"""A part needs at most three methods of its own: every other method of its base class has a default."""

import inspect

import pytest
import torch
from torch import Tensor

from tests.nn.conftest import SIGMA, make_model, make_system, simulate
from waveorder.data.sources import PairSource
from waveorder.data.transforms import Transform
from waveorder.nn import (
    Gaussian,
    GaussianObjectPrior,
    ImagingModel,
    ImagingSystem,
    IterativeSolver,
    Loop,
    Noise,
    NoPrior,
    ObjectPrior,
    Sampling,
    Score,
    SearchStage,
    ShiftInvariantSystem,
    Solver,
)
from waveorder.nn.networks import RestorationNetwork
from waveorder.nn.networks.bundle import Normalizer
from waveorder.nn.networks.conditioning import Condition, Conditioning
from waveorder.nn.networks.encoders import Encoder, EncoderModule
from waveorder.nn.networks.heads import Head
from waveorder.nn.networks.stems import Stem, StemModule
from waveorder.nn.scores import SharpnessScore
from waveorder.nn.verbs import InverseProblem
from waveorder.testing import assert_solver_descends, check_noise
from waveorder.train.losses import Loss
from waveorder.train.tasks import Task

BASES = [
    ImagingSystem,
    ShiftInvariantSystem,
    ObjectPrior,
    GaussianObjectPrior,
    Noise,
    NoPrior,
    Solver,
    IterativeSolver,
    Score,
    SharpnessScore,
    RestorationNetwork,
    Stem,
    StemModule,
    Encoder,
    EncoderModule,
    Head,
    Condition,
    Conditioning,
    Normalizer,
    PairSource,
    Transform,
    Loss,
    Task,
]


@pytest.mark.parametrize("base", BASES, ids=lambda base: base.__name__)
def test_a_base_class_requires_at_most_three_methods(base: type):
    required = sorted(getattr(base, "__abstractmethods__", ()))
    assert len(required) <= 3, required


def test_a_search_stage_is_one_method():
    methods = [name for name, value in vars(SearchStage).items() if inspect.isfunction(value) and name[0] != "_"]
    assert methods == ["run"]


class _PlainGaussian(Noise):
    """White Gaussian noise written with the three required methods only."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.register_quantity("sigma", sigma, positive=True)

    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        return 0.5 * (((data - mean) / self.get_buffer("sigma")) ** 2).sum()

    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        return mean + self.get_buffer("sigma") * torch.randn(mean.shape, generator=generator, dtype=mean.dtype)

    def variance(self, mean: Tensor) -> Tensor:
        return (self.get_buffer("sigma") ** 2).expand(mean.shape)


def test_noise_defaults_to_autograd_gradient_and_a_unit_scale():
    noise = _PlainGaussian(SIGMA)
    mean = torch.rand(2, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    data = noise.sample(mean, generator=torch.Generator().manual_seed(1))
    check_noise(noise, data, mean)
    with torch.no_grad():
        torch.testing.assert_close(noise.gradient(data, mean), Gaussian(SIGMA).gradient(data, mean))
    assert noise.scale().item() == 1.0
    torch.testing.assert_close(Gaussian(SIGMA).scale(), torch.tensor(SIGMA**2))


class _GradientDescent(IterativeSolver[Tensor]):
    """Gradient descent on the data term, written with the three required methods only (no ``check``)."""

    def start(self, problem: InverseProblem) -> Tensor:
        return torch.zeros(problem.operator.object_shape, dtype=problem.data.dtype)

    def step(self, state: Tensor, problem: InverseProblem) -> Tensor:
        operator = problem.operator
        gradient = operator.adjoint(problem.model.noise.gradient(problem.data, operator.apply(state)))
        return state - SIGMA**2 / operator.norm() ** 2 * gradient

    def estimate(self, state: Tensor) -> Tensor:
        return state


def test_an_iterative_solver_runs_on_any_model_without_a_check():
    sampling = Sampling(shape=(8, 16, 16), pixel_size=(0.25, 0.1, 0.1))
    model = ImagingModel(make_system(1.0), NoPrior(), Gaussian(SIGMA))
    _, data = simulate(model, sampling, scale=1.0)
    solver = _GradientDescent(loop=Loop(5))
    solver.check(make_model(1.0))
    assert_solver_descends(solver, data=data, model=model, sampling=sampling)


class _Identity(RestorationNetwork):
    """A network that declares no stride: it takes any ``Y`` and ``X``."""

    channels, fields, condition_sizes = 1, 1, {}

    def halo(self) -> int:
        return 0

    def planes_out(self, planes_in: int) -> int:
        return planes_in

    def forward(self, data: Tensor, condition=None) -> Tensor:
        del condition
        return data

    def arguments(self) -> dict[str, object]:
        return {}


def test_a_network_without_a_stride_takes_any_size():
    network = _Identity()
    assert network.stride == 1
    network.check_shapes(planes=[1, 3], sizes=[(5, 7)])
