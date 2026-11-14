"""Reconstruction and calibration end to end on simulated data, with interchangeable components."""

import re
from typing import override

import pytest
import torch
from torch import Tensor

from tests.nn.conftest import (
    FREE_NA,
    NA,
    SIGMA,
    WEIGHT,
    MatrixSystem,
    make_model,
    make_system,
    random_transfer,
    simulate,
    transfer_of,
)
from tests.nn.test_contracts import slice_weighted_blur
from waveorder.errors import ConvergenceWarning, IncompatibleModelError, ParameterError, PlacementError, ShapeError
from waveorder.nn import (
    FISTA,
    Evidence,
    Free,
    Gaussian,
    GridSearch,
    ImagingModel,
    ImagingSystem,
    MidbandPowerScore,
    Noise,
    NoPrior,
    ObjectPrior,
    ParameterPrior,
    Poisson,
    RichardsonLucy,
    Sampling,
    Tikhonov,
    TotalVariation,
    Wiener,
    calibrate,
    load_parameters,
    score_candidates,
)
from waveorder.nn.model import expect_on_system
from waveorder.nn.operators import LinearOperator
from waveorder.nn.parameters import same_device
from waveorder.nn.solvers import Loop, Solver
from waveorder.nn.verbs import InverseProblem
from waveorder.testing import assert_solver_descends

PRIOR_SCALE = (SIGMA**2 / WEIGHT) ** 0.5
SMALL = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.1))


def test_wiener_recovers_the_object_inside_the_passband(sampling):
    model = ImagingModel(make_system(1.0), Tikhonov(1e-6), Gaussian(1e-6))
    obj, data = simulate(model, sampling, scale=PRIOR_SCALE)
    estimate = Wiener()(data, model, sampling).estimate
    passband = transfer_of(model, sampling)[0, 0].abs() > 0.2
    true_spectrum, estimated_spectrum = torch.fft.rfftn(obj[0]), torch.fft.rfftn(estimate[0])
    error = (estimated_spectrum - true_spectrum)[passband].abs().norm() / true_spectrum[passband].abs().norm()
    assert error < 0.05


def test_wiener_reuses_a_bound_operator_and_rejects_one_bound_elsewhere(sampling):
    model = make_model(1.0)
    _, data = simulate(model, sampling, scale=PRIOR_SCALE)
    reused = Wiener()(data, model, sampling, operator=model.system.bind(sampling)).estimate
    assert torch.equal(reused, Wiener()(data, model, sampling).estimate)
    with pytest.raises(ShapeError, match="operator was bound to"):
        Wiener()(data, model, sampling, operator=model.system.bind(SMALL))


def test_the_objective_validates_data_and_estimate_as_every_verb_does():
    # 3-D data (no channel axis) and float64 data into a float32 model used to broadcast or promote silently.
    model = make_model(1.0)
    data, estimate = torch.zeros(1, *SMALL.shape), torch.zeros(1, *SMALL.shape)
    with pytest.raises(
        ShapeError, match=r"^ImagingModel\.objective data has shape \(8, 32, 32\), expected \(1, 8, 32, 32\)"
    ):
        model.objective(data[0], estimate, SMALL)
    with pytest.raises(
        ShapeError, match=r"^ImagingModel\.objective estimate has shape \(8, 32, 32\), expected \(1, 8, 32, 32\)"
    ):
        model.objective(data, estimate[0], SMALL)
    # A verb, or any caller checking on a model's behalf, names itself; the shape needs no binding.
    with pytest.raises(ShapeError, match=r"^Wiener data has shape \(8, 32, 32\), expected \(1, 8, 32, 32\)"):
        Wiener()(data[0], model, SMALL)
    with pytest.raises(ShapeError, match=r"^score_candidates data has shape \(8, 32, 32\)"):
        expect_on_system(
            data[0], model.system, where="score_candidates", name="data", shape=model.system.data_shape(SMALL)
        )
    with pytest.raises(PlacementError, match=r"ImagingModel\.objective data is torch\.float64 on cpu but "):
        model.objective(data.double(), estimate, SMALL)
    with pytest.raises(PlacementError, match=r"ImagingModel\.objective estimate is torch\.float64"):
        model.objective(data, estimate.double(), SMALL)
    # The same rule as every verb, and the error names the verb that was called.
    with pytest.raises(PlacementError, match=r"^Wiener data is torch\.float64 on cpu but "):
        Wiener()(data.double(), model, SMALL)
    with pytest.raises(PlacementError, match=r"^FISTA data is torch\.float64 on cpu but "):
        FISTA(loop=Loop(1))(data.double(), model, SMALL)
    with pytest.raises(PlacementError, match=r"^InverseProblem\.of data is torch\.float64"):  # called directly
        InverseProblem.of(data.double(), model, SMALL)
    # Complex data is complex, not "move it to float32": .to(float32) would silently drop the imaginary part.
    with pytest.raises(ShapeError, match=r"^Wiener data must be real, got torch\.complex64"):
        Wiener()(torch.complex(data, data), model, SMALL)
    with pytest.raises(ShapeError, match=r"^ImagingModel\.objective estimate must be real, got torch\.complex64"):
        model.objective(data, torch.complex(estimate, estimate), SMALL)


def test_every_path_that_takes_an_injected_operator_validates_it_before_use():
    # A two-channel operator on the right sampling against one-channel data used to reach the noise model, where the
    # objective broadcast it silently (or failed deep inside with a size mismatch).
    model = make_model(1.0)
    data = torch.zeros(1, *SMALL.shape)
    estimate = torch.zeros(1, *SMALL.shape)
    two_channels = MatrixSystem(random_transfer(2, 1, SMALL.shape), fields=("a",)).bind(SMALL)
    channels = r"operator maps data of shape \(2, 8, 32, 32\), but WidefieldFluorescence3D on this sampling maps data"
    with pytest.raises(ShapeError, match=f"ImagingModel.objective {channels}"):
        model.objective(data, estimate, SMALL, operator=two_channels)
    for verb in (Wiener(), FISTA(loop=Loop(1)), Evidence()):
        with pytest.raises(ShapeError, match=f"^{type(verb).__name__} {channels}"):
            verb(data, model, SMALL, operator=two_channels)
    # Placement: an operator bound from a float64 copy of the system does not stand in for the float32 one.
    elsewhere = make_model(1.0).double().system.bind(SMALL)
    with pytest.raises(PlacementError, match=r"objective operator computes in torch.float64 on cpu but "):
        model.objective(data, estimate, SMALL, operator=elsewhere)
    with pytest.raises(PlacementError, match=r"^Wiener operator computes in torch.float64"):
        Wiener()(data, model, SMALL, operator=elsewhere)
    # Sampling: bound elsewhere.
    with pytest.raises(ShapeError, match=r"ImagingModel\.objective operator was bound to"):
        model.objective(
            data, estimate, SMALL, operator=model.system.bind(Sampling(shape=(4, 16, 16), pixel_size=(0.25, 0.1, 0.1)))
        )
    # The model's own binding still passes through every check.
    torch.testing.assert_close(
        model.objective(data, estimate, SMALL, operator=model.system.bind(SMALL)),
        model.objective(data, estimate, SMALL),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
@pytest.mark.parametrize(
    "case",
    [
        lambda: (FISTA(loop=Loop(5)), make_model(1.0)),
        lambda: (RichardsonLucy(loop=Loop(5)), ImagingModel(make_system(1.0), NoPrior(), Poisson(gain=2.0))),
        lambda: (Wiener(), make_model(1.0)),
    ],
    ids=["fista", "richardson_lucy", "wiener"],
)
def test_a_solve_without_a_tolerance_never_waits_on_the_host(case):
    # A fixed-iteration solve used to read its final estimate back for a finiteness check: one sync per tile.
    solver, model = case()
    model = model.cuda()
    data = (10.0 + torch.rand(1, *SMALL.shape, generator=torch.Generator().manual_seed(0))).cuda()
    solver(data, model, SMALL)  # warm-up: one-time setup (cuFFT plans, caches) may sync
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        result = solver(data, model, SMALL)
        finite = result.finite
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert result.status in ("budget", "direct")  # nothing was checked on the host
    assert bool(finite)


def test_every_solver_call_checks_the_sampling():
    # The focus offset may reach 6 slices, beyond the +/-4 of an 8-slice stack: the defocus would wrap around, so
    # a direct call (no separate model.check) must refuse before solving.
    system = make_system(1.0, z_focus_offset=Free(0.0, bounds=(-6.0, 6.0)))
    model = ImagingModel(system, Tikhonov(WEIGHT), Gaussian(SIGMA))
    with pytest.raises(IncompatibleModelError, match="z_focus_offset"):
        Wiener()(torch.zeros(1, *SMALL.shape), model, SMALL)


class _WhiteNoise(Noise):
    """White Gaussian noise written from scratch, not a Gaussian subclass: it qualifies by its capability alone.

    With ``reports_variance=False`` it hides that capability, so solvers must treat it as generic noise.
    """

    has_curvature_bound = True

    def __init__(self, sigma: float, *, reports_variance: bool = True) -> None:
        super().__init__()
        self.register_quantity("sigma", sigma, positive=True)
        self.reports_variance = reports_variance

    @override
    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        return 0.5 * (((data - mean) / self.get_buffer("sigma")) ** 2).sum()

    @override
    def gradient(self, data: Tensor, mean: Tensor) -> Tensor:
        return (mean - data) / self.get_buffer("sigma") ** 2

    @override
    def curvature_bound(self) -> Tensor:
        return 1 / self.get_buffer("sigma") ** 2

    @override
    def scale(self) -> Tensor:
        return self.get_buffer("sigma") ** 2

    @override
    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        return mean + self.get_buffer("sigma") * torch.randn(mean.shape, generator=generator, dtype=mean.dtype)

    @override
    def variance(self, mean: Tensor) -> Tensor:
        return (self.get_buffer("sigma") ** 2).expand(mean.shape)

    @override
    def white_gaussian_variance(self) -> Tensor | None:
        return self.get_buffer("sigma") ** 2 if self.reports_variance else None


def test_wiener_accepts_any_noise_reporting_a_white_gaussian_variance():
    custom = ImagingModel(make_system(1.0), Tikhonov(WEIGHT), _WhiteNoise(SIGMA))
    _, data = simulate(custom, SMALL, scale=PRIOR_SCALE)
    expected = Wiener()(data, make_model(1.0), SMALL).estimate
    torch.testing.assert_close(Wiener()(data, custom, SMALL).estimate, expected)
    poisson = ImagingModel(make_system(1.0), Tikhonov(WEIGHT), Poisson())
    with pytest.raises(IncompatibleModelError, match="white Gaussian noise"):
        Wiener()(data, poisson, SMALL)


class _ShotNoise(Noise):
    """Unit-gain shot noise written from scratch: Richardson-Lucy must accept it by its photon counts, not by type."""

    has_photon_counts = True

    @override
    def photon_counts(self, data: Tensor) -> Tensor:
        return data.clamp_min(0.0)

    @override
    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        rate = mean.clamp_min(torch.finfo(mean.dtype).tiny)
        return (rate - self.photon_counts(data) * torch.log(rate)).sum()

    @override
    def gradient(self, data: Tensor, mean: Tensor) -> Tensor:
        return 1 - self.photon_counts(data) / mean.clamp_min(torch.finfo(mean.dtype).tiny)

    @override
    def scale(self) -> Tensor:
        return torch.ones(())

    @override
    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        return torch.poisson(mean.clamp_min(0.0), generator=generator)

    @override
    def variance(self, mean: Tensor) -> Tensor:
        return mean.clamp_min(0.0)


def _poisson_model(prior: ObjectPrior) -> ImagingModel:
    return ImagingModel(make_system(1.0), prior, Poisson(gain=2.0, offset=100.0))


def _poisson_data(model: ImagingModel, sampling: Sampling) -> Tensor:
    obj = 50.0 * torch.rand(model.system.object_shape(sampling), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return model.noise.sample(model(obj, sampling), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize(
    "case",
    [
        lambda: (Wiener(), make_model(1.0)),
        lambda: (FISTA(loop=Loop(20)), make_model(1.0)),
        lambda: (
            FISTA(loop=Loop(20)),
            ImagingModel(make_system(1.0), TotalVariation(0.05, nonnegative=True), Gaussian(SIGMA)),
        ),
        lambda: (RichardsonLucy(loop=Loop(10)), _poisson_model(NoPrior())),
    ],
    ids=["wiener", "fista", "fista_tv", "richardson_lucy"],
)
def test_solvers_descend_from_their_start(case):
    solver, model = case()
    if isinstance(model.noise, Poisson):
        data = _poisson_data(model, SMALL)
    else:
        _, data = simulate(model, SMALL, scale=PRIOR_SCALE)
    assert isinstance(solver, Solver)
    assert_solver_descends(solver, data=data, model=model, sampling=SMALL)


class _Flat(ObjectPrior):
    """A user-written flat prior: Richardson-Lucy must accept it by its is_flat capability, not by type."""

    @property
    @override
    def is_flat(self) -> bool:
        return True

    @override
    def penalty(self, obj: Tensor, *, scale: Tensor) -> Tensor:
        return obj.new_zeros(())

    @override
    def prox(self, obj: Tensor, step: Tensor, *, scale: Tensor) -> Tensor:
        return self.project(obj)


def test_richardson_lucy_asks_the_prior_whether_it_is_flat():
    data = _poisson_data(_poisson_model(NoPrior()), SMALL)
    custom = RichardsonLucy(loop=Loop(5))(data, _poisson_model(_Flat()), SMALL)
    reference = RichardsonLucy(loop=Loop(5))(data, _poisson_model(NoPrior()), SMALL)
    torch.testing.assert_close(custom.estimate, reference.estimate)
    with pytest.raises(IncompatibleModelError, match="prior Tikhonov"):
        RichardsonLucy(loop=Loop(5))(data, _poisson_model(Tikhonov(WEIGHT)), SMALL)


def test_richardson_lucy_asks_the_noise_for_photon_counts():
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson())
    data = _poisson_data(model, SMALL)
    custom = RichardsonLucy(loop=Loop(5))(data, ImagingModel(make_system(1.0), NoPrior(), _ShotNoise()), SMALL)
    torch.testing.assert_close(custom.estimate, RichardsonLucy(loop=Loop(5))(data, model, SMALL).estimate)
    with pytest.raises(IncompatibleModelError, match="photon_counts"):
        RichardsonLucy(loop=Loop(5))(data, ImagingModel(make_system(1.0), NoPrior(), Gaussian(SIGMA)), SMALL)


def test_fista_normal_operator_path_matches_the_generic_gradient():
    # White Gaussian noise lets FISTA use H* H f - H* g; a sign or scale slip there would change every iterate.
    def model(*, reports_variance: bool) -> ImagingModel:
        noise = _WhiteNoise(SIGMA, reports_variance=reports_variance)
        return ImagingModel(make_system(1.0), Tikhonov(WEIGHT), noise).double()

    _, data = simulate(make_model(1.0).double(), SMALL, scale=PRIOR_SCALE)
    fast = FISTA(loop=Loop(30))(data, model(reports_variance=True), SMALL)
    generic = FISTA(loop=Loop(30))(data, model(reports_variance=False), SMALL)
    assert fast.estimate.abs().max() > 0
    torch.testing.assert_close(fast.estimate, generic.estimate, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(fast.objective, generic.objective, rtol=1e-12, atol=0)


def test_fista_matches_wiener_on_the_same_quadratic_objective(sampling):
    model = make_model(1.0)
    _, data = simulate(model, sampling, scale=PRIOR_SCALE)
    closed = Wiener()(data, model, sampling)
    iterated = FISTA(loop=Loop(200))(data, model, sampling)
    assert iterated.objective.item() == pytest.approx(closed.objective.item(), rel=1e-4)
    torch.testing.assert_close(iterated.estimate, closed.estimate, atol=2e-3, rtol=0)


@pytest.mark.parametrize("prior", [Tikhonov(WEIGHT, nonnegative=True), TotalVariation(0.05, nonnegative=True)])
def test_fista_swaps_priors_and_keeps_their_support(prior, sampling):
    model = ImagingModel(make_system(1.0), prior, Gaussian(SIGMA))
    obj = torch.rand(model.system.object_shape(sampling), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        data = model.noise.sample(model(obj, sampling), generator=torch.Generator().manual_seed(1))
    start = model.object_prior.project(model.system.bind(sampling).adjoint(data))  # FISTA's initial point
    result = FISTA(loop=Loop(30))(data, model, sampling)
    assert result.estimate.min() >= 0
    assert result.objective < model.objective(data, start, sampling)


def test_fista_stops_at_tolerance_and_warns_when_it_cannot(sampling):
    model = make_model(1.0)
    _, data = simulate(model, sampling, scale=PRIOR_SCALE)
    loose = FISTA(loop=Loop(500, tolerance=1e-3))(data, model, sampling)
    assert loose.status == "converged"
    assert loose.iterations < 500
    with pytest.warns(ConvergenceWarning, match="without meeting tolerance"):
        tight = FISTA(loop=Loop(3, tolerance=1e-12))(data, model, sampling)
    assert tight.status == "max_iterations"
    # No criterion requested: the fixed budget ran, which is neither convergence nor failure.
    fixed = FISTA(loop=Loop(3))(data, model, sampling)
    assert fixed.status == "budget"
    assert bool(fixed.finite)


class _Understated(LinearOperator):
    """Another operator's actions with a norm five times too low: FISTA's step 1/L is then far too long."""

    def __init__(self, inner: LinearOperator) -> None:
        super().__init__(inner.sampling, object_shape=inner.object_shape, data_shape=inner.data_shape)
        self.inner = inner

    @property
    @override
    def placement(self) -> tuple[torch.dtype, torch.device]:
        return self.inner.placement

    @override
    def apply(self, obj: Tensor) -> Tensor:
        return self.inner.apply(obj)

    @override
    def adjoint(self, data: Tensor) -> Tensor:
        return self.inner.adjoint(data)

    @override
    def norm(self) -> Tensor:
        return 0.2 * self.inner.norm()


class _Understating(ImagingSystem):
    """Widefield fluorescence bound to a generic operator that understates its norm."""

    def __init__(self) -> None:
        super().__init__(fields=("a",), channel_count=1)
        self.inner = make_system(1.0)

    @override
    def compute_operator(self, sampling: Sampling) -> LinearOperator:
        return _Understated(self.inner.bind(sampling))


def test_a_diverging_solve_reports_it_on_the_device_and_with_a_tolerance_in_its_status():
    _, data = simulate(make_model(1.0).double(), SMALL, scale=PRIOR_SCALE)
    model = ImagingModel(_Understating(), Tikhonov(WEIGHT), Gaussian(SIGMA)).double()
    with torch.no_grad():
        diverged = FISTA(loop=Loop(300))(data, model, SMALL)
        assert diverged.status == "budget"
        assert not bool(diverged.finite)  # reported on the device; the solve itself read nothing back
        # With a tolerance the loop reads back at its checks, reports the divergence in the status and warns.
        with pytest.warns(ConvergenceWarning, match="non-finite"):
            checked = FISTA(loop=Loop(300, tolerance=1e-12))(data, model, SMALL)
        assert checked.status == "nonfinite"


def test_data_on_another_dtype_or_device_is_refused_naming_both_placements():
    model = make_model(1.0).double()
    data = torch.zeros(1, *SMALL.shape)
    with pytest.raises(PlacementError, match=r"float32 on cpu.*float64 on cpu"):
        FISTA(loop=Loop(1))(data, model, SMALL)
    if torch.cuda.is_available():
        with pytest.raises(PlacementError, match=r"float64 on cuda:0.*float64 on cpu"):
            FISTA(loop=Loop(1))(data.double().cuda(), model, SMALL)


def test_a_model_whose_children_were_moved_apart_names_the_child_before_solving():
    # A float32 prior under a float64 system used to fail deep inside the prox with a dtype error from torch.
    data = torch.zeros(1, *SMALL.shape, dtype=torch.float64)
    model = ImagingModel(make_system(1.0).double(), Tikhonov(WEIGHT), Gaussian(SIGMA).double())
    with pytest.raises(
        PlacementError, match=r"object_prior computes in torch.float32 on cpu but system computes in torch.float64"
    ):
        FISTA(loop=Loop(1))(data, model, SMALL)
    # A tree built mixed and then moved as a whole is fine: the check runs per call, not at construction.
    assert FISTA(loop=Loop(1))(data, model.double(), SMALL).status == "budget"
    if torch.cuda.is_available():
        model.noise.cuda()
        with pytest.raises(PlacementError, match=r"noise computes in torch.float64 on cuda:0 but system"):
            FISTA(loop=Loop(1))(data, model, SMALL)


def test_a_device_without_an_index_is_the_current_one():
    # torch.device("cuda") != torch.device("cuda:0"); a placement check comparing them directly refused valid input.
    assert same_device("cpu", torch.device("cpu", 0))
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        assert same_device("cuda", f"cuda:{current}")
        assert not same_device("cuda", "cpu")
        if torch.cuda.device_count() > 1:
            assert not same_device("cuda", f"cuda:{(current + 1) % torch.cuda.device_count()}")


def test_richardson_lucy_is_nonnegative_and_raises_the_likelihood(sampling):
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson(gain=2.0, offset=100.0))
    obj = 50.0 * torch.rand(model.system.object_shape(sampling), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        data = model.noise.sample(model(obj, sampling), generator=torch.Generator().manual_seed(1))
    short = RichardsonLucy(loop=Loop(2))(data, model, sampling)
    long = RichardsonLucy(loop=Loop(20))(data, model, sampling)
    assert long.estimate.min() >= 0
    assert long.objective < short.objective


def test_evidence_over_batched_candidates_finds_the_true_na(sampling):
    _, data = simulate(make_model(1.0), sampling, scale=PRIOR_SCALE)
    candidates = torch.linspace(0.8, 1.2, 9)
    with torch.no_grad():
        scores = score_candidates(data, make_model(FREE_NA), sampling, score=Evidence(), candidates={NA: candidates})
    assert scores.shape == candidates.shape
    assert candidates[scores.argmin()] == pytest.approx(1.0)


def test_candidate_names_are_validated(sampling):
    with pytest.raises(
        ParameterError, match=r"score_candidates candidates\.system\.detection\.numerical_apertur"
    ) as error:
        score_candidates(
            torch.zeros(1, *sampling.shape),
            make_model(FREE_NA),
            sampling,
            score=Evidence(),
            candidates={"system.detection.numerical_apertur": torch.ones(3)},
        )
    assert error.value.path == "candidates.system.detection.numerical_apertur"


def test_gradient_descent_with_the_parameter_prior(sampling):
    _, data = simulate(make_model(1.0), sampling, scale=PRIOR_SCALE)
    model = make_model(FREE_NA)
    prior = ParameterPrior.from_module(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(15):
        optimizer.zero_grad()
        loss = Evidence()(data, model, sampling) + prior.nll(dict(model.named_parameters()))
        loss.backward()
        optimizer.step()
        prior.clamp_(model)
    assert abs(model.get_parameter(NA).item() - 1.0) < 0.05


def test_load_parameters_applies_a_candidate():
    model = make_model(FREE_NA)
    load_parameters(model, {NA: torch.tensor(0.95)})
    assert model.get_parameter(NA).item() == pytest.approx(0.95)
    with pytest.raises(ParameterError, match=r"unknown parameter path\(s\) \['noise\.sigm'\]") as unknown:
        load_parameters(model, {"noise.sigm": torch.tensor(1.0)})
    assert unknown.value.path == "noise.sigm"


def test_load_parameters_says_a_fixed_quantity_is_fixed_and_what_to_do():
    """It used to call the right path of a fixed quantity "unknown" and list the free ones."""
    model = make_model(FREE_NA)
    fixed = r"^noise\.sigma is fixed: rebuild with the value or give it as Free\(\.\.\.\); free parameters are \['sys"
    with pytest.raises(ParameterError, match=fixed) as refused:
        load_parameters(model, {"noise.sigma": torch.tensor(1.0)})
    assert refused.value.path == "noise.sigma"
    with pytest.raises(ParameterError, match=r"^\['noise\.sigma', 'object_prior\.weight'\] are fixed: rebuild"):
        load_parameters(model, {"noise.sigma": 1.0, "object_prior.weight": 2.0, NA: 1.0})
    assert model.get_parameter(NA).item() == pytest.approx(1.1)  # nothing was copied


def test_load_parameters_refuses_a_value_outside_the_bounds():
    """A value past the bounds ran, and ``save`` then wrote the bound as the nominal: a file that is not the model."""
    model = make_model(FREE_NA)  # bounds (0.8, 1.2)
    for value in (1.3, torch.tensor(0.7), torch.tensor(1.21, dtype=torch.float64)):
        outside = rf"^load_parameters {re.escape(NA)} = [\d.]+ lies outside the bounds \(0\.8, 1\.2\) of its Free"
        with pytest.raises(ParameterError, match=outside) as refused:
            load_parameters(model, {NA: value})
        assert refused.value.path == NA
    assert model.get_parameter(NA).item() == pytest.approx(1.1)
    # On a bound, and the float32 rounding of a bound (where a calibration that stopped on it leaves the value).
    for value in (0.8, 1.2, torch.tensor(1.2), 1.2000000476837158, torch.tensor(0.8)):
        load_parameters(model, {NA: value})
    unbounded = ImagingModel(make_system(1.0, z_focus_offset=Free(0.0, tolerance=1.0)), Tikhonov(), Gaussian())
    load_parameters(unbounded, {"system.detection.z_focus_offset": 3.5})  # no bounds: any finite value


# --- messages name the fix --------------------------------------------------------------------------------------------


def test_a_refused_combination_names_what_to_use_instead():
    data = torch.zeros(1, *SMALL.shape)
    system = make_system(1.0)
    cases = [
        (Wiener(), TotalVariation(), Gaussian(), r"Wiener needs a Gaussian object prior, got TotalVariation; use FI"),
        (Wiener(), TotalVariation(), Gaussian(), r"use FISTA or ADMM \(ADMM takes TotalVariation only\)$"),
        (Wiener(), Tikhonov(nonnegative=True), Gaussian(), r"no closed form under a nonnegativity .*; use FISTA$"),
        (Wiener(), Tikhonov(), Poisson(), r"got Poisson; use RichardsonLucy with object_prior=NoPrior\(\) for Poisson"),
        (Evidence(), TotalVariation(), Gaussian(), r"got TotalVariation; use object_prior=Tikhonov\(\.\.\.\), or a sh"),
        (Evidence(), Tikhonov(), Poisson(), r"Evidence needs white Gaussian noise, got Poisson; use noise=Gaussian"),
        (RichardsonLucy(), Tikhonov(), Poisson(), r"got the prior Tikhonov: pass object_prior=NoPrior\(\)$"),
        (RichardsonLucy(), NoPrior(), Gaussian(), r"pass noise=Poisson\(\), or use Wiener or FISTA$"),
        (FISTA(), Tikhonov(), Poisson(), r"Use RichardsonLucy with object_prior=NoPrior\(\) for Poisson noise$"),
    ]
    for verb, prior, noise, message in cases:
        with pytest.raises(IncompatibleModelError, match=message):
            verb(data, ImagingModel(system, prior, noise), SMALL)


def test_poisson_noise_on_a_system_that_may_go_negative_names_the_real_fixes():
    """RichardsonLucy used to say "use FISTA" and FISTA "use RichardsonLucy" here; neither runs."""
    data = torch.zeros(1, *SMALL.shape)
    signed = MatrixSystem(random_transfer(1, 1, SMALL.shape), fields=("a",))
    assert not signed.preserves_nonnegativity
    model = ImagingModel(signed, NoPrior(), Poisson())
    with pytest.raises(IncompatibleModelError) as richardson_lucy:
        RichardsonLucy()(data, model, SMALL)
    with pytest.raises(IncompatibleModelError) as fista:
        FISTA()(data, model, SMALL)
    for error in (richardson_lucy, fista):
        message = str(error.value)
        assert "MatrixSystem does not declare preserves_nonnegativity" in message
        assert "set preserves_nonnegativity = True on the class" in message
        assert "noise=Gaussian(sigma)" in message
    assert "use FISTA" not in str(richardson_lucy.value).replace("use Wiener or FISTA", "")
    assert "Use RichardsonLucy" not in str(fista.value)
    assert "RichardsonLucy cannot run in its place" in str(fista.value)


def test_the_evidence_refusal_on_a_shift_variant_system_suggests_a_call_that_runs():
    model = ImagingModel(slice_weighted_blur(SMALL), Tikhonov(WEIGHT), Gaussian(0.1)).double()
    _, data = simulate(model, SMALL, scale=1.0)
    data = data.double()
    with pytest.raises(IncompatibleModelError, match="Evidence needs a system that binds to a Fourier") as refused:
        Evidence()(data, model, SMALL)
    assert "MidbandPowerScore(solver=FISTA(), cutoff=2 * NA / wavelength)" in str(refused.value)
    assert "score=MidbandPowerScore() on a waveorder.models model" in str(refused.value)
    score = MidbandPowerScore(solver=FISTA(), cutoff=2 * 1.0 / 0.52)  # the suggestion, for NA 1.0 at 0.52 um
    assert torch.isfinite(score(data, model, SMALL))


def test_calibrate_refuses_inference_mode_and_names_no_grad():
    model = make_model(FREE_NA)
    _, data = simulate(make_model(1.0), SMALL, scale=PRIOR_SCALE)
    message = r"^calibrate cannot run under torch\.inference_mode\(\): .* Use torch\.no_grad\(\) instead"
    with torch.inference_mode(), pytest.raises(IncompatibleModelError, match=message):
        calibrate(data, model, SMALL, score=Evidence())
    assert model.get_parameter(NA).item() == pytest.approx(1.1)  # refused before any stage moved it
    with torch.no_grad():
        result = calibrate(data, model, SMALL, score=Evidence(), search=(GridSearch(points=5),))
    assert result.parameters[NA].item() == pytest.approx(1.0, abs=0.11)


def test_half_precision_is_refused_before_the_fourier_transform():
    """torch has no half-precision FFT on the CPU; the model's own dtype is checked, as ``model.to`` checks it."""
    model, data = make_model(1.0).half(), torch.zeros(1, *SMALL.shape, dtype=torch.float16)
    message = r"^Wiener: the model computes in torch\.float16, but the Fourier transforms run in torch\.float32 and"
    with pytest.raises(IncompatibleModelError, match=message) as refused:
        Wiener()(data, model, SMALL)
    assert "ImagingModel.to(dtype=torch.float32)" in str(refused.value)
    with pytest.raises(IncompatibleModelError, match=r"^Evidence: the model computes in torch\.float16"):
        model.check(Evidence(), SMALL)


def test_placement_errors_say_what_to_move_and_how():
    model, data = make_model(1.0).double(), torch.zeros(1, *SMALL.shape)
    fix = r'convert the tensor with \.to\(dtype=torch\.float64, device="cpu"\), or move the model with ImagingModel\.to'
    with pytest.raises(PlacementError, match=fix) as error:
        Wiener()(data, model, SMALL)
    assert "model.to(device, dtype=...)" in str(error.value)  # the call on a waveorder.models model
    model.object_prior.float()
    with pytest.raises(PlacementError, match=r"move all three together with ImagingModel\.to\(dtype=\.\.\., device"):
        Wiener()(data.double(), model, SMALL)
    moved = make_model(1.0)
    stale = moved.system.bind(SMALL)
    moved.double()
    with pytest.raises(PlacementError, match=r"bind it again with model\.system\.bind\(sampling\), or pass operator="):
        Wiener()(data.double(), moved, SMALL, operator=stale)


def test_a_solve_result_repr_shows_shapes_not_tensors(sampling):
    model = make_model(1.0)
    _, data = simulate(model, sampling, scale=PRIOR_SCALE)
    result = Wiener()(data, model, sampling)
    assert repr(result) == (
        "SolveResult(estimate=Tensor(shape=(1, 16, 64, 64), dtype=torch.float32, device=cpu), "
        "objective=Tensor(shape=(), dtype=torch.float32, device=cpu), iterations=0, status='direct', history=None)"
    )
    iterated = repr(FISTA(loop=Loop(2))(data, model, sampling))
    assert "iterations=2, status='budget'" in iterated
    assert "tensor(" not in iterated
    assert "\n" not in iterated
