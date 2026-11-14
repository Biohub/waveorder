"""Solvers under vmap over candidates and over data; Loop settings; the transfer cache across grad modes."""

import pytest
import torch
from torch.func import grad, vmap

from tests.nn.conftest import FREE_NA, NA, SIGMA, WEIGHT, make_model, make_system
from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn import (
    ADMM,
    FISTA,
    Evidence,
    Gaussian,
    ImagingModel,
    NoPrior,
    Poisson,
    RichardsonLucy,
    Sampling,
    Tikhonov,
    TotalVariation,
    evaluate,
)
from waveorder.nn.solvers import Loop
from waveorder.testing import assert_solver_vmaps_over_candidates

SAMPLING = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.1))
SMALL = Sampling(shape=(4, 16, 16), pixel_size=(0.25, 0.1, 0.1))


def _data(seed: int = 0, batch: tuple[int, ...] = (), sampling: Sampling = SAMPLING) -> torch.Tensor:
    return torch.randn(*batch, 1, *sampling.shape, generator=torch.Generator().manual_seed(seed))


def _small_model() -> ImagingModel:
    """The review's case: float64, a free aperture, Tikhonov 1e-2 and sigma 0.05 (an ill-conditioned adjoint)."""
    return ImagingModel(make_system(FREE_NA), Tikhonov(WEIGHT), Gaussian(SIGMA)).double()


def test_tolerance_under_vmap_is_refused_not_read_back():
    # Deciding to stop needs a host read, which vmap cannot do; it used to crash the process instead of raising.
    model = make_model(1.0)
    with torch.no_grad(), pytest.raises(IncompatibleModelError, match="tolerance=None"):
        vmap(lambda d: FISTA(loop=Loop(4, tolerance=1e-3))(d, model, SAMPLING).estimate)(_data(batch=(2,)))


@pytest.mark.parametrize(
    "case",
    [
        lambda: (FISTA(loop=Loop(4)), Tikhonov(WEIGHT), Gaussian(SIGMA)),
        lambda: (ADMM(loop=Loop(4), rho=0.1), TotalVariation(0.05), Gaussian(SIGMA)),
        lambda: (RichardsonLucy(loop=Loop(4)), NoPrior(), Poisson(gain=2.0)),
    ],
    ids=["fista", "admm_tv", "richardson_lucy"],
)
def test_solvers_vmap_over_candidates_of_a_free_quantity(case):
    # A grid search maps its score, and a sharpness score the solver inside it, over candidate apertures.
    solver, prior, noise = case()
    model = ImagingModel(make_system(FREE_NA), prior, noise).double()
    data = _data(sampling=SMALL).double()
    if isinstance(noise, Poisson):
        data = 10.0 + data.abs()
    assert_solver_vmaps_over_candidates(solver, data=data, model=model, sampling=SMALL, rtol=1e-9)


def test_the_candidate_check_fails_a_solver_that_reads_values_back():
    # A tolerance decides when to stop by a host read, which vmap over candidates cannot do.
    model, data = _small_model(), _data(sampling=SMALL).double()
    with pytest.raises(AssertionError, match="FISTA does not run under vmap over candidate values"):
        assert_solver_vmaps_over_candidates(FISTA(loop=Loop(5, tolerance=1e-3)), data=data, model=model, sampling=SMALL)


def test_a_fista_gradient_computes_the_gram_once(monkeypatch):
    # Each FISTA step applies H* H: its Gram is computed once per solve by normal_action, not once per iteration.
    import waveorder.nn.operators as operators

    calls: list[None] = []
    original = operators.gram

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(None)
        return original(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(operators, "gram", counted)
    model, data, weights = _small_model(), _data(sampling=SMALL).double(), _data(seed=4, sampling=SMALL).double()
    solver = FISTA(loop=Loop(20))
    point = torch.tensor(1.0, dtype=torch.float64)
    value = grad(lambda na: evaluate(lambda m: (solver(data, m, SMALL).estimate * weights).sum(), model, {NA: na}))
    assert value(point).abs() > 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"iterations": 0}, "Loop.iterations"),
        ({"iterations": 5, "check_every": 0}, "Loop.check_every"),
        ({"iterations": 5, "tolerance": 0.0}, "Loop.tolerance"),
    ],
)
def test_loop_rejects_bad_settings_by_name(settings, match):
    with pytest.raises(ParameterError, match=match) as error:
        Loop(**settings)
    assert error.value.path == match.removeprefix("Loop.")  # the field, so config can locate it


def test_loop_settings_reach_every_iterative_solver():
    # check_every is a Loop setting, so Richardson-Lucy honours it although its constructor never names it.
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson(gain=2.0))
    data = 10.0 + _data(sampling=SMALL).abs()
    result = RichardsonLucy(loop=Loop(500, tolerance=1e-3, check_every=7))(data, model, SMALL)
    assert result.status == "converged"
    assert result.iterations % 7 == 0


def test_evidence_under_inference_mode_then_autograd():
    # A transfer computed in inference mode must not be cached and later reused in a graph.
    system = make_system(1.0)
    system.cache = "auto"
    model = ImagingModel(system, Tikhonov(), Gaussian())
    data = _data()
    with torch.inference_mode():
        Evidence()(data, model, SAMPLING)
    obj = torch.zeros(1, *SAMPLING.shape, requires_grad=True)
    (model(obj, SAMPLING) * data).sum().backward()
    assert obj.grad is not None
