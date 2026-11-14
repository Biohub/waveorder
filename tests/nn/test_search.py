"""Calibration recovers the imaging system's free quantities from data, stage by stage."""

import fractions
import gc
import math
import re
import warnings
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, override

import pytest
import torch

from tests.nn.conftest import FREE_NA, NA, make_model, make_system, simulate
from waveorder.errors import CalibrationWarning, ParameterError, PlacementError, ShapeError
from waveorder.nn import (
    Evidence,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    ImagingModel,
    Poisson,
    RandomSearch,
    Sampling,
    Score,
    Tikhonov,
    ZernikeAberration,
    calibrate,
    evaluate,
    load_parameters,
    score_candidates,
)
from waveorder.nn.functional import abs2, gram, half_spectrum_weights
from waveorder.nn.operators import FourierDiagonal
from waveorder.nn.optics import Detection
from waveorder.nn.parameters import CalibratableModule, free_specs
from waveorder.nn.search import LBFGSSearch, expect_known_names, keep_best, on_bounds, search_spans
from waveorder.nn.search.base import CalibrationObjective, Estimate, SearchStage
from waveorder.nn.systems.fluorescence import WidefieldFluorescence3D
from waveorder.nn.verbs import InverseProblem
from waveorder.testing import check_search_stage

SAMPLING = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.1))
SIGMA = 0.05
SCALE = 3.0
# A prior that matches the simulated object: a misspecified weight biases the evidence's aperture estimate.
WEIGHT = SIGMA**2 / SCALE**2


class Stay:
    """A search stage that returns its start."""

    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        del objective
        return start


STAY = (Stay(),)

TOY_LIKELIHOOD = pytest.mark.filterwarnings(
    r"ignore:calibrate. \w+ is a likelihood, and a likelihood does not determine:waveorder.errors.CalibrationWarning"
)
"""For the toy scores below that call themselves likelihoods and read a free focus offset directly: a real
likelihood cannot determine it on this system, which ``calibrate`` warns about; these scores do by construction."""


def _model(numerical_aperture, weight=WEIGHT) -> ImagingModel:
    return ImagingModel(make_system(numerical_aperture), Tikhonov(weight), Gaussian(SIGMA))


def _data(seed: int = 0) -> torch.Tensor:
    return simulate(_model(1.0), SAMPLING, scale=SCALE, seed=seed)[1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibration_recovers_the_aperture(seed):
    model = _model(FREE_NA)
    result = calibrate(
        _data(seed),
        model,
        SAMPLING,
        score=Evidence(),
        search=(GridSearch(points=9), GradientSearch(iterations=30)),
    )
    assert list(result.parameters) == [NA]
    estimate = result.parameters[NA].item()
    assert model.get_parameter(NA).item() == pytest.approx(estimate)

    assert [name for name, _ in result.trace] == ["GridSearch", "GradientSearch"]
    values = [value for _, value in result.trace]
    assert values[1] <= values[0]
    assert result.objective.item() == pytest.approx(values[-1])

    assert result.at_bound == ()
    assert abs(estimate - 1.0) < 0.02


def test_a_quantity_on_its_bound_is_reported():
    # Truth NA 1.3 lies above the bound 1.2: the estimate sits on the bound, where the objective still slopes.
    data = simulate(_model(1.3), SAMPLING, scale=SCALE, seed=0)[1]
    weight = Free(WEIGHT, bounds=(WEIGHT / 1000, WEIGHT * 1000))
    model = _model(FREE_NA, weight=weight)
    result = calibrate(
        data, model, SAMPLING, score=Evidence(), search=(GridSearch(points=5), GradientSearch(iterations=10))
    )
    assert result.parameters[NA].item() == pytest.approx(1.2)
    assert result.at_bound == (NA,)


def test_lbfgs_refines_the_grid_estimate():
    # A 4-point grid over (0.8, 1.2) has no point at the truth 1.0, so the refinement has work to do.
    data = _data()
    grid = calibrate(data, _model(FREE_NA), SAMPLING, score=Evidence(), search=(GridSearch(points=4),))
    refined = calibrate(
        data, _model(FREE_NA), SAMPLING, score=Evidence(), search=(GridSearch(points=4), LBFGSSearch(iterations=5))
    )
    assert [name for name, _ in refined.trace] == ["GridSearch", "LBFGSSearch"]
    assert refined.objective.item() < grid.objective.item()
    assert abs(refined.parameters[NA].item() - 1.0) < 0.02


class ScaledEvidence(Evidence):
    """A toy score that is not a likelihood: half the negative log evidence, minimized at the same point."""

    is_likelihood: ClassVar[bool] = False

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        return 0.5 * super().run(problem)


class Plain(Score):
    """A score that is neither a likelihood nor a sum of independent terms: no covariance exists for it."""

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        return (_value(problem.model, NA) - 1.0).square()


class WhittleTotal(Score):
    """The Whittle negative log-likelihood of one channel, with an assumed object and noise variance, as one total.

    With a wrong noise variance it is a misspecified likelihood: its estimate is still consistent for its own optimum,
    but the inverse Hessian is not its variance.
    """

    is_likelihood: ClassVar[bool] = True

    def __init__(self, *, noise_variance: float, object_variance: float = SCALE**2, factor: float = 1.0) -> None:
        super().__init__()
        self.noise_variance = noise_variance
        self.object_variance = object_variance
        self.factor = factor

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        per_frequency = self.per_frequency(problem)
        weights = half_spectrum_weights(SAMPLING.shape[-1], device=per_frequency.device, dtype=per_frequency.dtype)
        return (per_frequency * weights).sum()

    def per_frequency(self, problem: InverseProblem) -> torch.Tensor:
        """Return the unweighted term of each half-spectrum sample, ``(Z, Y, X // 2 + 1)``."""
        operator = problem.operator
        assert isinstance(operator, FourierDiagonal)
        variance = gram(operator.transfer)[0, 0] * self.object_variance + self.noise_variance
        spectrum = abs2(torch.view_as_real(torch.fft.rfftn(problem.data[0], norm="ortho")))
        return self.factor * 0.5 * (spectrum / variance + torch.log(variance))


FLAT_NA = Free(1.1, bounds=(0.8, 1.2))


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("history", [False, True])
def test_calibrate_reads_back_once(history):
    data = _data().cuda()
    model = _model(FREE_NA).cuda()
    stages = (GridSearch(points=5, chunk_size=2), RandomSearch(samples=4), GradientSearch(iterations=3))
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibrate(data, model, SAMPLING, score=Evidence(), search=stages, history=history)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    syncs = [str(warning.message) for warning in caught if "synchronizing CUDA operation" in str(warning.message)]
    assert len(syncs) == 1, syncs
    assert [name for name, _ in result.trace] == ["GridSearch", "RandomSearch", "GradientSearch"]


def test_the_history_holds_every_gradient_step_and_the_last_full_grid():
    data, model = _data(), _model(FREE_NA)
    stages = (GridSearch(points=5, chunk_size=2), GradientSearch(iterations=6), GradientSearch(iterations=2))
    result = calibrate(data, model, SAMPLING, score=Evidence(), search=stages, history=True)
    assert len(result.steps) == 7 + 3  # every evaluation, the last one after the final step included
    assert min(result.steps) == pytest.approx(result.trace[-1][1], rel=1e-6)
    assert result.grid is not None
    assert result.grid.names == (NA,)
    assert result.grid.values.dtype == torch.float64
    assert result.grid.values.device.type == "cpu"
    assert result.grid.values[:, 0].tolist() == pytest.approx(torch.linspace(0.8, 1.2, 5).tolist())
    assert result.grid.values[:, 1].min().item() == pytest.approx(result.trace[0][1], rel=1e-6)
    assert result.sweep is None
    plain = calibrate(data, _model(FREE_NA), SAMPLING, score=Evidence(), search=stages)
    assert (plain.steps, plain.grid) == ((), None)
    assert plain.parameters[NA].item() == pytest.approx(result.parameters[NA].item())


def test_evaluate_swaps_top_level_names_and_does_not_keep_the_module_alive():
    prior = Tikhonov(Free(1.0, bounds=(0.1, 2.0)))
    swapped = evaluate(lambda m: m.quantity("weight") * 2, prior, {"weight": torch.tensor(3.0)})
    assert swapped.item() == 6.0
    assert prior.get_parameter("weight").item() == 1.0
    alive = weakref.ref(prior)
    del prior
    gc.collect()
    assert alive() is None


def test_a_wide_weight_span_is_searched_on_a_geometric_grid():
    # Nominal 3e-2 is no grid point; the best of a 9-point grid over (1e-6, 1e2) must be a power of ten, which a
    # linear grid (1e-6, 12.5, 25, ...) would not contain below 12.5.
    model = _model(1.0, weight=Free(3e-2, bounds=(1e-6, 1e2)))
    result = calibrate(_data(), model, SAMPLING, score=Evidence(), search=(GridSearch(points=9),))
    weight = result.parameters["object_prior.weight"].item()
    exponent = math.log10(weight)
    assert exponent == pytest.approx(round(exponent), abs=1e-4)
    assert -6 <= round(exponent) <= 0
    assert weight != pytest.approx(3e-2)


def test_chunked_candidates_score_like_one_batch():
    data = _data()
    candidates = {NA: torch.linspace(0.9, 1.1, 5)}
    with torch.no_grad():
        whole = score_candidates(data, _model(FREE_NA), SAMPLING, score=Evidence(), candidates=candidates)
        chunked = score_candidates(
            data, _model(FREE_NA), SAMPLING, score=Evidence(), candidates=candidates, chunk_size=2
        )
    torch.testing.assert_close(chunked, whole)


def test_grid_batches_bound_storage_and_keep_the_grid_and_its_best():
    weight = Free(WEIGHT, bounds=(WEIGHT / 1000, WEIGHT * 1000))
    specs = free_specs(_model(FREE_NA, weight=weight))
    batches = list(GridSearch(points=3, chunk_size=4).batches(specs, like=torch.zeros(())))
    assert [len(batch[NA]) for batch in batches] == [4, 4, 1]
    mesh = torch.meshgrid(
        torch.linspace(0.8, 1.2, 3),
        torch.logspace(math.log10(WEIGHT / 1000), math.log10(WEIGHT * 1000), 3),
        indexing="ij",
    )
    torch.testing.assert_close(torch.cat([batch[NA] for batch in batches]), mesh[0].reshape(-1))
    torch.testing.assert_close(torch.cat([batch["object_prior.weight"] for batch in batches]), mesh[1].reshape(-1))

    data = _data()
    whole, chunked = (
        calibrate(
            data,
            _model(FREE_NA, weight=weight),
            SAMPLING,
            score=Evidence(),
            search=(GridSearch(points=3, chunk_size=size),),
        )
        for size in (None, 4)
    )
    torch.testing.assert_close(chunked.parameters, whole.parameters, atol=0, rtol=0)
    assert chunked.objective.item() == whole.objective.item()


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_candidate_batches_reach_the_device_without_a_host_sync():
    like = torch.zeros((), device="cuda")
    specs = {NA: FREE_NA, "object_prior.weight": Free(WEIGHT, bounds=(WEIGHT / 1000, WEIGHT * 1000))}
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        grid = list(GridSearch(points=3, chunk_size=4).batches(specs, like=like))
        drawn = list(RandomSearch(samples=6, seed=3, chunk_size=4).batches(specs, like=like))
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    cpu = like.cpu()
    torch.testing.assert_close(
        grid, list(GridSearch(points=3, chunk_size=4).batches(specs, like=cpu)), check_device=False
    )
    # The CPU generator draws the same points whatever the device.
    reference = list(RandomSearch(samples=6, seed=3, chunk_size=4).batches(specs, like=cpu))
    torch.testing.assert_close(drawn, reference, check_device=False)


def test_search_spans_are_the_bounds_or_three_tolerances():
    spans = search_spans({"a": Free(1.0, tolerance=0.1), "b": Free(0.0, tolerance=1.0, bounds=(-1.0, 2.0))})
    assert spans == {"a": pytest.approx((0.7, 1.3)), "b": (-1.0, 2.0)}
    with pytest.raises(ParameterError, match="c needs bounds or a tolerance"):
        search_spans({"c": Free(0.0)})


def test_grid_search_needs_a_span():
    unspanned = ZernikeAberration({"spherical": Free(0.0)})
    model = ImagingModel(make_system(1.0, pupil_aberrations=unspanned), Tikhonov(WEIGHT), Gaussian(SIGMA))
    with pytest.raises(ParameterError, match="needs bounds or a tolerance"):
        calibrate(_data(), model, SAMPLING, score=Evidence(), search=(GridSearch(points=3),))


FOCUS = "system.detection.z_focus_offset"


def _focus_model(numerical_aperture: float | Free, *, offset: float | Free) -> ImagingModel:
    system = make_system(numerical_aperture, z_focus_offset=offset)
    return ImagingModel(system, Tikhonov(WEIGHT), Gaussian(SIGMA))


FREE_FOCUS = Free(0.0, bounds=(-2.0, 2.0))


def test_a_likelihood_is_warned_off_a_free_focus_offset_of_a_3d_object():
    """The offset translates the point-spread function, which the evidence cannot see: its estimate means nothing."""
    data = _data()
    message = (
        rf"^calibrate: Evidence is a likelihood, and a likelihood does not determine {re.escape(FOCUS)} on "
        r"WidefieldFluorescence3D\. .* Hold it fixed: build the part with a number instead of Free\(\.\.\.\)$"
    )
    with pytest.warns(CalibrationWarning, match=message) as caught:
        result = calibrate(
            data, _focus_model(FREE_NA, offset=FREE_FOCUS), SAMPLING, score=Evidence(), search=(GridSearch(points=3),)
        )
    assert len(caught) == 1
    assert caught[0].filename == __file__  # attributed to the caller of calibrate
    assert set(result.parameters) == {NA, FOCUS}  # a warning, not a refusal
    # The evidence itself does not change with the offset, by a slice here.
    at = [Evidence()(data, _focus_model(1.0, offset=offset), SAMPLING).item() for offset in (0.0, 1.0)]
    assert at[0] == pytest.approx(at[1], rel=1e-6)


def test_no_warning_without_a_likelihood_or_without_the_free_offset():
    data = _data()
    stages = (GridSearch(points=3),)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibrate(data, _focus_model(FREE_NA, offset=FREE_FOCUS), SAMPLING, score=ScaledEvidence(), search=stages)
        calibrate(data, _focus_model(FREE_NA, offset=0.5), SAMPLING, score=Evidence(), search=stages)


class _Tiltable(WidefieldFluorescence3D):
    """A system that says the likelihood cannot see its aperture either: the method is the one hook."""

    @override
    def undetermined_by_likelihood(self) -> tuple[str, ...]:
        return (*super().undetermined_by_likelihood(), "detection.numerical_aperture", "not.a.quantity")


def test_a_system_names_what_a_likelihood_cannot_determine_and_only_free_ones_are_reported():
    system = _Tiltable(Detection(wavelength=0.52, numerical_aperture=FREE_NA), index_of_refraction_media=1.33)
    model = ImagingModel(system, Tikhonov(WEIGHT), Gaussian(SIGMA))
    assert system.undetermined_by_likelihood()[0] == "detection.z_focus_offset"
    with pytest.warns(CalibrationWarning, match=rf"does not determine {re.escape(NA)} on _Tiltable\."):
        calibrate(_data(), model, SAMPLING, score=Evidence(), search=(GridSearch(points=3),))


def test_calibration_needs_a_free_quantity():
    with pytest.raises(ParameterError, match="no free quantity"):
        calibrate(_data(), make_model(1.0), SAMPLING, score=Evidence())


@dataclass(frozen=True)
class TryApertures:
    """A user-written stage: score a few given apertures and keep the best."""

    apertures: tuple[float, ...]
    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        return keep_best(objective, {NA: torch.tensor(self.apertures)}, start=start)


class RunOnly:
    """A stage that forgot to declare whether it reads back."""

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        del objective
        return start


def test_a_stage_declares_whether_it_reads_back():
    assert isinstance(TryApertures((1.0,)), SearchStage)
    assert not isinstance(RunOnly(), SearchStage)
    assert [stage.reads_back for stage in (GridSearch(), RandomSearch(), GradientSearch(), LBFGSSearch())] == [
        False,
        False,
        False,
        True,
    ]


def test_a_new_search_stage_is_one_class():
    data = _data()
    stages = (TryApertures((0.9, 1.0)), GradientSearch(iterations=5))
    result = calibrate(data, _model(FREE_NA), SAMPLING, score=Evidence(), search=stages)
    assert [name for name, _ in result.trace] == ["TryApertures", "GradientSearch"]
    # FREE_NA: nominal 1.1, tolerance 0.2.
    at_truth = Evidence()(data, _model(1.0), SAMPLING).item() + 0.5 * ((1.0 - 1.1) / 0.2) ** 2
    assert result.trace[0][1] == pytest.approx(at_truth, rel=1e-6)
    assert result.trace[1][1] <= result.trace[0][1]


def test_the_objective_starts_at_the_model_values():
    data = _data()
    objective = CalibrationObjective.of(data, _model(FREE_NA), SAMPLING, score=Evidence())
    start = objective.start()
    assert list(start.point) == [NA]
    assert start.point[NA].item() == pytest.approx(1.1)
    # FREE_NA is centred on 1.1, so the prior adds nothing there.
    assert start.objective.item() == pytest.approx(Evidence()(data, _model(1.1), SAMPLING).item(), rel=1e-6)
    with pytest.raises(ParameterError, match="no free quantity"):
        CalibrationObjective.of(data, make_model(1.0), SAMPLING, score=Evidence())


def test_a_partial_batch_is_scored_at_the_best_point_of_the_other_quantities():
    data = _data()
    weight = Free(WEIGHT, bounds=(WEIGHT / 1000, WEIGHT * 1000))
    objective = CalibrationObjective.of(data, _model(FREE_NA, weight=weight), SAMPLING, score=Evidence())
    # A best weight far from the model's own value, so scoring at the model's value would be caught.
    best_weight = torch.tensor(WEIGHT * 30)
    point = {NA: torch.tensor(1.15), "object_prior.weight": best_weight}
    with torch.no_grad():
        start = Estimate(point, objective(point))
    candidates = (0.9, 1.0, 1.05)
    chosen = keep_best(objective, [{NA: torch.tensor(candidates[:2])}, {NA: torch.tensor(candidates[2:])}], start=start)
    assert chosen.point["object_prior.weight"].item() == best_weight.item()
    # The best of the candidates and the start, each scored at the best weight.
    with torch.no_grad():
        scored = {a: objective({NA: torch.tensor(a), "object_prior.weight": best_weight}).item() for a in candidates}
    scored[1.15] = start.objective.item()
    best = min(scored, key=lambda a: scored[a])
    assert chosen.point[NA].item() == pytest.approx(best)
    assert chosen.objective.item() == pytest.approx(scored[best], rel=1e-6)
    with pytest.raises(ParameterError, match="not searched"):
        keep_best(objective, {"system.detection.z_focus_offset": torch.zeros(2)}, start=start)


def test_free_takes_tolerance_and_bounds_by_keyword():
    with pytest.raises(TypeError):
        Free(1.0, (0.8, 1.2))  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize(
    "make",
    [
        lambda: GridSearch(5),  # ty: ignore[too-many-positional-arguments]
        lambda: RandomSearch(5),  # ty: ignore[too-many-positional-arguments]
        lambda: GradientSearch(10),  # ty: ignore[too-many-positional-arguments]
        lambda: LBFGSSearch(10),  # ty: ignore[too-many-positional-arguments]
    ],
    ids=["grid", "random", "gradient", "lbfgs"],
)
def test_stages_take_keywords_only(make):
    with pytest.raises(TypeError, match="positional argument"):
        make()


def test_on_bounds_names_a_missing_value():
    specs = free_specs(_model(FREE_NA))
    with pytest.raises(ParameterError, match="missing") as error:
        on_bounds(specs, {})
    assert error.value.path == NA


@pytest.mark.parametrize(
    ("shape", "pixel_size", "error", "field"),
    [
        ((8, 16), (0.25, 0.1, 0.1), ShapeError, "shape"),
        ((0, 16, 16), (0.25, 0.1, 0.1), ShapeError, "shape"),
        ((8, 16, 16), (0.25, 0.0, 0.1), ParameterError, "pixel_size"),
        ((8, 16, 16), (0.25, math.inf, 0.1), ParameterError, "pixel_size"),
    ],
    ids=["two_axes", "zero_count", "zero_size", "infinite_size"],
)
def test_sampling_takes_three_positive_counts_and_positive_finite_sizes(shape, pixel_size, error, field):
    with pytest.raises(error, match=rf"Sampling\.{field}"):
        Sampling(shape=shape, pixel_size=pixel_size)


@pytest.mark.parametrize(
    ("stage", "owner", "path"),
    [
        (
            GridSearch(points={"system.detection.numerical_apertur": 3}),
            "GridSearch.points",
            "points.system.detection.numerical_apertur",
        ),
        (
            GradientSearch(iterations=1, learning_rates={"object_prior.wieght": 0.1}),
            "GradientSearch.learning_rates",
            "learning_rates.object_prior.wieght",
        ),
    ],
    ids=["grid_points", "learning_rates"],
)
def test_a_name_keyed_setting_rejects_a_name_that_is_not_free(stage, owner, path):
    with pytest.raises(ParameterError, match=owner) as error:
        calibrate(_data(), _model(FREE_NA), SAMPLING, score=Evidence(), search=(stage,))
    assert error.value.path == path


class NaNAboveAperture(Evidence):
    """Evidence that is NaN for an aperture above 1.09, as a score can be outside the region it models."""

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        system = problem.model.system
        assert isinstance(system, WidefieldFluorescence3D)
        aperture = system.detection.numerical_aperture
        return torch.where(aperture > 1.09, torch.nan, super().run(problem))


def test_a_nan_start_is_replaced_by_a_finite_candidate():
    # The start (1.1) scores NaN; compared as NaN it would beat no candidate and be kept.
    model = _model(FREE_NA)
    objective = CalibrationObjective.of(_data(), model, SAMPLING, score=NaNAboveAperture())
    assert objective.start().objective.item() == math.inf
    result = calibrate(_data(), model, SAMPLING, score=NaNAboveAperture(), search=(GridSearch(points=5),))
    assert result.parameters[NA].item() <= 1.09
    assert math.isfinite(result.objective.item())


def _value(model: torch.nn.Module, name: str) -> torch.Tensor:
    """The current value of free quantity ``name``, including one swapped in by ``evaluate``."""
    prefix, _, attribute = name.rpartition(".")
    module = model.get_submodule(prefix)
    assert isinstance(module, CalibratableModule)
    return module.quantity(attribute)


class TwoDirections(Score):
    """``0.5 |A theta - b|^2 / s^2`` over the free quantities: a likelihood that informs only two directions."""

    is_likelihood: ClassVar[bool] = True

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names
        generator = torch.Generator().manual_seed(3)
        self.directions = torch.randn(2, len(names), generator=generator, dtype=torch.float64)

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        theta = torch.stack([_value(problem.model, name) for name in self.names])
        return 0.5 * ((self.directions @ theta - 1.0) / 0.01).square().sum()


def _six_quantity_model() -> ImagingModel:
    terms = {f"z{j}": Free(0.0, tolerance=0.05, bounds=(-0.5, 0.5)) for j in (5, 6, 7, 8)}
    system = make_system(
        Free(1.0, tolerance=0.1, bounds=(0.8, 1.2)),
        z_focus_offset=Free(0.0, tolerance=0.5, bounds=(-2.0, 2.0)),
        pupil_aberrations=ZernikeAberration(terms),
    )
    return ImagingModel(system, Tikhonov(WEIGHT), Gaussian(SIGMA)).double()


class Quadratic(Score):
    """``0.5 theta^T A theta - sum(theta)``: a likelihood whose Hessian over the free quantities is exactly ``A``."""

    is_likelihood: ClassVar[bool] = True

    def __init__(self, names: list[str], hessian: torch.Tensor) -> None:
        super().__init__()
        self.names = names
        self.hessian = hessian

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        theta = torch.stack([_value(problem.model, name) for name in self.names])
        return 0.5 * theta @ self.hessian @ theta - theta.sum()


def _rotated(spectrum: list[float], *, size: int) -> torch.Tensor:
    """A symmetric ``(size, size)`` matrix with eigenvalues ``spectrum`` (zero-padded) along fixed random axes."""
    generator = torch.Generator().manual_seed(5)
    axes, _ = torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))
    values = torch.tensor([*spectrum, *[0.0] * (size - len(spectrum))], dtype=torch.float64)
    return axes @ torch.diag(values) @ axes.mT


def _quantity_model(zernikes: int, *, tolerance: bool, dtype: torch.dtype = torch.float64) -> ImagingModel:
    """A model with the aperture, the focus and ``zernikes`` Zernike terms free, near the middle of their bounds;
    Gaussian priors when ``tolerance``, flat ones otherwise."""

    def free(nominal: float, half: float, bounds: tuple[float, float]) -> Free:
        return Free(nominal, tolerance=half if tolerance else None, bounds=bounds)

    terms = {f"z{j}": free(0.0, 0.05, (-50.0, 50.0)) for j in range(5, 5 + zernikes)}
    system = make_system(
        free(1.0, 0.1, (0.5, 1.3)),
        z_focus_offset=free(0.0, 0.5, (-3.0, 3.0)),
        pupil_aberrations=ZernikeAberration(terms),
    )
    return ImagingModel(system, Tikhonov(WEIGHT), Gaussian(SIGMA)).to(dtype)


def _exact_errors(model: ImagingModel, hessian: torch.Tensor) -> torch.Tensor:
    """The float64 standard errors of ``(hessian + prior precision)^-1``, the truth both covariances estimate."""
    specs = free_specs(model).values()
    precision = torch.tensor([0.0 if s.tolerance is None else s.tolerance**-2 for s in specs], dtype=torch.float64)
    return torch.linalg.inv(hessian + torch.diag(precision)).diagonal().sqrt()


@pytest.mark.parametrize(
    ("build", "path"),
    [
        (lambda: GridSearch(points=1), "points"),
        (lambda: GridSearch(points={NA: 1}), f"points.{NA}"),
        (lambda: GridSearch(chunk_size=0), "chunk_size"),
        (lambda: RandomSearch(samples=0), "samples"),
        (lambda: GradientSearch(iterations=0), "iterations"),
        (lambda: GradientSearch(learning_rate=-0.1), "learning_rate"),
        (lambda: GradientSearch(learning_rate=math.nan), "learning_rate"),
        (lambda: GradientSearch(learning_rates={NA: -1.0}), f"learning_rates.{NA}"),
        (lambda: LBFGSSearch(iterations=0), "iterations"),
        (lambda: LBFGSSearch(history=0), "history"),
    ],
    ids=[
        "grid_one",
        "grid_named_one",
        "grid_chunk",
        "random_samples",
        "gradient_iterations",
        "negative_rate",
        "nan_rate",
        "negative_named_rate",
        "lbfgs_iterations",
        "lbfgs_history",
    ],
)
def test_stage_settings_are_checked_at_construction(build, path):
    with pytest.raises(ParameterError) as error:
        build()
    assert error.value.path == path


@dataclass(frozen=True)
class MisdeclaredStage:
    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ("steps",)

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        del objective
        return start


def test_a_declared_quantity_keyed_field_must_exist():
    with pytest.raises(ParameterError, match=r"quantity_keyed names 'steps'") as error:
        calibrate(_data(), _model(FREE_NA), SAMPLING, score=Evidence(), search=(MisdeclaredStage(),))
    assert error.value.path == "steps"


def test_a_sampling_from_lists_is_the_tuple_sampling():
    from_lists = Sampling(shape=[8, 32, 32], pixel_size=[0.25, 0.1, 0.1])  # ty: ignore[invalid-argument-type]
    assert from_lists == SAMPLING
    assert hash(from_lists) == hash(SAMPLING)
    assert isinstance(from_lists.shape, tuple)


def test_load_parameters_takes_numbers_and_names_a_wrong_value():
    model = _model(FREE_NA)
    load_parameters(model, {NA: 1.05})
    assert model.get_parameter(NA).item() == pytest.approx(1.05)
    for bad in (math.nan, math.inf, torch.tensor(math.nan)):
        with pytest.raises(ParameterError, match=f"load_parameters {re.escape(NA)} must be finite") as error:
            load_parameters(model, {NA: bad})
        assert error.value.path == NA
    assert model.get_parameter(NA).item() == pytest.approx(1.05)


@pytest.mark.parametrize("wavelength", [math.inf, math.nan, -0.5], ids=["inf", "nan", "negative"])
def test_detection_takes_a_positive_finite_wavelength(wavelength):
    with pytest.raises(ParameterError, match="Detection wavelength must be positive and finite") as error:
        Detection(wavelength=wavelength, numerical_aperture=0.5)
    assert error.value.path == "wavelength"


def test_an_unknown_zernike_term_is_a_parameter_error_at_its_name():
    with pytest.raises(ParameterError, match="Unknown Zernike term 'bogus'") as error:
        ZernikeAberration({"bogus": 0.1})
    assert error.value.path == "bogus"


def test_score_candidates_names_an_unknown_candidate_under_its_own_name():
    with pytest.raises(ParameterError, match=r"score_candidates candidates\.system\.widht is not a parameter") as error:
        score_candidates(
            _data(), _model(FREE_NA), SAMPLING, score=Evidence(), candidates={"system.widht": torch.ones(2)}
        )
    assert error.value.path == "candidates.system.widht"
    assert "fixed quantities are buffers" in str(error.value)


def _calibrate(data: torch.Tensor) -> None:
    calibrate(data, _model(FREE_NA), SAMPLING, score=Evidence(), search=STAY)


def _score_candidates(data: torch.Tensor) -> None:
    score_candidates(data, _model(FREE_NA), SAMPLING, score=Evidence(), candidates={NA: torch.tensor([1.0])})


@pytest.mark.parametrize(
    ("call", "where"),
    [(_calibrate, "calibrate"), (_score_candidates, "score_candidates")],
    ids=["calibrate", "score_candidates"],
)
def test_data_errors_name_the_caller(call, where):
    data = _data()
    with pytest.raises(ShapeError, match=rf"{where} data has shape"):
        call(data[..., :-1])
    with pytest.raises(PlacementError, match=rf"{where} data is torch\.float64"):
        call(data.double())
    with pytest.raises(ShapeError, match=rf"{where} data must be real"):
        call(torch.complex(data, data))


@dataclass(frozen=True)
class KeyedStage:
    """A plugin stage with a field keyed by quantity name, declared in ``quantity_keyed``."""

    steps: Mapping[str, float]
    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ("steps",)

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        expect_known_names(self.steps, objective.specs, path="KeyedStage.steps")
        return start


def test_a_plugin_stage_names_are_checked_before_any_stage_runs():
    class Fails:
        reads_back: ClassVar[bool] = False
        quantity_keyed: ClassVar[tuple[str, ...]] = ()

        def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
            del objective, start
            raise AssertionError("a stage ran")

    # A list of stages is a valid Sequence[SearchStage].
    stages = [Fails(), KeyedStage({"system.detection.numerical_apertur": 0.1})]
    with pytest.raises(ParameterError, match=r"KeyedStage\.steps") as error:
        calibrate(_data(), _model(FREE_NA), SAMPLING, score=Evidence(), search=stages)
    assert error.value.path == "steps.system.detection.numerical_apertur"
    result = calibrate(_data(), _model(FREE_NA), SAMPLING, score=Evidence(), search=[KeyedStage({NA: 0.1})])
    assert [name for name, _ in result.trace] == ["KeyedStage"]


class Undeclared:
    """A stage written before ``quantity_keyed`` was a protocol member."""

    reads_back: ClassVar[bool] = False

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        del objective
        return start


def _calibration(data, model, sampling, *, score) -> tuple[CalibrationObjective, Estimate]:
    objective = CalibrationObjective.of(data, model, sampling, score=score)
    return objective, objective.start()


def test_the_stage_check_asserts_a_missing_quantity_keyed():
    objective, start = _calibration(_data(), _model(FREE_NA), SAMPLING, score=Evidence())
    with pytest.raises(AssertionError, match="must declare quantity_keyed"):
        check_search_stage(Undeclared(), objective, start)  # ty: ignore[invalid-argument-type]


def test_the_stage_check_requires_real_numbers_in_quantity_keyed_fields():
    objective, start = _calibration(_data(), _model(FREE_NA), SAMPLING, score=Evidence())
    check_search_stage(KeyedStage({NA: 0.1}), objective, start)
    with pytest.raises(AssertionError, match=r"KeyedStage\.steps\.system.* must be a real number, not a bool"):
        check_search_stage(KeyedStage({NA: True}), objective, start)
    with pytest.raises(AssertionError, match="must be a real number, not a str"):
        check_search_stage(KeyedStage({NA: "0.1"}), objective, start)  # ty: ignore[invalid-argument-type]


def test_a_float32_grid_on_its_bound_passes_the_stage_check():
    # The grid's 1.2 bound is 1.2000000476837158 in float32; compared in float64 it would read as outside.
    objective, start = _calibration(_data(), _model(FREE_NA), SAMPLING, score=Evidence())
    check_search_stage(GridSearch(points=5), objective, start)


def test_score_candidates_checks_every_argument():
    data, model = _data(), _model(FREE_NA)
    good = {NA: torch.tensor([1.0, 1.1])}
    cases = [
        (dict(chunk_size=0), "chunk_size", "score_candidates chunk_size must be at least 1"),
        (dict(candidates={}), "candidates", "score_candidates candidates must be non-empty"),
        (
            dict(candidates={NA: good[NA].double()}),
            f"candidates.{NA}",
            r"candidates\.system.* must be torch\.float32 on cpu, as .* is, got torch\.float64",
        ),
    ]
    for overrides, path, match in cases:
        kwargs = {"model": model, "score": Evidence(), "candidates": good} | overrides
        with pytest.raises(ParameterError, match=match) as error:
            score_candidates(data, kwargs.pop("model"), SAMPLING, **kwargs)  # ty: ignore[invalid-argument-type]
        assert error.value.path == path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_candidates_names_a_candidate_on_another_device():
    with pytest.raises(ParameterError, match=r"on cpu, as .* is, got torch\.float32 on cuda") as error:
        score_candidates(
            _data(), _model(FREE_NA), SAMPLING, score=Evidence(), candidates={NA: torch.tensor([1.0], device="cuda")}
        )
    assert error.value.path == f"candidates.{NA}"


def test_a_quantity_takes_any_real_number_as_a_float():
    """Values read from array metadata are often numpy scalars; ``numpy.float32`` is no ``float``."""
    eighth = Gaussian(fractions.Fraction(1, 8))  # ty: ignore[invalid-argument-type]
    assert eighth.fixed_quantities() == {"sigma": 0.125}
    assert isinstance(eighth.fixed_quantities()["sigma"], float)
    numpy = pytest.importorskip("numpy")
    for value in (numpy.float32(0.25), numpy.float64(0.25), numpy.int64(2)):
        fixed = Tikhonov(value).fixed_quantities()["weight"]
        assert type(fixed) is float
        assert fixed == float(value)
    photons = Poisson(gain=numpy.float32(2.0), offset=numpy.int16(100))
    assert photons.fixed_quantities() == {"gain": 2.0, "offset": 100.0}
    with pytest.raises(ParameterError, match=r"Gaussian\.sigma must be positive, got -0\.5"):
        Gaussian(numpy.float32(-0.5))


def test_calibrate_needs_a_stage():
    for stages in ((), []):
        with pytest.raises(ParameterError, match="at least one search stage") as error:
            calibrate(_data(), _model(FREE_NA), SAMPLING, score=Evidence(), search=stages)
        assert error.value.path == "search"


class _Quadratic(Score):
    """``(gain - 3)^2`` of the system's free gain: a known objective for the optimizers."""

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        return (problem.model.system.quantity("gain") - 3.0).square()


def _quadratic_problem() -> tuple[CalibrationObjective, Estimate]:
    from tests.nn.wo40_fixtures import GaussianTransfer

    model = ImagingModel(GaussianTransfer(Free(1.0, bounds=(-10.0, 10.0))), Tikhonov(1e-3), Gaussian(1.0)).double()
    data = torch.zeros(1, 4, 8, 8, dtype=torch.float64)
    return _calibration(data, model, Sampling(shape=(4, 8, 8), pixel_size=(1.0, 1.0, 1.0)), score=_Quadratic())


def _descend_by_hand(optimizer: str, *, scale: float, steps: int) -> float:
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    built = {"sgd": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
             "adam": lambda p: torch.optim.Adam(p, lr=0.1)}[optimizer]([x])  # fmt: skip
    for _ in range(steps):
        built.zero_grad()
        (scale * (x - 3.0) ** 2).backward()
        built.step()
    return float(x.detach())


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_the_last_iterate_is_the_torch_optimizer_s_with_the_loss_scaled(optimizer, scale):
    objective, start = _quadratic_problem()
    stage = GradientSearch(
        iterations=12, learning_rates={"system.gain": 0.1}, optimizer=optimizer, loss_scale=scale, last=True
    )
    estimate = stage.run(objective, start=start)
    expected = _descend_by_hand(optimizer, scale=scale, steps=12)
    assert float(estimate.point["system.gain"]) == pytest.approx(expected, rel=1e-12)
    assert float(estimate.objective) == pytest.approx((expected - 3.0) ** 2, rel=1e-12)
    if optimizer == "adam":  # Adam's steps do not depend on the scale, up to its epsilon
        assert expected == pytest.approx(_descend_by_hand("adam", scale=1.0, steps=12), rel=1e-6)


def test_the_best_point_is_kept_unless_the_last_is_asked_for():
    objective, start = _quadratic_problem()
    # Nesterov momentum overshoots 3 and swings back: the best visit is not the last iterate.
    stage = {"iterations": 25, "learning_rates": {"system.gain": 0.1}, "optimizer": "sgd"}
    best = GradientSearch(**stage).run(objective, start=start)  # ty: ignore[invalid-argument-type]
    last = GradientSearch(**stage, last=True).run(objective, start=start)  # ty: ignore[invalid-argument-type]
    assert float(best.objective) < float(last.objective)


def test_a_grid_over_named_quantities_holds_the_others():
    model = _model(FREE_NA, weight=Free(WEIGHT, bounds=(WEIGHT / 100, WEIGHT * 100)))
    data = _data()
    objective, start = _calibration(data, model, SAMPLING, score=Evidence())
    estimate = GridSearch(points=5, names=("object_prior.weight",)).run(objective, start=start)
    assert torch.equal(estimate.point[NA], start.point[NA])
    swept = torch.logspace(math.log10(WEIGHT / 100), math.log10(WEIGHT * 100), 5, dtype=torch.float32)
    picked = float(estimate.point["object_prior.weight"])
    assert any(math.isclose(picked, value, rel_tol=1e-6) for value in swept.tolist())
    with pytest.raises(ParameterError, match="which are not free quantities"):
        GridSearch(names=("noise.sigma",)).run(objective, start=start)
    spec = Free(0.0, tolerance=0.1)
    assert search_spans({"a": spec}, width=2.0) == {"a": (-0.2, 0.2)}


def test_the_noise_and_weight_profile_and_gradient_match_wo40():
    """wo-4.0 run_evidence_calibration's profile of (sigma, weight) and its gradient stage, on fixed aberrations."""
    from tests.nn.wo40_fixtures import GaussianTransfer, sweep_volume

    data = sweep_volume().double()[None]
    sampling = Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0))
    weight = Free(1e-3, bounds=(1e-6, 1.0))
    model = ImagingModel(GaussianTransfer(), Tikhonov(weight), Gaussian("estimate")).double()
    names = ("noise.sigma", "object_prior.weight")
    search = [GridSearch(points=13, names=names), GradientSearch(iterations=30, learning_rate=0.01, last=True)]
    profile = calibrate(data, model, sampling, score=Evidence(), search=search[:1]).parameters
    # wo-4.0: sigma in std(data) * geomspace(1e-3, 1, 13), weight in geomspace(1e-6, 1, 13), first strict minimum.
    assert float(profile["noise.sigma"]) == pytest.approx(5.394478196995732, rel=1e-6)
    assert float(profile["object_prior.weight"]) == pytest.approx(0.0031622776601683794, rel=1e-9)
    model = ImagingModel(GaussianTransfer(), Tikhonov(weight), Gaussian("estimate")).double()
    result = calibrate(data, model, sampling, score=Evidence(), search=search)
    # wo-4.0: then Adam(lr=0.01) on log sigma and log weight, 30 steps, the last iterate.
    assert float(result.parameters["noise.sigma"]) == pytest.approx(4.833106051413308, rel=1e-5)
    assert float(result.parameters["object_prior.weight"]) == pytest.approx(0.0034891498617156975, rel=1e-5)
    assert float(result.objective) == pytest.approx(25900.94140625, rel=1e-6)
