"""The search stages of :mod:`waveorder.nn.search`: 3.x optimizer parity, bounds, and the stage kit."""

from typing import ClassVar, override

import pytest
import torch

from tests.nn.conftest import FREE_NA, NA, make_model, make_system, simulate
from tests.nn.legacy_optim import minimize_neldermead, optimize_gradient
from waveorder.errors import ParameterError
from waveorder.nn import (
    Evidence,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    ImagingModel,
    NelderMeadSearch,
    Sampling,
    Score,
    Tikhonov,
    ZernikeAberration,
    calibrate,
    free_specs,
)
from waveorder.nn.parameters import CalibratableModule
from waveorder.nn.search import CalibrationObjective, Estimate
from waveorder.nn.search.grid import MAX_GRID_POINTS
from waveorder.nn.verbs import InverseProblem
from waveorder.testing import check_search_stage

FOCUS = "system.detection.z_focus_offset"
TOY_SAMPLING = Sampling(shape=(8, 16, 16), pixel_size=(0.25, 0.1, 0.1))
SAMPLING = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.1))


def _quantity(model: torch.nn.Module, name: str) -> torch.Tensor:
    prefix, _, attribute = name.rpartition(".")
    module = model.get_submodule(prefix)
    assert isinstance(module, CalibratableModule)
    return module.quantity(attribute)


def _calibration(data, model, sampling, *, score) -> tuple[CalibrationObjective, Estimate]:
    objective = CalibrationObjective.of(data, model, sampling, score=score)
    return objective, objective.start()


def _toy(aperture: torch.Tensor, focus: torch.Tensor, *, centre: float = 1.08) -> torch.Tensor:
    """A smooth, coupled, non-quadratic objective of two quantities, minimal near ``(centre, 0.35)``."""
    a, b = aperture - centre, focus - 0.35
    return 20 * a**2 + b**2 + 2 * a * b + 0.5 * b**4


class Toy(Score):
    """:func:`_toy` of the model's aperture and focus offset, ignoring the data."""

    is_likelihood: ClassVar[bool] = True

    def __init__(self, *, centre: float = 1.08) -> None:
        super().__init__()
        self.centre = centre

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        return _toy(_quantity(problem.model, NA), _quantity(problem.model, FOCUS), centre=self.centre)


def _toy_problem(*, nominal: float = 1.0):
    system = make_system(Free(nominal, bounds=(0.5, 1.3)), z_focus_offset=Free(0.0, bounds=(-3.0, 3.0)))
    model = ImagingModel(system, Tikhonov(1e-2), Gaussian(0.05)).double()
    data = torch.zeros(model.system.data_shape(TOY_SAMPLING), dtype=torch.float64)
    return model, data


def _legacy_toy(x: list[float]) -> float:
    return _toy(torch.tensor(x[0], dtype=torch.float64), torch.tensor(x[1], dtype=torch.float64)).item()


@pytest.mark.parametrize("iterations", [8, 60, 400], ids=["budget", "midway", "converged"])
def test_nelder_mead_follows_the_3x_scipy_path(iterations):
    # 3.x: scipy.optimize.minimize(method="Nelder-Mead", options={"maxfev": n, "maxiter": n}) from the nominal values.
    model, data = _toy_problem()
    objective, start = _calibration(data, model, TOY_SAMPLING, score=Toy())
    result = NelderMeadSearch(iterations=iterations).run(objective, start=start)
    x, fun, evaluations = minimize_neldermead(_legacy_toy, [1.0, 0.0], maxiter=iterations, maxfev=iterations)
    assert [result.point[NA].item(), result.point[FOCUS].item()] == pytest.approx(x, rel=1e-13, abs=1e-15)
    assert result.objective.item() == pytest.approx(fun, rel=1e-13)
    if iterations == 400:
        assert evaluations < 400  # stopped by scipy's default xatol and fatol of 1e-4
        assert result.point[NA].item() == pytest.approx(1.08, abs=1e-3)


@pytest.mark.parametrize("optimizer", ["adam", "nadam", "adamw"])
def test_gradient_search_follows_the_3x_optimizer(optimizer):
    # 3.x _optimize_gradient (adam, nadam) and recover_zernikes (adamw): one param group per quantity, absolute rates.
    model, data = _toy_problem()
    objective, start = _calibration(data, model, TOY_SAMPLING, score=Toy())
    rates = {NA: 0.01, FOCUS: 0.05}
    stage = GradientSearch(iterations=40, learning_rates=rates, optimizer=optimizer)
    result = stage.run(objective, start=start)

    def loss(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return _toy(params[NA], params[FOCUS])

    visited = optimize_gradient(
        loss,
        {NA: (1.0, rates[NA]), FOCUS: (0.0, rates[FOCUS])},
        method=optimizer,
        max_iterations=40,
        dtype=torch.float64,
    )
    best = min(range(len(visited)), key=lambda k: visited[k][1])  # the first lowest, as 4.x keeps a strict improvement
    params, lowest = visited[best]
    assert result.point[NA].item() == pytest.approx(params[NA], rel=1e-12)
    assert result.point[FOCUS].item() == pytest.approx(params[FOCUS], rel=1e-12)
    assert result.objective.item() == pytest.approx(lowest, rel=1e-12)


def test_nelder_mead_keeps_to_the_bounds_and_never_worsens_the_start():
    # The minimum, aperture 1.4, lies past the upper bound 1.3; a start near the bound puts a vertex past it at once.
    model, data = _toy_problem(nominal=1.28)
    objective, start = _calibration(data, model, TOY_SAMPLING, score=Toy(centre=1.4))
    result = NelderMeadSearch(iterations=200).run(objective, start=start)
    assert 0.5 <= result.point[NA].item() <= 1.3
    assert result.point[NA].item() == pytest.approx(1.3, abs=1e-3)
    assert result.objective.item() < start.objective.item()
    # A start already at the optimum: no vertex is better, so the start comes back unchanged.
    model, data = _toy_problem(nominal=1.08)
    objective, start = _calibration(data, model, TOY_SAMPLING, score=Toy())
    at_optimum = {NA: torch.tensor(1.08, dtype=torch.float64), FOCUS: torch.tensor(0.35, dtype=torch.float64)}
    exact = start._replace(point=at_optimum, objective=objective(at_optimum))
    assert NelderMeadSearch(iterations=50).run(objective, start=exact) is exact


class NaNAbove(Score):
    """:func:`_toy`, NaN where the aperture exceeds 1.1."""

    is_likelihood: ClassVar[bool] = True

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> torch.Tensor:
        aperture = _quantity(problem.model, NA)
        value = _toy(aperture, _quantity(problem.model, FOCUS), centre=1.2)
        return torch.where(aperture > 1.1, torch.nan, value)


def test_nelder_mead_reads_a_nan_objective_as_infinite():
    model, data = _toy_problem()
    objective, start = _calibration(data, model, TOY_SAMPLING, score=NaNAbove())
    result = NelderMeadSearch(iterations=200).run(objective, start=start)
    assert result.point[NA].item() <= 1.1
    assert torch.isfinite(result.objective)
    assert result.objective.item() < start.objective.item()


def _data() -> torch.Tensor:
    return simulate(make_model(1.0), SAMPLING, scale=3.0, seed=0)[1]


@pytest.mark.parametrize(
    "stage",
    [
        NelderMeadSearch(iterations=20),
        GradientSearch(iterations=5, optimizer="nadam"),
        GradientSearch(iterations=5, optimizer="adamw"),
    ],
    ids=["nelder_mead", "nadam", "adamw"],
)
def test_the_new_stages_pass_the_stage_kit(stage):
    data = _data()
    objective, start = _calibration(data, make_model(FREE_NA), SAMPLING, score=Evidence())
    check_search_stage(stage, objective, start)
    if torch.cuda.is_available():
        objective, start = _calibration(data.cuda(), make_model(FREE_NA).cuda(), SAMPLING, score=Evidence())
        check_search_stage(stage, objective, start)


def test_nelder_mead_calibrates_the_aperture():
    result = calibrate(
        _data(), make_model(FREE_NA), SAMPLING, score=Evidence(), search=(NelderMeadSearch(iterations=40),)
    )
    assert result.parameters[NA].item() == pytest.approx(1.0, abs=0.03)
    assert [name for name, _ in result.trace] == ["NelderMeadSearch"]


@pytest.mark.parametrize(
    ("make", "path"),
    [
        (lambda: GradientSearch(optimizer="rmsprop"), "optimizer"),  # ty: ignore[invalid-argument-type]
        (lambda: NelderMeadSearch(iterations=0), "iterations"),
        (lambda: NelderMeadSearch(tolerance=0.0), "tolerance"),
        (lambda: NelderMeadSearch(tolerance=float("nan")), "tolerance"),
    ],
    ids=["optimizer", "zero_iterations", "zero_tolerance", "nan_tolerance"],
)
def test_the_new_stage_settings_are_checked(make, path):
    with pytest.raises(ParameterError) as error:
        make()
    assert error.value.path == path


def _aberration_specs(count: int) -> dict[str, Free]:
    return {f"system.detection.pupil_aberrations.z{j}": Free(0.0, bounds=(-0.1, 0.1)) for j in range(4, 4 + count)}


def test_a_grid_of_more_than_a_million_points_is_refused_before_it_is_built():
    """Twelve free quantities at the default 7 points are 7 ** 12 = 1.4e10 grid points, whose index alone was a
    110 GB allocation. The refusal counts the points and names the stages that scale."""
    like = torch.zeros(())
    message = (
        r"^GridSearch would score 7 \*\* 12 \(points \*\* quantities\) = 13841287201 points, more than the 1000000 a "
        r"grid may hold; pass search=\[GradientSearch\(\)\] \(or RandomSearch\), or fewer points per quantity "
        r"\(points=\{name: n\}\)$"
    )
    for stage in (GridSearch(), GridSearch(chunk_size=8)):  # chunks bound the storage, not the count
        with pytest.raises(ParameterError, match=message) as error:
            stage.batches(_aberration_specs(12), like=like)
        assert error.value.path == "points"
    seven = _aberration_specs(7)
    with pytest.raises(ParameterError, match=r"would score 9 x 7 x 7 x 7 x 7 x 7 x 7 = 1058841 points"):
        GridSearch(points={next(iter(seven)): 9}).batches(seven, like=like)
    # At or below the cap a grid is built as before: 7 ** 7 points, and exactly 10 ** 6.
    assert MAX_GRID_POINTS == 10**6
    assert [len(axis) for axis in next(GridSearch(chunk_size=5).batches(seven, like=like)).values()] == [5] * 7
    assert len(next(GridSearch(points=10, chunk_size=3).batches(_aberration_specs(6), like=like))) == 6


@pytest.mark.parametrize("search", [None, (GridSearch(),)], ids=["default-search", "grid"])
def test_calibrate_refuses_a_grid_over_twelve_free_quantities(search):
    terms = {f"z{j}": Free(0.0, bounds=(-0.1, 0.1)) for j in range(4, 15)}
    model = make_model(FREE_NA, pupil_aberrations=ZernikeAberration(terms))
    assert len(free_specs(model)) == 12
    with pytest.raises(ParameterError, match=r"7 \*\* 12 \(points \*\* quantities\) = 13841287201 points") as error:
        calibrate(_data(), model, SAMPLING, score=Evidence(), search=search)
    assert str(error.value).endswith(
        "pass search=[GradientSearch()] (or RandomSearch), or fewer points per quantity (points={name: n})"
    )
