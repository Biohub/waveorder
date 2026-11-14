"""``solver.calibration.search``: the gradient optimizer, the Nelder-Mead stage, and the 3.x ``advanced.optimizer``."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.config.build import build
from waveorder.errors import ConfigurationWarning, Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import CalibrationResult, GradientSearch, GridSearch, LBFGSSearch, Sampling, load_parameters
from waveorder.nn.search import NelderMeadSearch

CONFIG = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.12}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection:
    wavelength: 0.52
    numerical_aperture: {nominal: 1.2, tolerance: free, bounds: [1.0, 1.3]}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
solver:
  calibration: {score: evidence}
"""


def _raw(**calibration: object) -> dict:
    raw = yaml.safe_load(CONFIG)
    raw["solver"]["calibration"].update(calibration)
    return raw


def _search(**calibration: object) -> tuple[object, ...]:
    _, parts = build(V4Config(**_raw(**calibration)))
    return parts["search"]


@pytest.mark.parametrize("optimizer", ["adam", "nadam", "adamw"])
def test_a_gradient_stage_runs_the_configured_torch_optimizer(optimizer):
    (stage,) = _search(search=[{"gradient": {"max_iter": 5, "optimizer": optimizer}}])
    assert isinstance(stage, GradientSearch)
    assert (stage.optimizer, stage.iterations) == (optimizer, 5)


def test_a_nelder_mead_stage_takes_its_iterations_and_tolerance():
    (stage,) = _search(search=[{"nelder_mead": {"iterations": 50, "tolerance": 1e-3}}])
    assert stage == NelderMeadSearch(iterations=50, tolerance=1e-3)
    (bare,) = _search(search=["nelder_mead"])
    assert bare == NelderMeadSearch()


def _calibrate(search: list[object]) -> tuple[float, float]:
    model = ReconstructionModel.from_config(_raw(search=search))
    sampling = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.12))  # data.pixel_size of CONFIG
    obj = torch.rand(1, *sampling.shape, generator=torch.Generator().manual_seed(0))
    load_parameters(model.imaging_model, {"system.detection.numerical_aperture": 1.15})
    with torch.no_grad():
        data = model.imaging_model(obj, sampling)
    load_parameters(model.imaging_model, {"system.detection.numerical_aperture": 1.2})
    result = model.calibrate(data, pixel_size=sampling.pixel_size, prepared=True)
    assert isinstance(result, CalibrationResult)
    return float(result.parameters["system.detection.numerical_aperture"]), float(result.objective)


def test_a_nelder_mead_stage_reaches_the_minimum_of_a_fine_grid():
    # The evidence of this mismatched prior is lowest near 1.07, not at the imaging aperture 1.15; the stage must
    # find the score's minimum, which a 61-point grid over the bounds (spacing 0.005) brackets.
    on_grid, grid_objective = _calibrate([{"grid": {"points": 61}}])
    found, objective = _calibrate([{"nelder_mead": {"iterations": 60, "tolerance": 1e-5}}])
    assert found == pytest.approx(on_grid, abs=0.005)
    assert objective <= grid_objective + 1e-3 * abs(grid_objective)


@pytest.mark.parametrize(
    ("advanced", "kind", "attributes"),
    [
        ({"optimizer": "adam"}, GradientSearch, {"optimizer": "adam"}),
        (
            {"optimizer": "nadam", "learning_rates": {"detection.numerical_aperture": 0.01}},
            GradientSearch,
            {"optimizer": "nadam", "learning_rates": {"system.detection.numerical_aperture": 0.01}},
        ),
        ({"optimizer": "lbfgs"}, LBFGSSearch, {"iterations": LBFGSSearch().iterations}),
        ({"optimizer": "nelder_mead"}, NelderMeadSearch, {"tolerance": NelderMeadSearch().tolerance}),
        ({"optimizer": "nelder_mead", "convergence_tol": 1e-6}, NelderMeadSearch, {"tolerance": 1e-6}),
        ({"optimizer": "grid_search"}, GridSearch, {"points": GridSearch().points}),
        ({"optimizer": "grid_search", "grid_points": 5}, GridSearch, {"points": 5}),
    ],
    ids=["adam", "nadam_with_rates", "lbfgs", "nelder_mead", "nelder_mead_tol", "grid", "grid_points"],
)
def test_the_3x_optimizer_names_the_search(advanced, kind, attributes):
    (stage,) = _search(advanced=advanced)
    assert type(stage) is kind
    assert {name: getattr(stage, name) for name in attributes} == attributes


@pytest.mark.parametrize(
    ("calibration", "location"),
    [
        ({"advanced": {"optimizer": "adam"}, "search": ["lbfgs"]}, "solver.calibration.advanced.optimizer"),
        ({"advanced": {"optimizer": "adam", "convergence_tol": 1e-3}}, "solver.calibration.advanced.convergence_tol"),
        ({"advanced": {"optimizer": "lbfgs", "grid_points": 5}}, "solver.calibration.advanced.grid_points"),
        (
            {"advanced": {"optimizer": "nelder_mead", "convergence_tol": -1.0}},
            "solver.calibration.advanced.convergence_tol",
        ),
        ({"advanced": {"optimizer": "grid_search", "grid_points": 1}}, "solver.calibration.advanced.grid_points"),
    ],
    ids=["two_searches", "tol_not_read", "points_not_read", "negative_tol", "one_point"],
)
def test_a_3x_optimizer_setting_it_cannot_honour_is_refused_at_its_key(calibration, location):
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(_raw(**calibration))
    (problem,) = rejected.value.problems
    assert problem.message.startswith(f"{location}: ")


def test_the_wo_40_optimizer_settings_this_version_does_not_execute_are_accepted_with_a_located_warning():
    ignored = {"convergence_patience": 5, "use_gradients": True, "log_dir": "logs"}
    with pytest.warns(ConfigurationWarning) as caught:
        model = ReconstructionModel.from_config(_raw(advanced={"optimizer": "adam", **ignored}))
    located = [warning.message for warning in caught if isinstance(warning.message, ConfigurationWarning)]
    assert len(located) == len(caught)
    assert sorted(warning.location or "" for warning in located) == [
        f"solver.calibration.advanced.{name}" for name in sorted(ignored)
    ]
    assert all("is ignored" in str(warning) for warning in located)
    assert model.search == ReconstructionModel.from_config(_raw(advanced={"optimizer": "adam"})).search
