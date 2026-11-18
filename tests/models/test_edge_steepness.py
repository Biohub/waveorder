"""A calibration on a softer sigmoid pupil edge than the reconstruction's, as wo-4.0's phase optimizer fitted."""

import copy
import json

import pytest
import torch
import yaml

from waveorder.config import load
from waveorder.config.migrate import migrate
from waveorder.errors import ParameterError, Rejected
from waveorder.models import Phase2D, ReconstructionModel
from waveorder.models._stores import system_record
from waveorder.nn import Detection, Illumination, Sampling
from waveorder.nn.optics import PupilEdge

PREVIEW = """\
version: 4
data:
  input_channel_names: [Brightfield]
  pixel_size: {yx: {y: 0.1, x: 0.1}, z: 0.25}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.3
  detection: {numerical_aperture: 1.2}
  illumination: {wavelength: 0.532, numerical_aperture: {nominal: 0.9, tolerance: free}}
object: {dimension: 3, prior: tikhonov, weight: 0.001}
solver:
  calibration:
    score: sharpness
    search: [{gradient: {max_iter: 10}}]
    advanced: {optimizer: adam, learning_rates: {illumination.numerical_aperture: 0.01}}
"""


def _phase_3x(**optimization: object) -> dict:
    """A 3.x 3D phase file whose optimizer calibrates the condenser aperture, then sweeps the weight."""
    return {
        "input_channel_names": ["BF"],
        "phase": {
            "transfer_function": {"numerical_aperture_illumination": {"init": 0.5, "lr": 0.01}},
            "apply_inverse": {"auto_regularization": {"rule": "otsu_cnr", "num_samples": 3}},
        },
        "optimization": {"max_iterations": 2, **optimization},
    }


def _fluorescence_3x() -> dict:
    return {
        "input_channel_names": ["GFP"],
        "fluorescence": {"transfer_function": {"numerical_aperture_detection": {"init": 1.2, "lr": 0.01}}},
        "optimization": {"max_iterations": 2},
    }


def _migrated(raw: dict) -> dict:
    result = migrate(raw)
    assert result.rejections == (), result.rejections
    assert result.config is not None
    config = copy.deepcopy(result.config)
    config["solver"]["device"] = None
    return config


@pytest.mark.parametrize(
    ("raw", "steepness"),
    [
        (_phase_3x(), 100.0),
        (_phase_3x(method="nelder_mead"), 100.0),  # wo-4.0 fitted on the soft edge whatever the method
        (yaml.safe_load(PREVIEW), 100.0),
        (_fluorescence_3x(), None),  # wo-4.0's fluorescence optimizer fitted at the default 1e4
    ],
    ids=["3x_adam", "3x_nelder_mead", "wo40_adam", "3x_fluorescence"],
)
def test_a_migrated_optimizer_fits_phase_on_the_soft_edge(raw, steepness):
    config = _migrated(raw)
    assert config["solver"]["calibration"].get("edge_steepness") == steepness
    assert "edge_steepness" not in config["imaging_system"]["detection"]  # the reconstruction's stays at 1e4
    assert ReconstructionModel.from_config(load(config)).calibration_edge_steepness == steepness


@pytest.mark.filterwarnings("ignore::waveorder.errors.CalibrationWarning")  # three weights: the pick is at an end
def test_a_migrated_phase_optimizer_fits_at_100_and_sweeps_and_reconstructs_at_1e4():
    model = ReconstructionModel.from_config(load(_migrated(_phase_3x())))
    detection, size = getattr(model.system, "detection", None), model.pixel_size
    assert isinstance(detection, Detection)
    assert size is not None
    sampling = Sampling(shape=(8, 32, 32), pixel_size=size)
    obj = 0.02 * torch.randn(model.system.object_shape(sampling), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        data = model.system(obj, sampling)
    seen: list[tuple[str, float]] = []

    def record(_: int, stage: object, __: object) -> None:
        seen.append((type(stage).__name__, detection.edge_steepness))

    model.calibrate(data, prepared=True, callback=record)
    assert seen == [("GradientSearch", 100.0), ("WeightSweep", 1e4)]  # the sweep, as wo-4.0's after its optimizer
    assert detection.edge_steepness == 1e4
    assert model.reconstruct(data, prepared=True).estimate.shape == (1, 8, 32, 32)


def _phase_2d(edge: PupilEdge = "ramp", steepness: float = 1e4) -> Phase2D:
    illumination = Illumination(wavelength=0.532, numerical_aperture=0.5)
    detection = Detection(numerical_aperture=1.2, edge=edge, edge_steepness=steepness)
    return Phase2D(illumination, detection, index_of_refraction_media=1.3)


def _recorded(model: Phase2D) -> tuple[str, dict]:
    """Return the store's system hash of ``model`` and the arguments of its detection in the record."""
    record = json.loads(json.dumps(system_record(model)))
    return record["imaging_system_sha256"], record["imaging_system"]["thin_phase_2d"]["detection"]["detection"]


def test_only_a_steepness_other_than_the_default_enters_the_store_hash():
    default, written = _recorded(_phase_2d("sigmoid"))
    assert "edge_steepness" not in written
    soft, written = _recorded(_phase_2d("sigmoid", 100.0))
    assert written["edge_steepness"] == 100.0
    assert soft != default
    # A calibration setting: the transfer functions a store holds are computed at the detection's own steepness.
    model = _phase_2d("sigmoid")
    model.calibration_edge_steepness = 100.0
    assert _recorded(model)[0] == default


def test_the_calibration_steepness_is_saved_with_the_model(tmp_path):
    model = _phase_2d("sigmoid")
    model.calibration_edge_steepness = 100.0
    assert Phase2D.load(model.save(tmp_path / "model")).calibration_edge_steepness == 100.0


def test_a_calibration_steepness_needs_the_sigmoid_edge():
    with pytest.raises(ParameterError) as error:
        _phase_2d().calibration_edge_steepness = 100.0
    assert error.value.path == "calibration_edge_steepness"
    config = _migrated(_phase_3x())
    config["imaging_system"]["detection"]["edge"] = "ramp"
    with pytest.raises(Rejected, match=r"solver\.calibration\.edge_steepness: is the slope of the sigmoid"):
        ReconstructionModel.from_config(load(config))
