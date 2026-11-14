"""``contrast: phase`` configs build the 3.x phase systems, with the illumination block's free quantities."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.errors import Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import Evidence, GridSearch, Sampling, ThickPhase3D, ThinPhase2D, Wiener, free_specs

PHASE = """
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection:
    numerical_aperture: 0.9
  illumination:
    wavelength: 0.532
    numerical_aperture: {nominal: 0.5, tolerance: 0.05, bounds: [0.35, 0.7]}
    tilt:
      zenith: {nominal: 0.1, tolerance: 0.05, bounds: [0.0, 0.3]}
      azimuth: 0.5
    invert_phase_contrast: true
object: {prior: tikhonov, weight: 1e-6}
noise: {gaussian: {sigma: 1e-4}}
solver:
  calibration: {score: evidence, search: [{grid: {points: 3}}]}
"""
PIXEL_SIZE = (0.4, 0.15, 0.15)  # data.pixel_size of PHASE


def _raw(**overrides: object) -> dict:
    raw = yaml.safe_load(PHASE)
    for path, value in overrides.items():
        *parents, leaf = path.split("__")
        node = raw
        for key in parents:
            node = node.setdefault(key, {})
        if value is None:
            node.pop(leaf)
        else:
            node[leaf] = value
    return raw


def _config(**overrides: object) -> V4Config:
    return V4Config(**_raw(**overrides))


def test_a_3d_phase_config_builds_the_thick_system_and_calibrates_its_illumination():
    model = ReconstructionModel.from_config(_config())
    system = model.system
    assert isinstance(system, ThickPhase3D)
    assert system.fields == ("phase",)
    assert system.invert_phase_contrast
    assert system.detection.wavelength == system.illumination.wavelength == 0.532
    assert set(free_specs(model.imaging_model)) == {
        "system.illumination.numerical_aperture",
        "system.illumination.tilt.zenith",
    }
    assert isinstance(model.solver, Wiener)
    assert isinstance(model.score, Evidence)
    assert model.search is not None
    assert isinstance(model.search[0], GridSearch)
    sampling = Sampling(shape=(6, 12, 12), pixel_size=PIXEL_SIZE)
    obj = 0.02 * torch.randn(1, *sampling.shape, generator=torch.Generator().manual_seed(0))
    data = model.imaging_model(obj, sampling)
    result = model.calibrate(data, pixel_size=PIXEL_SIZE, prepared=True)
    assert 0.0 <= float(result.parameters["system.illumination.tilt.zenith"]) <= 0.3
    assert model.solver(data, model.imaging_model, sampling).estimate.shape == (1, 6, 12, 12)


def test_a_2d_phase_config_builds_the_thin_system_with_a_closed_form():
    model = ReconstructionModel.from_config(_config(object__dimension=2, solver={}))
    system = model.system
    assert isinstance(system, ThinPhase2D)
    assert system.fields == ("phase", "absorption")
    sampling = Sampling(shape=(5, 12, 12), pixel_size=PIXEL_SIZE)
    data = torch.randn(1, *sampling.shape, generator=torch.Generator().manual_seed(1))
    assert model.solver(data, model.imaging_model, sampling).estimate.shape == (2, 1, 12, 12)


def test_a_free_azimuth_is_the_system_parameter_under_its_yaml_path():
    config = _config(imaging_system__illumination__tilt__azimuth={"nominal": 0.5, "tolerance": 0.1})
    assert "system.illumination.tilt.azimuth" in free_specs(ReconstructionModel.from_config(config).imaging_model)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"imaging_system__illumination__wavelength": None}, "imaging_system.illumination.wavelength"),
        ({"imaging_system__illumination__numerical_aperture": None}, "imaging_system.illumination.numerical_aperture"),
        ({"imaging_system__detection__wavelength": 0.6}, "imaging_system.detection.wavelength"),
        (
            {"imaging_system__illumination__tilt__zenith": {"nominal": 0.1, "tolerance": 0.05}},
            "imaging_system.illumination.tilt.zenith",
        ),
        ({"imaging_system__illumination__tilt__zenith": 0.8}, "imaging_system.illumination.tilt.zenith"),
        (
            {
                "object__dimension": 2,
                "imaging_system__illumination__numerical_aperture": {"nominal": 0.8, "bounds": [0.5, 0.95]},
            },
            "imaging_system.illumination.numerical_aperture",
        ),
    ],
    ids=[
        "no-wavelength",
        "no-condenser-aperture",
        "detection-wavelength-differs",
        "unbounded-free-zenith",
        "darkfield-tilt",
        "thin-condenser-reaches-objective",
    ],
)
def test_configs_the_phase_modules_cannot_represent_are_refused_at_their_field(overrides, field):
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(_raw(**overrides))
    (problem,) = rejected.value.problems
    assert problem.message.startswith(f"{field}: ")
