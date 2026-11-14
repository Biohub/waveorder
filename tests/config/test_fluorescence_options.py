"""``contrast: fluorescence``: thin (``object.dimension: 2``) and confocal (``detection.pinhole``) systems."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.errors import Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import (
    ConfocalFluorescence3D,
    Detection,
    Sampling,
    ThinFluorescence2D,
    free_specs,
    load_parameters,
)

CONFIG = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection: {wavelength: 0.52, numerical_aperture: 1.2, z_focus_offset: 1.0}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
"""
SHAPE = (8, 32, 32)
PIXEL_SIZE = (0.25, 0.1, 0.1)  # data.pixel_size of CONFIG


def _raw(**detection: object) -> dict:
    raw = yaml.safe_load(CONFIG)
    raw["imaging_system"]["detection"].update(detection)
    return raw


def _detection() -> Detection:
    return Detection(wavelength=0.52, numerical_aperture=1.2, z_focus_offset=1.0)


def _same_images(model, system, object_shape) -> None:
    """Check the configured model images an object as ``system`` does, on the configured grid."""
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    obj = torch.rand(*object_shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(model.system(obj, sampling), system(obj, sampling))


def test_a_2d_object_builds_thin_fluorescence_with_the_configured_detection():
    raw = _raw()
    raw["object"]["dimension"] = 2
    model = ReconstructionModel.from_config(V4Config(**raw))
    assert isinstance(model.system, ThinFluorescence2D)
    _same_images(model, ThinFluorescence2D(_detection(), index_of_refraction_media=1.33), (1, 1, *SHAPE[1:]))
    estimate = model.reconstruct(torch.ones(1, *SHAPE), pixel_size=PIXEL_SIZE, prepared=True).estimate
    assert estimate.shape == (1, 1, *SHAPE[1:])


@pytest.mark.parametrize(
    ("pinhole", "diameter"),
    [("point", 0.0), ({"diameter_um": 0.5}, 0.5), ({"diameter_au": 1.0}, 1.22 * 0.52 / 1.2)],
    ids=["point", "um", "airy_units"],
)
def test_a_pinhole_builds_confocal_fluorescence_with_its_diameter_in_um(pinhole, diameter):
    model = ReconstructionModel.from_config(V4Config(**_raw(pinhole=pinhole)))
    system = model.system
    assert isinstance(system, ConfocalFluorescence3D)
    expected = ConfocalFluorescence3D(_detection(), pinhole_diameter=diameter, index_of_refraction_media=1.33)
    _same_images(model, expected, (1, *SHAPE))


def test_a_free_airy_unit_diameter_is_the_um_parameter_with_converted_bounds():
    unit = 1.22 * 0.52 / 1.2
    pinhole = {"diameter_au": {"nominal": 1.0, "tolerance": 0.2, "bounds": [0.5, 2.0]}}
    model = ReconstructionModel.from_config(V4Config(**_raw(pinhole=pinhole)))
    spec = free_specs(model.imaging_model)["system.pinhole_diameter"]
    assert spec.nominal == pytest.approx(unit)
    assert spec.tolerance == pytest.approx(0.2 * unit)
    assert spec.bounds == pytest.approx((0.5 * unit, 2.0 * unit))


def test_a_free_diameter_is_calibrated_from_confocal_data():
    # An object drawn from the prior (unit variance: weight = sigma^2) with noise at sigma, so the evidence is
    # lowest at the true diameter (measured -13336 at 0.3 against -11566 and -11320 at its grid neighbours).
    raw = _raw(pinhole={"diameter_um": {"nominal": 0.5, "tolerance": "free", "bounds": [0.1, 0.9]}})
    raw["object"]["weight"] = 1e-4
    raw["solver"] = {"calibration": {"score": "evidence", "search": [{"grid": {"points": 5}}]}}
    model = ReconstructionModel.from_config(V4Config(**raw))
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    generator = torch.Generator().manual_seed(0)
    load_parameters(model.imaging_model, {"system.pinhole_diameter": 0.3})
    with torch.no_grad():
        data = model.imaging_model(torch.randn(1, *SHAPE, generator=generator), sampling)
    data += 0.01 * torch.randn(data.shape, generator=generator)
    load_parameters(model.imaging_model, {"system.pinhole_diameter": 0.5})
    result = model.calibrate(data, pixel_size=PIXEL_SIZE, prepared=True)
    assert float(result.parameters["system.pinhole_diameter"]) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("pinhole", "location"),
    [
        ({"diameter_um": {"nominal": 0.5, "bounds": [-0.1, 1.0]}}, "imaging_system.detection.pinhole.diameter_um"),
        ({"diameter_um": -0.5}, "imaging_system.detection.pinhole.diameter_um"),
    ],
    ids=["negative_bound", "negative"],
)
def test_an_impossible_diameter_is_refused_at_its_key(pinhole, location):
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(_raw(pinhole=pinhole))
    (problem,) = rejected.value.problems
    assert problem.message.startswith(f"{location}: ")
