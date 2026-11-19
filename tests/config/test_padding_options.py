"""``solver.padding.z``: the model pads each solve with mirrored edge slices (3.x ``z_padding``) and crops."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.errors import CalibrationWarning, ConfigurationError
from waveorder.models import ReconstructionModel
from waveorder.nn import CalibrationResult, Sampling, ZPadding

CONFIG = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection:
    wavelength: 0.52
    numerical_aperture: {nominal: 1.2, tolerance: free, bounds: [1.0, 1.3]}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
solver:
  padding: {z: 3}
  calibration: {score: evidence, search: [{grid: {points: 3}}]}
"""
SHAPE = (8, 16, 16)
PIXEL_SIZE = (0.25, 0.1, 0.1)  # data.pixel_size of CONFIG
SAMPLING = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)


def _model(z: int = 3) -> ReconstructionModel:
    raw = yaml.safe_load(CONFIG)
    raw["solver"]["padding"]["z"] = z
    return ReconstructionModel.from_config(V4Config(**raw))


def _data(model: ReconstructionModel) -> torch.Tensor:
    with torch.no_grad():
        return model.imaging_model(torch.rand(1, *SHAPE, generator=torch.Generator().manual_seed(0)), SAMPLING)


def test_reconstruct_solves_on_the_padded_grid_and_keeps_the_measured_slices():
    model = _model()
    data = _data(model)
    padding = ZPadding(3)
    padded = padding.sampling(SAMPLING)
    expected = padding.crop(model.solver(padding.pad(data), model.imaging_model, padded)).estimate
    estimate = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    assert estimate.shape == (1, *SHAPE)
    torch.testing.assert_close(estimate, expected, rtol=0, atol=0)
    unpadded = _model(0).reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    assert (estimate - unpadded).norm() > 1e-3 * unpadded.norm()  # the padding changed the solve


def test_calibrate_scores_the_padded_data():
    model = _model()
    data = _data(model)
    # The evidence takes the mirrored slices for data, which biases the estimate; the model says so.
    with pytest.warns(CalibrationWarning, match="mirrored slices"):
        result = model.calibrate(data, pixel_size=PIXEL_SIZE, prepared=True)
    padding = ZPadding(3)
    score = _model(0).score
    assert score is not None
    by_hand = score(padding.pad(data), model.imaging_model, padding.sampling(SAMPLING))
    assert isinstance(result, CalibrationResult)
    torch.testing.assert_close(result.objective, by_hand)


def test_reconstruct_tiles_pads_whole_stacks_and_refuses_a_plan_that_splits_z_at_its_field():
    model = _model()
    data = _data(model)
    whole = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    sink = torch.zeros_like(data)
    plan = model.tile_plan(SHAPE, pixel_size=PIXEL_SIZE, tile=SHAPE)
    model.reconstruct_tiles([(data, sink)], pixel_size=PIXEL_SIZE, plan=plan)
    assert (sink - whole).norm() <= 1e-5 * whole.norm()
    split = model.tile_plan(SHAPE, pixel_size=PIXEL_SIZE, tile=(4, 8, 8))
    with pytest.raises(ConfigurationError) as error:
        model.reconstruct_tiles([(data, torch.zeros_like(data))], pixel_size=PIXEL_SIZE, plan=split)
    assert error.value.field == "solver.padding.z"
