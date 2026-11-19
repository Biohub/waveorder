"""wo-4.0 options from YAML end to end: ``object.prior: beads``, ``sigma: estimate`` and the rule-based weight."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.models import ReconstructionModel
from waveorder.nn import CalibrationResult, Sampling, load_parameters
from waveorder.nn.priors import Beads
from waveorder.nn.search import BeadCalibration
from waveorder.nn.search.sweep import WeightSweep

CONFIG = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.532
    numerical_aperture: {nominal: 1.2, tolerance: free, bounds: [1.0, 1.3]}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 1.0}}
solver:
  calibration: {score: evidence, search: [{grid: {points: 7}}]}
"""
APERTURE = "system.detection.numerical_aperture"
PIXEL_SIZE = (0.25, 0.1, 0.1)  # data.pixel_size of CONFIG


def _model(**blocks: object) -> ReconstructionModel:
    raw = yaml.safe_load(CONFIG)
    raw.update(blocks)
    return ReconstructionModel.from_config(V4Config(**raw))


def test_a_beads_prior_builds_and_calibrates_through_the_bead_fit():
    model = _model(
        object={"prior": {"beads": {"diameter": 0.3}}},
        solver={"calibration": {"score": "evidence", "search": [{"gradient": {"max_iter": 20}}]}},
    )
    assert isinstance(model.imaging_model.object_prior, Beads)
    sampling = Sampling(shape=(16, 32, 32), pixel_size=PIXEL_SIZE)
    positions = torch.tensor([(8, 8, 8), (8, 8, 24), (8, 24, 16)], dtype=torch.float32)
    obj = Beads(0.3).render(positions, torch.full((3,), 3000.0), sampling)
    load_parameters(model.imaging_model, {APERTURE: 1.1})
    clean, _ = model.simulate(obj[None], noise=False)
    load_parameters(model.imaging_model, {APERTURE: 1.2})
    data = clean + 10.0 + torch.randn(clean.shape, generator=torch.Generator().manual_seed(3))
    result = model.calibrate(data, prepared=True)
    assert isinstance(result, BeadCalibration)
    assert result.positions.shape == (3, 3)
    assert result.trace.shape == (20,)
    assert float(result.trace[-1]) < float(result.trace[0])
    fitted = float(result.parameters[APERTURE])
    assert float(model.imaging_model.get_parameter(APERTURE).detach()) == pytest.approx(fitted)


def test_an_estimated_sigma_builds_calibrates_and_reconstructs():
    model = _model(noise={"gaussian": {"sigma": "estimate"}})
    data, _ = model.simulate(shape=(8, 32, 32), seed=0)
    result = model.calibrate(data, prepared=True)
    assert isinstance(result, CalibrationResult)
    sigma = float(result.parameters["noise.sigma"])
    assert 0 < sigma <= float(data.std())
    assert float(model.imaging_model.get_parameter("noise.sigma").detach()) == pytest.approx(sigma)
    estimate = model.reconstruct(data, prepared=True).estimate
    assert estimate.shape == (1, 8, 32, 32)
    assert torch.isfinite(estimate).all()


@pytest.mark.parametrize(("rule", "picked"), [("otsu_cnr", 1.0), ("l_curve", 1e-4)])
def test_a_rule_based_weight_builds_sweeps_and_reconstructs(rule, picked):
    imaging_system = yaml.safe_load(CONFIG)["imaging_system"]
    imaging_system["detection"] = {"wavelength": 0.532, "numerical_aperture": 1.2, "edge": "sigmoid"}
    blocks = {"imaging_system": imaging_system, "noise": {"gaussian": {"sigma": 0.01}}, "solver": {}}
    model = _model(object={"prior": "tikhonov", "weight": {"auto": {"rule": rule, "num_samples": 5}}}, **blocks)
    data, _ = model.simulate(shape=(4, 128, 128), seed=1)
    model.calibrate(data, prepared=True)
    weight = float(model.imaging_model.get_parameter("object_prior.weight").detach())
    reference = _model(object={"prior": "tikhonov", "weight": 1e-3}, **blocks).imaging_model
    expected = WeightSweep(rule=rule, samples=5).sweep(
        data, reference, Sampling(shape=(4, 128, 128), pixel_size=PIXEL_SIZE)
    )
    assert weight == pytest.approx(expected.weight, rel=1e-6)
    assert weight == pytest.approx(picked, rel=1e-5)  # inside the sweep of 10^-6 .. 10^2
    estimate = model.reconstruct(data, prepared=True).estimate
    assert estimate.shape == (1, 4, 128, 128)
    assert torch.isfinite(estimate).all()
