"""``solver.calibration.score``: gcv and sharpness (the 3.x losses of ``advanced.loss``)."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.errors import Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import FISTA, CalibrationResult, Sampling, Score, Wiener
from waveorder.nn.scores import (
    GCV,
    LaplacianVarianceScore,
    MidbandPowerScore,
    NormalizedVarianceScore,
    SpectralFlatnessScore,
    TotalVariationScore,
)

FLUORESCENCE = """
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
noise: {gaussian: {sigma: 2.0}}
solver:
  calibration: {score: evidence, search: [{grid: {points: 3}}]}
"""
PHASE = """
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection: {numerical_aperture: 0.9}
  illumination:
    wavelength: 0.532
    numerical_aperture: {nominal: 0.5, tolerance: 0.05, bounds: [0.35, 0.7]}
object: {prior: tikhonov, weight: 1e-6}
noise: {gaussian: {sigma: 1e-4}}
solver:
  calibration: {score: sharpness, search: [{grid: {points: 3}}]}
"""


def _raw(text: str = FLUORESCENCE, *, score: object = "evidence", loss: object = None) -> dict:
    raw = yaml.safe_load(text)
    raw["solver"]["calibration"]["score"] = score
    if loss is not None:
        raw["solver"]["calibration"]["advanced"] = {"loss": loss}
    return raw


def _refusal(raw: dict) -> str:
    """Return the message of the one problem ``from_config`` refuses ``raw`` with."""
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(raw)
    (problem,) = rejected.value.problems
    return problem.message


def _weight_calibration(score: str) -> tuple[Score | None, CalibrationResult]:
    raw = _raw(score=score)
    raw["object"]["weight"] = "auto"
    raw["imaging_system"]["detection"]["numerical_aperture"] = 1.2
    raw["solver"]["calibration"]["search"] = [{"grid": {"points": 5}}, "gradient"]
    model = ReconstructionModel.from_config(V4Config(**raw))
    sampling = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.12))  # data.pixel_size of FLUORESCENCE
    generator = torch.Generator().manual_seed(0)
    obj = torch.rand(1, *sampling.shape, generator=generator)
    with torch.no_grad():
        clean = model.imaging_model(obj, sampling)
    data = clean + 2.0 * torch.randn(clean.shape, generator=generator)  # the configured sigma
    result = model.calibrate(data, pixel_size=sampling.pixel_size, prepared=True)
    assert isinstance(result, CalibrationResult)
    return model.score, result


def test_gcv_builds_its_score_and_estimates_the_noise_variance():
    gcv, by_gcv = _weight_calibration("gcv")
    assert isinstance(gcv, GCV)
    # At its minimum GCV estimates the noise variance, 2.0^2 (measured 4.07).
    assert float(by_gcv.objective) == pytest.approx(4.0, rel=0.05)


def test_sharpness_without_a_loss_is_the_3x_default_midband_power_at_the_emission_cutoff():
    model = ReconstructionModel.from_config(V4Config(**_raw(score="sharpness")))
    score = model.score
    assert isinstance(score, MidbandPowerScore)
    assert score.cutoff == pytest.approx(2 * 1.2 / 0.52)
    assert score.fractions == (0.125, 0.25)
    assert isinstance(score.solver, Wiener)
    assert score.fields is None  # one fluorescence field: every field


def test_sharpness_on_phase_uses_the_illumination_wavelength_and_the_configured_solver():
    raw = yaml.safe_load(PHASE)
    raw["object"] = {"prior": "tikhonov", "weight": 1e-3}
    raw["solver"]["reconstruction"] = {"fista": {"iterations": 3}}
    score = ReconstructionModel.from_config(V4Config(**raw)).score
    assert isinstance(score, MidbandPowerScore)
    assert score.cutoff == pytest.approx(2 * 0.9 / 0.532)
    assert isinstance(score.solver, FISTA)


def test_sharpness_scores_the_solver_the_model_chooses_when_none_is_written():
    raw = _raw(score="sharpness")
    raw["object"] = {"prior": "tv", "weight": 1e-3}  # no solver.reconstruction: the model chooses FISTA for it
    model = ReconstructionModel.from_config(V4Config(**raw))
    score = model.score
    assert isinstance(score, MidbandPowerScore)
    assert isinstance(model.solver, FISTA)
    assert isinstance(score.solver, FISTA)
    assert score.unfilled == ()
    assert score.cutoff == pytest.approx(2 * 1.2 / 0.52)  # the system's cutoff, as with a written solver


def test_sharpness_on_thin_phase_scores_the_phase_alone_as_3x_did():
    raw = yaml.safe_load(PHASE)
    raw["object"] = {"prior": "tikhonov", "weight": 1e-3, "dimension": 2}
    model = ReconstructionModel.from_config(V4Config(**raw))
    score = model.score
    assert isinstance(score, MidbandPowerScore)
    sampling = Sampling(shape=(5, 32, 32), pixel_size=(0.4, 0.15, 0.15))  # data.pixel_size of PHASE
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        data = model.imaging_model(torch.randn(2, 1, 32, 32, generator=generator), sampling)  # phase and absorption
        configured = score(data, model.imaging_model, sampling)
        cutoff, fractions = score.cutoff, score.fractions
        phase_only = MidbandPowerScore(solver=Wiener(), cutoff=cutoff, fractions=fractions, fields=("phase",))
        every_field = MidbandPowerScore(solver=Wiener(), cutoff=cutoff, fractions=fractions)
        phase_only, every_field = (each(data, model.imaging_model, sampling) for each in (phase_only, every_field))
    torch.testing.assert_close(configured, phase_only, rtol=0, atol=0)
    assert not torch.isclose(configured, every_field)  # absorption would change the loss


@pytest.mark.parametrize(
    ("loss", "kind"),
    [
        ({"type": "total_variation"}, TotalVariationScore),
        ({"type": "laplacian_variance"}, LaplacianVarianceScore),
        ({"type": "normalized_variance"}, NormalizedVarianceScore),
        ({"type": "spectral_flatness"}, SpectralFlatnessScore),
        ({"type": "midband_power", "midband_fractions": [0.1, 0.3]}, MidbandPowerScore),
    ],
    ids=lambda value: value["type"] if isinstance(value, dict) else value.__name__,
)
def test_each_3x_loss_type_builds_its_score(loss, kind):
    score = ReconstructionModel.from_config(V4Config(**_raw(score="sharpness", loss=loss))).score
    assert type(score) is kind
    if "midband_fractions" in loss:
        assert isinstance(score, MidbandPowerScore)
        assert score.fractions == (0.1, 0.3)


@pytest.mark.parametrize(
    ("loss", "location"),
    [
        ({"type": "brenner"}, "solver.calibration.advanced.loss.type"),
        (
            {"type": "total_variation", "midband_fractions": [0.1, 0.3]},
            "solver.calibration.advanced.loss.midband_fractions",
        ),
        ({"type": "midband_power", "weight": 2}, "solver.calibration.advanced.loss.weight"),
        (
            {"type": "midband_power", "midband_fractions": [0.3, 0.1]},
            "solver.calibration.advanced.loss.midband_fractions",
        ),
    ],
    ids=["unknown_type", "fractions_on_a_spatial_loss", "unknown_key", "decreasing_fractions"],
)
def test_a_loss_setting_with_no_4x_meaning_is_refused_at_its_key(loss, location):
    assert _refusal(_raw(score="sharpness", loss=loss)).startswith(f"{location}: ")


def test_a_loss_with_a_score_that_does_not_read_it_is_refused():
    message = _refusal(_raw(score="evidence", loss={"type": "total_variation"}))
    assert message == "solver.calibration.advanced.loss: is a sharpness setting, and score is evidence; remove it"
