"""The verbs of a model on small synthetic data: simulate, train, export, and the compliance paths of calibrate."""

from typing import Any

import pytest
import torch
import yaml

from tests.data.conftest import phase_index, phase_pair
from tests.models.plates import thin_phase_model
from tests.nn.networks import tiny
from waveorder.config import V4Config
from waveorder.errors import ConfigurationError, IncompatibleModelError, ParameterError
from waveorder.io.bundle import load_bundle
from waveorder.models import Fluorescence2D, Fluorescence3D, Phase3D, ReconstructionModel, TrainResult
from waveorder.nn import (
    FISTA,
    Detection,
    Distilled,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    Illumination,
    Sampling,
    Tikhonov,
    ZernikeAberration,
)
from waveorder.nn.preprocess import apodize, most_structured_crop
from waveorder.nn.priors import Beads
from waveorder.nn.search.base import CalibrationResult
from waveorder.nn.search.beads import BeadCalibration
from waveorder.nn.search.sweep import WeightSweep
from waveorder.tiling import TilePlan

__all__ = ["phase_index", "phase_pair"]  # the fixtures of the data tests, used by the training test

PIXEL_SIZE = (0.25, 0.1, 0.1)
APERTURE = "system.detection.numerical_aperture"


def _fluorescence(numerical_aperture: float | Free = 1.2, **parts: object) -> Fluorescence3D:
    detection = Detection(wavelength=0.532, numerical_aperture=numerical_aperture, edge="sigmoid")
    settings: dict[str, Any] = {
        "object_prior": Tikhonov(1e-3),
        "noise": Gaussian(0.01),
        "pixel_size": PIXEL_SIZE,
        **parts,
    }
    return Fluorescence3D(detection, index_of_refraction_media=1.3, **settings)


# --- simulate ---


def test_simulate_images_an_object_with_the_models_noise_and_repeats_for_a_seed():
    model = _fluorescence()
    data, obj = model.simulate(shape=(8, 32, 32), seed=3)
    assert (data.shape, obj.shape) == ((1, 8, 32, 32), (1, 8, 32, 32))
    assert float(obj.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(obj.std()) == pytest.approx(1.0, rel=1e-5)  # the power-law field: zero mean, unit deviation
    again, _ = model.simulate(shape=(8, 32, 32), seed=3)
    assert torch.equal(data, again)
    clean, same = model.simulate(obj, noise=False)
    assert torch.equal(same, obj)
    with torch.no_grad():
        assert torch.equal(clean, model.imaging_model(obj, Sampling(shape=(8, 32, 32), pixel_size=PIXEL_SIZE)))
    assert float((data - clean).std()) == pytest.approx(0.01, rel=0.1)  # Gaussian(0.01)
    with pytest.raises(ParameterError, match="needs an object or the shape"):
        model.simulate()


def test_a_simulation_configuration_gives_the_shape_the_seed_and_the_signal_to_noise():
    text = """
version: 4
data: {pixel_size: {z: 0.25, y: 0.1, x: 0.1}}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.3
  detection: {wavelength: 0.532, numerical_aperture: 1.2}
object: {source: {gaussian_spectrum: {exponent: 3.0}}}
noise: {gaussian: {snr_db: 20}}
simulate: {seed: 5, shape: [8, 32, 32]}
"""
    model = ReconstructionModel.from_config(V4Config(**yaml.safe_load(text)))
    assert model.pixel_size == PIXEL_SIZE
    data, obj = model.simulate()
    assert data.shape == (1, 8, 32, 32)
    clean, _ = model.simulate(obj, noise=False)
    assert float((data - clean).std() / clean.std()) == pytest.approx(0.1, rel=0.1)  # 20 dB
    assert torch.equal(model.simulate()[0], data)  # the configuration's seed


# --- train and export ---


def test_train_fits_on_an_index_and_returns_a_new_model_whose_solver_runs_the_trained_network(phase_index, tmp_path):
    index = phase_index.save(tmp_path / "index")
    model = thin_phase_model(pixel_size=(0.4, 0.15))
    solver, before = model.solver, {name: value.clone() for name, value in model.imaging_model.state_dict().items()}
    cpu = {"accelerator": "cpu", "enable_progress_bar": False}
    result = model.train(index, student=tiny.strided(5), out=tmp_path / "run", patch=32, epochs=1, trainer=cpu)
    assert isinstance(result, TrainResult)
    assert (result.directory, result.bundle) == (tmp_path / "run", tmp_path / "run" / "best")
    assert result.checkpoint.is_file()
    assert (result.directory / "config.yaml").is_file()
    assert {"train/loss", "val/loss", "val/ssim"} <= set(result.metrics)
    assert model.solver is solver  # the original is untouched
    assert not isinstance(solver, Distilled)
    assert all(torch.equal(before[name], value) for name, value in model.imaging_model.state_dict().items())
    assert isinstance(result.model.solver, Distilled)
    data = 1000 + torch.rand(1, 5, 64, 64, generator=torch.Generator().manual_seed(0))
    estimate = result.model.reconstruct(data).estimate
    assert estimate.shape == (2, 1, 64, 64)
    assert estimate[0].abs().max() > 0
    saved = thin_phase_model(pixel_size=(0.4, 0.15), solver=Distilled(load_bundle(result.bundle)))
    torch.testing.assert_close(estimate, saved.reconstruct(data).estimate)  # the network the run saved
    onnx = result.model.export(tmp_path / "student.onnx")
    assert onnx.is_file()
    with pytest.raises(ConfigurationError, match="export writes a network"):
        model.export(tmp_path / "wiener.onnx")


def test_train_refuses_an_argument_at_its_configuration_key(phase_index, tmp_path):
    index = phase_index.save(tmp_path / "index")
    model = thin_phase_model(pixel_size=(0.4, 0.15))
    with pytest.raises(ConfigurationError, match=r"^data\.batch_size: input should be greater than 0$"):
        model.train(index, student=tiny.strided(5), out=tmp_path / "run", batch_size=0)
    with pytest.raises(ConfigurationError, match=r"^model\.weights: ParameterError: weights names \['l2'\]"):
        model.train(index, student=tiny.strided(5), out=tmp_path / "run", module={"weights": {"l2": 1.0}})


# --- apodization ---


def test_the_apodization_rolloff_filters_a_wiener_estimate_and_is_refused_elsewhere():
    data, _ = _fluorescence().simulate(shape=(8, 32, 32), seed=0)
    plain = _fluorescence().reconstruct(data, prepared=True).estimate
    rolled = _fluorescence(apodization_rolloff=0.3).reconstruct(data, prepared=True).estimate
    torch.testing.assert_close(rolled, apodize(plain, 0.3))
    with pytest.raises(ConfigurationError, match="applies to a Wiener estimate"):
        _fluorescence(apodization_rolloff=0.3, solver=FISTA()).reconstruct(data, prepared=True)


# --- the rule-based weight ---


def test_a_weight_sweep_picks_the_weight_on_the_most_structured_crop():
    sweep = WeightSweep(rule="otsu_cnr", samples=5)
    weight = Tikhonov(Free(1e-3, bounds=(1e-12, 1e12)))
    model = _fluorescence(object_prior=weight, search=[sweep], score=None)
    data, _ = model.simulate(shape=(4, 300, 300), seed=1)
    model.calibrate(data, prepared=True)
    y, x, side = most_structured_crop(data, size=256)
    crop = data[..., y : y + side, x : x + side]
    reference = _fluorescence()
    expected = sweep.sweep(crop, reference.imaging_model, Sampling(shape=(4, side, side), pixel_size=PIXEL_SIZE))
    picked = float(model.imaging_model.get_parameter("object_prior.weight").detach())
    assert picked == pytest.approx(expected.weight, rel=1e-6)


def test_a_weight_sweep_normalizes_its_crop_by_the_crops_own_mean_as_wo40():
    # An illumination ramp: the dim crop's mean is not the stack's, which offsets the data and moves the L-curve.
    def phase(**parts: Any) -> Phase3D:
        illumination = Illumination(wavelength=0.532, numerical_aperture=0.5)
        detection = Detection(wavelength=0.532, numerical_aperture=1.2, edge="sigmoid")
        return Phase3D(illumination, detection, index_of_refraction_media=1.33, pixel_size=PIXEL_SIZE, **parts)

    contrast, _ = phase(object_prior=Tikhonov(1e-3), noise=Gaussian(0.01)).simulate(shape=(6, 64, 320), seed=0)
    recorded = (1 + contrast) * torch.linspace(0.9, 1.1, 320)
    recorded[..., :64] += 0.3 * contrast[..., :64]  # the most structured crop is on the dim left side
    sweep = WeightSweep(rule="l_curve", search_min=-4, samples=9)  # picks index 6; normalized whole, it picks 7
    model = phase(object_prior=Tikhonov(Free(1e-3, bounds=(1e-12, 1e12))), noise=Gaussian(0.01), search=[sweep])
    result = model.calibrate(recorded, history=True)
    assert isinstance(result, CalibrationResult)
    assert result.sweep is not None
    assert result.sweep.region == most_structured_crop(recorded, size=256) == (0, 0, 64)
    crop = recorded[..., :64]
    reference = phase()
    sampling = Sampling(shape=(6, 64, 64), pixel_size=PIXEL_SIZE)
    expected = sweep.sweep(reference.prepare(crop), reference.imaging_model, sampling)
    assert result.sweep.residual_norms == pytest.approx(expected.residual_norms, rel=1e-5)
    assert result.sweep.index == expected.index == 6
    assert result.sweep.weight == pytest.approx(expected.weight, rel=1e-6)


# --- sigma estimated from the data ---


def test_an_estimated_sigma_starts_from_the_data_before_a_store_becomes_the_prior():
    truth = _fluorescence(1.2)
    data, _ = truth.simulate(shape=(8, 32, 32), seed=2)
    model = _fluorescence(Free(1.1, bounds=(1.0, 1.3)), noise=Gaussian("estimate"), search=[GridSearch(points=7)])
    result = model.calibrate(data, prepared=True, prior=model.to_tf_store())
    sigma = float(result.parameters["noise.sigma"])
    spread = float(data.std())
    assert 1e-3 * spread <= sigma <= spread  # within the bounds start_from set from the data
    assert float(result.parameters[APERTURE]) == pytest.approx(1.2, abs=0.05)


def test_a_tiled_calibration_estimates_the_noise_level_as_the_whole_field_does():
    data = 100 * torch.randn(1, 8, 32, 32, generator=torch.Generator().manual_seed(0)) + 500

    def model() -> Fluorescence3D:
        aperture = Free(1.1, bounds=(1.0, 1.3))
        return _fluorescence(aperture, noise=Gaussian("estimate"), search=[GridSearch(points=5)])

    whole = float(model().calibrate(data, prepared=True).parameters["noise.sigma"])
    tiled = model()
    plan = tiled.tile_plan((8, 32, 32), tile=(8, 16, 16))
    sigma = tiled.calibrate_tiles(data, plan=plan, prepared=True).parameters["noise.sigma"]
    assert sigma.shape == (len(plan.regions),)
    assert whole == pytest.approx(100, rel=0.1)
    assert sigma.tolist() == pytest.approx([whole] * len(plan.regions), rel=0.1)


# --- beads ---


def _bead_model(astigmatism: float | Free, **parts: object) -> Fluorescence3D:
    aberration = ZernikeAberration({"astigmatism_0": astigmatism})
    detection = Detection(wavelength=0.532, numerical_aperture=1.2, edge="sigmoid", pupil_aberrations=aberration)
    settings: dict[str, Any] = {"object_prior": Beads(0.3), "noise": Gaussian(1.0), "pixel_size": PIXEL_SIZE, **parts}
    return Fluorescence3D(detection, index_of_refraction_media=1.3, **settings).to(dtype=torch.float64)


def test_a_beads_prior_calibrates_through_the_bead_fit_with_wo40s_refusals():
    sampling = Sampling(shape=(16, 32, 32), pixel_size=PIXEL_SIZE)
    positions = torch.tensor([(8, 8, 8), (8, 8, 24), (8, 24, 16)], dtype=torch.float64)
    obj = Beads(0.3).render(positions, torch.full((3,), 3000.0, dtype=torch.float64), sampling)
    clean, _ = _bead_model(0.15).simulate(obj[None], noise=False)
    data = clean + 10.0 + torch.randn(clean.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    model = _bead_model(Free(0.0), search=[GradientSearch(iterations=150, learning_rate=0.01)])
    result = model.calibrate(data, prepared=True)
    assert isinstance(result, BeadCalibration)
    assert float(result.parameters["system.detection.pupil_aberrations.astigmatism_0"]) == pytest.approx(0.15, abs=0.01)
    assert result.trace.shape == (150,)
    with pytest.raises(ConfigurationError, match="bead calibration runs one gradient search"):
        _bead_model(Free(0.0), search=[GridSearch(points=3)]).calibrate(data, prepared=True)
    with pytest.raises(ParameterError, match="prior with prior: beads is not supported yet"):
        model.calibrate(data, prepared=True, prior=model.to_tf_store())
    phase = Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=Free(0.5, bounds=(0.3, 0.7))),
        Detection(wavelength=0.532, numerical_aperture=1.2),
        index_of_refraction_media=1.3,
        object_prior=Beads(0.3),
        noise=Gaussian(1.0),
        pixel_size=PIXEL_SIZE,
    )
    with pytest.raises(ConfigurationError, match="bead calibration supports fluorescence contrast"):
        phase.calibrate(data, prepared=True)
    thin = Fluorescence2D(
        Detection(wavelength=0.532, numerical_aperture=Free(1.2, bounds=(1.0, 1.3))),
        index_of_refraction_media=1.3,
        object_prior=Beads(0.2),
        noise=Gaussian(1.0),
        pixel_size=PIXEL_SIZE,
    )
    with pytest.raises(ConfigurationError, match="it fits a 3D bead volume, which a ThinFluorescence2D does not"):
        thin.check((8, 32, 32), calibration=True)


def test_a_beads_prior_calibrates_and_does_not_reconstruct():
    with pytest.raises(IncompatibleModelError, match="a Beads object prior calibrates the system"):
        _bead_model(Free(0.0), solver=FISTA())  # refused when the model is built
    model = _bead_model(Free(0.0))
    model.check((8, 32, 32), calibration=True)
    data = torch.rand(1, 8, 32, 32, dtype=torch.float64)
    for refused in (
        lambda: model.check((8, 32, 32)),
        lambda: model.reconstruct(data),
        lambda: model.reconstruct_tiles(
            [(data, torch.zeros_like(data))], plan=TilePlan(shape=(8, 32, 32), tile=(8, 32, 32))
        ),
    ):
        with pytest.raises(IncompatibleModelError, match="it does not reconstruct"):
            refused()
    tikhonov = _fluorescence(solver=FISTA())
    with pytest.raises(IncompatibleModelError, match="it does not reconstruct"):
        tikhonov.object_prior = Beads(0.3)  # the solver it was given cannot run with it
    assert isinstance(tikhonov.object_prior, Tikhonov)
