"""Bead calibration (wo-4.0 ``object.prior: {beads: ...}``): the sphere, the detection and the joint fit."""

import math
from typing import Any

import pytest
import torch

from waveorder.errors import CalibrationWarning, IncompatibleModelError, ParameterError
from waveorder.nn import (
    Detection,
    Free,
    Gaussian,
    ImagingModel,
    Sampling,
    ThinFluorescence2D,
    Tikhonov,
    WidefieldFluorescence3D,
    ZernikeAberration,
)
from waveorder.nn.noise import estimate_sigma
from waveorder.nn.priors import Beads
from waveorder.nn.search import calibrate_beads

# Nyquist-sampled for NA 1.2 at 0.532 um (lateral 0.111 um, axial 0.333 um), so the transfer function is computed on
# the data's own grid in both engines and, with the sigmoid edge, they agree (wo-4.0 computes it there always).
SAMPLING = Sampling(shape=(16, 32, 32), pixel_size=(0.25, 0.1, 0.1))
POSITIONS = [(8, 8, 8), (8, 8, 24), (8, 24, 16)]
DIAMETER = 0.3

# wo-4.0 calibrate.beads.sphere_form_factor((16, 32, 32), 0.1, 0.25, 0.3) at [z, y, x], float32.
OLD_FORM = {(0, 0, 1): 0.9913532137870789, (1, 2, 3): 0.8865720629692078, (3, 5, 7): 0.4588157534599304,
            (8, 16, 16): -0.04465780407190323, (5, 10, 2): 0.27960553765296936}  # fmt: skip
# wo-4.0 run_bead_calibration on its own rendering of this volume (fluorescence_otf, astigmatism_0 = 0.15, three
# beads of 3000 over 10, unit noise from seed 3), one gradient stage of 150 steps, astigmatism_0 free from 0.
OLD_DETECTED = [[8, 25, 16], [8, 8, 8], [8, 8, 24]]
OLD_FIT: dict[float | str, dict[str, Any]] = {
    1.0: {"astigmatism": 0.15278529523698084, "background": 9.997584373338553, "sigma": 1.0,
          "positions": [[7.9933, 23.9552, 15.964], [7.9923, 8.0373, 7.9235], [8.0018, 7.9935, 23.9502]],
          "amplitudes": [3086.034, 3030.074, 3024.029], "trace": (26483.227707264607, 23344.611212296528)},
    "estimate": {"astigmatism": 0.15254161455589035, "background": 10.000050211046494, "sigma": 1.0057659839643303,
                 "positions": [[7.9935, 23.9552, 15.964], [7.9924, 8.0372, 7.9236], [8.0019, 7.9937, 23.95]],
                 "amplitudes": [3075.994, 3020.575, 3009.939], "trace": (26101.553869286785, 23344.03362791425)},
}  # fmt: skip


def _system(astigmatism: float | Free) -> WidefieldFluorescence3D:
    aberration = ZernikeAberration({"astigmatism_0": astigmatism})
    detection = Detection(wavelength=0.532, numerical_aperture=1.2, edge="sigmoid", pupil_aberrations=aberration)
    return WidefieldFluorescence3D(detection, index_of_refraction_media=1.3).double()


def _bead_volume() -> torch.Tensor:
    """The wo-4.0 fixture rendered with this engine: ``(1, Z, Y, X)`` float64."""
    beads = Beads(DIAMETER)
    obj = beads.render(torch.tensor(POSITIONS, dtype=torch.float64), torch.full((3,), 3000.0, dtype=torch.float64),
                       SAMPLING)  # fmt: skip
    clean = _system(0.15)(obj[None], SAMPLING) + 10.0
    return clean + torch.randn(SAMPLING.shape, generator=torch.Generator().manual_seed(3)).double()


def test_the_form_factor_is_the_wo40_sphere() -> None:
    form = Beads(DIAMETER).form_factor(SAMPLING, dtype=torch.float64, device=torch.device("cpu"))
    assert form.shape == (16, 32, 17)
    assert float(form[0, 0, 0]) == 1.0
    assert float(form.abs().max()) <= 1.0
    for index, expected in OLD_FORM.items():
        assert float(form[index]) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_the_volume_matches_the_one_wo40_rendered() -> None:
    assert float(_bead_volume().sum()) == pytest.approx(172944.92068767548, rel=1e-5)


def test_detection_finds_what_wo40_finds_brightest_first() -> None:
    volume = _bead_volume()[0]
    assert Beads(DIAMETER).detect(volume).tolist() == OLD_DETECTED
    assert Beads(DIAMETER, max_beads=2).detect(volume).tolist() == OLD_DETECTED[:2]


def test_detection_relaxes_with_a_warning_then_refuses() -> None:
    volume = torch.zeros(8, 16, 16, dtype=torch.float64)
    volume[4, 8, 8] = 1.0
    volume += 0.1 * torch.randn(volume.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    mad = float((volume - volume.median()).abs().median() * 1.4826)
    peak = float(volume[4, 8, 8] - volume.median())
    with pytest.warns(CalibrationWarning, match="snr_threshold"):
        found = Beads(DIAMETER, snr_threshold=1.2 * peak / mad).detect(volume)
    assert [8, 8] in found[:, 1:].tolist()
    with pytest.raises(ParameterError, match="snr_threshold") as caught:
        Beads(DIAMETER, snr_threshold=3 * peak / mad).detect(volume)
    assert caught.value.path == "snr_threshold"


@pytest.mark.parametrize("sigma", [1.0, "estimate"])
def test_the_objective_at_wo40s_start_is_wo40s_first(sigma) -> None:
    """Rendering, system and objective at wo-4.0's own start (detected peaks, amplitudes at the peak height, the
    median, sigma 1 or the MAD's), measured within 3e-7 of its ``trace[0]``; the fit's first objective is the same
    function at its own start, the amplitudes divided by the peak of a unit bead's image."""
    model = ImagingModel(_system(Free(0.0)), Beads(DIAMETER), Gaussian(sigma)).double()
    volume, peaks = _bead_volume()[0], torch.tensor(OLD_DETECTED, dtype=torch.float64)
    deviation = volume.new_tensor(1.0) if sigma == 1.0 else estimate_sigma(volume, method="mad")
    median = volume.median()
    heights = volume[peaks.long().unbind(1)] - median

    @torch.no_grad()
    def objective(amplitudes: torch.Tensor) -> float:
        predicted = model.system(Beads(DIAMETER).render(peaks, amplitudes, SAMPLING)[None], SAMPLING)[0] + median
        variance = deviation.square()
        misfit = (predicted - volume).square().sum() / (2 * variance)
        return float(misfit + 0.5 * volume.numel() * torch.log(2 * math.pi * variance))

    assert objective(heights) == pytest.approx(OLD_FIT[sigma]["trace"][0], rel=1e-6)
    unit = Beads(DIAMETER).render(peaks.new_zeros(1, 3), peaks.new_ones(1), SAMPLING)
    expected = objective(heights / model.system(unit[None], SAMPLING).max())
    assert float(calibrate_beads(_bead_volume(), model, SAMPLING, iterations=1).trace[0]) == pytest.approx(expected)


@pytest.mark.parametrize("sigma", [1.0, "estimate"])
def test_the_joint_fit_reaches_wo40s_optimum_in_the_default_steps(sigma) -> None:
    """wo-4.0's 150 steps ended near its objective's optimum, which this fit (the same objective, above) reaches in
    its default 200 steps from its better start. Measured, against wo-4.0: aberration within 1.3e-4 waves (1000 steps
    move it by 3e-6), positions within 0.002 voxels, background within 1.4e-3, amplitudes within 0.25%, sigma
    within 1.2e-4, and a final objective 0.019 (0.007 with sigma estimated) below wo-4.0's."""
    model = ImagingModel(_system(Free(0.0)), Beads(DIAMETER), Gaussian(sigma)).double()
    result = calibrate_beads(_bead_volume(), model, SAMPLING)
    expected = OLD_FIT[sigma]
    astigmatism = float(result.parameters["system.detection.pupil_aberrations.astigmatism_0"])
    assert astigmatism == pytest.approx(expected["astigmatism"], abs=5e-4)
    assert float(model.get_parameter("system.detection.pupil_aberrations.astigmatism_0").detach()) == astigmatism
    assert float(result.background) == pytest.approx(expected["background"], abs=2e-3)
    assert float(result.noise_sigma) == pytest.approx(expected["sigma"], rel=2e-4)
    assert torch.allclose(result.positions, torch.tensor(expected["positions"], dtype=torch.float64), atol=0.01)
    assert torch.allclose(result.amplitudes, torch.tensor(expected["amplitudes"], dtype=torch.float64), rtol=3e-3)
    assert result.trace.shape == (200,)
    assert float(result.trace[-1]) <= expected["trace"][1]
    assert float(result.trace[-1]) == pytest.approx(float(result.trace.min()), abs=1e-3)  # settled


def test_a_short_fit_finds_the_aperture_from_beads_of_unknown_brightness() -> None:
    """The reported failure: 40 steps (its config's), aperture free from 1.15 in [1, 1.3], sigma estimated. Started
    at the peak heights, about 1/150 of the amplitudes, the beads grew for the whole search and the aperture drifted
    to 1.059 (sigma 2.063); now 1.2164 (sigma 2.018). 1000 steps from either start end at 1.2161 (sigma 2.015)."""
    truth = Detection(wavelength=0.532, numerical_aperture=1.2, edge="sigmoid")
    positions = torch.tensor(POSITIONS, dtype=torch.float64)
    obj = Beads(DIAMETER).render(positions, torch.full((3,), 3000.0, dtype=torch.float64), SAMPLING)
    noise = 2 * torch.randn(SAMPLING.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    data = WidefieldFluorescence3D(truth, index_of_refraction_media=1.3).double()(obj[None], SAMPLING) + 100 + noise
    detection = Detection(wavelength=0.532, numerical_aperture=Free(1.15, bounds=(1.0, 1.3)))
    system = WidefieldFluorescence3D(detection, index_of_refraction_media=1.3)
    model = ImagingModel(system, Beads(DIAMETER), Gaussian("estimate")).double()
    result = calibrate_beads(data, model, SAMPLING, iterations=40)
    aperture = float(result.parameters["system.detection.numerical_aperture"])
    assert aperture == pytest.approx(1.2161, rel=0.02)
    assert 1.0 + 0.05 < aperture < 1.3 - 0.05
    assert float(result.noise_sigma) == pytest.approx(2.015, rel=0.01)


def test_beads_refuse_to_reconstruct_and_need_a_free_system_quantity() -> None:
    with pytest.raises(IncompatibleModelError, match="Tikhonov"):
        Beads(DIAMETER).penalty(torch.zeros(1, 2, 2, 2), scale=torch.tensor(1.0))
    fixed = ImagingModel(_system(0.0), Beads(DIAMETER), Gaussian(1.0)).double()
    with pytest.raises(ParameterError, match="free quantity"):
        calibrate_beads(_bead_volume(), fixed, SAMPLING, iterations=1)
    tikhonov = ImagingModel(_system(Free(0.0)), Tikhonov(1e-3), Gaussian(1.0)).double()
    with pytest.raises(IncompatibleModelError, match="Beads prior"):
        calibrate_beads(_bead_volume(), tikhonov, SAMPLING, iterations=1)


def test_beads_refuse_a_thin_system_before_fitting() -> None:
    # wo-4.0 fitted 3D stacks only; the thin system's object is one plane, so a bead volume cannot be rendered for it.
    detection = Detection(wavelength=0.532, numerical_aperture=Free(1.2, bounds=(1.0, 1.25)))
    thin = ImagingModel(ThinFluorescence2D(detection, index_of_refraction_media=1.3), Beads(DIAMETER), Gaussian(1.0))
    with pytest.raises(IncompatibleModelError, match="ThinFluorescence2D takes an object of shape \\(1, 1, 32, 32\\)"):
        calibrate_beads(_bead_volume().float(), thin, SAMPLING, iterations=1)
