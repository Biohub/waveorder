"""Preprocessing: focus finding, z padding and the thin-model background filter, against unmodified 3.x.

The 3.x references live in :mod:`tests.nn.legacy_preprocess` (focus finding, ``inten_normalization``) and
:mod:`tests.nn.legacy_phase` (``pad_zyx_along_z``, the padded thick transfer and inverse).
"""

import math
import warnings
from typing import Any, Literal

import pytest
import torch
from torch import Tensor

from tests.nn import legacy_phase, legacy_preprocess
from waveorder.errors import ParameterError, ShapeError, WaveorderWarning
from waveorder.nn import (
    Detection,
    Gaussian,
    ImagingModel,
    Sampling,
    SolveResult,
    Tikhonov,
    Wiener,
)
from waveorder.nn.optics import Illumination, IlluminationTilt
from waveorder.nn.preprocess import Focus, TransverseBand, ZPadding, brightfield_contrast, focus_from_transverse_band
from waveorder.nn.systems.phase import ThickPhase3D, ThinPhase2D

WAVELENGTH = 0.532
INDEX = 1.33


def _relative(actual: Tensor, expected: Tensor) -> float:
    return float((actual.double() - expected.double()).norm() / expected.double().norm())


def _stack(shape: tuple[int, ...], *, seed: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Random positive planes ``(..., Z, Y, X)`` whose contrast varies smoothly along z, so the band power peaks."""
    generator = torch.Generator().manual_seed(seed)
    stack = torch.rand(shape, generator=generator, dtype=torch.float64)
    slope = 0.6 * torch.rand(1, generator=generator, dtype=torch.float64)
    profile = 1 + slope * torch.linspace(-1, 1, shape[-3], dtype=torch.float64).abs()
    return (stack * profile[:, None, None]).to(dtype)


# --- focus finding --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(("shape", "fractions"), [((5, 32, 32), (0.125, 0.25)), ((4, 33, 18), (0.1, 0.3))])
def test_band_power_matches_3x_midband_power(shape, fractions, dtype) -> None:
    stack = _stack(shape, seed=3, dtype=dtype)
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1, fractions=fractions)
    expected = legacy_preprocess.compute_midband_power(stack, 1.2, WAVELENGTH, 0.1, fractions)
    power = band.power(stack)
    assert power.dtype == dtype
    assert power.shape == (shape[0],)
    assert _relative(power, expected) <= (1e-13 if dtype == torch.float64 else 1e-6)
    # One plane gives a scalar, as 3.x did for 2D input.
    plane = band.power(stack[1])
    assert plane.shape == ()
    assert _relative(plane, legacy_preprocess.compute_midband_power(stack[1], 1.2, WAVELENGTH, 0.1, fractions)) <= 1e-6


def test_band_power_casts_integer_stacks_to_float32_as_3x() -> None:
    stack = (100 * _stack((3, 16, 16), seed=4)).to(torch.int32)
    band = TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=0.1)
    power = band.power(stack)
    assert power.dtype == torch.float32
    assert _relative(power, legacy_preprocess.compute_midband_power(stack.float(), 1.0, WAVELENGTH, 0.1)) <= 1e-6


def test_band_power_with_anisotropic_pixels_sums_the_full_spectrum_ring() -> None:
    """Oracle: the ring on the full spectrum with separate y and x frequency steps; any leading dimensions batch."""
    images = _stack((2, 3, 12, 15), seed=5)
    band = TransverseBand(numerical_aperture=0.9, wavelength=WAVELENGTH, pixel_size=(0.12, 0.08), fractions=(0.1, 0.6))
    fy = torch.fft.fftfreq(12, 0.12, dtype=torch.float64)
    fx = torch.fft.fftfreq(15, 0.08, dtype=torch.float64)
    radius = torch.sqrt(fx[None] ** 2 + fy[:, None] ** 2)
    cutoff = 2 * 0.9 / WAVELENGTH
    ring = (radius > 0.1 * cutoff) & (radius < 0.6 * cutoff)
    expected = (torch.fft.fft2(images).abs() * ring).sum(dim=(-2, -1))
    assert _relative(band.power(images), expected) <= 1e-13


def test_band_power_is_differentiable() -> None:
    images = _stack((2, 6, 7), seed=6).requires_grad_()
    band = TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=0.1, fractions=(0.05, 0.9))
    assert torch.autograd.gradcheck(band.power, (images,))


FOCUS_OPTIONS = [
    pytest.param({}, id="max"),
    pytest.param({"mode": "min"}, id="min"),
    pytest.param({"polynomial_order": 2}, id="quadratic-fit"),
    pytest.param({"polynomial_order": 3, "subpixel": True}, id="cubic-subpixel"),
    pytest.param({"polynomial_order": 4, "subpixel": True, "mode": "min"}, id="quartic-subpixel-min"),
    pytest.param({"threshold_fwhm": 3.0}, id="threshold"),
]


def _legacy_options(options: dict) -> dict:
    names = {"polynomial_order": "polynomial_fit_order", "subpixel": "enable_subpixel_precision"}
    names["threshold_fwhm"] = "threshold_FWHM"
    return {names.get(key, key): value for key, value in options.items()}


@pytest.mark.filterwarnings("ignore:focus_from_transverse_band polynomial fit:waveorder.errors.WaveorderWarning")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("options", FOCUS_OPTIONS)
def test_focus_matches_3x(options, dtype) -> None:
    """Over random stacks of 3 to 14 slices (underdetermined fits included), every decision matches 3.x.

    3.x raised for a sub-slice extreme within half a slice of the end (see the next test); those stacks are skipped.
    In float32 (the 3.x dtype) the half-spectrum power differs from the masked full-spectrum sum by rounding, which
    moves the fitted extreme and the peak width by about 1e-6 relative.
    """
    band = TransverseBand(numerical_aperture=1.1, wavelength=WAVELENGTH, pixel_size=0.12)
    tolerance = 1e-9 if dtype == torch.float64 else 1e-4
    compared = 0
    for seed in range(40):
        nz = 3 + seed % 12
        stack = _stack((nz, 20 + seed % 7, 24), seed=seed, dtype=dtype)
        try:
            index, stats = legacy_preprocess.focus_from_transverse_band(
                stack, 1.1, WAVELENGTH, 0.12, **_legacy_options(options)
            )
        except IndexError:
            continue
        focus = focus_from_transverse_band(stack, band, **options)
        assert focus.peak_index == stats["peak_index"]
        assert focus.peak_fwhm == pytest.approx(stats["peak_FWHM"], rel=tolerance, abs=tolerance)
        assert (focus.index is None) == (index is None)
        if index is not None:
            assert focus.index == pytest.approx(index, rel=tolerance, abs=tolerance)
            assert isinstance(focus.index, float) == isinstance(index, float)
        compared += 1
    assert compared >= 30


def _scaled(profile: list[float]) -> Tensor:
    """A stack whose band power is exactly proportional to ``profile``: one plane scaled per slice."""
    plane = _stack((1, 24, 24), seed=11)[0]
    return torch.stack([value * plane for value in profile])


def test_focus_finds_the_peak_slice_and_its_width() -> None:
    profile = [math.exp(-(((z - 6) / 2.5) ** 2)) + 0.1 for z in range(13)]
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1)
    focus = focus_from_transverse_band(_scaled(profile), band)
    assert isinstance(focus, Focus)
    assert focus.index == 6
    assert focus.peak_index == 6
    # Half prominence of a Gaussian of 1/e half-width 2.5: FWHM = 2 * 2.5 * sqrt(ln 2), to sampling error.
    assert focus.peak_fwhm == pytest.approx(2 * 2.5 * math.sqrt(math.log(2)), rel=0.05)
    expected = legacy_preprocess.peak_widths([float(v) for v in focus.power.double()], 6)[0]
    assert focus.peak_fwhm == pytest.approx(expected, rel=1e-12)
    wider = expected + 0.1
    assert focus_from_transverse_band(_scaled(profile), band, threshold_fwhm=wider).index is None
    assert focus_from_transverse_band(_scaled(profile), band, mode="min").index in (0, 12)


def test_subpixel_extreme_near_the_last_slice_is_measured_there() -> None:
    """A continuous extreme in [Z - 0.5, Z) rounds to Z: 3.x raised from peak_widths, 4.x measures the last slice."""
    profile = [30 - (z - 4.7) ** 2 for z in range(5)]
    stack = _scaled(profile)
    with pytest.raises(IndexError):
        legacy_preprocess.focus_from_transverse_band(
            stack, 1.2, WAVELENGTH, 0.1, polynomial_fit_order=2, enable_subpixel_precision=True
        )
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1)
    focus = focus_from_transverse_band(stack, band, polynomial_order=2, subpixel=True)
    assert focus.index == pytest.approx(4.7, abs=1e-9)
    assert focus.peak_index == 4
    assert focus.peak_fwhm == legacy_preprocess.peak_widths([float(v) for v in focus.power.double()], 4)[0]


def test_focus_keeps_the_power_graph() -> None:
    stack = _stack((5, 16, 16), seed=12).requires_grad_()
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1)
    focus = focus_from_transverse_band(stack, band)
    focus.power.sum().backward()
    assert stack.grad is not None
    assert torch.isfinite(stack.grad).all()


def test_a_single_slice_is_in_focus_with_a_warning() -> None:
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1)
    with pytest.warns(WaveorderWarning, match="single slice"):
        focus = focus_from_transverse_band(_stack((1, 10, 10), seed=1), band)
    assert (focus.index, focus.peak_index, focus.peak_fwhm) == (0, None, None)
    assert focus.power.shape == (1,)


def test_a_rank_deficient_fit_warns() -> None:
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1)
    with pytest.warns(WaveorderWarning, match="rank deficient"):
        focus_from_transverse_band(_stack((3, 16, 16), seed=2), band, polynomial_order=4)


def test_band_checks_its_fields_at_their_path() -> None:
    with pytest.raises(ParameterError) as error:
        TransverseBand(numerical_aperture=0.0, wavelength=WAVELENGTH, pixel_size=0.1)
    assert error.value.path == "numerical_aperture"
    with pytest.raises(ParameterError) as error:
        TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=(0.1, -0.1))
    assert error.value.path == "pixel_size.x"
    for fractions in [(0.3, 0.2), (0.1, 1.5), (-0.1, 0.2)]:
        with pytest.raises(ParameterError) as error:
            TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=0.1, fractions=fractions)
        assert error.value.path == "fractions"
    with pytest.warns(WaveorderWarning, match="same units"):
        TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=6.5)
    pair: Any = [0.1, 0.2]  # a list, as YAML gives a pair
    band = TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=0.1, fractions=pair)
    assert band.pixel_size == (0.1, 0.1)
    assert band.fractions == (0.1, 0.2)


def test_focus_checks_its_arguments_at_their_path() -> None:
    band = TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=0.1)
    stack = _stack((3, 8, 8), seed=0)
    with pytest.raises(ShapeError, match=r"\(Z, Y, X\)"):
        focus_from_transverse_band(stack[None], band)
    cases = [({"mode": "maxx"}, "mode"), ({"polynomial_order": -1}, "polynomial_order")]
    for options, path in cases:
        with pytest.raises(ParameterError) as error:
            focus_from_transverse_band(stack, band, **options)  # ty: ignore[invalid-argument-type]
        assert error.value.path == path
    with pytest.raises(ParameterError) as error:
        band.power(stack.to(torch.complex128))
    assert error.value.path == "images"


# --- z padding ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("slices", [0, 1, 3, 5, 6, 9], ids=["none", "1", "3", "5", "6-zeros", "9-zeros"])
def test_pad_is_3x_pad_zyx_along_z(slices) -> None:
    # 3.x mirrors the edge slices while the pad is shorter than the stack (6 slices), and pads zeros from 6 on.
    data = _stack((2, 6, 5, 4), seed=7)
    padding = ZPadding(slices)
    padded = padding.pad(data)
    assert padded.shape == (2, 6 + 2 * slices, 5, 4)
    for channel in range(2):
        assert torch.equal(padded[channel], legacy_phase.pad_along_z(data[channel], slices))
    assert torch.equal(padding.crop(SolveResult(padded, torch.tensor(0.0), 0, "direct")).estimate, data)
    sampling = Sampling(shape=(6, 5, 4), pixel_size=(0.4, 0.1, 0.2))
    assert padding.sampling(sampling) == Sampling(shape=(6 + 2 * slices, 5, 4), pixel_size=(0.4, 0.1, 0.2))


def test_pad_is_differentiable() -> None:
    data = _stack((1, 4, 3, 3), seed=8).requires_grad_()
    assert torch.autograd.gradcheck(ZPadding(2).pad, (data,))


def test_pad_rejects_data_without_z_and_a_negative_pad() -> None:
    with pytest.raises(ShapeError, match=r"\(\.\.\., Z, Y, X\)"):
        ZPadding(1).pad(torch.ones(4, 3))
    with pytest.raises(ParameterError) as error:
        ZPadding(-1)
    assert error.value.path == "slices"


def test_crop_keeps_the_central_slices_and_the_padded_solve() -> None:
    estimate = torch.arange(2 * 9 * 2 * 2, dtype=torch.float64).reshape(2, 9, 2, 2)
    result = SolveResult(estimate, torch.tensor(1.5), 7, "budget")
    cropped = ZPadding(2).crop(result)
    assert torch.equal(cropped.estimate, estimate[:, 2:7])
    assert (float(cropped.objective), cropped.iterations, cropped.status) == (1.5, 7, "budget")
    assert ZPadding().crop(result) is result
    with pytest.raises(ShapeError, match="padded grid"):
        ZPadding(5).crop(result)


THICK_PIXELS = (0.4, 0.15, 0.15)
"""A z step 3.x did not oversample (its padded transfer is then right; see the oversampled test)."""


def _transfer_3x(
    shape: tuple[int, int, int], pixel_size: tuple[float, float, float], *, zenith: float, z_padding: int = 0
) -> tuple[Tensor, Tensor]:
    """Unmodified 3.x thick transfers for the optics of :func:`_thick_phase`."""
    return legacy_phase.thick_transfer(
        shape,
        pixel_size,
        wavelength=WAVELENGTH,
        index=INDEX,
        na_illumination=0.5,
        na_detection=0.9,
        zenith=zenith,
        azimuth=1.0,
        z_padding=z_padding,
    )


def _thick_phase(*, zenith: float = 0.15, absorption: float | Literal["independent"] = 0.2) -> ThickPhase3D:
    illumination = Illumination(
        wavelength=WAVELENGTH, numerical_aperture=0.5, tilt=IlluminationTilt(zenith=zenith, azimuth=1.0)
    )
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=0.9, edge="sigmoid")
    return ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX, absorption=absorption).double()


def _thick_intensity(shape: tuple[int, int, int]) -> Tensor:
    """Raw 3.x intensities of a phase sphere wrapped across the top and bottom of the stack."""
    real_tf, imag_tf = _transfer_3x(shape, THICK_PIXELS, zenith=0.15)
    phantom = 0.05 * torch.roll(legacy_phase.sphere(shape, THICK_PIXELS, radius=0.6), 3, dims=0)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    contrast = torch.fft.ifftn(torch.fft.fftn(phantom) * (real_tf + 0.2 * imag_tf)).real
    return 1e3 * (1 + contrast) + 2.0 * noise


def _padded_4x(intensity: Tensor, sampling: Sampling, *, slices: int, weight: float) -> Tensor:
    padding = ZPadding(slices)
    model = ImagingModel(_thick_phase(), Tikhonov(weight), Gaussian(1.0)).double()
    result = Wiener()(brightfield_contrast(padding.pad(intensity[None])), model, padding.sampling(sampling))
    return padding.crop(result).estimate[0]


@pytest.mark.parametrize("slices", [1, 3])
def test_padded_thick_tikhonov_reconstruction_matches_unmodified_3x(slices) -> None:
    """Pad the raw intensities, normalize, solve on the padded grid and crop: the 3.x padded Tikhonov inverse."""
    sampling = Sampling(shape=(8, 16, 15), pixel_size=THICK_PIXELS)
    intensity = _thick_intensity(sampling.shape)
    real_tf, imag_tf = _transfer_3x(sampling.shape, THICK_PIXELS, zenith=0.15, z_padding=slices)
    expected = legacy_phase.thick_inverse(
        intensity, real_tf, imag_tf, z_padding=slices, absorption_ratio=0.2, regularization_strength=1e-6
    )
    estimate = _padded_4x(intensity, sampling, slices=slices, weight=1e-6)
    assert estimate.shape == sampling.shape
    assert _relative(estimate, expected) <= 1e-10
    assert _relative(_padded_4x(intensity, sampling, slices=0, weight=1e-6), expected) > 0.1


def _stored_half(full: Tensor, nx: int) -> Tensor:
    """A 3.x full transfer as its operator ``real(ifftn(H F))`` sees it: the half of its Hermitian part, real pairs."""
    dims = (-3, -2, -1)
    hermitian = 0.5 * (full + torch.roll(torch.flip(full, dims), (1, 1, 1), dims).conj())
    return torch.view_as_real(hermitian[..., : nx // 2 + 1].contiguous())


def test_padded_transfer_differs_from_3x_where_3x_oversampled_z() -> None:
    """With z oversampled (by 2 at a 1 um step), 3.x computed the padded transfer on ``2 Z + 2p`` fine slices (the pad
    not oversampled) and cropped ``Z + 2p`` frequencies, whose spacing is then not the padded grid's. 4.x binds the
    padded sampling, so its transfer is the 3.x transfer of a ``Z + 2p`` stack. Measured at 8 x 16 x 15 voxels of
    (1.0, 0.15, 0.15) um, p = 3, as each operator applies it: 3.x's padded transfer differs by 50.3% (phase) and 31.7%
    (absorption)."""
    pixel_size = (1.0, 0.15, 0.15)
    system = _thick_phase(zenith=0.0, absorption="independent")
    transfer = system.transfer(ZPadding(3).sampling(Sampling(shape=(8, 16, 15), pixel_size=pixel_size)))
    real_tf, imag_tf = _transfer_3x((14, 16, 15), pixel_size, zenith=0.0)
    assert _relative(transfer[0, 0], _stored_half(real_tf, 15)) <= 1e-10
    assert _relative(transfer[0, 1], _stored_half(imag_tf, 15)) <= 1e-10
    real_3x, imag_3x = _transfer_3x((8, 16, 15), pixel_size, zenith=0.0, z_padding=3)
    assert _relative(transfer[0, 0], _stored_half(real_3x, 15)) == pytest.approx(0.503, abs=5e-3)
    assert _relative(transfer[0, 1], _stored_half(imag_3x, 15)) == pytest.approx(0.317, abs=5e-3)


# --- thin-model background filter -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3, 32, 64), id="even"),
        pytest.param((2, 17, 30), id="odd-rows"),
        pytest.param((2, 9, 40), id="window-wider-than-the-rows"),
        pytest.param((2, 3, 11, 12), id="batched"),
    ],
)
def test_bg_filter_is_3x_inten_normalization(shape) -> None:
    generator = torch.Generator().manual_seed(9)
    intensity = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
    expected = legacy_preprocess.inten_normalization(intensity, bg_filter=True)
    assert _relative(brightfield_contrast(intensity, per_plane=True, bg_filter=True), expected) <= 1e-13


def test_bg_filter_default_is_the_unfiltered_contrast_bit_for_bit() -> None:
    intensity = 0.5 + torch.rand(2, 5, 8, 8, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    plain = intensity / torch.clamp(intensity.mean(dim=(-2, -1), keepdim=True), min=1e-12) - 1
    assert torch.equal(brightfield_contrast(intensity, per_plane=True), plain)
    assert torch.equal(brightfield_contrast(intensity, per_plane=True, bg_filter=False), plain)
    unfiltered = legacy_preprocess.inten_normalization(intensity, bg_filter=False)
    assert _relative(brightfield_contrast(intensity, per_plane=True), unfiltered) <= 1e-15


def test_thin_tikhonov_reconstruction_with_bg_filter_matches_unmodified_3x() -> None:
    """3.x ``bg_filter=True`` in the thin inverse: the filtered per-plane contrast through the same Tikhonov inverse."""
    sampling = Sampling(shape=(7, 32, 32), pixel_size=(0.6, 0.1, 0.1))
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=0.4)
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=0.9, edge="sigmoid")
    system = ThinPhase2D(illumination, detection, index_of_refraction_media=INDEX).double()
    ball = legacy_phase.sphere((3, 32, 32), (1.0, 0.1, 0.1), radius=0.8)[1]
    obj = torch.stack((0.3 * ball, 0.05 * torch.roll(ball, 3, -1)))[:, None]
    shading = 1 + 0.3 * torch.linspace(0, 1, 32, dtype=torch.float64)[None, :] * torch.ones(32, 1, dtype=torch.float64)
    intensity = 10 * (1 + system.forward(obj, sampling)) * shading
    z = legacy_phase.thin_z_positions(7, 0.6)
    absorption_tf, phase_tf = legacy_phase.thin_transfer(
        (32, 32), (0.1, 0.1), z, wavelength=WAVELENGTH, index=INDEX, na_illumination=0.4, na_detection=0.9
    )
    weight = 1e-3
    filtered = intensity[0] / torch.stack([legacy_preprocess.uniform_filter(plane, 16) for plane in intensity[0]])
    absorption, phase = legacy_phase.thin_inverse(
        filtered, legacy_phase.thin_singular_system(absorption_tf, phase_tf), regularization_strength=weight
    )
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    data = brightfield_contrast(intensity, per_plane=True, bg_filter=True)
    estimate = Wiener()(data, model, sampling).estimate
    assert _relative(estimate[0, 0], phase) <= 1e-9
    assert _relative(estimate[1, 0], absorption) <= 1e-9


def test_bg_filter_is_differentiable() -> None:
    intensity = 0.5 + torch.rand(1, 2, 5, 6, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    intensity.requires_grad_()
    assert torch.autograd.gradcheck(lambda x: brightfield_contrast(x, per_plane=True, bg_filter=True), (intensity,))


def test_bg_filter_refuses_wrong_arguments_at_its_path() -> None:
    intensity = torch.ones(1, 2, 4, 4)
    with pytest.raises(ParameterError, match="per_plane") as error:
        brightfield_contrast(intensity, bg_filter=True)
    assert error.value.path == "bg_filter"
    with pytest.raises(ShapeError, match="2 pixels wide"):
        brightfield_contrast(torch.ones(1, 2, 4, 1), per_plane=True, bg_filter=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        brightfield_contrast(intensity, per_plane=True, bg_filter=True)
