"""Preprocessing: focus finding, z padding and the thin-model background filter."""

import math
import warnings
from typing import Any

import pytest
import torch
from torch import Tensor

from waveorder.errors import ParameterError, ShapeError, WaveorderWarning
from waveorder.nn import (
    Sampling,
    SolveResult,
)
from waveorder.nn.preprocess import Focus, TransverseBand, ZPadding, brightfield_contrast, focus_from_transverse_band

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


def test_band_power_casts_integer_stacks_to_float32() -> None:
    stack = (100 * _stack((3, 16, 16), seed=4)).to(torch.int32)
    band = TransverseBand(numerical_aperture=1.0, wavelength=WAVELENGTH, pixel_size=0.1)
    power = band.power(stack)
    assert power.dtype == torch.float32


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
    wider = 1.1 * 2 * 2.5 * math.sqrt(math.log(2))  # past the measured width
    assert focus_from_transverse_band(_scaled(profile), band, threshold_fwhm=wider).index is None
    # The least power is on an end slice: no prominence, so its width is 0, which 3.x warned of through scipy.
    with pytest.warns(WaveorderWarning, match="prominence of 0, so its peak width is 0$"):
        minimum = focus_from_transverse_band(_scaled(profile), band, mode="min")
    assert (minimum.index, minimum.peak_fwhm) in ((0, 0.0), (12, 0.0))
    with pytest.warns(WaveorderWarning, match=r"and no slice is in focus \(threshold_fwhm = 1.0\)"):
        assert focus_from_transverse_band(_scaled(profile), band, mode="min", threshold_fwhm=1.0).index is None


def test_subpixel_extreme_near_the_last_slice_is_measured_there() -> None:
    """A continuous extreme in [Z - 0.5, Z) rounds to Z, and the last slice is measured (3.x raised there)."""
    stack = _scaled([30 - (z - 4.7) ** 2 for z in range(5)])
    band = TransverseBand(numerical_aperture=1.2, wavelength=WAVELENGTH, pixel_size=0.1)
    with pytest.warns(WaveorderWarning, match="band power at slice 4 has a prominence of 0"):
        focus = focus_from_transverse_band(stack, band, polynomial_order=2, subpixel=True)
    assert focus.index == pytest.approx(4.7, abs=1e-9)
    assert focus.peak_index == 4


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


@pytest.mark.filterwarnings("ignore:focus_from_transverse_band. the band power:waveorder.errors.WaveorderWarning")
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
def test_pad_mirrors_the_edge_slices_while_shorter_than_the_stack_and_pads_zeros_after(slices) -> None:
    # As 3.x pad_zyx_along_z: the 6 slices of this stack are mirrored while the pad is shorter, zeros from 6 on.
    data = _stack((2, 6, 5, 4), seed=7)
    padding = ZPadding(slices)
    padded = padding.pad(data)
    assert padded.shape == (2, 6 + 2 * slices, 5, 4)
    if slices < 6:
        edges = data[:, :slices].flip(1), data[:, 6 - slices :].flip(1)
    else:
        edges = (torch.zeros(2, slices, 5, 4, dtype=torch.float64),) * 2
    assert torch.equal(padded, torch.cat((edges[0], data, edges[1]), dim=1))
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


# --- thin-model background filter -----------------------------------------------------------------------------------


def test_bg_filter_default_is_the_unfiltered_contrast_bit_for_bit() -> None:
    intensity = 0.5 + torch.rand(2, 5, 8, 8, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    plain = intensity / torch.clamp(intensity.mean(dim=(-2, -1), keepdim=True), min=1e-12) - 1
    assert torch.equal(brightfield_contrast(intensity, per_plane=True), plain)
    assert torch.equal(brightfield_contrast(intensity, per_plane=True, bg_filter=False), plain)


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
