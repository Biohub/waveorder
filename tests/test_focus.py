"""The 3.x ``waveorder.focus.focus_from_transverse_band`` front door matches the 3.x engine."""

import math

import pytest
import torch

from waveorder.errors import ParameterError
from waveorder.focus import focus_from_transverse_band

DEPRECATED = "use waveorder.nn.focus_from_transverse_band"


def defocus_stack(slices: int, focus: float, n: int = 64) -> torch.Tensor:
    """A fixed pattern blurred by a Gaussian whose width grows with the distance to ``focus``."""
    y = torch.arange(n, dtype=torch.float32)
    pattern = torch.cos(2 * torch.pi * y[:, None] * 5 / n) * torch.cos(2 * torch.pi * y[None, :] * 7 / n) + torch.cos(
        2 * torch.pi * (y[:, None] + y[None, :]) * 3 / n
    )
    fy = torch.fft.fftfreq(n)
    fr2 = fy[:, None] ** 2 + fy[None, :] ** 2
    z = torch.arange(slices, dtype=torch.float32)
    sigma = 0.5 + 0.8 * (z - focus).abs()
    otf = torch.exp(-2 * torch.pi**2 * sigma[:, None, None] ** 2 * fr2 * 4)
    return torch.fft.ifftn(torch.fft.fftn(pattern) * otf, dim=(-2, -1)).real + 2.0


def index_of(stack: object, **options: object) -> int | float | None:
    """The 3.x call with NA_det=0.55, lambda_ill=0.532, pixel_size=0.325, expecting the index alone."""
    with pytest.warns(DeprecationWarning, match=DEPRECATED):
        result = focus_from_transverse_band(stack, 0.55, 0.532, 0.325, **options)  # ty: ignore[invalid-argument-type]
    assert not isinstance(result, tuple)
    return result


def with_statistics(stack: object, **options: object) -> tuple[int | float | None, dict[str, int | float | None]]:
    """The 3.x call as :func:`index_of`, with ``return_statistics=True``."""
    with pytest.warns(DeprecationWarning, match=DEPRECATED):
        result = focus_from_transverse_band(stack, 0.55, 0.532, 0.325, return_statistics=True, **options)  # ty: ignore[invalid-argument-type]
    assert isinstance(result, tuple)
    return result


# Values from the 3.x engine, commit 85282ce (waveorder.focus.focus_from_transverse_band), on defocus_stack(slices,
# focus) with NA_det=0.55, lambda_ill=0.532, pixel_size=0.325 (script /tmp/FocusCompat/old.py). "index": defaults,
# midband_fractions=(0.05, 0.4) and return_statistics=True; "subpixel": polynomial_fit_order=3 with
# enable_subpixel_precision=True; "fwhm": the statistics' peak_FWHM of both.
OLD = {
    (11, 4.3): {"index": 4, "subpixel": 3.5900512782222975, "fwhm": 1.432428378263927},
    (9, 6.0): {"index": 6, "subpixel": 5.996725306187311, "fwhm": 1.1440242838864112},
}
SUBPIXEL = {"polynomial_fit_order": 3, "enable_subpixel_precision": True}


@pytest.mark.parametrize(("slices", "focus"), list(OLD))
def test_matches_3x(slices: int, focus: float) -> None:
    old = OLD[slices, focus]
    stack = defocus_stack(slices, focus)
    index = index_of(stack)
    assert isinstance(index, int)
    assert index == old["index"]
    subpixel = index_of(stack, **SUBPIXEL)
    assert isinstance(subpixel, float)
    assert math.isclose(subpixel, old["subpixel"], rel_tol=1e-5)
    assert index_of(stack, midband_fractions=(0.05, 0.4)) == old["index"]

    index, stats = with_statistics(stack)
    assert index == old["index"]
    assert stats["peak_index"] == old["index"]
    assert stats["peak_FWHM"] == pytest.approx(old["fwhm"], rel=1e-5)
    index, stats = with_statistics(stack, **SUBPIXEL)
    assert index == pytest.approx(old["subpixel"], rel=1e-5)
    assert stats["peak_index"] == int(old["subpixel"])
    assert stats["peak_FWHM"] == pytest.approx(old["fwhm"], rel=1e-5)
    index, stats = with_statistics(stack, threshold_FWHM=100)
    assert index is None
    assert stats["peak_index"] == old["index"]


def test_accepts_numpy() -> None:
    numpy = pytest.importorskip("numpy")
    stack = defocus_stack(11, 4.3).numpy().astype(numpy.float64)
    assert index_of(stack) == OLD[11, 4.3]["index"]


def test_refuses_plot_path() -> None:
    with pytest.raises(ParameterError, match="plot_path"):
        index_of(defocus_stack(5, 2.0), plot_path="focus.png")


def test_docstring_points_at_nn() -> None:
    assert "waveorder.nn.focus_from_transverse_band" in (focus_from_transverse_band.__doc__ or "")
