"""Preprocessing around a solve: focus finding, z padding, brightfield contrast, cropping and apodization.

:func:`focus_from_transverse_band` finds the in-focus slice as the extreme of the power in a ring of transverse
frequencies (:class:`TransverseBand`); :class:`ZPadding` mirrors edge slices along z before a solve and crops after;
:func:`brightfield_contrast` turns raw intensities into the contrast the phase systems measure;
:func:`most_structured_crop` and :func:`apodize` are the steps around the 3.x data-driven weight sweep.
"""

import dataclasses
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from waveorder.errors import ParameterError, ShapeError, WaveorderWarning
from waveorder.nn.sampling import Sampling
from waveorder.nn.solvers import SolveResult

type FocusMode = Literal["min", "max"]
"""Whether the in-focus slice minimizes or maximizes the band power."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TransverseBand:
    """A ring of transverse spatial frequencies, as fractions of the detection cutoff ``2 NA / wavelength``.

    Parameters
    ----------
    numerical_aperture : float
        Detection numerical aperture.
    wavelength : float
        Illumination (or emission) wavelength, in the units of ``pixel_size``.
    pixel_size : float or tuple[float, float]
        Object-space pixel size (camera pixel over magnification) along y and x, or one number for square pixels.
    fractions : tuple[float, float]
        Inner and outer radius of the ring as fractions of the cutoff, ``0 <= inner < outer <= 1``. Both edges are
        excluded, as in 3.x.

    Raises
    ------
    ParameterError
        If a number is not positive and finite or ``fractions`` is not an increasing pair in ``[0, 1]``; ``path`` names
        the field.

    Warns
    -----
    WaveorderWarning
        If ``wavelength / pixel_size`` is outside ``(0.4, 10)`` along an axis, which usually means mixed units or a
        pixel size not divided by the magnification (the 3.x check).
    """

    numerical_aperture: float
    wavelength: float
    pixel_size: float | tuple[float, float]
    fractions: tuple[float, float] = (0.125, 0.25)

    def __post_init__(self) -> None:
        where = "TransverseBand"
        _expect_positive(self.numerical_aperture, where=where, name="numerical_aperture")
        _expect_positive(self.wavelength, where=where, name="wavelength")
        sizes = _pair(self.pixel_size, where=where, name="pixel_size")
        for axis, size in zip("yx", sizes, strict=True):
            _expect_positive(size, where=where, name=f"pixel_size.{axis}")
            if not 0.4 < self.wavelength / size < 10:
                warnings.warn(
                    f"{where} wavelength / pixel_size.{axis} = {self.wavelength / size:.3g}: are both in the same "
                    "units, and is the pixel size in (demagnified) object space?",
                    WaveorderWarning,
                    stacklevel=3,
                )
        object.__setattr__(self, "pixel_size", (float(sizes[0]), float(sizes[1])))
        inner, outer = _pair(self.fractions, where=where, name="fractions")
        if not 0 <= inner < outer <= 1:
            raise ParameterError(
                f"{where} fractions must satisfy 0 <= inner < outer <= 1, got ({inner}, {outer})", path="fractions"
            )
        object.__setattr__(self, "fractions", (float(inner), float(outer)))

    def power(self, images: Tensor) -> Tensor:
        """Return the summed spectral magnitude ``sum |FFT2(image)|`` over the ring, per image (3.x midband power).

        Each ``rfft2`` column is weighted by the number of full-spectrum columns it stands for.

        Parameters
        ----------
        images : Tensor
            Real images, ``(..., Y, X)``: one plane, a ``(Z, Y, X)`` stack, or any batch of planes. Integer images
            are cast to ``float32``.

        Returns
        -------
        Tensor
            The power of each image, shape ``(...)``, in the images' floating dtype, differentiable in ``images``.

        Raises
        ------
        ParameterError
            If ``images`` is complex.
        ShapeError
            If ``images`` has fewer than two dimensions.
        """
        where = "TransverseBand.power"
        if images.is_complex():
            raise ParameterError(f"{where} images must be real, got {images.dtype}", path="images")
        if images.ndim < 2:
            raise ShapeError(f"{where} images must be (..., Y, X), got shape {tuple(images.shape)}")
        if not images.is_floating_point():
            images = images.float()
        ny, nx = images.shape[-2:]
        dy, dx = _pair(self.pixel_size, where=where, name="pixel_size")
        dtype, device = images.dtype, images.device
        fy = torch.fft.fftfreq(ny, dy, dtype=dtype, device=device)
        fx = torch.fft.rfftfreq(nx, dx, dtype=dtype, device=device)
        radius = torch.sqrt(fx**2 + fy[:, None] ** 2)
        cutoff = 2 * self.numerical_aperture / self.wavelength
        inner, outer = self.fractions
        ring = (radius > cutoff * inner) & (radius < cutoff * outer)
        # Every half-spectrum column but zero (and, for even X, Nyquist) also stands for its mirror column.
        columns = torch.full((nx // 2 + 1,), 2.0, dtype=dtype, device=device)
        columns[0] = 1
        if nx % 2 == 0:
            columns[-1] = 1
        return (torch.fft.rfft2(images).abs() * (ring * columns)).sum(dim=(-2, -1))


@dataclass(frozen=True, slots=True)
class Focus:
    """The in-focus slice of a stack and the statistics it was chosen from (3.x ``return_statistics``).

    Attributes
    ----------
    index : int, float or None
        The in-focus slice: the extreme of the band power, or of its polynomial fit; a float with ``subpixel=True``.
        ``None`` when the peak is narrower than ``threshold_fwhm``.
    peak_index : int or None
        The extreme as an int (truncated toward zero, as 3.x reported it), kept when :attr:`index` is ``None``;
        ``None`` for a single slice.
    peak_fwhm : float or None
        Width in slices of the raw power's peak at half its prominence (scipy ``peak_widths``, as 3.x measured it);
        ``None`` for a single slice.
    power : Tensor
        Band power per slice, ``(Z,)``, differentiable in the stack.
    """

    index: int | float | None
    peak_index: int | None
    peak_fwhm: float | None
    power: Tensor


def focus_from_transverse_band(
    stack: Tensor,
    band: TransverseBand,
    *,
    mode: FocusMode = "max",
    polynomial_order: int | None = None,
    threshold_fwhm: float = 0.0,
    subpixel: bool = False,
) -> Focus:
    """Find the in-focus slice of a stack as the extreme of its band power (3.x ``focus_from_transverse_band``).

    The index is read back to the host; :attr:`Focus.power` keeps the autograd graph.

    Parameters
    ----------
    stack : Tensor
        Real data, ``(Z, Y, X)``.
    band : TransverseBand
        The ring of frequencies whose power is compared across slices.
    mode : {"max", "min"}
        Choose the slice of largest (the default) or smallest power.
    polynomial_order : int, optional
        Fit a polynomial of this degree to the power over the slice index (as ``numpy.polyfit``) and take the
        extreme of the fit instead; ``None`` uses the power itself.
    threshold_fwhm : float
        Smallest width, in slices, of the power's peak for the slice to count as in focus; a narrower peak gives
        ``Focus.index = None``.
    subpixel : bool
        With ``polynomial_order``, return the fit's best real critical point in ``[0, Z)`` as a float (the best
        fitted slice when there is none). Ignored without ``polynomial_order``, as in 3.x.

    Returns
    -------
    Focus
        The index, the peak statistics and the power curve.

    Raises
    ------
    ParameterError
        If ``mode`` is unknown or ``polynomial_order`` is negative; ``path`` names the argument.
    ShapeError
        If ``stack`` is not ``(Z, Y, X)``.

    Warns
    -----
    WaveorderWarning
        For a single slice, returned as in focus (``index = 0``) without statistics; or if the polynomial fit is rank
        deficient (``polynomial_order`` at least ``Z``).
    """
    _expect_focus_arguments(stack, mode=mode, polynomial_order=polynomial_order)
    power = band.power(stack)
    if stack.shape[0] == 1:
        warnings.warn(
            "focus_from_transverse_band got a single slice; returning slice 0", WaveorderWarning, stacklevel=2
        )
        return Focus(0, None, None, power)
    curve = power.detach().to(device="cpu", dtype=torch.float64)
    continuous = subpixel and polynomial_order is not None
    peak = _extreme(curve, mode=mode, polynomial_order=polynomial_order, subpixel=continuous)
    # The nearest slice of a continuous extreme in [Z - 0.5, Z) is the last one, not Z (3.x raised there).
    fwhm = _peak_width(curve.tolist(), min(round(peak), len(curve) - 1) if continuous else int(peak))
    return Focus(peak if fwhm >= threshold_fwhm else None, int(peak), fwhm, power)


@dataclass(frozen=True, slots=True)
class ZPadding:
    """Mirrored edge slices added at both ends of z before a solve, and cropped from the estimate after it.

    A shift-invariant inverse treats the stack as periodic along z, so an object that reaches an end slice wraps into
    the other. Padding mirrors ``p`` edge slices at each end (zeros instead when ``p >= Z``, as 3.x did); the solve runs
    on ``Z + 2p`` slices and the central ``Z`` of the estimate are kept. 3.x: ``z_padding``; config:
    ``solver.padding.z``::

        padding = ZPadding(4)
        result = padding.crop(Wiener()(padding.pad(data), model, padding.sampling(sampling)))

    For brightfield data, pad the raw intensities, then take :func:`~waveorder.nn.brightfield_contrast`, as 3.x did.
    Thin objects have nothing to pad, and the tile methods refuse padding.

    Parameters
    ----------
    slices : int
        Slices ``p`` added at each end, at least 0.

    Raises
    ------
    ParameterError
        If ``slices`` is negative.
    """

    slices: int = 0

    def __post_init__(self) -> None:
        if self.slices < 0:
            raise ParameterError(f"ZPadding slices must be at least 0, got {self.slices}", path="slices")

    def pad(self, data: Tensor) -> Tensor:
        """Return ``data``, ``(..., Z, Y, X)``, extended to ``(..., Z + 2p, Y, X)``: mirrored edge slices if ``p < Z``.

        With ``p >= Z`` there are not enough slices to mirror, and zeros are added instead (3.x ``pad_zyx_along_z``).

        Raises
        ------
        ShapeError
            If ``data`` has fewer than three dimensions.
        """
        where = "ZPadding.pad"
        if data.ndim < 3:
            raise ShapeError(f"{where} data must be (..., Z, Y, X), got shape {tuple(data.shape)}")
        p = self.slices
        if p == 0:
            return data
        if data.shape[-3] <= p:
            zeros = data.new_zeros((*data.shape[:-3], p, *data.shape[-2:]))
            return torch.cat((zeros, data, zeros), dim=-3)
        return torch.cat((data[..., :p, :, :].flip(-3), data, data[..., -p:, :, :].flip(-3)), dim=-3)

    def sampling(self, sampling: Sampling) -> Sampling:
        """Return ``sampling`` with ``2p`` more slices along z, the grid :meth:`pad` puts the data on."""
        nz, ny, nx = sampling.shape
        return dataclasses.replace(sampling, shape=(nz + 2 * self.slices, ny, nx))

    def crop(self, result: SolveResult) -> SolveResult:
        """Return ``result`` with the central ``Z`` slices of its estimate, ``(F, Z + 2p, Y, X)`` to ``(F, Z, Y, X)``.

        The objective, iteration count and status are those of the padded solve.

        Raises
        ------
        ShapeError
            If the estimate has at most ``2p`` slices, so it was not solved on the padded grid.
        """
        where = "ZPadding.crop"

        p = self.slices
        estimate = result.estimate
        if p == 0:
            return result
        if estimate.ndim < 3 or estimate.shape[-3] <= 2 * p:
            raise ShapeError(
                f"{where} needs an estimate of more than {2 * p} slices (solved on the padded grid), got shape "
                f"{tuple(estimate.shape)}"
            )
        return dataclasses.replace(result, estimate=estimate[..., p : estimate.shape[-3] - p, :, :])


def _expect_positive(value: float, *, where: str, name: str) -> None:
    if not (value > 0 and math.isfinite(value)):
        raise ParameterError(f"{where} {name} must be positive and finite, got {value}", path=name)


def _pair(value: float | Sequence[float], *, where: str, name: str) -> tuple[float, float]:
    """Return a number as ``(value, value)`` and a sequence of two as a tuple; raise for another length."""
    if not isinstance(value, Sequence):
        return (value, value)
    if len(value) != 2:
        raise ParameterError(f"{where} {name} must have 2 entries, got {len(value)}", path=name)
    first, second = value
    return (first, second)


def _expect_focus_arguments(stack: Tensor, *, mode: str, polynomial_order: int | None) -> None:
    where = "focus_from_transverse_band"
    if stack.ndim != 3:
        raise ShapeError(f"{where} stack must be (Z, Y, X), got shape {tuple(stack.shape)}")
    if mode not in ("min", "max"):
        raise ParameterError(f"{where} mode must be 'min' or 'max', got {mode!r}", path="mode")
    if polynomial_order is not None and polynomial_order < 0:
        raise ParameterError(
            f"{where} polynomial_order must be at least 0, got {polynomial_order}", path="polynomial_order"
        )


def _extreme(curve: Tensor, *, mode: FocusMode, polynomial_order: int | None, subpixel: bool) -> int | float:
    """Return the slice of extreme power, of extreme fitted power, or the fit's best critical point (3.x order)."""
    if polynomial_order is None:
        return _first_extreme(curve.tolist(), mode=mode)
    slices = torch.arange(curve.shape[0], dtype=torch.float64)
    coefficients = _polyfit(slices, curve, polynomial_order)
    fitted = _polyval(coefficients, slices).tolist()
    if not subpixel:
        return _first_extreme(fitted, mode=mode)
    critical = [root for root in _real_roots(_polyder(coefficients)) if 0 <= root < curve.shape[0]]
    if not critical:
        return float(_first_extreme(fitted, mode=mode))
    values = _polyval(coefficients, torch.tensor(critical, dtype=torch.float64)).tolist()
    return critical[_first_extreme(values, mode=mode)]


def _first_extreme(values: list[float], *, mode: FocusMode) -> int:
    """Index of the first largest (``"max"``) or smallest value, as ``numpy.argmax`` and ``numpy.argmin``."""
    best = 0
    for index, value in enumerate(values):
        if (value > values[best]) if mode == "max" else (value < values[best]):
            best = index
    return best


def _polyfit(x: Tensor, y: Tensor, order: int) -> Tensor:
    """``numpy.polyfit(x, y, order)``: column-scaled Vandermonde least squares, highest power first."""
    vandermonde = torch.linalg.vander(x, N=order + 1).flip(-1)
    scale = torch.sqrt((vandermonde * vandermonde).sum(dim=0))
    rcond = x.shape[0] * torch.finfo(x.dtype).eps
    fit = torch.linalg.lstsq(vandermonde / scale, y[:, None], rcond=rcond, driver="gelsd")
    if int(fit.rank) != order + 1:
        warnings.warn(
            f"focus_from_transverse_band polynomial fit of order {order} to {x.shape[0]} slices is rank deficient",
            WaveorderWarning,
            stacklevel=4,
        )
    return fit.solution[:, 0] / scale


def _polyval(coefficients: Tensor, x: Tensor) -> Tensor:
    """``numpy.polyval``: Horner's rule, highest power first."""
    value = torch.zeros_like(x)
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def _polyder(coefficients: Tensor) -> Tensor:
    """``numpy.polyder``: the derivative's coefficients, highest power first."""
    degree = coefficients.shape[0] - 1
    return coefficients[:-1] * torch.arange(degree, 0, -1, dtype=coefficients.dtype)


def _real_roots(coefficients: Tensor) -> list[float]:
    """Return a polynomial's real roots in ``numpy.roots`` order: companion-matrix eigenvalues, then trailing zeros."""
    nonzero = torch.nonzero(coefficients).flatten().tolist()
    if not nonzero:
        return []
    trailing = coefficients.shape[0] - nonzero[-1] - 1
    trimmed = coefficients[nonzero[0] : nonzero[-1] + 1]
    roots: list[float] = []
    if trimmed.shape[0] > 1:
        companion = torch.diag(torch.ones(trimmed.shape[0] - 2, dtype=trimmed.dtype), -1)
        companion[0] = -trimmed[1:] / trimmed[0]
        roots = [root.real for root in torch.linalg.eigvals(companion).tolist() if root.imag == 0]
    return roots + [0.0] * trailing


def peak_prominence(curve: list[float], peak: int) -> tuple[float, int, int]:
    """Return the prominence of the peak at ``peak`` and its left and right bases, as scipy ``peak_prominences``.

    Each base is the lowest sample between the peak and the nearest higher sample on that side (or the end of the
    curve); the prominence is the peak's height above the higher of the two bases.
    """
    top = curve[peak]
    left_min = right_min = top
    left_base = right_base = i = peak
    while i >= 0 and curve[i] <= top:
        if curve[i] < left_min:
            left_min, left_base = curve[i], i
        i -= 1
    i = peak
    while i < len(curve) and curve[i] <= top:
        if curve[i] < right_min:
            right_min, right_base = curve[i], i
        i += 1
    return top - max(left_min, right_min), left_base, right_base


def _peak_width(curve: list[float], peak: int) -> float:
    """Width of the peak at ``peak`` at half its prominence: scipy ``peak_widths`` with ``rel_height=0.5``."""
    prominence, left_base, right_base = peak_prominence(curve, peak)
    height = curve[peak] - prominence * 0.5
    i = peak
    while left_base < i and height < curve[i]:
        i -= 1
    left = float(i) + ((height - curve[i]) / (curve[i + 1] - curve[i]) if curve[i] < height else 0.0)
    i = peak
    while i < right_base and height < curve[i]:
        i += 1
    right = float(i) - ((height - curve[i]) / (curve[i - 1] - curve[i]) if curve[i] < height else 0.0)
    return right - left


def brightfield_contrast(intensity: Tensor, *, per_plane: bool = False, bg_filter: bool = False) -> Tensor:
    """Return the normalized brightfield contrast ``I / mean(I) - 1`` the phase systems measure.

    Parameters
    ----------
    intensity : Tensor
        Raw intensities, shape ``(..., Z, Y, X)``.
    per_plane : bool
        Divide each z plane by its own mean (3.x thin model); otherwise divide each stack by its mean over
        ``(Z, Y, X)`` (3.x thick model).
    bg_filter : bool
        First divide each plane by its background: its mean over a window ``X // 2`` pixels wide along y and x, with
        reflected edges (3.x ``bg_filter``, scipy ``uniform_filter``). Requires ``per_plane``.

    Returns
    -------
    Tensor
        Dimensionless contrast, same shape. Means are clamped at 1e-12 as in 3.x; the background is not.

    Raises
    ------
    ParameterError
        If ``bg_filter`` is set without ``per_plane``.
    ShapeError
        With ``bg_filter``, if the planes are narrower than 2 pixels along x (no window).
    """
    if bg_filter:
        if not per_plane:
            raise ParameterError(
                "brightfield_contrast bg_filter needs per_plane=True: 3.x filtered the background only for the thin "
                "model's per-plane normalization",
                path="bg_filter",
            )
        width = intensity.shape[-1] // 2
        if width < 1:
            raise ShapeError(
                "brightfield_contrast bg_filter needs planes at least 2 pixels wide, got shape "
                f"{tuple(intensity.shape)}"
            )
        intensity = intensity / _box_filter(intensity, width=width)
    dims = (-2, -1) if per_plane else (-3, -2, -1)
    return intensity / torch.clamp(intensity.mean(dim=dims, keepdim=True), min=1e-12) - 1


def _box_filter(planes: Tensor, *, width: int) -> Tensor:
    """Return the mean of ``planes`` over a ``width``-wide window along y, then x (scipy ``uniform_filter``).

    Pixel ``i`` averages ``i - width // 2`` through ``i - width // 2 + width - 1``; samples past an edge are mirrored
    about it with the edge sample repeated (``... b a | a b ...``), periodically for windows wider than the plane.
    """
    before = width // 2
    for dim in (-2, -1):
        n = planes.shape[dim]
        period = torch.arange(-before, n + width - before - 1, device=planes.device) % (2 * n)
        extended = planes.index_select(dim, torch.where(period < n, period, 2 * n - 1 - period)).movedim(dim, -1)
        mean = torch.nn.functional.avg_pool1d(extended.reshape(-1, 1, extended.shape[-1]), width, stride=1)
        planes = mean.reshape(*extended.shape[:-1], n).movedim(-1, dim)
    return planes


def apodize(estimate: Tensor, rolloff: float) -> Tensor:
    """Return ``estimate`` filtered by a raised-cosine window over its transverse frequencies (3.x apodization).

    Per axis the window is 1 below ``1 - rolloff`` of the Nyquist frequency and falls as a half cosine to 0 at
    Nyquist; the 2D window is the product over y and x. 3.x: ``solver.apodization_rolloff``.

    Parameters
    ----------
    estimate : Tensor
        Real, shape ``(..., Y, X)``.
    rolloff : float
        Fraction of the band, in ``[0, 1]``, over which the window falls; 0 returns ``estimate`` itself.

    Returns
    -------
    Tensor
        Same shape and dtype.
    """
    if rolloff == 0:
        return estimate
    ny, nx = estimate.shape[-2:]
    like = {"dtype": estimate.dtype, "device": estimate.device}

    def window(frequency: Tensor) -> Tensor:
        band = frequency.abs() * 2  # 1 at Nyquist
        start = 1 - rolloff
        return torch.where(band > start, 0.5 * (1 + torch.cos(math.pi * (band - start) / rolloff)), 1.0)

    weights = window(torch.fft.fftfreq(ny, **like))[:, None] * window(torch.fft.rfftfreq(nx, **like))
    return torch.fft.irfft2(torch.fft.rfft2(estimate) * weights, s=(ny, nx))


def most_structured_crop(volume: Tensor, *, size: int = 256) -> tuple[int, int, int]:
    """Return ``(y0, x0, side)`` of the square transverse crop of highest variance (3.x ``select_crop``).

    The side is ``min(size, Y, X)``. Candidates start every ``side // 2`` pixels along each axis, plus the last start
    that fits; the first highest in row-major order wins.

    Parameters
    ----------
    volume : Tensor
        Real, shape ``(..., Y, X)``.
    size : int
        Largest side of the crop.

    Returns
    -------
    tuple[int, int, int]
        First row, first column and side of the crop.
    """
    ny, nx = volume.shape[-2:]
    side = min(size, ny, nx)
    if side >= ny and side >= nx:
        return 0, 0, side
    step = max(side // 2, 1)

    def starts(extent: int) -> Tensor:
        return torch.tensor(sorted({*range(0, extent - side + 1, step), extent - side}), device=volume.device)

    rows, columns = starts(ny), starts(nx)
    planes = volume.double().reshape(-1, ny, nx)

    def window_totals(values: Tensor) -> Tensor:
        integral = torch.nn.functional.pad(values.sum(dim=0).cumsum(0).cumsum(1), (1, 0, 1, 0))
        top, bottom, left, right = rows[:, None], rows[:, None] + side, columns[None, :], columns[None, :] + side
        return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]

    count = planes.shape[0] * side * side
    mean = window_totals(planes) / count
    variance = window_totals(planes**2) / count - mean**2
    best = int(variance.argmax())
    return int(rows[best // len(columns)]), int(columns[best % len(columns)]), side
