"""Metrics that compare a student's estimate with its target, the reconstruction: pure functions on tensors.

Every function takes batched estimates and targets ``(B, F, Zout, Y, X)`` in the same units and returns one value per
sample, ``(B,)``; a sample that cannot be scored (a blank target) gets ``NaN``, which :func:`summarize` leaves out.
:func:`dynamic_range_ratio` falls below 1 for a student that regresses to the mean; :func:`passband_error` splits
the error into what the data determines (inside the transfer function's support) and invented content (outside).
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812 (the torch convention)
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn import ImagingSystem, Sampling
from waveorder.nn.functional import abs2, half_spectrum_weights, is_complex_matrix, to_half_spectrum

MIN_RANGE = 1e-3
"""The smallest range (max - min) of a target plane that is scored. A blank field of view spans about 1e-6 in
normalized units; dividing by such a range overflows float32, so flatter planes are left out."""

_K1, _K2 = 0.01, 0.03
"""The stabilizing constants of SSIM (Wang et al. 2004), as in scikit-image."""


def expect_pair(estimate: Tensor, target: Tensor, *, where: str) -> None:
    """Raise :class:`ShapeError` unless both are batched objects ``(B, F, Zout, Y, X)`` of one shape."""
    if estimate.ndim != 5 or estimate.shape != target.shape:
        raise ShapeError(
            f"{where} takes an estimate and a target of one shape (B, F, Zout, Y, X), got "
            f"{tuple(estimate.shape)} and {tuple(target.shape)}"
        )


def plane_ssim(estimate: Tensor, target: Tensor, *, window: int = 11, min_range: float = MIN_RANGE) -> Tensor:
    """Return the SSIM ``(B, F, Zout)`` of every plane, with ``NaN`` for planes whose target is blank.

    Matches scikit-image's ``structural_similarity`` with ``data_range = target.max() - target.min()``: a uniform
    ``window`` by ``window`` filter, ``K1 = 0.01``, ``K2 = 0.03``, sample covariance, mean over valid positions.
    Planes whose target spans less than ``min_range`` are blank and get no gradient.

    Raises
    ------
    ShapeError
        If the shapes differ or are not five-dimensional, or the window does not fit in a plane.
    ParameterError
        If ``window`` is below 2.
    """
    expect_pair(estimate, target, where="ssim")
    if window < 2:
        raise ParameterError(f"ssim window must be an int of at least 2 pixels, got {window!r}", path="window")
    *leading, ny, nx = estimate.shape
    if window > min(ny, nx):
        raise ShapeError(
            f"ssim window of {window} pixels does not fit in planes of {ny} by {nx} pixels; pass a smaller window "
            "or larger patches"
        )
    dtype = torch.promote_types(estimate.dtype, torch.float32)
    estimates = estimate.to(dtype).reshape(-1, 1, ny, nx)
    targets = target.to(dtype).reshape(-1, 1, ny, nx)
    span = targets.amax(dim=(-2, -1)) - targets.amin(dim=(-2, -1))
    keep = span.flatten() >= min_range
    values = torch.full((estimates.shape[0],), torch.nan, dtype=dtype, device=estimate.device)
    if keep.any():
        scale = span.flatten()[keep][:, None, None, None]
        values = values.masked_scatter(keep, _unit_ssim(estimates[keep] / scale, targets[keep] / scale, window))
    return values.reshape(leading)


def _unit_ssim(estimate: Tensor, target: Tensor, window: int) -> Tensor:
    """Return the mean SSIM ``(N,)`` of planes ``(N, 1, Y, X)`` whose data range is 1."""

    def mean(values: Tensor) -> Tensor:
        return F.avg_pool2d(values, window, stride=1)  # "valid": the border where the window does not fit is cropped

    count = window * window
    sample = count / (count - 1)
    mean_t, mean_e = mean(target), mean(estimate)
    var_t = sample * (mean(target * target) - mean_t * mean_t)
    var_e = sample * (mean(estimate * estimate) - mean_e * mean_e)
    covariance = sample * (mean(target * estimate) - mean_t * mean_e)
    c1, c2 = _K1**2, _K2**2
    numerator = (2 * mean_t * mean_e + c1) * (2 * covariance + c2)
    denominator = (mean_t * mean_t + mean_e * mean_e + c1) * (var_t + var_e + c2)
    return (numerator / denominator).mean(dim=(-3, -2, -1))


def ssim(estimate: Tensor, target: Tensor, *, window: int = 11) -> Tensor:
    """Return the structural similarity ``(B,)`` of each sample: the mean over its planes of :func:`plane_ssim`.

    1 is a perfect match; ``NaN`` for a sample whose target planes are all blank (see :data:`MIN_RANGE`). Raises as
    :func:`plane_ssim`.
    """
    return plane_ssim(estimate, target, window=window).flatten(1).nanmean(dim=1)


def quantiles(values: Tensor, fractions: Sequence[float]) -> Tensor:
    """Return quantiles ``(len(fractions), ...)`` of finite ``values`` ``(..., N)`` along the last axis.

    Linear interpolation, as :func:`torch.quantile` does, but for any ``N``; each fraction is in ``[0, 1]``.
    """
    ordered = values.sort(dim=-1).values
    positions = torch.tensor(list(fractions), dtype=ordered.dtype, device=ordered.device) * (ordered.shape[-1] - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    result = torch.lerp(ordered[..., lower], ordered[..., upper], positions - lower)
    return result.movedim(-1, 0)


def dynamic_range_ratio(estimate: Tensor, target: Tensor, *, low: float = 1, high: float = 99) -> Tensor:
    """Return the estimate's ``low``-to-``high`` percentile range over the target's, per sample, ``(B,)``.

    1 means the same range, below 1 a flatter estimate; ``NaN`` for a target without range.

    Raises
    ------
    ShapeError
        If the shapes differ or are not five-dimensional.
    ParameterError
        Unless ``0 <= low < high <= 100``.
    """
    expect_pair(estimate, target, where="dynamic_range_ratio")
    if not 0 <= low < high <= 100:
        raise ParameterError(
            f"dynamic_range_ratio takes percentiles 0 <= low < high <= 100, got low={low} and high={high}", path="low"
        )
    fractions = (low / 100, high / 100)
    dtype = torch.promote_types(estimate.dtype, torch.float32)
    estimate_low, estimate_high = quantiles(estimate.to(dtype).flatten(1), fractions)
    target_low, target_high = quantiles(target.to(dtype).flatten(1), fractions)
    span = target_high - target_low
    return torch.where(span > 0, (estimate_high - estimate_low) / span, torch.nan)


def transfer_support(system: ImagingSystem, sampling: Sampling, *, threshold: float = 1e-3) -> Tensor:
    """Return where a shift-invariant system's transfer function passes signal, on the object's half spectrum.

    A frequency is in the support when the transfer energy, summed over data channels, exceeds ``threshold**2``
    times the field's peak. Returns a boolean ``(F, Zobj, Y, X // 2 + 1)`` on the system's device (the
    ``torch.fft.rfftn`` layout; ``Zobj`` is 1 for a thin object).

    Raises
    ------
    IncompatibleModelError
        If the system is not shift-invariant on ``sampling`` or refuses it.
    """
    with torch.no_grad():
        operator = system.bind(sampling)
        fourier = operator.as_fourier()
        if fourier is None:
            raise IncompatibleModelError(
                f"transfer_support needs a shift-invariant system, whose operator is one transfer function, but "
                f"{type(system).__name__} binds a {type(operator).__name__} that has none; the passband error is "
                "defined for shift-invariant systems only, so leave it out for this one"
            )
        transfer = fourier.transfer
        energy = (abs2(transfer) if is_complex_matrix(transfer) else transfer.square()).sum(dim=0)
        peak = energy.amax(dim=(-3, -2, -1), keepdim=True)
        return energy > threshold**2 * peak


def passband_error(estimate: Tensor, target: Tensor, support: Tensor) -> dict[str, Tensor]:
    """Return the energy of ``estimate - target`` inside and outside a frequency support, per sample.

    ``support`` is a boolean mask broadcastable to ``(F, Zout, Y, X // 2 + 1)`` (:func:`transfer_support`).
    ``"inside"`` and ``"outside"`` are ``(B,)`` fractions of the target's energy (``NaN`` for a zero target); their
    sum is the normalized squared error.

    Raises
    ------
    ShapeError
        If the shapes differ or are not five-dimensional, or ``support`` is not a boolean mask that broadcasts.
    """
    expect_pair(estimate, target, where="passband_error")
    fields, nz, ny, nx = estimate.shape[1:]
    half = (fields, nz, ny, nx // 2 + 1)
    if support.dtype != torch.bool or not _broadcasts(tuple(support.shape), half):
        raise ShapeError(
            f"passband_error support must be a boolean mask that broadcasts to the half spectrum {half} of the "
            f"estimate, got {support.dtype} of shape {tuple(support.shape)}; transfer_support(system, sampling) "
            "returns one for the sampling of the data"
        )
    dtype = torch.promote_types(estimate.dtype, torch.float32)
    weights = half_spectrum_weights(nx, device=estimate.device, dtype=dtype)
    error = abs2(to_half_spectrum((estimate - target).to(dtype), norm="ortho")) * weights
    total = (abs2(to_half_spectrum(target.to(dtype), norm="ortho")) * weights).sum(dim=(1, 2, 3, 4))
    total = torch.where(total > 0, total, torch.nan)
    mask = support.to(error.device)
    inside = (error * mask).sum(dim=(1, 2, 3, 4))
    outside = (error * ~mask).sum(dim=(1, 2, 3, 4))
    return {"inside": inside / total, "outside": outside / total}


def _broadcasts(shape: tuple[int, ...], onto: tuple[int, ...]) -> bool:
    if len(shape) > len(onto):
        return False
    return all(have in (1, want) for have, want in zip(reversed(shape), reversed(onto), strict=False))


def summarize(values: Tensor) -> dict[str, float]:
    """Return ``"mean"``, ``"p10"``, ``"p50"`` and ``"p90"`` of the finite ``values`` (any shape); ``NaN`` for none."""
    flat = values.detach().flatten().to(torch.float64)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return dict.fromkeys(("mean", "p10", "p50", "p90"), float("nan"))
    p10, p50, p90 = quantiles(finite, (0.1, 0.5, 0.9)).tolist()
    return {"mean": float(finite.mean()), "p10": p10, "p50": p50, "p90": p90}
