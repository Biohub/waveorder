"""WaveOrder 3.x focus finding and thin-model background filter, as a test-only reference for the 4.x port.

Copied from WaveOrder 3.x at tag ``3.0.5`` (``waveorder/focus.py``: ``compute_midband_power``,
``focus_from_transverse_band``; ``waveorder/util.py``: ``gen_coordinate``, ``inten_normalization``; BSD-3-Clause,
Chan Zuckerberg Biohub). The z padding reference is :func:`tests.nn.legacy_phase.pad_along_z`.

3.x calls two scipy functions; scipy is not a 4.x dependency, so they are vendored here as minimal copies of the scipy
algorithms (BSD-3-Clause, SciPy developers): ``scipy.ndimage.uniform_filter`` with its defaults ``mode="reflect"``
(the half-sample symmetric extension, repeated for windows longer than the line) and ``origin=0``, and
``scipy.signal.peak_widths`` (``rel_height=0.5``) with the ``peak_prominences`` it calls (``wlen=None``), from
``scipy/signal/_peak_finding_utils.pyx``. Both copies matched scipy 1.18.1 (uniform filter to 2e-15 on line lengths
shorter and longer than the window; peak widths exactly on 3000 random and tied curves) when this file was written.

Changes from the originals, none of which alters a result beyond rounding:

- numpy calls replaced by torch: ``np.polyfit`` by the same column-scaled least squares (``gelsd``, ``rcond`` of
  ``len(x) * eps``; the scaling picks the minimum-norm solution when the fit is underdetermined), ``np.roots`` by the
  eigenvalues of the same companion matrix, ``np.poly1d`` by Horner's rule;
- the ``float32`` casts are dropped so the reference runs in the input's dtype (float64 for the tight checks);
- the diagnostic plot and the input warnings are dropped;
- the uniform filter averages each window directly where scipy keeps a running sum.
"""

import torch
from torch import Tensor


def gen_coordinate_frequencies(img_dim: tuple[int, int], ps: float, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """3.x ``util.gen_coordinate``, frequency outputs only: ``(fxx, fyy)``, each ``(Ny, Nx)``."""
    N, M = img_dim
    fx = torch.fft.fftfreq(M, float(ps), dtype=dtype)
    fy = torch.fft.fftfreq(N, float(ps), dtype=dtype)
    fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")
    return fxx, fyy


def compute_midband_power(
    input_tensor: Tensor,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
) -> Tensor:
    """3.x ``focus.compute_midband_power``: ``(Y, X)`` gives a scalar, ``(Z, Y, X)`` one power per slice."""
    squeeze = input_tensor.ndim == 2
    if squeeze:
        input_tensor = input_tensor.unsqueeze(0)
    _, Y, X = input_tensor.shape
    fxx, fyy = gen_coordinate_frequencies((Y, X), pixel_size, dtype=input_tensor.dtype)
    frr = torch.sqrt(fxx**2 + fyy**2)
    cutoff = 2 * NA_det / lambda_ill
    mask = torch.logical_and(frr > cutoff * midband_fractions[0], frr < cutoff * midband_fractions[1])
    abs_fft = torch.abs(torch.fft.fftn(input_tensor, dim=(-2, -1)))
    result = abs_fft[:, mask].sum(dim=-1)
    return result.squeeze(0) if squeeze else result


def peak_prominences(x: list[float], peak: int) -> tuple[float, int, int]:
    """scipy ``_peak_prominences`` for one peak with ``wlen=None``: ``(prominence, left_base, right_base)``."""
    i_min, i_max = 0, len(x) - 1
    i = left_base = peak
    left_min = x[peak]
    while i_min <= i and x[i] <= x[peak]:
        if x[i] < left_min:
            left_min = x[i]
            left_base = i
        i -= 1
    i = right_base = peak
    right_min = x[peak]
    while i <= i_max and x[i] <= x[peak]:
        if x[i] < right_min:
            right_min = x[i]
            right_base = i
        i += 1
    return x[peak] - max(left_min, right_min), left_base, right_base


def peak_widths(x: list[float], peak: int, rel_height: float = 0.5) -> tuple[float, float, float, float]:
    """scipy ``peak_widths`` for one peak: ``(width, width_height, left_ip, right_ip)``."""
    prominence, i_min, i_max = peak_prominences(x, peak)
    height = x[peak] - prominence * rel_height
    i = peak
    while i_min < i and height < x[i]:
        i -= 1
    left_ip = float(i)
    if x[i] < height:
        left_ip += (height - x[i]) / (x[i + 1] - x[i])
    i = peak
    while i < i_max and height < x[i]:
        i += 1
    right_ip = float(i)
    if x[i] < height:
        right_ip -= (height - x[i]) / (x[i - 1] - x[i])
    return right_ip - left_ip, height, left_ip, right_ip


def _argextreme(values: list[float], mode: str) -> int:
    """``np.argmax`` / ``np.argmin``: the first index of the extreme."""
    extreme = max(values) if mode == "max" else min(values)
    return values.index(extreme)


def _polyfit(y: Tensor, order: int) -> Tensor:
    x = torch.arange(y.shape[0], dtype=torch.float64)
    lhs = torch.stack([x**k for k in range(order, -1, -1)], dim=1)
    scale = torch.sqrt((lhs * lhs).sum(dim=0))
    rcond = len(x) * torch.finfo(torch.float64).eps
    c = torch.linalg.lstsq(lhs / scale, y[:, None].double(), rcond=rcond, driver="gelsd").solution[:, 0]
    return c / scale


def _poly1d(coeffs: Tensor, x: float | Tensor) -> Tensor:
    value = torch.zeros_like(torch.as_tensor(x, dtype=torch.float64))
    for c in coeffs:
        value = value * x + c
    return value


def _roots(p: Tensor) -> list[complex]:
    nonzero = [i for i, c in enumerate(p.tolist()) if c != 0]
    if not nonzero:
        return []
    trailing_zeros = len(p) - nonzero[-1] - 1
    p = p[nonzero[0] : nonzero[-1] + 1]
    roots: list[complex] = []
    if len(p) > 1:
        A = torch.diag(torch.ones(len(p) - 2, dtype=p.dtype), -1)
        A[0, :] = -p[1:] / p[0]
        roots = torch.linalg.eigvals(A).tolist()
    return roots + [0j] * trailing_zeros


def focus_from_transverse_band(
    zyx_array: Tensor,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
    *,
    mode: str = "max",
    polynomial_fit_order: int | None = None,
    threshold_FWHM: float = 0,
    enable_subpixel_precision: bool = False,
) -> tuple[int | float | None, dict]:
    """3.x ``focus.focus_from_transverse_band`` with ``return_statistics=True``: ``(in_focus_index, peak_stats)``."""
    peak_stats = {"peak_index": None, "peak_FWHM": None}
    if zyx_array.shape[0] == 1:
        return 0, peak_stats
    midband_sum = compute_midband_power(zyx_array, NA_det, lambda_ill, pixel_size, midband_fractions).double()
    values = midband_sum.tolist()
    if polynomial_fit_order is None:
        peak_index = _argextreme(values, mode)
    else:
        x = torch.arange(len(values), dtype=torch.float64)
        coeffs = _polyfit(midband_sum, polynomial_fit_order)
        fitted = _poly1d(coeffs, x).tolist()
        if enable_subpixel_precision:
            derivative = coeffs[:-1] * torch.arange(len(coeffs) - 1, 0, -1, dtype=torch.float64)
            real_critical_points = [cp.real for cp in _roots(derivative) if cp.imag == 0 and 0 <= cp.real < len(values)]
            if real_critical_points:
                critical_values = [float(_poly1d(coeffs, cp)) for cp in real_critical_points]
                peak_index = real_critical_points[_argextreme(critical_values, mode)]
            else:
                peak_index = float(_argextreme(fitted, mode))
        else:
            peak_index = _argextreme(fitted, mode)
    if enable_subpixel_precision and polynomial_fit_order is not None:
        integer_peak_index = round(peak_index)
    else:
        integer_peak_index = int(peak_index)
    peak_FWHM = peak_widths(values, integer_peak_index)[0]
    peak_stats.update({"peak_index": int(peak_index), "peak_FWHM": float(peak_FWHM)})
    in_focus_index = peak_index if peak_FWHM >= threshold_FWHM else None
    return in_focus_index, peak_stats


def uniform_filter1d(values: Tensor, size: int, dim: int) -> Tensor:
    """scipy ``ndimage.uniform_filter1d`` with ``mode="reflect"``, ``origin=0``: window ``[i - size // 2, ...]``."""
    n = values.shape[dim]
    before = size // 2
    after = size - before - 1
    period = torch.arange(-before, n + after) % (2 * n)
    extended = values.index_select(dim, torch.where(period < n, period, 2 * n - 1 - period))
    return extended.unfold(dim, size, 1).mean(dim=-1)


def uniform_filter(image: Tensor, size: int) -> Tensor:
    """scipy ``ndimage.uniform_filter`` with one ``size`` for every axis: 1D filters along each axis in order."""
    for dim in range(image.ndim):
        image = uniform_filter1d(image, size, dim)
    return image


def inten_normalization(img_stack: Tensor, *, bg_filter: bool = True) -> Tensor:
    """3.x ``util.inten_normalization``: per-plane ``I / mean(I) - 1``, after dividing by a uniform filter if asked."""
    batched = img_stack.ndim == 4
    if not batched:
        img_stack = img_stack.unsqueeze(0)
    B, Z, _, X = img_stack.shape
    if bg_filter:
        img_norm_stack = torch.zeros_like(img_stack)
        for b in range(B):
            for i in range(Z):
                img_norm_stack[b, i] = img_stack[b, i] / uniform_filter(img_stack[b, i], size=X // 2)
        means = torch.clamp(torch.mean(img_norm_stack, dim=(-2, -1), keepdim=True), min=1e-12)
        img_norm_stack = img_norm_stack / means - 1
    else:
        means = torch.clamp(torch.mean(img_stack, dim=(-2, -1), keepdim=True), min=1e-12)
        img_norm_stack = img_stack / means - 1
    return img_norm_stack if batched else img_norm_stack.squeeze(0)
