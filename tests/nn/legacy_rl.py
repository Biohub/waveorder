"""WaveOrder 3.x Richardson-Lucy, RLGC and back projectors, as a test-only reference for the 4.x port.

Copied from WaveOrder 3.x ``waveorder/rlgc.py`` and ``waveorder/backprojector.py`` (main, commit c27e6d9 "Add
RL/RLGC deconvolution for 3D fluorescence"; BSD-3-Clause, Chan Zuckerberg Biohub). The numerics are unmodified.
Changes, none of which alters a result: the argument validation, the MPS branches and the docstrings are dropped,
and ``ValueError`` raising helpers are inlined.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

_EPS = 1e-12


def clip(x: Tensor, eps: float = _EPS) -> Tensor:
    return torch.clamp(x, min=eps)


def _coinflip(counts: Tensor, probability: float, generator: torch.Generator | None) -> Tensor:
    counts = counts.clamp_min(0.0).round_()
    probs = counts.new_tensor(probability).expand_as(counts)
    return torch.binomial(counts, probs, generator=generator)


def scaled_gradient_step(
    estimate: Tensor,
    measured: Tensor,
    forward: Callable[[Tensor], Tensor],
    transpose: Callable[[Tensor], Tensor],
    transpose_ones: Tensor,
    *,
    method: str = "RL",
    background: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    rates = clip(forward(estimate) + background)
    gradient = transpose(measured / rates - 1.0)
    step_size = estimate / transpose_ones
    if method == "RLGC":
        heads = _coinflip(measured, 0.5, generator)
        heads_gradient = transpose(heads / rates - 0.5)
        tails_gradient = gradient - heads_gradient
        local_dot_product = transpose(forward(heads_gradient * tails_gradient))
        step_size = torch.where(local_dot_product <= 0, 0.0, step_size)
    return estimate + gradient * step_size, step_size


def richardson_lucy(
    measured: Tensor,
    forward: Callable[[Tensor], Tensor],
    transpose: Callable[[Tensor], Tensor],
    *,
    num_iterations: int,
    method: str = "RL",
    background: float = 0.0,
    guess: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    transpose_ones = clip(transpose(torch.ones_like(measured)))
    estimate = torch.ones_like(transpose_ones) if guess is None else clip(guess)
    for _ in range(num_iterations):
        updated, step_size = scaled_gradient_step(
            estimate,
            measured,
            forward,
            transpose,
            transpose_ones,
            method=method,
            background=background,
            generator=generator,
        )
        updated = clip(updated)
        if method == "RLGC" and torch.count_nonzero(step_size) == 0:
            estimate = updated
            break
        estimate = updated
    return estimate


def calculate_back_projector(
    optical_transfer_function: Tensor,
    back_projector: str = "matched",
    *,
    alpha: float | None = None,
    beta: float | None = None,
    order: int = 8,
) -> Tensor:
    """3.x ``calculate_back_projector`` with ``resolution_mode="fwhm"`` and ``beta_convention="reference"``."""
    if back_projector == "matched":
        return torch.conj_physical(optical_transfer_function)
    shape = tuple(optical_transfer_function.shape)
    device = optical_transfer_function.device
    real_dtype = optical_transfer_function.real.dtype
    if back_projector == "gaussian":
        fwhm_zyx_px = _psf_fwhm_zyx_px(optical_transfer_function)
        gaussian = _gaussian_back_projector(shape, fwhm_zyx_px, device, real_dtype)
        return gaussian.to(optical_transfer_function.dtype)
    resolution = _psf_fwhm_zyx_px(optical_transfer_function)
    cutoff_indices = tuple(size / res for size, res in zip(shape, resolution, strict=True))
    alpha_value = None if alpha is None else float(alpha)
    beta_value = None if beta is None else float(beta)
    normalized_otf = optical_transfer_function / torch.clamp(torch.max(torch.abs(optical_transfer_function)), min=_EPS)
    normalized_magnitude = torch.abs(normalized_otf)
    if alpha_value is None or beta_value is None:
        matched_cutoff_gain = _matched_cutoff_gain(normalized_magnitude, cutoff_indices)
        alpha_value = matched_cutoff_gain**2 if alpha_value is None else alpha_value
        beta_value = matched_cutoff_gain if beta_value is None else beta_value
    if back_projector == "butterworth":
        epsilon_squared = 1.0 / beta_value**2 - 1.0
        mask = _butterworth_mask(shape, cutoff_indices, epsilon_squared, order, device, real_dtype)
        return mask.to(optical_transfer_function.dtype)
    wiener = torch.conj(normalized_otf) / (normalized_magnitude.square() + alpha_value)
    if back_projector == "wiener":
        return wiener.to(optical_transfer_function.dtype)
    wiener_cutoff_gain = _wiener_cutoff_gain(wiener, cutoff_indices)
    effective_beta = beta_value * math.sqrt(wiener_cutoff_gain)
    epsilon_squared = (wiener_cutoff_gain / effective_beta) ** 2 - 1.0
    mask = _butterworth_mask(shape, cutoff_indices, epsilon_squared, order, device, real_dtype)
    return (wiener * mask).to(optical_transfer_function.dtype)


def _signed_frequency_indices(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.fft.fftfreq(size, device=device, dtype=dtype) * size


def _broadcast_along(values: Tensor, axis: int) -> Tensor:
    return values.reshape([-1 if a == axis else 1 for a in range(3)])


def _ellipsoidal_radius_squared(shape, scales, device, dtype) -> Tensor:
    radius_squared = torch.zeros(shape, device=device, dtype=dtype)
    for axis, (size, scale) in enumerate(zip(shape, scales, strict=True)):
        indices = _signed_frequency_indices(size, device, dtype) / scale
        radius_squared.add_(_broadcast_along(indices.square(), axis))
    return radius_squared


def _butterworth_mask(shape, cutoff_indices, epsilon_squared, order, device, dtype) -> Tensor:
    radius_squared = _ellipsoidal_radius_squared(shape, cutoff_indices, device, torch.float64)
    mask = radius_squared.pow_(order).mul_(epsilon_squared).add_(1.0).sqrt_().reciprocal_()
    return mask.to(dtype)


def _gaussian_back_projector(shape, fwhm_zyx_px, device, dtype) -> Tensor:
    sigmas = tuple(fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0))) for fwhm in fwhm_zyx_px)
    exponent = _ellipsoidal_radius_squared(shape, sigmas, device, torch.float64)
    kernel = torch.exp(-0.5 * exponent)
    kernel.div_(torch.clamp(kernel.sum(), min=_EPS))
    return torch.fft.fftn(kernel.to(dtype), dim=(-3, -2, -1))


def _psf_fwhm_zyx_px(optical_transfer_function: Tensor) -> tuple[float, float, float]:
    psf = torch.fft.fftshift(torch.real(torch.fft.ifftn(optical_transfer_function, dim=(-3, -2, -1))))
    peak_z, peak_y, peak_x = (int(index) for index in torch.unravel_index(torch.argmax(psf), psf.shape))
    return (
        _fwhm_1d(psf[:, peak_y, peak_x]),
        _fwhm_1d(psf[peak_z, :, peak_x]),
        _fwhm_1d(psf[peak_z, peak_y, :]),
    )


def _fwhm_1d(profile: Tensor) -> float:
    peak_index = int(torch.argmax(profile))
    peak_value = float(profile[peak_index])
    normalized = profile.cpu().to(torch.float64) / peak_value
    below_left = torch.nonzero(normalized[: peak_index + 1] < 0.5).flatten()
    below_right = torch.nonzero(normalized[peak_index:] < 0.5).flatten()

    def _interpolate(low_index: int, high_index: int) -> float:
        low_value = float(normalized[low_index])
        high_value = float(normalized[high_index])
        return low_index + (0.5 - low_value) / (high_value - low_value) * (high_index - low_index)

    left_index = int(below_left[-1])
    right_index = peak_index + int(below_right[0])
    return _interpolate(right_index - 1, right_index) - _interpolate(left_index, left_index + 1)


def _mean_gain_at_cutoff(profile: Tensor, cutoff: float) -> float:
    size = profile.shape[0]
    center = size // 2
    low = max(round(center - cutoff), 0)
    high = min(round(center + cutoff), size - 1)
    return float((profile[low] + profile[high]) / 2)


def _matched_cutoff_gain(normalized_magnitude: Tensor, cutoff_indices: tuple[float, ...]) -> float:
    magnitude = torch.fft.fftshift(normalized_magnitude)
    gains = []
    for axis, cutoff in enumerate(cutoff_indices):
        other_axes = tuple(a for a in range(3) if a != axis)
        gains.append(_mean_gain_at_cutoff(torch.amax(magnitude, dim=other_axes), cutoff))
    return sum(gains) / 3.0


def _wiener_cutoff_gain(wiener: Tensor, cutoff_indices: tuple[float, ...]) -> float:
    magnitude = torch.fft.fftshift(torch.abs(wiener))
    central_slice = magnitude[magnitude.shape[0] // 2]
    return _mean_gain_at_cutoff(torch.amax(central_slice, dim=0), cutoff_indices[2])
