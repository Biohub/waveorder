"""Metrics of a benchmark reconstruction.

Three groups:

- ``image_quality`` (always): distribution statistics and three sharpness metrics.
- ``with_reference`` (when an annotated reference is given): mse, ssim, correlation.
- ``with_phantom`` (synthetic cases): the same against the ground truth.

Every definition is copied from wo-4.0 (85282ce) so the numbers stay comparable with the saved baselines: midband
power, spectral flatness and total variation are its ``waveorder.optim.losses`` (negated back to "higher is
sharper"), and ``ssim`` is scikit-image's ``structural_similarity`` with its defaults (7-voxel uniform window,
reflected edges, sample covariance), written out here so the suite needs neither scikit-image nor scipy.
"""

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import mse_loss


def _frequency_radius(shape: tuple[int, int], pixel_size: float, device: torch.device | None = None) -> Tensor:
    """Radial spatial frequency of a ``(Y, X)`` grid in FFT order, float32 (3.x ``util.gen_coordinate``)."""
    ny, nx = shape
    fx = torch.fft.fftfreq(nx, pixel_size, device=device)
    fy = torch.fft.fftfreq(ny, pixel_size, device=device)
    fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")
    return torch.sqrt(fxx**2 + fyy**2)


def midband_power(
    volume: Tensor, *, NA_det: float, wavelength: float, pixel_size: float, fractions: tuple[float, float]
) -> float:
    """Mean Fourier amplitude per slice in the annulus ``fractions * 2 NA / wavelength``, averaged over slices."""
    slices = volume if volume.ndim == 3 else volume[None]
    radius = _frequency_radius(tuple(slices.shape[-2:]), pixel_size, device=slices.device)
    cutoff = 2 * NA_det / wavelength
    mask = (radius > cutoff * fractions[0]) & (radius < cutoff * fractions[1])
    amplitude = torch.abs(torch.fft.fftn(slices.float(), dim=(-2, -1)))
    power = amplitude[:, mask].sum(dim=-1)
    return (power / max(int(mask.sum()), 1)).mean().item()


def spectral_flatness(
    volume: Tensor, *, NA_det: float, wavelength: float, pixel_size: float, fractions: tuple[float, float]
) -> float:
    """Geometric over arithmetic mean of the Fourier amplitude in the midband annulus, averaged over slices."""
    slices = volume if volume.ndim == 3 else volume[None]
    radius = _frequency_radius(tuple(slices.shape[-2:]), pixel_size, device=slices.device)
    cutoff = 2 * NA_det / wavelength
    mask = (radius > cutoff * fractions[0]) & (radius < cutoff * fractions[1])
    values = []
    for plane in slices:
        band = torch.abs(torch.fft.fftn(plane.float(), dim=(-2, -1)))[mask]
        values.append(band.log().mean().exp() / (band.mean() + 1e-12))
    return torch.stack(values).mean().item()


def total_variation(volume: Tensor) -> float:
    """Mean absolute difference of neighbouring pixels along y and x, halved, averaged over slices."""
    slices = volume if volume.ndim == 3 else volume[None]
    values = [
        ((plane[1:, :] - plane[:-1, :]).abs().mean() + (plane[:, 1:] - plane[:, :-1]).abs().mean()) / 2
        for plane in slices
    ]
    return torch.stack(values).mean().item()


def distribution_stats(volume: Tensor, n_bins: int = 50) -> dict:
    """Return the maximum, minimum, mean and an ``n_bins`` histogram of ``volume``."""
    v = volume.float().flatten()
    hist = torch.histogram(v.cpu(), bins=n_bins)
    return {
        "max": volume.max().item(),
        "min": volume.min().item(),
        "mean": volume.mean().item(),
        "histogram": {
            "bin_edges": hist.bin_edges.tolist(),
            "counts": hist.hist.tolist(),
        },
    }


def _uniform_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Mean over a ``size`` window along every axis with reflected edges, axis by axis (``scipy.ndimage``)."""
    out = image
    half = size // 2
    for axis in range(image.ndim):
        pad = [(0, 0)] * image.ndim
        pad[axis] = (half + 1, half)
        padded = np.pad(out.astype(np.float64), pad, mode="symmetric")
        sums = np.cumsum(padded, axis=axis)
        upper = np.take(sums, range(size, sums.shape[axis]), axis=axis)
        lower = np.take(sums, range(sums.shape[axis] - size), axis=axis)
        out = ((upper - lower) / size).astype(image.dtype)
    return out


def structural_similarity(im1: np.ndarray, im2: np.ndarray, *, data_range: float, win_size: int = 7) -> float:
    """Mean structural similarity of two images or volumes (scikit-image's defaults, its float dtype)."""
    dtype = np.float32 if im1.dtype == np.float32 else np.float64
    im1, im2 = im1.astype(dtype, copy=False), im2.astype(dtype, copy=False)
    cov_norm = win_size**im1.ndim / (win_size**im1.ndim - 1)
    ux, uy = _uniform_filter(im1, win_size), _uniform_filter(im2, win_size)
    uxx, uyy = _uniform_filter(im1 * im1, win_size), _uniform_filter(im2 * im2, win_size)
    uxy = _uniform_filter(im1 * im2, win_size)
    vx, vy, vxy = cov_norm * (uxx - ux * ux), cov_norm * (uyy - uy * uy), cov_norm * (uxy - ux * uy)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    pad = (win_size - 1) // 2
    return float(s[tuple(slice(pad, n - pad) for n in s.shape)].mean(dtype=np.float64))


def comparison_metrics(recon: Tensor, target: Tensor) -> dict:
    """Return mse, ssim and Pearson correlation of ``recon`` against ``target`` (same shape).

    ``ssim`` uses the target's value range as its data range.
    """
    a, b = recon.float(), target.float()
    corr = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    a_np, b_np = a.cpu().numpy(), b.cpu().numpy()
    data_range = b_np.max() - b_np.min()
    return {
        "mse": mse_loss(a, b).item(),
        "ssim": structural_similarity(a_np, b_np, data_range=data_range),
        "correlation": corr,
    }


def compute_metrics(  # noqa: PLR0913 (the volume, the optics and the optional comparisons)
    recon: Tensor,
    *,
    NA_det: float,
    wavelength: float,
    pixel_size: float,
    reference: Tensor | None = None,
    phantom: Tensor | None = None,
    n_bins: int = 50,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
) -> dict:
    """Compute every metric group that applies.

    Parameters
    ----------
    recon : Tensor
        Reconstruction, ``(Y, X)`` or ``(Z, Y, X)``.
    NA_det : float
        Detection numerical aperture.
    wavelength : float
        Illumination (phase) or emission (fluorescence) wavelength, in the unit of ``pixel_size``.
    pixel_size : float
        Lateral pixel size of the object.
    reference : Tensor, optional
        Annotated reference, for ``with_reference``.
    phantom : Tensor, optional
        Ground truth, for ``with_phantom``.
    n_bins : int
        Histogram bins.
    midband_fractions : tuple[float, float]
        Inner and outer edge of the midband annulus as fractions of ``2 NA_det / wavelength``.

    Returns
    -------
    dict
        ``image_quality`` and, when given their inputs, ``with_reference`` and ``with_phantom``.
    """
    v = recon.float()
    result = {
        "image_quality": {
            **distribution_stats(v, n_bins=n_bins),
            "midband_power": midband_power(
                v, NA_det=NA_det, wavelength=wavelength, pixel_size=pixel_size, fractions=midband_fractions
            ),
            "spectral_flatness": spectral_flatness(
                v, NA_det=NA_det, wavelength=wavelength, pixel_size=pixel_size, fractions=midband_fractions
            ),
            "total_variation": total_variation(v),
        },
    }
    if reference is not None:
        result["with_reference"] = comparison_metrics(v, reference)
    if phantom is not None:
        result["with_phantom"] = comparison_metrics(v, phantom)
    return result
