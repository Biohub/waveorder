"""Ground-truth objects of the synthetic benchmark cases.

Copied unchanged from wo-4.0 (``waveorder/phantoms.py`` at 85282ce), so the simulated objects equal the saved
baselines' bit for bit. Only the two the regression suite uses are kept.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class Phantom:
    """A simulated sample with ground truth.

    Attributes
    ----------
    phase : Tensor
        Refractive index difference from background, shape ``(..., Y, X)``.
        Units: dimensionless (dn). Positive = higher RI than medium.
    absorption : Tensor
        Absorption coefficient, shape ``(..., Y, X)``.
        Units: arbitrary (non-negative).
    fluorescence : Tensor
        Fluorophore concentration, shape ``(..., Y, X)``.
        Units: arbitrary (non-negative).
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    metadata : dict
        All parameters used to generate this phantom, sufficient to
        reproduce it exactly.
    """

    phase: Tensor
    absorption: Tensor
    fluorescence: Tensor
    pixel_sizes: tuple[float, float, float]
    metadata: dict


def single_bead(  # noqa: PLR0913 (wo-4.0's signature)
    *,
    shape: tuple[int, int, int] = (64, 128, 128),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    bead_radius_um: float = 2.5,
    refractive_index_diff: float = 0.05,
    absorption_coefficient: float = 0.0,
    fluorescence_intensity: float = 1.0,
    fluorescence_background: float = 0.0,
    center: tuple[float, float, float] | None = None,
    blur_size_um: float = 0.1,
) -> Phantom:
    """Generate a single bead phantom.

    Creates a sphere with specified refractive index difference,
    absorption, and fluorescence intensity, blurred by a Gaussian
    to avoid sharp edges.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X) in pixels.
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    bead_radius_um : float
        Bead radius in um.
    refractive_index_diff : float
        RI difference from background (dn). Can be positive or negative.
    absorption_coefficient : float
        Peak absorption coefficient inside the bead.
    fluorescence_intensity : float
        Peak fluorophore concentration inside the bead.
    fluorescence_background : float
        Constant baseline added to the fluorescence channel everywhere
    center : tuple[float, float, float] or None
        Bead center in um, relative to volume center. None = (0, 0, 0).
    blur_size_um : float
        Gaussian blur standard deviation in um.

    Returns
    -------
    Phantom
        Phase, absorption, and fluorescence ground truth with metadata.

    Examples
    --------
    >>> phantom = single_bead(shape=(32, 64, 64), bead_radius_um=2.0)
    >>> phantom.phase.shape
    torch.Size([32, 64, 64])
    >>> phantom.metadata["bead_radius_um"]
    2.0
    """
    if center is None:
        center = (0.0, 0.0, 0.0)

    metadata = {
        "type": "single_bead",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "bead_radius_um": bead_radius_um,
        "refractive_index_diff": refractive_index_diff,
        "absorption_coefficient": absorption_coefficient,
        "fluorescence_intensity": fluorescence_intensity,
        "fluorescence_background": fluorescence_background,
        "center": list(center),
        "blur_size_um": blur_size_um,
    }

    mask = _sphere_mask(shape, pixel_sizes, bead_radius_um, center_um=center)
    blurred = _gaussian_blur(mask, pixel_sizes, blur_size_um)

    phase = blurred * refractive_index_diff
    absorption = blurred * absorption_coefficient
    fluorescence = blurred * fluorescence_intensity + fluorescence_background

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def grid_beads_gaussian(  # noqa: PLR0913 (wo-4.0's signature)
    *,
    shape: tuple[int, int, int] = (32, 256, 256),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    grid_shape: tuple[int, int] = (5, 5),
    sigma_um: tuple[float, float, float] = (0.15, 0.1, 0.1),
    fluorescence_intensity: float = 1.0,
    fluorescence_background: float = 0.0,
    refractive_index_diff: float = 0.0,
    z_plane_um: float = 0.0,
    margin_um: float | None = None,
) -> Phantom:
    """Pixel-aligned 2D grid of 3D Gaussian beads.

    Replaces the sphere-mask + blur composition used by ``grid_beads``
    with a direct anisotropic Gaussian intensity profile centered on the
    nearest *integer pixel* of each grid position. Both choices remove
    sub-pixel sampling artefacts (octagonal beads, peak-intensity drift
    across the grid) that surface when bead radii are close to one voxel.

    Each bead contributes ``exp(-(z^2 + y^2 + x^2) / (2 sigma^2))`` (with
    independent sigma per axis) and beads add together where they overlap.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape ``(Z, Y, X)`` in pixels.
    pixel_sizes : tuple[float, float, float]
        ``(z, y, x)`` pixel sizes in microns.
    grid_shape : tuple[int, int]
        Number of beads along ``(Y, X)``.
    sigma_um : tuple[float, float, float]
        Gaussian sigma along ``(z, y, x)`` in microns. Default ``(0.15, 0.1,
        0.1)`` matches a sub-diffraction bead at 100 nm pixels.
    fluorescence_intensity : float
        Peak fluorophore concentration at each bead center.
    fluorescence_background : float
        Constant baseline added to the fluorescence channel.
    refractive_index_diff : float
        RI difference from background (dn). Phase channel uses the same
        Gaussian profile multiplied by ``refractive_index_diff``.
    z_plane_um : float
        Axial position of the bead plane, relative to volume center.
    margin_um : float, optional
        Margin between beads and the FOV boundary in microns. If None,
        defaults to half a tile cell so beads sit at tile centers for
        the matching ``grid_shape`` tile partition.

    Returns
    -------
    Phantom
        Phase, absorption, and fluorescence ground truth, plus metadata
        with the snapped pixel centers (``centers_pix`` and ``centers_um``).
    """
    Z, Y, X = shape
    pz, py, px = pixel_sizes
    sz, sy, sx = sigma_um
    ny, nx = grid_shape

    y_extent = Y * py
    x_extent = X * px
    if margin_um is None:
        margin_y = y_extent / (2 * ny)
        margin_x = x_extent / (2 * nx)
    else:
        margin_y = margin_x = float(margin_um)

    ys_um = np.array([0.0]) if ny == 1 else np.linspace(-y_extent / 2 + margin_y, y_extent / 2 - margin_y, ny)
    xs_um = np.array([0.0]) if nx == 1 else np.linspace(-x_extent / 2 + margin_x, x_extent / 2 - margin_x, nx)
    z_pix = int(np.round(z_plane_um / pz)) + Z // 2

    ys_pix = (np.round(ys_um / py) + Y // 2).astype(int)
    xs_pix = (np.round(xs_um / px) + X // 2).astype(int)
    centers_pix = [(int(z_pix), int(yp), int(xp)) for yp in ys_pix for xp in xs_pix]
    centers_um = [
        (
            (cz - Z // 2) * pz,
            (cy - Y // 2) * py,
            (cx - X // 2) * px,
        )
        for cz, cy, cx in centers_pix
    ]

    metadata = {
        "type": "grid_beads_gaussian",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "grid_shape": list(grid_shape),
        "sigma_um": list(sigma_um),
        "fluorescence_intensity": fluorescence_intensity,
        "fluorescence_background": fluorescence_background,
        "refractive_index_diff": refractive_index_diff,
        "z_plane_um": z_plane_um,
        "centers_pix": [list(c) for c in centers_pix],
        "centers_um": [list(c) for c in centers_um],
    }

    z = (torch.arange(Z) - Z // 2) * pz
    y = (torch.arange(Y) - Y // 2) * py
    x = (torch.arange(X) - X // 2) * px
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    accum = torch.zeros(shape, dtype=torch.float32)
    for cz_um, cy_um, cx_um in centers_um:
        bead = torch.exp(
            -(((zz - cz_um) ** 2) / (2 * sz**2) + ((yy - cy_um) ** 2) / (2 * sy**2) + ((xx - cx_um) ** 2) / (2 * sx**2))
        )
        accum = accum + bead

    fluorescence = accum * fluorescence_intensity + fluorescence_background
    phase = accum * refractive_index_diff
    absorption = torch.zeros_like(accum)

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def _sphere_mask(
    shape: tuple[int, int, int],
    pixel_sizes: tuple[float, float, float],
    radius_um: float,
    *,
    center_um: tuple[float, float, float],
) -> Tensor:
    """Create a binary sphere mask.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X).
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    radius_um : float
        Sphere radius in um.
    center_um : tuple[float, float, float]
        Sphere center in um, relative to volume center.

    Returns
    -------
    Tensor
        Binary mask, shape (Z, Y, X), float32.
    """
    Z, Y, X = shape
    z = (torch.arange(Z) - Z // 2) * pixel_sizes[0] - center_um[0]
    y = (torch.arange(Y) - Y // 2) * pixel_sizes[1] - center_um[1]
    x = (torch.arange(X) - X // 2) * pixel_sizes[2] - center_um[2]

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    r_sq = xx**2 + yy**2 + zz**2

    mask = torch.zeros(shape, dtype=torch.float32)
    mask[r_sq < radius_um**2] = 1.0
    return mask


def _gaussian_blur(
    volume: Tensor,
    pixel_sizes: tuple[float, float, float],
    sigma_um: float,
) -> Tensor:
    """Apply Gaussian blur in Fourier domain.

    Parameters
    ----------
    volume : Tensor
        Input volume, shape (Z, Y, X).
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    sigma_um : float
        Gaussian standard deviation in um.

    Returns
    -------
    Tensor
        Blurred volume, non-negative, peak-normalized to [0, 1].
    """
    if sigma_um <= 0:
        peak = volume.max()
        if peak > 0:
            return volume / peak
        return volume

    Z, Y, X = volume.shape
    z = (torch.arange(Z) - Z // 2) * pixel_sizes[0]
    y = (torch.arange(Y) - Y // 2) * pixel_sizes[1]
    x = (torch.arange(X) - X // 2) * pixel_sizes[2]

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    r_sq = xx**2 + yy**2 + zz**2

    gaussian = torch.exp(-r_sq / (2 * sigma_um**2))

    blurred = torch.real(torch.fft.ifftn(torch.fft.fftn(volume) * torch.fft.fftn(torch.fft.ifftshift(gaussian))))
    blurred = torch.clamp(blurred, min=0)

    peak = blurred.max()
    if peak > 0:
        blurred = blurred / peak

    return blurred
