"""WaveOrder 3.x phase transfer functions and Tikhonov inverses, as a test-only reference for the 4.x port.

Copied from WaveOrder 3.x (``waveorder/optics.py``, ``waveorder/sampling.py``, ``waveorder/util.py``,
``waveorder/models/phase_thick_3d.py``, ``waveorder/models/isotropic_thin_3d.py``, ``waveorder/filter.py`` and
``waveorder/reconstruct.py`` at the commit this branch started from; BSD-3-Clause, Chan Zuckerberg Biohub). The
transfer functions are unmodified 3.x: same sigmoid detection pupil (steepness 1e4 per cycle/um), same Green's
function guard, same z flip for ``invert_phase_contrast``. Changes from the originals, none of which alters a result:

- numpy calls replaced by ``math`` and torch; ``float32`` casts replaced by an explicit ``dtype`` so the reference
  runs in float64 for the 1e-10 parity checks;
- scalar parameters only (the batched paths loop over the same code);
- the one option, ``crop="lowest"``, replaces the 3.x central crop (which moves the zero frequency by one sample when
  an odd axis is oversampled by an even factor) with the 4.x crop, to characterize that fix; it defaults to 3.x.
"""

import math
from typing import Literal

import torch
from torch import Tensor

type Crop = Literal["central", "lowest"]


def transverse_nyquist(wavelength: float, na_illumination: float, na_detection: float) -> float:
    return wavelength / (2 * (na_detection + na_illumination))


def axial_nyquist(wavelength: float, na_detection: float, index: float) -> float:
    n_on_lambda = index / wavelength
    cutoff = n_on_lambda - math.sqrt(n_on_lambda**2 - (na_detection / wavelength) ** 2)
    return 1 / (2 * cutoff)


def central_cuboid(source: Tensor, target_shape: tuple[int, ...], *, crop: Crop = "central") -> Tensor:
    """3.x ``nd_fourier_central_cuboid``, or with ``crop="lowest"`` the lowest frequencies in FFT order (4.x)."""
    if crop == "lowest":
        for dim, size in zip(range(-len(target_shape), 0), target_shape, strict=True):
            full = source.shape[dim]
            keep = torch.cat([torch.arange((size + 1) // 2), torch.arange(full - size // 2, full)])
            source = source.index_select(dim, keep)
        return source
    n_dim = len(target_shape)
    dims = tuple(range(source.ndim))[-n_dim:]
    center = tuple(
        slice((s - o) // 2, (s - o) // 2 + o) for s, o in zip(source.shape[-n_dim:], target_shape, strict=True)
    )
    center = (slice(None),) * (source.ndim - n_dim) + center
    return torch.fft.ifftshift(torch.fft.fftshift(source, dim=dims)[center], dim=dims)


def frequencies(yx_shape: tuple[int, int], dy: float, dx: float, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    fy = torch.fft.fftfreq(yx_shape[0], dy, dtype=dtype)
    fx = torch.fft.fftfreq(yx_shape[1], dx, dtype=dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    return fyy, fxx


def generate_pupil(frr: Tensor, na: float, wavelength: float, steepness: float = 1e4) -> Tensor:
    return torch.sigmoid(steepness * (na / wavelength - frr))


def generate_tilted_pupil(
    fxx: Tensor, fyy: Tensor, na: Tensor, wavelength: float, index: float, zenith: Tensor, azimuth: Tensor
) -> Tensor:
    slope = 4.0
    k = index / wavelength
    cos_alpha_max = torch.sqrt(torch.clamp(1 - (na / index) ** 2, min=0.0))
    with torch.no_grad():
        df = torch.min(torch.abs(fxx[0, 1] - fxx[0, 0]), torch.abs(fyy[1, 0] - fyy[0, 0]))
        pixel_slope = slope / df
        fz_sq = k**2 - fxx**2 - fyy**2
        inside_sphere = (fz_sq >= 0).to(fxx.dtype)
        fz = torch.sqrt(torch.clamp(fz_sq, min=0.0))
    sx = torch.sin(zenith) * torch.cos(azimuth)
    sy = torch.sin(zenith) * torch.sin(azimuth)
    sz = torch.cos(zenith)
    dot = fxx * sx + fyy * sy + fz * sz
    return torch.sigmoid(pixel_slope * (dot - k * cos_alpha_max)) * inside_sphere


def generate_propagation_kernel(frr: Tensor, support: Tensor, wavelength: float, z: Tensor) -> Tensor:
    oblique = torch.sqrt(torch.clamp(1 - wavelength**2 * frr**2, min=0.0)) / wavelength
    return support[None] * torch.exp(1j * 2 * math.pi * z[:, None, None] * oblique[None])


def generate_greens_function_z(frr: Tensor, support: Tensor, wavelength: float, z: Tensor) -> Tensor:
    """3.x ``generate_greens_function_z`` with ``axially_even=False``, as the thick phase model calls it."""
    oblique = torch.sqrt(torch.clamp(1 - wavelength**2 * frr**2, min=0.0)) / wavelength
    safe_denom = oblique + (1 - support) + 1e-15
    return (
        -1j
        / 4
        / math.pi
        * support[None]
        * torch.exp(1j * 2 * math.pi * z[:, None, None] * oblique[None])
        / safe_denom[None]
    )


def wotf_2d(illumination: Tensor, detection: Tensor) -> tuple[Tensor, Tensor]:
    sp_hat = torch.fft.fft2(illumination * detection)
    p_hat = torch.fft.fft2(detection)
    h1 = torch.fft.ifft2(torch.conj(sp_hat) * p_hat)
    h2 = torch.fft.ifft2(sp_hat * torch.conj(p_hat))
    inv_norm = 1 / torch.sum(illumination * detection * torch.conj(detection), dim=(-2, -1), keepdim=True)
    return (h1 + h2) * inv_norm, (h1 - h2) * (inv_norm * 1j)


def wotf_3d(
    illumination: Tensor, detection: Tensor, propagation: Tensor, greens: Tensor, dz: float
) -> tuple[Tensor, Tensor]:
    window = torch.fft.ifftshift(
        torch.hann_window(propagation.shape[0], periodic=False, dtype=illumination.dtype, device=propagation.device)
    )
    sphz = (illumination * detection).unsqueeze(-3) * propagation
    pg = detection.unsqueeze(0) * greens
    sphz_hat = torch.fft.fft2(sphz, dim=(-2, -1))
    pg_hat = torch.fft.fft2(pg, dim=(-2, -1))
    h1 = torch.fft.fft(torch.fft.ifft2(torch.conj(sphz_hat) * pg_hat, dim=(-2, -1)) * window[:, None, None], dim=-3)
    h2 = torch.fft.fft(torch.fft.ifft2(sphz_hat * torch.conj(pg_hat), dim=(-2, -1)) * window[:, None, None], dim=-3)
    direct = torch.sum(illumination * detection * torch.conj(detection), dim=(-2, -1), keepdim=True).unsqueeze(-3)
    return (h1 + h2) / direct * dz, 1j * (h1 - h2) / direct * dz


def thick_transfer(
    zyx_shape: tuple[int, int, int],
    pixel_size: tuple[float, float, float],
    *,
    wavelength: float,
    index: float,
    na_illumination: float | Tensor,
    na_detection: float,
    zenith: float | Tensor = 0.0,
    azimuth: float | Tensor = 0.0,
    invert_phase_contrast: bool = False,
    z_padding: int = 0,
    crop: Crop = "central",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """3.x ``phase_thick_3d.calculate_transfer_function``: ``(real_potential_tf, imag_potential_tf)``, full spectra."""
    dz, dy, dx = pixel_size
    na_ill = torch.as_tensor(na_illumination, dtype=dtype)
    nyquist = transverse_nyquist(wavelength, float(na_ill), na_detection)
    y_factor, x_factor = math.ceil(dy / nyquist), math.ceil(dx / nyquist)
    z_factor = math.ceil(dz / axial_nyquist(wavelength, na_detection, index))
    up_shape = (zyx_shape[0] * z_factor, zyx_shape[1] * y_factor, zyx_shape[2] * x_factor)
    up_dy, up_dx, up_dz = dy / y_factor, dx / x_factor, dz / z_factor
    out_shape = (zyx_shape[0] + 2 * z_padding, zyx_shape[1], zyx_shape[2])

    fyy, fxx = frequencies(up_shape[1:], up_dy, up_dx, dtype=dtype)
    frr = torch.sqrt(fyy**2 + fxx**2)
    z_total = up_shape[0] + 2 * z_padding
    z = torch.fft.ifftshift((torch.arange(z_total, dtype=dtype) - z_total // 2) * up_dz)
    if invert_phase_contrast:
        z = torch.flip(z, dims=(0,))
    det = generate_pupil(frr, na_detection, wavelength)
    support = det
    propagation = generate_propagation_kernel(frr, support, wavelength / index, z)
    greens = generate_greens_function_z(frr, support, wavelength / index, z)
    ill = generate_tilted_pupil(
        fxx, fyy, na_ill, wavelength, index, torch.as_tensor(zenith, dtype=dtype), torch.as_tensor(azimuth, dtype=dtype)
    )
    real_tf, imag_tf = wotf_3d(ill, det, propagation, greens, up_dz)
    return central_cuboid(real_tf, out_shape, crop=crop), central_cuboid(imag_tf, out_shape, crop=crop)


def thin_transfer(
    yx_shape: tuple[int, int],
    yx_pixel_size: tuple[float, float],
    z_positions: Tensor,
    *,
    wavelength: float,
    index: float,
    na_illumination: float | Tensor,
    na_detection: float,
    zenith: float | Tensor = 0.0,
    azimuth: float | Tensor = 0.0,
    invert_phase_contrast: bool = False,
    crop: Crop = "central",
) -> tuple[Tensor, Tensor]:
    """3.x ``isotropic_thin_3d.calculate_transfer_function``: ``(absorption_tf, phase_tf)``, ``(Z, Y, X)``."""
    dtype = z_positions.dtype
    dy, dx = yx_pixel_size
    na_ill = torch.as_tensor(na_illumination, dtype=dtype)
    nyquist = transverse_nyquist(wavelength, float(na_ill), na_detection)
    y_factor, x_factor = math.ceil(dy / nyquist), math.ceil(dx / nyquist)
    up_shape = (yx_shape[0] * y_factor, yx_shape[1] * x_factor)
    if float(na_ill) >= na_detection:  # 3.x clamps with a warning; 4.x rejects at construction
        na_ill = 0.9 * torch.as_tensor(na_detection, dtype=dtype)
    z = -z_positions if invert_phase_contrast else z_positions
    fyy, fxx = frequencies(up_shape, dy / y_factor, dx / x_factor, dtype=dtype)
    frr = torch.sqrt(fyy**2 + fxx**2)
    det = generate_pupil(frr, na_detection, wavelength)
    support = det
    propagation = generate_propagation_kernel(frr, support, wavelength / index, z)
    det_prop = det.unsqueeze(0) * propagation
    ill = generate_tilted_pupil(
        fxx, fyy, na_ill, wavelength, index, torch.as_tensor(zenith, dtype=dtype), torch.as_tensor(azimuth, dtype=dtype)
    )
    absorption_tf, phase_tf = wotf_2d(ill, det_prop)
    return central_cuboid(absorption_tf, yx_shape, crop=crop), central_cuboid(phase_tf, yx_shape, crop=crop)


def thin_z_positions(planes: int, dz: float, *, offset: float = 0.0, dtype: torch.dtype = torch.float64) -> Tensor:
    """3.x ``_position_list_from_shape_scale_offset``: slice ``i`` sits at ``(Z // 2 - i + offset) dz``."""
    return (-torch.arange(planes, dtype=dtype) + planes // 2 + offset) * dz


def pad_along_z(zyx: Tensor, z_padding: int) -> Tensor:
    """3.x ``pad_zyx_along_z``: flipped edge slices when the pad is shorter than the stack, zeros otherwise."""
    if z_padding == 0:
        return zyx
    padded = torch.nn.functional.pad(zyx[None], (0, 0, 0, 0, z_padding, z_padding), mode="constant", value=0)[0]
    if z_padding < zyx.shape[0]:
        padded[:z_padding] = torch.flip(zyx[:z_padding], dims=[0])
        padded[-z_padding:] = torch.flip(zyx[-z_padding:], dims=[0])
    return padded


def thick_inverse(
    zyx_data: Tensor,
    real_tf: Tensor,
    imag_tf: Tensor,
    *,
    z_padding: int = 0,
    absorption_ratio: float = 0.0,
    regularization_strength: float = 1e-3,
) -> Tensor:
    """3.x ``phase_thick_3d.apply_inverse_transfer_function`` (Tikhonov, no apodization)."""
    padded = pad_along_z(zyx_data, z_padding)
    zyx = padded / torch.clamp(padded.mean(), min=1e-12) - 1
    effective = real_tf + absorption_ratio * imag_tf
    inverse = torch.conj(effective) / (torch.conj(effective) * effective + regularization_strength)
    phase = torch.real(torch.fft.ifftn(torch.fft.fftn(zyx, dim=(-3, -2, -1)) * inverse, dim=(-3, -2, -1)))
    return phase[z_padding:-z_padding] if z_padding else phase


def thin_singular_system(absorption_tf: Tensor, phase_tf: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """3.x ``calculate_singular_system`` (``use_svd=True``).

    Returns ``U (2, 2, Y, X)``, ``S (2, Y, X)`` and ``Vh (2, Z, Y, X)``.
    """
    stacked = torch.stack((absorption_tf, phase_tf), dim=0)  # (s, Z, Y, X)
    u, s, vh = torch.linalg.svd(stacked.permute(2, 3, 0, 1), full_matrices=False)
    return u.permute(2, 3, 0, 1), s.permute(2, 0, 1), vh.permute(2, 3, 0, 1)


def thin_inverse(
    zyx_data: Tensor, singular_system: tuple[Tensor, Tensor, Tensor], *, regularization_strength: float = 1e-3
) -> tuple[Tensor, Tensor]:
    """3.x ``isotropic_thin_3d.apply_inverse_transfer_function`` (Tikhonov, no background filter or apodization)."""
    means = torch.clamp(torch.mean(zyx_data, dim=(-2, -1), keepdim=True), min=1e-12)
    zyx = zyx_data / means - 1
    u, s, vh = singular_system
    s_reg = s / (s**2 + regularization_strength)
    inverse = torch.einsum("sj...,j...,jf...->fs...", u, s_reg, vh)  # (Z, 2, Y, X)
    spectrum = torch.fft.fftn(zyx, dim=(-2, -1))
    out = torch.real(torch.fft.ifftn((inverse * spectrum[:, None]).sum(dim=0), dim=(-2, -1)))
    return out[0], out[1]


def thin_forward(absorption: Tensor, phase: Tensor, absorption_tf: Tensor, phase_tf: Tensor) -> Tensor:
    """3.x ``isotropic_thin_3d.apply_transfer_function`` without its constant background of 10."""
    data = torch.real(torch.fft.ifftn(torch.fft.fftn(absorption)[None] * absorption_tf, dim=(1, 2)))
    return data + torch.real(torch.fft.ifftn(torch.fft.fftn(phase)[None] * phase_tf, dim=(1, 2)))


def sphere(zyx_shape: tuple[int, int, int], pixel_size: tuple[float, float, float], *, radius: float) -> Tensor:
    """3.x ``generate_sphere_target``: a blurred unit sphere, ``(Z, Y, X)``, peak 1, float64."""
    dz, dy, dx = pixel_size
    nz, ny, nx = zyx_shape
    z = (torch.arange(nz, dtype=torch.float64) - nz // 2) * dz
    y = (torch.arange(ny, dtype=torch.float64) - ny // 2) * dy
    x = (torch.arange(nx, dtype=torch.float64) - nx // 2) * dx
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    rho2 = xx**2 + yy**2 + zz**2
    ball = (rho2 < radius**2).to(torch.float64)
    blur = 2 * min(dy, dx)
    gaussian = torch.exp(-rho2 / (2 * blur**2))
    blurred = torch.real(torch.fft.ifftn(torch.fft.fftn(ball) * torch.fft.fftn(torch.fft.ifftshift(gaussian))))
    blurred = torch.clamp(blurred, min=0)
    return blurred / blurred.max()
