"""WaveOrder 3.x fluorescence transfer functions and Tikhonov inverses, as a test-only reference for the 4.x port.

Copied from WaveOrder 3.x at tag ``3.0.5`` (``waveorder/models/isotropic_fluorescent_thin_3d.py``,
``waveorder/models/isotropic_fluorescent_thick_3d.py`` (its confocal path), ``waveorder/optics.py``,
``waveorder/sampling.py``, ``waveorder/util.py``, ``waveorder/filter.py``, ``waveorder/reconstruct.py`` and
``waveorder/api/_utils.py``; BSD-3-Clause, Chan Zuckerberg Biohub), with the clamp of the cropped thick PSF
(``_enforce_nonnegative_psf``) that 3.x added after 3.0.5 and the regression baselines were made with. The transfer
functions are unmodified 3.x: the sigmoid detection pupil (steepness 1e4 per cycle/um), the propagation kernel, the
incoherent transfer normalized by its largest magnitude, the confocal pinhole and PSF product, the central crop and
the clamp. Changes from the originals, none of which alters a result unless its option is set:

- numpy calls replaced by ``math`` and torch, and the ``float32`` casts by an explicit ``dtype``, so the reference
  runs in float64;
- scalar parameters only, and ``z_padding=0`` for the thick model (4.x pads outside the system);
- ``apply_filter_bank`` written out for one plane per channel (the shapes the thin model uses): ``sum_z
  IFFT2(filter_z FFT2(data_z))``;
- ``crop="lowest"`` replaces the 3.x central crop (which moves the zero frequency by one sample when an odd axis is
  oversampled by an even factor) with the 4.x crop, to characterize that fix; it defaults to 3.x;
- ``zero_frequency=math.pi / 4`` replaces the 3.x pinhole OTF value at zero frequency (0.5 before squaring) with
  the function's limit there, to characterize that fix; it defaults to 3.x;
- ``clamp=False`` leaves out the PSF clamp (the 3.0.5 thick transfer), to characterize it; it defaults to the clamp.
"""

import math

import torch
from torch import Tensor


def transverse_nyquist(wavelength: float, na_illumination: float, na_detection: float) -> float:
    return wavelength / (2 * (na_detection + na_illumination))


def central_cuboid(source: Tensor, target_shape: tuple[int, ...], *, crop: str = "central") -> Tensor:
    """3.x ``sampling.nd_fourier_central_cuboid``, or with ``crop="lowest"`` the 4.x lowest frequencies in FFT order."""
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


def radial_frequencies(yx_shape: tuple[int, int], pixel_size: float, *, dtype: torch.dtype) -> Tensor:
    """3.x ``util.generate_radial_frequencies``."""
    fy = torch.fft.fftfreq(yx_shape[0], pixel_size, dtype=dtype)
    fx = torch.fft.fftfreq(yx_shape[1], pixel_size, dtype=dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(fyy**2 + fxx**2)


def generate_pupil(frr: Tensor, na: float, wavelength: float, steepness: float = 1e4) -> Tensor:
    return torch.sigmoid(steepness * (na / wavelength - frr))


def generate_propagation_kernel(frr: Tensor, support: Tensor, wavelength: float, z: Tensor) -> Tensor:
    oblique = torch.sqrt(torch.clamp(1 - wavelength**2 * frr**2, min=0.0)) / wavelength
    return support[None] * torch.exp(1j * 2 * math.pi * z[:, None, None] * oblique[None])


def thin_z_positions(planes: int, dz: float, *, offset: float = 0.0, dtype: torch.dtype = torch.float64) -> Tensor:
    """3.x ``api._utils._position_list_from_shape_scale_offset``: ``(Z // 2 - i + offset) * dz``."""
    return (-torch.arange(planes, dtype=dtype) + planes // 2 + offset) * dz


def thin_fluorescence_transfer(
    yx_shape: tuple[int, int],
    yx_pixel_size: float,
    z: Tensor,
    *,
    wavelength: float,
    index: float,
    na_detection: float,
    crop: str = "central",
) -> Tensor:
    """3.x ``isotropic_fluorescent_thin_3d.calculate_transfer_function``: complex, shape ``(Z, Y, X)``."""
    factor = math.ceil(yx_pixel_size / transverse_nyquist(wavelength, na_detection, na_detection))
    fine = (yx_shape[0] * factor, yx_shape[1] * factor)
    frr = radial_frequencies(fine, yx_pixel_size / factor, dtype=z.dtype)
    pupil = generate_pupil(frr, na_detection, wavelength)
    kernel = generate_propagation_kernel(frr, pupil, wavelength / index, z)
    psf = torch.abs(torch.fft.ifft2(kernel, dim=(1, 2))) ** 2
    transfer = torch.fft.fft2(psf, dim=(1, 2))
    transfer = transfer / torch.clamp(torch.max(torch.abs(transfer)), min=1e-12)
    return torch.stack([central_cuboid(plane, yx_shape, crop=crop) for plane in transfer])


def thin_fluorescence_inverse(zyx_data: Tensor, transfer: Tensor, *, regularization_strength: float = 1e-3) -> Tensor:
    """3.x ``calculate_singular_system`` then ``apply_inverse_transfer_function`` (Tikhonov): shape ``(Y, X)``."""
    sfyx = transfer[None]
    s = torch.sqrt(torch.clamp(torch.sum(torch.abs(sfyx[0]) ** 2, dim=0), min=1e-12))
    vh = sfyx / (s[None, None] + 1e-12)
    u = torch.ones(1, 1, *s.shape, dtype=sfyx.dtype)
    s = s[None]
    s_reg = s / (s**2 + regularization_strength)
    inverse = torch.einsum("sj...,j...,jf...->fs...", u, s_reg, vh)  # (Z, 1, Y, X)
    spectrum = torch.fft.fft2(zyx_data, dim=(-2, -1))
    return torch.real(torch.fft.ifft2((inverse * spectrum[:, None]).sum(dim=0), dim=(-2, -1)))[0]


def thin_fluorescence_forward(yx_density: Tensor, transfer: Tensor) -> Tensor:
    """3.x ``apply_transfer_function`` without its constant background of 10: shape ``(Z, Y, X)``."""
    return torch.real(torch.fft.ifftn(torch.fft.fftn(yx_density)[None] * torch.real(transfer), dim=(1, 2)))


def axial_nyquist(wavelength: float, na_detection: float, index: float) -> float:
    n_on_lambda = index / wavelength
    return 1 / (2 * (n_on_lambda - math.sqrt(n_on_lambda**2 - (na_detection / wavelength) ** 2)))


def pinhole_aperture_otf(frr: Tensor, pinhole_diameter: float, *, zero_frequency: float = 0.5) -> Tensor:
    """3.x ``_calculate_pinhole_aperture_otf``: ``(J1(pi d k) / (2 d k))^2``, with ``0.5^2`` at ``d k <= 1e-10``.

    ``zero_frequency=math.pi / 4`` replaces the 3.x value there with the function's limit, to characterize that fix.
    """
    argument = pinhole_diameter * frr
    jinc = torch.where(argument > 1e-10, torch.special.bessel_j1(math.pi * argument) / (2 * argument), zero_frequency)
    return jinc**2


def thick_transfer(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    *,
    wavelength: float,
    index: float,
    na_detection: float,
    pinhole_diameter: float | None = None,
    zero_frequency: float = 0.5,
    clamp: bool = True,
) -> Tensor:
    """3.x ``isotropic_fluorescent_thick_3d.calculate_transfer_function`` with ``z_padding=0``: widefield when
    ``pinhole_diameter`` is ``None``, else confocal. ``clamp=False`` leaves out ``_enforce_nonnegative_psf`` (3.0.5)."""
    halve = 1 if pinhole_diameter is None else 2
    transverse = transverse_nyquist(wavelength, na_detection, na_detection) / halve
    axial = axial_nyquist(wavelength, na_detection, index) / halve
    yx_factor, z_factor = math.ceil(yx_pixel_size / transverse), math.ceil(z_pixel_size / axial)
    fine = (zyx_shape[0] * z_factor, zyx_shape[1] * yx_factor, zyx_shape[2] * yx_factor)
    frr = radial_frequencies(fine[1:], yx_pixel_size / yx_factor, dtype=torch.float64)
    z = torch.fft.ifftshift((torch.arange(fine[0], dtype=torch.float64) - fine[0] // 2) * (z_pixel_size / z_factor))
    pupil = generate_pupil(frr, na_detection, wavelength)
    kernel = generate_propagation_kernel(frr, pupil, wavelength / index, z)
    psf = torch.abs(torch.fft.ifft2(kernel, dim=(1, 2))) ** 2
    otf = torch.fft.fftn(psf, dim=(0, 1, 2))
    if pinhole_diameter is not None:
        pinhole = pinhole_aperture_otf(frr, pinhole_diameter, zero_frequency=zero_frequency)
        excitation = torch.abs(torch.fft.ifftn(otf, dim=(0, 1, 2)))
        detection = torch.abs(torch.fft.ifftn(otf * pinhole[None], dim=(0, 1, 2)))
        otf = torch.fft.fftn(excitation * detection, dim=(0, 1, 2))
    otf = otf / torch.clamp(torch.max(torch.abs(otf)), min=1e-12)
    otf = central_cuboid(otf, zyx_shape)
    return enforce_nonnegative_psf(otf) if clamp else otf


def enforce_nonnegative_psf(otf: Tensor) -> Tensor:
    """3.x ``_enforce_nonnegative_psf``: clamp the cropped PSF at 0, renormalize the OTF to unit peak."""
    psf = torch.clamp(torch.real(torch.fft.ifftn(otf, dim=(-3, -2, -1))), min=0)
    otf = torch.fft.fftn(psf, dim=(-3, -2, -1))
    return otf / torch.clamp(torch.max(torch.abs(otf)), min=1e-12)


def thick_forward(zyx_density: Tensor, otf: Tensor) -> Tensor:
    """3.x ``isotropic_fluorescent_thick_3d.apply_transfer_function`` without padding or its background of 10."""
    return torch.real(torch.fft.ifftn(torch.fft.fftn(zyx_density) * otf))


def thick_inverse(zyx_data: Tensor, otf: Tensor, *, regularization_strength: float = 1e-3) -> Tensor:
    """3.x ``apply_inverse_transfer_function`` (Tikhonov, no padding), ``tikhonov_regularized_inverse_filter``."""
    conj = torch.conj(otf)
    inverse = conj / (conj * otf + regularization_strength)
    return torch.real(torch.fft.ifftn(torch.fft.fftn(zyx_data) * inverse))
