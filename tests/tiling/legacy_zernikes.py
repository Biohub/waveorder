"""WaveOrder 3.x per-tile Zernike recovery, as a test-only reference for :func:`waveorder.tiling.recover_zernikes`.

Copied from WaveOrder 3.x at tag 3.0.5 (BSD-3-Clause, Chan Zuckerberg Biohub): ``waveorder/zernike.py``
(``zernike_noll``), ``waveorder/optics.py`` (``generate_pupil``, ``generate_propagation_kernel``) and
``waveorder/api/shift_variant_recovery.py`` (``_build_gaussian_bead_template``, ``_build_pupil_grids``,
``_psf_from_coefs`` and the ``loss="mse"`` path of ``_optimize``). Changes from the originals, none of which alters a
result:

- numpy and xarray replaced by torch; the tiles arrive already stacked ``(T, Z, Y, X)``;
- ``float32`` replaced by an explicit ``dtype`` (and ``complex64`` by its complex counterpart);
- only the default schedule (``constant``) and optimizers ``adam``, ``adamw`` and ``nadam``; no L1 or smoothness term;
- the Wiener deconvolution and stitching after the fit are left out: the tests compare the coefficients.
"""

import math

import torch
from torch import Tensor


def _noll_to_nm(j: int) -> tuple[int, int]:
    n = 0
    cumulative = 0
    while cumulative + (n + 1) < j:
        n += 1
        cumulative += n
    k = j - cumulative - 1
    if n % 2 == 0:
        ms = [0] + [v for pair in zip(range(2, n + 1, 2), range(-2, -n - 1, -2), strict=False) for v in pair]
    else:
        ms = [v for pair in zip(range(1, n + 1, 2), range(-1, -n - 1, -2), strict=False) for v in pair]
    m = ms[k]
    if j % 2 == 0 and m < 0:
        m = -m
    if j % 2 == 1 and m > 0:
        m = -m
    if j == 1:
        m = 0
    return n, m


def _radial_polynomial(n: int, m: int, rho: Tensor) -> Tensor:
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return torch.zeros_like(rho)
    out = torch.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        coef = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k) * math.factorial((n + m_abs) // 2 - k) * math.factorial((n - m_abs) // 2 - k)
        )
        out = out + coef * rho ** (n - 2 * k)
    return out


def zernike_noll(j: int, rho: Tensor, theta: Tensor) -> Tensor:
    n, m = _noll_to_nm(j)
    r = _radial_polynomial(n, abs(m), rho)
    if m > 0:
        return math.sqrt(2 * (n + 1)) * r * torch.cos(m * theta)
    if m < 0:
        return math.sqrt(2 * (n + 1)) * r * torch.sin(-m * theta)
    return math.sqrt(n + 1) * r


def generate_pupil(frr: Tensor, NA: float, lamb_in: float, steepness: float = 1e4) -> Tensor:
    return torch.sigmoid(steepness * (NA / lamb_in - frr))


def generate_propagation_kernel(
    frr: Tensor, pupil_support: Tensor, wavelength: float, z_position_list: Tensor
) -> Tensor:
    argument = 1 - wavelength**2 * frr**2
    oblique_factor = torch.sqrt(torch.clamp(argument, min=0.0)) / wavelength
    return pupil_support[None, :, :] * torch.exp(
        1j * 2 * math.pi * z_position_list[:, None, None] * oblique_factor[None, :, :]
    )


def _psf_from_coefs(coefs: Tensor, pupil_amp: Tensor, propagation_kernel: Tensor, zernike_basis: Tensor) -> Tensor:
    phi = torch.einsum("tm,myx->tyx", coefs, zernike_basis)
    pupil = pupil_amp[None] * torch.exp(2j * math.pi * phi.to(propagation_kernel.dtype))
    pupil_3d = propagation_kernel[None] * pupil[:, None, :, :]
    coherent = torch.fft.ifft2(pupil_3d, dim=(-2, -1))
    psf = torch.abs(coherent) ** 2
    return psf / psf.sum(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-12)


def recover(
    tiles: Tensor,
    *,
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength: float,
    numerical_aperture: float,
    index: float,
    noll_indices: tuple[int, ...],
    lr_per_mode: dict[int, float],
    n_iter: int,
    optimizer: str = "adam",
    bead_sigma: tuple[float, float, float] = (0.15, 0.1, 0.1),
) -> Tensor:
    """3.x ``recover_zernikes(loss="mse", scale_fit=True)``'s fit: return the ``(T, M)`` coefficients in waves."""
    dtype = tiles.dtype
    T, Z, Y, X = tiles.shape
    # _build_gaussian_bead_template
    sz, sy, sx = bead_sigma
    z = (torch.arange(Z, dtype=dtype) - Z // 2) * z_pixel_size
    y = (torch.arange(Y, dtype=dtype) - Y // 2) * yx_pixel_size
    x = (torch.arange(X, dtype=dtype) - X // 2) * yx_pixel_size
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    bead = torch.exp(-(zz**2 / (2 * sz**2) + yy**2 / (2 * sy**2) + xx**2 / (2 * sx**2)))
    bead_template_fft = torch.fft.fftn(bead[None].expand(T, -1, -1, -1), dim=(-3, -2, -1))
    # _build_pupil_grids
    fy = torch.fft.fftfreq(Y, d=yx_pixel_size, dtype=dtype)
    fx = torch.fft.fftfreq(X, d=yx_pixel_size, dtype=dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    frr = torch.sqrt(fxx**2 + fyy**2)
    theta = torch.atan2(fyy, fxx)
    pupil_amp = generate_pupil(frr, numerical_aperture, wavelength)
    z_position_list = torch.fft.ifftshift((torch.arange(Z, dtype=dtype) - Z // 2) * z_pixel_size)
    propagation_kernel = generate_propagation_kernel(frr, pupil_amp, wavelength / index, z_position_list)
    rho = (frr * wavelength / numerical_aperture).clamp(max=1.0)
    zernike_basis = torch.stack([zernike_noll(j, rho, theta) for j in noll_indices], dim=0)
    # _optimize, loss="mse", scale_fit=True, constant schedule
    coefs_per_mode = [torch.zeros(T, dtype=dtype, requires_grad=True) for _ in noll_indices]
    base_lrs = [lr_per_mode.get(int(j), 0.01) for j in noll_indices]
    param_groups = [{"params": [c], "lr": lr} for c, lr in zip(coefs_per_mode, base_lrs, strict=True)]
    if optimizer == "adamw":
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    elif optimizer == "nadam":
        opt = torch.optim.NAdam(param_groups)
    else:
        opt = torch.optim.Adam(param_groups)
    for _ in range(n_iter):
        opt.zero_grad()
        coefs_local = torch.stack(coefs_per_mode, dim=1)
        psf = _psf_from_coefs(coefs_local, pupil_amp, propagation_kernel, zernike_basis)
        psf_fft = torch.fft.fftn(psf, dim=(-3, -2, -1))
        predicted = torch.real(torch.fft.ifftn(bead_template_fft * psf_fft, dim=(-3, -2, -1)))
        scale = (predicted * tiles).sum(dim=(-3, -2, -1)) / (predicted * predicted).sum(dim=(-3, -2, -1)).clamp(
            min=1e-20
        )
        predicted = predicted * scale[:, None, None, None]
        loss = ((predicted - tiles) ** 2).mean(dim=(-3, -2, -1)).mean()
        loss.backward()
        opt.step()
    return torch.stack(coefs_per_mode, dim=1).detach()
