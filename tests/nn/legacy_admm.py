"""WaveOrder 3.x ADMM TV deconvolution, as a test-only reference for the 4.x :class:`~waveorder.nn.solvers.ADMM`.

Copied from WaveOrder 3.x ``waveorder/util.py`` (``single_variable_admm_tv_deconvolution_3D`` and
``softTreshold``; BSD-3-Clause, Chan Zuckerberg Biohub), CPU branch. Changes, none of which alters a result: numpy
replaced by torch; the volume is ``(Z, Y, X)`` instead of ``(Y, X, Z)`` (the three difference kernels and rolls follow
their axes); the GPU branch and the progress printing are dropped.
"""

import torch
from torch import Tensor


def soft_threshold(x: Tensor, threshold: float) -> Tensor:
    magnitude = torch.abs(x)
    ratio = torch.clamp(magnitude - threshold, min=0) / (magnitude + 1e-16)
    return x * ratio


def admm_tv_deconvolution_3d(
    stack: Tensor, transfer: Tensor, *, rho: float, reg_re: float, lambda_re: float, itr: int
) -> Tensor:
    """Minimize ``|A x - b|^2 / 2 + reg_re |x|^2 / 2 + lambda_re |D x|_1`` (anisotropic TV); ``transfer`` is full."""
    nz, ny, nx = stack.shape
    differences = torch.zeros((3, nz, ny, nx), dtype=stack.dtype)
    for axis, index in enumerate(((0, 0, -1), (0, -1, 0), (-1, 0, 0))):  # x, y, z as 3.x orders them
        differences[axis, 0, 0, 0] = 1
        differences[(axis, *index)] = -1
    dx, dy, dz = torch.fft.fftn(differences, dim=(1, 2, 3))
    stack_f = torch.fft.fftn(stack)
    rho_term = rho * (torch.conj(dx) * dx + torch.conj(dy) * dy + torch.conj(dz) * dz) + reg_re
    aha = torch.abs(transfer) ** 2 + rho_term
    b_vec = stack_f * torch.conj(transfer)
    z_para = torch.zeros((3, nz, ny, nx), dtype=stack.dtype)
    u_para = torch.zeros((3, nz, ny, nx), dtype=stack.dtype)
    d_vec = torch.zeros((3, nz, ny, nx), dtype=stack.dtype)
    f_real = torch.zeros_like(stack)
    for _ in range(itr):
        v_para = torch.fft.fftn(z_para - u_para, dim=(1, 2, 3))
        b_vec_new = b_vec + rho * (torch.conj(dx) * v_para[0] + torch.conj(dy) * v_para[1] + torch.conj(dz) * v_para[2])
        f_real = torch.real(torch.fft.ifftn(b_vec_new / aha))
        d_vec[0] = f_real - torch.roll(f_real, -1, dims=2)
        d_vec[1] = f_real - torch.roll(f_real, -1, dims=1)
        d_vec[2] = f_real - torch.roll(f_real, -1, dims=0)
        z_para = soft_threshold(d_vec + u_para, lambda_re / rho)
        u_para = u_para + d_vec - z_para
    return f_real
