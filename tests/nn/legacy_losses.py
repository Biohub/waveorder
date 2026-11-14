"""WaveOrder 3.x sharpness losses, as a test-only reference for the 4.x scores.

Copied from WaveOrder 3.x at tag ``3.0.5`` (``waveorder/optim/losses.py``, ``compute_midband_power`` from
``waveorder/focus.py`` and ``gen_coordinate`` from ``waveorder/util.py``; BSD-3-Clause, Chan Zuckerberg Biohub).
Changes from the originals, none of which alters a result in float32:

- numpy calls replaced by torch (``np.sqrt`` on the frequency grids);
- the ``.float()`` casts before the FFTs removed, and the frequency grids built in the input's dtype, so the reference
  also runs in float64 (3.x computed these two losses in float32 whatever the input);
- the pydantic settings classes replaced by the plain arguments of ``build_loss_fn``.
"""

import torch
from torch import Tensor


def gen_coordinate(img_dim: tuple[int, int], ps: float, *, dtype: torch.dtype, device=None):
    N, M = img_dim
    ps = float(ps)
    fx = torch.fft.fftfreq(M, ps, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(N, ps, device=device, dtype=dtype)
    x = torch.fft.ifftshift((torch.arange(M, dtype=dtype, device=device) - M / 2) * ps)
    y = torch.fft.ifftshift((torch.arange(N, dtype=dtype, device=device) - N / 2) * ps)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")
    return (xx, yy, fxx, fyy)


def compute_midband_power(input_tensor, NA_det, lambda_ill, pixel_size, midband_fractions=(0.125, 0.25)):
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    _, Y, X = input_tensor.shape
    _, _, fxx, fyy = gen_coordinate((Y, X), pixel_size, dtype=input_tensor.dtype, device=input_tensor.device)
    frr = torch.sqrt(fxx**2 + fyy**2)
    cutoff = 2 * NA_det / lambda_ill
    mask = torch.logical_and(frr > cutoff * midband_fractions[0], frr < cutoff * midband_fractions[1])
    abs_fft = torch.abs(torch.fft.fftn(input_tensor, dim=(-2, -1)))
    result = abs_fft[:, mask].sum(dim=-1)
    if squeeze:
        return result.squeeze(0)
    return result


def midband_power_loss(NA_det, wavelength, pixel_size, midband_fractions):
    def loss_fn(recon: Tensor) -> Tensor:
        power = compute_midband_power(
            recon, NA_det=NA_det, lambda_ill=wavelength, pixel_size=pixel_size, midband_fractions=midband_fractions
        )
        Y, X = recon.shape[-2:]
        _, _, fxx, fyy = gen_coordinate((Y, X), pixel_size, dtype=recon.dtype)
        frr = torch.sqrt(fxx**2 + fyy**2)
        cutoff = 2 * NA_det / wavelength
        n_masked = int(((frr > cutoff * midband_fractions[0]) & (frr < cutoff * midband_fractions[1])).sum())
        loss = -power / max(n_masked, 1)
        if loss.ndim > 0:
            loss = loss.mean()
        return loss

    return loss_fn


def total_variation_loss():
    def _tv_2d(img: Tensor) -> Tensor:
        dy = img[1:, :] - img[:-1, :]
        dx = img[:, 1:] - img[:, :-1]
        return (dy.abs().mean() + dx.abs().mean()) / 2

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_tv_2d(recon)
        tv = torch.stack([_tv_2d(recon[z]) for z in range(recon.shape[0])])
        return -tv.mean()

    return loss_fn


def laplacian_variance_loss():
    def _lap_var_2d(img: Tensor) -> Tensor:
        kernel = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=img.dtype, device=img.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        padded = img.unsqueeze(0).unsqueeze(0)
        lap = torch.nn.functional.conv2d(padded, kernel, padding=1)
        return lap.var()

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_lap_var_2d(recon)
        vals = torch.stack([_lap_var_2d(recon[z]) for z in range(recon.shape[0])])
        return -vals.mean()

    return loss_fn


def normalized_variance_loss():
    def _nvar_2d(img: Tensor) -> Tensor:
        mean = img.mean()
        var = img.var()
        return var / (mean**2 + 1e-12)

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_nvar_2d(recon)
        vals = torch.stack([_nvar_2d(recon[z]) for z in range(recon.shape[0])])
        return -vals.mean()

    return loss_fn


def spectral_flatness_loss(NA_det, wavelength, pixel_size, midband_fractions):
    def _flatness_2d(img: Tensor) -> Tensor:
        Y, X = img.shape
        _, _, fxx, fyy = gen_coordinate((Y, X), pixel_size, dtype=img.dtype, device=img.device)
        frr = torch.sqrt(fxx**2 + fyy**2)
        cutoff = 2 * NA_det / wavelength
        mask = torch.logical_and(frr > cutoff * midband_fractions[0], frr < cutoff * midband_fractions[1])
        power = torch.abs(torch.fft.fftn(img, dim=(-2, -1)))
        midband_power = power[mask]
        log_mean = midband_power.log().mean()
        arith_mean = midband_power.mean()
        return log_mean.exp() / (arith_mean + 1e-12)

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_flatness_2d(recon)
        vals = torch.stack([_flatness_2d(recon[z]) for z in range(recon.shape[0])])
        return -vals.mean()

    return loss_fn
