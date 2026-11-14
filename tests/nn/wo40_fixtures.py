"""Inputs shared with the wo-4.0 (85282ce) reference runs, built with torch alone so both environments build them alike.

The reference numbers in the parity tests came from running wo-4.0's own functions, in its own environment, on these
inputs; each test names the wo-4.0 function. A checksum beside each builder guards against a torch release that draws
other random numbers.
"""

from typing import override

import torch
from torch import Tensor

from waveorder.nn.parameters import Quantity
from waveorder.nn.sampling import Sampling
from waveorder.nn.systems.base import ShiftInvariantSystem


def otsu_fixture() -> Tensor:
    """``(9, 8, 48, 48)`` noise with a block whose contrast grows along the first axis (checksum 64266.7356)."""
    volumes = torch.randn((9, 8, 48, 48), generator=torch.Generator().manual_seed(20240101))
    volumes[:, :, 10:30, 10:30] += torch.linspace(0.5, 4.0, 9).reshape(9, 1, 1, 1)
    return volumes


def noisy_l_curve() -> tuple[list[float], list[float]]:
    """A gently bent, noisy L-curve of 25 samples (sums 138.330563 and 2898.802998)."""
    generator, wide = torch.Generator().manual_seed(7), torch.float64
    residual = torch.logspace(0, 1.2, 25, dtype=wide) * (1 + 0.02 * torch.randn(25, generator=generator, dtype=wide))
    solution = torch.logspace(2.6, 1.1, 25, dtype=wide) * (1 + 0.02 * torch.randn(25, generator=generator, dtype=wide))
    return residual.tolist(), solution.tolist()


def crop_volume() -> Tensor:
    """``(3, 70, 90)`` noise, four times stronger in one region (checksum -236.795003)."""
    volume = torch.randn((3, 70, 90), generator=torch.Generator().manual_seed(1))
    volume[:, 40:60, 50:75] *= 4
    return volume


def gaussian_otf(shape: tuple[int, int, int], *, half: bool = False) -> Tensor:
    """A real Gaussian transfer function over frequencies in cycles per voxel, full or half spectrum."""
    nz, ny, nx = shape
    fz = torch.fft.fftfreq(nz)[:, None, None]
    fy = torch.fft.fftfreq(ny)[None, :, None]
    fx = (torch.fft.rfftfreq(nx) if half else torch.fft.fftfreq(nx))[None, None, :]
    return torch.exp(-(fz**2 / (2 * 0.15**2) + (fy**2 + fx**2) / (2 * 0.12**2)))


def sweep_volume() -> Tensor:
    """``(8, 32, 32)`` float32 blocks blurred by :func:`gaussian_otf`, plus noise and an offset (checksum 458541.70)."""
    shape = (8, 32, 32)
    obj = torch.zeros(shape)
    obj[2:6, 5:12, 6:14] = 1.0
    obj[3:5, 18:26, 15:22] = 2.0
    obj[1:7, 20:24, 4:8] = 0.5
    blurred = torch.fft.ifftn(torch.fft.fftn(obj) * gaussian_otf(shape)).real
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(5))
    return (100 * blurred + 5 * noise + 50).float()


class GaussianTransfer(ShiftInvariantSystem):
    """One field, one channel, transfer ``gain`` times :func:`gaussian_otf` on any grid (pixel size 1)."""

    def __init__(self, gain: Quantity = 1.0) -> None:
        super().__init__(fields=("object",), channel_count=1)
        self.real_transfer = True
        self.register_quantity("gain", gain)

    @override
    def compute_transfer(self, sampling: Sampling) -> Tensor:
        gain = self.quantity("gain")
        return (gain * gaussian_otf(sampling.shape, half=True).to(gain))[None, None]
