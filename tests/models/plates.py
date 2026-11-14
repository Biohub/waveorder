"""Tiny HCS plates and the models that reconstruct them, for the tests of the position verbs."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Unpack

import iohub
import torch

from waveorder.models import Fluorescence3D, Phase2D, Phase3D
from waveorder.models._base import Parts
from waveorder.nn import Detection, Gaussian, Illumination, Sampling, Tikhonov

PIXEL_SIZE = (0.4, 0.15, 0.15)
SHAPE = (8, 32, 32)


def phase_model(**parts: Unpack[Parts]) -> Phase3D:
    """Thick phase with nothing free: ``Phase3D``."""
    settings: Parts = {"object_prior": Tikhonov(weight=4e-4), "noise": Gaussian(sigma=1e-3), **parts}
    return Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=0.52),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        **settings,
    )


def thin_phase_model(**parts: Unpack[Parts]) -> Phase2D:
    """Thin phase: one output plane with ``Phase2D`` and ``Absorption2D``."""
    settings: Parts = {"object_prior": Tikhonov(weight=1e-3), "noise": Gaussian(sigma=1e-3), **parts}
    return Phase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.52),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        **settings,
    )


def fluorescence_model(detection: Detection | None = None, **parts: Unpack[Parts]) -> Fluorescence3D:
    """Thick fluorescence of one channel: ``<channel>_Density3D``."""
    settings: Parts = {"object_prior": Tikhonov(weight=1e-3), "noise": Gaussian(sigma=0.01), **parts}
    detection = detection or Detection(wavelength=0.52, numerical_aperture=0.8)
    return Fluorescence3D(detection, index_of_refraction_media=1.33, **settings)


def contrast(shape: tuple[int, int, int] = SHAPE, *, seed: int = 0) -> torch.Tensor:
    """Return ``(Z, Y, X)`` brightfield contrast of a random phase object, as the thick phase system measures it."""
    model = phase_model()
    generator = torch.Generator().manual_seed(seed)
    obj = 0.05 * torch.randn(1, *shape, generator=generator)
    with torch.no_grad():
        data = model.imaging_model(obj, Sampling(shape=shape, pixel_size=PIXEL_SIZE))
    return (data + 1e-3 * torch.randn(data.shape, generator=generator))[0]


def series(volume: torch.Tensor, times: int) -> torch.Tensor:
    """Return ``times`` brightfield intensity stacks ``1000 * (1 + contrast)``, slightly different (``T, Z, Y, X``)."""
    return torch.stack([1000 * (1 + volume * (1 + 0.1 * t)) for t in range(times)])


def write_plate(
    path: Path,
    positions: Sequence[torch.Tensor],
    *,
    channel: str = "BF",
    pixel_size: tuple[float, float, float] = PIXEL_SIZE,
    version: Literal["0.4", "0.5"] = "0.5",
) -> list[Path]:
    """Write one position per ``(T, Z, Y, X)`` tensor (``A/1/0``, ``A/1/1``, ...) as a ``version`` plate."""
    paths = []
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=[channel], version=version) as plate:
        plate.zattrs["acquisition"] = {"microscope": "test"}
        for index, volume in enumerate(positions):
            position = plate.create_position("A", "1", str(index))
            position.create_image("0", volume[:, None].to(torch.float32).numpy())
            for axis, size in zip("zyx", pixel_size, strict=True):
                position.set_scale("0", axis, size)
            position.zattrs["biahub-deskew"] = {"angle": 30}
            paths.append(path / "A" / "1" / str(index))
    return paths


def read_output(path: Path, position: str = "A/1/0") -> tuple[list[str], torch.Tensor, dict]:
    """Return an output position's channel names, its image ``0`` and its attributes."""
    with iohub.open_ome_zarr(path / position, mode="r") as node:
        return list(node.channel_names), torch.as_tensor(node["0"][:]), dict(node.zattrs)
