"""Tiny HCS plates for the data and training tests: a raw plate and its reconstruction, with the same positions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import iohub
import torch

from waveorder.models import Phase2D
from waveorder.nn import Detection, Gaussian, Illumination, Sampling, Tikhonov

PIXEL_SIZE = (2.0, 0.65, 0.65)
"""Voxel size of the ``plates`` fixture along z, y, x in um."""

CHANNELS = ("GFP", "BF")
TARGET_CHANNELS = ("Phase2D", "Focus3D")
POSITIONS = tuple(f"A/{column}/{fov}" for column in (1, 2) for fov in range(4))
"""Two wells of four positions each."""

PHASE_PIXEL_SIZE = (0.4, 0.15, 0.15)
"""Voxel size of the ``phase_plates`` fixture, which a tiny ``Phase2D`` reconstructs."""


@dataclass(frozen=True)
class Plates:
    """A raw plate and the plate of its targets, with the same position names."""

    inputs: Path
    targets: Path
    positions: tuple[str, ...]
    channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    pixel_size: tuple[float, float, float]


def write_plate(
    path: Path, volumes: Mapping[str, torch.Tensor], *, channels: Sequence[str], pixel_size: Sequence[float]
) -> Path:
    """Write an HCS plate of one position per entry of ``volumes`` (``row/column/fov`` to ``(T, C, Z, Y, X)``).

    Each channel of each time point is one chunk, as in the plates the loader is made for.
    """
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=list(channels)) as plate:
        for name, volume in volumes.items():
            row, column, fov = name.split("/")
            position = plate.create_position(row, column, fov)
            position.create_image("0", volume.numpy(), chunks=(1, 1, *volume.shape[2:]))
            for axis, size in zip("zyx", pixel_size, strict=True):
                position.set_scale("0", axis, size)
    return path


def raw_field(name: str, shape: tuple[int, int, int, int] = (2, 5, 64, 64)) -> torch.Tensor:
    """Return a recorded field ``(1, C, Z, Y, X)`` uint16 whose values depend on the position's name only."""
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    return torch.randint(100, 4000, (1, *shape), generator=generator, dtype=torch.int32).to(torch.uint16)


def target_field(name: str, shape: tuple[int, int, int, int] = (2, 1, 64, 64)) -> torch.Tensor:
    """Return a target field ``(1, F, 1, Y, X)`` float32 whose values depend on the position's name only."""
    generator = torch.Generator().manual_seed(1000 + sum(map(ord, name)))
    return torch.randn((1, *shape), generator=generator)


def recorded_plates(directory: Path, positions: Sequence[str] = POSITIONS) -> Plates:
    """Write the ``plates`` fixture's pair under ``directory``."""
    inputs = write_plate(
        directory / "raw.zarr",
        {name: raw_field(name) for name in positions},
        channels=CHANNELS,
        pixel_size=PIXEL_SIZE,
    )
    targets = write_plate(
        directory / "reconstruction.zarr",
        {name: target_field(name) for name in positions},
        channels=TARGET_CHANNELS,
        pixel_size=(1.0, *PIXEL_SIZE[1:]),
    )
    return Plates(
        inputs=inputs,
        targets=targets,
        positions=tuple(positions),
        channels=CHANNELS,
        target_channels=TARGET_CHANNELS,
        pixel_size=PIXEL_SIZE,
    )


def thin_phase_model() -> Phase2D:
    """The tiny model the ``phase_plates`` reconstruction is made with: thin phase and absorption from brightfield."""
    return Phase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.52),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(weight=1e-3),
        noise=Gaussian(sigma=1e-3),
    )


def phase_plates(
    directory: Path, positions: Sequence[str] = POSITIONS, *, shape: tuple[int, int, int] = (5, 64, 64)
) -> Plates:
    """Write brightfield stacks of random phase objects and their ``Phase2D`` reconstruction (phase only).

    The raw plate has one channel ``BF`` of uint16 intensities; the target plate one channel ``Phase2D``, the
    model's phase estimate in float32.
    """
    model = thin_phase_model()
    sampling = Sampling(shape=shape, pixel_size=PHASE_PIXEL_SIZE)
    raw: dict[str, torch.Tensor] = {}
    reconstructed: dict[str, torch.Tensor] = {}
    for number, name in enumerate(positions):
        generator = torch.Generator().manual_seed(number)
        obj = torch.zeros(model.system.object_shape(sampling))
        noise = torch.randn(1, 1, *shape[1:], generator=generator)
        obj[0] = 0.2 * torch.nn.functional.avg_pool2d(noise, 5, stride=1, padding=2)[0]
        with torch.no_grad():
            contrast = model.imaging_model(obj, sampling)
        intensity = (1000 * (1 + contrast) + torch.randn(contrast.shape, generator=generator)).clamp(0, 65535)
        raw[name] = intensity.round().to(torch.int32).to(torch.uint16)[None]
        estimate = model.reconstruct(intensity, pixel_size=PHASE_PIXEL_SIZE).estimate
        reconstructed[name] = estimate[:1].to(torch.float32)[None]
    inputs = write_plate(directory / "bf.zarr", raw, channels=["BF"], pixel_size=PHASE_PIXEL_SIZE)
    targets = write_plate(directory / "phase.zarr", reconstructed, channels=["Phase2D"], pixel_size=PHASE_PIXEL_SIZE)
    return Plates(
        inputs=inputs,
        targets=targets,
        positions=tuple(positions),
        channels=("BF",),
        target_channels=("Phase2D",),
        pixel_size=PHASE_PIXEL_SIZE,
    )
