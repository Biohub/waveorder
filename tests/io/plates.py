"""Tiny HCS plates for the tests of ``waveorder.io``."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import iohub
import torch

PIXEL_SIZE = (0.4, 0.15, 0.15)


def write_plate(
    path: Path,
    volumes: Sequence[torch.Tensor],
    *,
    channels: Sequence[str] = ("BF",),
    pixel_size: tuple[float, float, float] = PIXEL_SIZE,
    version: Literal["0.4", "0.5"] = "0.5",
) -> list[Path]:
    """Write one position per ``(T, C, Z, Y, X)`` tensor (``A/1/0``, ``A/1/1``, ...); return their paths."""
    paths = []
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=list(channels), version=version) as plate:
        plate.zattrs["acquisition"] = {"microscope": "test"}
        for index, volume in enumerate(volumes):
            position = plate.create_position("A", "1", str(index))
            position.create_image("0", volume.numpy())
            for axis, size in zip("zyx", pixel_size, strict=True):
                position.set_scale("0", axis, size)
            position.zattrs["biahub-deskew"] = {"angle": 30}
            paths.append(path / "A" / "1" / str(index))
    return paths
