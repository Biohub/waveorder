"""Where a field and its target are read from: :class:`Recorded` plates, or a :class:`PairSource` of your own.

Targets are reconstructions recorded beforehand (``wo rec``). A source runs in a loader worker on the CPU.
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from waveorder.data.index import FovRow
from waveorder.errors import ReadError
from waveorder.io.positions import reading

if TYPE_CHECKING:
    from iohub.ngff.nodes import NGFFNode

type _Node = NGFFNode
"""An open iohub node; the alias is evaluated lazily, so importing this module does not import iohub."""


class PairSource(ABC):
    """Where the data and the target of each field come from."""

    @abstractmethod
    def read(self, row: FovRow, *, channels: Sequence[str], target_channels: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Read one field on the CPU: data ``(C, Z, Y, X)`` and target ``(F, Zout, Y, X)``, channels in order given."""


class Recorded(PairSource):
    """Data and its recorded reconstruction, the target, from the row's plates; each worker opens a plate once."""

    @override
    def read(self, row: FovRow, *, channels: Sequence[str], target_channels: Sequence[str]) -> tuple[Tensor, Tensor]:
        data = read_field(row.input, row.position, time=row.time, channels=channels)
        return data, read_field(row.target, row.position, time=row.time, channels=target_channels)


def read_field(plate: str | Path, position: str, *, time: int, channels: Sequence[str]) -> Tensor:
    """Read one time point of the named channels of a position as ``(C, Z, Y, X)`` in the stored dtype.

    The plate is opened read-only once per process.

    Raises
    ------
    ReadError
        If the plate or position cannot be read or lacks one of ``channels``.
    """
    path = Path(plate) / position
    with reading(path, time=time):
        node = _plate(str(plate))[position]
        names = list(node.channel_names)
        array = node["0"]
    missing = [name for name in channels if name not in names]
    if missing:
        raise ReadError(
            f"could not read time {time} of {path}: it has no channel {', '.join(missing)}; it has "
            f"{', '.join(names)}. Build the index again"
        )
    with reading(path, time=time):
        volumes = [torch.as_tensor(array[time, names.index(name)]) for name in channels]
    return volumes[0].unsqueeze(0) if len(volumes) == 1 else torch.stack(volumes)


@functools.cache
def _plate(path: str) -> _Node:
    """Open a plate read-only, once per process."""
    from iohub.ngff import open_ome_zarr

    return open_ome_zarr(path, mode="r")
