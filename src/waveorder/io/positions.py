"""Input positions: expanding paths, globs and iohub nodes to positions, and reading their metadata and volumes.

A position is named by its plate-relative name (``A/1/0``), or :data:`SINGLE_POSITION` for a single-position store
that no plate lists. A failed read raises :class:`~waveorder.errors.ReadError` naming the position and the time.
Importing this module imports neither torch nor iohub.
"""

import glob
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from waveorder.errors import ParameterError, Problems, ReadError, Rejected, named

if TYPE_CHECKING:  # iohub takes a second to import, torch longer; the aliases below are evaluated only when inspected
    from iohub.ngff.nodes import NGFFNode, Plate, Position
    from torch import Tensor

type OpenPosition = Position
"""An open iohub ``Position``."""

type OpenNode = Position | Plate
"""An open iohub ``Position`` or ``Plate``."""

type _Opened = NGFFNode
type _Volume = Tensor

_NOT_OME = (ValueError, OSError, KeyError, TypeError)
"""What iohub raises for a directory that is no OME-Zarr node (zarr's group-not-found is a ValueError)."""

_PREFIXES = {
    "yocto": -24, "zepto": -21, "atto": -18, "femto": -15, "pico": -12, "nano": -9, "micro": -6, "milli": -3,
    "centi": -2, "deci": -1, "": 0, "hecto": 2, "kilo": 3, "mega": 6, "giga": 9, "tera": 12, "peta": 15,
    "exa": 18, "zetta": 21, "yotta": 24,
}  # fmt: skip
_MICROMETERS = {f"{prefix}meter": 10.0 ** (exponent + 6) for prefix, exponent in _PREFIXES.items()} | {
    "angstrom": 1e-4,
    "inch": 25_400.0,
    "foot": 304_800.0,
    "yard": 914_400.0,
    "mile": 1_609_344_000.0,
    "parsec": 3.0856775814913673e22,
}
"""The OME-NGFF space units and their size in micrometers."""


def pixel_size(position: OpenPosition, *, image: str = "0") -> tuple[float, float, float]:
    """Return the voxel size of ``image`` along z, y and x in micrometers, from its OME-NGFF scale and axis units.

    Raises
    ------
    ParameterError
        If a spatial axis has a unit that is not an OME-NGFF length unit.
    """
    sizes = []
    for axis, value in zip(position.axes[-3:], position.get_effective_scale(image)[-3:], strict=True):
        unit = getattr(axis, "unit", None) or "micrometer"
        if unit not in _MICROMETERS:
            raise ParameterError(
                f"pixel_size: axis {axis.name} is in {unit!r}, which is not an OME-NGFF length unit",
                path=f"position.axes.{axis.name}",
            )
        sizes.append(float(value) * _MICROMETERS[unit])
    z, y, x = sizes
    return (z, y, x)


SINGLE_POSITION = "."
"""The name of the position of a single-position store, an OME-Zarr position that no HCS plate lists.

Its output is a single-position store too, whose root is the position: ``Path(output) / SINGLE_POSITION`` is
``output`` itself, so the joins that place a plate position in its output plate place this one at the root.
"""


def store_path(node: OpenNode) -> Path:
    """Return the directory of an iohub node: its store's root joined with the node's group path."""
    return Path(node.zgroup.store.root, node.zgroup.path).resolve()


def _located(node: OpenNode) -> Path:
    """Return the directory of an iohub node as reached, symlinks kept (a position linked into a plate stays there)."""
    return Path(os.path.abspath(Path(node.zgroup.store.root, node.zgroup.path)))


def listed_inputs(inputs: object) -> list[object]:
    """Return the ``positions`` argument as a list: one path, pattern or iohub node, or the ones a sequence lists."""
    from iohub.ngff import Plate, Position

    return [inputs] if isinstance(inputs, str | Path | Position | Plate) else list(inputs)  # ty: ignore[invalid-argument-type]


@contextmanager
def open_positions(inputs: object, *, problems: Problems | None = None) -> Generator[list[OpenPosition]]:
    """Open the ``positions`` argument as iohub positions, read-only, each once, in the order given.

    Each input is an open iohub ``Position`` or ``Plate`` (its positions), or a path or glob whose matching
    directories are opened; a plate stands for its positions. Existing files are skipped (a shell-expanded
    ``in.zarr/*/*/*`` also passes ``zarr.json``). Matches are naturally sorted (``A/2`` before ``A/10``). Nodes opened
    here are closed when the block ends; nodes given stay open.

    Raises
    ------
    Rejected
        At ``positions <pattern>`` if a pattern matches nothing, or only directories that are neither positions nor
        plates; at ``positions <file>`` if every argument is a file. With ``problems``, refusals are collected there.
    """
    from iohub.ngff import Plate, Position
    from natsort import natsorted

    given = listed_inputs(inputs)
    found: dict[Path, Position] = {}
    files: list[str] = []
    opened: list[NGFFNode] = []
    try:
        for item in given:
            nodes: list[Position | Plate] = []
            if isinstance(item, Position | Plate):
                nodes = [item]
            elif is_file_argument(item):  # ty: ignore[invalid-argument-type]
                files.append(str(item))
            else:
                with nullcontext() if problems is None else problems.collect():
                    nodes = _open_matches(str(item), opened)
            for node in nodes:
                members = [p for _, p in node.positions()] if isinstance(node, Plate) else [node]
                if isinstance(node, Plate) and any(node is mine for mine in opened):  # a plate path: natural order
                    members = natsorted(members, key=lambda member: member.zgroup.path)
                for position in members:
                    found.setdefault(store_path(position), position)
        if files and len(files) == len(given):  # a file is refused only when nothing else is named
            refusal = Rejected.one(
                f"{files[0]} is a file, not an OME-Zarr position or plate", location=f"{named('positions')} {files[0]}"
            )
            if problems is None:
                raise refusal
            problems.take(refusal)
        yield list(found.values())
    finally:
        for node in opened:
            node.close()


def expand_positions(patterns: object) -> list[Path]:
    """Return the store paths of the positions ``patterns`` names, as :func:`open_positions` opens them."""
    with open_positions(patterns) as positions:
        return [store_path(position) for position in positions]


def is_file_argument(pattern: str | Path) -> bool:
    """Whether a pattern is an existing file, which a shell-expanded glob passes and expansion skips."""
    path = Path(pattern)
    return path.exists() and not path.is_dir()


def _open_matches(pattern: str, opened: list[_Opened]) -> list[OpenNode]:
    """Open each directory ``pattern`` matches with iohub; keep the plates and positions (``opened`` takes all)."""
    from iohub.ngff import Plate, Position, open_ome_zarr
    from natsort import natsorted

    matches = [Path(match) for match in natsorted(glob.glob(pattern)) if Path(match).is_dir()]
    if not matches:
        raise Rejected.one(f"{pattern} matches nothing on disk", location=f"{named('positions')} {pattern}")
    nodes = []
    for match in matches:
        try:
            node = open_ome_zarr(match, mode="r")
        except _NOT_OME:
            continue
        opened.append(node)
        if isinstance(node, Plate | Position):
            nodes.append(node)
    if not nodes:
        what = (
            "is not an OME-Zarr position or plate"
            if matches == [Path(pattern)]
            else ("matches only directories that are not OME-Zarr positions or plates")
        )
        raise Rejected.one(f"{pattern} {what}", location=f"{named('positions')} {pattern}")
    return nodes


def plate_of(path: str | Path) -> Path | None:
    """Return the HCS plate that holds the position at ``path``, or ``None`` for a single-position store.

    The position is in a plate when the folder three levels up is an HCS plate listing its well and field. Symlinks
    are kept, so a store linked as ``plate.zarr/A/1/0`` is that position of the plate.
    """
    from iohub.ngff import Plate, open_ome_zarr

    path = Path(os.path.abspath(path))
    if len(path.parents) < 3:
        return None
    root = path.parents[2]
    row, column, field = path.parts[-3:]
    try:
        node = open_ome_zarr(root, mode="r")
    except _NOT_OME:
        return None
    with node:
        metadata = getattr(node, "metadata", None)  # iohub keeps none for a plate whose metadata it cannot read
        if not isinstance(node, Plate) or metadata is None:
            return None
        well = f"{row}/{column}"
        if well not in {listed.path for listed in metadata.wells}:
            return None
        try:
            images = node[well].metadata.images
        except (*_NOT_OME, AttributeError):
            return None
    return root if field in {image.path for image in images} else None


def plate_name(path: str | Path) -> str:
    """Return the name of the position at ``path``: ``row/column/field``, or :data:`SINGLE_POSITION` if no plate."""
    path = Path(os.path.abspath(path))
    return SINGLE_POSITION if plate_of(path) is None else "/".join(path.parts[-3:])


@dataclass(frozen=True, kw_only=True)
class PositionMeta:
    """An input position's metadata, read before any pixel.

    Attributes
    ----------
    path : Path
        The position's directory, symlinks resolved.
    name : str
        Plate-relative name (``A/1/0``); :data:`SINGLE_POSITION` for a single-position store.
    shape : tuple[int, int, int, int, int]
        ``(T, C, Z, Y, X)`` of image ``0``.
    dtype : str
        Its dtype, e.g. ``float32``.
    channel_names : tuple[str, ...]
        The position's channels.
    scale : tuple[float, ...]
        Image ``0``'s effective ``(T, C, Z, Y, X)`` scale, in the axes' own units.
    pixel_size : tuple[float, float, float]
        Voxel size along z, y, x in um.
    chunks : tuple[int, ...]
        Chunk shape of image ``0``.
    shards : tuple[int, ...], optional
        Its shard shape; ``None`` without shards (every v0.4 array).
    version : str
        OME-Zarr version, ``0.4`` or ``0.5``.
    plate_root : Path, optional
        The HCS plate that lists the position, symlinks kept; ``None`` for a single-position store.
    """

    path: Path
    name: str
    shape: tuple[int, int, int, int, int]
    dtype: str
    channel_names: tuple[str, ...]
    scale: tuple[float, ...]
    pixel_size: tuple[float, float, float]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    version: str
    plate_root: Path | None

    @property
    def zyx(self) -> tuple[int, int, int]:
        """The volume's ``(Z, Y, X)``."""
        _, _, z, y, x = self.shape
        return (z, y, x)

    @property
    def times(self) -> int:
        """The number of time points, ``T``."""
        return self.shape[0]

    @property
    def store_root(self) -> Path:
        """The store's root: the plate, or the position itself when no plate lists it."""
        return self.path if self.plate_root is None else self.plate_root


def read_position_meta(position: OpenPosition | str | Path) -> PositionMeta:
    """Read a position's metadata: name, shape, dtype, channels, scale and array layout.

    Parameters
    ----------
    position : iohub Position, str or Path
        An open position, or a path opened read-only for the read.

    Returns
    -------
    PositionMeta

    Raises
    ------
    Rejected
        If it is not an OME-Zarr position, image ``0`` is missing or not ``(T, C, Z, Y, X)``, or a spatial axis has a
        unit that is not a length unit.
    """
    from iohub.ngff import Position, open_ome_zarr

    if not isinstance(position, Position):
        given = Path(position)
        try:
            node = open_ome_zarr(given, mode="r")
        except _NOT_OME as error:
            raise Rejected.one(f"{given} is not an OME-Zarr position ({error})", location=str(given)) from None
        with node:
            if not isinstance(node, Position):
                raise Rejected.one(f"{given} is not an OME-Zarr position", location=str(given))
            return read_position_meta(node)
    path = store_path(position)
    located = _located(position)
    plate = plate_of(located)
    if "0" not in position.array_keys():
        raise Rejected.one(f"{path} has no image 0", location=str(path))
    array = position["0"]
    shape = tuple(int(n) for n in array.shape)
    if len(shape) != 5:
        raise Rejected.one(
            f"{path} image 0 has {len(shape)} axes ({'x'.join(map(str, shape))}), expected 5 (T, C, Z, Y, X)",
            location=str(path),
        )
    try:
        sizes = pixel_size(position)
    except ParameterError as error:
        raise Rejected.one(f"{path}: {error}", location=str(path)) from None
    t, c, z, y, x = shape
    shards = array.shards
    return PositionMeta(
        path=path,
        name=SINGLE_POSITION if plate is None else "/".join(located.parts[-3:]),
        shape=(t, c, z, y, x),
        dtype=str(array.dtype),
        channel_names=tuple(position.channel_names),
        scale=tuple(float(value) for value in position.get_effective_scale("0")),
        pixel_size=sizes,
        chunks=tuple(int(n) for n in array.chunks),
        shards=None if shards is None else tuple(int(n) for n in shards),
        version=str(position.version),
        plate_root=plate,
    )


def channel_indices(meta: PositionMeta, channels: Sequence[str]) -> tuple[int, ...]:
    """Return the indices of the channels named ``channels`` in a position, in the order given.

    Raises
    ------
    Rejected
        Naming the channels the position lacks and listing the ones it has.
    """
    missing = [name for name in channels if name not in meta.channel_names]
    if missing:
        raise Rejected.one(
            f"the input channel names list {', '.join(missing)}, which {meta.path} does not have; it has "
            f"{', '.join(meta.channel_names)}",
            location=str(meta.path),
        )
    return tuple(meta.channel_names.index(name) for name in channels)


@contextmanager
def reading(path: str | Path, *, time: int) -> Generator[None]:
    """Turn any error of the block, except ``MemoryError``, into a ``ReadError`` naming ``path`` and ``time``."""
    try:
        yield
    except (ReadError, MemoryError):
        raise
    except Exception as error:
        raise ReadError(
            f"could not read time {time} of {path}: {type(error).__name__}: {error}; the store is damaged or "
            "incomplete there: restore it, or use another time point or position"
        ) from error


def read_volume(position: OpenPosition | str | Path, *, time: int, channels: Sequence[int]) -> _Volume:
    """Read one time point of a position's image ``0``.

    Parameters
    ----------
    position : iohub Position, str or Path
        An open position, or a path opened read-only (as a worker process has).
    time : int
        Index along the time axis.
    channels : Sequence[int]
        Indices along the channel axis (:func:`channel_indices`), in the order to return them.

    Returns
    -------
    Tensor
        ``(C, Z, Y, X)`` in the stored dtype, on the CPU.

    Raises
    ------
    ReadError
        If the position cannot be opened or the volume read, naming the position, the time and the cause.
    """
    import torch
    from iohub.ngff import open_ome_zarr

    if isinstance(position, str | Path):
        with reading(position, time=time), open_ome_zarr(position, mode="r") as opened:
            return read_volume(opened, time=time, channels=channels)
    with reading(store_path(position), time=time):
        array = position["0"]
        return torch.stack([torch.as_tensor(array[time, channel]) for channel in channels])
