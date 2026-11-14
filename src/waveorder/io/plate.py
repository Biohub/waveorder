"""The output plate: provenance key, array geometry, creating and filling it, and the host-memory estimate.

The plate is created once with every position, channel and time point (iohub's ``create_empty_plate``), then filled
one (position, time) at a time; iohub records each finished time point in ``.iohub-progress`` beside the store, and a
resumed run skips those. One single-position input store gives a single-position output store instead. The
provenance key follows 3.x (``waveorder/cli/settings.py``). Importing this module imports neither torch nor iohub.
"""

import contextlib
import io
import json
import logging
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path

from waveorder.errors import Problem, ReadError, Rejected, named
from waveorder.io.positions import SINGLE_POSITION

type ZYX = tuple[int, int, int]
type TCZYX = tuple[int, int, int, int, int]

log = logging.getLogger(__name__)

DEFAULT_VERSION = "0.5"
"""The version of a new plate when none is asked for."""

_BYTES_PER_VOXEL = 4
"""float32."""
_FOURIER_MULTIPLIER = 32
"""Working copies of a volume a closed-form phase or fluorescence solve holds (3.x ``estimate_resources``)."""

_CHUNK_ZY_CAP = (16, 256)
_CHUNK_X_CAP = 256

_PLATE_KEYS = ("ome", "plate", "well")
"""Root attributes the OME-Zarr spec owns, which are not copied from the input plate."""


def provenance_key(channel_names: Sequence[str]) -> str:
    """Return the attribute key a reconstruction writing ``channel_names`` records its provenance under.

    ``waveorder-`` and the phase and density names joined by ``,`` (3.x): ``waveorder-Phase3D``, ``waveorder-Phase2D``
    (thin phase, with ``Absorption2D``), ``waveorder-GFP_Density3D``. With no such name, ``waveorder-`` and every
    channel name joined by ``-``: ``waveorder-GFP_Deblurred3D``.
    """
    parts = [
        name for name in channel_names if name in ("Phase2D", "Phase3D") or name.endswith(("_Density2D", "_Density3D"))
    ]
    if not parts:
        return "waveorder-" + "-".join(channel_names)
    return "waveorder-" + ",".join(parts)


def software_versions() -> dict[str, str]:
    """Return the installed versions of waveorder, iohub and torch, for a provenance record."""
    return {name: package_version(name) for name in ("waveorder", "iohub", "torch")}


@dataclass(frozen=True, kw_only=True)
class Geometry:
    """Chunks and shards of an output array, each ``(T, C, Z, Y, X)`` (``None``: iohub's default for the version)."""

    chunks: TCZYX | None = None
    shards: TCZYX | None = None


def _divisor(n: int, cap: int) -> int:
    """Return the largest divisor of ``n`` not above ``cap``."""
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def choose_geometry(core: ZYX | None, *, x: int) -> Geometry:
    """Return the array geometry of a new output position.

    Parameters
    ----------
    core : (Z, Y, X), optional
        The tile core the tile merger writes; ``None`` for a reconstruction that writes whole volumes.
    x : int
        The output's extent along x.

    Returns
    -------
    Geometry
        iohub's default without ``core``; else shards one tile row high (``core``'s z and y, all of x, one time point
        and channel), so the merger's writer threads never share a shard.
    """
    if core is None:
        return Geometry()
    cz, cy, _ = core
    chunk_x = min(x, _CHUNK_X_CAP)
    chunks = (1, 1, _divisor(cz, _CHUNK_ZY_CAP[0]), _divisor(cy, _CHUNK_ZY_CAP[1]), chunk_x)
    shards = (1, 1, cz, cy, math.ceil(x / chunk_x) * chunk_x)
    return Geometry(chunks=chunks, shards=shards)


@dataclass(frozen=True, kw_only=True)
class OutputPosition:
    """One position of the output plate, as :func:`create_plate` creates it.

    Attributes
    ----------
    name : str
        ``row/column/fov`` (``A/1/0``), or :data:`~waveorder.io.positions.SINGLE_POSITION` for a single-position store.
    shape : tuple[int, int, int, int, int]
        ``(T, C, Z, Y, X)`` of the output, ``C`` the number of output channels.
    scale : tuple[float, float, float, float, float]
        ``(T, C, Z, Y, X)`` scale to record.
    geometry : Geometry
        Chunks and shards of a new position's array (:func:`choose_geometry`).
    """

    name: str
    shape: TCZYX
    scale: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    geometry: Geometry = Geometry()


@dataclass(frozen=True, kw_only=True)
class ExistingPlate:
    """An existing output: its OME-Zarr ``version`` and the ``(T, C, Z, Y, X)`` shape of each position by name."""

    version: str
    shapes: dict[str, tuple[int, ...]]


def is_single(names: Sequence[str]) -> bool:
    """Whether the output positions ``names`` make a single-position store."""
    return list(names) == [SINGLE_POSITION]


def read_plate(output: str | Path, *, single: bool = False) -> ExistingPlate | None:
    """Read the version of an existing output and the shape of each of its positions.

    Parameters
    ----------
    output : str or Path
        The output store.
    single : bool
        Whether the run writes a single-position store (:func:`is_single`) rather than an HCS plate.

    Returns
    -------
    ExistingPlate or None
        ``None`` if ``output`` does not exist.

    Raises
    ------
    Rejected
        If ``output`` exists and is not the kind of store the run writes, or cannot be read.
    """
    from iohub.ngff import Plate, Position, open_ome_zarr

    output = Path(output)
    if not output.exists():
        return None
    kind = "a single-position OME-Zarr store" if single else "an OME-Zarr plate"
    try:
        node = open_ome_zarr(output, mode="r")
    except Exception as error:  # noqa: BLE001 (whatever iohub or zarr raise for something that is no plate)
        raise Rejected.one(
            f"exists and is not {kind} ({error}); use another {named('output')}", location=str(output)
        ) from None
    with node:
        if single and not isinstance(node, Position):
            raise Rejected.one(
                f"exists and is not a single-position store, which a reconstruction of one single-position store "
                f"writes; use another {named('output')}",
                location=str(output),
            )
        if single:
            shape = tuple(int(n) for n in node["0"].shape)
            return ExistingPlate(version=str(node.version), shapes={SINGLE_POSITION: shape})
        if not isinstance(node, Plate):
            raise Rejected.one(f"exists and is not an HCS plate; use another {named('output')}", location=str(output))
        shapes = {name: tuple(int(n) for n in position["0"].shape) for name, position in node.positions()}
        return ExistingPlate(version=str(node.version), shapes=shapes)


def name_problems(names: Sequence[str]) -> list[Problem]:
    """Return the problems of position names an output cannot hold.

    A plate names a position ``row/column/field``, each letters and digits only. A single-position store
    (:data:`~waveorder.io.positions.SINGLE_POSITION`) is refused among other positions.

    Parameters
    ----------
    names : Sequence[str]
        Plate-relative position names.

    Returns
    -------
    list[Problem]
        Problems at ``positions``; empty if every name is accepted.
    """
    problems = []
    singles = sum(name == SINGLE_POSITION for name in names)
    if singles and len(names) > 1:
        which = (
            f"all {singles} are single-position stores"
            if singles == len(names)
            else ("one is a single-position store" if singles == 1 else f"{singles} are single-position stores")
        )
        problems.append(
            Problem(
                f"{named('positions')} gives {len(names)} positions and {which} (no HCS plate lists it), but an "
                "output holds one single-position store or the positions of plates: reconstruct each "
                "single-position store in a run of its own, or link them into an HCS plate as "
                "plate.zarr/<row>/<column>/<field> (its metadata listing them) and give the plate",
                named("positions"),
            )
        )
    refused = [
        name
        for name in names
        if name != SINGLE_POSITION and (len(name.split("/")) != 3 or not all(p.isalnum() for p in name.split("/")))
    ]
    if not refused:
        return problems
    shown = ", ".join(refused[:3]) + (f" and {len(refused) - 3} more" if len(refused) > 3 else "")
    return [
        *problems,
        Problem(
            "an output plate names each position row/column/field, each of letters and digits only, but the last "
            f"three folders of the input give {shown}; rename those folders, or link the position under a path such "
            "as plate.zarr/A/1/0",
            named("positions"),
        ),
    ]


def create_plate(
    output: str | Path,
    positions: Sequence[OutputPosition],
    *,
    channel_names: Sequence[str],
    records: Mapping[str, Mapping[str, object]],
    version: str,
    sources: Sequence[str | Path] = (),
) -> dict[str, tuple[int, ...]]:
    """Create the output plate, or add the positions and channels it lacks, and record the run.

    New positions are float32 arrays. An existing position keeps its array and other reconstructions' channels and
    records, and gains the channels it lacks. Each position's record is written under :func:`provenance_key`,
    replacing an earlier run's record for these channels.

    Parameters
    ----------
    output : str or Path
        The output plate, or a single-position store when :func:`is_single` holds for ``positions``.
    positions : Sequence[OutputPosition]
        The positions the reconstruction writes.
    channel_names : Sequence[str]
        The channels it writes, in the order of the shapes' channel axis.
    records : Mapping[str, Mapping[str, object]]
        Per position name, the JSON-ready record of the reconstruction.
    version : str
        OME-Zarr version of a new plate, ``0.4`` or ``0.5``.
    sources : Sequence[str or Path]
        Input store roots; a new position (and a new plate's root) inherits their non-OME attributes, first wins.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Per position name, the indices of ``channel_names`` along the output's channel axis.
    """
    from iohub.ngff import open_ome_zarr
    from iohub.ngff.utils import create_empty_plate

    output = Path(output)
    if is_single([position.name for position in positions]):
        (position,) = positions
        indices = _create_position(
            output,
            position,
            channel_names=channel_names,
            record=records[position.name],
            version=version,
            sources=sources,
        )
        return {position.name: indices}
    created = not output.exists()
    key = provenance_key(channel_names)
    groups: dict[tuple[object, ...], list[OutputPosition]] = {}
    for position in positions:
        record = json.dumps(records[position.name], sort_keys=True, default=str)
        group = (position.shape, position.scale, position.geometry, record)
        groups.setdefault(group, []).append(position)
    for members in groups.values():
        first = members[0]
        create_empty_plate(
            store_path=output,
            position_keys=[tuple(member.name.split("/")) for member in members],  # ty: ignore[invalid-argument-type]
            channel_names=list(channel_names),
            shape=first.shape,
            chunks=first.geometry.chunks,
            shards=first.geometry.shards,
            version=version,  # ty: ignore[invalid-argument-type]
            scale=first.scale,
            metadata_sources=[str(source) for source in dict.fromkeys(map(Path, sources))] or None,
            extra_metadata={key: dict(records[first.name])},
        )
    with open_ome_zarr(output, mode="r+") as plate:
        if created and sources:
            with open_ome_zarr(sources[0], mode="r") as source:
                inherited = {name: value for name, value in dict(source.zattrs).items() if name not in _PLATE_KEYS}
            plate.zattrs.put({**dict(plate.zattrs), **inherited})
        return {
            position.name: tuple(plate[position.name].channel_names.index(name) for name in channel_names)
            for position in positions
        }


def _create_position(
    output: Path,
    position: OutputPosition,
    *,
    channel_names: Sequence[str],
    record: Mapping[str, object],
    version: str,
    sources: Sequence[str | Path],
) -> tuple[int, ...]:
    """Create the single-position output store (as ``create_empty_plate`` does a plate position), or add channels."""
    from iohub.ngff import open_ome_zarr
    from iohub.ngff.models import TransformationMeta
    from iohub.ngff.utils import _default_chunks, _selected_metadata_keys

    created = not output.exists()
    inherited: dict[str, object] = {}
    with open_ome_zarr(output, layout="fov", mode="a", channel_names=list(channel_names), version=version) as node:  # ty: ignore[no-matching-overload]
        if created:
            shards = position.geometry.shards
            node.create_zeros(
                "0",
                shape=position.shape,
                dtype="float32",
                chunks=position.geometry.chunks or _default_chunks(position.shape, "float32", version),  # ty: ignore[invalid-argument-type]
                shards="XYZ" if shards is None and version == "0.5" else shards,
                transform=[TransformationMeta(type="scale", scale=list(position.scale))],
            )
            for source in dict.fromkeys(map(Path, sources)):
                with open_ome_zarr(source, layout="fov", mode="r") as given:
                    attributes = dict(given.zattrs)
                for name in _selected_metadata_keys(attributes, None):
                    inherited.setdefault(name, attributes[name])
        for name in channel_names:
            if name not in node.channel_names:
                node.append_channel(name, resize_arrays=True)
        node.zattrs.put({**dict(node.zattrs), **inherited, provenance_key(channel_names): dict(record)})
        return tuple(node.channel_names.index(name) for name in channel_names)


def process_time(  # noqa: PLR0913 (what iohub's process_single_position takes, for one time point)
    solve: Callable[[object], object],
    *,
    source: str | Path,
    output: str | Path,
    inputs: Sequence[int],
    outputs: Sequence[int],
    time: int,
    token: str,
) -> bool:
    """Reconstruct one time point of a position with iohub's ``process_single_position`` and write it.

    iohub reads channels ``inputs`` at ``time`` as a numpy ``(C, Z, Y, X)`` array, calls ``solve`` on it unless a
    channel is all zero or NaN, writes the result to channels ``outputs`` and records the time point under ``token``.
    iohub's stdout is logged at DEBUG.

    Returns
    -------
    bool
        Whether ``solve`` ran and its result was written; ``False`` if iohub skipped an all-zero or NaN input.

    Raises
    ------
    ReadError
        If iohub failed before ``solve`` was called (the input could not be read).
    Exception
        Whatever ``solve`` or the write raises.
    """
    from iohub.ngff.utils import process_single_position

    called = False

    def run(czyx: object) -> object:
        nonlocal called
        called = True
        return solve(czyx)

    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            process_single_position(
                run,  # ty: ignore[invalid-argument-type]  # iohub types it with numpy's NDArray, which is not imported
                input_position_path=Path(source),
                output_position_path=Path(output),
                input_channel_indices=[list(inputs)],
                output_channel_indices=[list(outputs)],
                input_time_indices=[time],
                resume_token=token,
            )
    except MemoryError:
        raise
    except Exception as error:
        if called:
            raise
        raise ReadError(
            f"could not read time {time} of {source}: {type(error).__name__}: {error}; the store is damaged or "
            "incomplete there: restore it, or use another time point or position"
        ) from error
    finally:
        for line in said.getvalue().splitlines():
            log.debug("iohub: %s", line)
    return called


def finished(output: str | Path, *, time: int, channels: Sequence[int], token: str) -> bool:
    """Whether an earlier run under ``token`` finished writing ``channels`` at ``time`` (always ``False`` for v0.4)."""
    from iohub.ngff import open_ome_zarr
    from iohub.ngff._write_units import plan_write_unit, progress_dir_for, unit_is_complete

    progress = progress_dir_for(Path(output))
    with open_ome_zarr(output, layout="fov", mode="r") as position:
        unit = plan_write_unit(position.data, [time], list(channels), token=token, progress_dir=progress)
        return unit is not None and unit_is_complete(unit, position.data)


@contextlib.contextmanager
def tracked_time(output: str | Path, *, time: int, channels: Sequence[int], token: str) -> Generator[None]:
    """Record a time point that the block writes itself (the tile merger) as iohub records its own writes.

    The time point's old record and files are cleared before the block and it is recorded as finished after; a block
    that raises leaves it unfinished, so a resumed run rewrites it.
    """
    from iohub.ngff import open_ome_zarr
    from iohub.ngff._write_units import plan_write_unit, progress_dir_for

    progress = progress_dir_for(Path(output))
    with open_ome_zarr(output, layout="fov", mode="r") as position:
        unit = plan_write_unit(position.data, [time], list(channels), token=token, progress_dir=progress)
    if unit is not None:
        unit.begin()
    yield
    if unit is not None:
        unit.complete()


@dataclass(frozen=True)
class Resources:
    """Host memory one process needs to reconstruct the largest volume of a run.

    Attributes
    ----------
    shape_zyx : tuple[int, int, int]
        The largest input volume.
    host_bytes : int
        Host memory of one whole-volume solve: 32 float32 copies of the volume (3.x ``estimate_resources``).
    gb_ram : int
        ``host_bytes`` in GiB, rounded up, at least 1 (3.x ``gb_ram_per_cpu``).
    """

    shape_zyx: ZYX
    host_bytes: int
    gb_ram: int


def estimate_resources(shapes: ZYX | Sequence[ZYX]) -> Resources:
    """Estimate the host memory one process needs to reconstruct the largest of ``shapes`` whole.

    Parameters
    ----------
    shapes : (Z, Y, X) or sequence of (Z, Y, X)
        The shape of one input volume, or of several.

    Returns
    -------
    Resources
        The estimate for the largest volume.

    Raises
    ------
    ValueError
        If ``shapes`` is empty.
    """
    one = bool(shapes) and isinstance(shapes[0], int)
    listed: list[ZYX] = [shapes] if one else list(shapes)  # ty: ignore[invalid-assignment]
    if not listed:
        raise ValueError("estimate_resources needs at least one input shape")
    largest = max(listed, key=math.prod)
    host = math.prod(largest) * _BYTES_PER_VOXEL * _FOURIER_MULTIPLIER
    return Resources(shape_zyx=largest, host_bytes=host, gb_ram=max(1, math.ceil(host / 2**30)))
