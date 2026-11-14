"""The pieces of a model's ``reconstruct_positions``: the units, the output plate and the unit runner.

After every check (:mod:`waveorder.models._checks`) the output plate is created once; reconstructing fills it one
(position, time) :class:`Unit` at a time, through iohub for a whole volume or the tile driver for a tiled one. iohub
records each finished unit under the :func:`fingerprint` of the reconstruction, which ``resume`` skips.
"""

import contextlib
import hashlib
import json
import logging
import shutil
import socket
import time as clock
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from iohub.ngff import Position, open_ome_zarr

from waveorder.config import V4Config
from waveorder.errors import ConfigurationError, UnitResult, UnitStatus
from waveorder.io.plate import create_plate, finished, process_time, software_versions, tracked_time
from waveorder.io.positions import channel_indices, reading
from waveorder.models._base import ReconstructionModel
from waveorder.models._checks import ZYX, PixelSize, Plan, Times
from waveorder.nn import Distilled
from waveorder.nn.context import solve_context
from waveorder.nn.systems.base import ShiftInvariantSystem
from waveorder.tiling import TileOptions, TilePlan
from waveorder.tiling.merge import ArrayLike, Index, volumes

log = logging.getLogger(__name__)

type Timings = tuple[float, float, float] | tuple[None, None, None]

RESUME_UNTRACKED = (
    "resume was requested, but progress can only be tracked in an OME-Zarr v0.5 output; recomputing every timepoint."
)
"""The 3.x warning for ``resume`` into a v0.4 output."""


@dataclass(frozen=True, order=True)
class Unit:
    """One (position, time) of a reconstruction: the plate-relative position name and the absolute time index."""

    position: str
    time: int

    @property
    def name(self) -> str:
        """``<position>/t<time>``."""
        return f"{self.position}/t{self.time}"


@dataclass(frozen=True, kw_only=True)
class PositionWork:
    """How one position reconstructs: its input, voxel size, channel indices in and out, path and fingerprint.

    ``tiles`` is the tile plan the checks made on the ``"tiles"`` and ``"field_tiles"`` paths: planning computes
    the solver's filter on two grids to size the halo, so a unit reuses it instead of planning again.
    """

    source: str
    pixel_size: PixelSize
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    path: str
    token: str
    tiles: TilePlan | None = None


@dataclass(frozen=True, kw_only=True)
class Work:
    """What a process needs, besides the model, to reconstruct units; it pickles, so workers receive it as it is."""

    output: str
    channels: tuple[str, ...]
    fields: tuple[str, ...]
    positions: dict[str, PositionWork]
    tile: ZYX | None = None
    tile_overlap: ZYX | None = None
    resume: bool = False


@contextlib.contextmanager
def transfer_kept(model: ReconstructionModel) -> Generator[None]:
    """Keep the system's transfer function from the checks to the units of one run, then drop it.

    The checks and the units bind the same grids with the same parameters, so ``cache="auto"`` for the run computes
    (or reads from a loaded store) each transfer function once per grid. Workers receive the model without it.
    """
    system = model.system
    if not isinstance(system, ShiftInvariantSystem) or system.cache == "auto":
        yield
        return
    system.cache = "auto"
    try:
        yield
    finally:
        system.cache = "off"
        system._cache_entry = None  # noqa: SLF001 (the model keeps no transfer function after the run)
        if (stored := model._stored_transfers) is not None:  # noqa: SLF001
            stored.last = None


def write_plate(plan: Plan, records: Mapping[str, Mapping[str, object]]) -> dict[str, tuple[int, ...]]:
    """Create the plate (deleting the old one first for ``overwrite``); return each position's output channels."""
    if plan.replaces:
        log.info("replacing %s", plan.output)
        shutil.rmtree(plan.output)
    return create_plate(
        plan.output,
        plan.outputs,
        channel_names=plan.names,
        records=records,
        version=plan.version,
        sources=[meta.store_root for meta in plan.positions],
    )


def output_records(model: ReconstructionModel, plan: Plan, *, times: Times) -> dict[str, dict[str, object]]:
    """Return the record of each output position, one per distinct voxel size (and input grid, for z padding auto).

    With ``z_padding`` ``"auto"`` a record also holds ``z_padding``, the slices auto chose on the position's grid, so
    that the reconstruction can be made again with that integer.
    """
    selected: Times = times
    if times is not None:
        selected = [int(times)] if isinstance(times, int) else [int(t) for t in times]
    grids = {meta.name: meta.zyx for meta in plan.positions} if model.z_padding == "auto" else {}
    made: dict[tuple[PixelSize, ZYX | None], dict[str, object]] = {}
    records = {}
    for position in plan.outputs:
        size = plan.pixel_sizes[position.name]
        grid = grids.get(position.name)
        key = (size, grid)
        if key not in made:
            made[key] = provenance(model, plan.channels, pixel_size=size, times=selected)
            if grid is not None:
                slices = model.z_padding_for(grid, pixel_size=size)
                log.info("z_padding auto pads %s slices at each end of z on %s", slices, "x".join(map(str, grid)))
                made[key]["z_padding"] = slices
        records[position.name] = made[key]
    return records


def provenance(
    model: ReconstructionModel, channels: Sequence[str], *, pixel_size: PixelSize, times: Times
) -> dict[str, object]:
    """Return what an output position records about the reconstruction that writes it.

    ``config`` (the model's configuration as JSON) or ``model`` (its plain-data description, else the class names
    of its parts); ``bundle`` (path and sha256) for a distilled solver loaded from one; ``versions`` and ``host``.
    """
    record: dict[str, object]
    try:
        config = model.to_config(channels=list(channels), pixel_size=pixel_size, times=times)
        record = {"config": config.model_dump(mode="json")} if isinstance(config, V4Config) else {"model": config}
    except ConfigurationError as error:
        log.info("the model has no plain-data form (%s); recording the classes of its parts", error)
        parts = {"system": model.system, "object_prior": model.object_prior, "noise": model.noise}
        described = {name: type(part).__name__ for name, part in (parts | {"solver": model.solver}).items()}
        record = {"model": {"class": type(model).__name__, **described}}
    solver = model.solver
    if isinstance(solver, Distilled) and solver.bundle.path is not None:
        record["bundle"] = {"path": str(solver.bundle.path.absolute()), "sha256": solver.bundle.sha256}
    return record | {"versions": software_versions(), "host": socket.gethostname()}


_PLACEMENT = ("device",)
"""Entries of a configuration's ``solver`` block (a description's ``settings``) that say where a job computes."""


def fingerprint(
    model: ReconstructionModel, record: Mapping[str, object], *, tile: ZYX | None, tile_overlap: ZYX | None
) -> str:
    """Return 16 hex digits of the sha256 of what a reconstruction writes, under which iohub records finished units.

    It covers the record without the host, versions, selected times and solver placement; the tile arguments; and
    the model's ``tile_parameters``.
    """
    kept = {name: value for name, value in record.items() if name not in ("host", "versions")}
    for key in ("config", "model"):
        described = kept.get(key)
        if isinstance(described, Mapping):
            trimmed = dict(described)
            for block, left in (("data", ("time_indices",)), ("solver", _PLACEMENT), ("settings", _PLACEMENT)):
                if isinstance(described.get(block), Mapping):
                    trimmed[block] = {name: value for name, value in described[block].items() if name not in left}
            kept[key] = trimmed
    digest = hashlib.sha256(json.dumps([kept, tile, tile_overlap], sort_keys=True, default=str).encode())
    for name, values in sorted((model.tile_parameters or {}).items()):
        digest.update(name.encode())
        digest.update(values.detach().to(device="cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def unit_work(
    model: ReconstructionModel,
    plan: Plan,
    indices: Mapping[str, tuple[int, ...]],
    *,
    records: Mapping[str, Mapping[str, object]],
    resume: bool,
) -> Work:
    positions = {
        meta.name: PositionWork(
            source=str(meta.path),
            pixel_size=plan.pixel_sizes[meta.name],
            inputs=channel_indices(meta, plan.channels),
            outputs=tuple(indices[meta.name]),
            path=plan.grids[meta.name].path,
            token=fingerprint(model, records[meta.name], tile=plan.tile, tile_overlap=plan.tile_overlap),
            tiles=plan.grids[meta.name].tiles,
        )
        for meta in plan.positions
    }
    return Work(
        output=str(plan.output),
        channels=plan.channels,
        fields=plan.names,
        positions=positions,
        tile=plan.tile,
        tile_overlap=plan.tile_overlap,
        resume=resume,
    )


# --- one unit --------------------------------------------------------------------------------------------------------


class UnitRunner:
    """Reconstruct units in this process with one model (on its device)."""

    def __init__(self, model: ReconstructionModel, work: Work) -> None:
        self.model = model
        self.work = work
        self.host = socket.gethostname()
        self.device = str(model.device)

    def run(self, unit: Unit) -> UnitResult:
        """Reconstruct ``unit`` and write it, or skip it when ``resume`` finds it finished; never raises.

        An all-zero or all-NaN input writes nothing (``"empty"``); a unit that raises is ``"failed"``, its stack
        logged at INFO.
        """
        started = clock.perf_counter()
        item = self.work.positions[unit.position]
        output = Path(self.work.output) / unit.position
        timings: Timings = (None, None, None)
        error = None
        status: UnitStatus
        try:
            if self.work.resume and finished(output, time=unit.time, channels=item.outputs, token=item.token):
                status = "skipped"
            elif item.path == "whole":
                status, timings = self._whole(unit, item, output)
            else:
                status, timings = self._tiles(unit, item, output)
        except Exception as failure:  # any failure of a unit is reported as its result
            log.info("unit %s failed", unit.name, exc_info=True)
            status, error = "failed", failure
        read_s, solve_s, write_s = timings
        return UnitResult(
            unit=unit.name,
            position=unit.position,
            time=unit.time,
            status=status,
            error=None if error is None else f"{type(error).__name__}: {error}",
            error_type=None if error is None else type(error).__name__,
            wall_s=clock.perf_counter() - started,
            read_s=read_s,
            solve_s=solve_s,
            write_s=write_s,
            host=self.host,
            device=self.device,
        )

    def _whole(self, unit: Unit, item: PositionWork, output: Path) -> tuple[UnitStatus, Timings]:
        """Reconstruct the whole volume in one solve, which iohub reads and writes."""
        marks: list[float] = []

        def solve(czyx: object) -> object:
            marks.append(clock.perf_counter())
            data = torch.as_tensor(czyx)
            height, width = (int(n) for n in data.shape[-2:])
            with torch.no_grad(), solve_context(position=unit.position, time=unit.time, field_shape=(height, width)):
                estimate = self.model.reconstruct(data, pixel_size=item.pixel_size).estimate
            values = estimate.detach().to(device="cpu", dtype=torch.float32).numpy()
            marks.append(clock.perf_counter())
            return values

        tic = clock.perf_counter()
        written = process_time(
            solve,
            source=item.source,
            output=output,
            inputs=item.inputs,
            outputs=item.outputs,
            time=unit.time,
            token=item.token,
        )
        if not written:
            log.info("unit %s: the input is all zero or NaN; nothing is written", unit.name)
            return "empty", (None, None, None)
        read, solved = marks
        return "complete", (read - tic, solved - read, clock.perf_counter() - solved)

    def _tiles(self, unit: Unit, item: PositionWork, output: Path) -> tuple[UnitStatus, Timings]:
        """Reconstruct through the tile driver, which writes the blocks; iohub records the time point once written."""
        model = self.model
        with (
            tracked_time(output, time=unit.time, channels=item.outputs, token=item.token),
            open_ome_zarr(item.source, mode="r") as source,
            open_ome_zarr(output, mode="r+") as sink,
        ):
            image = source["0"]
            with reading(item.source, time=unit.time):
                empty = _empty(image, unit.time, item.inputs)
            if empty:
                log.info("unit %s: the input is all zero or NaN; nothing is written", unit.name)
                return "empty", (None, None, None)
            pairs = volumes(
                source, sink, channels=list(self.work.channels), fields=list(self.work.fields), times=[unit.time]
            )
            recorded = [(_Recorded(view, path=item.source, time=unit.time), into) for view, into in pairs]
            plan = item.tiles
            assert plan is not None  # the checks plan the tiles of every position on a tiled path
            _, y, x = plan.shape
            with solve_context(position=unit.position, time=unit.time, field_shape=(y, x)):
                stats = model.reconstruct_tiles(
                    recorded,
                    pixel_size=item.pixel_size,
                    plan=plan,
                    options=TileOptions(write_workers=_writers(sink, plan)),
                    parameters=model.tile_parameters if item.path == "field_tiles" else None,
                )
        return "complete", (stats.read_s, stats.solve_s, stats.write_s)


def _writers(sink: Position, plan: TilePlan) -> int:
    """Two writer threads when every row of tile cores owns its blocks (a plate made for these tiles), else one."""
    array = sink["0"]
    blocks = tuple(array.shards or array.chunks)[2:4]
    owned = all(t >= n or t % b == 0 for t, n, b in zip(plan.tile[:2], plan.shape[:2], blocks, strict=True))
    return 2 if owned else 1


class _Recorded:
    """A unit's recorded ``(C, Z, Y, X)`` view whose failed reads are ``ReadError`` naming the position and time."""

    def __init__(self, view: ArrayLike, *, path: str, time: int) -> None:
        self._view = view
        self._path = path
        self._time = time

    @property
    def shape(self) -> tuple[int, ...]:
        """The view's shape."""
        return tuple(self._view.shape)

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._view, name)

    def __getitem__(self, key: Index, /) -> object:
        with reading(self._path, time=self._time):
            return self._view[key]


def _empty(image: object, time: int, channels: Sequence[int]) -> bool:
    """Whether a selected channel at ``time`` is all zero or all NaN (as iohub skips), read slab by slab."""
    _, _, z, _, _ = (int(n) for n in image.shape)  # ty: ignore[missing-attribute]
    chunks = getattr(image, "chunks", None)
    step = max(1, int(chunks[2])) if chunks is not None else z
    for channel in channels:
        for start in range(0, z, step):
            block = torch.as_tensor(image[time, channel, start : start + step])  # ty: ignore[not-subscriptable]
            if ((block != 0) & ~torch.isnan(block)).any():
                break
        else:
            return True
    return False
