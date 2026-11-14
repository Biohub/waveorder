"""The driver: read tiles, reconstruct each, merge; with the tile loader and per-tile parameters.

:func:`reconstruct_series` runs a :class:`TileLoader` (tiles read ahead on threads, through a :class:`ChunkCache` for
a chunked source, uploaded on a side CUDA stream), binds and solves each tile, and blends the estimates with a
:class:`~waveorder.tiling.merge.TileMerger`. :func:`stack_tiles` reads every tile into one stack for a per-tile
calibration; :func:`recover_zernikes` fits per-tile Zernike coefficients with these pieces.
"""

import dataclasses
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, final

import torch
from torch import Tensor

from waveorder.errors import ParameterError, PlacementError, ShapeError
from waveorder.io.staging import Prefetcher
from waveorder.nn.context import solve_context
from waveorder.nn.model import ImagingModel
from waveorder.nn.operators import LinearOperator
from waveorder.nn.parameters import placement, same_device
from waveorder.nn.sampling import ZYX, Sampling
from waveorder.nn.scores import Score
from waveorder.nn.search.descent import GradientSearch
from waveorder.nn.solvers import Solver, SolveResult, Wiener
from waveorder.nn.systems.base import ImagingSystem, StateKey
from waveorder.nn.systems.tiled import (
    BeadFit,
    TileCalibration,
    TileOperators,
    TileParameters,
    calibrate_tiles,
    expect_tile_parameters,
    zernike_search,
    zernike_terms,
)
from waveorder.observe import emit, observing, operation
from waveorder.tiling.merge import (
    OUTPUT_DTYPES,
    ArrayLike,
    StagingRing,
    TileMerger,
    WritableArray,
    accumulate_dtype,
    block_shape,
    channel_axis,
    row_buffers,
    row_voxels,
)
from waveorder.tiling.plan import ChunkKey, Region, TilePlan, TileRegion, chunk_keys, peak_chunk_bytes

_MERGERS_IN_FLIGHT = 2
"""Volumes with an open merger: the one being reconstructed and the previous one finishing its writes."""

_FAILED = ("max_iterations", "inaccurate", "nonfinite")
"""Statuses of a solve that requested a criterion and missed it; ``"direct"`` and ``"budget"`` are not failures."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TileOptions:
    """How tiles move: read-ahead depth, thread counts, staging slots and the host memory budget.

    Parameters
    ----------
    prefetch : int
        Tiles read and uploaded ahead of the one being reconstructed; ``0`` reads each tile when requested.
    read_workers : int
        Reader threads, which decode and stage.
    pinned_slots : int
        Host staging buffers of one read region each (pinned on CUDA).
    write_workers : int
        Threads writing finished rows of cores. Above 1, ``plan.tile`` z and y must be multiples of the sink's chunks.
    memory_budget : int, optional
        Host bytes the run may hold (see :func:`estimate_memory`); a run that needs more is refused.
    chunk_cache : bool
        For a chunked source, decode each chunk once (:class:`ChunkCache`) instead of once per tile touching it.
    output_dtype : torch.dtype
        Dtype handed to the sink. Blending accumulates in float64 for float64, else in float32.
    operator_cache_bytes : int
        Device bytes of per-tile operators kept across volumes with a per-tile parameter table.
    """

    prefetch: int = 2
    read_workers: int = 2
    pinned_slots: int = 2
    write_workers: int = 1
    memory_budget: int | None = None
    chunk_cache: bool = True
    output_dtype: torch.dtype = torch.float32
    operator_cache_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        for name, minimum in (("prefetch", 0), ("read_workers", 1), ("pinned_slots", 1), ("write_workers", 1)):
            value = getattr(self, name)
            if value < minimum:
                raise ParameterError(f"TileOptions {name} must be an integer >= {minimum}, got {value!r}", path=name)
        for name in ("memory_budget", "operator_cache_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ParameterError(f"TileOptions {name} must be a nonnegative byte count, got {value}", path=name)
        if self.output_dtype not in OUTPUT_DTYPES:
            raise ParameterError(
                f"TileOptions output_dtype must be one of {OUTPUT_DTYPES}, got {self.output_dtype}", path="output_dtype"
            )


@dataclass(slots=True)
class TileStats:
    """Time in seconds (summed over tiles and threads, so it can exceed wall time) and bytes of a tiled run.

    Attributes
    ----------
    read_s, stage_s, upload_s : float
        Reader threads reading, copying into the staging ring, and uploading.
    wait_s, bind_s, solve_s, blend_s : float
        The consumer waiting for its next tile, binding operators, enqueueing solves, and blending estimates.
    gpu_s, write_s, wall_s : float
        Device time of solving and blending (0 on the CPU), writer threads writing rows, and the run's wall time.
    bytes_read, bytes_decoded, bytes_written : int
        Read regions delivered to tiles, data decoded from the source, and values handed to the sink.
    peak_cache_bytes : int
        Most bytes of decoded chunks held at once; 0 without the chunk cache.
    tiles, unconverged, nonfinite : int
        Tiles reconstructed; tiles whose solve missed a requested criterion; tiles with a non-finite estimate.
    """

    read_s: float = 0.0
    stage_s: float = 0.0
    upload_s: float = 0.0
    wait_s: float = 0.0
    bind_s: float = 0.0
    solve_s: float = 0.0
    gpu_s: float = 0.0
    blend_s: float = 0.0
    write_s: float = 0.0
    bytes_read: int = 0
    bytes_decoded: int = 0
    peak_cache_bytes: int = 0
    bytes_written: int = 0
    tiles: int = 0
    unconverged: int = 0
    nonfinite: int = 0
    wall_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, name: str, amount: float) -> None:
        """Add ``amount`` to the counter ``name``, from any thread."""
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)


def _read_block(array: ArrayLike, region: Region, *, channels: bool) -> Tensor:
    """Read ``region`` of ``array`` as a tensor ``(C, z, y, x)``, sharing memory with the backend's result."""
    key = tuple(slice(a, b) for a, b in region)
    value = torch.as_tensor(array[(slice(None), *key)] if channels else array[key])
    return value if channels else value.unsqueeze(0)


@dataclass
class _Entry:
    ready: threading.Event = field(default_factory=threading.Event)
    value: Tensor = field(default_factory=lambda: torch.empty(0))
    error: BaseException | None = None


class ChunkCache:
    """Decoded chunks of ``sources``, each decoded once and freed after the last tile that reads it.

    ``sources`` are all ``(C, Z, Y, X)`` (``channels``) or all ``(Z, Y, X)``, chunked as ``chunks`` along z, y, x.
    ``reads`` lists ``(volume, read region)`` for every tile; each must be passed to :meth:`read` once.
    """

    def __init__(
        self, sources: Sequence[ArrayLike], *, chunks: ZYX[int], reads: Iterable[tuple[int, Region]], channels: bool
    ) -> None:
        self.sources = sources
        self.chunks = chunks
        self.channels = channels
        self._uses: dict[ChunkKey, int] = defaultdict(int)
        for volume, region in reads:
            for key in chunk_keys(volume, region, chunks):
                self._uses[key] += 1
        self._entries: dict[ChunkKey, _Entry] = {}
        self._lock = threading.Lock()
        self.decoded_bytes = 0
        self.live_bytes = 0
        self.peak_live_bytes = 0

    def _box(self, key: ChunkKey) -> Region:
        volume, *index = key
        shape = self.sources[volume].shape[-3:]
        z, y, x = ((i * c, min((i + 1) * c, n)) for i, c, n in zip(index, self.chunks, shape, strict=True))
        return (z, y, x)

    def _decode(self, keys: list[ChunkKey]) -> None:
        """Decode ``keys`` (owned by this thread), one source read per run of chunks contiguous along z."""
        runs: list[list[ChunkKey]] = []
        for key in sorted(keys, key=lambda k: (k[0], k[2], k[3], k[1])):
            previous = runs[-1][-1] if runs else None
            if previous is not None and (previous[0], previous[1] + 1, *previous[2:]) == key:
                runs[-1].append(key)
            else:
                runs.append([key])
        for run in runs:
            boxes = [self._box(key) for key in run]
            box = ((boxes[0][0][0], boxes[-1][0][1]), boxes[0][1], boxes[0][2])
            block = _read_block(self.sources[run[0][0]], box, channels=self.channels)
            nbytes = block.numel() * block.element_size()
            start = box[0][0]
            with self._lock:
                self.decoded_bytes += nbytes
                self.live_bytes += nbytes
                self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
            for key, chunk in zip(run, boxes, strict=True):
                entry = self._entries[key]
                entry.value = block[:, chunk[0][0] - start : chunk[0][1] - start]
                entry.ready.set()

    def _claim(self, keys: Sequence[ChunkKey]) -> list[_Entry]:
        """Return the entries of ``keys``, decoding on this thread the ones no reader has claimed yet."""
        owned: list[ChunkKey] = []
        with self._lock:
            entries = []
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    entry = self._entries[key] = _Entry()
                    owned.append(key)
                entries.append(entry)
        try:
            if owned:
                self._decode(owned)
        except BaseException as error:
            for key in owned:
                self._entries[key].error = error
                self._entries[key].ready.set()
            raise
        return entries

    def read(
        self, volume: int, region: Region, *, into: Callable[[tuple[int, ...], torch.dtype], Tensor] | None = None
    ) -> Tensor:
        """Return ``region`` of volume ``volume`` as ``(C, z, y, x)``, into ``into(shape, dtype)`` if given."""
        keys = chunk_keys(volume, region, self.chunks)
        entries = self._claim(keys)
        try:
            for entry in entries:
                entry.ready.wait()
                if entry.error is not None:
                    raise entry.error
            first = entries[0].value
            shape = (first.shape[0], *(b - a for a, b in region))
            out = torch.empty(shape, dtype=first.dtype) if into is None else into(shape, first.dtype)
            for key, entry in zip(keys, entries, strict=True):
                dst, src = [], []
                for (a, b), (c0, c1) in zip(region, self._box(key), strict=True):
                    lo, hi = max(a, c0), min(b, c1)
                    dst.append(slice(lo - a, hi - a))
                    src.append(slice(lo - c0, hi - c0))
                out[(slice(None), *dst)].copy_(entry.value[(slice(None), *src)])
            return out
        finally:
            with self._lock:
                for key in keys:
                    self._uses[key] -= 1
                    if self._uses[key] <= 0:
                        entry = self._entries.pop(key, None)
                        if entry is not None:
                            self.live_bytes -= entry.value.numel() * entry.value.element_size()


class Tile(NamedTuple):
    """One loaded tile: its read region ``(C, z, y, x)`` on the loader's device (never reused), sampling and region."""

    data: Tensor
    sampling: Sampling
    region: TileRegion


@final
class TileLoader:
    """Iterate over ``(volume, tile)`` pairs of ``plan`` in each volume of ``sources``, volume by volume.

    Reader threads read ``options.prefetch`` tiles ahead, across volume boundaries, and upload them on a side stream;
    a reader's exception is raised from the iterator. ``sources`` hold one ``(C, Z, Y, X)`` or ``(Z, Y, X)`` array per
    volume, all one shape; tiles are converted to ``dtype`` on ``device``.
    """

    def __init__(
        self,
        sources: Sequence[ArrayLike],
        plan: TilePlan,
        *,
        pixel_size: ZYX[float],
        device: torch.device,
        dtype: torch.dtype,
        options: TileOptions,
    ) -> None:
        shapes = {tuple(source.shape) for source in sources}
        shape = next(iter(shapes), ())
        if len(shapes) != 1 or shape[-3:] != plan.shape or len(shape) not in (3, 4):
            raise ParameterError(
                f"plan.shape is {plan.shape} but the sources have shapes {sorted(shapes)}; expected one (C, *shape)",
                path="sources",
            )
        for index, source in enumerate(sources):
            declared = getattr(source, "dtype", None)
            # A torch dtype says ``is_complex``, a numpy one has ``kind == "c"``: no numpy import needed.
            if getattr(declared, "is_complex", False) or getattr(declared, "kind", None) == "c":
                raise ShapeError(f"source {index} is {declared}; data must be real")
        self.channels = len(shape) == 4
        self.sources = tuple(sources)
        self.plan = plan
        self.pixel_size = pixel_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.options = options
        chunks = {block_shape(source, "chunks") for source in sources}
        self.chunks = chunks.pop() if options.chunk_cache and len(chunks) == 1 else None
        self.stats = TileStats()
        self.ring = StagingRing(options.pinned_slots, pin=self.device.type == "cuda")
        self._prefetch = Prefetcher(self.device, workers=options.read_workers, ahead=options.prefetch, dtype=dtype)

    @contextmanager
    def _staged(self, read: tuple[int, TileRegion], *, cache: ChunkCache | None) -> Generator[tuple[Tensor]]:
        """Read a tile's region into a staging slot, held while the prefetcher uploads it (on its reader thread)."""
        volume, region = read
        cuda = self.device.type == "cuda"
        tic = time.perf_counter()
        held: list[Tensor] = []

        def slot(shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
            buffer, view = self.ring.acquire(shape, dtype)
            held.append(buffer)
            return view

        try:
            if cache is not None:
                host = cache.read(volume, region.read, into=slot)
            else:
                block = _read_block(self.sources[volume], region.read, channels=self.channels)
                decoded = time.perf_counter()
                self.stats.add("read_s", decoded - tic)
                tic = decoded
                host = slot(tuple(block.shape), block.dtype)
                host.copy_(block)
                self.stats.add("bytes_decoded", block.numel() * block.element_size())
            if host.is_complex():
                raise ShapeError(f"source {volume} returned {host.dtype} data; data must be real")
            self.stats.add("read_s" if cache is not None else "stage_s", time.perf_counter() - tic)
            self.stats.add("bytes_read", host.numel() * host.element_size())
            uploading = time.perf_counter()
            yield (host,)
            if cuda:
                self.stats.add("upload_s", time.perf_counter() - uploading)
        finally:
            for buffer in held:
                self.ring.release(buffer)

    def __iter__(self) -> Generator[tuple[int, Tile]]:
        reads = [(volume, region) for volume in range(len(self.sources)) for region in self.plan.regions]
        cache = None
        if self.chunks is not None:
            cache = ChunkCache(
                self.sources, chunks=self.chunks, reads=[(v, r.read) for v, r in reads], channels=self.channels
            )
        tiles = self._prefetch(partial(self._staged, cache=cache), reads)
        try:
            tic = time.perf_counter()
            for (volume, region), (data,) in tiles:
                self.stats.add("wait_s", time.perf_counter() - tic)
                yield volume, Tile(data, Sampling(shape=region.read_shape, pixel_size=self.pixel_size), region)
                tic = time.perf_counter()
        finally:
            tiles.close()  # stops the reader threads
            if cache is not None:
                self.stats.add("bytes_decoded", cache.decoded_bytes)
                self.stats.peak_cache_bytes = max(self.stats.peak_cache_bytes, cache.peak_live_bytes)


def stack_tiles(
    source: ArrayLike,
    plan: TilePlan,
    *,
    pixel_size: ZYX[float],
    device: torch.device | str,
    dtype: torch.dtype,
    options: TileOptions | None = None,
) -> tuple[Tensor, Sampling]:
    """Read every tile of ``plan`` from ``source`` into one ``(tiles, C, *plan.read_shape)`` tensor on ``device``.

    Row ``i`` is the tile whose ``TileRegion.index`` is ``i``, the row order of a parameter table. All tiles share the
    returned sampling.
    """
    loader = TileLoader(
        [source],
        plan,
        pixel_size=pixel_size,
        device=torch.device(device),
        dtype=dtype,
        options=options or TileOptions(),
    )
    return torch.stack([tile.data for _, tile in loader]), Sampling(shape=plan.read_shape, pixel_size=pixel_size)


@final
class OperatorCache:
    """Bind ``system`` once per sampling; reuse the operator while ``state_key()`` is unchanged and not ``None``."""

    def __init__(self, system: ImagingSystem) -> None:
        self.system = system
        self._operators: dict[Sampling, LinearOperator] = {}
        self._key: StateKey | None = None
        self.binds = 0

    def __call__(self, sampling: Sampling) -> LinearOperator:
        """Return the system bound to ``sampling``, from the cache when nothing that determines it changed."""
        key = self.system.state_key()
        if key != self._key:
            self._operators.clear()
            self._key = key
        operator = None if key is None else self._operators.get(sampling)
        if operator is None:
            self.binds += 1
            operator = self.system.bind(sampling)
            if key is not None:
                self._operators[sampling] = operator
        return operator


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Bytes a tiled reconstruction holds, by owner.

    Attributes
    ----------
    staging, decoded, output_staging : int
        Host memory: the staging ring, decoded source data (an upper bound without the chunk cache), and rows of
        cores waiting for their write.
    accumulators, device_prefetch : int
        Device memory: partial sums of cores, and tiles uploaded ahead.
    on_host : bool
        Whether the reconstruction device is the CPU, so the device terms are host memory.
    read_shape : tuple[int, int, int]
        Voxels every tile reads along z, y, x.
    """

    staging: int
    decoded: int
    output_staging: int
    accumulators: int
    device_prefetch: int
    on_host: bool
    read_shape: ZYX[int]

    @property
    def host(self) -> int:
        """Host bytes, what ``options.memory_budget`` bounds: the device terms count only on the CPU."""
        device = self.accumulators + self.device_prefetch if self.on_host else 0
        return self.staging + self.decoded + self.output_staging + device


def estimate_memory(  # noqa: PLR0913 (the plan, the data layout, the source's storage, the options and the device)
    plan: TilePlan,
    *,
    channels: int,
    fields: int,
    source_itemsize: int,
    options: TileOptions,
    chunks: ZYX[int] | None = None,
    volumes: int = 1,
    device: torch.device | str = "cpu",
) -> MemoryEstimate:
    """Estimate the memory :func:`reconstruct_series` needs for ``plan``.

    Parameters
    ----------
    plan : TilePlan
        The tiling.
    channels, fields : int
        Data channels and object fields.
    source_itemsize : int
        Bytes per source element.
    options : TileOptions
        How tiles move.
    chunks : tuple[int, int, int], optional
        The source's chunk shape, used with ``options.chunk_cache``.
    volumes : int
        Volumes in the run.
    device : torch.device or str
        The reconstruction device.

    Returns
    -------
    MemoryEstimate
        Bytes by owner; ``.host`` is what ``options.memory_budget`` bounds.
    """
    read = channels * source_itemsize * math.prod(plan.read_shape)
    voxels = fields * max(math.prod(b - a for a, b in r.core) for r in plan.regions)
    if chunks is not None and options.chunk_cache:
        decoded = peak_chunk_bytes(
            [(v, r.read) for v in range(volumes) for r in plan.regions],
            chunks=chunks,
            chunk_bytes=channels * source_itemsize * math.prod(chunks),
            ahead=options.prefetch + options.read_workers,
        )
    else:
        decoded = options.read_workers * read
    mergers = min(volumes, _MERGERS_IN_FLIGHT)
    rows = row_buffers(plan, options.write_workers) * fields * row_voxels(plan)
    return MemoryEstimate(
        staging=options.pinned_slots * read,
        decoded=decoded,
        output_staging=mergers * rows * options.output_dtype.itemsize,
        accumulators=mergers * plan.peak_pending_blocks * voxels * accumulate_dtype(options.output_dtype).itemsize,
        device_prefetch=max(options.prefetch, 1) * read,
        on_host=torch.device(device).type == "cpu",
        read_shape=plan.read_shape,
    )


type Volume = tuple[ArrayLike, WritableArray]
"""One volume of a series: its data source and the sink of its estimate."""


def expect_volumes(volumes: Sequence[Volume], *, where: str) -> tuple[Volume, ...]:
    """Return the ``(source, sink)`` pairs of ``volumes`` as a tuple of tuples.

    Raises
    ------
    ShapeError
        If ``volumes`` is empty.
    ParameterError
        If one sink appears twice (their writes would interleave); ``path`` is ``volumes.<i>.sink``.
    """
    if not volumes:
        raise ShapeError(f"{where}: volumes is empty; give one (source, sink) pair per volume")
    pairs: list[Volume] = []
    seen: dict[int, int] = {}
    for index, (source, sink) in enumerate(volumes):
        if (first := seen.setdefault(id(sink), index)) != index:
            raise ParameterError(
                f"{where}: volumes.{index}.sink is the sink of volumes.{first} too; give each volume its own sink",
                path=f"volumes.{index}.sink",
            )
        pairs.append((source, sink))
    return tuple(pairs)


def reconstruct_series(  # noqa: PLR0913 (the volumes, the model and solver, the grid's voxel size and tiling, a device)
    volumes: Sequence[Volume],
    *,
    model: ImagingModel,
    solver: Solver,
    pixel_size: ZYX[float],
    plan: TilePlan,
    device: torch.device | str,
    options: TileOptions | None = None,
    parameters: TileParameters | None = None,
) -> TileStats:
    """Reconstruct each ``(source, sink)`` volume tile by tile.

    Reading, uploading, solving and writing overlap, also across volumes. Runs under ``torch.no_grad()``; an observer
    receives one ``tiles.end`` event with every :class:`TileStats` field.

    Parameters
    ----------
    volumes : sequence of (ArrayLike, WritableArray)
        Per volume, data ``(C, Z, Y, X)`` (``(Z, Y, X)`` for one channel, real) and a sink ``(F, Z, Y, X)``
        (``(Z, Y, X)`` for one field); :func:`~waveorder.tiling.merge.volumes` builds them from OME-Zarr positions.
    model, solver : ImagingModel, Solver
        The model, on ``device``, and the solver run once per tile.
    pixel_size : tuple[float, float, float]
        Voxel size along z, y, x in micrometers.
    plan, device, options : TilePlan, torch.device or str, TileOptions or None
        The tiling of every volume, where tiles are reconstructed, and how tiles move.
    parameters : Mapping[str, Tensor], optional
        Per-tile values of free parameters, one row per tile of the plan
        (:data:`~waveorder.nn.systems.tiled.TileParameters`).

    Returns
    -------
    TileStats
        Timings and byte counts of the run.

    Raises
    ------
    PlacementError
        If the model lives on another device.
    ShapeError
        If ``volumes`` is empty, or a source or sink does not match the plan and the model's channels and fields.
    ParameterError
        If a sink appears twice, ``pixel_size`` is not three positive finite sizes, the estimated host memory exceeds
        ``options.memory_budget``, or ``parameters`` is invalid (``path`` is ``parameters.<name>``).
    """
    where = "reconstruct_series"
    volumes = expect_volumes(volumes, where=where)
    Sampling(shape=plan.shape, pixel_size=tuple(pixel_size))  # three positive finite sizes, or ParameterError
    options = options or TileOptions()
    device = torch.device(device)
    system = model.system
    dtype, model_device = placement(system)
    if not same_device(model_device, device):
        raise PlacementError(
            f"device is {device} but {type(system).__name__} computes in {dtype} on {model_device}; "
            "move the model with .to(device) or pass its device"
        )
    channels, fields = system.channel_count, len(system.fields)
    sources = [source for source, _ in volumes]
    for source in sources:
        channel_axis(source, count=channels, volume=plan.shape, what="source")
    loader = TileLoader(sources, plan, pixel_size=pixel_size, device=device, dtype=dtype, options=options)
    declared = getattr(getattr(sources[0], "dtype", None), "itemsize", None)
    estimate = estimate_memory(
        plan,
        channels=channels,
        fields=fields,
        source_itemsize=declared if isinstance(declared, int) else 4,
        options=options,
        chunks=loader.chunks,
        volumes=len(volumes),
        device=device,
    )
    if options.memory_budget is not None and estimate.host > options.memory_budget:
        raise ParameterError(
            f"options.memory_budget = {options.memory_budget} bytes is below the {estimate.host} the plan needs "
            f"({estimate}); shrink plan.tile or options.pinned_slots, read_workers or write_workers",
            path="options.memory_budget",
        )
    table = None
    if parameters is not None:
        table = expect_tile_parameters(parameters, model=model, tiles=len(plan.regions), where=where)
    tiles = TileOperators(model, table, budget=options.operator_cache_bytes) if table else None
    run = _Run(loader, [sink for _, sink in volumes], model=model, solver=solver, tiles=tiles)
    with operation():
        stats = run()
        if observing():
            emit(
                "tiles.end",
                {item.name: getattr(stats, item.name) for item in dataclasses.fields(stats) if item.init and item.repr},
            )
    return stats


@final
class _Run:
    """One call of :func:`reconstruct_series`: the tile loop, one merger per volume and the timing."""

    def __init__(
        self,
        loader: TileLoader,
        sinks: Sequence[WritableArray],
        *,
        model: ImagingModel,
        solver: Solver,
        tiles: TileOperators | None,
    ) -> None:
        self.loader, self.sinks, self.model, self.solver, self.tiles = loader, sinks, model, solver, tiles
        self.operators = OperatorCache(model.system)
        self.stats = loader.stats
        self.device = loader.device
        self.cuda = self.device.type == "cuda"
        self.timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._nonfinite = torch.zeros((), dtype=torch.int64, device=self.device)
        self._open: TileMerger | None = None
        self._closing: list[Future[None]] = []

    def __call__(self) -> TileStats:
        start = time.perf_counter()
        with (
            torch.no_grad(),
            torch.cuda.device(self.device) if self.cuda else nullcontext(),
            ThreadPoolExecutor(1, thread_name_prefix="waveorder-tile-close") as closer,
        ):
            try:
                self._loop(closer)
            except BaseException:
                if self._open is not None:
                    self._open.abort()
                for future in self._closing:
                    future.exception()
                raise
        self.stats.gpu_s = sum(begin.elapsed_time(end) for begin, end in self.timings) / 1e3
        self.stats.nonfinite = int(self._nonfinite)  # the one read-back, after the run
        self.stats.wall_s = time.perf_counter() - start
        return self.stats

    def _loop(self, closer: ThreadPoolExecutor) -> None:
        current = -1
        for volume, tile in self.loader:
            if self._open is None or volume != current:
                self._hand_off(closer)
                self._open, current = self._merger(volume), volume
            tic = time.perf_counter()
            solve = self._bind(tile)
            bound = time.perf_counter()
            begin = self._event()
            # The region the solver's data covers within the field: the tile's read region, halo included.
            _, (y0, y1), (x0, x1) = tile.region.read
            context = solve_context(field_shape=self.loader.plan.shape[1:], region=(slice(y0, y1), slice(x0, x1)))
            with context:
                result = solve()
            self._open.submit(tile.region, result.estimate)
            end = self._event()
            if begin is not None and end is not None:
                self.timings.append((begin, end))
            self.stats.add("bind_s", bound - tic)
            self.stats.add("solve_s", time.perf_counter() - bound)
            self.stats.add("tiles", 1)
            self.stats.add("unconverged", int(result.status in _FAILED))
            self._nonfinite += ~result.finite
        self._hand_off(closer)
        for future in self._closing:
            future.result()
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _merger(self, volume: int) -> TileMerger:
        options = self.loader.options
        return TileMerger(
            self.sinks[volume],
            self.loader.plan,
            fields=len(self.model.system.fields),
            device=self.device,
            output_dtype=options.output_dtype,
            workers=options.write_workers,
            record=self.stats.add,
        )

    def _bind(self, tile: Tile) -> Callable[[], SolveResult]:
        """Bind the tile's operator and return its solve: with the tile's own values when there is a table."""
        if self.tiles is not None:
            index = tile.region.index
            operator = self.tiles(index, tile.sampling)
            return partial(self.tiles.solve, index, tile.data, tile.sampling, solver=self.solver, operator=operator)
        # A solver that never reads H (a learned one) costs no bind.
        shared = self.operators(tile.sampling) if self.solver.needs_operator else None
        return partial(self.solver, tile.data, self.model, tile.sampling, operator=shared)

    def _hand_off(self, closer: ThreadPoolExecutor) -> None:
        """Close the open merger in the background; keep at most ``_MERGERS_IN_FLIGHT`` mergers alive."""
        if self._open is None:
            return
        self._closing.append(closer.submit(self._open.close))
        self._open = None
        if len(self._closing) >= _MERGERS_IN_FLIGHT:
            self._closing.pop(0).result()

    def _event(self) -> torch.cuda.Event | None:
        if not self.cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event


@dataclass(frozen=True, slots=True)
class ZernikeRecovery:
    """What :func:`recover_zernikes` returns: the per-tile calibration and the blended deconvolution (F, Z, Y, X)."""

    calibration: TileCalibration
    reconstruction: Tensor

    def coefficients(self, model: ImagingModel) -> dict[int, Tensor]:
        """Return the recovered coefficients by Noll index, a ``(T,)`` tensor each, in waves."""
        return {noll: self.calibration.parameters[name] for name, noll in zernike_terms(model).items()}


def recover_zernikes(
    data: ArrayLike,
    model: ImagingModel,
    plan: TilePlan,
    *,
    pixel_size: ZYX[float],
    search: GradientSearch | None = None,
    score: Score | None = None,
) -> ZernikeRecovery:
    """Recover the model's free Zernike coefficients on every tile of a bead field, then deconvolve and blend.

    Each tile is fitted alone, ``tile ~ bead * PSF(c)``, then Wiener-deconvolved with its own coefficients.

    Parameters
    ----------
    data : ArrayLike
        The volume, ``(Z, Y, X)`` or ``(C, Z, Y, X)``, one bead at the centre of each tile.
    model : ImagingModel
        A model with a :class:`~waveorder.nn.ZernikeAberration` with free bounded terms; other free quantities are
        fitted per tile too.
    plan : TilePlan
        The tiling.
    pixel_size : tuple[float, float, float]
        Voxel size along z, y, x in micrometers.
    search : GradientSearch, optional
        The fit; by default :func:`~waveorder.nn.systems.tiled.zernike_search`.
    score : Score, optional
        The per-tile loss; by default :meth:`~waveorder.nn.systems.tiled.BeadFit.gaussian`.

    Returns
    -------
    ZernikeRecovery
        The calibration and the reconstruction, on the model's device.

    Raises
    ------
    ParameterError
        If the model has no free Zernike term (``path="model"``).
    IncompatibleModelError
        As :func:`~waveorder.nn.calibrate_tiles` raises it.
    """
    terms = zernike_terms(model)
    if not terms:
        raise ParameterError(
            "recover_zernikes needs a model with free ZernikeAberration terms; it has none", path="model"
        )
    dtype, device = placement(model)
    tiles, sampling = stack_tiles(data, plan, pixel_size=pixel_size, device=device, dtype=dtype)
    score = BeadFit.gaussian(sampling, like=tiles).to(device) if score is None else score
    search = zernike_search(terms) if search is None else search
    calibration = calibrate_tiles(tiles, model, sampling, score=score, search=(search,))
    fields = model.system.object_shape(sampling)[0]
    reconstruction = torch.zeros((fields, *data.shape[-3:]), dtype=dtype, device=device)
    reconstruct_series(
        [(data, reconstruction)],
        model=model,
        solver=Wiener(),
        pixel_size=pixel_size,
        plan=plan,
        device=device,
        options=TileOptions(output_dtype=dtype),  # a float64 model blends and writes in float64
        parameters=calibration.parameters,
    )
    return ZernikeRecovery(calibration, reconstruction)
