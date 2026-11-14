"""The tile merger, and the arrays tiles are read from and written to.

A source is anything with ``shape`` and numpy-style basic slicing (:class:`ArrayLike`); a sink also takes item
assignment (:class:`WritableArray`): a zarr or iohub array, a ``numpy.memmap``, a torch tensor. :class:`Selection`
fixes leading axes lazily; :func:`volumes` pairs the images of a source and a sink OME-Zarr position, one
``(source, sink)`` volume per time point. :class:`TileMerger` blends weighted estimates into cores and writes each
finished row of cores (same z and y grid position) in one call; rows partition the volume, so no voxel is written twice.
"""

import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, final, runtime_checkable

import torch
from torch import Tensor

from waveorder.errors import ParameterError, ShapeError
from waveorder.nn.sampling import ZYX
from waveorder.tiling.plan import Region, TilePlan, TileRegion, intersect, relative

if TYPE_CHECKING:  # importing iohub takes a second; the aliases below are evaluated only when inspected
    from iohub.ngff.nodes import ImageArray, Position

type _Position = Position
type _ImageArray = ImageArray

type Index = tuple[int | slice, ...]
"""A numpy-style basic selection: integers and unit-step slices, one per leading axis."""

OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
"""Dtypes a merger hands to its sink."""


@runtime_checkable
class ArrayLike(Protocol):
    """A readable array: ``shape`` and ``array[index]`` with basic slicing."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Size along each axis."""
        ...

    def __getitem__(self, index: Index, /) -> Any:  # noqa: ANN401 (numpy arrays, tensors and backend buffers all qualify)
        """Return the selected block."""
        ...


@runtime_checkable
class WritableArray(ArrayLike, Protocol):
    """A writable array: ``array[index] = value`` with basic slicing."""

    def __setitem__(self, index: Index, value: Any, /) -> None:  # noqa: ANN401 (the backend decides what it accepts)
        """Store ``value`` into the selected block."""
        ...


@final
class Selection:
    """A lazy view of ``array`` with the leading axes fixed at ``index`` and, optionally, some channels picked.

    With ``channels`` (indices along the axis after the fixed ones) the view is ``(C', Z, Y, X)`` and reads and writes
    the parent one channel at a time, so the channels need not be contiguous.
    """

    def __init__(self, array: ArrayLike, index: tuple[int, ...], *, channels: tuple[int, ...] | None = None) -> None:
        if len(index) >= len(array.shape):
            raise ShapeError(f"select: {len(index)} indices fix every axis of an array of shape {tuple(array.shape)}")
        for axis, (i, n) in enumerate(zip(index, array.shape, strict=False)):
            if not 0 <= i < n:
                raise ShapeError(f"select: index {i} on axis {axis} is outside [0, {n})")
        self.array = array
        self.index = index
        self.channels = channels
        rest = tuple(int(n) for n in array.shape[len(index) :])
        self.shape = rest if channels is None else (len(channels), *rest[1:])
        self.dtype = getattr(array, "dtype", None)
        self.chunks = self._blocks("chunks")
        self.shards = self._blocks("shards")

    def _blocks(self, attribute: str) -> tuple[int, ...] | None:
        value = getattr(self.array, attribute, None)
        if value is None:
            return None
        rest = tuple(int(n) for n in value[len(self.index) :])
        return rest if self.channels is None else (1, *rest[1:])

    def _picked(self, key: Index) -> tuple[tuple[slice, ...], tuple[int, ...], tuple[slice, ...], list[int]]:
        """Return the fixed axes as slices, the parent channels, the spatial slices and the axes an integer drops."""
        if self.channels is None or len(key) != len(self.shape):
            raise ShapeError(f"a channel selection takes a key of {len(self.shape)} parts (channel first), got {key!r}")
        channel, *spatial = key
        channels = (self.channels[channel],) if isinstance(channel, int) else self.channels[channel]
        slices = tuple(slice(part, part + 1) if isinstance(part, int) else part for part in spatial)
        fixed = tuple(slice(i, i + 1) for i in self.index)
        return fixed, channels, slices, [axis for axis, part in enumerate(key) if isinstance(part, int)]

    def __getitem__(self, key: Index, /) -> Any:  # noqa: ANN401 (the parent's block, or a tensor of picked channels)
        if self.channels is None:
            return self.array[(*self.index, *key)]
        fixed, channels, spatial, drop = self._picked(key)
        first = (0,) * (len(fixed) + 1)
        block = torch.stack([torch.as_tensor(self.array[(*fixed, slice(c, c + 1), *spatial)])[first] for c in channels])
        for axis in reversed(drop):
            block = block.squeeze(axis)
        return block

    def __setitem__(self, key: Index, value: Any, /) -> None:  # noqa: ANN401 (numpy array or tensor)
        if not isinstance(self.array, WritableArray):
            raise ParameterError(f"sink: {type(self.array).__name__} under select() does not support item assignment")
        if self.channels is None:
            self.array[(*self.index, *key)] = value
            return
        fixed, channels, spatial, drop = self._picked(key)
        if drop:
            raise ShapeError("writes to a channel selection take slices only, not integer indices")
        for index, c in enumerate(channels):
            self.array[(*fixed, slice(c, c + 1), *spatial)] = value[index][(None,) * (len(fixed) + 1)]


def block_shape(array: ArrayLike, attribute: str) -> ZYX[int] | None:
    """Return the last three entries of ``array.<attribute>`` (``"chunks"`` or ``"shards"``), or ``None``."""
    value = getattr(array, attribute, None)
    if value is None:
        return None
    *_, z, y, x = (int(n) for n in value)
    return (z, y, x)


def channel_axis(array: ArrayLike, *, count: int, volume: ZYX[int], what: str) -> bool:
    """Return whether ``array`` is ``(count, Z, Y, X)``; ``(Z, Y, X)`` for one channel has no axis; else ShapeError."""
    shape = tuple(array.shape)
    if shape == (count, *volume):
        return True
    if count == 1 and shape == volume:
        return False
    raise ShapeError(f"{what} has shape {shape}, expected ({count}, *{volume}) or, for one channel, {volume}")


def _image(position: _Position, *, name: str) -> _ImageArray:
    if "0" not in position.array_keys():
        raise ParameterError(
            f"volumes: {name} has no image '0'; it has {sorted(position.array_keys())}", path=f"{name}_image"
        )
    array = position["0"]
    if len(array.shape) != 5:
        raise ShapeError(f"volumes: {name} image '0' has shape {tuple(array.shape)}, expected (T, C, Z, Y, X)")
    return array


def _channel_indices(position: _Position, names: Sequence[str] | None, *, name: str) -> tuple[int, ...]:
    if names is None:
        return tuple(range(len(position.channel_names)))
    if not names:
        raise ParameterError(f"volumes: {name} must name at least one channel", path=name)
    for index, channel in enumerate(names):
        if channel not in position.channel_names:
            raise ParameterError(
                f"volumes: {name}.{index} = {channel!r} is not a channel of the position; it has "
                f"{list(position.channel_names)}",
                path=f"{name}.{index}",
            )
    if len(set(names)) != len(names):
        raise ParameterError(f"volumes: {name} names a channel twice: {list(names)}", path=name)
    return tuple(position.get_channel_index(channel) for channel in names)


def volumes(
    source: _Position,
    sink: _Position,
    *,
    channels: Sequence[str],
    fields: Sequence[str] | None = None,
    times: Sequence[int] | None = None,
) -> list[tuple[Selection, Selection]]:
    """Return the ``(source, sink)`` volumes of the images ``"0"`` of ``source`` and ``sink``, one per time point.

    Parameters
    ----------
    source : iohub.ngff.Position
        The data, opened for reading.
    sink : iohub.ngff.Position
        Where estimates go, opened for writing, with the source's time, z, y and x extents
        (:func:`waveorder.io.plate.create_plate` makes one).
    channels : sequence of str
        Source channel names, in the order of the model's data channels.
    fields : sequence of str, optional
        Sink channel names, in the order of the model's object fields; all of the sink's channels by default.
    times : sequence of int, optional
        Time indices to reconstruct; all by default.

    Returns
    -------
    list of (Selection, Selection)
        Lazy ``(C, Z, Y, X)`` source and ``(F, Z, Y, X)`` sink views, one pair per selected time point.

    Raises
    ------
    ParameterError
        If an image is missing, a channel name is unknown or repeated, a time is out of range, or the sink's chunks
        span several time points; ``path`` names the argument (``channels.1``, ``times.0``, ``sink``).
    ShapeError
        If an image is not ``(T, C, Z, Y, X)``, or the sink's time, z, y and x extents differ from the source's.
    """
    read = _image(source, name="source")
    write = _image(sink, name="sink")
    inputs = _channel_indices(source, channels, name="channels")
    outputs = _channel_indices(sink, fields, name="fields")
    if (read.shape[0], *read.shape[2:]) != (write.shape[0], *write.shape[2:]):
        raise ShapeError(
            f"volumes: the sink image has shape {tuple(write.shape)} but the source {tuple(read.shape)}; their time, "
            "z, y and x extents must match"
        )
    for blocks in (getattr(write, "chunks", None), getattr(write, "shards", None)):
        if blocks is not None and blocks[0] != 1:
            raise ParameterError(
                f"volumes: the sink's chunks {tuple(blocks)} span {blocks[0]} time points; write one time point per "
                "chunk (and shard), or the volumes' writes would interleave",
                path="sink",
            )
    count = int(read.shape[0])
    selected = range(count) if times is None else times
    for index, t in enumerate(selected):
        if not 0 <= t < count:
            raise ParameterError(f"volumes: times.{index} = {t} is outside [0, {count})", path=f"times.{index}")
    return [(Selection(read, (t,), channels=inputs), Selection(write, (t,), channels=outputs)) for t in selected]


@final
class StagingRing:
    """At most ``slots`` reusable host buffers, allocated on first use, pinned if ``pin``, grown only when too small."""

    def __init__(self, slots: int, *, pin: bool) -> None:
        self.pin = pin
        self._free: list[Tensor] = []
        self._lock = threading.Lock()
        self._available = threading.Semaphore(slots)
        self.allocated = 0
        self.in_use = 0
        self.peak_in_use = 0

    def acquire(self, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Block until a buffer is free; return it and a contiguous ``shape``, ``dtype`` view of it.

        Pass the buffer to :meth:`release`, from any thread, once nothing reads the view.
        """
        nbytes = math.prod(shape) * dtype.itemsize
        self._available.acquire()
        with self._lock:
            buffer = self._free.pop() if self._free else None
            self.in_use += 1
            self.peak_in_use = max(self.peak_in_use, self.in_use)
            if buffer is None or buffer.numel() < nbytes:
                self.allocated += 1
        if buffer is None or buffer.numel() < nbytes:
            try:
                buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self.pin)
            except BaseException:
                self.release(None)
                raise
        return buffer, buffer[:nbytes].view(dtype).view(shape)

    def release(self, buffer: Tensor | None) -> None:
        """Return a buffer from :meth:`acquire` to the ring (``None`` gives back a slot whose buffer never existed)."""
        with self._lock:
            if buffer is not None:
                self._free.append(buffer)
            self.in_use -= 1
        self._available.release()


def accumulate_dtype(output_dtype: torch.dtype) -> torch.dtype:
    """Return the dtype contributions are weighted and summed in: float64 for a float64 output, else float32."""
    return torch.float64 if output_dtype == torch.float64 else torch.float32


def row_buffers(plan: TilePlan, workers: int) -> int:
    """Rows of cores a merger holds on the host: open rows, then per write thread two waiting and one being written."""
    return plan.peak_open_rows + 3 * workers


def row_voxels(plan: TilePlan) -> int:
    """Voxels of the largest row of cores: one core's z and y extent across the whole x extent."""
    return max((b - a) * (d - c) for (a, b), (c, d), _ in (r.core for r in plan.regions)) * plan.shape[2]


def _expect_write_alignment(sink: WritableArray, plan: TilePlan, *, workers: int) -> None:
    """Raise unless ``workers`` threads can write rows into ``sink`` without two sharing a chunk (shard)."""
    blocks = block_shape(sink, "shards") or block_shape(sink, "chunks")
    if workers == 1 or blocks is None:
        return
    for name, n, tile, block in zip("zy", plan.shape, plan.tile, blocks, strict=False):
        if tile < n and tile % block:
            raise ParameterError(
                f"options.write_workers = {workers} needs plan.tile.{name} = {tile} to be a multiple of the sink's "
                f"block size {block} along {name}, or options.write_workers = 1",
                path="options.write_workers",
            )


@dataclass
class _Row:
    """A row of cores on the host: its box, one contiguous tensor per core, the last download event, slots to free."""

    box: Region
    cores: list[Tensor | None]
    remaining: int
    buffers: list[Tensor] = field(default_factory=list)
    ready: torch.cuda.Event | None = None


@final
class TileMerger:
    """Accumulate weighted tile estimates into cores and write finished cores to ``sink`` asynchronously.

    ``sink`` is ``(F, Z, Y, X)`` or, for one field, ``(Z, Y, X)``, and is assigned ``output_dtype`` values (a tensor,
    or a numpy view of one for a non-tensor sink). Estimates of ``fields`` fields arrive on ``device``, where partial
    sums live, in :func:`accumulate_dtype`. ``workers`` is ``TileOptions.write_workers``; ``record(name, amount)``
    adds to a :class:`~waveorder.tiling.TileStats` counter from any thread.
    """

    def __init__(  # noqa: PLR0913 (the sink and its tiling, then what arrives and how it is written)
        self,
        sink: WritableArray,
        plan: TilePlan,
        *,
        fields: int,
        device: torch.device,
        output_dtype: torch.dtype,
        workers: int,
        record: Callable[[str, float], None],
    ) -> None:
        self.channels = channel_axis(sink, count=fields, volume=plan.shape, what="sink")
        _expect_write_alignment(sink, plan, workers=workers)
        if output_dtype == torch.bfloat16 and not isinstance(sink, Tensor):
            raise ParameterError(
                "options.output_dtype = bfloat16 needs a tensor sink; numpy has no bfloat16",
                path="options.output_dtype",
            )
        self.sink = sink
        self.plan = plan
        self.fields = fields
        self.device = torch.device(device)
        self.output_dtype = output_dtype
        self.accumulate = accumulate_dtype(output_dtype)
        self.record = record
        self._weights = self._upload_weights(plan)
        self._remaining = [len(plan.contributors(region)) for region in plan.regions]
        self._partial: dict[int, Tensor] = {}
        self._rows: dict[tuple[int, int], _Row] = {}
        cuda = self.device.type == "cuda"
        self._ring = StagingRing(row_buffers(plan, workers) * plan.grid[2], pin=True) if cuda else None
        self._download = torch.cuda.Stream(self.device) if cuda else None
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="waveorder-tile-write")
        self._in_flight = threading.Semaphore(2 * workers)  # finished rows waiting for a write; more block submit
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()

    def _upload_weights(self, plan: TilePlan) -> tuple[tuple[Tensor, ...], ...]:
        """Move every per-axis weight to the device in one copy; from pinned memory it does not block the host."""
        flat = [w.to(self.accumulate) for axis in plan.weights for w in axis]
        host = torch.cat(flat)
        if self.device.type == "cuda":
            self._weights_host = host.pin_memory()  # kept alive until the asynchronous copy has read it
            host = self._weights_host.to(self.device, non_blocking=True)
        parts = iter(host.split([w.numel() for w in flat]))
        return tuple(tuple(next(parts) for _ in axis) for axis in plan.weights)

    def submit(self, region: TileRegion, estimate: Tensor) -> None:
        """Blend a tile's estimate ``(F, *region.read_shape)`` into its cores and queue the cores it completes.

        Blocks only while every output slot waits for its write. Raises a ParameterError for an estimate of the wrong
        shape, and re-raises the first error a write thread hit.
        """
        if self._error is not None:
            raise self._error
        expected = (self.fields, *region.read_shape)
        if tuple(estimate.shape) != expected:
            raise ParameterError(f"estimate of tile {region.index} has shape {tuple(estimate.shape)}, not {expected}")
        tic = time.perf_counter()
        wz, wy, wx = (self._weights[a][g] for a, g in enumerate(region.grid))
        weights = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
        values = estimate[(slice(None), *region.crop)].to(self.accumulate) * weights
        for index in self.plan.blocks(region):
            block = self.plan.regions[index]
            overlap = intersect(block.core, region.blend)
            if overlap is None:
                continue
            part = values[(slice(None), *relative(overlap, region.blend))]
            self._remaining[index] -= 1
            partial = self._partial.pop(index, None)
            if partial is None and self._remaining[index] == 0:
                self._finish(block, part)  # the only contributor: its weights are 1 on this core
                continue
            if partial is None:
                shape = (self.fields, *(b - a for a, b in block.core))
                partial = torch.zeros(shape, dtype=self.accumulate, device=self.device)
            partial[(slice(None), *relative(overlap, block.core))] += part
            if self._remaining[index] == 0:
                self._finish(block, partial)
            else:
                self._partial[index] = partial
        self.record("blend_s", time.perf_counter() - tic)

    def _finish(self, block: TileRegion, values: Tensor) -> None:
        """Stage a finished core on the host in its own contiguous slot (one asynchronous copy); write full rows."""
        key = (block.grid[0], block.grid[1])
        row = self._rows.get(key)
        if row is None:
            box = (block.core[0], block.core[1], (0, self.plan.shape[2]))
            row = self._rows[key] = _Row(box, [None] * self.plan.grid[2], self.plan.grid[2])
        values = values.to(self.output_dtype).contiguous()
        if self._ring is None or self._download is None:
            row.cores[block.grid[2]] = values  # nothing writes to ``values`` after this
        else:
            buffer, host = self._ring.acquire(tuple(values.shape), values.dtype)
            row.buffers.append(buffer)
            current = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self._download):
                self._download.wait_stream(current)
                host.copy_(values, non_blocking=True)
                row.ready = torch.cuda.Event()
                row.ready.record(self._download)  # the download stream is in order: this covers the row's copies
            values.record_stream(self._download)
            row.cores[block.grid[2]] = host
        row.remaining -= 1
        if row.remaining == 0:
            del self._rows[key]
            self._in_flight.acquire()
            try:
                self._pool.submit(self._store, row)
            except BaseException:
                self._release(row)
                raise

    def _release(self, row: _Row) -> None:
        if self._ring is not None:
            for buffer in row.buffers:
                self._ring.release(buffer)
        self._in_flight.release()

    def _store(self, row: _Row) -> None:
        try:
            if row.ready is not None:
                row.ready.synchronize()
            if self._error is not None:
                return
            tic = time.perf_counter()
            cores = [core for core in row.cores if core is not None]
            values = cores[0] if len(cores) == 1 else torch.cat(cores, dim=-1)
            key = tuple(slice(a, b) for a, b in row.box)
            value = values if self.channels else values[0]
            self.sink[(slice(None), *key) if self.channels else key] = (
                value if isinstance(self.sink, Tensor) else value.numpy()
            )
            self.record("write_s", time.perf_counter() - tic)
            self.record("bytes_written", value.numel() * value.element_size())
        except BaseException as error:  # noqa: BLE001 (handed to the consumer thread, which re-raises it)
            self._fail(error)
        finally:
            self._release(row)

    def _fail(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def close(self) -> None:
        """Wait for every queued write; raise the first error any of them hit."""
        self._pool.shutdown(wait=True)
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        """Stop without writing pending cores and without raising: for a run that is already failing."""
        self._fail(RuntimeError("aborted"))
        self._pool.shutdown(wait=True, cancel_futures=True)
