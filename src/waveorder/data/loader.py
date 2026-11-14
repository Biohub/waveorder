"""The loader: batches of patches cut from whole fields of view, for one GPU or many.

:class:`PatchData` turns a :class:`~waveorder.data.FovIndex` into batches. Workers start by spawn, so a script that
makes loaders needs an ``if __name__ == "__main__":`` guard. With ``device=None`` the workers do all the work and
batches arrive in pinned host memory; with a CUDA ``device`` the workers only decode and this process does the rest
on the GPU (:class:`DeviceLoader`). Both give the same patches for a seed. In training every rank yields exactly
``len(loader)`` batches; other splits read each field once over all ranks and keep the last incomplete batch.
"""

import contextlib
import copy
import dataclasses
import functools
import itertools
import logging
import math
import os
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

import pyarrow as pa
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, default_collate, get_worker_info

from waveorder.data.index import FovIndex, FovRow, iter_rows
from waveorder.data.samplers import GridPatches, PatchMixer, RandomPatches, draw_seed, field_share, permutation
from waveorder.data.sources import PairSource, Recorded
from waveorder.errors import BackendUnavailableError, ParameterError, ReadError, ShapeError
from waveorder.io.staging import Prefetcher
from waveorder.nn import Sampling
from waveorder.nn.networks import Condition, Normalizer, Place, PlaneOffsets
from waveorder.nn.sampling import ZYX

_LOG = logging.getLogger(__name__)

type Patches = RandomPatches | GridPatches

type Prepare = Callable[[Tensor], Tensor]
"""``prepare(data) -> Tensor``: recorded data ``(C, Z, Y, X)`` to what the system measures; picklable."""


@dataclass(slots=True)
class Batch:
    """A batch of patches.

    Parameters
    ----------
    data : Tensor
        ``(B, C, Z, Y, X)``, prepared and normalized, float32; the stored dtype without ``prepare`` or input normalizer.
    target : Tensor
        ``(B, F, Zout, Y, X)``, through the target normalizer.
    condition : dict[str, Tensor]
        Condition name to its values ``(B, K)``, float32.
    position : list[str]
        Each patch's field: ``plate/row/column/fov``, with ``@t`` after time 0.
    origin : Tensor
        ``(B, 2)`` int64: ``(y, x)`` of each patch's top-left pixel in its field.
    """

    data: Tensor
    target: Tensor
    condition: dict[str, Tensor]
    position: list[str]
    origin: Tensor

    def to(self, device: torch.device | str, *, non_blocking: bool = True) -> Self:
        """Return the batch with its tensors on ``device``."""
        return self._map(lambda tensor: tensor.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> Self:
        """Return the batch in page-locked memory."""
        return self._map(Tensor.pin_memory)

    def _map(self, function: Callable[[Tensor], Tensor]) -> Self:
        return type(self)(
            data=function(self.data),
            target=function(self.target),
            condition={name: function(value) for name, value in self.condition.items()},
            position=self.position,
            origin=function(self.origin),
        )

    def __len__(self) -> int:
        return self.data.shape[0]


@dataclass(frozen=True, slots=True)
class Field:
    """One whole field, prepared and not normalized, for fitting normalizers and for evaluation.

    Parameters
    ----------
    data : Tensor
        ``(C, Z, Y, X)`` float32, through ``prepare``.
    target : Tensor
        ``(F, Zout, Y, X)`` float32, as recorded.
    condition : dict[str, Tensor]
        Condition name to its values for the whole field, ``(K,)``.
    row : FovRow
        The index row the field was read from.
    """

    data: Tensor
    target: Tensor
    condition: dict[str, Tensor]
    row: FovRow


def _shapes(rows: FovIndex, *, planes: bool = False) -> set:
    """Return the distinct ``(height, width)``, or ``(planes, height, width)``, of the rows."""
    names = ["planes", "height", "width"] if planes else ["height", "width"]
    unique = rows.table.select(names).group_by(names).aggregate([])
    return set(zip(*(unique[name].to_pylist() for name in names), strict=True))


def distributed_shard() -> tuple[int, int]:
    """Return ``(rank, world_size)`` of this process.

    From ``torch.distributed`` when initialized, else ``RANK`` (or ``LOCAL_RANK``) and ``WORLD_SIZE`` as ``torchrun``
    sets them, else ``(0, 1)``.

    Raises
    ------
    ParameterError
        If the environment's values are not ints with ``0 <= rank < world_size``.
    """
    import torch.distributed as distributed

    if distributed.is_available() and distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return distributed.get_rank(), distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK"))
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None or world_size is None:
        return 0, 1
    try:
        found = (int(rank), int(world_size))
    except ValueError:
        found = (-1, 0)
    if not 0 <= found[0] < found[1]:
        raise ParameterError(
            f"the environment gives RANK (or LOCAL_RANK) = {rank!r} and WORLD_SIZE = {world_size!r}, which is no rank "
            "among a number of processes; unset them, or pass rank and world_size to PatchData"
        )
    return found


@dataclass(slots=True)
class _Loaded:
    """One field as a worker holds it: read, prepared and normalized."""

    data: Tensor
    target: Tensor
    row: FovRow
    planes: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Recipe:
    """What is done to a field between the store and the patches; picklable."""

    channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    pixel_size: ZYX[float]
    source: PairSource
    conditions: tuple[Condition, ...]
    prepare: Prepare | None
    input_normalizer: Normalizer | None
    target_normalizer: Normalizer | None

    def load(self, row: FovRow, *, normalize: bool = True) -> _Loaded | None:
        """Read and prepare one field; ``None`` if it is blank."""
        data, target = self.source.read(row, channels=self.channels, target_channels=self.target_channels)
        return self.finish(row, data, target, normalize=normalize)

    def finish(self, row: FovRow, data: Tensor, target: Tensor, *, normalize: bool = True) -> _Loaded | None:
        """Prepare a field read as stored, on its device; ``None`` if it is blank."""
        with torch.no_grad():
            if _blank_field(data):
                return None
            if self.prepare is not None or self.input_normalizer is not None:
                data = data.to(torch.float32)
            if self.prepare is not None:
                data = self._prepared(data, row).to(torch.float32)
            if normalize and self.input_normalizer is not None:
                data = self.input_normalizer(data)
            if target.ndim != 4 or target.shape[2:] != data.shape[2:]:
                raise ShapeError(
                    f"the target of {row.name} has shape {tuple(target.shape)} but its data {tuple(data.shape)}: a "
                    "target is (F, Zout, Y, X) with the data's Y and X. Leave the field out of the index or "
                    "reconstruct it again"
                )
            target = target.to(torch.float32)
            if normalize and self.target_normalizer is not None:
                target = self.target_normalizer(target)
        return _Loaded(data=data, target=target, row=row, planes=data.shape[1])

    def _prepared(self, data: Tensor, row: FovRow) -> Tensor:
        assert self.prepare is not None
        prepared = self.prepare(data)
        if not isinstance(prepared, Tensor) or prepared.ndim != 4 or prepared.shape[1:] != data.shape[1:]:
            got = tuple(prepared.shape) if isinstance(prepared, Tensor) else type(prepared).__name__
            raise ShapeError(
                f"prepare returned {got} for the field {row.name} of shape {tuple(data.shape)}; it must return a "
                "tensor (C, Z, Y, X) with the same Z, Y and X"
            )
        return prepared

    def condition(
        self, loaded: _Loaded, origins: Sequence[tuple[int, int]], size: tuple[int, int]
    ) -> dict[str, Tensor]:
        """Return each condition's values ``(N, K)`` for the patches of ``size`` at ``origins``."""
        if not self.conditions:
            return {}
        field_shape = (loaded.data.shape[2], loaded.data.shape[3])
        sampling = Sampling(shape=(loaded.planes, *size), pixel_size=self.pixel_size)
        places = [
            Place(
                field_shape=field_shape,
                region=(slice(y, y + size[0]), slice(x, x + size[1])),
                sampling=sampling,
                attributes=loaded.row.attributes,
            )
            for y, x in origins
        ]
        return {
            condition.name: torch.stack([condition.values(place, None) for place in places])
            for condition in self.conditions
        }


@dataclass(slots=True)
class _Cut:
    """The patches of one field."""

    loaded: _Loaded
    origins: list[tuple[int, int]]
    condition: dict[str, Tensor]
    key: tuple[int, int]


@dataclass(eq=False, kw_only=True)
class _Stream(IterableDataset[Batch]):
    """The batches of one split for one rank; each loader worker iterates its own share of the fields."""

    recipe: _Recipe
    rows: pa.Table
    attributes: tuple[str, ...]
    split: str
    train: bool
    patches: Patches
    batch_size: int
    buffer_fields: int
    workers: int
    seed: int
    rank: int
    world_size: int
    stride: int
    cache_fields: int
    pad_planes: int | None
    shared_epoch: Tensor
    cache: OrderedDict[int, _Loaded | None] = field(default_factory=OrderedDict, repr=False)
    blank: Tensor = field(init=False, repr=False)
    per_field: dict[tuple[int, int], int] = field(init=False, repr=False)
    fewest: int = field(init=False, repr=False)
    packed_rows: Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        count = self.rows.num_rows
        if self.train and count < self.world_size:
            raise ParameterError(
                f"the split {self.split!r} has {count} fields, fewer than the {self.world_size} processes that "
                "share it in training; use fewer processes or more fields",
                path="world_size",
            )
        unique = self.rows.select(["height", "width"]).group_by(["height", "width"]).aggregate([])
        shapes = zip(unique["height"].to_pylist(), unique["width"].to_pylist(), strict=True)
        self.per_field = {shape: self.patches.count(shape, stride=self.stride) for shape in shapes}
        self.fewest = min(self.per_field.values(), default=0)
        self.blank = torch.zeros(self.workers, dtype=torch.int64)  # blank fields met, one slot per worker

    def __reduce__(self) -> tuple[object, tuple[dict[str, object], Tensor]]:
        """Pickle the rows as a shared-memory handle so spawned workers start at once and share one copy."""
        state = {name: value for name, value in self.__dict__.items() if name not in ("rows", "packed_rows")}
        if self.packed_rows is None:  # kept, so its memory stays mapped for every worker
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, self.rows.schema) as writer:
                writer.write_table(self.rows)
            self.packed_rows = torch.frombuffer(bytearray(sink.getvalue()), dtype=torch.uint8).share_memory_()
        return (_unpacked, (state, self.packed_rows))

    # --- which fields ------------------------------------------------------------------------------------------------

    def positions(self, worker: int, *, epoch: int) -> list[int]:
        """Return the row numbers ``worker`` of this rank reads in ``epoch``, in order."""
        return field_share(
            self.rows.num_rows,
            train=self.train,
            seed=self.seed,
            epoch=epoch,
            shard=(self.rank, self.world_size),
            worker=(worker, self.workers),
            fixed=self.cache_fields > 0,
        )

    def assignment(self, worker: int, *, epoch: int) -> Iterator[FovRow]:
        """Yield the rows ``worker`` of this rank reads in ``epoch``, in order, made a chunk at a time."""
        return iter_rows(self.rows, self.positions(worker, epoch=epoch), self.attributes)

    def _quota(self, worker: int) -> int:
        """Return the batches ``worker`` yields per training epoch, the same on every rank."""
        fields = len(range(worker, self.rows.num_rows // self.world_size, self.workers))
        return fields * self.fewest // self.batch_size

    def plan(self) -> dict[str, int]:
        """Return this rank's fields, patches and batches per epoch, from the rows alone."""
        if self.train:
            batches = sum(self._quota(worker) for worker in range(self.workers))
            return {
                "fields_per_epoch": self.rows.num_rows // self.world_size,
                "patches_per_epoch": batches * self.batch_size,
                "batches_per_epoch": batches,
            }
        fields = patches = batches = 0
        for worker in range(self.workers):
            positions = pa.array(self.positions(worker, epoch=0), pa.int64())  # typed: a rank may have no field
            taken = self.rows.select(["planes", "height", "width"]).take(positions)
            waiting: dict[int, int] = {}
            for planes, height, width in zip(*(taken[name].to_pylist() for name in taken.column_names), strict=True):
                key = planes if self.pad_planes is None else self.pad_planes
                waiting[key] = waiting.get(key, 0) + self.per_field[height, width]
                fields += 1
            patches += sum(waiting.values())
            batches += sum(-(-count // self.batch_size) for count in waiting.values())
        return {"fields_per_epoch": fields, "patches_per_epoch": patches, "batches_per_epoch": batches}

    def __len__(self) -> int:
        return self.plan()["batches_per_epoch"]

    # --- reading -----------------------------------------------------------------------------------------------------

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        yield from self.run(0 if info is None else info.id, epoch=int(self.shared_epoch[0]), load=self._load)

    def run(self, worker: int, *, epoch: int, load: Callable[[FovRow], _Loaded | None]) -> Iterator[Batch]:
        """Yield the batches ``worker`` of this rank gives in ``epoch``, taking fields from ``load``."""
        rows = self.assignment(worker, epoch=epoch)
        quota = self._quota(worker) if self.train else None
        blank: list[str] = []
        batches = self._epoch(rows, epoch=epoch, worker=worker, blank=blank, load=load)
        yield from batches if quota is None else itertools.islice(batches, quota)
        if blank:
            _LOG.warning(
                "skipped %d blank (all-zero) fields of split %s in epoch %d (rank %d, worker %d): %s",
                len(blank),
                self.split,
                epoch,
                self.rank,
                worker,
                ", ".join(blank[:8]) + (", ..." if len(blank) > 8 else ""),
            )

    def _epoch(
        self,
        rows: Iterable[FovRow],
        *,
        epoch: int,
        worker: int,
        blank: list[str],
        load: Callable[[FovRow], _Loaded | None],
    ) -> Iterator[Batch]:
        """Yield the batches of ``rows``; in training keep drawing patches from the last fields, without end."""
        generator = None
        if self.train:
            generator = torch.Generator().manual_seed(draw_seed(self.seed, "mix", epoch, self.rank, worker))
        mixer: PatchMixer[_Cut] = PatchMixer(batch_size=self.batch_size, fields=self.buffer_fields, generator=generator)

        def added(loaded: _Loaded, draw: int) -> Iterator[Batch]:
            cut = self._cut(loaded, epoch=epoch, draw=draw)
            return map(self._batch, mixer.add(cut, len(cut.origins), cut.key))

        held: deque[_Loaded] = deque(maxlen=self.buffer_fields)
        for row in rows:
            loaded = load(row)
            if loaded is None:
                blank.append(row.name)
                self.blank[worker] += 1
                continue
            held.append(loaded)
            yield from added(loaded, 0)
        yield from map(self._batch, mixer.flush(incomplete=not self.train))
        if not self.train:
            return
        if not held:
            raise ReadError(
                f"all {len(blank)} fields of split {self.split!r} given to worker {worker} of rank {self.rank} are "
                "blank (all zero), so it has no batch to give; leave the blank fields out of the index"
            )
        draw = 0
        while True:  # blank fields left this worker short of its batches: draw again from the fields it holds
            draw += 1
            for loaded in list(held):
                yield from added(loaded, draw)
            yield from map(self._batch, mixer.flush(incomplete=False))

    def _batch(self, patches: list[tuple[_Cut, int]]) -> Batch:
        """Stack patches into a batch on the fields' device."""
        height, width = self.patches.size
        data, targets, origins = [], [], []
        for cut, number in patches:
            y, x = cut.origins[number]
            origins.append((y, x))
            data.append(_padded(cut.loaded.data[:, :, y : y + height, x : x + width], self.pad_planes))
            targets.append(cut.loaded.target[:, :, y : y + height, x : x + width])
        device = data[0].device
        names = patches[0][0].condition
        condition = {name: torch.stack([cut.condition[name][number] for cut, number in patches]) for name in names}
        return Batch(
            data=default_collate(data),
            target=default_collate(targets),
            condition={name: value.to(device) for name, value in condition.items()},
            position=[cut.loaded.row.name for cut, _ in patches],
            origin=torch.tensor(origins, dtype=torch.int64, device=device).reshape(-1, 2),
        )

    def _load(self, row: FovRow) -> _Loaded | None:
        """Read a field, or take it from the cache."""
        if not self.cache_fields:
            return self.recipe.load(row)
        key = row.field_id
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        loaded = self.recipe.load(row)
        self.cache[key] = loaded
        while len(self.cache) > self.cache_fields:
            self.cache.popitem(last=False)
        return loaded

    def _cut(self, loaded: _Loaded, *, epoch: int, draw: int) -> _Cut:
        """Place the patches of a field and compute their conditions."""
        shape = (loaded.data.shape[2], loaded.data.shape[3])
        generator = torch.Generator().manual_seed(draw_seed(self.seed, "origins", epoch, loaded.row.field_id, draw))
        origins = self.patches.origins(shape, generator=generator, stride=self.stride).tolist()
        return _Cut(
            loaded=loaded,
            origins=[(y, x) for y, x in origins],
            condition=self.recipe.condition(loaded, origins, self.patches.size),
            key=(loaded.data.shape[1] if self.pad_planes is None else self.pad_planes, loaded.target.shape[1]),
        )


def _unpacked(state: dict[str, object], packed_rows: Tensor) -> _Stream:
    """Rebuild a ``_Stream`` in a worker from ``_Stream.__reduce__``, reading the rows in place."""
    stream = _Stream.__new__(_Stream)
    stream.__dict__.update(state)
    shared = pa.foreign_buffer(packed_rows.data_ptr(), packed_rows.numel(), base=packed_rows)
    stream.rows = pa.ipc.open_stream(shared).read_all()
    stream.packed_rows = packed_rows
    return stream


# --- the device pipeline ----------------------------------------------------------------------------------------------

_PREFETCH = 2
"""Batches (fields, with a device) each loader worker prepares ahead."""

_UPLOADED = 2
"""Fields a device loader uploads ahead of the one it prepares."""

_PAGE = 4096
"""Staging buffers start on page boundaries."""


class _Ring:
    """Page-locked staging buffers in shared memory: workers write fields, this process uploads.

    ``full[slot]`` is the generation (loader iteration) whose field the slot holds, 0 when free.
    """

    def __init__(self, slots: int, data_bytes: int, target_bytes: int) -> None:
        self.target_offset = -(-data_bytes // _PAGE) * _PAGE
        self.slot_bytes = max(_PAGE, self.target_offset + -(-target_bytes // _PAGE) * _PAGE)
        self.buffer = torch.empty(slots * self.slot_bytes, dtype=torch.uint8).share_memory_()
        self.full = torch.zeros(slots, dtype=torch.int64).share_memory_()
        self.generation = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.pinned = False

    def pin(self) -> None:
        """Page-lock the buffers; raise :class:`BackendUnavailableError` if CUDA cannot."""
        try:
            torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(self.buffer.data_ptr(), self.buffer.numel(), 0))
        except RuntimeError as error:
            raise BackendUnavailableError(
                f"CUDA could not page-lock the {self.buffer.numel() / 1e9:.2f} GB of staging buffers of a device "
                f"loader ({error}); use fewer workers, or device=None"
            ) from error
        self.pinned = True

    def unpin(self) -> None:
        if self.pinned:
            torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr()))
            self.pinned = False

    def view(self, slot: int, *, target: bool, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        """Return the data (or the target) part of a slot as a ``shape``, ``dtype`` tensor."""
        start = slot * self.slot_bytes + (self.target_offset if target else 0)
        room = (self.slot_bytes - self.target_offset) if target else self.target_offset
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes > room:
            raise ShapeError(
                f"a field's {'target' if target else 'data'} {tuple(shape)} {dtype} takes {nbytes} bytes, more than "
                f"the {room} its staging buffer has: the index says another dtype or shape than the plate holds; "
                "build the index again"
            )
        return self.buffer[start : start + nbytes].view(dtype).view(shape)


@dataclass(frozen=True, slots=True)
class _Staged:
    """A field written into a staging slot: slot, iteration, row, and data and target shapes and dtypes."""

    slot: int
    generation: int
    row: FovRow
    data: tuple[tuple[int, ...], torch.dtype]
    target: tuple[tuple[int, ...], torch.dtype]


@dataclass(eq=False, kw_only=True)
class _Reader(IterableDataset[_Staged]):
    """Device-loader worker: read the rank's fields, in order, into the staging buffers.

    Reader ``r`` takes every ``readers``-th field from the ``r``-th, so the round-robin over workers keeps the order.
    """

    stream: _Stream
    ring: _Ring
    readers: int
    slots_per_reader: int
    cached: Tensor

    def __iter__(self) -> Iterator[_Staged]:
        info = get_worker_info()
        return self.read(0 if info is None else info.id)

    def read(self, reader: int) -> Iterator[_Staged]:
        """Yield the fields of ``reader`` in this iteration, each written into a free slot."""
        generation = int(self.ring.generation[0])
        positions = self.stream.positions(0, epoch=int(self.stream.shared_epoch[0]))
        if bool(self.cached.any()):
            positions = [position for position in positions if not self.cached[position]]
        recipe = self.stream.recipe
        rows = iter_rows(self.stream.rows, positions[reader :: self.readers], self.stream.attributes)
        for number, row in enumerate(rows):
            data, target = recipe.source.read(row, channels=recipe.channels, target_channels=recipe.target_channels)
            slot = reader * self.slots_per_reader + number % self.slots_per_reader
            while int(self.ring.full[slot]) == generation:  # its field is not uploaded yet
                if int(self.ring.generation[0]) != generation:
                    return
                time.sleep(0.0005)
            if int(self.ring.generation[0]) != generation:  # the loader started another iteration
                return
            self.ring.view(slot, target=False, shape=tuple(data.shape), dtype=data.dtype).copy_(data)
            self.ring.view(slot, target=True, shape=tuple(target.shape), dtype=target.dtype).copy_(target)
            self.ring.full[slot] = generation
            yield _Staged(slot, generation, row, (tuple(data.shape), data.dtype), (tuple(target.shape), target.dtype))


class DeviceLoader:
    """The loader of one split of a :class:`PatchData` with a ``device``: batches on the device.

    Workers (or the upload thread with ``workers=0``) decode whole fields into page-locked staging buffers; a thread
    uploads up to 2 fields ahead on a side CUDA stream. On the device each field is prepared, normalized, cut and
    batched as the CPU pipeline does. Device memory holds the shuffle buffer, the field being prepared, two uploaded
    fields and ``cache_fields`` cached fields; pinned host memory holds ``max(1, workers) * 3`` stored fields.

    Attributes
    ----------
    device : torch.device
        Where the batches are.
    stats : dict[str, int]
        Of the last iteration: ``fields`` uploaded, ``cached`` fields reused and ``blank`` fields.
    """

    def __init__(
        self, stream: _Stream, *, device: torch.device, workers: int, cache_fields: int, meta: Mapping[str, object]
    ) -> None:
        self.device = device
        self.stream = stream
        self.stats: dict[str, int] = {}
        self._recipe = dataclasses.replace(
            stream.recipe,
            input_normalizer=_on(stream.recipe.input_normalizer, device),
            target_normalizer=_on(stream.recipe.target_normalizer, device),
        )
        rows = stream.rows
        readers, per_reader = max(1, workers), _PREFETCH + 1
        recipe = stream.recipe
        staged = _staged_bytes(rows, meta, channels=len(recipe.channels), target_channels=len(recipe.target_channels))
        self._ring = _Ring(readers * per_reader, *staged)
        self._ring.pin()
        self._unpin = weakref.finalize(self, self._ring.unpin)
        self._cache_fields = cache_fields
        self._cache: dict[int, _Loaded | None] = {}
        self._index = {key: number for number, key in enumerate(rows["field_id"].to_pylist())}
        self._cached = torch.zeros(rows.num_rows, dtype=torch.int8)  # what this process keeps, as it fills the cache
        self._snapshot = torch.zeros(rows.num_rows, dtype=torch.int8).share_memory_()  # what the readers skip
        reading = dataclasses.replace(  # the readers only read: no user callable or module is sent to them
            stream.recipe, conditions=(), prepare=None, input_normalizer=None, target_normalizer=None
        )
        self._reader = _Reader(
            stream=dataclasses.replace(stream, recipe=reading),
            ring=self._ring,
            readers=readers,
            slots_per_reader=per_reader,
            cached=self._snapshot,
        )
        self._loader: DataLoader[_Staged] | None = None
        if workers:
            self._loader = DataLoader(
                self._reader,
                batch_size=None,
                num_workers=workers,
                multiprocessing_context="spawn",
                persistent_workers=True,
                prefetch_factor=_PREFETCH,
                worker_init_fn=_one_thread,
            )
        self._prefetch = Prefetcher(device, workers=1, ahead=_UPLOADED)  # one thread takes the readers' fields in order
        self._fields: Generator[tuple[FovRow, tuple[Tensor, ...]]] | None = None  # the uploads of the open iteration

    def __len__(self) -> int:
        return len(self.stream)

    def __iter__(self) -> Iterator[Batch]:
        self._stop()  # an iteration left unfinished: stop its uploads before its readers restart
        epoch = int(self.stream.shared_epoch[0])
        self._snapshot.copy_(self._cached)
        self._ring.generation += 1
        rows = [row for row in self.stream.assignment(0, epoch=epoch) if row.field_id not in self._cache]
        stage = functools.partial(self._staged, fields=self._read(int(self._ring.generation[0])))
        fields = self._fields = self._prefetch(stage, rows)
        stats = self.stats = {"fields": 0, "cached": 0, "blank": 0}
        blank = int(self.stream.blank[0])
        try:
            with torch.cuda.device(self.device):
                load = functools.partial(self._field, fields=fields, stats=stats)
                yield from self.stream.run(0, epoch=epoch, load=load)
        finally:
            if self._fields is fields:  # else a later iteration stopped it already
                self._stop()
            stats["blank"] = int(self.stream.blank[0]) - blank

    def close(self) -> None:
        """Stop the uploads and the workers, and release the page-locked staging buffers."""
        self._stop()
        if self._loader is not None:
            _shutdown(self._loader)
        self._unpin()

    def _stop(self) -> None:
        """Make the readers abandon the open iteration and stop its uploads."""
        if self._fields is not None:
            self._ring.generation += 1
            self._fields.close()
            self._fields = None

    def _read(self, generation: int) -> Iterator[_Staged]:
        """Yield the fields staged in iteration ``generation``, in order."""
        staged_fields = iter(self._loader) if self._loader is not None else self._reader.read(0)
        yield from (staged for staged in staged_fields if staged.generation == generation)  # not an abandoned one's

    @contextlib.contextmanager
    def _staged(self, row: FovRow, *, fields: Iterator[_Staged]) -> Generator[tuple[Tensor, ...]]:
        """Hold the next staged field as views of its buffer; free the buffer once uploaded."""
        staged = next(fields, None)
        assert staged is not None, "the device loader's readers ended before the fields of the epoch did"
        assert staged.row.field_id == row.field_id, f"the device loader read {staged.row.name} where {row.name} was due"
        parts = [(False, *staged.data), (True, *staged.target)]
        yield tuple(self._ring.view(staged.slot, target=part, shape=shape, dtype=dtype) for part, shape, dtype in parts)
        self._ring.full[staged.slot] = 0

    def _field(
        self, row: FovRow, *, fields: Iterator[tuple[FovRow, tuple[Tensor, ...]]], stats: dict[str, int]
    ) -> _Loaded | None:
        """Return the next field, prepared on the device; ``None`` if it is blank."""
        key = row.field_id
        if key in self._cache:
            stats["cached"] += 1
            return self._cache[key]
        item = next(fields, None)
        assert item is not None, "the device loader's readers ended before the fields of the epoch did"
        _, (data, target) = item
        stats["fields"] += 1
        loaded = self._recipe.finish(row, data, target)
        if len(self._cache) < self._cache_fields:
            self._cache[key] = loaded
            self._cached[self._index[key]] = 1
        return loaded


def _shutdown(loader: DataLoader) -> None:
    """Stop the persistent workers of a ``DataLoader``."""
    running = getattr(loader, "_iterator", None)
    shutdown = getattr(running, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def _staged_bytes(
    rows: pa.Table, meta: Mapping[str, object], *, channels: int, target_channels: int
) -> tuple[int, int]:
    """Return the most bytes a field's data and target take as stored, from the index alone."""
    pixels = max((h * w for h, w in zip(rows["height"].to_pylist(), rows["width"].to_pylist(), strict=True)), default=0)
    planes = max(rows["planes"].to_pylist(), default=0)
    data_bytes = channels * planes * pixels * _dtype(str(meta.get("dtype", "float32"))).itemsize
    target_planes = meta.get("target_planes", 1)
    depth = target_planes if isinstance(target_planes, int) else 1
    itemsize = _dtype(str(meta.get("target_dtype", "float32"))).itemsize
    return data_bytes, target_channels * depth * pixels * itemsize


def _dtype(name: str) -> torch.dtype:
    """Return the torch dtype named ``name``, or raise."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ParameterError(f"the index records the dtype {name!r}, which torch has no dtype for", path="meta.dtype")
    return dtype


def _device(value: torch.device | str | None) -> torch.device | None:
    """Return the CUDA device a ``PatchData`` works on, or ``None`` for the loader workers."""
    if value is None or value == "cpu":
        return None
    if value == "auto":
        if not torch.cuda.is_available():
            return None
        local = os.environ.get("LOCAL_RANK", "")
        index = int(local) if local.isdigit() and int(local) < torch.cuda.device_count() else None
        return torch.device("cuda", torch.cuda.current_device() if index is None else index)
    try:
        device = torch.device(value)
    except RuntimeError:
        device = None
    if device is None or device.type not in ("cpu", "cuda"):
        raise ParameterError(
            f"PatchData device must be None, 'cpu', 'auto', 'cuda' or a CUDA device such as 'cuda:1', got {value!r}",
            path="device",
        )
    if device.type == "cpu":
        return None
    if not torch.cuda.is_available():
        raise BackendUnavailableError(f"PatchData device={value!r}, but CUDA is not available here; use device=None")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


def _blank_field(data: Tensor) -> bool:
    """Whether a field is all zero; on the CPU a sparse look first spares most fields a full pass."""
    if data.is_cuda:  # CUDA has no ``any`` of uint16; int16 has the same zeros
        return not bool((data.view(torch.int16) if data.dtype == torch.uint16 else data).any())
    return not data[..., ::61, ::61].any() and not data.any()


def _on(normalizer: Normalizer | None, device: torch.device) -> Normalizer | None:
    """Return a copy of a normalizer on ``device``."""
    return None if normalizer is None else copy.deepcopy(normalizer).to(device)


def _padded(data: Tensor, planes: int | None) -> Tensor:
    """Zero-pad ``data`` ``(..., Z, Y, X)`` after its last plane up to ``planes``."""
    if planes is None or data.shape[-3] == planes:
        return data
    if data.shape[-3] > planes:
        raise ShapeError(
            f"a stack of {data.shape[-3]} planes cannot be padded to {planes}: give PlaneOffsets at least as many "
            "planes as the deepest field has"
        )
    return torch.nn.functional.pad(data, (0, 0, 0, 0, 0, planes - data.shape[-3]))


def _one_thread(worker: int) -> None:
    """Limit a loader worker to one thread."""
    del worker
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLOSC_NTHREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    import zarr
    from numcodecs import blosc

    blosc.set_nthreads(1)
    zarr.config.set({"threading.max_workers": 1})


def _expect_at_least(name: str, value: int | None, minimum: int) -> None:
    if value is not None and value < minimum:
        raise ParameterError(f"PatchData {name} must be at least {minimum}, got {value!r}", path=name)


def _expect_shard(rank: int | None, world_size: int | None) -> tuple[int, int] | None:
    """Return ``(rank, world_size)`` as given, or ``None`` when both are left to be found."""
    if rank is None and world_size is None:
        return None
    if rank is None or world_size is None:
        raise ParameterError(
            f"PatchData takes rank and world_size together, got rank={rank!r} and world_size={world_size!r}; leave "
            "both out to find them from torch.distributed or the environment",
            path="rank",
        )
    _expect_at_least("world_size", world_size, 1)
    _expect_at_least("rank", rank, 0)
    if rank >= world_size:
        raise ParameterError(f"PatchData rank must be below world_size={world_size}, got {rank}", path="rank")
    return rank, world_size


class PatchData:
    """Batches of patches from the fields of an index, sharded over processes and workers.

    Parameters
    ----------
    index : FovIndex
        The fields, their recorded targets and their splits.
    patches : RandomPatches or GridPatches
        Patch placement in training; other splits use the grid of the same patch size.
    batch_size, workers : int
        Patches per batch; loader processes (0 reads in the calling process).
    buffer_fields : int
        Fields whose patches are mixed in training; each worker holds about that many.
    source : PairSource, optional
        Where data and targets are read from; :class:`~waveorder.data.Recorded` by default.
    conditions : Sequence[Condition]
        Conditions computed for every patch from its place in the field and the row's attributes.
    prepare : callable, optional
        ``prepare(data) -> Tensor``: recorded field ``(C, Z, Y, X)`` to what the system measures. Runs once per
        field (on the device with ``device``); must be picklable.
    input_normalizer, target_normalizer : Normalizer, optional
        Applied to each whole prepared field (target field) before patches are cut.
    seed : int
        Seed of the field order, the patch origins and the mixing.
    rank, world_size : int, optional
        This process among ``world_size``; both ``None`` finds them with :func:`distributed_shard`.
    stride : int
        Random patch origins are multiples of this; give the network's stride.
    limit_fields : int, optional
        Use only that many fields per split, chosen by the seed: for smoke runs.
    cache_fields : int
        Decoded, prepared fields kept in each worker (on the device, per rank, with ``device``); with a cache each
        worker keeps the same fields every epoch, so ``limit_fields / (world_size * workers)`` should fit.
    device : torch.device, str or None
        ``None`` or ``"cpu"``: work in the loader workers. A CUDA device, ``"cuda"`` or ``"auto"``
        (``cuda:$LOCAL_RANK`` if available, else ``None``): work on that device; the loader is a :class:`DeviceLoader`.

    Raises
    ------
    ParameterError
        If a count is out of range, conditions share a name, only one of ``rank`` and ``world_size`` is given, or
        ``device`` is not the CPU, ``"auto"`` or a CUDA device.
    BackendUnavailableError
        If ``device`` names a CUDA device and CUDA is not available.
    ShapeError
        If a field of the index is smaller than a patch.
    """

    def __init__(  # noqa: PLR0913 (the design's arguments: one object describes a loader)
        self,
        index: FovIndex,
        *,
        patches: Patches,
        batch_size: int = 16,
        buffer_fields: int = 4,
        workers: int = 8,
        source: PairSource | None = None,
        conditions: Sequence[Condition] = (),
        prepare: Prepare | None = None,
        input_normalizer: Normalizer | None = None,
        target_normalizer: Normalizer | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        stride: int = 1,
        limit_fields: int | None = None,
        cache_fields: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        self._arguments: dict[str, object] = {name: value for name, value in locals().items() if name != "self"}
        del self._arguments["index"]
        for name, value, minimum in (
            ("batch_size", batch_size, 1),
            ("buffer_fields", buffer_fields, 1),
            ("workers", workers, 0),
            ("stride", stride, 1),
            ("cache_fields", cache_fields, 0),
            ("limit_fields", limit_fields, 1),
        ):
            _expect_at_least(name, value, minimum)
        names = [condition.name for condition in conditions]
        if len(set(names)) != len(names):
            raise ParameterError(f"PatchData conditions must have distinct names, got {names}", path="conditions")
        self.index = index
        self.source = Recorded() if source is None else source
        self.patches = patches
        self.batch_size = batch_size
        self.buffer_fields = buffer_fields
        self.workers = workers
        self.conditions = tuple(conditions)
        self.prepare = prepare
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        self.seed = seed
        self.stride = stride
        self.limit_fields = limit_fields
        self.cache_fields = cache_fields
        self.device = _device(device)
        self.channels, self.pixel_size = tuple(index.channels), index.pixel_size
        self._shard = _expect_shard(rank, world_size)
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()  # read by the workers of every loader
        self._skipped = 0  # blank fields met by fields()
        self._loaders: dict[tuple[str, bool], DataLoader[Batch] | DeviceLoader] = {}
        self._shards: dict[tuple[int, int, str | None], Self] = {}
        for split in self._splits():  # a field smaller than a patch is refused now, not in a worker
            for shape in _shapes(self._rows(split)):
                self.patches.count(shape, stride=stride)

    # --- shards and epochs -------------------------------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Rank of this process among :attr:`world_size`: as given, else found."""
        return self._resolved_shard()[0]

    @property
    def world_size(self) -> int:
        """Number of processes sharing the fields: as given, else found."""
        return self._resolved_shard()[1]

    def _resolved_shard(self) -> tuple[int, int]:
        return self._shard or distributed_shard()

    @property
    def epoch(self) -> int:
        """The epoch the next iteration of a loader reads; see :meth:`set_epoch`."""
        return int(self._epoch[0])

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the next iteration of every loader of this object, its copies and shards reads.

        Shared memory, so persistent workers see it; without it every epoch reads the same order.
        """
        _expect_at_least("epoch", epoch, 0)
        self._epoch[0] = epoch

    @property
    def skipped(self) -> int:
        """Blank (all-zero) fields met so far by loaders, :meth:`fields` and shards, once per epoch each."""
        counted = 0
        for loader in self._loaders.values():
            stream = loader.stream if isinstance(loader, DeviceLoader) else loader.dataset
            assert isinstance(stream, _Stream)
            counted += int(stream.blank.sum())
        return self._skipped + counted + sum(shard.skipped for shard in self._shards.values())

    def replace(self, **changes: object) -> Self:
        """Return a copy with some constructor arguments (``index`` included) changed, sharing the epoch."""
        arguments = {"index": self.index, **self._arguments, **changes}
        changed = type(self)(arguments.pop("index"), **arguments)  # ty: ignore[invalid-argument-type]
        changed._epoch = self._epoch  # noqa: SLF001 (a copy of this class)
        return changed

    def shard(self, *, rank: int, world_size: int, device: torch.device | str | None = None) -> Self:
        """Return the (cached) copy that reads the share of ``rank`` among ``world_size``.

        ``device`` replaces this object's ``device`` in the copy; ``None`` keeps it.
        """
        chosen = self._arguments["device"] if device is None else device
        key = (rank, world_size, None if chosen is None else str(chosen))
        if key not in self._shards:
            self._shards[key] = self.replace(rank=rank, world_size=world_size, device=chosen)
        return self._shards[key]

    # --- loading -----------------------------------------------------------------------------------------------------

    def loader(
        self, split: str, *, epoch: int | None = None, train: bool | None = None
    ) -> DataLoader[Batch] | DeviceLoader:
        """Return the loader of a split, made once and kept with its workers.

        ``epoch`` calls :meth:`set_epoch` first. ``train`` (default ``split == "train"``) shuffles fields each epoch,
        places patches by ``patches``, mixes them and drops the incomplete last batch. ``len(loader)`` is this rank's
        batches per epoch: the same on every rank in training.

        Raises
        ------
        ParameterError
            If the index has no such split, or in training it has fewer fields than ``world_size``.
        """
        if epoch is not None:
            self.set_epoch(epoch)
        train = split == "train" if train is None else train

        if (split, train) not in self._loaders:
            if self.device is not None:
                self._loaders[split, train] = DeviceLoader(
                    self._stream(split, train=train),
                    device=self.device,
                    workers=self.workers,
                    cache_fields=self.cache_fields,
                    meta=self.index.meta,
                )
                return self._loaders[split, train]
            stream = self._stream(split, train=train)
            stream.blank.share_memory_()  # each worker counts its blank fields in its own slot
            self._loaders[split, train] = DataLoader(
                stream,
                batch_size=None,
                num_workers=self.workers,
                multiprocessing_context="spawn" if self.workers else None,
                persistent_workers=self.workers > 0,
                prefetch_factor=_PREFETCH if self.workers else None,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=_one_thread if self.workers else None,
            )
        return self._loaders[split, train]

    def batches(self, split: str, *, epoch: int | None = None, train: bool | None = None) -> Iterator[Batch]:
        """Iterate one epoch of a split: ``iter(self.loader(split, epoch=epoch, train=train))``."""
        return iter(self.loader(split, epoch=epoch, train=train))

    def close(self) -> None:
        """Stop the workers of every loader of this object and its shards, and forget the loaders."""
        for loader in self._loaders.values():
            if isinstance(loader, DeviceLoader):
                loader.close()
            else:
                _shutdown(loader)
        self._loaders.clear()
        for shard in self._shards.values():
            shard.close()

    def fields(self, split: str, *, limit: int | None = None) -> Iterator[Field]:
        """Yield the whole fields of a split, prepared and not normalized, read in this process.

        Every field of the split (after ``limit_fields``) whatever the rank; ``limit`` reads that many, evenly spaced.
        Blank fields are skipped and counted.
        """
        _expect_at_least("limit", limit, 1)
        rows = self._rows(split)
        numbers = range(len(rows))
        if limit is not None and limit < len(rows):
            numbers = [number * len(rows) // limit for number in range(limit)]
        recipe = self._recipe()
        for row in rows.take(numbers):
            loaded = recipe.load(row, normalize=False)
            if loaded is None:
                self._skipped += 1
                _LOG.warning("skipped the blank (all-zero) field %s", row.name)
                continue
            height, width = loaded.data.shape[2:]
            condition = recipe.condition(loaded, [(0, 0)], (height, width))
            conditions = {name: value[0] for name, value in condition.items()}
            yield Field(data=loaded.data.to(torch.float32), target=loaded.target, condition=conditions, row=row)

    def assignment(self, split: str, *, worker: int = 0, train: bool | None = None) -> tuple[FovRow, ...]:
        """Return the fields ``worker`` of this rank reads in the current epoch, in order.

        With ``device`` worker 0 has all the rank's fields.
        """
        train = split == "train" if train is None else train
        return tuple(self._stream(split, train=train).assignment(worker, epoch=self.epoch))

    def describe(self) -> dict[str, object]:
        """Return what the loaders will read, from the index alone.

        Returns
        -------
        dict[str, object]
            ``splits`` (per split ``fields``, ``fields_per_epoch``, ``patches_per_epoch``, ``batches_per_epoch``),
            ``planes``, byte estimates (``bytes_per_field``, per-worker buffer and cache, per-rank device and pinned
            with a device) and the loader settings.
        """
        splits: dict[str, dict[str, int]] = {}
        shapes: set[tuple[int, int, int]] = set()
        for split in self._splits():
            rows = self._rows(split)
            shapes |= _shapes(rows, planes=True)
            splits[split] = {"fields": len(rows), **self._stream(split, train=split == "train").plan()}
        bytes_per_field = max((self._bytes(shape) for shape in shapes), default=0)
        on_device: dict[str, object] = {"device": None if self.device is None else str(self.device)}
        if self.device is not None:
            stored = max((self._bytes(shape, stored=True) for shape in shapes), default=0)
            on_device["device_bytes_per_rank"] = (
                bytes_per_field * (self.buffer_fields + 1 + self.cache_fields) + stored * _UPLOADED
            )
            on_device["pinned_bytes_per_rank"] = stored * max(1, self.workers) * (_PREFETCH + 1)
        return {
            "splits": splits,
            "planes": sorted({planes for planes, _, _ in shapes}),
            "bytes_per_field": bytes_per_field,
            "buffer_bytes_per_worker": bytes_per_field * (self.buffer_fields + 1),
            "cache_bytes_per_worker": bytes_per_field * self.cache_fields,
            **on_device,
            "rank": self.rank,
            "world_size": self.world_size,
            "workers": self.workers,
            "batch_size": self.batch_size,
            "patch_size": list(self.patches.size),
            "channels": list(self.channels),
            "target_channels": list(self.index.target_channels),
            "pixel_size": list(self.pixel_size),
        }

    # --- internals ---------------------------------------------------------------------------------------------------

    def _splits(self) -> list[str]:
        return list(self.index.counts)

    def _rows(self, split: str) -> FovIndex:
        """Return the rows of a split after ``limit_fields``, in the index's order."""
        rows = self.index.split(split)
        if self.limit_fields is not None and self.limit_fields < len(rows):
            kept = sorted(permutation(len(rows), self.seed, "limit")[: self.limit_fields])
            rows = replace(rows, table=rows.table.take(pa.array(kept, pa.int64())))
        return rows

    def _bytes(self, shape: tuple[int, int, int], *, stored: bool = False) -> int:
        """Estimate the bytes of a ``(planes, height, width)`` field with its target, prepared or as stored."""
        planes, height, width = shape
        meta = self.index.meta
        floats = self.prepare is not None or self.input_normalizer is not None
        digits = "".join(character for character in str(meta.get("dtype", "")) if character.isdigit())
        itemsize = 4 if (floats and not stored) or not digits else int(digits) // 8
        target_planes = meta.get("target_planes", 1)
        target = len(self.index.target_channels) * (target_planes if isinstance(target_planes, int) else 1) * 4
        return height * width * (max(1, len(self.channels)) * planes * itemsize + target)

    def _pad_planes(self) -> int | None:
        """Return the plane count of a ``PlaneOffsets`` condition, if any: stacks are zero-padded to it."""
        return next((condition.planes for condition in self.conditions if isinstance(condition, PlaneOffsets)), None)

    def _recipe(self) -> _Recipe:
        """Return the field recipe, with normalizers copied to the CPU."""
        return _Recipe(
            channels=self.channels,
            target_channels=tuple(self.index.target_channels),
            pixel_size=self.pixel_size,
            source=self.source,
            conditions=self.conditions,
            prepare=self.prepare,
            input_normalizer=_on(self.input_normalizer, torch.device("cpu")),
            target_normalizer=_on(self.target_normalizer, torch.device("cpu")),
        )

    def _stream(self, split: str, *, train: bool) -> _Stream:
        rank, world_size = self._resolved_shard()
        patches = self.patches if train or isinstance(self.patches, GridPatches) else GridPatches(self.patches.size)
        return _Stream(
            recipe=self._recipe(),
            rows=self._rows(split).table,
            attributes=tuple(self.index.attributes),
            split=split,
            train=train,
            patches=patches,
            batch_size=self.batch_size,
            buffer_fields=self.buffer_fields,
            workers=1 if self.device is not None else max(1, self.workers),
            seed=self.seed,
            rank=rank,
            world_size=world_size,
            stride=self.stride,
            cache_fields=self.cache_fields,
            pad_planes=self._pad_planes(),
            shared_epoch=self._epoch,
        )
