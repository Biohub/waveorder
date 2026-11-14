"""Sampling: where patches sit in a field, which fields each process reads, and how patches are mixed into batches.

A patch strategy's ``origins`` returns the top-left ``(y, x)`` in pixels of every patch of a field, never past its
edge; ``count`` says how many. Every random choice is seeded by :func:`draw_seed`, so a seed gives the same epoch
anywhere.
"""

import hashlib
from collections.abc import Hashable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from waveorder.errors import ParameterError, ShapeError


def draw_seed(*parts: object) -> int:
    """Return a 63-bit seed from a hash of ``parts``."""
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def permutation(count: int, *parts: object) -> list[int]:
    """Return a permutation of ``range(count)`` that depends on ``parts`` only."""
    return torch.randperm(count, generator=torch.Generator().manual_seed(draw_seed(*parts))).tolist()


def field_share(  # noqa: PLR0913 (what an epoch's order depends on)
    count: int, *, train: bool, seed: int, epoch: int, shard: tuple[int, int], worker: tuple[int, int], fixed: bool
) -> list[int]:
    """Return the fields of ``count`` that worker ``(number, workers)`` of ``shard = (rank, world_size)`` reads.

    In training the fields are permuted from ``(seed, epoch)``, a tail of fewer than ``world_size`` is dropped, and
    rank then worker take every ``world_size``-th (``workers``-th): no field is read twice in an epoch. With ``fixed``
    a worker keeps the same fields every epoch, in a new order, so a cache serves them. Without ``train`` the fields
    are taken in order, each once over all ranks and workers.
    """
    (rank, world_size), (number, workers) = shard, worker
    if not train:
        return list(range(count))[rank::world_size][number::workers]
    order = permutation(count, seed, "fields", 0 if fixed else epoch)
    mine = order[: count // world_size * world_size][rank::world_size][number::workers]
    if fixed:
        mine = [mine[index] for index in permutation(len(mine), seed, "order", epoch, rank, number)]
    return mine


class PatchMixer[T]:
    """Mix the patches of several fields into groups of ``batch_size``, one key per group.

    With a ``generator`` (training) a random ``1 / fields`` of the pool is drawn per added field and the pool holds
    patches of at most ``fields`` fields. Without one, patches are grouped in the order they come.
    """

    def __init__(self, *, batch_size: int, fields: int, generator: torch.Generator | None) -> None:
        self.batch_size, self.fields, self.generator = batch_size, fields, generator
        self.added = 0
        self.pool: list[tuple[int, Hashable, T, int]] = []
        self.pending: dict[Hashable, list[tuple[T, int]]] = {}

    def add(self, field: T, count: int, key: Hashable) -> Iterator[list[tuple[T, int]]]:
        """Add the ``count`` patches of a field; yield the ``(field, patch number)`` groups this fills.

        Patches are grouped only with those of the same ``key`` (the data and target plane counts).
        """
        self.added += 1
        patches = [(self.added, key, field, number) for number in range(count)]
        if self.generator is None:
            yield from self._queue(patches)
            return
        self.pool.extend(patches)
        order = torch.randperm(len(self.pool), generator=self.generator).tolist()
        taken = len(self.pool) // self.fields
        oldest = self.added - self.fields + 1
        drawn = [self.pool[number] for number in order[:taken]]
        kept = [self.pool[number] for number in order[taken:]]
        drawn.extend(entry for entry in kept if entry[0] < oldest)
        self.pool = [entry for entry in kept if entry[0] >= oldest]
        yield from self._queue(drawn)

    def flush(self, *, incomplete: bool) -> Iterator[list[tuple[T, int]]]:
        """Yield the groups the remaining patches fill and, with ``incomplete``, the groups left unfilled."""
        if self.pool:
            order = torch.randperm(len(self.pool), generator=self.generator).tolist()
            drawn = [self.pool[number] for number in order]
            self.pool = []
            yield from self._queue(drawn)
        if incomplete:
            yield from (patches for patches in self.pending.values() if patches)
            self.pending = {}

    def _queue(self, patches: list[tuple[int, Hashable, T, int]]) -> Iterator[list[tuple[T, int]]]:
        for _, key, field, number in patches:
            waiting = self.pending.setdefault(key, [])
            waiting.append((field, number))
            if len(waiting) == self.batch_size:
                self.pending[key] = []
                yield waiting


@dataclass(frozen=True, slots=True)
class RandomPatches:
    """``per_field`` patches of ``size`` (height, width) pixels at random places of a field, for training.

    Raises
    ------
    ParameterError
        If a size or ``per_field`` is below 1.
    """

    size: tuple[int, int] = (256, 256)
    per_field: int = 64

    def __post_init__(self) -> None:
        _expect_size(self, "RandomPatches")
        _expect_at_least(self.per_field, 1, where="RandomPatches", name="per_field")

    def origins(self, field_shape: tuple[int, int], *, generator: torch.Generator, stride: int = 1) -> Tensor:
        """Draw the origins ``(per_field, 2)`` int64 ``(y, x)`` of a field of ``field_shape`` pixels.

        Origins are multiples of ``stride``: give the network's stride.

        Raises
        ------
        ShapeError
            If the field is smaller than a patch.
        ParameterError
            If ``stride`` is below 1.
        """
        _expect_at_least(stride, 1, where="RandomPatches.origins", name="stride")
        limits = _room(field_shape, self.size, where="RandomPatches.origins")
        columns = [
            torch.randint(0, limit // stride + 1, (self.per_field,), generator=generator) * stride for limit in limits
        ]
        return torch.stack(columns, dim=1)

    def count(self, field_shape: tuple[int, int], *, stride: int = 1) -> int:
        """Return how many patches :meth:`origins` gives for a field; raise :class:`ShapeError` if it is too small."""
        del stride
        _room(field_shape, self.size, where="RandomPatches.count")
        return self.per_field


@dataclass(frozen=True, slots=True)
class GridPatches:
    """Patches of ``size`` pixels tiling a field, for validation and test; the last row and column shift inward.

    Raises
    ------
    ParameterError
        If a size is below 1.
    """

    size: tuple[int, int] = (256, 256)

    def __post_init__(self) -> None:
        _expect_size(self, "GridPatches")

    def origins(
        self, field_shape: tuple[int, int], *, generator: torch.Generator | None = None, stride: int = 1
    ) -> Tensor:
        """Return the grid's origins ``(N, 2)`` int64 ``(y, x)``, row by row; ``generator`` is unused.

        Origins round down to multiples of ``stride``, so up to ``stride - 1`` edge pixels may be in no patch. Raises
        as :meth:`RandomPatches.origins`.
        """
        del generator
        _expect_at_least(stride, 1, where="GridPatches.origins", name="stride")
        rows, columns = self._starts(field_shape, stride, where="GridPatches.origins")
        return torch.tensor([(y, x) for y in rows for x in columns], dtype=torch.int64).reshape(-1, 2)

    def count(self, field_shape: tuple[int, int], *, stride: int = 1) -> int:
        """Return how many patches :meth:`origins` gives for a field."""
        rows, columns = self._starts(field_shape, stride, where="GridPatches.count")
        return len(rows) * len(columns)

    def _starts(self, field_shape: tuple[int, int], stride: int, *, where: str) -> tuple[list[int], list[int]]:
        """Return the patch starts along y and along x."""
        starts = []
        for limit, size in zip(_room(field_shape, self.size, where=where), self.size, strict=True):
            last = limit // stride * stride
            along = [start // stride * stride for start in range(0, last, size)]
            starts.append(list(dict.fromkeys([*along, last])))
        return starts[0], starts[1]


def _expect_size(patches: RandomPatches | GridPatches, where: str) -> None:
    """Check a patch size is positive and store it as a tuple (YAML gives a list)."""
    for entry in patches.size:
        _expect_at_least(entry, 1, where=where, name="size")
    object.__setattr__(patches, "size", tuple(patches.size))


def _expect_at_least(value: int, minimum: int, *, where: str, name: str) -> None:
    if value < minimum:
        raise ParameterError(f"{where} {name} must be an int of at least {minimum}, got {value!r}", path=name)


def _room(field_shape: tuple[int, int], size: tuple[int, int], *, where: str) -> tuple[int, int]:
    """Return the largest origin along y and x that keeps a patch inside the field."""
    height, width = field_shape
    if height < size[0] or width < size[1]:
        raise ShapeError(
            f"{where}: a field of {height} x {width} pixels is smaller than a patch of {size[0]} x {size[1]}; use "
            "smaller patches or leave the field out of the index"
        )
    return (height - size[0], width - size[1])
