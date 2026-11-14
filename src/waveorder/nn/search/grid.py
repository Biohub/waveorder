"""Batched candidate search: score points of the free quantities' spans in batches and keep the best.

A new candidate stage can reuse :func:`search_spans` (where each quantity is searched) and :func:`keep_best` (score
candidate batches and keep the best on the device).
"""

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.nn.parameters import Free
from waveorder.nn.search.base import (
    CalibrationObjective,
    Estimate,
    ScoreGrid,
    _expect_count,
    _geometric,
    expect_known_names,
    keep_record,
    keeping_records,
)
from waveorder.registry import register

MAX_GRID_POINTS = 10**6
"""The most points a :class:`GridSearch` scores; a larger grid is refused before any of it is built."""


def _expect_chunk_size(value: int | None, *, where: str) -> None:
    if value is not None:
        _expect_count(value, where=where, name="chunk_size", minimum=1)


@register("grid", kind="search")
@dataclass(frozen=True, slots=True, kw_only=True)
class GridSearch:
    """Score every point of a grid over the free quantities and keep the best.

    Each quantity spans its bounds, or ``nominal +/- width * tolerance`` without bounds, spaced geometrically when the
    span is positive and wider than a factor of 100, linearly otherwise. The grid grows as ``points ** quantities``;
    more than :data:`MAX_GRID_POINTS` points is refused.

    Parameters
    ----------
    points : int or Mapping[str, int]
        Grid size along every quantity, or per dotted name (7 for a quantity not listed); each at least 2.
    chunk_size : int, optional
        Build and score at most this many points at a time, bounding memory; the whole grid by default.
    names : tuple[str, ...], optional
        Grid only these free quantities, holding the others at the best point so far; every free quantity by
        default. The 3.x profile of noise level and weight is ``names=("noise.sigma", "object_prior.weight")``.
    width : float
        Half-width of an unbounded span in tolerances; 3 by default (2 in the 3.x evidence path); positive.

    Raises
    ------
    ParameterError
        For a grid size below 2, a ``chunk_size`` below 1 or a ``width`` that is not positive; when the stage runs,
        for more than :data:`MAX_GRID_POINTS` points or a name that is no free quantity.
    """

    points: int | Mapping[str, int] = 7
    chunk_size: int | None = None
    names: tuple[str, ...] | None = None
    width: float = 3.0
    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ("points",)

    def __post_init__(self) -> None:
        where = "GridSearch"
        if isinstance(self.points, Mapping):
            for key, count in self.points.items():
                _expect_count(count, where=where, name=f"points.{key}", minimum=2)
        else:
            _expect_count(self.points, where=where, name="points", minimum=2)
        _expect_chunk_size(self.chunk_size, where=where)
        if not (math.isfinite(self.width) and self.width > 0):
            raise ParameterError(f"{where} width must be positive and finite, got {self.width}", path="width")

    def points_for(self, name: str) -> int:
        """Return the grid size along ``name``."""
        return self.points if isinstance(self.points, int) else self.points.get(name, 7)

    def batches(self, specs: Mapping[str, Free], *, like: Tensor) -> Iterator[dict[str, Tensor]]:
        """Return the grid in batches of at most ``chunk_size`` points, as equal-length 1-D tensors by name.

        Points follow the flattened ``meshgrid(..., indexing="ij")``, in the dtype and on the device of ``like``.

        Raises
        ------
        ParameterError
            If ``points`` names a quantity not in ``specs``, a quantity has neither bounds nor a tolerance, or the
            grid would hold more than :data:`MAX_GRID_POINTS` points.
        """
        if not isinstance(self.points, int):
            expect_known_names(self.points, specs, path="GridSearch.points")
        spans = search_spans(specs, width=self.width)
        if self.names is not None:
            expect_known_names(self.names, specs, path="GridSearch.names")
            spans = {name: spans[name] for name in self.names}
        counts = {name: self.points_for(name) for name in spans}
        shape = tuple(counts.values())
        total = math.prod(shape)
        if total > MAX_GRID_POINTS:
            raise ParameterError(
                f"GridSearch would score {_product_text(shape)} = {total} points, more than the {MAX_GRID_POINTS} a "
                "grid may hold; pass search=[GradientSearch()] (or RandomSearch), or fewer points per quantity "
                "(points={name: n})",
                path="points",
            )
        axes = [_grid_axis(span, points=counts[name], like=like) for name, span in spans.items()]
        size = total if self.chunk_size is None else self.chunk_size
        # Row-major strides as Python ints: torch.unravel_index would copy its strides to the device, a host sync.
        strides = [math.prod(shape[k + 1 :]) for k in range(len(shape))]

        def generate() -> Iterator[dict[str, Tensor]]:
            for first in range(0, total, size):
                flat = torch.arange(first, min(first + size, total), device=like.device)
                yield {
                    name: axis[(flat // stride) % count]
                    for name, axis, stride, count in zip(spans, axes, strides, shape, strict=True)
                }

        return generate()

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best grid point, or ``start`` if no grid point is better; keeps the grid for ``history``."""
        batches = self.batches(objective.specs, like=_like(start))
        return keep_best(objective, batches, start=start, keep_scores=True)


def _product_text(shape: tuple[int, ...]) -> str:
    """``7 ** 12`` for equal grid sizes over 12 quantities, else the product written out, ``7 x 3 x 7``."""
    if len(set(shape)) == 1:
        return f"{shape[0]} ** {len(shape)} (points ** quantities)"
    return " x ".join(map(str, shape))


@register("random", kind="search")
@dataclass(frozen=True, slots=True, kw_only=True)
class RandomSearch:
    """Score ``samples`` uniformly random points of the same spans as :class:`GridSearch` and keep the best.

    Parameters
    ----------
    samples : int
        Number of points drawn, at least 1.
    seed : int
        Seed of the CPU generator the points are drawn from.
    chunk_size : int, optional
        Score at most this many points at a time, bounding memory; all by default.
    """

    samples: int = 32
    seed: int = 0
    chunk_size: int | None = None
    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        _expect_count(self.samples, where="RandomSearch", name="samples", minimum=1)
        _expect_chunk_size(self.chunk_size, where="RandomSearch")

    def batches(self, specs: Mapping[str, Free], *, like: Tensor) -> Iterator[dict[str, Tensor]]:
        """Return the drawn points in batches of at most ``chunk_size``, like :meth:`GridSearch.batches`.

        Raises
        ------
        ParameterError
            If a quantity has neither bounds nor a tolerance.
        """
        spans = search_spans(specs)
        generator = torch.Generator().manual_seed(self.seed)
        drawn = {}
        for name, (low, high) in spans.items():
            unit = torch.rand(self.samples, generator=generator, dtype=like.dtype)
            if like.device.type == "cuda":
                # Drawn on the CPU generator for the same points on every device; pinned, the copy does not sync.
                unit = unit.pin_memory().to(like.device, non_blocking=True)
            else:
                unit = unit.to(like.device)
            if _geometric(low, high):
                drawn[name] = low * (high / low) ** unit
            else:
                drawn[name] = low + (high - low) * unit
        size = self.samples if self.chunk_size is None else self.chunk_size
        return (
            {name: values[first : first + size] for name, values in drawn.items()}
            for first in range(0, self.samples, size)
        )

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best drawn point, or ``start`` if none is better."""
        return keep_best(objective, self.batches(objective.specs, like=_like(start)), start=start)


def _like(start: Estimate) -> Tensor:
    return next(iter(start.point.values()))


def _span(name: str, spec: Free, *, width: float = 3.0) -> tuple[float, float]:
    if spec.bounds is not None:
        return spec.bounds
    if spec.tolerance is not None:
        return (spec.nominal - width * spec.tolerance, spec.nominal + width * spec.tolerance)
    raise ParameterError(f"{name} needs bounds or a tolerance to be searched", path=name)


def search_spans(specs: Mapping[str, Free], *, width: float = 3.0) -> dict[str, tuple[float, float]]:
    """Return the interval each free quantity is searched over, by dotted name, in the quantity's units.

    A quantity spans its ``bounds``, or ``nominal +/- width * tolerance`` without bounds. A span is geometric
    (searched on a log scale) when it is positive and wider than a factor of 100.

    Parameters
    ----------
    specs : Mapping[str, Free]
        The free quantities by dotted name, :attr:`~waveorder.nn.search.CalibrationObjective.specs`.
    width : float
        Half-width of an unbounded span in tolerances.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``(low, high)`` per name, in the quantity's own units.

    Raises
    ------
    ParameterError
        If a quantity has neither bounds nor a tolerance.
    """
    return {name: _span(name, spec, width=width) for name, spec in specs.items()}


def _grid_axis(span: tuple[float, float], *, points: int, like: Tensor) -> Tensor:
    low, high = span
    if _geometric(low, high):
        return torch.logspace(math.log10(low), math.log10(high), points, dtype=like.dtype, device=like.device)
    return torch.linspace(low, high, points, dtype=like.dtype, device=like.device)


def keep_best(
    objective: CalibrationObjective,
    candidates: Mapping[str, Tensor] | Iterable[Mapping[str, Tensor]],
    *,
    start: Estimate,
    keep_scores: bool = False,
) -> Estimate:
    """Score candidates and return the best of them, or ``start`` if none is better.

    Parameters
    ----------
    objective : CalibrationObjective
        What to minimize.
    candidates : Mapping[str, Tensor] or Iterable of them
        One batch of equal-length 1-D tensors by dotted name, or batches of them; a batch's length bounds memory. A
        searched name a batch leaves out is held at its value in the best point so far.
    start : Estimate
        The best estimate so far. Its values may carry a leading tile axis ``(B,)``; every tile then keeps its own
        best candidate.
    keep_scores : bool
        Hand each batch's points and objectives, as a :class:`~waveorder.nn.search.base.ScoreGrid`, to a
        ``calibrate(..., history=True)``.

    Returns
    -------
    Estimate
        On the device of ``start``, in its shapes; nothing is read back.

    Raises
    ------
    ParameterError
        If a batch names a quantity that is not in ``start.point``.
    """
    # A Mapping is itself an Iterable (of its keys), so isinstance cannot narrow the union for a type checker.
    batches: Iterable[Mapping[str, Tensor]] = (
        (cast(Mapping[str, Tensor], candidates),) if isinstance(candidates, Mapping) else candidates
    )
    point, value = dict(start.point), start.objective
    for batch in batches:
        unknown = sorted(set(batch) - set(point))
        if unknown:
            raise ParameterError(f"candidates name {unknown}, which are not searched; searched are {sorted(point)}")
        length = next(iter(batch.values())).shape[0]
        full = {
            name: batch[name] if name in batch else point[name].expand(length, *point[name].shape) for name in point
        }
        scores = objective.batch(full)  # (N,), or (N, B) over tiles
        if keep_scores and keeping_records():
            columns = [batch[name].to(scores.dtype) for name in batch]
            keep_record(ScoreGrid(tuple(batch), torch.stack([*columns, scores], dim=-1).detach()))
        lowest, index = scores.min(dim=0)
        better = lowest < value
        # Indexing with a tensor index reads it back to the host; take_along_dim gathers on the device.
        point = {
            name: torch.where(better, torch.take_along_dim(_tiled(values, scores), index[None], dim=0)[0], point[name])
            for name, values in full.items()
        }
        value = torch.where(better, lowest, value)
    return Estimate(point, value)


def _tiled(values: Tensor, scores: Tensor) -> Tensor:
    """``values`` ``(N,)`` (shared by every tile) or ``(N, B)``, broadcast to the shape of ``scores``."""
    return values.reshape(*values.shape, *(1,) * (scores.dim() - values.dim())).expand_as(scores)
