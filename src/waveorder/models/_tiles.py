"""The pieces of a model's tile methods: sources read as the model prepares them, per-tile values, z padding.

A z-padded model reads each source with its mirrored slices (:class:`_ZPadded`) and crops each estimate back
(:class:`_ZCropped`), as a whole stack is padded.
"""

import dataclasses

import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.models._base import ReconstructionModel
from waveorder.models._prepare import Prepared
from waveorder.nn.preprocess import ZPadding
from waveorder.nn.sampling import ZYX
from waveorder.nn.systems.tiled import TileParameters
from waveorder.tiling import TilePlan
from waveorder.tiling.merge import ArrayLike, Index, WritableArray


def as_measured(model: ReconstructionModel, source: ArrayLike, *, plan: TilePlan, where: str) -> ArrayLike:
    """Return ``source`` read as the model prepares it; refuse several tiles for a ``"block"`` prepare scope."""
    kind = type(model)
    if kind.prepare is ReconstructionModel.prepare:  # the identity: there is nothing to prepare
        return source
    if kind.prepare_scope == "block" and len(plan.regions) > 1:
        raise ParameterError(
            f"{where}: the plan has {len(plan.regions)} tiles, but {kind.__name__}.prepare_scope is 'block' (its "
            "prepare depends on the block it is given), so tiles would not be prepared alike; use one tile, or pass "
            "data as the system measures it with prepared=True",
            path="plan",
        )
    return Prepared(source, model=model)


def field_tile_parameters(model: ReconstructionModel, *, plan: TilePlan, where: str) -> TileParameters | None:
    """Return the model's per-field-tile values for ``plan``; ``None`` when it holds none or has no field tiles."""
    table = model.tile_parameters
    if not table or model.field_tiles is None:
        return None
    count = len(plan.regions)
    held = {values.shape[0] for values in table.values()}
    if held != {count}:
        raise ParameterError(
            f"{where}: the model's tile_parameters hold {sorted(held)} tiles, but the plan has {count}; plan the "
            "volume with field_tile_plan, pass parameters=, or set model.tile_parameters = None",
            path="plan",
        )
    return table


def z_padded(
    model: ReconstructionModel,
    plan: TilePlan,
    pairs: tuple[tuple[ArrayLike, WritableArray], ...],
    *,
    pixel_size: ZYX[float],
) -> tuple[TilePlan, tuple[tuple[ArrayLike, WritableArray], ...]]:
    """Return the plan and volumes a z-padded model solves; refuse a plan that splits z."""
    if not model.z_padding:
        return plan, pairs
    z, y, x = plan.shape
    if plan.tile[0] < z:
        raise model._refused(  # noqa: SLF001
            "solver.padding.z",
            "z_padding",
            f"is {model.z_padding}, but the plan splits z into tiles of {plan.tile[0]} of {z} slices; padding mirrors "
            "whole stacks, so give tiles the stack's z, or no padding",
        )
    slices = model.z_padding_for(plan.shape, pixel_size=pixel_size)
    depth = z + 2 * slices
    _, oy, ox = plan.overlap
    padded = dataclasses.replace(
        plan, shape=(depth, y, x), tile=(depth, plan.tile[1], plan.tile[2]), overlap=(0, oy, ox)
    )
    padding = ZPadding(slices)
    cropped = tuple((_ZPadded(source, padding), _ZCropped(sink, slices)) for source, sink in pairs)
    return padded, cropped  # ty: ignore[invalid-return-type]  # a sink is only written, never read back


class _ZPadded:
    """A ``(..., Z, Y, X)`` source seen as ``(..., Z + 2p, Y, X)``; the last padded read is kept for the next."""

    def __init__(self, source: ArrayLike, padding: ZPadding) -> None:
        *lead, z, y, x = (int(n) for n in source.shape)
        self.shape = (*lead, z + 2 * padding.slices, y, x)
        self.dtype = getattr(source, "dtype", None)
        self._source = source
        self._padding = padding
        self._last: tuple[tuple[object, ...], Tensor] | None = None

    def __getitem__(self, key: Index, /) -> Tensor:
        *lead, depth, rows, columns = key
        whole = (*lead, slice(None), rows, columns)
        last = self._last
        if last is None or last[0] != whole:
            last = (whole, self._padding.pad(torch.as_tensor(self._source[whole])))
            self._last = last
        return last[1][..., depth, :, :]


class _ZCropped:
    """A ``(..., Z, Y, X)`` sink seen as ``(..., Z + 2p, Y, X)``: writes keep the slices of the measured stack only."""

    def __init__(self, sink: WritableArray, slices: int) -> None:
        *lead, z, y, x = (int(n) for n in sink.shape)
        self.shape = (*lead, z + 2 * slices, y, x)
        self.dtype = getattr(sink, "dtype", None)
        self.chunks = getattr(sink, "chunks", None)
        self.shards = getattr(sink, "shards", None)
        self._sink = sink
        self._slices = slices
        self._z = z

    def __setitem__(self, key: Index, value: object, /) -> None:
        *lead, depth, rows, columns = key
        start, stop, _ = depth.indices(self.shape[-3])  # ty: ignore[missing-union-attribute]  # the merger writes slices
        low, high = max(start, self._slices), min(stop, self._slices + self._z)
        if low < high:
            kept = value[..., low - start : high - start, :, :]  # ty: ignore[not-subscriptable]
            if isinstance(self._sink, Tensor):  # the merger writes host arrays; a tensor sink takes tensors only
                kept = torch.as_tensor(kept)
            self._sink[(*lead, slice(low - self._slices, high - self._slices), rows, columns)] = kept


def expect_pairs(volumes: object, *, where: str) -> None:
    """Refuse ``volumes`` unless it is a non-empty list or tuple of ``(source, sink)`` pairs of arrays.

    Raises
    ------
    ParameterError
        At ``volumes`` or ``volumes.<i>``; a bare pair gets the hint to pass ``[(source, sink)]``, an OME-Zarr
        position the hint to call ``reconstruct_positions``.
    """
    if not isinstance(volumes, list | tuple) or not volumes:
        raise ParameterError(
            f"{where}: volumes must be a non-empty list of (source, sink) pairs, got {type(volumes).__qualname__}",
            path="volumes",
        )
    from iohub.ngff.nodes import Position

    for index, pair in enumerate(volumes):
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ParameterError(
                f"{where}: volumes.{index} is a {type(pair).__qualname__}, not a (source, sink) pair; for one volume "
                "pass [(source, sink)], not (source, sink)",
                path=f"volumes.{index}",
            )
        if any(isinstance(part, Position) for part in pair):
            raise ParameterError(
                f"{where}: volumes.{index} holds an OME-Zarr position, not an array; call reconstruct_positions, or "
                "build the pairs with waveorder.tiling.merge.volumes(source, sink, channels=[...])",
                path=f"volumes.{index}",
            )
