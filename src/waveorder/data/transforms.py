"""Augmentations of a batch on its device that keep its conditions true.

A :class:`Transform` draws all random numbers from a CPU :class:`torch.Generator`, so a seed gives the same
augmentations on any device.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor

from waveorder.data.loader import Batch
from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn.networks import FieldPosition, PlaneOffsets

_AXES = {"y": (-2, 0), "x": (-1, 1)}
"""Axis name to its dimension in ``(B, C, Z, Y, X)`` and its entry in a field position ``(y, x)``."""


class Transform(ABC):
    """A random change of a batch on its device, with the conditions kept in step."""

    @abstractmethod
    def __call__(self, batch: Batch, *, generator: torch.Generator) -> Batch:
        """Return the transformed batch; ``generator`` is a CPU generator all random numbers come from."""

    def arguments(self) -> dict[str, object]:
        """Return the constructor arguments: ``type(self)(**self.arguments())`` is an equal transform."""
        return dict(vars(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{name}={value!r}' for name, value in self.arguments().items())})"


class PlaneCrop(Transform):
    """Keep a random contiguous run of z planes, the same run for the whole batch.

    With probability ``probability`` a run of ``min_planes`` to ``Z`` planes is kept. The plane-offsets condition keeps
    the kept planes' offsets and is padded with NaN. A target with one plane per data plane is cropped alike; a
    one-plane target is left as it is.

    Parameters
    ----------
    min_planes : int
        The fewest planes kept.
    probability : float
        Chance that a batch is cropped, in [0, 1].
    pad : bool
        Zero-pad after the kept planes to keep the plane count; else the network must take any plane count.
    condition : str
        Name of the plane-offsets condition.

    Raises
    ------
    ParameterError
        If ``min_planes`` is below 1 or ``probability`` is not in [0, 1].
    """

    def __init__(
        self, min_planes: int, probability: float = 0.5, *, pad: bool = False, condition: str = PlaneOffsets.name
    ) -> None:
        if min_planes < 1:
            raise ParameterError(f"PlaneCrop min_planes must be a positive int, got {min_planes!r}", path="min_planes")
        self.min_planes = min_planes
        self.probability = _probability(probability, where="PlaneCrop")
        self.pad = pad
        self.condition = condition

    @override
    def __call__(self, batch: Batch, *, generator: torch.Generator) -> Batch:
        """Crop the batch's planes; raise :class:`ShapeError` if the offsets have fewer entries than planes."""
        planes = batch.data.shape[2]
        if planes <= self.min_planes or torch.rand((), generator=generator).item() >= self.probability:
            return batch
        kept = int(torch.randint(self.min_planes, planes + 1, (), generator=generator))
        start = int(torch.randint(0, planes - kept + 1, (), generator=generator))
        target = batch.target
        if target.shape[2] == planes:
            target = self._run(target, start, kept)
        condition = dict(batch.condition)
        offsets = condition.get(self.condition)
        if offsets is not None:
            if offsets.shape[1] < planes:
                raise ShapeError(
                    f"PlaneCrop: the condition {self.condition!r} has {offsets.shape[1]} entries but the data "
                    f"{planes} planes; give PlaneOffsets at least as many planes as the data has"
                )
            cropped = torch.full_like(offsets, float("nan"))
            cropped[:, :kept] = offsets[:, start : start + kept]
            condition[self.condition] = cropped
        return dataclasses.replace(batch, data=self._run(batch.data, start, kept), target=target, condition=condition)

    def _run(self, stack: Tensor, start: int, kept: int) -> Tensor:
        """Return planes ``start`` to ``start + kept`` of ``(B, C, Z, Y, X)``, zero-padded with ``pad``."""
        run = stack[:, :, start : start + kept]
        if not self.pad:
            return run
        return torch.nn.functional.pad(run, (0, 0, 0, 0, 0, stack.shape[2] - kept))


class FocusShift(Transform):
    """Shift each sample's plane offsets by one random value, a simulated focus error.

    The shift is uniform in ``[-max_shift, max_shift]`` axial optical units (see
    :class:`~waveorder.nn.networks.PlaneOffsets`). The data is unchanged; NaN offsets stay NaN.

    Raises
    ------
    ParameterError
        If ``max_shift`` is negative or not finite.
    """

    def __init__(self, max_shift: float, *, condition: str = PlaneOffsets.name) -> None:
        if not 0 <= max_shift < float("inf"):
            raise ParameterError(
                f"FocusShift max_shift must be a non-negative number of axial units, got {max_shift!r}",
                path="max_shift",
            )
        self.max_shift = float(max_shift)
        self.condition = condition

    @override
    def __call__(self, batch: Batch, *, generator: torch.Generator) -> Batch:
        """Shift the batch's plane offsets; raise :class:`IncompatibleModelError` if it has none."""
        offsets = batch.condition.get(self.condition)
        if offsets is None:
            raise IncompatibleModelError(
                f"FocusShift shifts the condition {self.condition!r}, which the batch does not have (it has "
                f"{sorted(batch.condition) or 'none'}): give the loader a PlaneOffsets condition, or drop the "
                "transform"
            )
        shift = (torch.rand(offsets.shape[0], 1, generator=generator) * 2 - 1) * self.max_shift
        condition = {**batch.condition, self.condition: offsets + shift.to(offsets)}
        return dataclasses.replace(batch, condition=condition)


class Flip(Transform):
    """Mirror samples along y, x or both, each sample and axis with its own coin.

    Data and target flip together and the field-position condition is mirrored (``c`` becomes ``1 - c``). Plane
    offsets are unchanged. Any other condition must be listed in ``invariant``, else the batch is refused.
    ``batch.origin`` keeps the patch's place in the recorded field.

    Parameters
    ----------
    axes : Sequence[str]
        Axes that may be flipped: ``"y"``, ``"x"`` or both.
    probability : float
        Chance that a sample is flipped along an axis, in [0, 1].
    invariant : Sequence[str]
        Conditions a flip does not change.
    position : FieldPosition, optional
        The loader's field-position condition; ``FieldPosition()`` by default.

    Raises
    ------
    ParameterError
        If ``axes`` is empty, names another axis or an axis twice, or ``probability`` is not in [0, 1].
    """

    def __init__(
        self,
        axes: Sequence[str] = ("y", "x"),
        *,
        probability: float = 0.5,
        invariant: Sequence[str] = (),
        position: FieldPosition | None = None,
    ) -> None:
        if not axes or any(axis not in _AXES for axis in axes) or len(set(axes)) != len(axes):
            raise ParameterError(f"Flip axes must be 'y', 'x' or both, each once, got {axes!r}", path="axes")
        self.axes = tuple(axes)
        self.probability = _probability(probability, where="Flip")
        self.invariant = tuple(invariant)
        self.position = FieldPosition() if position is None else position

    @override
    def __call__(self, batch: Batch, *, generator: torch.Generator) -> Batch:
        """Flip the batch; raise :class:`IncompatibleModelError` for a condition it does not know."""
        name = self.position.name
        known = {name, PlaneOffsets.name, *self.invariant}
        unknown = sorted(set(batch.condition) - known)
        if unknown:
            raise IncompatibleModelError(
                f"Flip does not know what a flip does to the conditions {', '.join(unknown)}: name them in "
                "invariant=(...) if a flip leaves them unchanged, or drop the transform"
            )
        data, target = batch.data, batch.target
        position = batch.condition.get(name)
        for axis in self.axes:
            dimension, entry = _AXES[axis]
            coins = torch.rand(data.shape[0], generator=generator) < self.probability  # on the CPU: no device sync
            if not coins.any():
                continue
            flipped = coins.to(data.device)
            each = flipped[:, None, None, None, None]
            data = torch.where(each, data.flip(dimension), data)
            target = torch.where(each, target.flip(dimension), target)
            if position is not None:
                index = entry if self.position.axes == "yx" else 1 - entry
                offset, scale = self.position.offset[index], self.position.scale[index]
                mirrored = (1 - 2 * offset) / scale - position[:, index]  # (1 - c - offset) / scale
                position = position.clone()
                position[:, index] = torch.where(flipped.to(position.device), mirrored, position[:, index])
        condition = dict(batch.condition) if position is None else {**batch.condition, name: position}
        return dataclasses.replace(batch, data=data, target=target, condition=condition)


def _probability(value: float, *, where: str) -> float:
    if not 0 <= value <= 1:
        raise ParameterError(f"{where} probability must be a number in [0, 1], got {value!r}", path="probability")
    return float(value)
