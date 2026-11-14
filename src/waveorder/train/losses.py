"""The terms of a training loss.

A :class:`Loss` is a module called as ``loss(estimate, batch)`` that returns a scalar, lower is better. ``estimate``
is the student's output ``(B, F, Zout, Y, X)`` in normalized target units, and ``batch`` the
:class:`~waveorder.data.Batch` it was computed from, whose ``target`` is the recorded reconstruction. :class:`L1`,
:class:`MSE` and :class:`SSIM` compare the two; a loss of your own is named by its class path in a configuration.
"""

from abc import ABC, abstractmethod
from typing import Annotated, override

import torch
from pydantic import Field, PositiveFloat
from torch import Tensor, nn

from waveorder.data import Batch
from waveorder.errors import ParameterError, ShapeError
from waveorder.train.metrics import MIN_RANGE, plane_ssim


class Loss(nn.Module, ABC):
    """One term of a training loss: ``loss(estimate, batch)`` returns a scalar tensor, lower is better.

    Subclass it and implement :meth:`forward`.
    """

    @abstractmethod
    @override
    def forward(self, estimate: Tensor, batch: Batch) -> Tensor:
        """Return the scalar loss of ``estimate`` ``(B, F, Zout, Y, X)``, in normalized target units, on ``batch``."""


def _target(batch: Batch, estimate: Tensor, *, where: str) -> Tensor:
    """Return ``batch.target`` after checking it is shaped as the estimate."""
    target = batch.target
    if target.shape != estimate.shape:
        raise ShapeError(
            f"{where}: the estimate has shape {tuple(estimate.shape)} but batch.target has {tuple(target.shape)}; "
            "both are (B, F, Zout, Y, X), so the student's fields and planes_out must match the reconstruction's"
        )
    return target


class L1(Loss):
    """The mean absolute difference between the estimate and ``batch.target``, in normalized target units."""

    @override
    def forward(self, estimate: Tensor, batch: Batch) -> Tensor:
        return (estimate.float() - _target(batch, estimate, where="L1")).abs().mean()


class MSE(Loss):
    """The mean squared difference between the estimate and ``batch.target``, in normalized target units."""

    @override
    def forward(self, estimate: Tensor, batch: Batch) -> Tensor:
        return (estimate.float() - _target(batch, estimate, where="MSE")).square().mean()


class SSIM(Loss):
    """One minus the mean plane-by-plane SSIM of the estimate with ``batch.target`` (the validation ``ssim``).

    Blank target planes (flatter than ``min_range``) are left out: dividing by their range overflows float32.

    Parameters
    ----------
    window : int
        Side of the square averaging window in pixels, at least 2; patches must be at least this large.
    min_range : float
        The range (max - min, normalized target units) below which a target plane is blank; positive.
    """

    def __init__(self, window: Annotated[int, Field(ge=2)] = 11, *, min_range: PositiveFloat = MIN_RANGE) -> None:
        super().__init__()
        if window < 2:
            raise ParameterError(f"SSIM window must be an int of at least 2 pixels, got {window!r}", path="window")
        if not min_range > 0:
            raise ParameterError(f"SSIM min_range must be positive, got {min_range!r}", path="min_range")
        self.window = window
        self.min_range = float(min_range)

    @override
    def extra_repr(self) -> str:
        return f"window={self.window}, min_range={self.min_range:g}"

    @override
    def forward(self, estimate: Tensor, batch: Batch) -> Tensor:
        target = _target(batch, estimate, where="SSIM")
        values = plane_ssim(estimate.float(), target, window=self.window, min_range=self.min_range)
        scored = values[~torch.isnan(values)]
        if scored.numel() == 0:
            return estimate.float().sum() * 0.0
        return 1 - scored.mean()
