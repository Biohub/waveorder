"""Layers the parts of a student share: the normalization choice and the 3x3 convolution block."""

from typing import Any, Literal, get_args, override

import torch
from torch import Tensor, nn

from waveorder.errors import ParameterError

type Norm = Literal["batch", "group"]
"""The normalization of a network: ``"batch"`` (``BatchNorm2d``), or ``"group"``.

With ``"group"`` nothing in the network depends on the other items of a batch: the stem, the decoder and the head
use ``GroupNorm``, and the batch norms of an encoder are frozen at their stored statistics.
"""

GROUPS = 8
"""The most groups a ``GroupNorm`` layer uses; fewer where the channel count is not a multiple."""


def expect_norm(norm: object, *, where: str) -> None:
    """Raise :class:`~waveorder.errors.ParameterError` unless ``norm`` is ``"batch"`` or ``"group"``."""
    if norm not in get_args(Norm.__value__):
        raise ParameterError(f"{where} norm must be 'batch' or 'group', got {norm!r}", path="norm")


def norm_layer(channels: int, norm: Norm) -> nn.Module:
    """Return ``BatchNorm2d(channels)``, or a ``GroupNorm`` of up to :data:`GROUPS` groups that divide ``channels``."""
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    groups = min(GROUPS, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    """A 3x3 convolution without bias, a normalization layer and a ReLU: ``(B, in, Y, X)`` to ``(B, out, Y, X)``."""

    def __init__(self, in_channels: int, out_channels: int, norm: Norm) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, norm),
            nn.ReLU(inplace=True),
        )


class FrozenBatchNorm2d(nn.Module):
    """A batch norm that always applies its stored statistics and affine, and trains nothing.

    Its four tensors are buffers under the names of ``BatchNorm2d`` (``weight``, ``bias``, ``running_mean``,
    ``running_var``), so a checkpoint of a ``BatchNorm2d`` loads into it.
    """

    weight: Tensor
    bias: Tensor
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, channels: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Drop the step counter of a ``BatchNorm2d`` checkpoint, which a frozen layer does not keep."""
        state_dict.pop(f"{prefix}num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` ``(B, channels, Y, X)`` normalized by the stored statistics."""
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        shift = self.bias - self.running_mean * scale
        return x * scale[None, :, None, None] + shift[None, :, None, None]

    @override
    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


def block_norm(channels: int, norm: Norm) -> nn.Module:
    """Return the normalization of an encoder block: ``BatchNorm2d``, frozen when ``norm`` is ``"group"``."""
    return nn.BatchNorm2d(channels) if norm == "batch" else FrozenBatchNorm2d(channels)
