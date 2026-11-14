"""Heads: the decoder's last feature map to the object fields ``(B, F, Y, X)`` at the data's resolution."""

from abc import ABC, abstractmethod
from typing import ClassVar, override

from torch import Tensor, nn

from waveorder.errors import ParameterError
from waveorder.nn.networks.base import expect_count
from waveorder.nn.networks.layers import ConvNormAct, Norm
from waveorder.registry import register


class Head(ABC):
    """A description of a head; :meth:`module` builds its layers for one network.

    Parameters
    ----------
    fields : int
        The number of object fields ``F`` the network returns, at least 1.

    Attributes
    ----------
    upsampling : int
        How much finer along y and x the head's output is than its input: the decoder stops at ``1 / upsampling``
        of the data's resolution.
    """

    upsampling: ClassVar[int]

    def __init__(self, fields: int = 1) -> None:
        expect_count(fields, where=type(self).__name__, name="fields")
        self.fields = fields

    @abstractmethod
    def module(self, *, channels: int, norm: Norm) -> nn.Module:
        """Build the layers that map ``(B, channels, Y / upsampling, X / upsampling)`` to ``(B, fields, Y, X)``."""


@register("plane", kind="head")
class PlaneHead(Head):
    """One 1x1 convolution at the data's resolution: the decoder runs up to full resolution."""

    upsampling = 1

    @override
    def module(self, *, channels: int, norm: Norm) -> nn.Module:
        del norm
        return _PlaneHead(channels, self.fields)


class _PlaneHead(nn.Module):
    def __init__(self, channels: int, fields: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, fields, kernel_size=1)

    @override
    def forward(self, features: Tensor) -> Tensor:
        return self.conv(features)


@register("sub_pixel", kind="head")
class SubPixelHead(Head):
    """Two sub-pixel steps from 1/4 resolution, with one 3x3 convolution at 1/2 resolution between them.

    The decoder stops at 1/4 resolution with ``channels`` channels, a multiple of 4. A pixel shuffle rearranges them
    into ``channels / 4`` at 1/2 resolution, a 3x3 convolution block refines them, a 1x1 convolution gives 4 values
    per field, and a second pixel shuffle lays those out at full resolution. No convolution runs at full resolution.
    """

    upsampling = 4

    @override
    def module(self, *, channels: int, norm: Norm) -> nn.Module:
        if channels % 4:
            raise ParameterError(
                f"SubPixelHead needs a last decoder width that is a multiple of 4, got {channels}",
                path="decoder_channels",
            )
        return _SubPixelHead(channels // 4, self.fields, norm)


class _SubPixelHead(nn.Module):
    def __init__(self, channels: int, fields: int, norm: Norm) -> None:
        super().__init__()
        self.shuffle = nn.PixelShuffle(2)
        self.refine = ConvNormAct(channels, channels, norm)
        self.conv = nn.Conv2d(channels, 4 * fields, kernel_size=1)

    @override
    def forward(self, features: Tensor) -> Tensor:
        return self.shuffle(self.conv(self.refine(self.shuffle(features))))
