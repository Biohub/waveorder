"""The U-Net student: a stem, an encoder, a decoder with skips and a head, with conditions read at every level."""

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional

from waveorder.errors import ParameterError, ShapeError
from waveorder.nn.networks.base import RestorationNetwork, expect_count
from waveorder.nn.networks.conditioning import Conditioning, condition_offset
from waveorder.nn.networks.encoders import Encoder
from waveorder.nn.networks.heads import Head
from waveorder.nn.networks.layers import ConvNormAct, Norm, expect_norm
from waveorder.nn.networks.stems import Stem
from waveorder.registry import Spec, register, resolve


class DecoderBlock(nn.Module):
    """Upsample by 2, concatenate the skip (``skip_channels`` 0 for none), two 3x3 convolution blocks, dropout.

    With ``condition`` (the condition vector's length; 0 for none), a learned per-channel offset from that vector is
    added after the first convolution, before its normalization.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float, norm: Norm, condition: int
    ) -> None:
        super().__init__()
        joined = in_channels + skip_channels
        self.conv1 = ConvNormAct(joined, out_channels, norm)
        self.conv2 = ConvNormAct(out_channels, out_channels, norm)
        self.condition = condition_offset(condition, out_channels, fan_in=joined + condition) if condition else None
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    @override
    def forward(self, features: Tensor, skip: Tensor | None, vector: Tensor | None = None) -> Tensor:
        """Return ``(B, out, 2 y, 2 x)`` from ``features`` ``(B, in, y, x)`` and ``skip`` ``(B, skip, 2 y, 2 x)``."""
        features = functional.interpolate(features, scale_factor=2, mode="nearest")
        if skip is not None:
            features = torch.cat([features, skip], dim=1)
        convolution, normalization, activation = self.conv1
        features = convolution(features)
        if self.condition is not None and vector is not None:
            features = features + self.condition(vector)[:, :, None, None]
        return self.drop(self.conv2(activation(normalization(features))))


@register("unet", kind="network")
class UNet(RestorationNetwork):
    """A 2D U-Net over a stack folded into channels: stem, encoder, decoder with skips, head.

    The stem turns the stack ``(B, C, Z, Y, X)`` into a 2D feature map, so the network returns ``(B, F, 1, Y, X)``. Each
    decoder block doubles the resolution and takes the next finer encoder map as its skip, up to the head's input. Use
    a stride-1 stem (:class:`~waveorder.nn.networks.PlaneProjection`) with a
    :class:`~waveorder.nn.networks.PlaneHead`, or a strided stem (:class:`~waveorder.nn.networks.StridedVolume`,
    :class:`~waveorder.nn.networks.DefocusKernel`) with a :class:`~waveorder.nn.networks.SubPixelHead`, which runs no
    layer at full resolution. With a ResNet encoder the decoder has 5 blocks in the first layout and 3 in the second.

    Parameters
    ----------
    stem : Stem
        How the stack becomes a 2D feature map. Every part may also be given as its spec ``{name: arguments}``.
    encoder : Encoder
        The feature pyramid.
    head : Head
        The decoder's last map to the fields.
    decoder_channels : sequence of int
        Channels of each decoder block, coarse to fine: one per block.
    conditioning : Conditioning, optional
        Which conditions the network reads and how they enter (at the stem and at every decoder level); ``None`` for
        none, though a stem may still read one itself.
    dropout : float
        Probability of dropping a whole channel after each decoder block while training.
    norm : {"batch", "group"}
        ``"batch"``: batch norm everywhere. ``"group"``: group norm in the stem, the decoder and the head, and the
        encoder's batch norms frozen, so no output depends on the other items of its batch.
    full_resolution_channels : int
        With a stride-1 stem and a plane head: channels of a full-resolution skip (two 3x3 convolution blocks on the
        stem's output) into the last decoder block; 0 for none.
    halo : int, optional
        The pixels of context a tile needs on each side; by default :attr:`stride`. A deep encoder's receptive field
        is wider, so compare a tiled reconstruction with a whole one and widen it if they differ.

    Raises
    ------
    ParameterError
        If the parts do not fit together, ``decoder_channels`` does not have one width per block, ``dropout`` is not
        in [0, 1), or ``norm`` is unknown.
    """

    def __init__(  # noqa: PLR0913 (three parts and one keyword per setting of the network)
        self,
        stem: Stem | Spec,
        encoder: Encoder | Spec,
        head: Head | Spec,
        *,
        decoder_channels: Sequence[int],
        conditioning: Conditioning | Spec | None = None,
        dropout: float = 0.0,
        norm: Norm = "batch",
        full_resolution_channels: int = 0,
        halo: int | None = None,
    ) -> None:
        super().__init__()
        expect_norm(norm, where="UNet")
        if not 0 <= dropout < 1:
            raise ParameterError(f"UNet dropout must be in [0, 1), got {dropout!r}", path="dropout")
        expect_count(full_resolution_channels, where="UNet", name="full_resolution_channels", minimum=0)
        if halo is not None:
            expect_count(halo, where="UNet", name="halo", minimum=0)
        widths = tuple(decoder_channels)
        for width in widths:
            expect_count(width, where="UNet", name="decoder_channels")
        self._stem, self._encoder = resolve(stem, "stem", Stem), resolve(encoder, "encoder", Encoder)
        self._head = resolve(head, "head", Head)
        self._conditioning = None if conditioning is None else resolve(conditioning, "conditioning", Conditioning)
        self._halo = halo

        vector = 0 if self._conditioning is None else self._conditioning.size
        self._feature_planes = vector if self._conditioning is not None and self._conditioning.planes else 0
        self._stem_planes = self._feature_planes if self._stem.takes_planes else 0
        if self._stem.stride == 1:
            self.stem = self._stem.module(width=None, condition_planes=self._stem_planes, norm=norm)
            self.encoder = self._encoder.module(in_channels=self.stem.out_channels, norm=norm)
        else:
            self.encoder = self._encoder.module(in_channels=None, norm=norm)
            self.stem = self._stem.module(width=self.encoder.in_channels, condition_planes=self._stem_planes, norm=norm)
        # Where the stem takes no planes (offsets, or a stem with no fixed input layout), the vector is an offset.
        self.stem_condition = (
            condition_offset(vector, self.stem.out_channels, fan_in=self.stem.fan_in + vector)
            if vector and not self._stem_planes
            else None
        )

        skips = self._skips(full_resolution_channels)
        if len(widths) != len(skips):
            raise ParameterError(
                f"UNet decoder_channels has {len(widths)} widths, but the decoder has {len(skips)} blocks "
                f"({len(self.encoder.channels)} encoder levels, the finest at 1/{self._finest}, up to the head's "
                f"input at 1/{self._head.upsampling}); give {len(skips)}",
                path="decoder_channels",
            )
        self.full_resolution = (
            nn.Sequential(
                ConvNormAct(self.stem.out_channels, full_resolution_channels, norm),
                ConvNormAct(full_resolution_channels, full_resolution_channels, norm),
            )
            if full_resolution_channels
            else None
        )
        offsets = vector if self._conditioning is not None and self._conditioning.offsets else 0
        blocks = []
        previous = self.encoder.channels[-1] + self._feature_planes
        for width, skip in zip(widths, skips, strict=True):
            blocks.append(DecoderBlock(previous, skip, width, dropout=float(dropout), norm=norm, condition=offsets))
            previous = width
        self.decoder = nn.ModuleList(blocks)
        self.head = self._head.module(channels=widths[-1], norm=norm)

        self.channels = self._stem.data_channels
        self.fields = self._head.fields
        self.condition_sizes = self._condition_sizes()

    def _skips(self, full_resolution_channels: int) -> list[int]:
        """Return the skip channels of each decoder block, coarse to fine; 0 for a block without a skip."""
        reductions = self.encoder.reductions
        if any(coarse != 2 * fine for fine, coarse in pairwise(reductions)):
            raise ParameterError(
                f"UNet needs an encoder whose feature maps each halve the resolution, got reductions {reductions}",
                path="encoder",
            )
        self._finest = self._stem.stride * reductions[0]
        ratio, remainder = divmod(self._finest, self._head.upsampling)
        extra = ratio.bit_length() - 1  # blocks beyond the encoder's levels: one per factor of 2 up to the head
        if remainder or ratio < 1 or 2**extra != ratio:
            raise ParameterError(
                f"{type(self._head).__name__} starts from 1/{self._head.upsampling} resolution, but the finest "
                f"encoder map behind {type(self._stem).__name__} is at 1/{self._finest}; use a PlaneHead with a "
                "stem of stride 1 and a SubPixelHead with a strided stem",
                path="head",
            )
        skips = [channels + self._feature_planes for channels in self.encoder.channels[-2::-1]] + [0] * extra
        if full_resolution_channels:
            if self._stem.stride != 1 or self._head.upsampling != 1 or not extra:
                raise ParameterError(
                    "UNet full_resolution_channels needs a stem of stride 1 and a head at full resolution "
                    f"(PlaneProjection with PlaneHead); {type(self._stem).__name__} with {type(self._head).__name__} "
                    "has no full-resolution path: pass 0",
                    path="full_resolution_channels",
                )
            skips[-1] = full_resolution_channels
        return skips

    def _condition_sizes(self) -> dict[str, int]:
        sizes = {} if self._conditioning is None else dict(self._conditioning.sizes)
        for name, size in self._stem.condition_sizes.items():
            if sizes.setdefault(name, size) != size:
                raise ParameterError(
                    f"the stem reads the condition {name!r} with {size} values, the conditioning with {sizes[name]}",
                    path="conditioning",
                )
        return sizes

    @property
    @override
    def stride(self) -> int:
        """The reduction of the coarsest feature map: 32 for a ResNet encoder."""
        return self._stem.stride * self.encoder.reductions[-1]

    @override
    def halo(self) -> int:
        """Return the ``halo`` the network was built with, or :attr:`stride` when none was given."""
        return self.stride if self._halo is None else self._halo

    @override
    def planes_out(self, planes_in: int) -> int:
        """Return 1 for a plane count the stem takes.

        Raises
        ------
        ShapeError
            For any other plane count.
        """
        return self._stem.planes_out(planes_in)

    @override
    def forward(self, data: Tensor, condition: Mapping[str, Tensor] | None = None) -> Tensor:
        if data.dim() != 5 or data.shape[1] != self.channels:
            raise ShapeError(
                f"UNet data has shape {tuple(data.shape)}, expected (B, {self.channels}, Z, Y, X): a batch of stacks"
            )
        if data.shape[-2] % self.stride or data.shape[-1] % self.stride:
            raise ShapeError(
                f"UNet data has Y, X = {tuple(data.shape[-2:])}, which must be multiples of its stride "
                f"{self.stride}; pad the data (Distilled pads by reflection and crops the estimate)"
            )
        vector = None if self._conditioning is None else self._conditioning.vector(condition, like=data)
        features = self.stem(
            data,
            condition,
            planes=vector if self._stem_planes else None,
            offset=None if self.stem_condition is None or vector is None else self.stem_condition(vector),
        )
        levels = self.encoder(features)[::-1]  # coarse to fine
        if self._feature_planes and vector is not None:
            levels = [
                torch.cat([level, vector[:, :, None, None].expand(-1, -1, *level.shape[-2:])], dim=1)
                for level in levels
            ]
        skips: list[Tensor | None] = [*levels[1:], *([None] * (len(self.decoder) - len(levels) + 1))]
        if self.full_resolution is not None:
            skips[-1] = self.full_resolution(features)
        hidden = levels[0]
        for block, skip in zip(self.decoder, skips, strict=True):
            hidden = block(hidden, skip, vector)
        return self.head(hidden)[:, :, None]
