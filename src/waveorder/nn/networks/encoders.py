"""Encoders: a 2D feature map to a pyramid of feature maps, fine to coarse.

:class:`ResNetEncoder` needs torch only; :class:`TimmEncoder` imports ``timm`` when its layers are built. The built-in
ResNets have the layers and state-dict keys of ``timm``'s ``resnet10t``, ``resnet18`` and ``resnet34``, so a
``TimmEncoder`` checkpoint of those loads into a ``ResNetEncoder`` once the prefix ``encoder.model.`` is renamed to
``encoder.``.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast, override

from torch import Tensor, nn

from waveorder.errors import ParameterError
from waveorder.nn.networks.layers import Norm, block_norm
from waveorder.registry import register


class EncoderModule(nn.Module, ABC):
    """The layers of an encoder.

    Attributes
    ----------
    in_channels : int
        Channels of the feature map it takes.
    channels : tuple[int, ...]
        Channels of each feature map it returns, fine to coarse.
    reductions : tuple[int, ...]
        The y and x reduction of each feature map relative to its input, fine to coarse; each is twice the one
        before.
    """

    in_channels: int
    channels: tuple[int, ...]
    reductions: tuple[int, ...]

    @override
    @abstractmethod
    def forward(self, features: Tensor) -> list[Tensor]:
        """Return the feature maps of ``features`` ``(B, in_channels, Y, X)``, fine to coarse."""


class Encoder(ABC):
    """A description of an encoder; :meth:`module` builds its layers for one network."""

    @abstractmethod
    def module(self, *, in_channels: int | None, norm: Norm) -> EncoderModule:
        """Build the layers.

        ``in_channels`` is the stem's output channels, for an encoder that keeps its own stride-4 stem; ``None`` drops
        that stem, and the network's stem must return :attr:`~EncoderModule.in_channels`. With ``norm="group"`` the
        batch norms are frozen.
        """


_PRESETS: dict[str, tuple[tuple[int, int, int, int], bool]] = {
    "resnet10t": ((1, 1, 1, 1), True),
    "resnet18": ((2, 2, 2, 2), False),
    "resnet34": ((3, 4, 6, 3), False),
}
"""Blocks per stage, and whether the stem is tiered (three 3x3 convolutions, average-pool shortcuts), by preset."""

_WIDTHS = (64, 128, 256, 512)


@register("resnet", kind="encoder")
class ResNetEncoder(Encoder):
    """A ResNet of basic blocks (two 3x3 convolutions with a shortcut), four stages, each halving the resolution.

    With its own stem (a stride-2 convolution block and a stride-2 max pool) it returns five maps, at 1/2 to 1/32 of
    its input; without (a strided network stem replaces it), the four stage outputs, at 1/1 to 1/8.

    Parameters
    ----------
    preset : {"resnet10t", "resnet18", "resnet34"}
        The depths and the stem: 1, 2 or (3, 4, 6, 3) blocks per stage. ``"resnet10t"`` has a stem of three 3x3
        convolutions and average-pooled shortcuts; the others a 7x7 stem convolution.
    widths : sequence of four ints, optional
        Channels of the four stages; by default 64, 128, 256, 512.
    depths : sequence of four ints, optional
        Blocks per stage, in place of the preset's.

    Raises
    ------
    ParameterError
        If ``preset`` is unknown, ``widths`` or ``depths`` is not four positive ints, or ``widths[0]`` is not a
        multiple of 8 for ``"resnet10t"``.
    """

    def __init__(
        self, preset: str = "resnet18", *, widths: Sequence[int] | None = None, depths: Sequence[int] | None = None
    ) -> None:
        if preset not in _PRESETS:
            raise ParameterError(
                f"ResNetEncoder preset must be one of {', '.join(_PRESETS)}, got {preset!r}", path="preset"
            )
        self.preset = preset
        self.widths = _four(widths, default=_WIDTHS, name="widths")
        self.depths = _four(depths, default=_PRESETS[preset][0], name="depths")
        self.tiered = _PRESETS[preset][1]
        if self.tiered and self.widths[0] % 8:
            raise ParameterError(
                f"ResNetEncoder widths[0] must be a multiple of 8 for {preset}, got {self.widths[0]}", path="widths"
            )

    @override
    def module(self, *, in_channels: int | None, norm: Norm) -> EncoderModule:
        return _ResNet(self, in_channels=in_channels, norm=norm)


def _four(values: Sequence[int] | None, *, default: tuple[int, int, int, int], name: str) -> tuple[int, int, int, int]:
    if values is None:
        return default
    counts = tuple(values)
    if len(counts) != 4 or not all(n >= 1 for n in counts):
        raise ParameterError(f"ResNetEncoder {name} must be four positive ints, got {values!r}", path=name)
    a, b, c, d = counts
    return (a, b, c, d)


class _BasicBlock(nn.Module):
    """Two 3x3 convolutions with a shortcut: ``(B, in, Y, X)`` to ``(B, out, Y / stride, X / stride)``."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int, average: bool, norm: Norm) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = block_norm(out_channels, norm)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = block_norm(out_channels, norm)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            if average:
                pool = (
                    nn.AvgPool2d(2, stride, ceil_mode=True, count_include_pad=False) if stride != 1 else nn.Identity()
                )
                layers = [pool, nn.Conv2d(in_channels, out_channels, 1, bias=False)]
            else:
                layers = [nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)]
            self.downsample = nn.Sequential(*layers, block_norm(out_channels, norm))

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class _ResNet(EncoderModule):
    def __init__(self, encoder: ResNetEncoder, *, in_channels: int | None, norm: Norm) -> None:
        super().__init__()
        widths, first = encoder.widths, encoder.widths[0]
        self.own_stem = in_channels is not None
        if in_channels is not None:
            self.conv1 = _stem_convolution(in_channels, first, tiered=encoder.tiered, norm=norm)
            self.bn1 = block_norm(first, norm)
            self.act1 = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        previous = first
        for index, (width, depth) in enumerate(zip(widths, encoder.depths, strict=True)):
            blocks = []
            for block in range(depth):
                stride = 2 if block == 0 and index > 0 else 1
                blocks.append(_BasicBlock(previous, width, stride=stride, average=encoder.tiered, norm=norm))
                previous = width
            setattr(self, f"layer{index + 1}", nn.Sequential(*blocks))
        self.in_channels = first if in_channels is None else in_channels
        self.channels = (first, *widths) if self.own_stem else tuple(widths)
        self.reductions = (2, 4, 8, 16, 32) if self.own_stem else (1, 2, 4, 8)
        _initialize(self)

    @override
    def forward(self, features: Tensor) -> list[Tensor]:
        maps = []
        if self.own_stem:
            features = self.act1(self.bn1(self.conv1(features)))
            maps.append(features)
            features = self.maxpool(features)
        for index in range(1, 5):
            features = getattr(self, f"layer{index}")(features)
            maps.append(features)
        return maps


def _stem_convolution(in_channels: int, width: int, *, tiered: bool, norm: Norm) -> nn.Module:
    """Return the stride-2 convolution of a ResNet stem: one 7x7, or three 3x3 of 3/8, 1/2 and 1/1 of ``width``."""
    if not tiered:
        return nn.Conv2d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False)
    first, second = 3 * width // 8, width // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, first, 3, stride=2, padding=1, bias=False),
        block_norm(first, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(first, second, 3, padding=1, bias=False),
        block_norm(second, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(second, width, 3, padding=1, bias=False),
    )


def _initialize(network: nn.Module) -> None:
    """Initialize as ResNets are: He-normal convolutions, and each block starting as its shortcut."""
    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, _BasicBlock) and isinstance(module.bn2, nn.BatchNorm2d):
            nn.init.zeros_(module.bn2.weight)


@register("timm", kind="encoder")
class TimmEncoder(Encoder):
    """Any ``timm`` backbone, through its ``features_only`` interface.

    With its own stem the backbone returns its feature levels as ``timm`` reports them. Without (a strided network stem
    replaces it), the backbone must be ResNet-style (``conv1``, ``bn1``, ``act1``, ``maxpool``): those are removed and
    the four stage outputs returned. ``timm`` is an optional dependency.

    Parameters
    ----------
    name : str
        The ``timm`` model name, e.g. ``"resnet34"``.
    pretrained : bool
        Start from ``timm``'s pretrained weights. :meth:`arguments` always records ``False``: a bundle holds the
        weights.
    drop_path_rate : float
        Stochastic depth rate while training, for backbones that take one.
    """

    def __init__(self, name: str, *, pretrained: bool = False, drop_path_rate: float = 0.0) -> None:
        if not name:
            raise ParameterError(f"TimmEncoder name must be a timm model name, got {name!r}", path="name")
        self.name, self.pretrained, self.drop_path_rate = name, bool(pretrained), float(drop_path_rate)

    @override
    def module(self, *, in_channels: int | None, norm: Norm) -> EncoderModule:
        return _TimmFeatures(self, in_channels=in_channels, norm=norm)

    def arguments(self) -> dict[str, object]:
        """Return the constructor arguments with ``pretrained`` false: a bundle holds the weights."""
        return {"name": self.name, "pretrained": False, "drop_path_rate": self.drop_path_rate}


_RESNET_STEM = ("conv1", "bn1", "act1", "maxpool")


class _TimmFeatures(EncoderModule):
    def __init__(self, encoder: TimmEncoder, *, in_channels: int | None, norm: Norm) -> None:
        super().__init__()
        import timm
        from timm.layers import freeze_batch_norm_2d

        options: dict[str, Any] = {"features_only": True, "pretrained": encoder.pretrained}
        if in_channels is None:
            options["out_indices"] = (1, 2, 3, 4)  # without the 1/2-resolution level of the stem
        else:
            options["in_chans"] = in_channels
        try:
            created = timm.create_model(encoder.name, drop_path_rate=encoder.drop_path_rate, **options)
        except TypeError:  # a backbone without stochastic depth
            created = timm.create_model(encoder.name, **options)
        model = cast(Any, created)  # timm's feature_info and layer attributes are untyped
        reductions = [int(reduction) for reduction in model.feature_info.reduction()]
        if in_channels is None:
            if not all(hasattr(model, name) for name in _RESNET_STEM):
                raise ParameterError(
                    "a strided stem replaces the stem of a ResNet-style timm encoder (with "
                    f"{', '.join(_RESNET_STEM)}); {encoder.name} has another layout: use a stem of stride 1, such "
                    "as PlaneProjection",
                    path="name",
                )
            self.in_channels = int(model.bn1.num_features)
            for name in _RESNET_STEM:
                setattr(model, name, nn.Identity())
            reductions = [reduction // 4 for reduction in reductions]
        else:
            self.in_channels = in_channels
        self.channels = tuple(int(channels) for channels in model.feature_info.channels())
        self.reductions = tuple(reductions)
        self.model: nn.Module = freeze_batch_norm_2d(model) if norm == "group" else model

    @override
    def forward(self, features: Tensor) -> list[Tensor]:
        return list(self.model(features))
