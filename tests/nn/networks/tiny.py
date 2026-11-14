"""Tiny students for the tests: a few thousand weights, seconds on the CPU."""

from waveorder.nn.networks import (
    Conditioning,
    DefocusKernel,
    PlaneHead,
    PlaneProjection,
    ResNetEncoder,
    SlidingPlanes,
    StridedVolume,
    SubPixelHead,
    UNet,
)
from waveorder.nn.networks.layers import Norm

WIDTHS = (8, 16, 32, 64)


def tiny_unet(*, planes: int = 3, conditioning: Conditioning | None = None, norm: Norm = "batch") -> UNet:
    """A strided U-Net of 8 channels a stage, with batch norm unless told otherwise."""
    return UNet(
        StridedVolume(planes),
        ResNetEncoder(widths=(8, 8, 8, 8), depths=(1, 1, 1, 1)),
        SubPixelHead(1),
        decoder_channels=(8, 8, 8),
        conditioning=conditioning,
        norm=norm,
    )


def encoder() -> ResNetEncoder:
    """A ResNet18 layout with one block per stage and narrow stages."""
    return ResNetEncoder("resnet18", widths=WIDTHS, depths=(1, 1, 1, 1))


def strided(
    planes: int = 5, *, fields: int = 1, conditioning: Conditioning | None = None, halo: int | None = None
) -> UNet:
    """A strided stem and a sub-pixel head: the default layout, nothing at full resolution."""
    return UNet(
        StridedVolume(planes=planes),
        encoder(),
        SubPixelHead(fields=fields),
        decoder_channels=(32, 16, 8),
        conditioning=conditioning,
        norm="group",
        halo=halo,
    )


def projected(planes: int = 3, *, fields: int = 1, conditioning: Conditioning | None = None) -> UNet:
    """A stride-1 projection stem and a plane head: the encoder keeps its own stem."""
    return UNet(
        PlaneProjection(planes=planes, channels=3),
        encoder(),
        PlaneHead(fields=fields),
        decoder_channels=(32, 16, 8, 8, 8),
        conditioning=conditioning,
        norm="group",
    )


def defocus(planes: int = 5, *, conditioning: Conditioning | None = None) -> UNet:
    """A defocus-kernel stem, which reads ``plane_offsets``, and a sub-pixel head."""
    return UNet(
        DefocusKernel(planes=planes, hidden=8),
        encoder(),
        SubPixelHead(),
        decoder_channels=(32, 16, 8),
        conditioning=conditioning,
        norm="group",
    )


def sliding(planes: int = 3, *, fields: int = 1) -> SlidingPlanes:
    """A volume student: a projection student over windows of ``planes`` planes."""
    return SlidingPlanes(projected(planes, fields=fields), planes=planes)
