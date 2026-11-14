"""Stems: how a stack ``(B, C, Z, Y, X)`` becomes the 2D feature map ``(B, channels, Y / stride, X / stride)``.

A 2D network folds the planes of a stack into channels in its stem, so everything about z lives here. A :class:`Stem`
is a description; the network builds its layers (:meth:`Stem.module`) once it knows what the encoder needs.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from waveorder.errors import ParameterError, ShapeError
from waveorder.nn.networks.base import expect_count
from waveorder.nn.networks.layers import Norm, norm_layer
from waveorder.registry import register

PATCH = 4
"""The y and x reduction of a strided stem: the reduction of the encoder stem it replaces."""


class StemModule(nn.Module, ABC):
    """The layers of a stem: a projection, an optional offset per channel, then normalization.

    Attributes
    ----------
    out_channels : int
        Channels of the feature map returned.
    fan_in : int
        Input channels of the projection without condition planes; sets the starting scale of a condition offset.
    """

    out_channels: int
    fan_in: int

    @abstractmethod
    def project(self, data: Tensor, condition: Mapping[str, Tensor] | None, planes: Tensor | None) -> Tensor:
        """Return the projection of ``data`` ``(B, C, Z, Y, X)``, ``(B, out_channels, Y / stride, X / stride)``.

        ``planes`` is the condition vector ``(B, K)`` to append as constant planes, or ``None``; ``condition`` holds
        the named conditions for a stem that reads one itself.
        """

    @abstractmethod
    def finish(self, features: Tensor) -> Tensor:
        """Return the projection normalized (and activated, where the stem replaces an encoder stem)."""

    @override
    def forward(
        self,
        data: Tensor,
        condition: Mapping[str, Tensor] | None = None,
        *,
        planes: Tensor | None = None,
        offset: Tensor | None = None,
    ) -> Tensor:
        """Return the stem's feature map ``(B, out_channels, Y / stride, X / stride)`` of ``data`` ``(B, C, Z, Y, X)``.

        ``planes`` is as :meth:`project` takes it; ``offset`` ``(B, out_channels)`` is added to the projection before
        it is normalized.
        """
        features = self.project(data, condition, planes)
        if offset is not None:
            features = features + offset[:, :, None, None]
        return self.finish(features)


class Stem(ABC):
    """A description of a stem; :meth:`module` builds its layers for one network.

    Attributes
    ----------
    stride : int
        The y and x reduction. A stem of stride 1 feeds an encoder that keeps its own stem; a stem of stride 4
        replaces the encoder's stride-4 stem, so nothing in the network runs above 1/4 resolution.
    takes_planes : bool
        Whether constant condition planes can be appended to its input. A stem that cannot take them reads the
        condition vector as an offset on its output instead.
    data_channels : int
        The number of data channels ``C`` it takes.
    """

    stride: ClassVar[int]
    takes_planes: ClassVar[bool] = True
    data_channels: int

    @property
    def condition_sizes(self) -> Mapping[str, int]:
        """The name and size of every condition the stem reads itself; none by default."""
        return {}

    @abstractmethod
    def planes_out(self, planes_in: int) -> int:
        """Return the planes the network returns for a stack of ``planes_in`` planes: 1 for the built-in stems.

        Raises
        ------
        ShapeError
            If the stem does not take ``planes_in`` planes.
        """

    @abstractmethod
    def module(self, *, width: int | None, condition_planes: int, norm: Norm) -> StemModule:
        """Build the layers.

        ``width`` is the channels the encoder needs from a stem that replaces its own (``None`` for a stride-1 stem,
        which chooses its width); ``condition_planes`` the constant planes appended to the projection's input.
        """


def _expect_planes(data: Tensor, planes: int, *, where: str) -> None:
    if data.shape[2] != planes:
        raise ShapeError(f"{where} takes stacks of {planes} planes, got {data.shape[2]} (data {tuple(data.shape)})")


@register("plane_projection", kind="stem")
class PlaneProjection(Stem):
    """A learned projection of a fixed stack to a 2D image at full resolution, for an encoder with its own stem.

    The ``C * planes`` planes of the stack (and any condition planes) are the input channels of one 3x3 convolution
    with ``channels`` outputs, followed by normalization. The output has the data's ``Y`` and ``X``.

    Parameters
    ----------
    planes : int
        The number of planes ``Z`` of the stacks it takes.
    channels : int
        Channels of the projection, which the encoder takes as its input (3 lets a pretrained encoder keep its
        first layer).
    data_channels : int
        The number of data channels ``C``.

    Raises
    ------
    ParameterError
        If a count is below 1.
    """

    stride = 1

    def __init__(self, planes: int, channels: int, *, data_channels: int = 1) -> None:
        for name, value in (("planes", planes), ("channels", channels), ("data_channels", data_channels)):
            expect_count(value, where="PlaneProjection", name=name)
        self.planes, self.channels, self.data_channels = planes, channels, data_channels

    @override
    def planes_out(self, planes_in: int) -> int:
        if planes_in != self.planes:
            raise ShapeError(f"PlaneProjection takes stacks of {self.planes} planes, got {planes_in}")
        return 1

    @override
    def module(self, *, width: int | None, condition_planes: int, norm: Norm) -> StemModule:
        if width not in (None, self.channels):
            raise ParameterError(
                f"PlaneProjection returns {self.channels} channels, but the encoder needs {width}", path="channels"
            )
        return _PlaneProjection(self, condition_planes=condition_planes, norm=norm)


class _PlaneProjection(StemModule):
    def __init__(self, stem: PlaneProjection, *, condition_planes: int, norm: Norm) -> None:
        super().__init__()
        self.planes = stem.planes
        self.fan_in = stem.data_channels * stem.planes
        self.out_channels = stem.channels
        self.conv = nn.Conv2d(self.fan_in + condition_planes, stem.channels, kernel_size=3, padding=1)
        self.norm = norm_layer(stem.channels, norm)

    @override
    def project(self, data: Tensor, condition: Mapping[str, Tensor] | None, planes: Tensor | None) -> Tensor:
        del condition
        _expect_planes(data, self.planes, where="PlaneProjection")
        folded = data.flatten(1, 2)
        if planes is not None:
            folded = torch.cat([folded, planes[:, :, None, None].expand(-1, -1, *folded.shape[-2:])], dim=1)
        return self.conv(folded)

    @override
    def finish(self, features: Tensor) -> Tensor:
        return self.norm(features)


@register("strided_volume", kind="stem")
class StridedVolume(Stem):
    """One 3D convolution whose kernel equals its stride, ``(kernel_planes, 4, 4)``, replacing the encoder's stem.

    The stack (and any condition planes, repeated over z) goes through one ``Conv3d``; the
    ``planes / kernel_planes`` depth planes left are folded into channels, then normalized and activated. The output
    is at 1/4 resolution with the channels the encoder's first stage takes.

    Parameters
    ----------
    planes : int
        The number of planes ``Z`` of the stacks it takes.
    kernel_planes : int, optional
        Depth of the kernel, a divisor of ``planes``; by default ``planes``. ``planes / kernel_planes`` must divide
        the channels the encoder takes.
    data_channels : int
        The number of data channels ``C``.

    Raises
    ------
    ParameterError
        If a count is below 1, or ``kernel_planes`` does not divide ``planes``.
    """

    stride = PATCH

    def __init__(self, planes: int, *, kernel_planes: int | None = None, data_channels: int = 1) -> None:
        kernel = planes if kernel_planes is None else kernel_planes
        for name, value in (("planes", planes), ("kernel_planes", kernel), ("data_channels", data_channels)):
            expect_count(value, where="StridedVolume", name=name)
        if planes % kernel:
            raise ParameterError(
                f"StridedVolume kernel_planes = {kernel} does not divide planes = {planes}", path="kernel_planes"
            )
        self.planes, self.kernel_planes, self.data_channels = planes, kernel, data_channels

    @override
    def planes_out(self, planes_in: int) -> int:
        if planes_in != self.planes:
            raise ShapeError(f"StridedVolume takes stacks of {self.planes} planes, got {planes_in}")
        return 1

    @override
    def module(self, *, width: int | None, condition_planes: int, norm: Norm) -> StemModule:
        width = _expect_width(width, where="StridedVolume")
        depth = self.planes // self.kernel_planes
        if width % depth:
            raise ParameterError(
                f"StridedVolume leaves planes / kernel_planes = {depth} depth planes, which do not divide the "
                f"{width} channels the encoder takes; choose kernel_planes so that they do",
                path="kernel_planes",
            )
        return _StridedVolume(self, width=width, condition_planes=condition_planes, norm=norm)


def _expect_width(width: int | None, *, where: str) -> int:
    if width is None:
        raise ParameterError(
            f"{where} replaces an encoder's stem and takes its width from the encoder; it cannot be built without one"
        )
    return width


class _StridedVolume(StemModule):
    def __init__(self, stem: StridedVolume, *, width: int, condition_planes: int, norm: Norm) -> None:
        super().__init__()
        self.planes = stem.planes
        self.fan_in = stem.data_channels
        self.out_channels = width
        kernel = (stem.kernel_planes, PATCH, PATCH)
        depth = stem.planes // stem.kernel_planes
        self.conv = nn.Conv3d(stem.data_channels + condition_planes, width // depth, kernel, stride=kernel)
        self.norm = norm_layer(width, norm)

    @override
    def project(self, data: Tensor, condition: Mapping[str, Tensor] | None, planes: Tensor | None) -> Tensor:
        del condition
        _expect_planes(data, self.planes, where="StridedVolume")
        if planes is not None:
            data = torch.cat([data, planes[:, :, None, None, None].expand(-1, -1, *data.shape[-3:])], dim=1)
        return self.conv(data).flatten(1, 2)

    @override
    def finish(self, features: Tensor) -> Tensor:
        return functional.relu(self.norm(features))


DEFOCUS_PERIODS = (16.0, 8.0, 4.0, 2.0)
"""Periods, in axial optical units, of the sines and cosines a :class:`DefocusKernel` encodes the defocus with. The
optics pass no z detail finer than a period of about 2 units; 16 is longer than any stack, so every plane in range
gets a distinct encoding."""


def encode_defocus(defocus: Tensor, z_range: float) -> Tensor:
    """Return the encoding ``(..., N, 1 + 2 * 4)`` of a defocus ``(..., N)`` in axial optical units.

    The defocus divided by ``z_range``, then its sine for each of :data:`DEFOCUS_PERIODS`, then its cosines.
    """
    frequencies = 2 * torch.pi / torch.tensor(DEFOCUS_PERIODS, dtype=defocus.dtype, device=defocus.device)
    phase = defocus.unsqueeze(-1) * frequencies
    return torch.cat([defocus.unsqueeze(-1) / z_range, phase.sin(), phase.cos()], dim=-1)


@register("defocus_kernel", kind="stem")
class DefocusKernel(Stem):
    """A strided stem whose kernel is a learned function of each plane's defocus, for any plane count and spacing.

    The defocus of each plane, in axial optical units, is read from the condition ``condition``
    (:class:`~waveorder.nn.networks.PlaneOffsets`), encoded by :func:`encode_defocus`, and mapped by a small network
    to that plane's 4x4 stride-4 kernels. The stem sums kernel times plane over the planes present, scaled by their
    spacing, so the sum approximates an integral over z whatever the plane count. Planes defocused by more than
    ``z_range`` and padded slots (NaN defocus) get zero weight, so stacks of different depths share a batch once
    padded. It takes no condition planes: a conditioning's vector is an offset on its output.

    Parameters
    ----------
    planes : int
        The most planes a stack may have: the size of the defocus condition.
    hidden : int
        Width of the two hidden layers of the kernel network.
    z_range : float
        The largest defocus, in axial optical units, that gets a kernel; by default half the longest encoding period.
    reference_spacing : float
        Plane spacing, in axial optical units, at which a plane counts once, also assumed for a single plane; 1.344
        is 2 um at a wavelength of 0.45 um and an NA of 0.55.
    data_channels : int
        The number of data channels ``C``.
    condition : str
        Name of the condition that holds the defocus of each plane, ``(B, planes)`` in axial optical units.

    Raises
    ------
    ParameterError
        If a count is below 1, or ``z_range`` or ``reference_spacing`` is not positive.
    """

    stride = PATCH
    takes_planes = False

    def __init__(
        self,
        planes: int,
        *,
        hidden: int = 32,
        z_range: float = DEFOCUS_PERIODS[0] / 2,
        reference_spacing: float = 1.344,
        data_channels: int = 1,
        condition: str = "plane_offsets",
    ) -> None:
        for name, value in (("planes", planes), ("hidden", hidden), ("data_channels", data_channels)):
            expect_count(value, where="DefocusKernel", name=name)
        for name, value in (("z_range", z_range), ("reference_spacing", reference_spacing)):
            if not value > 0:
                raise ParameterError(
                    f"DefocusKernel {name} must be a positive number of axial units, got {value!r}", path=name
                )
        self.planes, self.hidden, self.data_channels = planes, hidden, data_channels
        self.z_range, self.reference_spacing = float(z_range), float(reference_spacing)
        self.condition = condition

    @property
    @override
    def condition_sizes(self) -> Mapping[str, int]:
        return {self.condition: self.planes}

    @override
    def planes_out(self, planes_in: int) -> int:
        if not 1 <= planes_in <= self.planes:
            raise ShapeError(f"DefocusKernel takes stacks of 1 to {self.planes} planes, got {planes_in}")
        return 1

    @override
    def module(self, *, width: int | None, condition_planes: int, norm: Norm) -> StemModule:
        if condition_planes:
            raise ParameterError("DefocusKernel cannot take condition planes", path="condition_planes")
        return _DefocusKernel(self, width=_expect_width(width, where="DefocusKernel"), norm=norm)


class _DefocusKernel(StemModule):
    def __init__(self, stem: DefocusKernel, *, width: int, norm: Norm) -> None:
        super().__init__()
        self.planes, self.condition = stem.planes, stem.condition
        self.z_range, self.reference_spacing = stem.z_range, stem.reference_spacing
        self.fan_in = self.data_channels = stem.data_channels
        self.out_channels = width
        self.kernel = nn.Sequential(
            nn.Linear(1 + 2 * len(DEFOCUS_PERIODS), stem.hidden),
            nn.GELU(),
            nn.Linear(stem.hidden, stem.hidden),
            nn.GELU(),
            nn.Linear(stem.hidden, width * stem.data_channels * PATCH * PATCH),
        )
        self.norm = norm_layer(width, norm)

    def kernels(self, offsets: Tensor) -> Tensor:
        """Return the kernels ``(B, out_channels, C, N, 4, 4)`` at the defocus ``offsets`` ``(B, N)`` (NaN: padding)."""
        valid = torch.isfinite(offsets) & (offsets.abs() <= self.z_range)
        defocus = torch.where(valid, offsets, torch.zeros_like(offsets))
        weights = self.kernel(encode_defocus(defocus, self.z_range)) * valid.unsqueeze(-1).to(defocus.dtype)
        # The spacing of the planes present, (max - min) / (n - 1); a single plane takes the reference spacing.
        count = valid.sum(1)
        low = torch.where(valid, defocus, torch.full_like(defocus, float("inf"))).amin(1)
        high = torch.where(valid, defocus, torch.full_like(defocus, float("-inf"))).amax(1)
        spacing = torch.where(
            count > 1, (high - low) / (count - 1).clamp(min=1), torch.full_like(low, self.reference_spacing)
        )
        weights = weights * (spacing / self.reference_spacing)[:, None, None]
        batch, planes = offsets.shape
        return weights.view(batch, planes, self.out_channels, self.data_channels, PATCH, PATCH).permute(
            0, 2, 3, 1, 4, 5
        )

    @override
    def project(self, data: Tensor, condition: Mapping[str, Tensor] | None, planes: Tensor | None) -> Tensor:
        del planes
        batch, channels, count, height, width = data.shape
        offsets = None if condition is None else condition.get(self.condition)
        if offsets is None:
            raise ShapeError(
                f"DefocusKernel reads the defocus of each plane from the condition {self.condition!r}, but none was "
                f"given; pass condition={{{self.condition!r}: tensor of shape ({batch}, {self.planes})}} in axial "
                "optical units"
            )
        if tuple(offsets.shape) != (batch, self.planes):
            raise ShapeError(
                f"the condition {self.condition!r} has shape {tuple(offsets.shape)}, expected ({batch}, {self.planes})"
            )
        if not 1 <= count <= self.planes:
            raise ShapeError(f"DefocusKernel takes stacks of 1 to {self.planes} planes, got {count}")
        kernels = self.kernels(offsets[:, :count].to(dtype=data.dtype, device=data.device))
        kernels = kernels.reshape(batch * self.out_channels, channels * count, PATCH, PATCH)
        stacked = data.reshape(1, batch * channels * count, height, width)
        features = functional.conv2d(stacked, kernels, stride=PATCH, groups=batch)
        return features.view(batch, self.out_channels, height // PATCH, width // PATCH)

    @override
    def finish(self, features: Tensor) -> Tensor:
        return functional.relu(self.norm(features))
