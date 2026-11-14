"""Conditions (named vectors a student reads besides the data), how they enter a network, and sliding planes.

A :class:`Condition` computes its vector from a :class:`Place`, alike in training and at inference. A network reads
its conditions as one vector ``(B, K)``: the named ``(B, k)`` values concatenated in the order its
:class:`Conditioning` declares. That order is part of the trained weights.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Self, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from waveorder.errors import ParameterError, ShapeError
from waveorder.nn.context import SolveContext
from waveorder.nn.model import ImagingModel
from waveorder.nn.networks.base import RestorationNetwork, expect_count
from waveorder.nn.sampling import Sampling
from waveorder.registry import Spec, register, resolve


@dataclass(frozen=True, slots=True, kw_only=True)
class Place:
    """Where a block of data sits: what a condition may read.

    Parameters
    ----------
    field_shape : tuple[int, int]
        Pixels of the whole field of view along y and x.
    region : tuple[slice, slice]
        The y and x extent of the block within the field; the whole field when the data is not tiled or cropped.
    sampling : Sampling
        The sampling of the block: its ``(Z, Y, X)`` shape and voxel size in micrometres.
    attributes : Mapping[str, float]
        Values per position, by name: index columns in training, the result of a user function at inference.

    Raises
    ------
    ShapeError
        If the region does not lie within the field, or its extent is not the block's ``(Y, X)``.
    """

    field_shape: tuple[int, int]
    region: tuple[slice, slice]
    sampling: Sampling
    attributes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for axis, extent, size, block in zip("yx", self.region, self.field_shape, self.sampling.shape[1:], strict=True):
            if not 0 <= extent.start < extent.stop <= size:
                raise ShapeError(
                    f"Place region along {axis} is {extent.start}:{extent.stop}, outside the field's 0:{size}"
                )
            if extent.stop - extent.start != block:
                raise ShapeError(
                    f"Place region along {axis} spans {extent.stop - extent.start} pixels, but the block has {block}"
                )

    @classmethod
    def of(
        cls, sampling: Sampling, *, context: SolveContext | None = None, attributes: Mapping[str, float] | None = None
    ) -> Self:
        """Return the place of a block with ``sampling`` in the solve ``context``.

        Without a context, or with one that names neither a field nor a region, the block is the whole field.

        Raises
        ------
        ShapeError
            If the context's region or field does not fit the block, or names a region without a field.
        """
        _, height, width = sampling.shape
        context = context or SolveContext()
        if context.region is not None and context.field_shape is None:
            raise ShapeError(
                "the solve context names a region but no field: set both, solve_context(field_shape=(Y, X), region=...)"
            )
        field_shape = context.field_shape or (height, width)
        region = context.region or (slice(0, field_shape[0]), slice(0, field_shape[1]))
        try:
            return cls(field_shape=field_shape, region=region, sampling=sampling, attributes=dict(attributes or {}))
        except ShapeError as error:
            raise ShapeError(
                f"the solve context (field_shape={context.field_shape}, region={context.region}) does not describe "
                f"data of Y, X = {(height, width)}: {error}. Set the region the data covers with "
                "solve_context(field_shape=(Y, X), region=(slice(y0, y1), slice(x0, x1)))"
            ) from error


class Condition(ABC):
    """A named vector computed from where a block sits and, for some, from the imaging model.

    Attributes
    ----------
    name : str
        The name a student reads it by.
    size : int
        The number of values, ``K``.
    """

    name: str
    size: int

    @abstractmethod
    def values(self, place: Place, model: ImagingModel | None) -> Tensor:
        """Return the condition at ``place``: ``(size,)``, float32, on the CPU.

        ``model`` is the imaging model being solved, or ``None`` where there is none (a loader worker).
        """


def _names(names: Sequence[str], *, where: str) -> tuple[str, ...]:
    if not names or len(set(names)) != len(names):
        raise ParameterError(f"{where} names must be distinct strings, at least one, got {names!r}", path="names")
    return tuple(names)


def _per_name(value: float | Sequence[float], count: int, *, where: str, name: str) -> tuple[float, ...]:
    """Return ``value`` once per name, refusing a non-finite value, a wrong length or (as ``scale``) a zero."""
    values = [value] * count if isinstance(value, int | float) else list(value)
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ParameterError(
            f"{where} {name} must be a finite number, or one for each of the {count} values, got {value!r}", path=name
        )
    if name == "scale" and 0 in values:
        raise ParameterError(f"{where} scale must not be 0, got {value!r}", path="scale")
    return tuple(float(v) for v in values)


@register("field_position", kind="condition")
class FieldPosition(Condition):
    """The centre of the block in its field, as ``(centre - offset) / scale`` per axis.

    The centre is in [0, 1] of the field along each axis (0.5 for the whole field), ordered ``(y, x)`` by default.
    With ``axes="xy"``, ``offset=0.5`` and ``scale=0.4375`` the values are ``(x, y)`` in [-1, 1] at the centres of the
    outer patches of an 8 x 8 grid.

    Parameters
    ----------
    axes : {"yx", "xy"}
        The order of the two values.
    offset, scale : float or sequence of two floats
        Subtracted from, then divided into, each centre; one number for both axes or one per axis, in the order of
        ``axes``. ``scale`` must not be 0.
    name : str
        The name of the condition.
    """

    name = "field_position"
    size = 2

    def __init__(
        self,
        *,
        axes: Literal["yx", "xy"] = "yx",
        offset: float | Sequence[float] = 0.0,
        scale: float | Sequence[float] = 1.0,
        name: str = "field_position",
    ) -> None:
        if axes not in ("yx", "xy"):
            raise ParameterError(f"FieldPosition axes must be 'yx' or 'xy', got {axes!r}", path="axes")
        self.axes = axes
        self.offset = _per_name(offset, 2, where="FieldPosition", name="offset")
        self.scale = _per_name(scale, 2, where="FieldPosition", name="scale")
        self.name = name

    @override
    def values(self, place: Place, model: ImagingModel | None) -> Tensor:
        del model
        extents = zip(place.region, place.field_shape, strict=True)
        centre = [(extent.start + extent.stop) / 2 / size for extent, size in extents]
        ordered = centre if self.axes == "yx" else centre[::-1]
        scaled = [
            (value - offset) / scale for value, offset, scale in zip(ordered, self.offset, self.scale, strict=True)
        ]
        return torch.tensor(scaled, dtype=torch.float32)


@register("attributes", kind="condition")
class Attributes(Condition):
    """Values per position looked up by name, as ``(value - offset) / scale``: well coordinates, say.

    In training the values are numeric index columns; at inference, what the ``attributes`` function of
    :class:`~waveorder.nn.Distilled` returns for the solve context.

    Parameters
    ----------
    names : sequence of str
        The distinct attributes read, in the order of the values returned.
    offset, scale : float or sequence of float
        Subtracted from, then divided into, each value; one number for all names or one per name. ``scale`` must
        not be 0.
    name : str
        The name of the condition, so a student can read several sets of attributes.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        offset: float | Sequence[float] = 0.0,
        scale: float | Sequence[float] = 1.0,
        name: str = "attributes",
    ) -> None:
        self.names = _names(names, where="Attributes")
        self.offset = _per_name(offset, len(self.names), where="Attributes", name="offset")
        self.scale = _per_name(scale, len(self.names), where="Attributes", name="scale")
        self.name = name
        self.size = len(self.names)

    @override
    def values(self, place: Place, model: ImagingModel | None) -> Tensor:
        del model
        missing = [name for name in self.names if name not in place.attributes]
        if missing:
            raise ParameterError(
                f"the condition {self.name!r} reads the attributes {missing}, which this place does not have (it "
                f"has {sorted(place.attributes)}); at inference, pass Distilled(attributes=function) returning them "
                "for a solve context, in training, add them as numeric columns of the index",
                path="attributes",
            )
        pairs = zip(self.names, self.offset, self.scale, strict=True)
        scaled = [(float(place.attributes[name]) - offset) / scale for name, offset, scale in pairs]
        return torch.tensor(scaled, dtype=torch.float32)


@register("plane_offsets", kind="condition")
class PlaneOffsets(Condition):
    """The defocus of each plane in axial optical units, padded with NaN to ``planes`` entries.

    The axial optical unit is ``n * wavelength / NA ** 2``, so one number means the same blur for any objective and
    wavelength. The focus plane sits ``focus_offset`` planes from plane ``Z // 2`` and z runs down the stack (as the
    detection's ``z_focus_offset``), so plane ``i`` spaced ``dz`` um has the defocus
    ``(Z // 2 + focus_offset - i) * dz / unit``. A :class:`~waveorder.nn.networks.DefocusKernel` reads NaN as padding.

    Parameters
    ----------
    planes : int
        The size of the condition: the most planes a stack may have.
    wavelength : float
        The detection wavelength in um.
    numerical_aperture : float
        The detection's numerical aperture.
    index_of_refraction : float
        The index of refraction of the medium.
    focus_offset : float or str
        The focus plane in planes from ``Z // 2``, or the name of the position attribute that holds it.

    Raises
    ------
    ParameterError
        If ``planes`` is below 1 or an optical value is not positive.
    """

    name = "plane_offsets"

    def __init__(
        self,
        planes: int,
        *,
        wavelength: float,
        numerical_aperture: float,
        index_of_refraction: float = 1.0,
        focus_offset: float | str = 0.0,
    ) -> None:
        expect_count(planes, where="PlaneOffsets", name="planes")
        optics = {"wavelength": wavelength, "numerical_aperture": numerical_aperture}
        for name, value in {**optics, "index_of_refraction": index_of_refraction}.items():
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise ParameterError(f"PlaneOffsets {name} must be a positive number, got {value!r}", path=name)
        self.planes = self.size = planes
        self.wavelength, self.numerical_aperture = float(wavelength), float(numerical_aperture)
        self.index_of_refraction, self.focus_offset = float(index_of_refraction), focus_offset

    @property
    def unit(self) -> float:
        """The axial optical unit in um: ``index_of_refraction * wavelength / numerical_aperture ** 2``."""
        return self.index_of_refraction * self.wavelength / self.numerical_aperture**2

    @override
    def values(self, place: Place, model: ImagingModel | None) -> Tensor:
        """Return the defocus ``(planes,)``.

        Raises
        ------
        ShapeError
            If the stack is deeper than ``planes``.
        ParameterError
            If ``focus_offset`` names an attribute the place does not have.
        """
        del model
        depth, spacing = place.sampling.shape[0], place.sampling.pixel_size[0]
        if depth > self.planes:
            raise ShapeError(f"PlaneOffsets holds {self.planes} planes, but the stack has {depth}")
        focus = self.focus_offset
        if isinstance(focus, str):
            if focus not in place.attributes:
                raise ParameterError(
                    f"PlaneOffsets reads the focus offset from the attribute {focus!r}, which this place does not "
                    f"have (it has {sorted(place.attributes)})",
                    path="focus_offset",
                )
            focus = float(place.attributes[focus])
        offsets = torch.full((self.planes,), float("nan"), dtype=torch.float32)
        planes = torch.arange(depth, dtype=torch.float64)
        offsets[:depth] = ((depth // 2 + focus - planes) * spacing / self.unit).to(torch.float32)
        return offsets


@register("quantities", kind="condition")
class Quantities(Condition):
    """The current values of named scalar quantities of the imaging system, e.g. an illumination tilt.

    Parameters
    ----------
    names : sequence of str
        Distinct dotted names relative to ``model.system``, e.g. ``"illumination.numerical_aperture"``, in the
        order of the values returned.
    name : str
        The name of the condition.
    """

    def __init__(self, names: Sequence[str], *, name: str = "quantities") -> None:
        self.names = _names(names, where="Quantities")
        self.name = name
        self.size = len(self.names)

    @override
    def values(self, place: Place, model: ImagingModel | None) -> Tensor:
        """Return the quantities ``(size,)``; reading them from a CUDA model synchronizes once.

        Raises
        ------
        ParameterError
            If there is no model, or a name is not a one-element tensor of its system.
        """
        del place
        if model is None:
            raise ParameterError(
                f"the condition {self.name!r} reads quantities of the imaging system, but no model was given",
                path="model",
            )
        found = []
        for name in self.names:
            value: object = model.system
            for part in name.split("."):
                value = getattr(value, part, None)
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ParameterError(
                    f"the condition {self.name!r} reads {name!r}, which is not a scalar quantity of "
                    f"{type(model.system).__name__}",
                    path=name,
                )
            found.append(value.detach().reshape(()))
        return torch.stack(found).to(device="cpu", dtype=torch.float32)


type Sizes = Mapping[str, int] | Sequence[tuple[str, int]]
"""Condition sizes in order: a mapping ``{name: size}`` or a sequence of ``(name, size)`` pairs."""


class Conditioning(ABC):
    """Which conditions a network reads, in which order, and how their values enter it.

    Parameters
    ----------
    sizes : mapping or sequence of pairs
        The size ``k`` of each named condition, in the order they are concatenated; ``K`` is their sum.

    Attributes
    ----------
    planes : bool
        The vector enters as ``K`` constant planes, appended to the stem's input (when the stem takes planes) and to
        every encoder feature map on its way into the decoder.
    offsets : bool
        The vector enters as a learned per-channel offset, added after the stem's convolution and after the first
        convolution of every decoder block.

    Raises
    ------
    ParameterError
        If ``sizes`` is empty, names a condition twice, or holds a size below 1.
    """

    planes: ClassVar[bool]
    offsets: ClassVar[bool]

    def __init__(self, sizes: Sizes) -> None:
        pairs = list(sizes.items()) if isinstance(sizes, Mapping) else list(sizes)
        where = type(self).__name__
        self.sizes: dict[str, int] = {}
        for pair in pairs:
            if len(pair) != 2:
                raise ParameterError(f"{where} sizes must be {{name: size}} or (name, size) pairs, got {pair!r}")
            name, size = pair
            expect_count(size, where=where, name=f"sizes.{name}")
            if name in self.sizes:
                raise ParameterError(f"{where} sizes names the condition {name!r} twice", path=f"sizes.{name}")
            self.sizes[name] = size
        if not self.sizes:
            raise ParameterError(f"{where} sizes is empty; leave conditioning out for a network without conditions")

    @property
    def size(self) -> int:
        """``K``, the length of the vector: the sum of the sizes."""
        return sum(self.sizes.values())

    def vector(self, condition: Mapping[str, Tensor] | None, *, like: Tensor) -> Tensor:
        """Return the conditions concatenated in declared order, ``(B, K)``, in the dtype and on the device of ``like``.

        ``condition`` holds ``(B, k)`` values by name; undeclared names are ignored. ``like`` is the data.

        Raises
        ------
        ShapeError
            If a declared condition is missing or is not ``(B, k)`` with its declared size.
        """
        batch = like.shape[0]
        parts = []
        for name, size in self.sizes.items():
            value = None if condition is None else condition.get(name)
            if value is None:
                raise ShapeError(
                    f"the network reads the condition {name!r} ({size} values per item), but none was given; pass "
                    f"condition={{{name!r}: tensor of shape ({batch}, {size}), ...}}"
                )
            if tuple(value.shape) != (batch, size):
                raise ShapeError(f"the condition {name!r} has shape {tuple(value.shape)}, expected ({batch}, {size})")
            parts.append(value.to(dtype=like.dtype, device=like.device))
        return torch.cat(parts, dim=1)

    def arguments(self) -> dict[str, object]:
        """Return ``{"sizes": [[name, size], ...]}``: the order survives any JSON writer."""
        return {"sizes": [[name, size] for name, size in self.sizes.items()]}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sizes={self.sizes})"


@register("concat_planes", kind="conditioning")
class ConcatPlanes(Conditioning):
    """Conditions as constant planes: each value of the vector is broadcast to a plane and concatenated.

    A stem that cannot take planes (:class:`~waveorder.nn.networks.DefocusKernel`) gets an offset instead.
    """

    planes = True
    offsets = False


@register("bias", kind="conditioning")
class Bias(Conditioning):
    """Conditions as offsets: a linear layer without bias maps the vector to one offset per channel."""

    planes = False
    offsets = True


def condition_offset(size: int, channels: int, *, fan_in: int) -> nn.Linear:
    """Return the linear layer, without bias, that maps a condition vector ``(B, size)`` to offsets ``(B, channels)``.

    Its weights start uniform within ``fan_in ** -0.5``, the scale constant planes have through a default-initialized
    convolution; ``nn.Linear``'s default would start several times larger.
    """
    offset = nn.Linear(size, channels, bias=False)
    bound = fan_in**-0.5
    nn.init.uniform_(offset.weight, -bound, bound)
    return offset


@register("sliding_planes", kind="network")
class SlidingPlanes(RestorationNetwork):
    """Any student that takes ``planes`` planes and returns one, used as ``Z`` planes in, ``Z`` planes out.

    Output plane ``z`` is the student's estimate from the ``planes`` data planes centred on ``z``; the first and last
    planes are repeated past the ends of the stack. Every window gets its stack's conditions unchanged, so a condition
    must not depend on which planes a window holds.

    Parameters
    ----------
    student : RestorationNetwork
        The plane student, or the spec ``{name: arguments}`` that describes one.
    planes : int
        The window: an odd number of planes that ``student`` takes and maps to one plane.
    chunk : int, optional
        The most windows passed to ``student`` in one call, to bound memory; by default all ``B * Z`` at once.

    Raises
    ------
    ParameterError
        If ``planes`` is not positive and odd, ``chunk`` is below 1, or ``student`` does not map ``planes`` planes
        to one.
    """

    def __init__(self, student: RestorationNetwork | Spec, *, planes: int, chunk: int | None = None) -> None:
        super().__init__()
        if planes < 1 or planes % 2 == 0:
            raise ParameterError(f"SlidingPlanes planes must be a positive odd int, got {planes!r}", path="planes")
        if chunk is not None:
            expect_count(chunk, where="SlidingPlanes", name="chunk")
        self.student = resolve(student, "network", RestorationNetwork)
        try:
            planes_out = self.student.planes_out(planes)
        except ShapeError as error:
            raise ParameterError(f"SlidingPlanes planes = {planes}: {error}", path="planes") from error
        if planes_out != 1:
            raise ParameterError(
                f"SlidingPlanes needs a student that returns one plane from {planes}; "
                f"{type(self.student).__name__} returns {planes_out}",
                path="student",
            )
        self.planes, self.chunk = planes, chunk
        self.channels, self.fields = self.student.channels, self.student.fields
        self.condition_sizes = dict(self.student.condition_sizes)

    @property
    @override
    def stride(self) -> int:
        """The student's stride."""
        return self.student.stride

    @override
    def halo(self) -> int:
        """Return the student's halo."""
        return self.student.halo()

    @override
    def halo_planes(self) -> int:
        """Return half the window, rounded down: the planes on each side that an output plane reads."""
        return self.planes // 2

    @override
    def planes_out(self, planes_in: int) -> int:
        """Return ``planes_in``: one output plane per data plane; raise ``ShapeError`` below 1."""
        if planes_in < 1:
            raise ShapeError(f"SlidingPlanes takes stacks of at least one plane, got {planes_in}")
        return planes_in

    @override
    def forward(self, data: Tensor, condition: Mapping[str, Tensor] | None = None) -> Tensor:
        if data.dim() != 5:
            raise ShapeError(f"SlidingPlanes data has shape {tuple(data.shape)}, expected (B, C, Z, Y, X)")
        batch, channels, depth, height, width = data.shape
        half = self.planes // 2
        padded = functional.pad(data, (0, 0, 0, 0, half, half), mode="replicate") if half else data
        # (B, C, Z, Y, X, planes) -> (B * Z, C, planes, Y, X): the windows of one stack are consecutive items.
        windows = padded.unfold(2, self.planes, 1).permute(0, 2, 1, 5, 3, 4)
        windows = windows.reshape(batch * depth, channels, self.planes, height, width)
        repeated = {name: value.repeat_interleave(depth, dim=0) for name, value in (condition or {}).items()}
        size = self.chunk or batch * depth
        estimates = [
            self.student(windows[start : start + size], {name: v[start : start + size] for name, v in repeated.items()})
            for start in range(0, batch * depth, size)
        ]
        estimate = torch.cat(estimates) if len(estimates) > 1 else estimates[0]  # (B * Z, F, 1, Y, X)
        return estimate.reshape(batch, depth, self.fields, height, width).transpose(1, 2)
