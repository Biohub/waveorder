"""The network interface with its shape check, and the contract that records what a student was trained for."""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Self, override

import torch
from torch import Tensor, nn

from waveorder import registry
from waveorder.errors import ParameterError, ShapeError
from waveorder.nn.model import ImagingModel
from waveorder.nn.sampling import ZYX, Sampling

type Precision = Literal["float32", "bf16", "fp16"]
"""What a student computes in: float32, or autocast to bfloat16 (``"bf16"``) or to float16 (``"fp16"``)."""

PRECISIONS: tuple[Precision, ...] = ("float32", "bf16", "fp16")
"""The values of :data:`Precision`."""


def expect_count(value: int, *, where: str, name: str, minimum: int = 1) -> None:
    """Raise :class:`~waveorder.errors.ParameterError` unless the count ``value`` is at least ``minimum``."""
    if value < minimum:
        raise ParameterError(f"{where} {name} must be an int of at least {minimum}, got {value!r}", path=name)


class Case(NamedTuple):
    """One forward pass :meth:`RestorationNetwork.check_shapes` ran: the input was ``(batch, C, planes, *size)``."""

    planes: int
    size: tuple[int, int]
    batch: int


class RestorationNetwork(nn.Module, ABC):
    """A network mapping data ``(B, C, Z, Y, X)`` and named conditions to an object estimate ``(B, F, Zout, Y, X)``.

    The estimate has the data's ``Y`` and ``X``; ``Zout`` is 1 for a plane network or ``Z`` for a volume one
    (:meth:`planes_out`). Each condition is a ``(B, K)`` tensor read by name.

    Attributes
    ----------
    channels : int
        The number of data channels ``C`` it takes.
    fields : int
        The number of object fields ``F`` it returns.
    condition_sizes : Mapping[str, int]
        The name and the size ``K`` of every condition it reads; empty for none.
    """

    channels: int
    fields: int
    condition_sizes: Mapping[str, int]

    @property
    def stride(self) -> int:
        """The number ``Y`` and ``X`` must be multiples of; 1 unless the network downsamples."""
        return 1

    @abstractmethod
    def halo(self) -> int:
        """Return the pixels of context along y and x that a tile needs on each side."""

    def halo_planes(self) -> int:
        """Return the planes of context along z that a tile needs on each side; 0 unless the student slides along z."""
        return 0

    @abstractmethod
    def planes_out(self, planes_in: int) -> int:
        """Return ``Zout`` for data of ``planes_in`` planes.

        Raises
        ------
        ShapeError
            If the student does not take ``planes_in`` planes.
        """

    @override
    @abstractmethod
    def forward(self, data: Tensor, condition: Mapping[str, Tensor] | None = None) -> Tensor:
        """Return the estimate ``(B, F, Zout, Y, X)`` of ``data`` ``(B, C, Z, Y, X)`` under ``condition``.

        Raises
        ------
        ShapeError
            If ``data`` is not five-dimensional with ``C`` channels, a plane count the student takes and ``Y`` and
            ``X`` multiples of :attr:`stride`, or a condition it reads is missing or not ``(B, K)``.
        """

    def check_shapes(
        self,
        *,
        planes: Sequence[int],
        sizes: Sequence[tuple[int, int]],
        batch_sizes: Sequence[int] = (1, 2),
        conditions: Mapping[str, Tensor] | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> list[Case]:
        """Run the network over the shapes an instance must accept, and raise for the first case it gets wrong.

        For the tests of a new network or part; :class:`waveorder.models.DistillationModule` runs it before training.
        The cases are every combination of ``planes``, ``batch_sizes`` and ``sizes`` plus one taller and one wider
        size, run in eval mode without gradients on random data. Checked: the class spec is JSON and rebuilds a
        network with the same state shapes (what loading a bundle needs); every case returns a finite
        ``(B, F, planes_out(Z), Y, X)``; a sample's output does not depend on its batch; a plane count
        :meth:`planes_out` refuses is refused by the forward pass too. The network is left in its mode and on its
        device.

        Parameters
        ----------
        planes : sequence of int
            Plane counts ``Z`` the instance must take.
        sizes : sequence of (int, int)
            ``(Y, X)`` sizes, multiples of :attr:`stride`; keep them small.
        batch_sizes : sequence of int
            Batch sizes to run.
        conditions : Mapping[str, Tensor], optional
            One ``(K,)`` value per condition the network reads, used for every sample; by default uniform in [0, 1).
        device : str or torch.device
            Where to run.
        seed : int
            Seed of the random data.

        Returns
        -------
        list[Case]
            The cases that ran, in order.

        Raises
        ------
        AssertionError
            If the network breaks the contract; the message names the failing case.
        ParameterError
            If ``planes``, ``sizes`` or ``batch_sizes`` is empty, a size is not a multiple of :attr:`stride`, or
            ``conditions`` does not match :attr:`condition_sizes`.
        """
        name = type(self).__name__
        all_sizes = _case_sizes(sizes, self.stride)
        if not planes or not batch_sizes:
            raise ParameterError("check_shapes needs at least one plane count and one batch size", path="planes")
        wanted = {key: (size,) for key, size in self.condition_sizes.items()}
        if conditions is not None and {key: tuple(value.shape) for key, value in conditions.items()} != wanted:
            raise ParameterError(
                f"check_shapes conditions must give one (K,) tensor for each condition the network reads, "
                f"{dict(self.condition_sizes)}, got shapes {[tuple(value.shape) for value in conditions.values()]}",
                path="conditions",
            )
        _check_rebuild(self)
        was_training = self.training
        origin = next((parameter.device for parameter in self.parameters()), None)
        generator = torch.Generator().manual_seed(seed)
        self.to(device).eval()
        cases = [Case(count, size, batch) for count in planes for size in all_sizes for batch in batch_sizes]
        try:
            with torch.no_grad():
                refused = [count for count in range(1, max(planes) + 2) if _refuses(self, count)]
                required = [count for count in refused if count in planes]
                if required:
                    raise AssertionError(
                        f"check_shapes: {name}.planes_out({required[0]}) raises ShapeError, but {required[0]} is "
                        f"one of the plane counts {tuple(planes)} the instance must take"
                    )
                for case in cases:
                    _check_case(self, case, _inputs(self, case, conditions, generator=generator))
                for count in refused:
                    case = Case(count, all_sizes[0], 1)
                    _check_refusal(self, case, _inputs(self, case, conditions, generator=generator))
        finally:
            self.train(was_training)
            if origin is not None:
                self.to(origin)
        return cases


_TOLERANCE = 1e-4
"""Largest difference, relative to the output's largest magnitude, between a sample run alone and in a batch."""


def _case_sizes(sizes: Sequence[tuple[int, int]], stride: int) -> list[tuple[int, int]]:
    """Return the given sizes followed by a taller and a wider one, each a multiple of ``stride``."""
    if not sizes:
        raise ParameterError("check_shapes needs at least one (Y, X) size", path="sizes")
    given = [(int(y), int(x)) for y, x in sizes]
    for y, x in given:
        if y < 1 or x < 1 or y % stride or x % stride:
            raise ParameterError(
                f"check_shapes size {(y, x)} is not a positive multiple of the network's stride {stride} along both "
                f"axes; pass sizes such as {(stride * max(1, y // stride), stride * max(1, x // stride))} "
                "(Distilled pads data to the stride at inference)",
                path="sizes",
            )
    y, x = min(given, key=lambda size: size[0] * size[1])
    derived = [(y + stride, x), (y, x + 2 * stride)]
    return [*given, *(size for size in derived if size not in given)]


def _inputs(
    network: RestorationNetwork, case: Case, conditions: Mapping[str, Tensor] | None, *, generator: torch.Generator
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return random data of the case's shape and its conditions, on the network's device."""
    device = next((parameter.device for parameter in network.parameters()), None)
    data = torch.randn((case.batch, network.channels, case.planes, *case.size), generator=generator)
    condition = {}
    for key, size in network.condition_sizes.items():
        if conditions is None:
            values = torch.rand((case.batch, size), generator=generator)
        else:
            values = conditions[key].to(torch.float32).expand(case.batch, size).clone()
        condition[key] = values.to(device)
    return data.to(device), condition


def _check_case(network: RestorationNetwork, case: Case, inputs: tuple[Tensor, dict[str, Tensor]]) -> None:
    where = f"check_shapes: {type(network).__name__} with {case}"
    data, condition = inputs
    try:
        planes_out = network.planes_out(case.planes)
        output = network(data, condition or None)
    except Exception as error:
        raise AssertionError(f"{where}: the forward pass raised {type(error).__name__}: {error}") from error
    expected = (case.batch, network.fields, planes_out, *case.size)
    if not isinstance(output, Tensor) or tuple(output.shape) != expected:
        got = tuple(output.shape) if isinstance(output, Tensor) else type(output).__name__
        raise AssertionError(f"{where}: the output is {got}, expected (B, F, planes_out, Y, X) = {expected}")
    if not bool(torch.isfinite(output).all()):
        raise AssertionError(f"{where}: the output has non-finite values for finite input")
    scale = max(float(output.abs().max()), 1.0)
    for index in {0, case.batch - 1} if case.batch > 1 else ():
        alone = network(data[index : index + 1], {key: value[index : index + 1] for key, value in condition.items()})
        difference = float((alone[0] - output[index]).abs().max())
        if difference > _TOLERANCE * scale:
            raise AssertionError(
                f"{where}: sample {index} gives another output alone than in its batch (max difference "
                f"{difference:.3g}); in eval mode a sample must not depend on its batch mates (batch statistics, a "
                "normalization over the batch axis)"
            )


def _refuses(network: RestorationNetwork, planes: int) -> bool:
    try:
        network.planes_out(planes)
    except ShapeError:
        return True
    return False


def _check_refusal(network: RestorationNetwork, case: Case, inputs: tuple[Tensor, dict[str, Tensor]]) -> None:
    where = f"check_shapes: {type(network).__name__} with {case}: planes_out({case.planes}) raises ShapeError, but"
    data, condition = inputs
    advice = "refuse the plane count with ShapeError in forward too"
    try:
        network(data, condition or None)
    except ShapeError:
        return
    except Exception as error:
        raise AssertionError(f"{where} the forward pass raised {type(error).__name__} ({error}); {advice}") from error
    raise AssertionError(f"{where} the forward pass accepted {case.planes} planes; {advice}")


def _check_rebuild(network: RestorationNetwork) -> None:
    name = type(network).__name__
    try:
        spec = registry.class_spec(network)
        json.dumps(spec)
    except (TypeError, ValueError, ParameterError) as error:
        raise AssertionError(
            f"check_shapes: the arguments of {name} are not JSON ({error}); a bundle stores them in bundle.json"
        ) from error
    try:
        rebuilt = registry.build(spec)
    except Exception as error:
        raise AssertionError(
            f"check_shapes: rebuilding {name} from its class spec raised {type(error).__name__}: {error}; a bundle "
            "rebuilds the network that way"
        ) from error
    assert isinstance(rebuilt, RestorationNetwork)
    ours = {key: tuple(value.shape) for key, value in network.state_dict().items()}
    theirs = {key: tuple(value.shape) for key, value in rebuilt.state_dict().items()}
    if ours != theirs:
        differing = sorted(key for key in ours.keys() | theirs.keys() if ours.get(key) != theirs.get(key))
        raise AssertionError(
            f"check_shapes: {name} rebuilt from its class spec has another state than the network, so its weights "
            f"would not load from a bundle; {len(differing)} entries differ, the first: {differing[:5]}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Contract:
    """What a student was trained for: recorded with it in the bundle, checked on every solve.

    "The model" is the reconstruction model the training targets were made with; the student runs in it as its
    solver.

    Parameters
    ----------
    channels : int
        The number of data channels the student takes.
    planes : tuple[int, ...] or None
        The plane counts ``Z`` accepted; ``None`` for whatever the student accepts.
    fields : tuple[str, ...]
        Names of the object fields the student returns, in output order: some or all of the system's fields.
    pixel_size : tuple[float, float, float]
        Voxel size along z, y, x, in micrometres, of the data the student was trained on.
    pixel_size_tolerance : float
        Relative difference in voxel size, per axis, up to which data is accepted.
    system : Mapping[str, object]
        ``{"class": ..., "sha256": ...}`` of the model's imaging system; ``sha256`` is ``None`` when not computed.
    model : str
        The model's class name, e.g. ``"Phase2D"``.
    prepare_scope : str
        The model's prepare scope: what its ``prepare`` reads to prepare one voxel.

    Raises
    ------
    ParameterError
        If a count is not a positive int, ``fields`` is empty or not distinct names, ``pixel_size`` is not three
        positive numbers, or ``pixel_size_tolerance`` is not in [0, 1).
    """

    channels: int
    planes: tuple[int, ...] | None
    fields: tuple[str, ...]
    pixel_size: ZYX[float]
    pixel_size_tolerance: float = 0.05
    system: Mapping[str, object]
    model: str
    prepare_scope: str

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ParameterError(f"Contract channels must be a positive int, got {self.channels!r}", path="channels")
        if self.planes is not None:
            planes = tuple(self.planes)
            if not planes or min(planes) < 1:
                raise ParameterError(
                    f"Contract planes must be positive plane counts, or None, got {self.planes!r}", path="planes"
                )
            object.__setattr__(self, "planes", planes)
        fields = tuple(self.fields)
        if not fields or len(set(fields)) != len(fields):
            raise ParameterError(
                f"Contract fields must be distinct names, at least one, got {self.fields!r}", path="fields"
            )
        object.__setattr__(self, "fields", fields)
        size = tuple(self.pixel_size)
        if len(size) != 3 or not all(value > 0 for value in size):
            raise ParameterError(
                f"Contract pixel_size must be three positive numbers in um (z, y, x), got {self.pixel_size!r}",
                path="pixel_size",
            )
        object.__setattr__(self, "pixel_size", tuple(float(value) for value in size))
        if not 0 <= self.pixel_size_tolerance < 1:
            raise ParameterError(
                f"Contract pixel_size_tolerance must be in [0, 1), got {self.pixel_size_tolerance!r}",
                path="pixel_size_tolerance",
            )
        object.__setattr__(self, "system", dict(self.system))

    @classmethod
    def of(  # noqa: PLR0913 (the model, and one keyword per field the model does not determine)
        cls,
        imaging_model: ImagingModel,
        *,
        pixel_size: Sequence[float],
        model: str,
        prepare_scope: str,
        planes: Sequence[int] | None = None,
        fields: Sequence[str] | None = None,
        sha256: str | None = None,
        pixel_size_tolerance: float = 0.05,
    ) -> Self:
        """Return the contract of a student distilled for ``imaging_model``.

        The channels and the system's class name come from the model's system; ``fields`` defaults to all of its
        fields, and ``sha256`` is the hash of the system's record. The other arguments are as the constructor's.

        Raises
        ------
        ParameterError
            As the constructor; and if ``fields`` names a field the system does not have.
        """
        system = imaging_model.system
        chosen = tuple(system.fields if fields is None else fields)
        unknown = [name for name in chosen if name not in system.fields]
        if unknown:
            raise ParameterError(
                f"Contract fields {unknown} are not fields of {type(system).__name__}, which has {system.fields}",
                path="fields",
            )
        return cls(
            channels=system.channel_count,
            planes=None if planes is None else tuple(planes),
            fields=chosen,
            pixel_size=tuple(pixel_size),  # ty: ignore[invalid-argument-type] (validated by the constructor)
            pixel_size_tolerance=pixel_size_tolerance,
            system={"class": type(system).__name__, "sha256": sha256},
            model=model,
            prepare_scope=prepare_scope,
        )

    def model_problems(self, model: ImagingModel) -> list[str]:
        """Return what in ``model`` breaks the contract (channels, fields, system class), one sentence each."""
        system = model.system
        name = type(system).__name__
        problems = []
        if system.channel_count != self.channels:
            problems.append(
                f"the student takes {self.channels} data channel(s), but {name} measures {system.channel_count}"
            )
        missing = [field for field in self.fields if field not in system.fields]
        if missing:
            problems.append(
                f"the student returns the field(s) {', '.join(missing)}, which {name} does not have (its fields are "
                f"{', '.join(system.fields)})"
            )
        trained = self.system.get("class")
        if trained is not None and trained != name:
            problems.append(f"the student was distilled from a {trained} system, but the model's system is a {name}")
        return problems

    def sampling_problems(self, sampling: Sampling) -> list[str]:
        """Return what in ``sampling`` breaks the contract (plane count, voxel size), one sentence each."""
        problems = []
        depth = sampling.shape[0]
        if self.planes is not None and depth not in self.planes:
            accepted = " or ".join(str(count) for count in self.planes)
            problems.append(f"the data has {depth} plane(s), but the student takes {accepted}")
        for axis, size, trained in zip("zyx", sampling.pixel_size, self.pixel_size, strict=True):
            if abs(size / trained - 1) > self.pixel_size_tolerance:
                problems.append(
                    f"the pixel size along {axis} is {size:g} um, but the student was trained at {trained:g} um "
                    f"(tolerance {self.pixel_size_tolerance:.0%}): resample the data, or distil a student at this "
                    "sampling"
                )
        return problems

    def as_dict(self) -> dict[str, object]:
        """Return the contract as JSON-able values; :meth:`from_dict` reads them back."""
        return {
            "channels": self.channels,
            "planes": None if self.planes is None else list(self.planes),
            "fields": list(self.fields),
            "pixel_size": list(self.pixel_size),
            "pixel_size_tolerance": self.pixel_size_tolerance,
            "system": dict(self.system),
            "model": self.model,
            "prepare_scope": self.prepare_scope,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Return the contract :meth:`as_dict` wrote.

        Raises
        ------
        ParameterError
            If a key is missing or unknown, or a value is invalid.
        """
        try:
            return cls(**dict(values))  # ty: ignore[invalid-argument-type] (validated by the constructor)
        except TypeError as error:
            raise ParameterError(f"not a contract: {error}") from error
