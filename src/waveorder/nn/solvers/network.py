"""The distilled solver: a student network that returns the estimate a physics solver returned."""

from collections.abc import Callable, Mapping
from typing import Self, override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError, PlacementError, ShapeError
from waveorder.nn.context import SolveContext, current_context
from waveorder.nn.model import ImagingModel
from waveorder.nn.networks.base import PRECISIONS, Contract, Precision, RestorationNetwork
from waveorder.nn.networks.bundle import Bundle, Normalizer, predict
from waveorder.nn.networks.conditioning import Condition, Place
from waveorder.nn.parameters import placement, same_device
from waveorder.nn.sampling import ZYX
from waveorder.nn.solvers.base import Solver, SolveResult
from waveorder.nn.verbs import InverseProblem
from waveorder.registry import register

type AttributeFunction = Callable[[SolveContext], Mapping[str, float]]
"""Values per position for a solve context, by name; it must pickle to reach worker processes (no lambda)."""


@register("distilled", kind="solver")
class Distilled(Solver):
    """A solver that is a network: the student's estimate of what the model's physics solver returns.

    ``model.solver = Distilled(waveorder.io.bundle.load_bundle("bundle"))``. The estimate has every field of the
    system, ``(F, Z, Y, X)``; fields the student was not trained on are zeros. The student's weights stay float32;
    data in another dtype are cast on the way in and the estimate is returned in the data's dtype. The solver stays
    in evaluation mode, and ``problem.operator`` is not read unless ``score_objective`` is set. Plain-data form:
    ``{"distilled": {"path": ..., "precision": ...}}``.

    Parameters
    ----------
    bundle : Bundle
        The student, its contract, its conditions and its normalizers. An input normalizer of scope ``"field"``
        refuses a solve of part of a field (a tile).
    attributes : callable, optional
        ``attributes(context) -> {name: value}``: values per position for
        :class:`~waveorder.nn.networks.Attributes` conditions. It must pickle (a module-level function, not a lambda).
    precision : {"float32", "bf16", "fp16"}
        What the student computes in: float32, or autocast to bfloat16, or to float16 (which needs CUDA).
    score_objective : bool
        Return the model's objective at the estimate, which binds ``H``. By default the objective is NaN.

    Raises
    ------
    ParameterError
        If ``precision`` is unknown, the student's channels, fields or plane counts differ from the contract's, the
        conditions differ from those the student reads by name or size, or the target normalizer is not invertible
        voxel by voxel.
    """

    def __init__(
        self,
        bundle: Bundle,
        *,
        attributes: AttributeFunction | None = None,
        precision: Precision = "float32",
        score_objective: bool = False,
    ) -> None:
        super().__init__()
        if precision not in PRECISIONS:
            raise ParameterError(
                f"Distilled precision must be one of {', '.join(PRECISIONS)}, got {precision!r}", path="precision"
            )
        self.student = bundle.student
        self.input_normalizer = bundle.input_normalizer
        self.target_normalizer = bundle.target_normalizer
        self.contract = bundle.contract
        self.conditions = bundle.conditions
        self.attributes = attributes
        self.precision: Precision = precision
        self.bundle = bundle
        self.score_objective = score_objective
        _expect_fit(bundle.student, bundle.contract, target_normalizer=bundle.target_normalizer)
        _expect_conditions(bundle.student, bundle.conditions)
        self.train(mode=False)

    def arguments(self) -> dict[str, object]:
        """Return ``{"path": ..., "precision": ...}``, from which :mod:`waveorder.io.bundle` rebuilds the solver.

        Raises
        ------
        ParameterError
            If the bundle was built in memory, so no path describes it.
        """
        if self.bundle.path is None:
            raise ParameterError(
                "Distilled runs a bundle built in memory, so it has no plain-data form; write the bundle with "
                "waveorder.io.bundle.save_bundle and use Distilled(waveorder.io.bundle.load_bundle(path))",
                path="bundle",
            )
        return {"path": str(self.bundle.path), "precision": self.precision}

    @property
    @override
    def needs_operator(self) -> bool:
        """Whether a solve reads ``H``: only with ``score_objective``."""
        return self.score_objective

    @override
    def train(self, mode: bool = True) -> Self:
        """Keep the solver in evaluation mode, whatever ``mode`` is: a solve never trains the student."""
        del mode
        return super().train(mode=False)

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """Move as any module, then return the weights to float32: the student computes in no other dtype."""
        super()._apply(fn, recurse=recurse)
        return super()._apply(lambda tensor: tensor.float() if tensor.is_floating_point() else tensor, recurse=recurse)

    @override
    def extra_repr(self) -> str:
        return f"precision={self.precision!r}, score_objective={self.score_objective}"

    def halo(self) -> int:
        """Return the student's halo: the pixels of context along y and x that a tile needs on each side."""
        return self.student.halo()

    @override
    def tile_halo(self) -> ZYX[int]:
        """Return the student's halo along z, y and x, which the tile planner uses as it is."""
        halo = self.student.halo()
        return (self.student.halo_planes(), halo, halo)

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise :class:`IncompatibleModelError` unless the system matches the contract's class, channels and fields.

        The sha256 of the system record is compared by ``waveorder.models``, not here.
        """
        problems = self.contract.model_problems(model)
        if problems:
            raise IncompatibleModelError(f"Distilled cannot run on this model: {'; '.join(problems)}")

    @override
    def run(self, problem: InverseProblem) -> SolveResult:
        """Return the student's estimate ``(F, Z, Y, X)`` with status ``"distilled"``.

        Raises
        ------
        IncompatibleModelError
            If the data's sampling breaks the contract (another plane count, a pixel size outside the tolerance), or
            the data is part of a field while the input normalizer needs the whole field.
        PlacementError
            If the student is on another device than the data.
        ShapeError
            If the solve context does not describe the data, or the student returns another number of planes than
            the system's object has.
        BackendUnavailableError
            If the device cannot compute in the solver's precision.
        """
        data, model, sampling = problem.data, problem.model, problem.sampling
        system = model.system
        problems = self.contract.sampling_problems(sampling)
        if problems:
            raise IncompatibleModelError(f"Distilled cannot run on this data: {'; '.join(problems)}")
        _, device = placement(self.student)
        if not same_device(device, data.device):
            raise PlacementError(
                f"Distilled: the student is on {device} but the data is on {data.device}; move the solver with "
                ".to(device) (on a waveorder.models model: model.to(device), which moves its solver)"
            )
        place = self._place(problem)
        condition = {item.name: item.values(place, model)[None].to(device) for item in self.conditions}
        estimate = predict(
            self.student,
            data[None],
            condition,
            input_normalizer=self.input_normalizer,
            target_normalizer=self.target_normalizer,
            precision=self.precision,
        )[0].to(data.dtype)
        shape = system.object_shape(sampling)
        if tuple(estimate.shape[1:]) != shape[1:]:
            raise ShapeError(
                f"Distilled: the student returns {estimate.shape[1]} plane(s) of {tuple(estimate.shape[2:])}, but the "
                f"object of {type(system).__name__} on this sampling is {shape}: a plane student fits a thin-object "
                "system; wrap it in SlidingPlanes for a volume"
            )
        if self.contract.fields != tuple(system.fields):
            placed = estimate.new_zeros(shape)
            placed[[system.fields.index(name) for name in self.contract.fields]] = estimate
            estimate = placed
        if self.score_objective:
            objective = problem.objective(estimate)
        else:
            objective = torch.full((), float("nan"), dtype=data.dtype, device=data.device)
        return SolveResult(estimate, objective, 0, "distilled")

    def _place(self, problem: InverseProblem) -> Place:
        """Return where the problem's data sits, from the solve context, with the attributes of its position."""
        context = current_context()
        attributes = {} if self.attributes is None else self.attributes(context)
        place = Place.of(problem.sampling, context=context, attributes=attributes)
        extents = zip(place.region, place.field_shape, strict=True)
        whole = all(extent.stop - extent.start == size for extent, size in extents)
        if self.input_normalizer.scope == "field" and not whole:
            raise IncompatibleModelError(
                f"Distilled: {type(self.input_normalizer).__name__} normalizes by statistics of the whole field of "
                f"view {place.field_shape}, but this solve is given the region {place.region} of it; reconstruct "
                "whole fields (no tiles), or distil a student with a voxel-scope input normalizer"
            )
        return place


def _expect_fit(student: RestorationNetwork, contract: Contract, *, target_normalizer: Normalizer) -> None:
    """Raise :class:`ParameterError` unless the student and the target normalizer fit the contract."""
    if student.channels != contract.channels:
        raise ParameterError(
            f"Distilled: the student takes {student.channels} channel(s), the contract says {contract.channels}",
            path="contract",
        )
    if student.fields != len(contract.fields):
        raise ParameterError(
            f"Distilled: the student returns {student.fields} field(s), the contract names {len(contract.fields)} "
            f"({', '.join(contract.fields)})",
            path="contract",
        )
    for planes in contract.planes or ():
        try:
            student.planes_out(planes)
        except ShapeError as error:
            raise ParameterError(
                f"Distilled: the contract accepts {planes} planes, but {error}", path="contract"
            ) from error
    if not target_normalizer.invertible or target_normalizer.scope != "voxel":
        raise ParameterError(
            f"Distilled: the target normalizer {type(target_normalizer).__name__} cannot be inverted voxel by voxel; "
            "use Identity or Affine",
            path="target_normalizer",
        )


def _expect_conditions(student: RestorationNetwork, conditions: tuple[Condition, ...]) -> None:
    """Raise :class:`ParameterError` unless the conditions are those the student reads, by name and size."""
    given: dict[str, int] = {}
    for condition in conditions:
        if condition.name in given:
            raise ParameterError(f"Distilled: two conditions are named {condition.name!r}", path="conditions")
        given[condition.name] = condition.size
    for name, size in student.condition_sizes.items():
        if name not in given:
            raise ParameterError(
                f"Distilled: the student reads the condition {name!r} ({size} values), but no such condition was "
                f"given (given: {', '.join(given) or 'none'})",
                path="conditions",
            )
        if given[name] != size:
            raise ParameterError(
                f"Distilled: the condition {name!r} has {given[name]} values, the student reads {size}",
                path="conditions",
            )
    unread = [name for name in given if name not in student.condition_sizes]
    if unread:
        raise ParameterError(
            f"Distilled: the student does not read the condition(s) {', '.join(unread)}; leave them out",
            path="conditions",
        )
