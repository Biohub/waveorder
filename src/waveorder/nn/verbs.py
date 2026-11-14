"""The base of every solver and score: one inverse problem, checked on every call, with ``H`` bound on first use."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self, final, override

import torch
from torch import Tensor

from waveorder.nn.model import ImagingModel, bound_operator, expect_on_system
from waveorder.nn.operators import LinearOperator
from waveorder.nn.parameters import CalibratableModule
from waveorder.nn.sampling import Data, Object, Sampling
from waveorder.observe import capturing, observing, operation


@dataclass(frozen=True, slots=True)
class InverseProblem:
    """One inverse problem: data, the model, the data's sampling, and ``H`` bound to that sampling at most once.

    :attr:`operator` binds the model's system on first use, so whatever ``H`` needs for the sampling is computed at
    most once per verb call, and a verb that never reads it never binds. Build one with :meth:`of`.
    """

    data: Data
    model: ImagingModel
    sampling: Sampling
    _where: str = field(default="InverseProblem.of", repr=False, compare=False)
    _grad: bool = field(default=True, repr=False, compare=False)
    _bound: dict[str, LinearOperator] = field(default_factory=dict, repr=False, compare=False)
    _normal_actions: dict[bool, Callable[[Tensor], Tensor]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def of(
        cls,
        data: Data,
        model: ImagingModel,
        sampling: Sampling,
        *,
        operator: LinearOperator | None = None,
        where: str = "InverseProblem.of",
    ) -> Self:
        """Validate ``data`` against the model's system on ``sampling``, and adopt ``operator`` when one is given.

        Without ``operator`` nothing is bound here; :attr:`operator` binds later in the grad mode of this call.
        ``where`` names the caller in every error.

        Raises
        ------
        PlacementError
            If ``data`` has another real dtype or device than the system's parameters, or ``operator`` does.
        ShapeError
            If ``data`` does not have the system's data shape, or ``operator`` was bound to another sampling or maps
            other shapes (:func:`~waveorder.nn.model.bound_operator`).
        IncompatibleModelError
            If the operator is not of the system's declared
            :attr:`~waveorder.nn.systems.base.ImagingSystem.operator_type`.
        """
        system = model.system
        # The shape comes from the sampling alone; bound_operator holds the operator to the same shape when it binds.
        expect_on_system(data, system, where=where, name="data", shape=system.data_shape(sampling))
        bound = (
            {} if operator is None else {"operator": bound_operator(system, sampling, operator=operator, where=where)}
        )
        return cls(data, model, sampling, where, torch.is_grad_enabled(), bound)

    @property
    def operator(self) -> LinearOperator:
        """``H`` on :attr:`sampling`: the injected operator, or the model's system bound on the first read.

        The first read raises what :meth:`of` documents for an operator.
        """
        bound = self._bound.get("operator")
        if bound is None:
            with torch.set_grad_enabled(self._grad):
                bound = bound_operator(self.model.system, self.sampling, operator=None, where=self._where)
            self._bound["operator"] = bound
        return bound

    def objective(self, estimate: Object) -> Tensor:
        """Return the model's MAP objective at ``estimate``, through the bound operator."""
        return self.model.objective(self.data, estimate, self.sampling, operator=self.operator)

    def normal_action(self) -> Callable[[Tensor], Tensor]:
        """Return ``f -> H* H f`` from :meth:`LinearOperator.normal_action`, built once per grad mode per problem.

        Kept per grad mode: an action built without a graph must not be reused where gradients must reach the system.
        """
        grad = torch.is_grad_enabled()
        action = self._normal_actions.get(grad)
        if action is None:
            action = self.operator.normal_action()
            self._normal_actions[grad] = action
        return action


class Verb[R](CalibratableModule, ABC):
    """A module that runs on ``(data, model, sampling)``: it checks the model, validates the data, then works.

    Every call runs :meth:`ImagingModel.check <waveorder.nn.model.ImagingModel.check>`, so an incompatible model or
    sampling fails before any computation, then :meth:`run` on one :class:`InverseProblem`. Subclasses implement
    :meth:`run` and, when they cannot run on every model, :meth:`check`. Free tuning quantities (step sizes,
    weights) are ``nn.Parameter`` objects. Each eager call is one :func:`~waveorder.observe.operation`.
    """

    @property
    def needs_operator(self) -> bool:
        """Whether :meth:`run` reads ``problem.operator``; ``True`` unless a subclass says otherwise.

        Callers that bind ``H`` ahead to pass it as ``operator=`` skip that for a verb that returns ``False``.
        """
        return True

    def check(self, model: ImagingModel) -> None:
        """Raise :class:`~waveorder.errors.IncompatibleModelError` if this verb cannot run on ``model``.

        Every model is accepted by default.
        """
        del model

    @abstractmethod
    def run(self, problem: InverseProblem) -> R:
        """Do the work on a checked problem; ``problem.operator`` is ``H`` on its sampling, bound on first use."""

    def report(self, problem: InverseProblem, result: object) -> None:
        """Emit the events of a finished outermost call (:func:`waveorder.observe.emit`); none by default.

        Called only when an observer listens; it must not read device values. ``result`` is what :meth:`run` returned.
        """
        del problem, result

    @final
    @override
    def forward(
        self, data: Data, model: ImagingModel, sampling: Sampling, *, operator: LinearOperator | None = None
    ) -> R:
        """Check, validate and run; pass ``operator`` to reuse ``H`` already bound to ``sampling``."""
        where = type(self).__name__
        model.check(self, sampling)
        if capturing():  # inside a transform the verb runs without an operation
            return self.run(InverseProblem.of(data, model, sampling, operator=operator, where=where))
        with operation():
            problem = InverseProblem.of(data, model, sampling, operator=operator, where=where)
            result = self.run(problem)
            if observing():
                self.report(problem, result)
        return result
