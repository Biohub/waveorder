"""The solver contract, the result every solver returns, and the loop iterative solvers share."""

import math
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor

from waveorder.errors import ConvergenceWarning, IncompatibleModelError, ParameterError
from waveorder.nn.parameters import can_read_back, read_flag, repr_parts
from waveorder.nn.sampling import Object, Status
from waveorder.nn.verbs import InverseProblem, Verb


@dataclass(frozen=True, slots=True, repr=False)
class SolveResult:
    """An object estimate with the evidence a caller needs to trust it.

    Attributes
    ----------
    estimate : Tensor
        The object estimate, shape ``(F, Z, Y, X)``.
    objective : Tensor
        The objective the solver minimized at :attr:`estimate`, without terms constant in ``f``.
    iterations : int
        Iterations run: 0 for a closed-form solve, else the count at which the solve ended.
    status : Status
        How the solve ended: ``"direct"`` (closed form), ``"budget"`` (no tolerance), ``"converged"``,
        ``"max_iterations"`` or ``"nonfinite"`` (the last two warn). The first two read nothing back: check
        :attr:`finite`.
    """

    estimate: Object
    objective: Tensor
    iterations: int
    status: Status

    @override
    def __repr__(self) -> str:
        """Return one line that describes each tensor by shape, dtype and device."""
        return (
            f"SolveResult(estimate={_described(self.estimate)}, objective={_described(self.objective)}, "
            f"iterations={self.iterations}, status={self.status!r})"
        )

    @property
    def finite(self) -> Tensor:
        """Whether the estimate and the objective are finite, as a 0-d bool tensor on their device.

        Reads nothing back, so many solves can be checked with one host read.
        """
        return torch.isfinite(self.estimate).all() & torch.isfinite(self.objective).all()


def _described(value: object) -> str:
    """``Tensor(shape=(1, 16, 64, 64), dtype=torch.float32, device=cpu)``; anything else as its own repr."""
    if not isinstance(value, Tensor):
        return repr(value)
    return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"


class Solver(Verb[SolveResult]):
    """Estimates the object: ``solver(data, model, sampling, *, operator=None) -> SolveResult``.

    A module, so a solver defined outside this package (with trainable weights, say) subclasses it and implements
    :meth:`run` (and :meth:`check` if it cannot run on every model).
    """


class IterativeSolver[S](Solver):
    """A solver defined by its iteration: :meth:`start`, :meth:`step` and :meth:`estimate` over a state ``S``.

    The shared loop owns stopping, non-finite detection and the :class:`SolveResult`. Gradients reach the model's
    parameters through every iteration. A call's ``callback(iteration, estimate)`` runs after each iteration.

    Parameters
    ----------
    iterations : int
        Maximum number of iterations.
    tolerance : float, optional
        Stop once ``||f_k+1 - f_k|| <= tolerance * ||f_k||``, checked every ``check_every`` iterations (one device
        sync per check). ``None`` runs exactly ``iterations`` steps and reads nothing back.
    check_every : int
        Iterations between tolerance and finiteness checks.

    Raises
    ------
    ParameterError
        If a count is below 1 or the tolerance is not positive and finite.
    IncompatibleModelError
        From a call, if there is a tolerance and the data cannot be read back to the host.
    """

    def __init__(self, *, iterations: int, tolerance: float | None = None, check_every: int = 10) -> None:
        super().__init__()
        name = type(self).__name__
        for field, count in (("iterations", iterations), ("check_every", check_every)):
            if count < 1:
                raise ParameterError(f"{name}.{field} must be an integer of at least 1, got {count!r}", path=field)
        if tolerance is not None and not (math.isfinite(tolerance) and tolerance > 0):
            raise ParameterError(f"{name}.tolerance must be positive and finite, got {tolerance!r}", path="tolerance")
        self.iterations = iterations
        self.tolerance = tolerance
        self.check_every = check_every

    @override
    def extra_repr(self) -> str:
        """Return the iteration count, each stopping setting that differs from its default, then the quantities."""
        stopping = [f"iterations={self.iterations}"]
        if self.tolerance is not None:
            stopping.append(f"tolerance={self.tolerance!r}")
        if self.check_every != 10:
            stopping.append(f"check_every={self.check_every}")
        return repr_parts(", ".join(stopping), super().extra_repr())

    @abstractmethod
    def start(self, problem: InverseProblem) -> S:
        """Return the initial state."""

    @abstractmethod
    def step(self, state: S, problem: InverseProblem) -> S:
        """Return the state after one iteration."""

    @abstractmethod
    def estimate(self, state: S) -> Object:
        """Return the object estimate held by ``state``."""

    def objective(self, state: S, problem: InverseProblem) -> Tensor:
        """Return the objective this solver minimizes at ``state``; the model's MAP objective by default."""
        return problem.objective(self.estimate(state))

    def stopped(self, state: S) -> bool:
        """Return whether ``state`` is a fixed point, so the solve ends there as ``"converged"``; never by default.

        Checked after every step; an override reads the state back, so only a solver that runs eagerly has one.
        """
        del state
        return False

    @final
    @override
    def run(self, problem: InverseProblem) -> SolveResult:
        if self.tolerance is not None and not can_read_back(problem.data):
            raise IncompatibleModelError(
                f"{type(self).__name__}(tolerance=...) reads values back to decide when to stop, which vmap and "
                "grad cannot; use tolerance=None there"
            )
        state, iteration, status = self._iterate(problem)
        return SolveResult(self.estimate(state), self.objective(state, problem), iteration, status)

    def _iterate(self, problem: InverseProblem) -> tuple[S, int, Status]:
        tolerance, callback = self.tolerance, problem.callback
        state = self.start(problem)
        for iteration in range(1, self.iterations + 1):
            previous = None if tolerance is None else self.estimate(state)
            state = self.step(state, problem)
            if callback is not None:
                callback(iteration, self.estimate(state))
            if self.stopped(state):
                return state, iteration, "converged"
            if tolerance is None or previous is None:
                continue
            if iteration % self.check_every == 0 or iteration == self.iterations:
                status = _checked(previous, self.estimate(state), tolerance=tolerance)
                if status == "nonfinite":
                    warnings.warn(
                        f"{type(self).__name__} produced a non-finite estimate at iteration {iteration}",
                        ConvergenceWarning,
                        stacklevel=4,
                    )
                if status is not None:
                    return state, iteration, status
        if tolerance is None:
            return state, self.iterations, "budget"
        warnings.warn(
            f"{type(self).__name__} reached {self.iterations} iterations without meeting tolerance {tolerance}",
            ConvergenceWarning,
            stacklevel=4,
        )
        return state, self.iterations, "max_iterations"


def _checked(previous: Tensor, current: Tensor, *, tolerance: float) -> Status | None:
    """``"nonfinite"`` or ``"converged"`` if the check ends the solve, else ``None``; at most two host reads."""
    # A diverging iterate can stay finite while its norm overflows, and inf <= tolerance * inf would read as converged.
    change = torch.linalg.vector_norm(current - previous)
    size = torch.linalg.vector_norm(previous)
    if not read_flag(torch.isfinite(current).all() & torch.isfinite(change) & torch.isfinite(size)):
        return "nonfinite"
    if read_flag(change <= tolerance * size):
        return "converged"
    return None
