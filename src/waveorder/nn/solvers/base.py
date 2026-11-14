"""The solver contract, the result every solver returns, and the loop iterative solvers share."""

import math
import warnings
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, fields
from typing import final, override

import torch
from torch import Tensor

from waveorder.errors import ConvergenceWarning, IncompatibleModelError, ParameterError
from waveorder.nn.parameters import can_read_back, read_flag, repr_parts
from waveorder.nn.sampling import ZYX, Object, Status
from waveorder.nn.verbs import InverseProblem, Verb
from waveorder.observe import emit, recording
from waveorder.registry import VALUE, register


@dataclass(frozen=True, slots=True, repr=False)
class SolveResult:
    """An object estimate with the evidence a caller needs to trust it.

    Attributes
    ----------
    estimate : Tensor
        The object estimate, shape ``(F, Z, Y, X)``.
    objective : Tensor
        The objective the solver minimized at :attr:`estimate`, without terms constant in ``f``; NaN for a
        ``"distilled"`` solve unless the solver was asked to score it.
    iterations : int
        Iterations run: 0 for a closed-form solve, else the count at which the solve ended.
    status : Status
        How the solve ended: ``"direct"`` (closed form), ``"budget"`` (no tolerance), ``"converged"``,
        ``"max_iterations"`` or ``"nonfinite"`` (the last two warn). The first two read nothing back: check
        :attr:`finite`.
    history : Tensor or None
        The relative change ``||f_k - f_k-1|| / ||f_k||`` at each iteration, shape ``(iterations,)``, detached;
        recorded inside ``waveorder.observe(..., iterations=True)``, else ``None``.
    """

    estimate: Object
    objective: Tensor
    iterations: int
    status: Status
    history: Tensor | None = None

    @override
    def __repr__(self) -> str:
        """Return one line that describes each tensor by shape, dtype and device."""
        history = None if self.history is None else _described(self.history)
        return (
            f"SolveResult(estimate={_described(self.estimate)}, objective={_described(self.objective)}, "
            f"iterations={self.iterations}, status={self.status!r}, history={history})"
        )

    @property
    def finite(self) -> Tensor:
        """Whether the estimate and the objective are finite, as a 0-d bool tensor on their device.

        Reads nothing back, so many solves can be checked with one host read. A ``"distilled"`` solve's NaN objective
        is not held against it.
        """
        objective = torch.isfinite(self.objective)
        if self.status == "distilled":
            objective = objective | torch.isnan(self.objective)
        return torch.isfinite(self.estimate).all() & objective.all()


def _described(value: object) -> str:
    """``Tensor(shape=(1, 16, 64, 64), dtype=torch.float32, device=cpu)``; anything else as its own repr."""
    if not isinstance(value, Tensor):
        return repr(value)
    return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"


class Solver(Verb[SolveResult]):
    """Estimates the object: ``solver(data, model, sampling, *, operator=None) -> SolveResult``.

    A module, so a learned network is a solver too: subclass and implement :meth:`run` (and :meth:`check` if it
    cannot run on every model).
    """

    def tile_halo(self) -> ZYX[int] | None:
        """Return the voxels of context along z, y, x a tile needs on each side, when the solver knows them.

        ``None`` (the default) lets :func:`waveorder.tiling.halo_for` measure it from the impulse response, which
        holds for a solver linear in the data; a nonlinear solver (a learned network) returns its own.
        """
        return None

    @override
    def report(self, problem: InverseProblem, result: object) -> None:
        if not isinstance(result, SolveResult):  # a run that broke its contract; the call itself reports that
            return
        payload = {
            "solver": type(self).__name__,
            "system": type(problem.model.system).__name__,
            "status": result.status,
            "iterations": result.iterations,
        }
        history = result.history
        emit("solve.end", payload, deferred=None if history is None else {"history": history})


@register("loop", kind=VALUE)
@dataclass(frozen=True, slots=True, repr=False)
class Loop:
    """How an :class:`IterativeSolver` iterates and stops, e.g. ``FISTA(loop=Loop(200, tolerance=1e-4))``.

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
        If a count is below 1 or a tolerance is not positive and finite.
    """

    iterations: int
    _: KW_ONLY
    tolerance: float | None = None
    check_every: int = 10

    def __post_init__(self) -> None:
        for name in ("iterations", "check_every"):
            value = getattr(self, name)
            if value < 1:
                raise ParameterError(f"Loop.{name} must be an integer of at least 1, got {value!r}", path=name)
        tolerance = self.tolerance
        if tolerance is not None and not (math.isfinite(tolerance) and tolerance > 0):
            raise ParameterError(f"Loop.tolerance must be positive and finite, got {tolerance!r}", path="tolerance")

    @override
    def __repr__(self) -> str:
        """Return ``Loop(50)`` followed by each setting that differs from its default."""
        changed = [
            f"{field.name}={getattr(self, field.name)!r}"
            for field in fields(self)
            if field.name != "iterations" and getattr(self, field.name) != field.default
        ]
        return f"Loop({', '.join([str(self.iterations), *changed])})"


class IterativeSolver[S](Solver):
    """A solver defined by its iteration: :meth:`start`, :meth:`step` and :meth:`estimate` over a state ``S``.

    The shared loop owns stopping, non-finite detection and the :class:`SolveResult`. Gradients reach the model's
    parameters through every iteration.

    Parameters
    ----------
    loop : Loop
        Iteration count and stopping.

    Raises
    ------
    IncompatibleModelError
        From a call, if ``loop`` has a tolerance and the data cannot be read back to the host.
    """

    def __init__(self, *, loop: Loop) -> None:
        super().__init__()
        self.loop = loop

    @override
    def extra_repr(self) -> str:
        """Return the loop, then the solver's quantities."""
        return repr_parts(f"loop={self.loop!r}", super().extra_repr())

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
        loop = self.loop
        if loop.tolerance is not None and not can_read_back(problem.data):
            raise IncompatibleModelError(
                f"{type(self).__name__} Loop(tolerance=...) reads values back to decide when to stop, which vmap and "
                "grad cannot; use tolerance=None there"
            )
        state, iteration, status, history = self._iterate(problem)
        return SolveResult(self.estimate(state), self.objective(state, problem), iteration, status, history=history)

    def _iterate(self, problem: InverseProblem) -> tuple[S, int, Status, Tensor | None]:
        loop = self.loop
        history = _History(on=recording())
        state = self.start(problem)
        for iteration in range(1, loop.iterations + 1):
            previous = self.estimate(state) if history.on or loop.tolerance is not None else None
            state = self.step(state, problem)
            history.add(previous, self.estimate(state))
            if self.stopped(state):
                return state, iteration, "converged", history.tensor()
            if loop.tolerance is None or previous is None:
                continue
            if iteration % loop.check_every == 0 or iteration == loop.iterations:
                status = _checked(previous, self.estimate(state), tolerance=loop.tolerance)
                if status == "nonfinite":
                    warnings.warn(
                        f"{type(self).__name__} produced a non-finite estimate at iteration {iteration}",
                        ConvergenceWarning,
                        stacklevel=4,
                    )
                if status is not None:
                    return state, iteration, status, history.tensor()
        if loop.tolerance is None:
            return state, loop.iterations, "budget", history.tensor()
        warnings.warn(
            f"{type(self).__name__} reached {loop.iterations} iterations without meeting tolerance {loop.tolerance}",
            ConvergenceWarning,
            stacklevel=4,
        )
        return state, loop.iterations, "max_iterations", history.tensor()


class _History:
    """The relative change of the estimate at each iteration, collected when ``on``; see :attr:`SolveResult.history`."""

    __slots__ = ("changes", "on")

    def __init__(self, *, on: bool) -> None:
        self.on = on
        self.changes: list[Tensor] = []

    def add(self, previous: Object | None, current: Object) -> None:
        if self.on and previous is not None:
            self.changes.append(_relative_change(previous, current))

    def tensor(self) -> Tensor | None:
        return torch.stack(self.changes) if self.on else None


def _relative_change(previous: Object, current: Object) -> Tensor:
    """Return ``||current - previous|| / ||current||``, detached: a 0-d tensor on the device, nothing read back."""
    current, previous = current.detach(), previous.detach()
    return torch.linalg.vector_norm(current - previous) / torch.linalg.vector_norm(current)


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
