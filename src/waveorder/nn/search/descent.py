"""Gradient descent stages: refine the free quantities with a torch optimizer, clamped to their bounds.

:class:`GradientSearch` and :class:`LBFGSSearch` share one loop: geometric spans
(:func:`~waveorder.nn.search.search_spans`) are optimized in log space, every evaluated point is clamped to the
bounds, and the best point visited is tracked on the device. With a ``tolerance`` the loop stops as wo-4.0's
``optimize_reconstruction`` did: after ``patience`` consecutive steps whose objective did not improve on the lowest
so far by ``tolerance``.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal, cast

import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.nn.search.base import (
    CalibrationObjective,
    Estimate,
    _expect_count,
    _geometric,
    expect_known_names,
    record_step,
)
from waveorder.nn.search.grid import _span
from waveorder.registry import register


def _expect_rate(rate: float, *, where: str, name: str) -> None:
    if not (math.isfinite(rate) and rate > 0):
        raise ParameterError(f"{where} {name} must be positive and finite, got {rate}", path=name)


def _expect_stop(tolerance: float | None, patience: int, *, where: str) -> None:
    if tolerance is not None and not math.isfinite(tolerance):
        raise ParameterError(f"{where} tolerance must be finite, got {tolerance}", path="tolerance")
    _expect_count(patience, where=where, name="patience", minimum=1)


@dataclass(frozen=True, slots=True)
class _Stop:
    """wo-4.0's early stop: ``patience`` steps in a row without an improvement of ``tolerance``.

    The stop is read back to the host every ``check_every`` steps.
    """

    tolerance: float
    patience: int
    check_every: int


@dataclass(frozen=True, slots=True)
class _Space:
    """The span of each quantity a descent stage moves in, and which spans are geometric (optimized in log space)."""

    spans: dict[str, tuple[float, float]]
    logged: frozenset[str]


type _OptimizerFactory = Callable[[Mapping[str, Tensor], _Space], torch.optim.Optimizer]
"""Build an optimizer over the descent variables (by dotted name) in a given :class:`_Space`."""


type Optimizer = Literal["adam", "nadam", "adamw", "sgd"]
"""The torch optimizers :class:`GradientSearch` can run: ``torch.optim.Adam``, ``NAdam``, ``AdamW`` and ``SGD``."""

_OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "nadam": torch.optim.NAdam,
    # 3.x recover_zernikes ran AdamW with this decay and SGD with Nesterov momentum 0.9.
    "adamw": lambda groups: torch.optim.AdamW(groups, weight_decay=1e-3),
    "sgd": lambda groups: torch.optim.SGD(groups, momentum=0.9, nesterov=True),
}


@register("gradient", kind="search")
@dataclass(frozen=True, slots=True, kw_only=True)
class GradientSearch:
    """Refine with Adam (or NAdam, AdamW, SGD) on the objective, clamping to the bounds after every step.

    A quantity with a geometric span is optimized in log space, so a step is a relative change. Adam and NAdam use
    torch's defaults; AdamW a weight decay of 1e-3 and SGD Nesterov momentum 0.9, as 3.x ``recover_zernikes``.

    Parameters
    ----------
    iterations : int
        Optimizer steps, at least 1; 200 by default (the config's ``max_iter``).
    learning_rate : float
        Step size as a fraction of each quantity's span (:func:`~waveorder.nn.search.search_spans`: Adam's first step
        moves a quantity by about ``learning_rate * (high - low)``), or in log units for a geometric span; positive
        and finite.
    learning_rates : Mapping[str, float]
        Step sizes by dotted name, overriding ``learning_rate``: absolute, in the quantity's own units (log units for
        a geometric span) and not scaled by its span, as wo-4.0's ``Adam(lr=...)``.
    optimizer : {"adam", "nadam", "adamw", "sgd"}
        The torch optimizer; ``"adam"`` by default.
    loss_scale : float
        Factor on the objective before each backward pass; positive and finite. It matters for SGD, not Adam.
    last : bool
        Return the last iterate instead of the best point visited, as the 3.x gradient stages did.
    tolerance : float, optional
        Stop early, as wo-4.0's ``convergence_tol``: once ``patience`` consecutive steps start from an objective
        (summed over a leading tile axis) that is not below the lowest so far minus ``tolerance``; finite. ``None``
        (the default) runs every step without a host read.
    patience : int
        Those consecutive steps, at least 1; wo-4.0's ``convergence_patience``, 5 by default.
    check_every : int
        With a ``tolerance``: read the stop back to the host every this many steps, at least 1. The stop is decided
        on the device at the step it happens; the up to ``check_every - 1`` steps run after it change nothing and
        are not reported, so they cost time only.

    Raises
    ------
    ParameterError
        For ``iterations``, ``patience`` or ``check_every`` below 1, a rate or ``loss_scale`` that is not positive
        and finite, a ``tolerance`` that is not finite, or an unknown ``optimizer``.
    """

    iterations: int = 200
    learning_rate: float = 0.02
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    optimizer: Optimizer = "adam"
    loss_scale: float = 1.0
    last: bool = False
    tolerance: float | None = None
    patience: int = 5
    check_every: int = 10
    quantity_keyed: ClassVar[tuple[str, ...]] = ("learning_rates",)

    def __post_init__(self) -> None:
        where = "GradientSearch"
        _expect_count(self.iterations, where=where, name="iterations", minimum=1)
        _expect_rate(self.learning_rate, where=where, name="learning_rate")
        _expect_rate(self.loss_scale, where=where, name="loss_scale")
        for key, rate in self.learning_rates.items():
            _expect_rate(rate, where=where, name=f"learning_rates.{key}")
        if self.optimizer not in _OPTIMIZERS:
            raise ParameterError(
                f"{where} optimizer must be one of {sorted(_OPTIMIZERS)}, got {self.optimizer!r}", path="optimizer"
            )
        _expect_stop(self.tolerance, self.patience, where=where)
        _expect_count(self.check_every, where=where, name="check_every", minimum=1)

    @property
    def reads_back(self) -> bool:
        """Whether :meth:`run` reads back to the host: with a ``tolerance``, every ``check_every`` steps."""
        return self.tolerance is not None

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best point the optimizer visits from ``start``, or its last one with :attr:`last`."""
        expect_known_names(self.learning_rates, objective.specs, path="GradientSearch.learning_rates")
        stop = None if self.tolerance is None else _Stop(self.tolerance, self.patience, self.check_every)
        return _descend(
            objective,
            start=start,
            optimizer=self._build,
            steps=self.iterations,
            span_units=False,
            loss_scale=self.loss_scale,
            last=self.last,
            stop=stop,
        )

    def _build(self, variables: Mapping[str, Tensor], space: _Space) -> torch.optim.Optimizer:
        groups = []
        for name, (low, high) in space.spans.items():
            rate = self.learning_rates.get(name)
            if rate is None:
                rate = self.learning_rate * (1.0 if name in space.logged else high - low)
            groups.append({"params": [variables[name]], "lr": rate})
        return _OPTIMIZERS[self.optimizer](groups)


@register("lbfgs", kind="search")
@dataclass(frozen=True, slots=True, kw_only=True)
class LBFGSSearch:
    """Refine with L-BFGS and a strong-Wolfe line search, evaluating the objective only at points clamped to the bounds.

    Quantities are scaled to a tenth of their (log) span, the length of the first step. Each iteration may evaluate
    the objective a few times.

    Parameters
    ----------
    iterations : int
        L-BFGS steps, at least 1.
    history : int
        Number of past updates kept for the inverse-Hessian estimate, at least 1.
    tolerance : float, optional
        Stop early, as wo-4.0's ``convergence_tol``: once ``patience`` consecutive iterations start from an
        objective not below the lowest so far minus ``tolerance``; finite. The iterations then run as one optimizer
        step each, so torch's own stopping tests within a step (a loss change below ``1e-9``, five evaluations per
        four iterations) do not apply. ``None`` (the default): one step of ``iterations`` iterations.
    patience : int
        Those consecutive iterations, at least 1; wo-4.0's ``convergence_patience``, 5 by default.
    """

    iterations: int = 20
    history: int = 10
    tolerance: float | None = None
    patience: int = 5
    reads_back: ClassVar[bool] = True
    """The strong-Wolfe line search reads objective values back to the host to choose its step."""
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        _expect_count(self.iterations, where="LBFGSSearch", name="iterations", minimum=1)
        _expect_count(self.history, where="LBFGSSearch", name="history", minimum=1)
        _expect_stop(self.tolerance, self.patience, where="LBFGSSearch")

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best point L-BFGS visits from ``start``."""
        if self.tolerance is None:
            # One optimizer step runs every iteration, so the line search keeps its state across them.
            return _descend(objective, start=start, optimizer=self._lbfgs, steps=1, span_units=True)
        stop = _Stop(self.tolerance, self.patience, check_every=1)  # the line search reads back every step anyway
        return _descend(
            objective, start=start, optimizer=self._lbfgs, steps=self.iterations, span_units=True, stop=stop
        )

    def _lbfgs(self, variables: Mapping[str, Tensor], space: _Space) -> torch.optim.Optimizer:
        del space  # one step size for all variables: span_units puts them on a common scale instead
        iterations = self.iterations if self.tolerance is None else 1
        return torch.optim.LBFGS(
            list(variables.values()), max_iter=iterations, history_size=self.history, line_search_fn="strong_wolfe"
        )


class _Stall:
    """The state of a :class:`_Stop` on the device, and the evaluations it holds back until it knows the stop."""

    def __init__(self, stop: _Stop, like: Tensor) -> None:
        self.stop = stop
        # wo-4.0 compared Python floats; MPS has no float64, so it compares in float32 there.
        self.dtype = torch.float32 if like.device.type == "mps" else torch.float64
        self.lowest = torch.full((), math.inf, dtype=self.dtype, device=like.device)
        self.stalled = torch.zeros((), dtype=torch.int64, device=like.device)
        self.counted = torch.zeros((), dtype=torch.int64, device=like.device)
        self.stopped = torch.zeros((), dtype=torch.bool, device=like.device)
        self.step = 0
        self.held: dict[str, Tensor] = {}
        self.pending: list[tuple[int, Tensor, dict[str, Tensor]]] = []

    def begin(self, step: int, variables: Mapping[str, Tensor]) -> None:
        """Start step ``step``: keep the variables, to undo it if the stop came before."""
        self.step = step
        self.held = {name: v.detach().clone() for name, v in variables.items()}

    def keep(self, objective: Tensor, point: Mapping[str, Tensor]) -> None:
        """Hold back one evaluation of the current step (``-1``: after the last step) until :meth:`report`."""
        self.pending.append((self.step, objective, {name: value.detach() for name, value in point.items()}))

    def end(self, started: Tensor, variables: Mapping[str, Tensor]) -> bool:
        """Count the step that started from objective ``started``, or undo it if stopped; return whether to break.

        The rule runs on the device; only every ``check_every`` steps is the stop read back.
        """
        ran = ~self.stopped
        total = started.sum().to(self.dtype)  # wo-4.0 summed the losses of a batch of tiles
        improved = total < self.lowest - self.stop.tolerance
        self.lowest = torch.where(ran & improved, total, self.lowest)
        self.stalled = torch.where(ran, torch.where(improved, 0, self.stalled + 1), self.stalled)
        self.counted = self.counted + ran.long()
        self.stopped = self.stopped | (self.stalled >= self.stop.patience)
        with torch.no_grad():
            for name, v in variables.items():
                v.copy_(torch.where(ran, v, self.held[name]))
        return (self.step + 1) % self.stop.check_every == 0 and bool(self.stopped)

    def report(self) -> None:
        """Hand the evaluations of the steps run, and the last one, to :func:`record_step`."""
        counted = int(self.counted)
        for step, objective, point in self.pending:
            if step < counted:
                record_step(objective, point)


def _take_steps(
    built: torch.optim.Optimizer,
    closure: Callable[[], Tensor],
    project: Callable[[], None],
    *,
    variables: Mapping[str, Tensor],
    steps: int,
    stall: _Stall | None,
) -> None:
    """Take ``steps`` optimizer steps, projecting after each, or fewer once ``stall`` stops."""
    for index in range(steps):
        if stall is not None:
            stall.begin(index, variables)
        # The stubs ask for a float closure; a tensor works with both optimizers and spares Adam a host sync. Both
        # return the objective the step started from.
        started = cast(Tensor, built.step(cast(Callable[[], float], closure)))
        project()
        if stall is not None and stall.end(started, variables):
            break
    if stall is not None:
        stall.step = -1  # the evaluation after the last step is always reported


def _descend(  # noqa: PLR0913 (the objective and one keyword per loop setting)
    objective: CalibrationObjective,
    *,
    start: Estimate,
    optimizer: _OptimizerFactory,
    steps: int,
    span_units: bool,
    loss_scale: float = 1.0,
    last: bool = False,
    stop: _Stop | None = None,
) -> Estimate:
    """Minimize ``objective`` from ``start`` with ``steps`` optimizer steps, tracking the best point.

    Variables are the quantity, or its log for a geometric span, divided by a tenth of the (log) span with
    ``span_units``. Values may carry a leading tile axis ``(B,)``; each tile then keeps its own best point. With
    ``stop``, a step after the stop leaves the variables where they were, and only the evaluations of the steps
    before it (and the last one) are reported.
    """
    specs = objective.specs
    names = list(specs)
    spans = {
        name: _span(name, spec) if spec.bounds or spec.tolerance else (0.0, max(abs(spec.nominal), 1.0))
        for name, spec in specs.items()
    }
    space = _Space(spans, frozenset(name for name, (low, high) in spans.items() if _geometric(low, high)))
    scales = {
        name: 0.1 * (math.log(high / low) if name in space.logged else high - low) if span_units else 1.0
        for name, (low, high) in spans.items()
    }

    def to_variable(name: str, value: Tensor) -> Tensor:
        return (value.log() if name in space.logged else value) / scales[name]

    variables = {
        name: to_variable(name, value).detach().clone().requires_grad_() for name, value in start.point.items()
    }
    # The best point so far stays on the device as one stacked tensor (quantities last), so tracking it never waits.
    best_point = torch.stack([start.point[name].detach() for name in names], dim=-1)
    best_value = torch.where(torch.isfinite(start.objective), start.objective.detach(), math.inf)

    def point() -> dict[str, Tensor]:
        raw = {
            name: (v * scales[name]).exp() if name in space.logged else v * scales[name]
            for name, v in variables.items()
        }
        # Clamped in value, but the gradient passes through: a line-search trial past a bound then still sees the
        # slope instead of zero, which would stall L-BFGS there.
        clamped = objective.prior.clamp(raw)
        return {name: value + (clamped[name] - value).detach() for name, value in raw.items()}

    last_point, last_value = start.point, start.objective
    stall = None if stop is None else _Stall(stop, start.objective)
    # The objective and point at each evaluation, for calibrate(..., history=True); a stop holds them back.
    keep = record_step if stall is None else stall.keep

    def evaluate_and_track() -> Tensor:
        nonlocal best_point, best_value, last_point, last_value
        last_point = point()
        value = objective(last_point)
        detached = last_value = value.detach()
        keep(detached, last_point)
        improved = torch.isfinite(detached) & (detached < best_value)
        stacked = torch.stack([last_point[name].detach() for name in names], dim=-1)
        best_point = torch.where(improved[..., None], stacked, best_point)
        best_value = torch.where(improved, detached, best_value)
        return value

    def project() -> None:
        with torch.no_grad():
            for name, tensor in point().items():
                variables[name].copy_(to_variable(name, tensor))

    built = optimizer(variables, space)

    def closure() -> Tensor:
        built.zero_grad()
        value = evaluate_and_track()
        (value.sum() * loss_scale).backward()
        return value.detach()

    _take_steps(built, closure, project, variables=variables, steps=steps, stall=stall)
    with torch.no_grad():
        evaluate_and_track()
    if stall is not None:
        stall.report()
    if last:
        return Estimate({name: value.detach().clone() for name, value in last_point.items()}, last_value)
    best = {name: best_point[..., index].reshape_as(start.point[name]).clone() for index, name in enumerate(names)}
    return Estimate(best, best_value)
