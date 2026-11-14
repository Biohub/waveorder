"""Gradient descent stages: refine the free quantities with a torch optimizer, clamped to their bounds.

:class:`GradientSearch` and :class:`LBFGSSearch` share one loop: geometric spans
(:func:`~waveorder.nn.search.search_spans`) are optimized in log space, every evaluated point is clamped to the
bounds, and the best point visited is tracked on the device.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal, cast

import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.nn.search.base import CalibrationObjective, Estimate, _expect_count, _geometric, expect_known_names
from waveorder.nn.search.grid import _span
from waveorder.observe import record_step
from waveorder.registry import register


def _expect_rate(rate: float, *, where: str, name: str) -> None:
    if not (math.isfinite(rate) and rate > 0):
        raise ParameterError(f"{where} {name} must be positive and finite, got {rate}", path=name)


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
        Step size as a fraction of each quantity's span, or in log units for a geometric span; positive and finite.
    learning_rates : Mapping[str, float]
        Step sizes by dotted name, in the same units, overriding ``learning_rate``.
    optimizer : {"adam", "nadam", "adamw", "sgd"}
        The torch optimizer; ``"adam"`` by default.
    loss_scale : float
        Factor on the objective before each backward pass; positive and finite. It matters for SGD, not Adam.
    last : bool
        Return the last iterate instead of the best point visited, as the 3.x gradient stages did.

    Raises
    ------
    ParameterError
        For ``iterations`` below 1, a rate or ``loss_scale`` that is not positive and finite, or an unknown
        ``optimizer``.
    """

    iterations: int = 200
    learning_rate: float = 0.02
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    optimizer: Optimizer = "adam"
    loss_scale: float = 1.0
    last: bool = False
    reads_back: ClassVar[bool] = False
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

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best point the optimizer visits from ``start``, or its last one with :attr:`last`."""
        expect_known_names(self.learning_rates, objective.specs, path="GradientSearch.learning_rates")
        return _descend(
            objective,
            start=start,
            optimizer=self._build,
            steps=self.iterations,
            span_units=False,
            loss_scale=self.loss_scale,
            last=self.last,
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
    """

    iterations: int = 20
    history: int = 10
    reads_back: ClassVar[bool] = True
    """The strong-Wolfe line search reads objective values back to the host to choose its step."""
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        _expect_count(self.iterations, where="LBFGSSearch", name="iterations", minimum=1)
        _expect_count(self.history, where="LBFGSSearch", name="history", minimum=1)

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best point L-BFGS visits from ``start``."""
        # One optimizer step runs every iteration, so the line search keeps its state across them.
        return _descend(objective, start=start, optimizer=self._lbfgs, steps=1, span_units=True)

    def _lbfgs(self, variables: Mapping[str, Tensor], space: _Space) -> torch.optim.Optimizer:
        del space  # one step size for all variables: span_units puts them on a common scale instead
        return torch.optim.LBFGS(
            list(variables.values()), max_iter=self.iterations, history_size=self.history, line_search_fn="strong_wolfe"
        )


def _descend(  # noqa: PLR0913 (the objective and one keyword per loop setting)
    objective: CalibrationObjective,
    *,
    start: Estimate,
    optimizer: _OptimizerFactory,
    steps: int,
    span_units: bool,
    loss_scale: float = 1.0,
    last: bool = False,
) -> Estimate:
    """Minimize ``objective`` from ``start`` with ``steps`` optimizer steps, tracking the best point.

    Variables are the quantity, or its log for a geometric span, divided by a tenth of the (log) span with
    ``span_units``. Values may carry a leading tile axis ``(B,)``; each tile then keeps its own best point.
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

    def evaluate_and_track() -> Tensor:
        nonlocal best_point, best_value, last_point, last_value
        last_point = point()
        value = objective(last_point)
        detached = last_value = value.detach()
        record_step(detached)  # the objective at each evaluation, for an observer (calibrate collects it)
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

    for _ in range(steps):
        # The stubs ask for a float closure; a tensor works with both optimizers and spares Adam a host sync.
        built.step(cast(Callable[[], float], closure))
        project()
    with torch.no_grad():
        evaluate_and_track()
    if last:
        return Estimate({name: value.detach().clone() for name, value in last_point.items()}, last_value)
    best = {name: best_point[..., index].reshape_as(start.point[name]).clone() for index, name in enumerate(names)}
    return Estimate(best, best_value)
