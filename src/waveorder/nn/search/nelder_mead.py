"""Nelder-Mead simplex search: a gradient-free stage that decides on the host (3.x ``method="nelder_mead"``).

It reimplements scipy's ``minimize(method="Nelder-Mead")`` with ``maxiter = maxfev``, as 3.x ran it: the same
initial simplex, coefficients, stopping rule and bounded variant, without depending on scipy.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch

from waveorder.nn.search.base import CalibrationObjective, Estimate, _expect_count
from waveorder.nn.search.descent import _expect_rate
from waveorder.registry import register

type _Point = list[float]

_REFLECT, _EXPAND, _CONTRACT, _SHRINK = 1.0, 2.0, 0.5, 0.5
_STEP, _STEP_FROM_ZERO = 0.05, 0.00025


@register("nelder_mead", kind="search")
@dataclass(frozen=True, slots=True, kw_only=True)
class NelderMeadSearch:
    """Refine with the Nelder-Mead simplex method, gradient-free, in each quantity's own units.

    The simplex starts at ``start`` with one vertex per quantity moved by 5% of its value (``0.00025`` from zero);
    new vertices are clipped to the bounds. The search stops after ``iterations`` iterations or evaluations, or when
    every vertex is within ``tolerance`` of the best in every coordinate and in value. A NaN objective reads as
    ``+inf``.

    Parameters
    ----------
    iterations : int
        Cap on both the simplex iterations and the objective evaluations, at least 1; 3.x's ``max_iterations``.
    tolerance : float
        Stopping tolerance in each quantity's units and in objective units, positive and finite; scipy's 1e-4.

    Raises
    ------
    ParameterError
        For ``iterations`` below 1 or a ``tolerance`` that is not positive and finite.
    """

    iterations: int = 200
    tolerance: float = 1e-4
    reads_back: ClassVar[bool] = True
    """Each simplex step compares objective values on the host."""
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        _expect_count(self.iterations, where="NelderMeadSearch", name="iterations", minimum=1)
        _expect_rate(self.tolerance, where="NelderMeadSearch", name="tolerance")

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best vertex found from ``start``, or ``start`` if none is better."""
        specs = objective.specs
        names = list(specs)
        like = next(iter(start.point.values()))

        def tensors(x: Sequence[float]) -> dict[str, torch.Tensor]:
            return {
                name: torch.tensor(value, dtype=like.dtype, device=like.device)
                for name, value in zip(names, x, strict=True)
            }

        def value(x: Sequence[float]) -> float:
            with torch.no_grad():
                return float(objective(tensors(x)))

        x0 = [float(start.point[name]) for name in names]
        bounds = [specs[name].bounds for name in names]
        best, lowest = _minimize(value, x0, bounds=bounds, iterations=self.iterations, tolerance=self.tolerance)
        if not lowest < float(start.objective):
            return start
        point = tensors(best)
        return Estimate(point, torch.tensor(lowest, dtype=start.objective.dtype, device=like.device))


class _OutOfEvaluations(Exception):  # noqa: N818 (a control-flow signal, as scipy's _MaxFuncCallError)
    """The evaluation budget is spent: the step in progress stops, as scipy's ``maxfev`` check does."""


def _minimize(
    function: Callable[[_Point], float],
    x0: _Point,
    *,
    bounds: Sequence[tuple[float, float] | None],
    iterations: int,
    tolerance: float,
) -> tuple[_Point, float]:
    """Return the best vertex and its value: scipy's ``_minimize_neldermead`` with ``maxiter = maxfev``."""
    size = len(x0)
    low = [-math.inf if span is None else span[0] for span in bounds]
    high = [math.inf if span is None else span[1] for span in bounds]
    calls = 0

    def clip(x: _Point) -> _Point:
        return [min(max(v, lo), hi) for v, lo, hi in zip(x, low, high, strict=True)]

    def counted(x: _Point) -> float:
        nonlocal calls
        if calls >= iterations:
            raise _OutOfEvaluations
        calls += 1
        result = function(x)
        return math.inf if math.isnan(result) else result

    x0 = clip(x0)
    simplex = [list(x0)]
    for k in range(size):
        vertex = list(x0)
        vertex[k] = (1 + _STEP) * vertex[k] if vertex[k] != 0 else _STEP_FROM_ZERO
        simplex.append(vertex)
    # A vertex moved past its upper bound is reflected inside, then clipped, so the simplex does not collapse.
    simplex = [clip([2 * hi - v if v > hi else v for v, hi in zip(vertex, high, strict=True)]) for vertex in simplex]
    values = [math.inf] * (size + 1)
    try:
        for k in range(size + 1):
            values[k] = counted(simplex[k])
    except _OutOfEvaluations:
        pass
    simplex, values = _sorted(simplex, values)
    iteration = 1
    while calls < iterations and iteration < iterations:
        try:
            if _converged(simplex, values, tolerance=tolerance):
                break
            _step(simplex, values, evaluate=counted, clip=clip)
        except _OutOfEvaluations:
            pass
        finally:
            simplex, values = _sorted(simplex, values)
        iteration += 1
    return simplex[0], values[0]


def _converged(simplex: list[_Point], values: list[float], *, tolerance: float) -> bool:
    spread = max(abs(v - b) for vertex in simplex[1:] for v, b in zip(vertex, simplex[0], strict=True))
    return spread <= tolerance and max(abs(values[0] - value) for value in values[1:]) <= tolerance


def _step(
    simplex: list[_Point],
    values: list[float],
    *,
    evaluate: Callable[[_Point], float],
    clip: Callable[[_Point], _Point],
) -> None:
    """One Nelder-Mead iteration on a sorted simplex, in place: reflect, then expand, contract or shrink."""
    size = len(simplex) - 1
    centroid = [sum(vertex[j] for vertex in simplex[:-1]) / size for j in range(size)]
    worst = simplex[-1]

    def toward(a: float, b: float) -> _Point:
        return clip([a * c - b * w for c, w in zip(centroid, worst, strict=True)])

    reflected = toward(1 + _REFLECT, _REFLECT)
    at_reflected = evaluate(reflected)
    if at_reflected < values[0]:
        expanded = toward(1 + _REFLECT * _EXPAND, _REFLECT * _EXPAND)
        at_expanded = evaluate(expanded)
        simplex[-1], values[-1] = (expanded, at_expanded) if at_expanded < at_reflected else (reflected, at_reflected)
        return
    if at_reflected < values[-2]:
        simplex[-1], values[-1] = reflected, at_reflected
        return
    if at_reflected < values[-1]:
        contracted = toward(1 + _CONTRACT * _REFLECT, _CONTRACT * _REFLECT)
        at_contracted = evaluate(contracted)
        if at_contracted <= at_reflected:
            simplex[-1], values[-1] = contracted, at_contracted
            return
    else:
        inside = clip([(1 - _CONTRACT) * c + _CONTRACT * w for c, w in zip(centroid, worst, strict=True)])
        at_inside = evaluate(inside)
        if at_inside < values[-1]:
            simplex[-1], values[-1] = inside, at_inside
            return
    for j in range(1, size + 1):
        simplex[j] = clip([b + _SHRINK * (v - b) for v, b in zip(simplex[j], simplex[0], strict=True)])
        values[j] = evaluate(simplex[j])


def _sorted(simplex: list[_Point], values: list[float]) -> tuple[list[_Point], list[float]]:
    """Order the vertices by value, ties kept in place (numpy's argsort is stable on so few)."""
    order = sorted(range(len(values)), key=values.__getitem__)
    return [simplex[k] for k in order], [values[k] for k in order]
