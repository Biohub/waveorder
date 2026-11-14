"""FISTA: accelerated proximal gradient for a smooth data term plus any prior with a proximal map."""

from dataclasses import dataclass, replace
from typing import override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError
from waveorder.nn.model import ImagingModel
from waveorder.nn.sampling import Object
from waveorder.nn.solvers.base import IterativeSolver, Loop
from waveorder.nn.verbs import InverseProblem
from waveorder.registry import register

_DEFAULT_LOOP = Loop(50)


@dataclass(frozen=True, slots=True)
class FISTAState:
    """Iterate ``x``, extrapolated point ``z``, momentum ``t`` and step size.

    For white Gaussian noise, ``backprojection`` is ``H* g`` and ``noise_variance`` is ``sigma^2``; else ``None``.
    """

    x: Tensor
    z: Tensor
    t: Tensor
    step_size: Tensor
    backprojection: Tensor | None
    noise_variance: Tensor | None


@register("fista", kind="solver")
class FISTA(IterativeSolver[FISTAState]):
    """Minimize ``D(g; H f) + P(f)`` by FISTA (Beck and Teboulle, SIAM J. Imaging Sci. 2009).

    Works with any system, any noise with :attr:`~waveorder.nn.noise.Noise.has_curvature_bound` and any object prior
    with a ``prox``. The step size is ``1 / L`` with ``L = curvature * ||H||^2``. For white Gaussian noise the
    gradient is ``(H* H f - H* g) / sigma^2`` with ``H* g`` computed once. Gradient restart (O'Donoghue and Candes,
    2015) resets the momentum when it stops descending.

    Parameters
    ----------
    loop : Loop
        Iterations and stopping (default ``Loop(50)``).
    """

    def __init__(self, *, loop: Loop = _DEFAULT_LOOP) -> None:
        super().__init__(loop=loop)

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise :class:`IncompatibleModelError` unless the noise declares a bounded curvature (Lipschitz gradient)."""
        if model.noise.has_curvature_bound:
            return
        noise, system = type(model.noise).__name__, type(model.system).__name__
        refusal = (
            f"FISTA needs a data term with a Lipschitz gradient; {noise} noise declares has_curvature_bound = False"
        )
        if model.system.preserves_nonnegativity:
            raise IncompatibleModelError(f"{refusal}. Use RichardsonLucy with object_prior=NoPrior() for Poisson noise")
        raise IncompatibleModelError(
            f"{refusal}, and RichardsonLucy cannot run in its place: {system} does not declare "
            f"preserves_nonnegativity. Pass noise=Gaussian(sigma); or, if the point-spread function of {system} is "
            "nonnegative, set preserves_nonnegativity = True on the class and use RichardsonLucy with "
            "object_prior=NoPrior()"
        )

    @override
    def start(self, problem: InverseProblem) -> FISTAState:
        operator = problem.operator
        lipschitz = problem.model.noise.curvature_bound() * operator.norm() ** 2
        # A zero operator has no data term to descend; any finite step then minimizes the prior alone.
        step_size = 1 / lipschitz.clamp_min(torch.finfo(lipschitz.dtype).tiny)
        backprojection, variance = _white_gaussian_terms(problem)
        adjoint = operator.adjoint(problem.data) if backprojection is None else backprojection
        x = problem.model.object_prior.project(adjoint)
        one = torch.ones((), dtype=x.dtype, device=x.device)
        return FISTAState(x=x, z=x, t=one, step_size=step_size, backprojection=backprojection, noise_variance=variance)

    @override
    def step(self, state: FISTAState, problem: InverseProblem) -> FISTAState:
        gradient = _smooth_gradient(
            state.z, problem, backprojection=state.backprojection, variance=state.noise_variance
        )
        x = _prox_step(state.z, gradient, step_size=state.step_size, problem=problem)
        ascending = ((state.z - x) * (x - state.x)).sum() > 0
        t = torch.where(ascending, torch.ones_like(state.t), state.t)
        t_next = (1 + torch.sqrt(1 + 4 * t * t)) / 2
        z = x + ((t - 1) / t_next) * (x - state.x)
        return replace(state, x=x, z=z, t=t_next)

    @override
    def estimate(self, state: FISTAState) -> Object:
        return state.x


def _white_gaussian_terms(problem: InverseProblem) -> tuple[Object | None, Tensor | None]:
    """``(H* g, sigma^2)`` for white Gaussian noise (its gradient is then affine in ``H* H f``), else ``None``s."""
    variance = problem.model.noise.white_gaussian_variance()
    if variance is None:
        return None, None
    return problem.operator.adjoint(problem.data), variance


def _smooth_gradient(
    point: Object, problem: InverseProblem, *, backprojection: Object | None, variance: Tensor | None
) -> Object:
    """Return the gradient of the data term ``D(g; H f)`` at ``point``, given :func:`_white_gaussian_terms`."""
    operator = problem.operator
    if backprojection is None or variance is None:
        return operator.adjoint(problem.model.noise.gradient(problem.data, operator.apply(point)))
    return (problem.normal_action()(point) - backprojection) / variance


def _prox_step(point: Object, gradient: Object, *, step_size: Tensor, problem: InverseProblem) -> Object:
    model = problem.model
    return model.object_prior.prox(point - step_size * gradient, step_size, scale=model.noise.scale())
