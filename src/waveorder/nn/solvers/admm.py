"""ADMM for total variation under white Gaussian noise (3.x ``util.single_variable_admm_tv_...``)."""

import math
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.functional import (
    abs2,
    add_diagonal,
    cholesky_hermitian,
    cholesky_solve,
    forward_differences,
    forward_differences_adjoint,
    from_half_spectrum,
    gram,
    to_half_spectrum,
)
from waveorder.nn.model import ImagingModel
from waveorder.nn.operators import FourierDiagonal
from waveorder.nn.parameters import repr_parts
from waveorder.nn.priors import TotalVariation
from waveorder.nn.sampling import Object
from waveorder.nn.solvers.base import IterativeSolver, Loop
from waveorder.nn.verbs import InverseProblem
from waveorder.registry import register

_DEFAULT_LOOP = Loop(10)


@dataclass(frozen=True, slots=True)
class ADMMState:
    """Iterate ``x``, split gradient ``z``, scaled dual ``u`` (both ``(3, F, Z, Y, X)``) and ``H* g``."""

    x: Tensor
    z: Tensor
    u: Tensor
    backprojection: Tensor


@register("admm", kind="solver")
class ADMM(IterativeSolver[ADMMState]):
    """Minimize ``||H f - g||^2 / 2 + weight ||grad f||_{2,1} + damping ||f||^2 / 2`` by ADMM on ``z = grad f``.

    This is ``s`` times the MAP objective under white Gaussian noise of variance ``s`` and a
    :class:`~waveorder.nn.priors.TotalVariation` prior, plus 3.x's Tikhonov term (``reg_re``). Each iteration
    solves ``(H* H + rho grad* grad + damping) f = H* g + rho grad*(z - u)`` exactly per frequency, group-soft-
    thresholds ``grad f + u`` at ``weight / rho`` per voxel over the three axes, and sets ``u = u + grad f - z``.
    Gradients are periodic forward differences. Unlike 3.x (anisotropic TV), the threshold is isotropic.

    Parameters
    ----------
    loop : Loop
        Iterations and stopping (default ``Loop(10)``); ``N`` iterations make ``N`` ``f``-updates from zero.
    rho : float
        Augmented Lagrangian weight, in the units of ``s`` times the objective; positive and finite.
    damping : float
        Tikhonov weight, in the same units; nonnegative and finite. Zero needs ``H`` to pass zero frequency.

    Raises
    ------
    ParameterError
        If ``rho`` is not positive and finite or ``damping`` not nonnegative and finite.
    """

    def __init__(self, *, loop: Loop = _DEFAULT_LOOP, rho: float = 1e-3, damping: float = 1e-3) -> None:
        super().__init__(loop=loop)
        for name, value, positive in (("rho", rho, True), ("damping", damping, False)):
            if not math.isfinite(value) or value < 0 or (positive and value == 0):
                bound = "positive" if positive else "nonnegative"
                raise ParameterError(f"ADMM {name} must be {bound} and finite, got {value}", path=name)
        self.rho = float(rho)
        self.damping = float(damping)

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise unless the system is Fourier-diagonal, the noise white Gaussian and the prior total variation."""
        if not issubclass(model.system.operator_type, FourierDiagonal):
            raise IncompatibleModelError(
                f"ADMM solves its f-update per frequency, so it needs a system that binds to a FourierDiagonal; "
                f"{type(model.system).__name__} binds to {model.system.operator_type.__name__}. Use FISTA"
            )
        if model.noise.white_gaussian_variance() is None:
            raise IncompatibleModelError(
                f"ADMM needs white Gaussian noise, got {type(model.noise).__name__}; use noise=Gaussian(sigma), or "
                "RichardsonLucy with object_prior=NoPrior() for Poisson noise"
            )
        prior = model.object_prior
        if not isinstance(prior, TotalVariation) or prior.nonnegative:
            raise IncompatibleModelError(
                f"ADMM splits the total-variation prior without a support constraint; got "
                f"{type(prior).__name__}{' (nonnegative)' if prior.nonnegative else ''}. Use FISTA"
            )

    @override
    def extra_repr(self) -> str:
        return repr_parts(super().extra_repr(), f"rho={self.rho:g}", f"damping={self.damping:g}")

    @override
    def start(self, problem: InverseProblem) -> ADMMState:
        fourier = _fourier(problem)
        backprojection = fourier.adjoint(problem.data.reshape(fourier.data_shape))
        x = torch.zeros_like(backprojection)
        split = torch.zeros_like(forward_differences(x))
        return ADMMState(x=x, z=split, u=split.clone(), backprojection=backprojection)

    @override
    def step(self, state: ADMMState, problem: InverseProblem) -> ADMMState:
        rhs = state.backprojection + self.rho * forward_differences_adjoint(state.z - state.u)
        x = _solve_update(_fourier(problem), rhs, rho=self.rho, damping=self.damping)
        gradient = forward_differences(x)
        z = _shrink(gradient + state.u, _total_variation(problem).weight / self.rho)
        return ADMMState(x=x, z=z, u=state.u + gradient - z, backprojection=state.backprojection)

    @override
    def estimate(self, state: ADMMState) -> Object:
        return state.x


def _fourier(problem: InverseProblem) -> FourierDiagonal:
    fourier = problem.operator.as_fourier()
    if fourier is None:
        raise IncompatibleModelError(f"ADMM needs a Fourier-diagonal operator, got {type(problem.operator).__name__}")
    return fourier


def _total_variation(problem: InverseProblem) -> TotalVariation:
    prior = problem.model.object_prior
    if not isinstance(prior, TotalVariation):  # check() refused the model before binding
        raise IncompatibleModelError(f"ADMM needs a TotalVariation prior, got {type(prior).__name__}")
    return prior


def _solve_update(fourier: FourierDiagonal, rhs: Tensor, *, rho: float, damping: float) -> Tensor:
    """Return ``x`` with ``(H* H + rho grad* grad + damping I) x = rhs``, solved per frequency."""
    shape = fourier.sampling.shape
    differences = _difference_transfers(shape, like=fourier.transfer)  # (3, Z, Y, X', 2)
    diagonal = rho * abs2(differences).sum(dim=0) + damping
    factor = cholesky_hermitian(add_diagonal(gram(fourier.transfer), diagonal))
    return from_half_spectrum(cholesky_solve(factor, to_half_spectrum(rhs)), shape)


def _difference_transfers(shape: tuple[int, int, int], *, like: Tensor) -> Tensor:
    """Half-spectrum transfers of the periodic forward differences ``f[n + 1] - f[n]`` along z, y, x."""
    kernels = torch.zeros((3, *shape), dtype=like.dtype, device=like.device)
    for axis in range(3):
        index = [axis, 0, 0, 0]
        kernels[tuple(index)] = -1.0
        index[axis + 1] = -1  # h[-1] = 1 gives (h * f)[n] = f[n + 1]
        kernels[tuple(index)] += 1.0
    return to_half_spectrum(kernels)


def _shrink(values: Tensor, threshold: Tensor) -> Tensor:
    """Group soft threshold over the axis dimension: ``v * max(0, 1 - threshold / ||v||)``."""
    norm = torch.linalg.vector_norm(values, dim=0, keepdim=True)
    return values * (1 - threshold / norm.clamp_min(torch.finfo(values.dtype).tiny)).clamp_min(0.0)
