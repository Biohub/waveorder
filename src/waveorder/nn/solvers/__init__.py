"""Solvers: the inner problem, ``f_hat = argmin_f D(g; H f) + P(f)``.

Every solver returns a :class:`SolveResult`. :class:`Wiener` is closed form and needs a linear-Gaussian model whose
system binds to a Fourier-diagonal operator; iterative solvers subclass :class:`IterativeSolver` and take one
:class:`Loop`.
"""

from waveorder.nn.solvers.admm import ADMM, ADMMState
from waveorder.nn.solvers.base import IterativeSolver, Loop, Solver, SolveResult
from waveorder.nn.solvers.fista import FISTA, FISTAState
from waveorder.nn.solvers.richardson_lucy import BACK_PROJECTORS, BackProjector, RichardsonLucy, RichardsonLucyState
from waveorder.nn.solvers.wiener import Wiener

__all__ = [
    "ADMM",
    "BACK_PROJECTORS",
    "FISTA",
    "ADMMState",
    "BackProjector",
    "FISTAState",
    "IterativeSolver",
    "Loop",
    "RichardsonLucy",
    "RichardsonLucyState",
    "SolveResult",
    "Solver",
    "Wiener",
]
