"""Exact inference for a shift-invariant system with a Gaussian object prior and white Gaussian noise.

Everything is per spatial frequency: the data's orthonormal Fourier coefficients ``G_k`` have ``C x C`` covariance
``V_k = H_k S_k H_k^H + sigma^2 I``, and the object's posterior has ``F x F`` precision
``A_k = S_k^-1 + H_k^H H_k / sigma^2``. The Wiener solver and the evidence score read from :class:`LinearGaussian`.
Spectra are orthonormal ``rfftn`` half spectra stored real.
"""

import math
from dataclasses import dataclass
from typing import Self, override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError
from waveorder.nn.functional import (
    HermitianFactor,
    abs2,
    add_diagonal,
    channel_covariance,
    cholesky_hermitian,
    cholesky_logdet,
    cholesky_solve,
    from_half_spectrum,
    gram,
    half_spectrum_weights,
    matrix_vector,
    to_half_spectrum,
)
from waveorder.nn.model import ImagingModel
from waveorder.nn.operators import FourierDiagonal, LinearOperator, ReshapedFourierDiagonal
from waveorder.nn.priors import GaussianObjectPrior
from waveorder.nn.sampling import ZYX, Data, Object
from waveorder.nn.solvers.base import Solver, SolveResult
from waveorder.nn.verbs import InverseProblem
from waveorder.registry import register


@dataclass(frozen=True, slots=True)
class LinearGaussianParts:
    """What :func:`check_linear_gaussian` verified: the narrowed prior and the noise variance (data units^2)."""

    prior: GaussianObjectPrior
    noise_variance: Tensor


@dataclass(frozen=True, slots=True)
class Fixes:
    """What :func:`check_linear_gaussian` tells the caller to do, per refused component of the model."""

    operator: str
    prior: str
    nonnegative: str
    noise: str


SOLVER_FIXES = Fixes(
    operator="use FISTA",
    prior="use FISTA or ADMM (ADMM takes TotalVariation only)",
    nonnegative="use FISTA",
    noise="use RichardsonLucy with object_prior=NoPrior() for Poisson noise, or FISTA for noise with a curvature bound",
)
"""For a closed-form solver: the solvers that take the model as it is."""

SCORE_FIXES = Fixes(
    operator=(
        "use a sharpness score, e.g. MidbandPowerScore(solver=FISTA(), cutoff=2 * NA / wavelength) "
        "(score=MidbandPowerScore() on a waveorder.models model)"
    ),
    prior="use object_prior=Tikhonov(...), or a sharpness score such as MidbandPowerScore",
    nonnegative="pass nonnegative=False, or use a sharpness score such as MidbandPowerScore",
    noise="use noise=Gaussian(sigma), or a sharpness score such as MidbandPowerScore",
)
"""For a closed-form score: the parts to change, or the scores that take any model."""

PART_FIXES = Fixes(
    operator="use a shift-invariant system",
    prior="use object_prior=Tikhonov(...)",
    nonnegative="pass nonnegative=False",
    noise="use noise=Gaussian(sigma)",
)
"""For linear-Gaussian inference: the parts to change."""


def check_linear_gaussian(model: ImagingModel, *, verb: str, fixes: Fixes = PART_FIXES) -> LinearGaussianParts:
    """Raise :class:`IncompatibleModelError`, naming the component and the fix, unless ``model`` is linear-Gaussian.

    Linear-Gaussian means a system that binds to a :class:`~waveorder.nn.operators.FourierDiagonal` or a
    :class:`~waveorder.nn.operators.ReshapedFourierDiagonal`, a Gaussian object prior without a nonnegativity
    constraint, and noise that reports a :meth:`~waveorder.nn.noise.Noise.white_gaussian_variance`. Each message
    ends with the fix from ``fixes``.
    """
    system, prior, noise = model.system, model.object_prior, model.noise
    if not issubclass(system.operator_type, FourierDiagonal | ReshapedFourierDiagonal):
        raise IncompatibleModelError(
            f"{verb} needs a system that binds to a FourierDiagonal operator; {type(system).__name__} binds to "
            f"{system.operator_type.__name__}; {fixes.operator}"
        )
    if not isinstance(prior, GaussianObjectPrior):
        raise IncompatibleModelError(f"{verb} needs a Gaussian object prior, got {type(prior).__name__}; {fixes.prior}")
    if prior.nonnegative:
        raise IncompatibleModelError(f"{verb} has no closed form under a nonnegativity constraint; {fixes.nonnegative}")
    noise_variance = noise.white_gaussian_variance()
    if noise_variance is None:
        raise IncompatibleModelError(f"{verb} needs white Gaussian noise, got {type(noise).__name__}; {fixes.noise}")
    return LinearGaussianParts(prior, noise_variance)


@dataclass(frozen=True, slots=True)
class GaussianPosterior:
    """The posterior ``p(f | g)`` per frequency: precision ``A_k`` and information vector ``b_k = H_k^H G_k / s``.

    ``precision`` is ``(F, F, Z, Y, X // 2 + 1)`` real, or with a trailing real/imaginary axis when complex;
    ``information`` is ``(F, Z, Y, X // 2 + 1, 2)``.
    """

    precision: Tensor
    information: Tensor
    shape: ZYX[int]

    def factor(self) -> HermitianFactor:
        """Return the Cholesky factor of the precision (see :func:`~waveorder.nn.functional.cholesky_hermitian`)."""
        return cholesky_hermitian(self.precision)

    def mean(self) -> Object:
        """Return the posterior mean, which is also the MAP estimate: shape ``(F, Z, Y, X)``."""
        if self.precision.shape[0] == 1:  # a real scalar per frequency: divide, no factor
            spectrum = self.information / self.precision[0][..., None]
        else:
            spectrum = cholesky_solve(self.factor(), self.information)
        return from_half_spectrum(spectrum, self.shape, norm="ortho")


@dataclass(frozen=True, slots=True)
class LinearGaussian:
    """A linear-Gaussian model on one sampling: half-spectrum transfer, prior spectrum and noise variance.

    Data of any shape with ``C * prod(shape)`` elements are read as ``(C, *shape)``.
    """

    transfer: Tensor
    spectrum: Tensor
    noise_variance: Tensor
    shape: ZYX[int]

    @classmethod
    def of(cls, model: ImagingModel, operator: LinearOperator) -> Self:
        """Assemble from ``model`` and its system bound to a sampling.

        Raises
        ------
        IncompatibleModelError
            If ``model`` is not linear-Gaussian or ``operator`` is not Fourier-diagonal.
        """
        parts = check_linear_gaussian(model, verb="linear-Gaussian inference")
        fourier = operator.as_fourier()
        if fourier is None:
            raise IncompatibleModelError(
                f"linear-Gaussian inference needs a Fourier-diagonal operator, got {type(operator).__name__}"
            )
        spectrum = parts.prior.spectrum(fourier.sampling, scale=parts.noise_variance)
        return cls(fourier.transfer, spectrum, parts.noise_variance, fourier.sampling.shape)

    def _coefficients(self, data: Data) -> Tensor:
        return to_half_spectrum(data.reshape(self.transfer.shape[0], *self.shape), norm="ortho")

    def posterior(self, data: Data) -> GaussianPosterior:
        """Return ``p(f | data)``."""
        information = matrix_vector(self.transfer, self._coefficients(data), adjoint=True) / self.noise_variance
        return GaussianPosterior(self._precision(), information, self.shape)

    def _precision(self) -> Tensor:
        return add_diagonal(gram(self.transfer) / self.noise_variance, 1 / self.spectrum)

    def neg_log_evidence(self, data: Data) -> Tensor:
        """Return ``-log p(data | H) = sum_k G_k^H V_k^-1 G_k / 2 + log det(2 pi V_k) / 2`` without the zero frequency.

        The zero frequency is left out, as in 3.x, so a constant background the prior does not model cannot drag the
        noise level and the weight.
        """
        terms = self.evidence_terms(data)
        weights = half_spectrum_weights(self.shape[-1], device=terms.device, dtype=terms.dtype)
        count = self.transfer.shape[0] * (math.prod(self.shape) - 1)
        return (terms * weights).sum() - terms[0, 0, 0] + 0.5 * count * math.log(2 * math.pi)

    def evidence_terms(self, data: Data) -> Tensor:
        """Return ``G_k^H V_k^-1 G_k / 2 + log det V_k / 2`` per half-spectrum frequency, shape ``(Z, Y, X // 2 + 1)``.

        Unweighted: :func:`~waveorder.nn.functional.half_spectrum_weights` sums them to the full spectrum. With more
        channels than fields, the Woodbury identity avoids forming the ``C x C`` covariance.
        """
        G = self._coefficients(data)
        channels, fields = self.transfer.shape[:2]
        if channels > fields:
            return self._woodbury_terms(G)
        if channels == 1:  # V is a real scalar per frequency: |G|^2 / V + log V
            variance = channel_covariance(self.transfer, self.spectrum, self.noise_variance)[0, 0]
            return 0.5 * (abs2(G[0]) / variance + torch.log(variance))
        factor = cholesky_hermitian(channel_covariance(self.transfer, self.spectrum, self.noise_variance))
        quadratic = (G * cholesky_solve(factor, G)).sum(dim=(0, -1))  # Re(G^H V^-1 G)
        return 0.5 * (quadratic + cholesky_logdet(factor))

    def _woodbury_terms(self, G: Tensor) -> Tensor:
        channels, fields = self.transfer.shape[:2]
        information = matrix_vector(self.transfer, G, adjoint=True) / self.noise_variance
        factor = cholesky_hermitian(self._precision())
        explained = (information * cholesky_solve(factor, information)).sum(dim=(0, -1))  # Re(b^H A^-1 b)
        quadratic = abs2(G).sum(dim=0) / self.noise_variance - explained
        prior_logdet = torch.log(self.spectrum).broadcast_to((fields, *G.shape[1:-1])).sum(dim=0)
        logdet = channels * torch.log(self.noise_variance) + prior_logdet + cholesky_logdet(factor)
        return 0.5 * (quadratic + logdet)


@register("wiener", kind="solver")
class Wiener(Solver):
    """Closed-form MAP estimate under a Gaussian object prior and white Gaussian noise.

    Per frequency, ``f_hat = A^-1 H^H g / sigma^2`` with ``A = S^-1 + H^H H / sigma^2``. For one field and channel
    with :class:`~waveorder.nn.priors.Tikhonov` this is the Tikhonov inverse filter ``conj(H) / (|H|^2 + weight)``.
    """

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise unless the system is shift-invariant and the prior and noise are Gaussian."""
        check_linear_gaussian(model, verb="Wiener", fixes=SOLVER_FIXES)

    @override
    def run(self, problem: InverseProblem) -> SolveResult:
        estimate = LinearGaussian.of(problem.model, problem.operator).posterior(problem.data).mean()
        return SolveResult(estimate, problem.objective(estimate), 0, "direct")
