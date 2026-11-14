"""Scores: the outer problem's objective over the imaging system, lower is better."""

import math
from abc import abstractmethod
from typing import ClassVar, override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.functional import (
    abs2,
    cholesky_solve,
    half_spectrum_weights,
    hermitian_inverse,
    is_complex_matrix,
    matrix_vector,
    to_half_spectrum,
)
from waveorder.nn.model import ImagingModel
from waveorder.nn.optics import Detection
from waveorder.nn.parameters import repr_parts
from waveorder.nn.sampling import Sampling
from waveorder.nn.solvers.base import Solver
from waveorder.nn.solvers.wiener import SCORE_FIXES, LinearGaussian, check_linear_gaussian
from waveorder.nn.systems.base import ImagingSystem
from waveorder.nn.verbs import InverseProblem, Verb
from waveorder.registry import register


class Score(Verb[Tensor]):
    """A scalar objective of the data under a model, differentiable in the model's free quantities.

    ``score(data, model, sampling, *, operator=None) -> Tensor`` of shape ``()``. A new score implements
    :meth:`~waveorder.nn.verbs.Verb.run`, and :meth:`~waveorder.nn.verbs.Verb.check` if it cannot run on every model.

    Attributes
    ----------
    is_likelihood : ClassVar[bool]
        Whether the score is a negative log-likelihood of the free quantities (up to a constant); calibration then
        warns about quantities a likelihood cannot determine. False by default.
    """

    is_likelihood: ClassVar[bool] = False


@register("evidence", kind="score")
class Evidence(Score):
    """Negative log marginal likelihood ``-log p(g | H)`` with the object integrated out.

    Under a Gaussian object prior and white Gaussian noise the data's orthonormal Fourier coefficients are independent
    across frequencies with ``C x C`` covariance ``V_k = H_k S_k H_k^H + sigma^2 I``, so the evidence is exact:
    ``sum_k G_k^H V_k^-1 G_k / 2 + log det(2 pi V_k) / 2``. The zero frequency is left out, as in 3.x.

    A phase factor on ``H_k`` leaves ``V_k`` unchanged, so a quantity that only translates the point-spread function
    is not determined: ``detection.z_focus_offset`` of a 3D-object system. :func:`~waveorder.nn.calibrate` warns when
    one is free.
    """

    is_likelihood: ClassVar[bool] = True

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise unless the system is shift-invariant and the prior and noise are Gaussian."""
        check_linear_gaussian(model, verb="Evidence", fixes=SCORE_FIXES)

    @override
    def run(self, problem: InverseProblem) -> Tensor:
        return LinearGaussian.of(problem.model, problem.operator).neg_log_evidence(problem.data)


# --- Linear-Gaussian risk estimate of the closed-form reconstruction -----------------------------------------------


def _hat_statistics(gaussian: LinearGaussian, data: Tensor) -> tuple[Tensor, Tensor, int]:
    """Return the residual energy ``|g - S g|^2``, the trace of the hat matrix ``S`` and the data count ``n``."""
    channels, fields = gaussian.transfer.shape[:2]
    coefficients = to_half_spectrum(data.reshape(channels, *gaussian.shape), norm="ortho")
    posterior = gaussian.posterior(data)
    if fields == 1:  # a real scalar precision per frequency: divide, no factor
        precision = posterior.precision[0]
        mean = posterior.information / precision[..., None]
        covariance_diagonal = 1 / precision
    else:
        mean = cholesky_solve(posterior.factor(), posterior.information)
        covariance = hermitian_inverse(posterior.precision)
        real = covariance[..., 0] if is_complex_matrix(covariance) else covariance
        covariance_diagonal = torch.diagonal(real, dim1=0, dim2=1).movedim(-1, 0)
    residual = abs2(coefficients - matrix_vector(gaussian.transfer, mean)).sum(dim=0)
    trace = fields - (covariance_diagonal / gaussian.spectrum).sum(dim=0)
    weights = half_spectrum_weights(gaussian.shape[-1], device=residual.device, dtype=residual.dtype)
    count = channels * math.prod(gaussian.shape)
    return (residual * weights).sum(), (trace * weights).sum(), count


@register("gcv", kind="score")
class GCV(Score):
    """Generalized cross-validation of the closed-form reconstruction: ``n |g - S g|^2 / (n - tr S)^2``.

    ``S`` is the hat matrix of the Wiener estimate, ``g -> H f_hat``, and ``n`` the number of data samples. GCV
    estimates the prediction error without knowing the noise variance, so minimizing it over a free Tikhonov weight
    picks the regularization from the data. Data units squared; not a likelihood.
    """

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise unless the system is shift-invariant and the prior and noise are Gaussian."""
        check_linear_gaussian(model, verb="GCV", fixes=SCORE_FIXES)

    @override
    def run(self, problem: InverseProblem) -> Tensor:
        residual, trace, count = _hat_statistics(LinearGaussian.of(problem.model, problem.operator), problem.data)
        return count * residual / (count - trace).square()


# --- Sharpness of the reconstruction ---------------------------------------------------------------------------------


def transverse_cutoff(system: ImagingSystem) -> float:
    """Return the transverse cutoff ``2 NA / wavelength`` of a system's detection, in cycles per um.

    The aperture is the number the detection was built with (the nominal value of a free one), so the cutoff does not
    follow a calibration. For the brightfield phase systems the wavelength is the illumination's.

    Parameters
    ----------
    system : ImagingSystem
        A system with a :class:`~waveorder.nn.Detection` as ``system.detection``, as every built-in system has.

    Raises
    ------
    ParameterError
        With ``path="cutoff"``, if ``system`` has no such detection; then pass ``cutoff=`` to the score.
    """
    detection = getattr(system, "detection", None)
    if not isinstance(detection, Detection):
        raise ParameterError(
            f"transverse_cutoff reads 2 NA / wavelength from system.detection, a waveorder.nn.Detection; "
            f"{type(system).__name__} has none, so pass cutoff= to the score, in cycles per um",
            path="cutoff",
        )
    spec = detection.free_quantities().get("numerical_aperture")
    aperture = detection.fixed_quantities()["numerical_aperture"] if spec is None else spec.nominal
    return 2 * aperture / detection.wavelength


_UNFILLED = {
    "solver": "solver= (a Solver, e.g. Wiener())",
    "cutoff": "cutoff= (2 NA / wavelength in cycles per um; waveorder.nn.scores.transverse_cutoff(system) returns it)",
}
"""How the message for an unfilled sharpness score names each argument it was built without."""


class SharpnessScore(Score):
    """The negated mean sharpness of the reconstruction's 2D slices: lower is sharper.

    The solver reconstructs the object ``(F, Z, Y, X)``; :meth:`sharpness` scores each ``(Y, X)`` slice of the
    selected fields and the score is minus their mean. These are the 3.x optimization losses
    (``waveorder/optim/losses.py``); for 3.x thin phase optimization pass ``fields=("phase",)``. Not a likelihood.
    Built without a solver, the score is filled by a :mod:`waveorder.models` model (:meth:`fill`); used directly
    while :attr:`unfilled`, it raises and names what to pass.

    Parameters
    ----------
    solver : Solver, optional
        Reconstructs the object the score judges, differentiably in the model's free quantities (e.g. ``Wiener()``).
        ``None`` leaves it to :meth:`fill`.
    fields : tuple[str, ...] or None
        Names of the system's object fields to score, or None for every field.

    Raises
    ------
    ParameterError
        If ``fields`` is empty or repeats a name (path ``fields``).
    """

    solver: Solver | None

    def __init__(self, *, solver: Solver | None = None, fields: tuple[str, ...] | None = None) -> None:
        super().__init__()
        where = type(self).__name__
        if fields is not None and (not fields or len(set(fields)) != len(fields)):
            raise ParameterError(f"{where} fields must be distinct names, at least one, got {fields!r}", path="fields")
        self.register_module("solver", solver)  # a child slot even when empty, so a later solver is a child too
        self._solver_given = solver is not None
        self.fields = fields

    @property
    def unfilled(self) -> tuple[str, ...]:
        """The arguments the score still needs before it can run, of ``"solver"`` and ``"cutoff"``; empty when ready."""
        return self._missing()

    def _missing(self) -> tuple[str, ...]:
        return () if self.solver is not None else ("solver",)

    def fill(self, system: ImagingSystem, *, solver: Solver) -> None:
        """Give the score what it was built without: ``solver``, and for a midband score the cutoff of ``system``.

        Arguments given to the constructor are never replaced; ones a previous call filled are.

        Parameters
        ----------
        system : ImagingSystem
            The system the score will judge; a :class:`MidbandScore` built without ``cutoff`` takes
            :func:`transverse_cutoff` of it.
        solver : Solver
            The solver a score built without one reconstructs with (the model's own).

        Raises
        ------
        ParameterError
            If a cutoff is needed and ``system`` has no detection to read it from (path ``cutoff``).
        """
        del system  # read by MidbandScore
        if not self._solver_given:
            self.solver = solver

    def _ready_solver(self) -> Solver:
        """Return the solver, or raise naming the arguments the score was built without."""
        missing, solver = self.unfilled, self.solver
        if missing or solver is None:
            name = type(self).__name__
            raise ParameterError(
                f"{name} was built without {' and '.join(_UNFILLED[argument] for argument in missing)}. Pass "
                f"{'them' if len(missing) > 1 else 'it'}, or give the score to a waveorder.models model "
                f"(score={name}()), which fills what is missing from its own solver and system",
                path=missing[0] if missing else "solver",
            )
        return solver

    @override
    def extra_repr(self) -> str:
        """Return the quantities, then the scored ``fields`` when they are a selection."""
        return repr_parts(super().extra_repr(), "" if self.fields is None else f"fields={self.fields!r}")

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise as the solver does on ``model``, or if a selected field is not one of the system's.

        Raises :class:`~waveorder.errors.ParameterError` first if the score is still :attr:`unfilled`.
        """
        self._ready_solver().check(model)
        missing = [name for name in self.fields or () if name not in model.system.fields]
        if missing:
            raise IncompatibleModelError(
                f"{type(self).__name__} scores fields {missing} that {type(model.system).__name__} does not have; "
                f"its fields are {model.system.fields}"
            )

    @override
    def run(self, problem: InverseProblem) -> Tensor:
        estimate = self._ready_solver().run(problem).estimate
        if self.fields is not None:  # integer indexing per field: an index tensor would copy from the host
            names = problem.model.system.fields
            estimate = torch.stack([estimate[names.index(name)] for name in self.fields])
        return -self.sharpness(self.slices(estimate), problem.sampling).mean()

    def slices(self, estimate: Tensor) -> Tensor:
        """Return the ``(B, Y, X)`` slices of the scored fields ``(F', Z, Y, X)`` that :meth:`sharpness` judges: all."""
        return estimate.reshape(-1, *estimate.shape[-2:])

    @abstractmethod
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        """Return the sharpness of each slice ``(B, Y, X)``, shape ``(B,)``, higher for sharper slices."""


def _midband_mask(slices: Tensor, sampling: Sampling, *, cutoff: float, fractions: tuple[float, float]) -> Tensor:
    """Return the ``(Y, X)`` FFT-order mask of transverse frequencies strictly inside the midband annulus."""
    _, pixel_y, pixel_x = sampling.pixel_size
    ny, nx = slices.shape[-2:]
    fy = torch.fft.fftfreq(ny, d=pixel_y, dtype=slices.dtype, device=slices.device)
    fx = torch.fft.fftfreq(nx, d=pixel_x, dtype=slices.dtype, device=slices.device)
    radius = torch.hypot(fy[:, None], fx[None, :])
    return (radius > cutoff * fractions[0]) & (radius < cutoff * fractions[1])


class MidbandScore(SharpnessScore):
    """A sharpness score over the transverse midband annulus between two fractions of a cutoff frequency.

    Built without ``solver`` and ``cutoff``, a :mod:`waveorder.models` model fills both, the cutoff with
    :func:`transverse_cutoff` of its system.

    Parameters
    ----------
    solver : Solver, optional
        Reconstructs the object the score judges; see :class:`SharpnessScore`.
    fields : tuple[str, ...] or None
        Object fields to score, or None for every field; see :class:`SharpnessScore`.
    cutoff : float, optional
        Optical cutoff ``2 NA / wavelength`` in cycles per um, with the detection NA (3.x: the emission wavelength
        for fluorescence, the illumination wavelength for phase). Fixed: it does not follow a free numerical
        aperture. ``None`` leaves it to :meth:`~SharpnessScore.fill`.
    fractions : tuple[float, float]
        Inner and outer radius of the annulus as fractions of ``cutoff``, ``0 <= inner < outer``.

    Raises
    ------
    ParameterError
        If ``cutoff`` is not ``None`` or positive and finite (path ``cutoff``), or ``fractions`` is not two
        increasing nonnegative finite numbers (path ``fractions``). An annulus holding no sample gives a NaN score.
    """

    def __init__(
        self,
        *,
        solver: Solver | None = None,
        cutoff: float | None = None,
        fractions: tuple[float, float] = (0.125, 0.25),
        fields: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(solver=solver, fields=fields)
        where = type(self).__name__
        self._cutoff: float | None = None
        self._cutoff_given = cutoff is not None
        if cutoff is not None:
            self.cutoff = cutoff
        if len(fractions) != 2:
            raise ParameterError(f"{where} fractions must be two numbers (inner, outer), got {fractions!r}",
                                 path="fractions")  # fmt: skip
        inner, outer = (float(value) for value in fractions)
        if not (math.isfinite(outer) and 0 <= inner < outer):
            raise ParameterError(f"{where} fractions must satisfy 0 <= inner < outer, got {fractions!r}",
                                 path="fractions")  # fmt: skip
        self.fractions = (inner, outer)

    @property
    def cutoff(self) -> float:
        """The cutoff frequency in cycles per um; assigning a number sets it, checked as at construction.

        Raises
        ------
        ParameterError
            While the score has none (path ``cutoff``): it was built without one and not filled since.
        """
        if self._cutoff is None:
            raise ParameterError(
                f"{type(self).__name__} was built without {_UNFILLED['cutoff']}. Pass it, or give the score to a "
                "waveorder.models model, which fills it from its system",
                path="cutoff",
            )
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        where = type(self).__name__
        if not (math.isfinite(value) and value > 0):
            raise ParameterError(f"{where} cutoff must be positive and finite, got {value}", path="cutoff")
        self._cutoff = float(value)
        self._cutoff_given = True

    @override
    def _missing(self) -> tuple[str, ...]:
        return (*super()._missing(), *(() if self._cutoff is not None else ("cutoff",)))

    @override
    def fill(self, system: ImagingSystem, *, solver: Solver) -> None:
        super().fill(system, solver=solver)
        if not self._cutoff_given:
            self._cutoff = transverse_cutoff(system)

    @override
    def extra_repr(self) -> str:
        cutoff = "None" if self._cutoff is None else f"{self._cutoff:g}"
        return repr_parts(f"cutoff={cutoff}", f"fractions={self.fractions!r}", super().extra_repr())

    def _amplitude_and_mask(self, slices: Tensor, sampling: Sampling) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``|FFT|`` of each slice, the annulus mask and its sample count (NaN-producing when 0)."""
        mask = _midband_mask(slices, sampling, cutoff=self.cutoff, fractions=self.fractions)
        amplitude = torch.fft.fft2(slices).abs()
        return amplitude, mask, mask.sum().to(slices.dtype)


@register("midband_power_score", kind="score")
class MidbandPowerScore(MidbandScore):
    """Mean Fourier amplitude ``|F|`` in the midband annulus (3.x ``midband_power``, the 3.x default loss).

    Divided by the annulus's sample count, so the score does not grow with the tile. Parameters: :class:`MidbandScore`.
    """

    @override
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        amplitude, mask, count = self._amplitude_and_mask(slices, sampling)
        return (amplitude * mask).sum(dim=(-2, -1)) / count


@register("centre_slice_midband_score", kind="score")
class CentreSliceMidbandScore(MidbandScore):
    """Mean of ``|F|^2`` times the midband mask over all pixels of the centre z slice (3.x recovery ``midband``).

    The loss of 3.x ``shift_variant_recovery`` (``loss: midband``): the squared Fourier amplitude of plane ``Z // 2``,
    averaged over every pixel rather than over the annulus. 3.x used ``fractions=(0.2, 0.4)``. Parameters:
    :class:`MidbandScore`.
    """

    @override
    def slices(self, estimate: Tensor) -> Tensor:
        return estimate[:, estimate.shape[1] // 2]

    @override
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        mask = _midband_mask(slices, sampling, cutoff=self.cutoff, fractions=self.fractions)
        return (torch.fft.fft2(slices).abs().square() * mask).mean(dim=(-2, -1))


@register("spectral_flatness_score", kind="score")
class SpectralFlatnessScore(MidbandScore):
    """Geometric over arithmetic mean of the Fourier amplitude in the midband annulus (3.x ``spectral_flatness``).

    Rewards power spread evenly across the annulus; between 0 and 1, up to the ``1e-12`` 3.x adds to the arithmetic
    mean. Parameters: :class:`MidbandScore`.
    """

    @override
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        amplitude, mask, count = self._amplitude_and_mask(slices, sampling)
        log_mean = torch.where(mask, amplitude, 1).log().sum(dim=(-2, -1)) / count  # log 1 = 0 off the annulus
        arithmetic_mean = (amplitude * mask).sum(dim=(-2, -1)) / count
        return log_mean.exp() / (arithmetic_mean + 1e-12)


@register("total_variation_score", kind="score")
class TotalVariationScore(SharpnessScore):
    """Mean absolute difference of neighboring pixels along y and along x, averaged (3.x ``total_variation``).

    In the object's units; independent of the tile size.
    """

    @override
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        dy = slices.diff(dim=-2).abs().mean(dim=(-2, -1))
        dx = slices.diff(dim=-1).abs().mean(dim=(-2, -1))
        return (dy + dx) / 2


@register("laplacian_variance_score", kind="score")
class LaplacianVarianceScore(SharpnessScore):
    """Variance of the 5-point Laplacian of each slice, zero padded (3.x ``laplacian_variance``).

    The classic autofocus metric; the unbiased variance over the slice, in object units squared per pixel^4.
    """

    @override
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        # The 3.x kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]] as shifts: a kernel tensor would be copied from the host.
        padded = torch.nn.functional.pad(slices, (1, 1, 1, 1))
        neighbors = padded[..., :-2, 1:-1] + padded[..., 2:, 1:-1] + padded[..., 1:-1, :-2] + padded[..., 1:-1, 2:]
        return (neighbors - 4 * slices).var(dim=(-2, -1))


@register("normalized_variance_score", kind="score")
class NormalizedVarianceScore(SharpnessScore):
    """Unbiased variance over squared mean of each slice, ``var / (mean^2 + 1e-12)`` (3.x ``normalized_variance``).

    Contrast relative to brightness; dimensionless. Near-zero-mean slices (phase) make it large and noisy.
    """

    @override
    def sharpness(self, slices: Tensor, sampling: Sampling) -> Tensor:
        return slices.var(dim=(-2, -1)) / (slices.mean(dim=(-2, -1)).square() + 1e-12)
