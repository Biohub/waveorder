"""Richardson-Lucy for Poisson noise without a prior, with gradient consensus and unmatched back projectors.

Gradient consensus follows Andrew G. York (doi.org/10.5281/zenodo.10278918). The unmatched back projectors
(3.x ``backprojector.py``; Guo et al., Nat. Biotechnol. 2020, Supplementary Note 2) replace ``conj(OTF)`` with a
``B`` that flattens ``|OTF * B|``, so 1 to 5 iterations do the work of many: ``"wiener"``, ``"butterworth"``,
``"wiener_butterworth"``, and ``"gaussian"``, a unit-sum Gaussian of the PSF's FWHM. Their defaults are Eq. 28's
``alpha`` and ``beta``, a Butterworth of order 8 and a cutoff at the PSF FWHM, measured on the host once per solve.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Literal, override

import torch
from torch import Tensor

from waveorder.errors import ConfigurationWarning, IncompatibleModelError, ParameterError
from waveorder.nn.functional import abs2, from_half_spectrum, is_complex_matrix, matrix_vector, to_half_spectrum
from waveorder.nn.model import ImagingModel
from waveorder.nn.parameters import Quantity, can_read_back, read_flag, repr_parts, value_range
from waveorder.nn.sampling import Data, Object
from waveorder.nn.solvers.base import IterativeSolver, Loop
from waveorder.nn.verbs import InverseProblem
from waveorder.registry import register

type BackProjector = Literal["matched", "gaussian", "butterworth", "wiener", "wiener_butterworth"]
"""The back projectors of 3.x ``backprojector.py``; ``"matched"`` is the adjoint."""

BACK_PROJECTORS: tuple[BackProjector, ...] = ("matched", "gaussian", "butterworth", "wiener", "wiener_butterworth")

_EPS = 1e-12
_ORDER = 8


def back_projector(
    transfer: Tensor, shape: tuple[int, int, int], *, kind: BackProjector, alpha: float | None, beta: float | None
) -> Tensor:
    """Return the back projector ``B`` that replaces ``conj(OTF)`` for one field and channel, as real storage.

    Parameters
    ----------
    transfer : Tensor
        The forward transfer of one channel and field, ``(1, 1, Z, Y, X // 2 + 1, 2)`` or real
        ``(1, 1, Z, Y, X // 2 + 1)``.
    shape : tuple of int
        The volume ``(Z, Y, X)``.
    kind : BackProjector
        Which back projector; ``"matched"`` returns ``conj(transfer)``.
    alpha : float, optional
        Wiener regularization added to ``|OTF|^2`` (unit-peak OTF). ``None`` uses the squared mean cutoff gain.
    beta : float, optional
        Butterworth gain at the resolution cutoff. ``None`` uses the mean cutoff gain.

    Returns
    -------
    Tensor
        ``B`` as half-spectrum storage ``(Z, Y, X // 2 + 1, 2)``: the back-projection is ``IRFFT(RFFT(g) * B)``.

    Raises
    ------
    ParameterError
        If ``kind`` is unknown, ``alpha`` or ``beta`` is not positive, or ``beta`` is too large for ``kind``.
    IncompatibleModelError
        If the PSF is undersampled, so its FWHM cannot be measured.
    """
    if kind not in BACK_PROJECTORS:
        raise ParameterError(f"back_projector must be one of {BACK_PROJECTORS}, got {kind!r}", path="back_projector")
    storage = transfer[0, 0] if is_complex_matrix(transfer) else _with_zero_imaginary(transfer[0, 0])
    conjugate = torch.stack((storage[..., 0], -storage[..., 1]), dim=-1)
    if kind == "matched":
        return conjugate
    if kind == "gaussian":
        kernel = _gaussian_kernel(shape, _psf_fwhm(storage, shape), device=storage.device, dtype=storage.dtype)
        return to_half_spectrum(kernel)
    return _filtered(storage, shape, kind=kind, alpha=alpha, beta=beta)


def _filtered(
    storage: Tensor, shape: tuple[int, int, int], *, kind: BackProjector, alpha: float | None, beta: float | None
) -> Tensor:
    """Return the Wiener and Butterworth back projectors (3.x ``calculate_back_projector`` past ``"gaussian"``)."""
    cutoffs = tuple(size / width for size, width in zip(shape, _psf_fwhm(storage, shape), strict=True))
    normalized = storage / abs2(storage).sqrt().max().clamp_min(_EPS)
    magnitude = _full_magnitude(normalized, shape)
    if alpha is None or beta is None:
        gain = _matched_cutoff_gain(magnitude, cutoffs)
        alpha = gain**2 if alpha is None else alpha
        beta = gain if beta is None else beta
    if not alpha > 0:
        raise ParameterError(f"bp_alpha must be positive, got {alpha}", path="bp_alpha")
    if not beta > 0:
        raise ParameterError(f"bp_beta must be positive, got {beta}", path="bp_beta")
    placed = {"device": storage.device, "dtype": storage.dtype}
    if kind == "butterworth":
        if beta > 1.0:
            raise ParameterError(f"butterworth needs bp_beta <= 1 (a gain at the cutoff), got {beta}", path="bp_beta")
        mask = _butterworth(shape, cutoffs, 1.0 / beta**2 - 1.0, **placed)
        return torch.stack((mask, torch.zeros_like(mask)), dim=-1)
    normalized_conjugate = torch.stack((normalized[..., 0], -normalized[..., 1]), dim=-1)
    wiener = normalized_conjugate / (abs2(normalized) + alpha)[..., None]
    if kind == "wiener":
        return wiener
    wiener_gain = _wiener_cutoff_gain(magnitude / (magnitude.square() + alpha), cutoffs)
    effective_beta = beta * math.sqrt(wiener_gain)  # 3.x's "reference" convention, its default
    if effective_beta > wiener_gain:
        raise ParameterError(
            f"wiener_butterworth needs an effective bp_beta <= the Wiener cutoff gain ({wiener_gain:.4g}), got "
            f"{effective_beta:.4g}; lower bp_beta or raise bp_alpha",
            path="bp_beta",
        )
    return wiener * _butterworth(shape, cutoffs, (wiener_gain / effective_beta) ** 2 - 1.0, **placed)[..., None]


def _with_zero_imaginary(transfer: Tensor) -> Tensor:
    return torch.stack((transfer, torch.zeros_like(transfer)), dim=-1)


def _full_magnitude(storage: Tensor, shape: tuple[int, int, int]) -> Tensor:
    """``|OTF|`` on the full spectrum ``(Z, Y, X)``, FFT order, rebuilt from the half spectrum."""
    psf = from_half_spectrum(storage, shape)
    return torch.view_as_real(torch.fft.fftn(psf, dim=(-3, -2, -1))).square().sum(-1).sqrt()


def _frequency_index(size: int, *, half: bool, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Signed FFT index ``[0, 1, ..., -1]`` (``half``: the ``rfft`` indices ``[0, ..., size // 2]``)."""
    if half:
        return torch.fft.rfftfreq(size, device=device, dtype=dtype) * size
    return torch.fft.fftfreq(size, device=device, dtype=dtype) * size


def _radius_squared(
    shape: tuple[int, int, int], scales: tuple[float, ...], *, device: torch.device, dtype: torch.dtype, half: bool
) -> Tensor:
    """Squared ellipsoidal radius ``sum_a (k_a / scale_a)^2`` on the (half) spectrum grid."""
    axes = []
    for axis, (size, scale) in enumerate(zip(shape, scales, strict=True)):
        index = _frequency_index(size, half=half and axis == 2, device=device, dtype=dtype) / scale
        axes.append(index.square().reshape([-1 if a == axis else 1 for a in range(3)]))
    return axes[0] + axes[1] + axes[2]


def _butterworth(
    shape: tuple[int, int, int],
    cutoffs: tuple[float, ...],
    epsilon_squared: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """``1 / sqrt(1 + eps^2 r^(2 order))`` over the ellipsoidal cutoff, on the half spectrum, computed in float64."""
    radius = _radius_squared(shape, cutoffs, device=device, dtype=torch.float64, half=True)
    return radius.pow(_ORDER).mul(epsilon_squared).add(1.0).sqrt().reciprocal().to(dtype)


def _gaussian_kernel(
    shape: tuple[int, int, int], fwhm: tuple[float, ...], *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return a unit-sum Gaussian kernel ``(Z, Y, X)``, FFT order, whose FWHM per axis is ``fwhm`` voxels."""
    sigmas = tuple(width / (2.0 * math.sqrt(2.0 * math.log(2.0))) for width in fwhm)
    kernel = torch.exp(-0.5 * _radius_squared(shape, sigmas, device=device, dtype=torch.float64, half=False))
    return (kernel / kernel.sum().clamp_min(_EPS)).to(dtype)


def _psf_fwhm(storage: Tensor, shape: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return the PSF's full width at half maximum per axis, in voxels: three read-backs."""
    psf = torch.fft.fftshift(from_half_spectrum(storage, shape), dim=(-3, -2, -1))
    peak = torch.unravel_index(torch.argmax(psf), psf.shape)
    z, y, x = (int(index) for index in peak)
    return _fwhm(psf[:, y, x], "z"), _fwhm(psf[z, :, x], "y"), _fwhm(psf[z, y, :], "x")


def _fwhm(profile: Tensor, axis: str) -> float:
    """Full width at half maximum of a peaked 1D profile, in voxels, by linear interpolation (as in 3.x)."""
    peak_index = int(torch.argmax(profile))
    peak = float(profile[peak_index])
    if not peak > 0:
        raise IncompatibleModelError(f"cannot measure the PSF FWHM along {axis}: the profile peak is not positive")
    normalized = profile.cpu().to(torch.float64) / peak
    left = torch.nonzero(normalized[: peak_index + 1] < 0.5).flatten()
    right = torch.nonzero(normalized[peak_index:] < 0.5).flatten()
    if left.numel() == 0 or right.numel() == 0:
        raise IncompatibleModelError(
            f"cannot measure the PSF FWHM along {axis}: the profile never falls below half maximum, so the PSF is "
            "undersampled along this axis; reconstruct on a finer grid or use back_projector='matched'"
        )

    def crossing(low: int, high: int) -> float:
        low_value, high_value = float(normalized[low]), float(normalized[high])
        return low + (0.5 - low_value) / (high_value - low_value) * (high - low)

    left_index, right_index = int(left[-1]), peak_index + int(right[0])
    return crossing(right_index - 1, right_index) - crossing(left_index, left_index + 1)


def _gain_at(profile: Tensor, cutoff: float) -> float:
    """Mean of a centered (fftshifted) 1D profile at the two cutoff frequencies."""
    size = profile.shape[0]
    center = size // 2
    low = max(round(center - cutoff), 0)
    high = min(round(center + cutoff), size - 1)
    return float((profile[low] + profile[high]) / 2)


def _matched_cutoff_gain(magnitude: Tensor, cutoffs: tuple[float, ...]) -> float:
    """Eq. 28: per axis, the maximum projection of ``|OTF|`` sampled at the two cutoffs, averaged over the axes."""
    shifted = torch.fft.fftshift(magnitude, dim=(-3, -2, -1))
    gains = [
        _gain_at(torch.amax(shifted, dim=tuple(a for a in range(3) if a != axis)), cutoff)
        for axis, cutoff in enumerate(cutoffs)
    ]
    return sum(gains) / 3.0


def _wiener_cutoff_gain(wiener_magnitude: Tensor, cutoffs: tuple[float, ...]) -> float:
    """Return the Wiener term's gain at the lateral (X) cutoff, read on the central Z slice (as 3.x does)."""
    shifted = torch.fft.fftshift(wiener_magnitude, dim=(-3, -2, -1))
    return _gain_at(torch.amax(shifted[shifted.shape[0] // 2], dim=0), cutoffs[2])


_DEFAULT_LOOP = Loop(25)

_UNMATCHED_ITERATIONS = 5
"""Guo et al.: an unmatched back projector reaches its result in 1 to 5 iterations and adds artifacts beyond."""


@dataclass(frozen=True, slots=True)
class RichardsonLucyState:
    """Iterate ``x``, signal ``s``, sensitivity ``B 1``, back projector ``B`` (``None``: adjoint), RLGC all frozen."""

    x: Tensor
    signal: Tensor
    sensitivity: Tensor
    back_projector: Tensor | None
    frozen: bool = False


@register("richardson_lucy", kind="solver")
class RichardsonLucy(IterativeSolver[RichardsonLucyState]):
    """Maximize the Poisson likelihood by ``x <- x * B(s / (H x + b)) / B 1``, with ``B = H*`` by default.

    Each step keeps ``x >= 0`` and never increases the negative log likelihood (Shepp and Vardi, IEEE TMI 1982).
    ``s`` is the noise's :meth:`~waveorder.nn.noise.Noise.photon_counts` and the denominator adds its
    :meth:`~waveorder.nn.noise.Noise.photon_background`. It needs a system that declares
    :attr:`~waveorder.nn.ImagingSystem.preserves_nonnegativity`, Poisson noise and a flat prior.

    Parameters
    ----------
    loop : Loop
        Iterations and stopping (default ``Loop(25)``).
    background : float or Free
        Expected signal ``b >= 0`` added to ``H x``, in data units above the noise offset.
    back_projector : {"matched", "gaussian", "butterworth", "wiener", "wiener_butterworth"}
        The back-projection filter ``B``; ``"matched"`` is ``H*``. An unmatched one is not an adjoint, so the
        likelihood need not rise; it needs one channel and one field on a Fourier-diagonal operator, and more than 5
        iterations warn.
    bp_alpha, bp_beta : float, optional
        Wiener regularization and Butterworth cutoff gain of the unmatched back projectors (unit-peak OTF);
        ``None`` uses 3.x's defaults.
    gradient_consensus : bool
        Use the RLGC update (3.x ``rlgc``): split the photons into two binomial halves each iteration, from a
        generator seeded with 0 per solve, and freeze voxels where the halves disagree. The solve ends as
        ``"converged"`` once every voxel is frozen, as 3.x's did. Needs ``"matched"``.
    start : {"mean", "ones"}
        The first estimate: the uniform object ``mean(s) - b`` (default), or 1, 3.x's start.

    Raises
    ------
    ParameterError
        If ``background`` can be negative, ``back_projector`` or ``start`` is unknown, ``bp_alpha`` or ``bp_beta``
        is not positive, or ``gradient_consensus`` is combined with an unmatched back projector.
    """

    background: Tensor

    def __init__(  # noqa: PLR0913 (the 3.x rl block: background, back projector, its two knobs, the RLGC switch)
        self,
        *,
        loop: Loop = _DEFAULT_LOOP,
        background: Quantity = 0.0,
        back_projector: BackProjector = "matched",
        bp_alpha: float | None = None,
        bp_beta: float | None = None,
        gradient_consensus: bool = False,
        start: Literal["mean", "ones"] = "mean",
    ) -> None:
        super().__init__(loop=loop)
        reach = value_range(background)
        if reach is not None and reach[0] < 0:
            raise ParameterError(f"RichardsonLucy.background must be nonnegative, got {background}", path="background")
        if back_projector not in BACK_PROJECTORS:
            raise ParameterError(
                f"RichardsonLucy back_projector must be one of {BACK_PROJECTORS}, got {back_projector!r}",
                path="back_projector",
            )
        for name, value in (("bp_alpha", bp_alpha), ("bp_beta", bp_beta)):
            if value is not None and not value > 0:
                raise ParameterError(f"RichardsonLucy {name} must be positive, got {value}", path=name)

        if start not in ("mean", "ones"):
            raise ParameterError(f"RichardsonLucy start must be 'mean' or 'ones', got {start!r}", path="start")
        if gradient_consensus and back_projector != "matched":
            # RLGC reads the sign of B(H(.)): only an adjoint makes it mean "the two halves agree" (3.x refuses too).
            raise ParameterError(
                f"RichardsonLucy gradient_consensus needs back_projector='matched', got {back_projector!r}",
                path="gradient_consensus",
            )
        if back_projector != "matched" and loop.iterations > _UNMATCHED_ITERATIONS:
            warnings.warn(
                f"RichardsonLucy with back_projector={back_projector!r} and {loop.iterations} iterations: unmatched "
                f"back projectors reach their result in 1 to {_UNMATCHED_ITERATIONS} iterations and add artifacts "
                "beyond that; consider Loop(1)",
                ConfigurationWarning,
                stacklevel=2,
            )
        self.register_quantity("background", background)
        self.back_projector = back_projector
        self.bp_alpha = bp_alpha
        self.bp_beta = bp_beta
        self.gradient_consensus = gradient_consensus
        self.start_from = start
        self._generator: torch.Generator | None = None

    @override
    def check(self, model: ImagingModel) -> None:
        """Raise unless the system preserves nonnegativity, the noise declares photon counts and the prior is flat."""
        if not model.system.preserves_nonnegativity:
            system = type(model.system).__name__
            raise IncompatibleModelError(
                f"RichardsonLucy needs a nonnegative point-spread function; {system} does not declare "
                f"preserves_nonnegativity. If the point-spread function of {system} is nonnegative, set "
                "preserves_nonnegativity = True on the class; if it is not, no built-in solver takes photon-counting "
                "noise on it: pass noise=Gaussian(sigma) and use Wiener or FISTA"
            )
        if not model.noise.has_photon_counts:
            raise IncompatibleModelError(
                f"RichardsonLucy needs Poisson noise; {type(model.noise).__name__} declares has_photon_counts = False "
                "and reports no photon_counts; pass noise=Poisson(), or use Wiener or FISTA"
            )
        if not model.object_prior.is_flat:
            raise IncompatibleModelError(
                f"RichardsonLucy maximizes the likelihood alone; got the prior {type(model.object_prior).__name__}: "
                "pass object_prior=NoPrior()"
            )

    @override
    def extra_repr(self) -> str:
        settings = [
            f"back_projector={self.back_projector!r}" if self.back_projector != "matched" else "",
            "gradient_consensus=True" if self.gradient_consensus else "",
            f"start={self.start_from!r}" if self.start_from != "mean" else "",
        ]
        return repr_parts(super().extra_repr(), *settings)

    @override
    def start(self, problem: InverseProblem) -> RichardsonLucyState:
        signal = problem.model.noise.photon_counts(problem.data)
        tiny = torch.finfo(signal.dtype).tiny
        eager = can_read_back(problem.data)
        if (self.gradient_consensus or self.back_projector != "matched") and not eager:
            raise IncompatibleModelError(
                "RichardsonLucy with gradient_consensus or an unmatched back_projector draws random numbers or "
                "measures the PSF on the host, which vmap and grad cannot; run it eagerly"
            )
        kernel = None if self.back_projector == "matched" else self._back_projector(problem)
        sensitivity = self._back_project(torch.ones_like(signal), problem, kernel).clamp_min(tiny)
        background = self.background + problem.model.noise.photon_background()
        if self.start_from == "ones":
            x = torch.ones(problem.operator.object_shape, dtype=signal.dtype, device=signal.device)
        else:
            level = (signal.mean() - background).clamp_min(tiny)
            x = level.expand(problem.operator.object_shape).clone()
        if self.gradient_consensus:
            self._generator = torch.Generator(device=signal.device).manual_seed(0)
        return RichardsonLucyState(x=x, signal=signal, sensitivity=sensitivity, back_projector=kernel)

    @override
    def step(self, state: RichardsonLucyState, problem: InverseProblem) -> RichardsonLucyState:
        tiny = torch.finfo(state.x.dtype).tiny
        rates = (problem.operator.apply(state.x) + self.background + problem.model.noise.photon_background()).clamp_min(
            tiny
        )
        back = self._back_project(state.signal / rates, problem, state.back_projector)
        if self.gradient_consensus:
            x, step_size = self._consensus_step(state, problem, rates=rates, back=back)
            frozen = not read_flag(step_size.ne(0).any())  # RLGC runs eagerly, so this read is allowed
        else:
            x = state.x * back / state.sensitivity
            frozen = False
        return RichardsonLucyState(
            x=x.clamp_min(0.0),
            signal=state.signal,
            sensitivity=state.sensitivity,
            back_projector=state.back_projector,
            frozen=frozen,
        )

    @override
    def stopped(self, state: RichardsonLucyState) -> bool:
        """Whether RLGC froze every voxel, where 3.x stopped: every later step would draw new halves and unfreeze."""
        return state.frozen

    def _consensus_step(
        self, state: RichardsonLucyState, problem: InverseProblem, *, rates: Tensor, back: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the RLGC update and its step size (3.x ``rlgc.scaled_gradient_step``): zero where halves disagree."""
        operator = problem.operator
        gradient = back - operator.adjoint(torch.ones_like(rates))
        heads = torch.binomial(
            state.signal.clamp_min(0.0).round(), torch.full_like(state.signal, 0.5), generator=self._generator
        )
        heads_gradient = operator.adjoint(heads / rates - 0.5)
        tails_gradient = gradient - heads_gradient
        agreement = operator.adjoint(operator.apply(heads_gradient * tails_gradient))
        step_size = torch.where(agreement <= 0, torch.zeros_like(state.x), state.x / state.sensitivity)
        return state.x + gradient * step_size, step_size

    def _back_projector(self, problem: InverseProblem) -> Tensor:
        fourier = problem.operator.as_fourier()
        if fourier is None or fourier.transfer.shape[:2] != (1, 1):
            raise IncompatibleModelError(
                f"RichardsonLucy back_projector={self.back_projector!r} needs a Fourier-diagonal operator with one "
                f"channel and one field; got {type(problem.operator).__name__}"
            )
        return back_projector(
            fourier.transfer, fourier.sampling.shape, kind=self.back_projector, alpha=self.bp_alpha, beta=self.bp_beta
        )

    def _back_project(self, data: Data, problem: InverseProblem, kernel: Tensor | None) -> Object:
        """``B data``: ``H* data`` for the matched back projector, else ``IRFFT(RFFT(data) * B)``."""
        if kernel is None:
            return problem.operator.adjoint(data)
        fourier = problem.operator.as_fourier()
        if fourier is None:  # _back_projector refused this operator before any step
            raise IncompatibleModelError(
                f"RichardsonLucy cannot back-project through {type(problem.operator).__name__}"
            )
        volume = data.reshape(fourier.data_shape)
        spectrum = matrix_vector(kernel[None, None], to_half_spectrum(volume))
        return from_half_spectrum(spectrum, fourier.sampling.shape)

    @override
    def estimate(self, state: RichardsonLucyState) -> Object:
        return state.x

    @override
    def objective(self, state: RichardsonLucyState, problem: InverseProblem) -> Tensor:
        return problem.model.noise.nll(problem.data, problem.operator.apply(state.x) + self.background)
