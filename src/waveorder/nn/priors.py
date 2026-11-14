"""Object priors p(f): the penalty ``P(f) = -log p(f)`` up to a constant, its proximal map, and Gaussian spectra.

Every prior is scaled by the noise model's :meth:`~waveorder.nn.noise.Noise.scale` ``s``: ``P(f) = weight * R(f) / s``.
Under Gaussian noise ``s = sigma^2``, so ``weight`` is the 3.x regularization ``lambda``. The MAP objective is
``D(g; H f) + P(f)``.
"""

import math
import warnings
from abc import ABC, abstractmethod
from typing import override

import torch
from torch import Tensor

from waveorder.errors import CalibrationWarning, IncompatibleModelError, ParameterError
from waveorder.nn.functional import (
    forward_differences,
    forward_differences_adjoint,
    half_spectrum_frequencies,
    irfftn,
)
from waveorder.nn.parameters import CalibratableModule, Quantity, repr_parts
from waveorder.nn.sampling import Sampling
from waveorder.registry import register


class ObjectPrior(CalibratableModule, ABC):
    """Prior on the object ``f``, shape ``(F, Z, Y, X)``.

    Parameters
    ----------
    nonnegative : bool
        Restrict the support to ``f >= 0``: the penalty is infinite elsewhere and :meth:`prox` projects onto it.
    """

    def __init__(self, *, nonnegative: bool = False) -> None:
        super().__init__()

        self.nonnegative = nonnegative

    @override
    def extra_repr(self) -> str:
        """Return the quantities, then ``nonnegative=True`` when the support is constrained."""
        return repr_parts(super().extra_repr(), "nonnegative=True" if self.nonnegative else "")

    @abstractmethod
    def penalty(self, obj: Tensor, *, scale: Tensor) -> Tensor:
        """Return ``P(f)``, excluding the support constraint, as a 0-d tensor."""

    @abstractmethod
    def prox(self, obj: Tensor, step: Tensor, *, scale: Tensor) -> Tensor:
        """Return ``argmin_x step * P(x) + ||x - f||^2 / 2`` over the support."""

    def project(self, obj: Tensor) -> Tensor:
        """Return the nearest point of the support: ``f`` itself, or ``max(f, 0)`` if :attr:`nonnegative`."""
        return obj.clamp_min(0.0) if self.nonnegative else obj

    @property
    def is_flat(self) -> bool:
        """Whether the penalty is zero everywhere, so only the support constrains ``f``."""
        return False


@register("none", kind="object_prior")
class NoPrior(ObjectPrior):
    """The flat prior: no regularization, only the optional support constraint."""

    @property
    @override
    def is_flat(self) -> bool:
        return True

    @override
    def penalty(self, obj: Tensor, *, scale: Tensor) -> Tensor:
        del scale
        return obj.new_zeros(())

    @override
    def prox(self, obj: Tensor, step: Tensor, *, scale: Tensor) -> Tensor:
        del step, scale
        return self.project(obj)


class GaussianObjectPrior(ObjectPrior, ABC):
    """A stationary Gaussian prior, described by its spectral variance ``S_k``.

    With :attr:`nonnegative` it is a truncated Gaussian and no longer admits the closed-form verbs.
    """

    @abstractmethod
    def spectrum(self, sampling: Sampling, *, scale: Tensor) -> Tensor:
        """Return ``S_k`` for orthonormal coefficients, broadcastable to the half spectrum ``(F, Z, Y, X // 2 + 1)``.

        ``S_k`` is in voxel units (frequencies in cycles per voxel) and must not depend on ``sampling.pixel_size``,
        since :meth:`~ObjectPrior.penalty` never sees the sampling; only ``sampling.shape`` may enter.
        """


@register("tikhonov", kind="object_prior")
class Tikhonov(GaussianObjectPrior):
    """White Gaussian prior ``P(f) = weight * ||f||^2 / (2 s)``: spectral variance ``S_k = s / weight``.

    With Gaussian noise the Wiener filter is ``conj(H) / (|H|^2 + weight)``, the 3.x Tikhonov inverse filter.

    Parameters
    ----------
    weight : float or Free
        Regularization weight: the noise scale ``s`` over the object's spectral variance. Positive; a free weight
        needs bounds with a positive lower limit.
    nonnegative : bool
        Restrict the support to ``f >= 0``; the closed-form verbs then refuse the prior.

    Raises
    ------
    ParameterError
        If ``weight`` is not positive; ``path`` is ``weight``.
    """

    weight: Tensor

    def __init__(self, weight: Quantity = 1e-3, *, nonnegative: bool = False) -> None:
        super().__init__(nonnegative=nonnegative)
        self.register_quantity("weight", weight, positive=True)

    @override
    def penalty(self, obj: Tensor, *, scale: Tensor) -> Tensor:
        return 0.5 * self.weight / scale * (obj**2).sum()

    @override
    def prox(self, obj: Tensor, step: Tensor, *, scale: Tensor) -> Tensor:
        return self.project(obj / (1 + step * self.weight / scale))

    @override
    def spectrum(self, sampling: Sampling, *, scale: Tensor) -> Tensor:
        del sampling
        return scale / self.weight


@register("tv", kind="object_prior")
class TotalVariation(ObjectPrior):
    """Isotropic total variation ``P(f) = weight * sum_voxels ||grad f||_2 / s``, per field, in voxel units.

    Gradients are periodic forward differences, matching the circular convolution of the imaging model. :meth:`prox`
    runs a fixed number of iterations of Beck and Teboulle's fast gradient projection (IEEE TIP 2009) on the dual.

    Parameters
    ----------
    weight : float or Free
        Regularization weight.
    nonnegative : bool
        Restrict the support to ``f >= 0``.
    inner_iterations : int
        Dual iterations per :meth:`prox` call.
    """

    weight: Tensor

    def __init__(self, weight: Quantity = 1e-3, *, nonnegative: bool = False, inner_iterations: int = 20) -> None:
        super().__init__(nonnegative=nonnegative)
        if inner_iterations < 1:
            raise ParameterError(f"TotalVariation inner_iterations must be at least 1, got {inner_iterations}")
        self.register_quantity("weight", weight, positive=True)
        self.inner_iterations = inner_iterations

    @override
    def extra_repr(self) -> str:
        return repr_parts(super().extra_repr(), f"inner_iterations={self.inner_iterations}")

    @override
    def penalty(self, obj: Tensor, *, scale: Tensor) -> Tensor:
        return self.weight / scale * torch.linalg.vector_norm(forward_differences(obj), dim=0).sum()

    @override
    def prox(self, obj: Tensor, step: Tensor, *, scale: Tensor) -> Tensor:
        # A zero step makes the prox the projection; clamping keeps the dual step finite on the way there.
        strength = (step * self.weight / scale).clamp_min(torch.finfo(obj.dtype).tiny)
        # ||grad||^2 <= 4 per axis, so 12 in 3D bounds the dual gradient's Lipschitz constant, times strength^2.
        dual_step = 1 / (12 * strength)
        dual = torch.zeros_like(forward_differences(obj))
        momentum, t = dual, 1.0
        for _ in range(self.inner_iterations):
            primal = self.project(obj - strength * forward_differences_adjoint(momentum))
            ascended = momentum + dual_step * forward_differences(primal)
            projected = ascended / torch.linalg.vector_norm(ascended, dim=0, keepdim=True).clamp_min(1.0)
            t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
            momentum = projected + ((t - 1) / t_next) * (projected - dual)
            dual, t = projected, t_next
        return self.project(obj - strength * forward_differences_adjoint(dual))


@register("beads", kind="object_prior")
class Beads(ObjectPrior):
    """Spheres of known diameter at unknown positions and amplitudes: the prior of a bead calibration.

    The object is ``sum_i a_i s(r - r_i)`` with ``s`` a solid sphere, rendered in Fourier space so positions are
    sub-voxel exact. Only :func:`~waveorder.nn.search.calibrate_beads` uses it; it has no penalty, so reconstruct with
    a Tikhonov prior. 3.x: ``object.prior.beads``.

    Parameters
    ----------
    diameter : float
        Sphere diameter, um.
    snr_threshold : float
        A bead's peak must exceed ``median + snr_threshold * 1.4826 * MAD`` of the data.
    max_beads : int
        Keep at most this many beads, brightest first.
    min_separation : int
        Side of the non-maximum-suppression window, voxels (rounded up to odd).
    """

    def __init__(
        self, diameter: float, *, snr_threshold: float = 6.0, max_beads: int = 50, min_separation: int = 5
    ) -> None:
        super().__init__()
        self.diameter = diameter
        self.snr_threshold = snr_threshold
        self.max_beads = max_beads
        self.min_separation = min_separation

    @override
    def extra_repr(self) -> str:
        return (
            f"diameter={self.diameter:g}, snr_threshold={self.snr_threshold:g}, max_beads={self.max_beads}, "
            f"min_separation={self.min_separation}"
        )

    @override
    def penalty(self, obj: Tensor, *, scale: Tensor) -> Tensor:
        raise IncompatibleModelError("Beads is a calibration prior with no penalty; reconstruct with Tikhonov")

    @override
    def prox(self, obj: Tensor, step: Tensor, *, scale: Tensor) -> Tensor:
        raise IncompatibleModelError("Beads is a calibration prior with no proximal map; reconstruct with Tikhonov")

    def form_factor(self, sampling: Sampling, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Return the sphere's transform, 1 at zero frequency, on the half spectrum ``(Z, Y, X // 2 + 1)``.

        ``3 (sin u - u cos u) / u^3`` with ``u = pi * diameter * |k|``, ``k`` in cycles per um; 1 where ``u < 1e-4``.
        """
        fz, fy, fx = half_spectrum_frequencies(sampling, device=device, dtype=dtype)
        u = math.pi * self.diameter * torch.sqrt(fz**2 + fy**2 + fx**2)
        small = u < 1e-4
        safe = torch.where(small, 1.0, u)
        return torch.where(small, 1.0, 3 * (torch.sin(safe) - safe * torch.cos(safe)) / safe**3)

    def detect(self, volume: Tensor) -> Tensor:
        """Return the integer ``(z, y, x)`` positions ``(N, 3)`` of the beads in ``(Z, Y, X)``, brightest first.

        A bead is a voxel equal to the maximum of its ``min_separation`` window and above
        ``median + t * 1.4826 * MAD`` for the first ``t`` of ``snr_threshold`` times 1, 0.75 and 0.5 that finds any;
        a relaxed ``t`` warns with :class:`~waveorder.errors.CalibrationWarning`.

        Raises
        ------
        ParameterError
            If no voxel passes even half the threshold (``path`` is ``snr_threshold``).
        """
        median = volume.median()
        mad = (volume - median).abs().median().clamp(min=1e-12)
        kernel = 2 * (self.min_separation // 2) + 1
        pooled = torch.nn.functional.max_pool3d(volume[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]
        peaks = volume == pooled
        for factor in (1.0, 0.75, 0.5):
            positions = (peaks & (volume > median + factor * self.snr_threshold * 1.4826 * mad)).nonzero()
            if positions.shape[0] > 0:
                break
        else:
            raise ParameterError(
                f"no beads above median + {0.5 * self.snr_threshold:g} * 1.4826 * MAD; lower snr_threshold "
                f"(currently {self.snr_threshold:g}) or check that the volume holds beads",
                path="snr_threshold",
            )
        if factor < 1:
            warnings.warn(
                f"no beads above median + {self.snr_threshold:g} * 1.4826 * MAD; relaxed the threshold to "
                f"{factor * self.snr_threshold:g}. Set snr_threshold to control this",
                CalibrationWarning,
                stacklevel=2,
            )
        order = torch.argsort(volume[positions.unbind(1)], descending=True)[: self.max_beads]
        return positions[order]

    def render(self, positions: Tensor, amplitudes: Tensor, sampling: Sampling) -> Tensor:
        """Return the object ``(Z, Y, X)`` of spheres with amplitudes ``(N,)`` at sub-voxel ``positions`` ``(N, 3)``.

        The spectrum is ``sum_i a_i exp(-2 pi i k . r_i)`` times :meth:`form_factor`, ``k`` in cycles per voxel,
        computed in the dtype of ``positions`` and differentiable in both arguments.
        """
        like = {"dtype": positions.dtype, "device": positions.device}
        spectrum = amplitudes.to(torch.complex128 if positions.dtype == torch.float64 else torch.complex64)
        spectrum = spectrum[:, None, None, None]
        for axis, count in enumerate(sampling.shape):
            frequency = torch.fft.rfftfreq(count, **like) if axis == 2 else torch.fft.fftfreq(count, **like)
            phase = -2 * math.pi * frequency * positions[:, axis, None]
            shape = [positions.shape[0], 1, 1, 1]
            shape[axis + 1] = frequency.shape[0]
            spectrum = spectrum * torch.polar(torch.ones_like(phase), phase).reshape(shape)
        form = self.form_factor(sampling, **like)
        return irfftn(spectrum.sum(dim=0) * form, sampling.shape)
