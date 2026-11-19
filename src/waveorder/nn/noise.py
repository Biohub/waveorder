"""Noise models p(n): the data likelihood, its gradient, sampling, and variance given the noiseless image."""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, Literal, override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError
from waveorder.nn.parameters import CalibratableModule, Free, Quantity
from waveorder.registry import register


def estimate_sigma(data: Tensor, *, method: Literal["std", "mad"] = "std") -> Tensor:
    """Return a noise standard deviation read off ``data``. 3.x: the start of ``sigma: estimate``.

    Parameters
    ----------
    data : Tensor
        Measured data, any shape.
    method : {"std", "mad"}
        ``"std"``: the unbiased standard deviation of all samples (the 3.x evidence path).
        ``"mad"``: ``1.4826`` times the median absolute deviation from the median, at least ``1e-6`` (the 3.x bead
        calibration).

    Returns
    -------
    Tensor
        0-d, in the dtype and on the device of ``data``.
    """
    if method == "mad":
        median = data.median()
        return ((data - median).abs().median() * 1.4826).clamp_min(1e-6)
    return data.std()


class Noise(CalibratableModule, ABC):
    """Distribution of the data ``g`` around the noiseless image ``mean = H f``, both in data units.

    A subclass implements :meth:`nll`, :meth:`sample` and :meth:`variance`; :meth:`gradient` and :meth:`scale` have
    defaults. The class attributes below declare what a solver may rely on.

    Attributes
    ----------
    has_curvature_bound : ClassVar[bool]
        Whether ``d^2 D / d mean^2`` is bounded, so :meth:`curvature_bound` returns it; FISTA needs this.
    has_photon_counts : ClassVar[bool]
        Whether the noise is shot noise, so :meth:`photon_counts` returns the signal; Richardson-Lucy needs this.
    """

    has_curvature_bound: ClassVar[bool] = False
    has_photon_counts: ClassVar[bool] = False

    @abstractmethod
    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        """Return the data-fit term ``D = -log p(data | mean)``, summed over voxels."""

    def gradient(self, data: Tensor, mean: Tensor) -> Tensor:
        """Return ``dD / d mean``, the same shape as ``mean``; by default autograd of :meth:`nll`."""
        return torch.func.grad(lambda point: self.nll(data, point))(mean)

    def curvature_bound(self) -> Tensor:
        """Return an upper bound on ``d^2 D / d mean^2`` over all means, in 1 / data units^2.

        First-order solvers multiply it by ``||H||^2`` for the Lipschitz constant of the data term's gradient.

        Raises
        ------
        IncompatibleModelError
            By default: the noise declares ``has_curvature_bound = False``.
        """
        raise IncompatibleModelError(
            f"{type(self).__name__} declares has_curvature_bound = False: its data term has no Lipschitz gradient"
        )

    def scale(self) -> Tensor:
        """Return the scale ``s`` that object priors divide their weight by: ``P(f) = weight * R(f) / s``.

        By default :meth:`white_gaussian_variance` (``sigma^2``, so Tikhonov's ``weight`` is the 3.x ``lambda`` in
        ``conj(H)/(|H|^2+lambda)``) where the noise reports one, else 1.
        """
        variance = self.white_gaussian_variance()
        return torch.ones(()) if variance is None else variance

    @abstractmethod
    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """Draw data around ``mean``."""

    @abstractmethod
    def variance(self, mean: Tensor) -> Tensor:
        """Return the per-voxel variance of the data given ``mean``."""

    def white_gaussian_variance(self) -> Tensor | None:
        """Return ``sigma^2`` if the noise is additive, white and Gaussian, else ``None``.

        The closed-form verbs (Wiener, the evidence) require it.
        """
        return None

    def photon_counts(self, data: Tensor) -> Tensor:
        """Return ``data`` as a nonnegative shot-noise signal in data units, comparable with ``mean = H f``.

        For Poisson noise it is ``gain * max((data - offset) / gain, 0)``.

        Raises
        ------
        IncompatibleModelError
            By default: the noise declares ``has_photon_counts = False``.
        """
        del data
        raise IncompatibleModelError(
            f"{type(self).__name__} declares has_photon_counts = False: it reports no photon_counts"
        )

    def photon_background(self) -> Tensor | float:
        """Return the expected signal, in :meth:`photon_counts` units, that the noise itself adds to ``mean = H f``.

        Richardson-Lucy divides :meth:`photon_counts` by ``H f + b + photon_background()``.
        """
        return 0.0


@register("gaussian", kind="noise")
class Gaussian(Noise):
    """Additive white Gaussian noise with standard deviation ``sigma``, in data units.

    ``D = |g - H f|^2 / (2 sigma^2)`` plus constants, with curvature bound ``1 / sigma^2``.

    Parameters
    ----------
    sigma : float or Free or "estimate"
        Standard deviation in data units; a :class:`~waveorder.nn.parameters.Free` value makes it calibratable as
        ``noise.sigma``. ``"estimate"`` makes it free with a start and bounds read from the data when a calibration
        begins (:meth:`start_from`); until then it is 1.

    Raises
    ------
    ParameterError
        If ``sigma`` is not positive and finite, or its ``Free`` bounds reach zero or below; ``path`` is ``"sigma"``.
    """

    has_curvature_bound: ClassVar[bool] = True
    sigma: Tensor

    def __init__(self, sigma: Quantity | Literal["estimate"] = 1.0) -> None:
        super().__init__()
        self.estimated = isinstance(sigma, str)  # "estimate"
        self.register_quantity(
            "sigma", Free(1.0, bounds=(1e-3, 1.0)) if isinstance(sigma, str) else sigma, positive=True
        )

    def start_from(self, data: Tensor) -> None:
        """With ``sigma="estimate"``, start at the data's standard deviation ``s``, bounded to ``(1e-3 s, s)``.

        This is the start and range of the 3.x evidence path; otherwise it does nothing.
        """
        if not self.estimated:
            return
        scale = float(estimate_sigma(data))
        self._free["sigma"] = Free(scale, bounds=(1e-3 * scale, scale))
        with torch.no_grad():
            self.sigma.fill_(scale)

    @override
    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        residual = (data - mean) / self.sigma
        return 0.5 * (residual**2).sum() + data.numel() * (torch.log(self.sigma) + 0.5 * math.log(2 * math.pi))

    @override
    def gradient(self, data: Tensor, mean: Tensor) -> Tensor:
        return (mean - data) / self.sigma**2

    @override
    def curvature_bound(self) -> Tensor:
        return 1 / self.sigma**2

    @override
    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + self.sigma * noise

    @override
    def variance(self, mean: Tensor) -> Tensor:
        return (self.sigma**2).expand(mean.shape)

    @override
    def white_gaussian_variance(self) -> Tensor:
        return self.sigma**2


@register("poisson", kind="noise")
class Poisson(Noise):
    """Shot noise: ``(data - offset) / gain`` counts photons, Poisson-distributed with rate ``mean / gain``.

    ``mean`` is the expected signal above ``offset``, in data units. Richardson-Lucy accepts it; FISTA does not.

    Parameters
    ----------
    gain : float or Free
        Data units per photon (the camera gain), positive.
    offset : float or Free
        Camera value of zero photons, in data units. Values below it count as zero photons.

    Raises
    ------
    ParameterError
        If ``gain`` is not positive and finite, or ``offset`` not finite; ``path`` names the quantity.
    """

    has_photon_counts: ClassVar[bool] = True
    gain: Tensor
    offset: Tensor

    def __init__(self, *, gain: Quantity = 1.0, offset: Quantity = 0.0) -> None:
        super().__init__()
        self.register_quantity("gain", gain, positive=True)
        self.register_quantity("offset", offset)

    def counts(self, data: Tensor) -> Tensor:
        """Return the photon counts ``max((data - offset) / gain, 0)`` of ``data``.

        A negative count has no Poisson probability, so it is clamped to zero.
        """
        return ((data - self.offset) / self.gain).clamp_min(0.0)

    def _rate(self, mean: Tensor) -> Tensor:
        return (mean / self.gain).clamp_min(torch.finfo(mean.dtype).tiny)

    @override
    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        counts = self.counts(data)
        rate = self._rate(mean)
        return (rate - counts * torch.log(rate) + torch.lgamma(counts + 1)).sum()

    @override
    def gradient(self, data: Tensor, mean: Tensor) -> Tensor:
        # The rate is clamped at the smallest normal number; below it the likelihood is flat in the mean.
        inside = mean / self.gain > torch.finfo(mean.dtype).tiny
        return torch.where(inside, (1 - self.counts(data) / self._rate(mean)) / self.gain, torch.zeros_like(mean))

    @override
    def photon_counts(self, data: Tensor) -> Tensor:
        return self.gain * self.counts(data)

    @override
    def scale(self) -> Tensor:
        return torch.ones_like(self.gain)

    @override
    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        return self.gain * torch.poisson(self._rate(mean), generator=generator) + self.offset

    @override
    def variance(self, mean: Tensor) -> Tensor:
        return self.gain * mean.clamp_min(0.0)


@register("poisson_gaussian", kind="noise")
class PoissonGaussian(Noise):
    """Shot noise plus Gaussian read noise: ``data = gain * Poisson(mean / gain) + offset + sigma * N(0, 1)``.

    The likelihood uses the shifted-Poisson approximation of Snyder et al. (JOSA A 1993): ``(data - offset) / gain +
    sigma^2 / gain^2`` is Poisson with rate ``mean / gain + sigma^2 / gain^2``, which has the exact mean and variance
    and is accurate when the read noise spans a few photons or more. Richardson-Lucy treats the shift as background
    (:meth:`photon_background`); FISTA refuses it. :meth:`sample` draws from the exact model. No 3.x counterpart.

    Parameters
    ----------
    gain : float or Free
        Data units per photon (the camera gain), positive.
    offset : float or Free
        Camera value of zero photons, in data units.
    sigma : float or Free
        Standard deviation of the read noise, in data units, positive.

    Raises
    ------
    ParameterError
        If ``gain`` or ``sigma`` is not positive and finite, or ``offset`` not finite; ``path`` names the quantity.
    """

    has_photon_counts: ClassVar[bool] = True
    gain: Tensor
    offset: Tensor
    sigma: Tensor

    def __init__(self, *, gain: Quantity = 1.0, offset: Quantity = 0.0, sigma: Quantity = 1.0) -> None:
        super().__init__()
        self.register_quantity("gain", gain, positive=True)
        self.register_quantity("offset", offset)
        self.register_quantity("sigma", sigma, positive=True)

    def _shift(self) -> Tensor:
        """``sigma^2 / gain^2``: the read-noise variance in photons squared, added to counts and rate alike."""
        return (self.sigma / self.gain) ** 2

    def counts(self, data: Tensor) -> Tensor:
        """Return the shifted counts ``max((data - offset) / gain + sigma^2 / gain^2, 0)``."""
        return ((data - self.offset) / self.gain + self._shift()).clamp_min(0.0)

    def _rate(self, mean: Tensor) -> Tensor:
        return (mean / self.gain + self._shift()).clamp_min(torch.finfo(mean.dtype).tiny)

    @override
    def nll(self, data: Tensor, mean: Tensor) -> Tensor:
        counts = self.counts(data)
        rate = self._rate(mean)
        return (rate - counts * torch.log(rate) + torch.lgamma(counts + 1)).sum()

    @override
    def gradient(self, data: Tensor, mean: Tensor) -> Tensor:
        # As for Poisson: below the clamped rate the likelihood is flat in the mean.
        inside = mean / self.gain + self._shift() > torch.finfo(mean.dtype).tiny
        return torch.where(inside, (1 - self.counts(data) / self._rate(mean)) / self.gain, torch.zeros_like(mean))

    @override
    def photon_counts(self, data: Tensor) -> Tensor:
        return self.gain * self.counts(data)

    @override
    def photon_background(self) -> Tensor:
        return self.gain * self._shift()

    @override
    def scale(self) -> Tensor:
        return torch.ones_like(self.gain)

    @override
    def sample(self, mean: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        shot = torch.poisson((mean / self.gain).clamp_min(0.0), generator=generator)
        read = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return self.gain * shot + self.offset + self.sigma * read

    @override
    def variance(self, mean: Tensor) -> Tensor:
        return self.gain * mean.clamp_min(0.0) + self.sigma**2
