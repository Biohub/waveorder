"""Object-oriented, one-``nn.Module``-per-distribution layer over the calibrate numerics.

WaveOrder 4.0 writes the forward model as ``g = H f + n`` with three
distributions:

* ``p(H)`` -- the imaging system and its prior,
* ``p(f)`` -- the object prior,
* ``p(n)`` -- the noise.

This module gives each distribution a :class:`torch.nn.Module` (or a plain
class where there is nothing to carry), mirroring Sri's design. It is a thin,
*additive* layer: every numeric here calls the existing, tested functions in
:mod:`waveorder.calibrate` and :mod:`waveorder.models.phase_thick_3d`. Nothing
reimplements the math, so the regression suite stays byte-for-byte valid.

Distribution -> module -> config map
------------------------------------
======================  =========================  ================================
distribution            module                     config block
======================  =========================  ================================
``p(H)`` imaging system :class:`Fluorescence3D`     ``imaging_system`` (fluorescence)
``p(H)`` imaging system :class:`Phase3D`            ``imaging_system`` (phase)
``p(H)`` parameter prior :class:`ParameterPrior`    ``imaging_system`` tolerances
``p(f)`` object prior   :class:`Tikhonov`           ``object.prior: tikhonov``
``p(f)`` object prior   :class:`TikhonovSpectral`   ``object.prior.tikhonov_spectral``
``p(f)`` object prior   :class:`FittedSpectrum`     ``object.prior`` fitted (iso-Whittle)
``p(n)`` noise          :class:`Gaussian`           ``noise.gaussian``
``p(n)`` noise          :class:`Poisson`            ``noise.poisson``
======================  =========================  ================================

A configuration value written ``{nominal, tolerance}`` becomes an
:class:`torch.nn.Parameter` (a free parameter ``P`` of the design, estimated
from the data); a plain number becomes a registered buffer. The dotted names
from :meth:`torch.nn.Module.named_parameters` are the ``parameter_names`` of a
``tf.zarr`` store (:mod:`waveorder.calibrate.tfzarr`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from waveorder import sampling
from waveorder.calibrate.scores import calibration_penalty, whittle_evidence
from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.calibrate.zernike_names import name_to_noll, noll_to_name
from waveorder.models import phase_thick_3d
from waveorder.reconstruct import tikhonov_regularized_inverse_filter

__all__ = [
    "Sampling",
    "ImagingSystem",
    "Fluorescence3D",
    "Phase3D",
    "ParameterPrior",
    "ObjectPrior",
    "Tikhonov",
    "TikhonovSpectral",
    "FittedSpectrum",
    "Noise",
    "Gaussian",
    "Poisson",
    "evidence",
    "calibrate",
    "reconstruct",
    "CalibrationResult",
]


# --------------------------------------------------------------------------- #
# Sampling: grid / geometry                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class Sampling:
    """Grid geometry shared by an imaging system and an object prior.

    A thin container around the pixel sizes and shape; the Nyquist helpers
    delegate to :mod:`waveorder.sampling` so there is a single source of truth
    for the sample-spacing formulas.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        ``(Z, Y, X)`` data shape (before axial padding).
    yx_pixel_size : float
        Transverse pixel size, um.
    z_pixel_size : float
        Axial pixel size, um.
    z_padding : int
        Axial padding added on each side before building the transfer
        function.
    """

    zyx_shape: tuple[int, int, int]
    yx_pixel_size: float
    z_pixel_size: float
    z_padding: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        """The unpadded ``(Z, Y, X)`` shape."""
        return tuple(self.zyx_shape)

    @property
    def padded_shape(self) -> tuple[int, int, int]:
        """The ``(Z + 2 z_padding, Y, X)`` shape used for transfer functions."""
        z, y, x = self.zyx_shape
        return (z + 2 * self.z_padding, y, x)

    @property
    def yx(self) -> tuple[int, int]:
        """The transverse ``(Y, X)`` shape."""
        return tuple(self.zyx_shape[1:])

    def transverse_nyquist(self, na_illumination: float, na_detection: float, wavelength: float) -> float:
        """Transverse Nyquist spacing (see :func:`waveorder.sampling.transverse_nyquist`)."""
        return sampling.transverse_nyquist(wavelength, na_illumination, na_detection)

    def axial_nyquist(self, na_detection: float, wavelength: float, index_of_refraction_media: float) -> float:
        """Axial Nyquist spacing (see :func:`waveorder.sampling.axial_nyquist`)."""
        return sampling.axial_nyquist(wavelength, na_detection, index_of_refraction_media)

    def shell_index(self, n_shells: int, device=None) -> tuple[Tensor, Tensor]:
        """Isotropic ``|k|`` shell index per voxel, on the transfer-function grid.

        Builds the physical radial frequency ``|k|`` (cycles/um) over the
        padded ``(Z, Y, X)`` grid in FFT order and bins it into ``n_shells``
        equal-width shells. Used by :class:`FittedSpectrum` to profile the
        object spectrum per shell.

        Parameters
        ----------
        n_shells : int
            Number of equal-width ``|k|`` shells.
        device : torch.device, optional
            Device for the returned tensors.

        Returns
        -------
        idx : Tensor
            Integer shell index per voxel, shape ``padded_shape``, in ``[0,
            n_shells)``.
        centers : Tensor
            Shell center frequencies (cycles/um), shape ``(n_shells,)``.
        """
        zz, yy, xx = self.padded_shape
        fz = np.fft.fftfreq(zz, d=self.z_pixel_size)
        fy = np.fft.fftfreq(yy, d=self.yx_pixel_size)
        fx = np.fft.fftfreq(xx, d=self.yx_pixel_size)
        fr = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
        kmax = float(fr.max()) * 1.0001 + 1e-12
        edges = np.linspace(0.0, kmax, n_shells + 1)
        idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, n_shells - 1).reshape(fr.shape)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return (
            torch.as_tensor(idx, dtype=torch.long, device=device),
            torch.as_tensor(centers, dtype=torch.float64, device=device),
        )


# --------------------------------------------------------------------------- #
# Parameter bookkeeping: {nominal, tolerance} -> Parameter, number -> buffer  #
# --------------------------------------------------------------------------- #


@dataclass
class _Free:
    """A ``{nominal, tolerance}`` configuration value: becomes an nn.Parameter."""

    nominal: float
    tolerance: float | str = "free"


def free(nominal: float, tolerance: float | str = "free") -> _Free:
    """Mark a value free for calibration: ``free(nominal, tolerance)``.

    A :class:`Fluorescence3D` or :class:`Phase3D` built with a :class:`_Free`
    value registers it as an :class:`torch.nn.Parameter` (appears in
    ``named_parameters``); a plain number becomes a buffer.

    Parameters
    ----------
    nominal : float
        Prior center.
    tolerance : float or "free"
        Gaussian prior standard deviation, or ``"free"`` for an improper
        (flat) prior.

    Returns
    -------
    _Free
    """
    return _Free(float(nominal), tolerance)


class ImagingSystem(nn.Module):
    """Base class for ``p(H)``: an imaging system with free parameters.

    Subclasses own named optical parameters as buffers (fixed numbers) or
    :class:`torch.nn.Parameter` (free ``{nominal, tolerance}`` values), build
    the complex transfer function ``H_k`` with :meth:`transfer`, and apply it
    with :meth:`forward`. The tolerances/nominals of the free parameters are
    carried alongside so a :class:`ParameterPrior` can score them.
    """

    #: Optical-parameter dtype. ``float32`` matches the function API
    #: (:func:`fluorescence_otf`, :func:`phase_thick_3d.calculate_transfer_function`)
    #: exactly, so :meth:`transfer` reproduces it to the bit; the scores still
    #: accumulate in ``float64`` internally.
    param_dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        # name -> (nominal, tolerance) for the free parameters only.
        self._prior: dict[str, tuple[float, float | str]] = {}

    def _register_value(self, name: str, value: float | _Free | Tensor) -> None:
        """Register ``value`` as a Parameter (if free) or a buffer (if a number)."""
        if isinstance(value, _Free):
            tensor = torch.tensor(float(value.nominal), dtype=self.param_dtype)
            self.register_parameter(name, nn.Parameter(tensor))
            self._prior[name] = (float(value.nominal), value.tolerance)
        else:
            tensor = torch.as_tensor(float(value), dtype=self.param_dtype)
            self.register_buffer(name, tensor)

    def free_parameter_names(self) -> list[str]:
        """Dotted names of the free parameters (``tf.zarr`` ``parameter_names``)."""
        return [name for name, _ in self.named_parameters()]

    def parameter_values(self) -> dict[str, Tensor]:
        """Current values of the free parameters by dotted name."""
        return {name: param for name, param in self.named_parameters()}

    def nominals(self) -> dict[str, float]:
        """Prior centers of the free parameters by dotted name."""
        return {name: nominal for name, (nominal, _) in self._prior.items()}

    def tolerances(self) -> dict[str, float | str]:
        """Prior tolerances of the free parameters by dotted name."""
        return {name: tolerance for name, (_, tolerance) in self._prior.items()}

    def transfer(self, s: Sampling) -> Tensor:  # pragma: no cover - abstract
        """Complex transfer function ``H_k`` in FFT order."""
        raise NotImplementedError

    def prepare(self, raw: Tensor, s: Sampling) -> Tensor:
        """Turn a raw stack into the modeled data ``g`` (identity by default)."""
        return torch.as_tensor(raw)

    def forward(self, f: Tensor, s: Sampling) -> Tensor:
        """Apply the transfer function: ``g = H f`` (real part of the IFFT)."""
        otf = self.transfer(s)
        f_fft = torch.fft.fftn(torch.as_tensor(f, dtype=otf.dtype if otf.is_complex() else torch.complex64))
        return torch.real(torch.fft.ifftn(f_fft * otf))


class Fluorescence3D(ImagingSystem):
    """Widefield fluorescence ``p(H)``; wraps :func:`fluorescence_otf`.

    Owns the detection NA, named Zernike coefficients, focus offset, emission
    wavelength, medium index and pixel sizes. Any value passed as
    :func:`free` becomes a calibratable :class:`torch.nn.Parameter`; a plain
    number becomes a buffer. :meth:`transfer` calls :func:`fluorescence_otf`
    with the current values, so the OTF is identical to the function API.

    Parameters
    ----------
    wavelength_emission : float
        Emission wavelength, um.
    index_of_refraction_media : float
        Medium refractive index.
    numerical_aperture_detection : float or _Free
        Detection NA.
    zernike_coefficients : mapping, optional
        ``{name_or_noll: value}`` in waves; each value a number (buffer) or a
        :func:`free` value (Parameter). The dotted parameter name is
        ``pupil_aberrations.<canonical_name>`` (e.g.
        ``pupil_aberrations.astigmatism_0``).
    z_focus_offset : float or _Free
        Focus offset in slices.
    normalize : {"peak", "none"}
        OTF normalization, passed to :func:`fluorescence_otf`.
    """

    def __init__(
        self,
        wavelength_emission: float,
        index_of_refraction_media: float,
        numerical_aperture_detection: float | _Free,
        zernike_coefficients: Mapping[str | int, float | _Free] | None = None,
        z_focus_offset: float | _Free = 0.0,
        normalize: str = "peak",
    ) -> None:
        super().__init__()
        self.wavelength_emission = float(wavelength_emission)
        self.index_of_refraction_media = float(index_of_refraction_media)
        self.normalize = normalize

        self._register_value("numerical_aperture_detection", numerical_aperture_detection)
        self._register_value("z_focus_offset", z_focus_offset)

        # Zernike coefficients live under a dotted "pupil_aberrations." prefix
        # so their free names round-trip through tf.zarr parameter_names.
        self._zernike_nolls: list[int] = []
        self.pupil_aberrations = nn.Module()
        aberrations_prior: dict[str, tuple[float, float | str]] = {}
        for key, value in (zernike_coefficients or {}).items():
            noll = key if isinstance(key, int) else name_to_noll(key)
            canonical = noll_to_name(noll)
            self._zernike_nolls.append(noll)
            if isinstance(value, _Free):
                tensor = torch.tensor(float(value.nominal), dtype=self.param_dtype)
                self.pupil_aberrations.register_parameter(canonical, nn.Parameter(tensor))
                aberrations_prior[f"pupil_aberrations.{canonical}"] = (float(value.nominal), value.tolerance)
            else:
                tensor = torch.as_tensor(float(value), dtype=self.param_dtype)
                self.pupil_aberrations.register_buffer(canonical, tensor)
        self._prior.update(aberrations_prior)

    def _coefficients(self) -> dict[int, float | Tensor]:
        """Current Zernike coefficients keyed by Noll index."""
        out: dict[int, float | Tensor] = {}
        for noll in self._zernike_nolls:
            canonical = noll_to_name(noll)
            value = getattr(self.pupil_aberrations, canonical)
            out[noll] = value
        return out

    def transfer(self, s: Sampling) -> Tensor:
        """Complex OTF from :func:`fluorescence_otf` at the current parameters."""
        return fluorescence_otf(
            zyx_shape=s.padded_shape,
            yx_pixel_size=s.yx_pixel_size,
            z_pixel_size=s.z_pixel_size,
            wavelength_emission=self.wavelength_emission,
            index_of_refraction_media=self.index_of_refraction_media,
            numerical_aperture_detection=self.numerical_aperture_detection,
            zernike_coefficients=self._coefficients(),
            z_focus_offset=self.z_focus_offset,
            normalize=self.normalize,
        )


class Phase3D(ImagingSystem):
    """Thick-phase ``p(H)``; wraps :func:`phase_thick_3d.calculate_transfer_function`.

    The weak-object model has two transfer functions: the real (phase) WOTF
    and the imaginary (absorption) WOTF, both returned by WaveOrder as complex
    tensors. :meth:`transfer_real_imag` exposes the pair directly for the
    label-free reconstruction path. :meth:`transfer` returns the phase WOTF as
    the single ``H_k`` (the usual pure-phase case; combine with the absorption
    WOTF via an ``absorption_ratio`` at reconstruction time, as
    :func:`phase_thick_3d.apply_inverse_transfer_function` does).

    Parameters
    ----------
    wavelength_illumination : float
        Illumination wavelength, um.
    index_of_refraction_media : float
        Medium refractive index.
    numerical_aperture_illumination : float or _Free
        Condenser NA.
    numerical_aperture_detection : float or _Free
        Detection NA.
    tilt_angle_zenith, tilt_angle_azimuth : float or _Free
        Illumination tilt angles, radians.
    invert_phase_contrast : bool
        Flip the defocus axis (matches 3.x).
    pupil_steepness : float
        Sigmoid pupil-edge steepness.
    """

    def __init__(
        self,
        wavelength_illumination: float,
        index_of_refraction_media: float,
        numerical_aperture_illumination: float | _Free = 0.9,
        numerical_aperture_detection: float | _Free = 1.2,
        tilt_angle_zenith: float | _Free = 0.0,
        tilt_angle_azimuth: float | _Free = 0.0,
        invert_phase_contrast: bool = False,
        pupil_steepness: float = 1e4,
    ) -> None:
        super().__init__()
        self.wavelength_illumination = float(wavelength_illumination)
        self.index_of_refraction_media = float(index_of_refraction_media)
        self.invert_phase_contrast = bool(invert_phase_contrast)
        self.pupil_steepness = float(pupil_steepness)

        self._register_value("numerical_aperture_illumination", numerical_aperture_illumination)
        self._register_value("numerical_aperture_detection", numerical_aperture_detection)
        self._register_value("tilt_angle_zenith", tilt_angle_zenith)
        self._register_value("tilt_angle_azimuth", tilt_angle_azimuth)

    def transfer_real_imag(self, s: Sampling) -> tuple[Tensor, Tensor]:
        """Real and imaginary potential WOTFs from the phase model.

        The optical parameters are ``float32`` (``param_dtype``), matching the
        model's internal cast, so the returned WOTFs reproduce the function
        API exactly while preserving autograd through the free parameters.
        """
        return phase_thick_3d.calculate_transfer_function(
            zyx_shape=s.shape,
            yx_pixel_size=s.yx_pixel_size,
            z_pixel_size=s.z_pixel_size,
            wavelength_illumination=self.wavelength_illumination,
            z_padding=s.z_padding,
            index_of_refraction_media=self.index_of_refraction_media,
            numerical_aperture_illumination=self.numerical_aperture_illumination,
            numerical_aperture_detection=self.numerical_aperture_detection,
            invert_phase_contrast=self.invert_phase_contrast,
            tilt_angle_zenith=self.tilt_angle_zenith,
            tilt_angle_azimuth=self.tilt_angle_azimuth,
            pupil_steepness=self.pupil_steepness,
        )

    def transfer(self, s: Sampling) -> Tensor:
        """The phase (real-potential) WOTF ``H_k`` for the pure-phase case."""
        real_tf, _ = self.transfer_real_imag(s)
        return real_tf


# --------------------------------------------------------------------------- #
# ParameterPrior: C(H) = -log p(H)                                            #
# --------------------------------------------------------------------------- #


class ParameterPrior:
    """The calibration penalty ``C(H) = -log p(H)``; wraps :func:`calibration_penalty`.

    Parameters
    ----------
    nominals : dict[str, float]
        Prior centers by parameter name.
    tolerances : dict[str, float or "free"]
        Gaussian prior standard deviations by parameter name (or ``"free"``).
    """

    def __init__(self, nominals: Mapping[str, float], tolerances: Mapping[str, float | str]) -> None:
        self.nominals = dict(nominals)
        self.tolerances = dict(tolerances)

    @classmethod
    def from_system(cls, system: ImagingSystem) -> "ParameterPrior":
        """Build the prior from an imaging system's free-parameter nominals/tolerances."""
        return cls(system.nominals(), system.tolerances())

    def nll(self, params: Mapping[str, float | Tensor]) -> Tensor:
        """Negative log prior ``C = sum_i (h_i - h_i0)^2 / (2 tau_i^2)``."""
        return calibration_penalty(dict(params), self.nominals, self.tolerances)

    def sample(self, generator: torch.Generator | None = None) -> dict[str, float]:
        """Draw a parameter dict from the Gaussian prior (``free`` -> nominal)."""
        out: dict[str, float] = {}
        for name, nominal in self.nominals.items():
            tolerance = self.tolerances.get(name, "free")
            if tolerance == "free":
                out[name] = float(nominal)
            else:
                noise = torch.randn((), generator=generator, dtype=torch.float64)
                out[name] = float(nominal) + float(tolerance) * float(noise)
        return out


# --------------------------------------------------------------------------- #
# ObjectPrior: p(f) -> S_k                                                     #
# --------------------------------------------------------------------------- #


class ObjectPrior(nn.Module):
    """Base class for ``p(f)``: returns the object power spectrum ``S_k``.

    Priors come in two flavours. A *fixed* prior (``Tikhonov``,
    ``TikhonovSpectral``) ignores the data and returns ``S_k`` from stored
    parameters. A *data-profiled* (empirical-Bayes) prior (``FittedSpectrum``,
    the winning iso-Whittle method) estimates ``S_k`` from the data
    periodogram and the current transfer function; :func:`evidence` passes
    ``data_fft`` and ``otf`` through for those. The optional keywords keep the
    fixed priors' call sites unchanged.
    """

    #: True for priors whose ``spectrum`` consumes ``data_fft``/``otf``.
    data_profiled: bool = False

    def spectrum(
        self,
        s: Sampling,
        noise: "Noise",
        *,
        data_fft: Tensor | None = None,
        otf: Tensor | None = None,
    ) -> Tensor | None:  # pragma: no cover - abstract
        """Object power spectrum ``S_k`` (same shape as ``H_k``), or ``None``."""
        raise NotImplementedError


class Tikhonov(ObjectPrior):
    """Flat Tikhonov prior ``S_k = sigma^2 / lambda`` (the current default).

    Parameters
    ----------
    weight : float
        Regularization weight ``lambda``.
    """

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(float(weight), dtype=torch.float64))

    def spectrum(self, s: Sampling, noise: "Noise", **_: Tensor) -> Tensor:
        """``S_k = sigma^2 / lambda`` (a scalar; broadcast against ``H_k``)."""
        return noise.variance() / self.weight


class TikhonovSpectral(ObjectPrior):
    """Explicit per-frequency object spectrum ``S_k``.

    Parameters
    ----------
    spectrum : Tensor
        The object power spectrum ``S_k`` in FFT order, same shape as ``H_k``.
    """

    def __init__(self, spectrum: Tensor) -> None:
        super().__init__()
        self.register_buffer("spectrum_buffer", torch.as_tensor(spectrum, dtype=torch.float64))

    def spectrum(self, s: Sampling, noise: "Noise", **_: Tensor) -> Tensor:
        """Return the stored ``S_k``."""
        return self.spectrum_buffer


class FittedSpectrum(ObjectPrior):
    """Isotropic shell-fit object spectrum ``S(|k|)`` estimated from the data.

    This is the winning "iso-Whittle" object prior of the 2026-10-30 blind
    sprint, ported onto the module interface. It is a data-profiled
    (empirical-Bayes) prior: instead of a fixed ``S_k``, it estimates the
    object power spectrum per isotropic frequency shell directly from the
    data periodogram and the current transfer function, which is what beat a
    flat Tikhonov prior on real blind data (see the theory page and
    ``winning_method_spec.md``).

    Per shell of ``|k|``, with periodogram ``P_k = |FFT(g)|^2`` (ortho),
    transfer magnitude ``|H_k|^2`` and noise floor ``sigma^2``, the fit is the
    threshold-free, ``|H|^2``-weighted least squares estimate

        S_shell = sum_k |H_k|^2 (P_k - sigma^2) / sum_k |H_k|^4   (clamped >= 0)

    (``whittle3d._wls_shell_S``). The weighting by ``|H|^2`` is what makes it
    threshold-free: a fixed ``|H|^2`` cutoff would turn the evidence into an
    OTF-support indicator (the round-1 failure). The shell profile is then
    broadcast back to every frequency. Being a positive, radially-binned
    spectrum, ``S_k`` keeps the Gaussian evidence exact.

    Parameters
    ----------
    n_shells : int
        Number of isotropic ``|k|`` shells.
    min_power : float
        Floor on the per-shell spectrum (keeps ``S_k > 0`` so ``log v_k`` and
        the Wiener inverse stay finite).
    """

    data_profiled = True

    def __init__(self, n_shells: int = 64, min_power: float = 1e-10) -> None:
        super().__init__()
        self.n_shells = int(n_shells)
        self.min_power = float(min_power)

    def spectrum(
        self,
        s: Sampling,
        noise: "Noise",
        *,
        data_fft: Tensor | None = None,
        otf: Tensor | None = None,
    ) -> Tensor:
        """Data-profiled ``S_k`` via an ``|H|^2``-weighted shell fit.

        Parameters
        ----------
        s : Sampling
            Grid geometry (supplies the physical ``|k|`` grid).
        noise : Noise
            Supplies ``sigma^2``.
        data_fft : Tensor
            ``fftn(g, norm="ortho")`` of the prepared data (``evidence`` passes
            this). Required.
        otf : Tensor
            Current transfer function ``H_k`` (``evidence`` passes this).
            Required.
        """
        if data_fft is None or otf is None:
            raise ValueError(
                "FittedSpectrum needs data_fft and otf; call it through distributions.evidence (which passes both)."
            )
        sigma2 = noise.variance().to(torch.float64)
        power = (data_fft.abs() ** 2).to(torch.float64)
        h2 = (otf.abs() ** 2).to(torch.float64)

        shell, _centers = s.shell_index(self.n_shells, device=power.device)
        shell = shell.reshape(-1)
        p = power.reshape(-1)
        h = h2.reshape(-1)
        num = torch.zeros(self.n_shells, dtype=torch.float64, device=p.device)
        den = torch.zeros(self.n_shells, dtype=torch.float64, device=p.device)
        num.scatter_add_(0, shell, h * torch.clamp(p - sigma2, min=0.0))
        den.scatter_add_(0, shell, h * h)
        s_shell = (num / den.clamp_min(1e-30)).clamp_min(self.min_power)
        return s_shell[shell].reshape(power.shape)


# --------------------------------------------------------------------------- #
# Noise: p(n)                                                                  #
# --------------------------------------------------------------------------- #


class Noise(nn.Module):
    """Base class for ``p(n)``."""

    def variance(self) -> Tensor:  # pragma: no cover - abstract
        """Noise variance ``sigma^2``."""
        raise NotImplementedError

    def nll(self, g: Tensor, mean: Tensor) -> Tensor:  # pragma: no cover - abstract
        """Negative log likelihood of ``g`` given its mean (the data misfit ``D``)."""
        raise NotImplementedError

    def sample(self, mean: Tensor, generator: torch.Generator | None = None) -> Tensor:  # pragma: no cover
        """Draw a noisy observation around ``mean``."""
        raise NotImplementedError


class Gaussian(Noise):
    """Additive white Gaussian noise ``n ~ N(0, sigma^2 I)``.

    Parameters
    ----------
    sigma : float
        Noise standard deviation in data units.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.register_buffer("sigma", torch.as_tensor(float(sigma), dtype=torch.float64))

    def variance(self) -> Tensor:
        """``sigma^2``."""
        return self.sigma**2

    def nll(self, g: Tensor, mean: Tensor) -> Tensor:
        """Gaussian data misfit ``||g - mean||^2 / (2 sigma^2) + const``."""
        residual = torch.as_tensor(g, dtype=torch.float64) - torch.as_tensor(mean, dtype=torch.float64)
        return (residual**2).sum() / (2.0 * self.variance())

    def sample(self, mean: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """``mean + sigma * N(0, 1)``."""
        mean = torch.as_tensor(mean, dtype=torch.float64)
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
        return mean + self.sigma * noise


class Poisson(Noise):
    """Poisson (shot) noise with a linear gain.

    TODO (next pass): the full Poisson evidence / Richardson-Lucy path. For
    now ``variance`` returns the gain so that a Gaussian-approximate evidence
    can still be scored, and ``sample`` draws true Poisson counts for the
    simulator.

    Parameters
    ----------
    gain : float
        Electrons (or ADU) per photon.
    offset : float
        Constant dark offset in data units.
    """

    def __init__(self, gain: float = 1.0, offset: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("gain", torch.as_tensor(float(gain), dtype=torch.float64))
        self.register_buffer("offset", torch.as_tensor(float(offset), dtype=torch.float64))

    def variance(self) -> Tensor:
        """Gaussian-approximate variance (the gain); a true shot-noise map is per-voxel."""
        return self.gain

    def nll(self, g: Tensor, mean: Tensor) -> Tensor:
        """Poisson negative log likelihood ``sum(mean - g log mean)`` (up to const)."""
        g = torch.as_tensor(g, dtype=torch.float64)
        rate = (torch.as_tensor(mean, dtype=torch.float64) - self.offset) / self.gain
        rate = rate.clamp(min=1e-12)
        return (rate - (g / self.gain) * torch.log(rate)).sum()

    def sample(self, mean: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Draw Poisson counts ``offset + gain * Poisson((mean - offset) / gain)``."""
        rate = (torch.as_tensor(mean, dtype=torch.float64) - self.offset) / self.gain
        counts = torch.poisson(rate.clamp(min=0.0), generator=generator)
        return self.offset + self.gain * counts


# --------------------------------------------------------------------------- #
# evidence / calibrate / reconstruct                                          #
# --------------------------------------------------------------------------- #


def evidence(
    g: Tensor,
    system: ImagingSystem,
    obj: ObjectPrior,
    noise: Noise,
    s: Sampling,
    *,
    exclude_dc: bool = False,
    data_fft: Tensor | None = None,
) -> Tensor:
    """Negative log evidence ``-log p(g | H)`` of the imaging system.

    Exactly Sri's snippet: per frequency ``v_k = |H_k|^2 S_k + sigma^2`` and
    the Whittle NLL. It calls :func:`whittle_evidence` with the module-derived
    ``H_k``, ``S_k`` and ``sigma`` so the value matches the function API.

    Parameters
    ----------
    g : Tensor
        Observed stack ``(Z, Y, X)``.
    system : ImagingSystem
        ``p(H)``; supplies ``H_k`` via :meth:`ImagingSystem.transfer`.
    obj : ObjectPrior
        ``p(f)``; supplies ``S_k`` via :meth:`ObjectPrior.spectrum`.
    noise : Noise
        ``p(n)``; supplies ``sigma``.
    s : Sampling
        Grid geometry.
    exclude_dc : bool
        Drop the ``k = 0`` term (for real data with a large background).
    data_fft : Tensor, optional
        Precomputed ``fftn(g, norm="ortho")``.

    Returns
    -------
    Tensor
        Scalar negative log evidence.
    """
    otf = system.transfer(s)
    # data-profiled priors (FittedSpectrum) need the periodogram and H_k; fixed
    # priors (Tikhonov, TikhonovSpectral) ignore these kwargs.
    if getattr(obj, "data_profiled", False) and data_fft is None:
        data_fft = torch.fft.fftn(g, norm="ortho")
    s_k = obj.spectrum(s, noise, data_fft=data_fft, otf=otf)
    sigma = torch.sqrt(noise.variance())
    return whittle_evidence(
        data=None if data_fft is not None else g,
        data_fft=data_fft,
        otf=otf,
        noise_sigma=sigma,
        object_spectrum=s_k,
        exclude_dc=exclude_dc,
    )


def _laplace_covariance(objective, params: list[nn.Parameter]) -> Tensor:
    """Inverse Hessian of ``objective`` w.r.t. ``params`` by double backprop.

    Parameters
    ----------
    objective : callable
        Scalar objective evaluated at the current parameter values.
    params : list of nn.Parameter
        The free parameters; the Hessian is taken over their flattened stack.

    Returns
    -------
    Tensor
        ``[P, P]`` Laplace covariance (inverse Hessian at the optimum).
    """
    value = objective()
    grads = torch.autograd.grad(value, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    rows = []
    for i in range(flat_grad.numel()):
        second = torch.autograd.grad(flat_grad[i], params, retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in second]))
    hessian = torch.stack(rows).to(torch.float64)
    return torch.linalg.inv(hessian).detach()


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate`."""

    parameters: dict[str, float]
    parameter_names: list[str]
    covariance: np.ndarray
    score_trace: list[float]
    best_score: float


def calibrate(
    g: Tensor,
    system: ImagingSystem,
    prior: ParameterPrior,
    obj: ObjectPrior,
    noise: Noise,
    s: Sampling,
    *,
    score=evidence,
    max_iter: int = 200,
    lr: float = 0.01,
    exclude_dc: bool = False,
) -> CalibrationResult:
    """Estimate ``H``'s free parameters by minimizing ``score + prior.nll``.

    The outer argmin over ``system``'s :meth:`torch.nn.Module.named_parameters`
    (the design's free params ``P``), by Adam, with the Laplace covariance
    (inverse Hessian of the total objective at the optimum).

    Parameters
    ----------
    g : Tensor
        Observed stack.
    system : ImagingSystem
        The imaging system; its free parameters are optimized in place.
    prior : ParameterPrior
        ``C(H)``, added to the data score.
    obj : ObjectPrior
        ``p(f)``.
    noise : Noise
        ``p(n)``.
    s : Sampling
        Grid geometry.
    score : callable
        Data score, ``score(g, system, obj, noise, s, exclude_dc=...)``;
        defaults to :func:`evidence`.
    max_iter : int
        Adam iterations.
    lr : float
        Adam step size.
    exclude_dc : bool
        Passed to the score.

    Returns
    -------
    CalibrationResult
    """
    names = system.free_parameter_names()
    params = [param for _, param in system.named_parameters()]
    if not params:
        raise ValueError("system has no free parameters to calibrate; mark values with free(...)")

    g = torch.as_tensor(g, dtype=torch.float64)
    data_fft = torch.fft.fftn(g, norm="ortho")

    def objective() -> Tensor:
        data_term = score(g, system, obj, noise, s, exclude_dc=exclude_dc, data_fft=data_fft)
        return data_term + prior.nll(system.parameter_values())

    optimizer = torch.optim.Adam(params, lr=lr)
    score_trace: list[float] = []
    for _ in range(max_iter):
        optimizer.zero_grad()
        value = objective()
        value.backward()
        optimizer.step()
        score_trace.append(float(value.detach()))

    with torch.no_grad():
        best_score = float(objective())
    estimated = {name: float(param.detach()) for name, param in zip(names, params)}

    # Laplace covariance: inverse Hessian of the total objective at the
    # optimum. Build it by double backprop through the live parameters so the
    # graph stays intact (a copy_/no_grad probe would detach it).
    try:
        covariance = _laplace_covariance(objective, params).numpy()
    except RuntimeError:
        covariance = np.full((len(params), len(params)), np.nan)

    return CalibrationResult(
        parameters=estimated,
        parameter_names=names,
        covariance=covariance,
        score_trace=score_trace,
        best_score=best_score,
    )


def reconstruct(
    g: Tensor,
    system: ImagingSystem,
    obj: ObjectPrior,
    noise: Noise,
    s: Sampling,
) -> Tensor:
    """Inner argmin over ``f``: the closed-form Wiener estimate.

    For Gaussian noise and a Tikhonov object prior this is exactly the
    Wiener filter ``f_hat = H* / (|H|^2 + lambda) . g``, identical to
    :func:`waveorder.reconstruct.tikhonov_regularized_inverse_filter` applied
    to ``g`` with ``lambda = sigma^2 / S_k``.

    Parameters
    ----------
    g : Tensor
        Observed stack ``(Z, Y, X)``.
    system : ImagingSystem
        Supplies ``H_k``.
    obj : ObjectPrior
        Supplies ``S_k`` (hence ``lambda = sigma^2 / S_k``).
    noise : Noise
        Supplies ``sigma^2``.
    s : Sampling
        Grid geometry.

    Returns
    -------
    Tensor
        Real-space object estimate ``f_hat``, same shape as ``g``.

    Raises
    ------
    NotImplementedError
        For a non-Gaussian noise or an improper (``None``) object spectrum;
        the iterative / Richardson-Lucy path lands in a later pass.
    """
    if not isinstance(noise, Gaussian):
        raise NotImplementedError("reconstruct's closed form is Gaussian-only for now; RL path follows")
    s_k = obj.spectrum(s, noise)
    if s_k is None:
        raise NotImplementedError("improper object prior (S_k=None) has no closed-form Wiener estimate")

    otf = system.transfer(s)
    g = torch.as_tensor(g)
    if g.shape[-3:] != otf.shape[-3:]:
        raise ValueError(
            f"g shape {tuple(g.shape)} does not match the transfer function {tuple(otf.shape)}; "
            f"z-padded reconstruction is not handled by the closed form yet "
            f"(use s.z_padding == 0, or pad g to the padded_shape)"
        )
    regularization = noise.variance() / s_k
    inverse_filter = tikhonov_regularized_inverse_filter(otf, regularization)
    g = g.to(torch.complex64)
    g_fft = torch.fft.fftn(g)
    return torch.real(torch.fft.ifftn(g_fft * inverse_filter))


# --------------------------------------------------------------------------- #
# Config adapter                                                              #
# --------------------------------------------------------------------------- #


def _value_or_free(value):
    """Map a config float or ``FreeParameter`` to a plain number or :func:`free`."""
    from waveorder.config.schema import FreeParameter

    if isinstance(value, FreeParameter):
        return free(value.nominal, value.tolerance)
    return float(value)


def build_system_from_config(cfg):
    """Build ``(system, obj, noise, prior, sampling)`` from a 4.0 config.

    A thin adapter from the ``imaging_system`` / ``object`` / ``noise`` /
    ``solver`` blocks of a :class:`waveorder.config.schema.V4Config` to the
    distribution modules of this file. A config value ``{nominal, tolerance}``
    (a :class:`waveorder.config.schema.FreeParameter`) becomes an
    :class:`torch.nn.Parameter`; a plain number becomes a buffer.

    Parameters
    ----------
    cfg : V4Config
        The configuration. Shift-invariant fluorescence or phase contrast.

    Returns
    -------
    system : ImagingSystem
    obj : ObjectPrior
    noise : Noise
    prior : ParameterPrior
    sampling : Sampling

    Raises
    ------
    NotImplementedError
        For tiled field dependence, non-gaussian noise with a spectral
        reconstruction, or object priors without a closed-form spectrum. These
        follow in later passes; the evidence path already restricts to the
        supported subset.
    """
    from waveorder.config.schema import (
        PriorTikhonovSpectral,
        TikhonovSpectralPrior,
    )

    imaging = cfg.imaging_system
    if imaging.field_dependence != "shift_invariant":
        raise NotImplementedError("build_system_from_config is shift-invariant for now; tiled follows")

    yx = cfg.data.pixel_size.yx
    yx_pixel_size = float(yx) if isinstance(yx, (int, float)) else float(yx.y)
    zyx_shape_source = cfg.simulate.shape if cfg.simulate is not None else None
    sampling_geometry = Sampling(
        zyx_shape=tuple(zyx_shape_source) if zyx_shape_source is not None else (0, 0, 0),
        yx_pixel_size=yx_pixel_size,
        z_pixel_size=float(cfg.data.pixel_size.z),
        z_padding=int(cfg.solver.padding.z),
    )

    detection = imaging.detection
    if imaging.contrast == "fluorescence":
        if detection.wavelength is None:
            raise ValueError("fluorescence imaging_system.detection.wavelength is required")
        zernike_coefficients: dict[str | int, object] = {}
        if detection.pupil_aberrations is not None:
            if detection.pupil_aberrations.zernike_up_to_order is not None:
                from waveorder.calibrate.zernike_names import noll_indices_up_to_order

                shorthand = detection.pupil_aberrations.zernike_up_to_order
                for noll in noll_indices_up_to_order(shorthand.order):
                    zernike_coefficients[noll] = free(0.0, shorthand.tolerance)
            for name, value in detection.pupil_aberrations.terms.items():
                zernike_coefficients[name] = _value_or_free(value)
        system: ImagingSystem = Fluorescence3D(
            wavelength_emission=detection.wavelength,
            index_of_refraction_media=imaging.index_of_refraction_media,
            numerical_aperture_detection=_value_or_free(detection.numerical_aperture),
            zernike_coefficients=zernike_coefficients,
            z_focus_offset=_value_or_free(detection.z_focus_offset),
        )
    elif imaging.contrast == "phase":
        illumination = imaging.illumination
        if illumination.wavelength is None:
            raise ValueError("phase imaging_system.illumination.wavelength is required")
        na_ill = illumination.numerical_aperture
        system = Phase3D(
            wavelength_illumination=illumination.wavelength,
            index_of_refraction_media=imaging.index_of_refraction_media,
            numerical_aperture_illumination=_value_or_free(na_ill) if na_ill is not None else 0.9,
            numerical_aperture_detection=_value_or_free(detection.numerical_aperture),
            tilt_angle_zenith=_value_or_free(illumination.tilt.zenith),
            tilt_angle_azimuth=_value_or_free(illumination.tilt.azimuth),
            invert_phase_contrast=illumination.invert_phase_contrast,
        )
    else:  # pragma: no cover - schema restricts contrast
        raise NotImplementedError(f"contrast {imaging.contrast!r} is not supported")

    # Noise.
    if cfg.noise.kind == "gaussian":
        gaussian = cfg.noise.gaussian
        sigma = gaussian.sigma if isinstance(gaussian.sigma, (int, float)) else 1.0
        noise: Noise = Gaussian(float(sigma))
    elif cfg.noise.kind == "poisson":
        poisson = cfg.noise.poisson
        noise = Poisson(gain=poisson.gain, offset=poisson.offset)
    else:
        raise NotImplementedError(f"noise kind {cfg.noise.kind!r} is not supported by the builder yet")

    # Object prior.
    prior_spec = cfg.object.prior
    weight = cfg.object.weight if isinstance(cfg.object.weight, (int, float)) else 1e-3
    if isinstance(prior_spec, PriorTikhonovSpectral) or isinstance(prior_spec, TikhonovSpectralPrior):
        # A fitted / explicit spectrum: deferred to FittedSpectrum (TODO).
        obj: ObjectPrior = FittedSpectrum()
    elif prior_spec in ("tikhonov", "tv", "none") or prior_spec is None:
        obj = Tikhonov(float(weight))
    else:
        obj = Tikhonov(float(weight))

    prior = ParameterPrior.from_system(system)
    return system, obj, noise, prior, sampling_geometry
