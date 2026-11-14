"""The optics parts the imaging systems are built from.

:class:`Detection` with its :class:`ZernikeAberration`, :class:`Illumination` with its :class:`IlluminationTilt`,
and the confocal pinhole's OTF, :func:`pinhole_otf`.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self, override

import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.nn.functional import complex_multiply, lateral_frequencies, zernike
from waveorder.nn.parameters import CalibratableModule, Quantity, expect_fixed, repr_parts, value_range
from waveorder.nn.sampling import Sampling
from waveorder.registry import register
from waveorder.zernike_names import canonical_name, name_to_noll

type PupilEdge = Literal["ramp", "sigmoid"]
"""How :class:`Detection` draws the aperture edge: ``"ramp"`` antialiased (default), ``"sigmoid"`` as in 3.x."""

SIGMOID_EDGE_STEEPNESS = 1e4
"""Slope of the ``"sigmoid"`` aperture edge, per cycle/um: 3.x ``generate_pupil``."""

_EDGE_STEPS = 4.0
"""Slope of the illumination cap's sigmoid edge, per frequency-grid step (3.x ``generate_tilted_pupil``)."""


@dataclass(frozen=True, slots=True)
class PupilGrid:
    """Lateral spatial frequencies of one transfer-function grid, in cycles per um, FFT order.

    ``spacing`` is the smaller of the two frequency steps, a Python float so it never syncs the device.
    """

    radius: Tensor
    azimuth: Tensor
    spacing: float

    @classmethod
    def from_sampling(cls, sampling: Sampling, *, device: torch.device, dtype: torch.dtype) -> Self:
        """Build the lateral frequency grid of ``sampling``."""
        fy, fx = lateral_frequencies(sampling, device=device, dtype=dtype)
        (_, ny, nx), (_, dy, dx) = sampling.shape, sampling.pixel_size
        return cls(
            radius=torch.hypot(fy[:, None], fx[None, :]),
            azimuth=torch.atan2(fy[:, None], fx[None, :]),
            spacing=min(1 / (ny * dy), 1 / (nx * dx)),
        )


@dataclass(frozen=True, slots=True)
class PupilCoordinates:
    """Unit-disk polar coordinates of a pupil: ``rho = 1`` at the aperture edge, ``theta = 0`` along +x."""

    rho: Tensor
    theta: Tensor


@register("zernike", kind="pupil_factor")
class ZernikeAberration(CalibratableModule):
    """Pupil phase ``exp(i 2 pi sum_j c_j Z_j)`` from named Zernike terms, coefficients in waves (RMS).

    ``Z_j`` are the orthonormal Zernike polynomials in Noll's order (``j = 4`` defocus, ``11`` spherical), so a
    coefficient is the RMS wavefront error of its term. ``rho = 1`` is the aperture edge at the current numerical
    aperture and ``theta`` runs from +x toward +y. A positive defocus coefficient ``c`` acts like a focus offset of
    about ``-4 sqrt(3) c n wavelength / NA^2`` um (paraxial). Each term is a quantity under its canonical name
    (:mod:`waveorder.zernike_names`), e.g. ``detection.pupil_aberrations.spherical``.

    Parameters
    ----------
    terms : Mapping[str, float or Free]
        Coefficient of each term by name; ``z<j>`` spellings are accepted and stored under the canonical name.

    Raises
    ------
    ParameterError
        For an unknown term name or one given twice (``path`` is the name).
    """

    def __init__(self, terms: Mapping[str, Quantity]) -> None:
        super().__init__()
        self.noll: dict[str, int] = {}
        for name, value in terms.items():
            try:
                key = canonical_name(name)
                noll = name_to_noll(key)
            except ValueError as error:
                raise ParameterError(str(error), path=name) from error
            if key in self.noll:
                raise ParameterError(f"Zernike term {key!r} is given twice", path=key)
            self.noll[key] = noll
            self.register_quantity(key, value)

    def profile(self, pupil: PupilCoordinates) -> Tensor:
        """Return the complex factor at every pupil point, stored as ``(Y, X, 2)`` real and imaginary parts."""
        waves = torch.zeros_like(pupil.rho)
        for name, j in self.noll.items():
            waves = waves + self.quantity(name) * zernike(j, pupil.rho, pupil.theta)
        phase = 2 * math.pi * waves
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


@register("detection", kind="detection")
class Detection(CalibratableModule):
    """Detection arm: the aperture cut-off ``NA / wavelength``, a focus offset, and the pupil aberrations.

    ``edge="ramp"`` (default) weights each frequency sample by the fraction of a one-sample ramp inside the cut-off,
    which gives the numerical aperture a gradient. ``edge="sigmoid"`` reproduces 3.x ``generate_pupil``,
    ``sigmoid(1e4 (NA / wavelength - radius))``, a hard edge: use it for parity with 3.x, not to calibrate the NA.

    Parameters
    ----------
    wavelength : float, optional
        Detected wavelength in vacuum, um; fixed. ``None`` lets a phase system give it the illumination's
        wavelength (:meth:`inherit_wavelength`); the fluorescence systems require it.
    numerical_aperture : float or Free
        Detection numerical aperture.
    z_focus_offset : float or Free
        Focal plane relative to the stack's reference slice, in slices (multiply by the z step for um); positive moves
        focus toward higher z. For a 3D object it translates the estimate along z, which leaves the data statistics
        unchanged, so give it as a number: :func:`~waveorder.nn.calibrate` warns if it is free. For a thin object the
        object plane is in focus at slice ``Z // 2 + z_focus_offset``, and it can be calibrated.
    edge : {"ramp", "sigmoid"}
        Aperture edge (see above).
    pupil_aberrations : ZernikeAberration, optional
        Aberrations multiplied into the pupil, named ``detection.pupil_aberrations.<term>``; ``None`` for none.

    Raises
    ------
    ParameterError
        For a ``wavelength`` that is not ``None`` or positive and finite, an unknown ``edge``, or an aperture that
        is not positive; ``path`` is the argument.
    """

    numerical_aperture: Tensor
    z_focus_offset: Tensor

    def __init__(
        self,
        *,
        wavelength: float | None = None,
        numerical_aperture: Quantity,
        z_focus_offset: Quantity = 0.0,
        edge: PupilEdge = "ramp",
        pupil_aberrations: ZernikeAberration | None = None,
    ) -> None:
        super().__init__()
        self._wavelength: float | None = None
        if wavelength is not None:
            self.inherit_wavelength(wavelength)
        if edge not in ("ramp", "sigmoid"):
            raise ParameterError(f"Detection.edge must be 'ramp' or 'sigmoid', got {edge!r}", path="edge")
        self.edge: PupilEdge = edge
        self.register_quantity("numerical_aperture", numerical_aperture, positive=True)
        self.register_quantity("z_focus_offset", z_focus_offset)
        self.numerical_aperture_range = value_range(numerical_aperture)
        self.z_focus_offset_range = value_range(z_focus_offset)
        self.pupil_aberrations = pupil_aberrations

    @property
    def has_wavelength(self) -> bool:
        """Whether a wavelength was given, or inherited from a system since."""
        return self._wavelength is not None

    @property
    def wavelength(self) -> float:
        """The detected wavelength in vacuum, um.

        Raises
        ------
        ParameterError
            While the detection has none (``path="wavelength"``): it was built with ``wavelength=None`` and no
            phase system has given it the illumination's yet.
        """
        if self._wavelength is None:
            raise ParameterError(
                "Detection has no wavelength: pass Detection(wavelength=...) in um, or build a phase system "
                "(ThickPhase3D, ThinPhase2D) around it, which gives it the illumination's",
                path="wavelength",
            )
        return self._wavelength

    def inherit_wavelength(self, wavelength: float) -> None:
        """Set the detected wavelength, in um, of a detection built without one.

        The phase systems call this with the illumination's wavelength; a detection shared with another system must
        agree with that system's illumination.

        Raises
        ------
        ParameterError
            If ``wavelength`` is not positive and finite (or is free), or the detection already has another one;
            ``path`` is ``"wavelength"``.
        """
        expect_fixed(wavelength, what="Detection wavelength", path="wavelength")
        if self._wavelength is not None and self._wavelength != wavelength:
            raise ParameterError(
                f"Detection already has wavelength = {self._wavelength} um, not {wavelength} um", path="wavelength"
            )
        self._wavelength = wavelength

    @override
    def extra_repr(self) -> str:
        """Return the wavelength (``None`` until given or inherited), the quantities and the edge."""
        wavelength = "None" if self._wavelength is None else f"{self._wavelength:g}"
        return repr_parts(f"wavelength={wavelength}", super().extra_repr(), f"edge={self.edge!r}")

    @override
    def current_argument(self, name: str, given: object) -> object:
        """Return ``wavelength`` as held now (inherited from a phase system when built without one), else the base's."""
        if name == "wavelength" and given is None and self.has_wavelength:
            return self.wavelength
        return super().current_argument(name, given)

    def pupil(self, grid: PupilGrid) -> Tensor:
        """Return the complex detection pupil on ``grid``, stored as ``(Y, X, 2)`` real and imaginary parts."""
        cutoff = self.numerical_aperture / self.wavelength
        if self.edge == "sigmoid":
            aperture = torch.sigmoid(SIGMOID_EDGE_STEEPNESS * (cutoff - grid.radius))
        else:
            aperture = torch.clamp((cutoff - grid.radius) / grid.spacing + 0.5, min=0.0, max=1.0)
        pupil = torch.stack((aperture, torch.zeros_like(aperture)), dim=-1)
        coordinates = PupilCoordinates(rho=grid.radius / cutoff, theta=grid.azimuth)
        if self.pupil_aberrations is not None:
            pupil = complex_multiply(pupil, self.pupil_aberrations.profile(coordinates))
        return pupil


@register("illumination_tilt", kind="tilt")
class IlluminationTilt(CalibratableModule):
    """Direction of the illumination axis, in radians.

    Parameters
    ----------
    zenith : float or Free
        Angle between the illumination axis and +z inside the immersion medium, radians, in ``(-pi/2, pi/2)``; the
        angle in air is ``asin(n sin(zenith))``. A free zenith needs bounds, so the system can check that unscattered
        light stays inside the detection aperture.
    azimuth : float or Free
        Angle of the tilt in the xy-plane from +x toward +y, radians. ``ThickPhase3D`` and ``ThinPhase2D`` take
        opposite z directions (as in 3.x), so azimuth ``a`` in one is ``a + pi`` in the other: calibrate it with the
        system that will use it.
    """

    zenith: Tensor
    azimuth: Tensor

    def __init__(self, *, zenith: Quantity = 0.0, azimuth: Quantity = 0.0) -> None:
        super().__init__()
        self.register_quantity("zenith", zenith)
        self.register_quantity("azimuth", azimuth)
        zenith_range = value_range(zenith)
        if zenith_range is None:
            raise ParameterError(
                f"illumination.tilt.zenith must have bounds when free, got {zenith}", path="illumination.tilt.zenith"
            )
        low, high = zenith_range
        if not (-math.pi / 2 < low and high < math.pi / 2):
            raise ParameterError(
                f"illumination.tilt.zenith must stay inside (-pi/2, pi/2) radians, got {zenith}",
                path="illumination.tilt.zenith",
            )
        self.largest_zenith = max(abs(low), abs(high))


@register("illumination", kind="illumination")
class Illumination(CalibratableModule):
    """Transmitted brightfield illumination: a uniform cone of half-angle ``asin(NA / n)`` around a tilted axis.

    The source is the cap of the Ewald sphere (radius ``n / wavelength``) within that cone, projected onto the
    lateral frequency plane, with a sigmoid edge about one grid step wide. 3.x: ``generate_tilted_pupil``.

    Parameters
    ----------
    wavelength : float
        Illumination wavelength in vacuum, um. Fixed, because it sets the frequency grid.
    numerical_aperture : float or Free
        Condenser numerical aperture, at most the medium's refractive index.
    tilt : IlluminationTilt, optional
        Direction of the illumination axis; on axis by default. Its quantities are named
        ``illumination.tilt.zenith`` and ``illumination.tilt.azimuth``.

    Raises
    ------
    ParameterError
        If ``wavelength`` is not a positive finite number, or the numerical aperture is not positive or a free one
        has no bounds; ``path`` names the field.
    """

    numerical_aperture: Tensor
    tilt: IlluminationTilt

    def __init__(
        self, *, wavelength: float, numerical_aperture: Quantity, tilt: IlluminationTilt | None = None
    ) -> None:
        super().__init__()

        expect_fixed(wavelength, what="illumination.wavelength", path="illumination.wavelength")

        self.wavelength = wavelength
        self.register_quantity("numerical_aperture", numerical_aperture, positive=True)
        na_range = value_range(numerical_aperture)
        if na_range is None:  # positive=True already requires bounds on a free aperture
            raise ParameterError(
                f"illumination.numerical_aperture must have bounds when free, got {numerical_aperture}",
                path="illumination.numerical_aperture",
            )
        self.numerical_aperture_range = na_range
        self.tilt = IlluminationTilt() if tilt is None else tilt

    @override
    def extra_repr(self) -> str:
        """Return the wavelength, then the quantities."""
        return repr_parts(f"wavelength={self.wavelength:g}", super().extra_repr())

    @override
    def current_argument(self, name: str, given: object) -> object:
        """Return ``tilt`` as the tilt module held now (a zero tilt when built without one), else the base's."""
        return self.tilt if name == "tilt" and given is None else super().current_argument(name, given)

    def source(self, fy: Tensor, fx: Tensor, *, index_of_refraction_media: float, spacing: float) -> Tensor:
        """Return the real source intensity on a lateral frequency grid.

        Parameters
        ----------
        fy, fx : Tensor
            Lateral frequencies in cycles per um, broadcastable to ``(Y, X)``, FFT order.
        index_of_refraction_media : float
            Refractive index of the medium.
        spacing : float
            Frequency-grid step in cycles per um (the finer of the two axes); sets the edge width.

        Returns
        -------
        Tensor
            Source weights in ``[0, 1]``, shape ``(Y, X)``.
        """
        radius = index_of_refraction_media / self.wavelength
        cone = torch.sqrt(torch.clamp(1 - (self.numerical_aperture / index_of_refraction_media) ** 2, min=0.0))
        fz_squared = radius**2 - fx**2 - fy**2
        inside = (fz_squared >= 0).to(fz_squared.dtype)
        fz = torch.sqrt(torch.clamp(fz_squared, min=0.0))
        zenith, azimuth = self.tilt.zenith, self.tilt.azimuth
        lateral = torch.sin(zenith)
        along = fx * lateral * torch.cos(azimuth) + fy * lateral * torch.sin(azimuth) + fz * torch.cos(zenith)
        return torch.sigmoid(_EDGE_STEPS / spacing * (along - radius * cone)) * inside


def pinhole_otf(diameter: Tensor, sampling: Sampling) -> Tensor:
    """Return a pinhole's lateral OTF ``(2 J1(x) / x)^2`` with ``x = pi d k`` on a lateral half-spectrum grid.

    As in 3.x, ``d`` is an effective diameter: about ``1 / sqrt(2)`` (0.71) times the diameter of the hard pinhole
    it stands for.

    Parameters
    ----------
    diameter : Tensor
        Effective pinhole diameter ``d`` in um, in the object plane; a scalar tensor, whose device and dtype the grid
        takes.
    sampling : Sampling
        Grid whose lateral frequencies ``k`` (cycles per um) the OTF is sampled on.

    Returns
    -------
    Tensor
        Real OTF, 1 at zero frequency, shape ``(Y, X // 2 + 1)``.
    """
    _, ny, nx = sampling.shape
    _, dy, dx = sampling.pixel_size
    fy = torch.fft.fftfreq(ny, d=dy, device=diameter.device, dtype=diameter.dtype)
    fx = torch.fft.rfftfreq(nx, d=dx, device=diameter.device, dtype=diameter.dtype)
    x = math.pi * diameter * torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    small = x < 1e-6
    safe = torch.where(small, torch.ones_like(x), x)
    # 2 J1(x) / x -> 1 - x^2 / 8 as x -> 0; the safe branch keeps the gradient finite at x = 0.
    jinc = torch.where(small, 1 - x**2 / 8, 2 * torch.special.bessel_j1(safe) / safe)
    return jinc**2
