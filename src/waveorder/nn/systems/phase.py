"""Brightfield phase contrast: weak-object transfer functions of partially coherent, possibly tilted illumination.

:class:`ThickPhase3D` images a 3D object as a 3D stack (3.x ``phase_thick_3d``); :class:`ThinPhase2D` images one
plane at several focal planes (3.x ``isotropic_thin_3d``). Both are first order in the object (first Born
approximation), so they hold while the contrast ``|I / mean(I) - 1|`` is well below 1, and both measure one channel of
that contrast (:func:`~waveorder.nn.preprocess.brightfield_contrast`). ``Detection(edge="sigmoid")`` reproduces 3.x
transfer functions to rounding.
"""

import math
from typing import Literal, override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.functional import (
    abs2,
    axial_nyquist,
    axial_spectrum,
    complex_multiply,
    crop_lateral,
    lateral_frequencies,
    plane_slabs,
    transverse_nyquist,
)
from waveorder.nn.operators import FourierDiagonal, LinearOperator, ReshapedFourierDiagonal
from waveorder.nn.optics import Detection, Illumination, PupilEdge, PupilGrid
from waveorder.nn.parameters import Quantity, evaluate, repr_parts
from waveorder.nn.sampling import ZYX, HalfTransfer, Sampling
from waveorder.nn.systems._checks import (
    expect_feasible_grid,
    expect_index,
    expect_resolved_aperture,
    expect_z_positions,
)
from waveorder.nn.systems.base import CacheMode, ImagingSystem, ShiftInvariantSystem
from waveorder.registry import register


def _check_brightfield(
    illumination: Illumination, detection: Detection, *, index_of_refraction_media: float, where: str
) -> None:
    """Reject optics whose unscattered light leaves the detection or apertures beyond ``n``; fill in the wavelength."""
    if not detection.has_wavelength:
        detection.inherit_wavelength(illumination.wavelength)
    expect_index(index_of_refraction_media, where=where)
    if illumination.wavelength != detection.wavelength:
        raise ParameterError(
            f"detection.wavelength = {detection.wavelength} um must equal illumination.wavelength = "
            f"{illumination.wavelength} um: brightfield detects the illumination's own light",
            path="detection.wavelength",
        )
    n = index_of_refraction_media
    na_ill = illumination.numerical_aperture_range[1]
    if na_ill > n:
        raise ParameterError(
            f"illumination.numerical_aperture can reach {na_ill}, above index_of_refraction_media = {n}",
            path="illumination.numerical_aperture",
        )
    na_range = detection.numerical_aperture_range
    if na_range is None or na_range[1] > n:
        raise ParameterError(
            f"detection.numerical_aperture can reach {None if na_range is None else na_range[1]}, above "
            f"index_of_refraction_media = {n}",
            path="detection.numerical_aperture",
        )
    axis = n * math.sin(illumination.tilt.largest_zenith)
    if axis >= na_range[0]:
        raise ParameterError(
            f"illumination.tilt.zenith can reach {illumination.tilt.largest_zenith} rad, where the illumination axis "
            f"n sin(zenith) = {axis:.4g} is outside the detection aperture (detection.numerical_aperture can be "
            f"{na_range[0]}); brightfield needs the unscattered light inside it",
            path="illumination.tilt.zenith",
        )


def _oversampling(sampling: Sampling, *, lateral_step: float, axial_step: float) -> ZYX[int]:
    dz, dy, dx = sampling.pixel_size
    return (math.ceil(dz / axial_step), math.ceil(dy / lateral_step), math.ceil(dx / lateral_step))


def _stored_pupil(detection: Detection, grid: PupilGrid) -> tuple[Tensor, Tensor]:
    """Return the detection pupil stored ``(Y, X, 2)`` and its aperture ``|P|``, ``(Y, X)``."""
    # vector_norm's gradient is zero where the pupil is zero; that of sqrt(re^2 + im^2) is not finite there.
    pupil = detection.pupil(grid)
    return pupil, torch.linalg.vector_norm(pupil, dim=-1)


def _defocus(oblique: Tensor, z: Tensor) -> Tensor:
    """``exp(i 2 pi z k_z)`` per plane, stored ``(Z, Y, X, 2)``."""
    phase = 2 * math.pi * z[:, None, None] * oblique
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def _conjugate(values: Tensor) -> Tensor:
    """Return the complex conjugate of values stored ``(..., 2)``."""
    real, imaginary = values.unbind(-1)
    return torch.stack((real, -imaginary), dim=-1)


def _times_i(values: Tensor) -> Tensor:
    """``i z`` for values ``z`` stored ``(..., 2)``."""
    real, imaginary = values.unbind(-1)
    return torch.stack((-imaginary, real), dim=-1)


def _fft(values: Tensor, *, dim: tuple[int, ...], inverse: bool = False) -> Tensor:
    """Return the (inverse) FFT over ``dim`` of the complex view of values stored ``(..., 2)``, stored alike."""
    # The inverse runs unscaled and divides here: torch's MKL single-precision scaled inverse applies 1 / N twice for a
    # 2048 x 2048 grid on more than two CPU threads.
    spectrum = torch.view_as_complex(values.contiguous())
    if not inverse:
        return torch.view_as_real(torch.fft.fftn(spectrum, dim=dim))
    points = math.prod(spectrum.shape[axis] for axis in dim)
    return torch.view_as_real(torch.fft.ifftn(spectrum, dim=dim, norm="forward")) / points


def _greens_denominator(oblique: Tensor, support: Tensor, edge: PupilEdge) -> Tensor:
    """Return the ``k_z`` the Green's function ``-i exp(i 2 pi z k_z) / (4 pi k_z)`` divides by, made finite."""
    # 3.x's k_z + (1 - support) + 1e-15 would bias the antialiased ramp's edge ring by about 1%, so the ramp divides by
    # k_z itself, and by 1 beyond the Ewald sphere where the pupil is zero. The sigmoid edge keeps the 3.x form.
    if edge == "sigmoid":
        return oblique + (1 - support) + 1e-15
    return torch.where(oblique > 0, oblique, torch.ones_like(oblique))


def _half(full: Tensor, nx: int) -> Tensor:
    """Keep the half-spectrum columns of an ``nx``-wide grid from a full spectrum stored ``(..., X, 2)``.

    For even ``nx`` the Nyquist column is taken at ``-nx / 2``, as 3.x's crop did.
    """
    columns = torch.arange(nx // 2 + 1, device=full.device)
    if nx % 2 == 0:
        columns[-1] = full.shape[-2] - nx // 2
    return full.index_select(-2, columns)


def _crop_full(full: Tensor, shape: ZYX[int]) -> Tensor:
    """Keep the lateral half-spectrum frequencies of a ``shape`` grid from full spectra ``(..., P, Y', X', 2)``."""
    return crop_lateral(_half(full, shape[2]), shape)


@register("thick_phase_3d", kind="system")
class ThickPhase3D(ShiftInvariantSystem):
    """Brightfield phase contrast of a thick object: the 3D weak-object transfer function (3.x ``phase_thick_3d``).

    The fields are scattering potentials in um^-2. ``phase`` is ``(2 pi / wavelength)^2 (n_object^2 - n_media^2)``;
    times :meth:`radians_per_voxel` it is the phase delay of one voxel in radians. ``absorption`` (with
    ``absorption="independent"``) is the imaginary part; times the same factor it is the log of the amplitude
    transmittance of one voxel, negative where the object absorbs. The data are one channel of normalized contrast
    over the whole stack, ``(1, Z, Y, X)``.

    Parameters
    ----------
    illumination : Illumination
        Condenser aperture and tilt; its wavelength is the imaging wavelength.
    detection : Detection
        Detection arm. Leave out its wavelength to take the illumination's; a different one is refused. Its
        ``z_focus_offset`` (slices) translates the estimate along z.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    invert_phase_contrast : bool
        Negate the defocus of every plane, for stacks acquired in the opposite z direction.
    absorption : float, Free or "independent"
        Absorption per unit phase (3.x ``absorption_ratio``, default 0: a pure phase object), so that
        ``H = H_phase + absorption * H_absorption``; or ``"independent"`` for a second object field.
    cache : {"off", "auto"}
        See :class:`~waveorder.nn.ShiftInvariantSystem`.

    Raises
    ------
    ParameterError
        If ``index_of_refraction_media`` is not a positive finite number, or the optics fail the brightfield checks
        (equal wavelengths, apertures within the medium, illumination axis inside the detection aperture).

    Notes
    -----
    Let ``u`` be a lateral frequency (cycles/um), ``kappa(u) = sqrt((n / wavelength)^2 - |u|^2)``, ``S(u)`` the
    source, ``P(u)`` the detection pupil, ``D_z(u) = P(u) |P(u)| exp(2 pi i z kappa(u))`` and
    ``G_z(u) = -i D_z(u) / (4 pi kappa(u))``. With sums over the lateral frequency grid::

        c1(q, z) = sum_u S(u) conj(D_z(u)) G_z(u + q)
        c2(q, z) = sum_u S(u) D_z(u) conj(G_z(u - q))
        B        = sum_u S(u) |P(u)|^2

        H_phase(q, q_z)      = (dz / B) FFT_z[w(z) (c1 + c2)]
        H_absorption(q, q_z) = (dz / B) FFT_z[w(z) i (c1 - c2)]

    on a grid oversampled along z, y and x for the largest reachable apertures, then cropped to the data's
    frequencies. ``z = k dz + z_focus_offset dz_data`` for plane ``k`` in FFT order, and ``w`` is a Hann window
    centred on ``k = 0``, as in 3.x. The oversampling ignores an illumination tilt, so a tilted transfer function
    aliases, as in 3.x.
    """

    illumination: Illumination
    detection: Detection

    def __init__(
        self,
        illumination: Illumination,
        detection: Detection,
        *,
        index_of_refraction_media: float,
        invert_phase_contrast: bool = False,
        absorption: Quantity | Literal["independent"] = 0.0,
        cache: CacheMode = "off",
    ) -> None:
        fields = ("phase", "absorption") if absorption == "independent" else ("phase",)
        super().__init__(fields=fields, channel_count=1, cache=cache)
        _check_brightfield(
            illumination, detection, index_of_refraction_media=index_of_refraction_media, where="ThickPhase3D"
        )
        self.illumination = illumination
        self.detection = detection
        self.index_of_refraction_media = index_of_refraction_media
        self.invert_phase_contrast = invert_phase_contrast
        if absorption != "independent":
            self.register_quantity("absorption_ratio", absorption)

    @override
    def extra_repr(self) -> str:
        """Return the medium's index, the inversion when set, then the quantities or the independent absorption."""
        settings = [
            f"index_of_refraction_media={self.index_of_refraction_media:g}",
            "invert_phase_contrast=True" if self.invert_phase_contrast else "",
            super().extra_repr(),
            'absorption="independent"' if "absorption" in self.fields else "",
        ]
        return repr_parts(*settings)

    @override
    def check_sampling(self, sampling: Sampling) -> None:
        """Reject a focus offset beyond half the stack (the defocus would wrap) and a grid that cannot hold the optics.

        Raises
        ------
        IncompatibleModelError
            If the offset can leave the stack, the cut-off ``NA / wavelength`` is below one lateral frequency step
            (a wavelength in nm), or the transfer function grid is too large (a pixel size in nm).
        """
        expect_resolved_aperture(
            sampling,
            numerical_aperture=_upper(self.detection),
            wavelength=self.detection.wavelength,
            where="ThickPhase3D",
        )
        expect_feasible_grid(sampling, self.upsampling(sampling), where="ThickPhase3D")
        offsets = self.detection.z_focus_offset_range
        half_stack = sampling.shape[0] / 2
        if offsets is not None and max(abs(offsets[0]), abs(offsets[1])) > half_stack:
            raise IncompatibleModelError(
                f"detection.z_focus_offset can reach {offsets} slices, outside the +/-{half_stack} slices of a "
                f"{sampling.shape[0]}-slice stack; the defocus would wrap around"
            )

    @override
    def undetermined_by_likelihood(self) -> tuple[str, ...]:
        """Return the focus offset: for a 3D object it only translates the point-spread function along z."""
        return ("detection.z_focus_offset",)

    def upsampling(self, sampling: Sampling) -> ZYX[int]:
        """Return the per-axis oversampling factors ``(z, y, x)``: 3.x's, for the largest reachable apertures."""
        wavelength = self.illumination.wavelength
        na_ill = self.illumination.numerical_aperture_range[1]
        na_det = _upper(self.detection)
        return _oversampling(
            sampling,
            lateral_step=transverse_nyquist(wavelength, na_ill + na_det),
            axial_step=axial_nyquist(wavelength, na_det, self.index_of_refraction_media),
        )

    def radians_per_voxel(self, sampling: Sampling) -> float:
        """Return the factor that turns the ``phase`` field into radians of phase delay per voxel on ``sampling``.

        ``estimate[0] * system.radians_per_voxel(sampling)`` is the phase each voxel adds; its sum over z is the
        unit of a :class:`ThinPhase2D` phase. The ``absorption`` field takes the same factor.

        Parameters
        ----------
        sampling : Sampling
            The grid of the estimate; only its z step enters.

        Returns
        -------
        float
            ``dz * wavelength / (4 pi n_media)`` in um^2.
        """
        dz = sampling.pixel_size[0]
        return dz * self.illumination.wavelength / (4 * math.pi * self.index_of_refraction_media)

    def potential_transfers(self, sampling: Sampling) -> Tensor:
        """Return the 3.x real- and imaginary-potential transfer functions on ``sampling``, before projection.

        Returns
        -------
        Tensor
            Half spectra stored real, shape ``(2, Z, Y, X // 2 + 1, 2)``: ``[H_phase, H_absorption]``.
        """
        reference = self.detection.numerical_aperture
        dtype, device = reference.dtype, reference.device
        fine = sampling.upsampled(self.upsampling(sampling))
        grid = PupilGrid.from_sampling(fine, device=device, dtype=dtype)
        fy, fx = lateral_frequencies(fine, device=device, dtype=dtype)
        pupil, support = _stored_pupil(self.detection, grid)
        source = self.illumination.source(
            fy[:, None], fx[None, :], index_of_refraction_media=self.index_of_refraction_media, spacing=grid.spacing
        )
        medium_wavelength = self.illumination.wavelength / self.index_of_refraction_media
        oblique = torch.sqrt(torch.clamp(1 - medium_wavelength**2 * grid.radius**2, min=0.0)) / medium_wavelength
        nz, dz = fine.shape[0], fine.pixel_size[0]
        planes = torch.fft.fftfreq(nz, d=1 / nz, device=device, dtype=dtype)
        z = planes * dz + self.detection.z_focus_offset * sampling.pixel_size[0]
        if self.invert_phase_contrast:
            z = -z
        denominator = (-4 * math.pi * _greens_denominator(oblique, support, self.detection.edge))[..., None]
        illumination = source[..., None] * pupil
        window = torch.fft.ifftshift(torch.hann_window(nz, periodic=False, dtype=dtype, device=device))
        scale = dz / (source * abs2(pupil)).sum()
        shape, lateral = sampling.shape, (-2, -1)  # dims of the complex view, without the (real, imaginary) axis

        def slab(part: Tensor, weight: Tensor) -> Tensor:
            """Return ``c1 + c2`` and ``i (c1 - c2)`` on the planes ``part``, windowed, scaled and cropped laterally."""
            propagation = support[..., None] * _defocus(oblique, part)
            # The Green's function -i exp(i 2 pi z k_z) / (4 pi k_z), on the pupil support.
            greens = _times_i(propagation / denominator)
            illuminated = _fft(complex_multiply(illumination, propagation), dim=lateral)
            scattered = _fft(complex_multiply(pupil, greens), dim=lateral)
            del propagation, greens
            first = _fft(complex_multiply(_conjugate(illuminated), scattered), dim=lateral, inverse=True)
            first = _crop_full(first, shape)
            second = _fft(complex_multiply(illuminated, _conjugate(scattered)), dim=lateral, inverse=True)
            second = _crop_full(second, shape)
            del illuminated, scattered
            weight = (weight * scale)[:, None, None, None]
            return torch.stack(((first + second) * weight, _times_i(first - second) * weight))

        # Everything but the transform along z is per plane, so it runs in slabs of planes. The window, the transform
        # along z and the scale act on each lateral frequency alone, so cropping to the data grid's lateral frequencies
        # right after the lateral transforms gives the same result as cropping after the transform along z.
        plane = oblique.numel()
        slabs = zip(plane_slabs(z, plane), plane_slabs(window, plane), strict=True)
        return axial_spectrum([slab(part, weight) for part, weight in slabs], shape)

    @override
    def compute_transfer(self, sampling: Sampling) -> HalfTransfer:
        phase, absorption = self.potential_transfers(sampling).unbind(0)
        if "absorption" in self.fields:
            return torch.stack((phase, absorption))[None]
        return (phase + self.quantity("absorption_ratio") * absorption)[None, None]


def _upper(detection: Detection) -> float:
    na_range = detection.numerical_aperture_range
    if na_range is None:  # Detection requires bounds on a free aperture; kept for the type checker
        raise ParameterError(
            "detection.numerical_aperture must have bounds when free", path="detection.numerical_aperture"
        )
    return na_range[1]


@register("thin_phase_2d", kind="system")
class ThinPhase2D(ImagingSystem):
    """Brightfield phase contrast of a thin object from a defocus stack (3.x ``isotropic_thin_3d``).

    The object is one plane, ``(2, 1, Y, X)``, with amplitude transmittance ``exp(absorption + i phase)``: ``phase``
    in radians, ``absorption`` the log of the amplitude transmittance, negative where the object absorbs. The data are
    one channel at ``Z`` focal planes, ``(1, Z, Y, X)``, each plane normalized by its own mean
    (``brightfield_contrast(..., per_plane=True)``).

    Plane ``i`` has defocus ``(Z // 2 + z_focus_offset - i) * dz``, as in 3.x. ``phase`` is positive where the
    object delays the light when slices before the focused one have their focal plane on the objective's side of the
    object; for a stack acquired the other way, set ``invert_phase_contrast``.

    Parameters
    ----------
    illumination : Illumination
        Condenser aperture and tilt; its numerical aperture must not exceed the detection's. Where it reaches it, the
        source is computed at 0.9 times the detection aperture, as in 3.x.
    detection : Detection
        Detection arm. Leave out its wavelength to take the illumination's; a different one is refused.
    index_of_refraction_media : float
        Refractive index of the medium.
    invert_phase_contrast : bool
        Negate the defocus of every plane.
    z_positions : tuple[float, ...], optional
        Defocus of each plane in um, in data order, before the focus offset; replaces ``(Z // 2 - i) * dz``.

    Raises
    ------
    ParameterError
        As :class:`ThickPhase3D`, or if the illumination aperture can exceed the detection's, or ``z_positions`` is
        not a nonempty tuple of finite numbers.

    Notes
    -----
    With ``u``, ``S``, ``P`` and ``D_z`` as in :class:`ThickPhase3D`, and sums over the lateral frequency grid::

        h1(q, z) = sum_u S(u) conj(D_z(u)) D_z(u + q)
        h2(q, z) = sum_u S(u) D_z(u) conj(D_z(u - q))
        B_z      = sum_u S(u) |D_z(u)|^2

        H_phase(q; z)      = i (h1 - h2) / B_z
        H_absorption(q; z) = (h1 + h2) / B_z

    and the plane at defocus ``z`` is ``FFT(contrast_z) = H_phase FFT(phase) + H_absorption FFT(absorption)``. Only y
    and x are oversampled, as in 3.x.
    """

    operator_type: type[LinearOperator] = ReshapedFourierDiagonal
    illumination: Illumination
    detection: Detection

    def __init__(
        self,
        illumination: Illumination,
        detection: Detection,
        *,
        index_of_refraction_media: float,
        invert_phase_contrast: bool = False,
        z_positions: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(fields=("phase", "absorption"), channel_count=1)
        _check_brightfield(
            illumination, detection, index_of_refraction_media=index_of_refraction_media, where="ThinPhase2D"
        )
        na_ill, na_det = illumination.numerical_aperture_range[1], _lower(detection)
        if na_ill > na_det:
            raise ParameterError(
                f"illumination.numerical_aperture can reach {na_ill}, above detection.numerical_aperture = {na_det}",
                path="illumination.numerical_aperture",
            )
        self.clamps_source = na_ill == na_det
        expect_z_positions(z_positions, where="ThinPhase2D")
        self.illumination = illumination
        self.detection = detection
        self.index_of_refraction_media = index_of_refraction_media
        self.invert_phase_contrast = invert_phase_contrast
        self.z_positions = z_positions

    @override
    def extra_repr(self) -> str:
        """Return the medium's index, the inversion when set and the plane positions when given."""
        settings = [
            f"index_of_refraction_media={self.index_of_refraction_media:g}",
            "invert_phase_contrast=True" if self.invert_phase_contrast else "",
            "" if self.z_positions is None else f"z_positions={self.z_positions!r}",
            super().extra_repr(),
        ]
        return repr_parts(*settings)

    @override
    def object_shape(self, sampling: Sampling) -> tuple[int, ...]:
        """Return ``(F, 1, Y, X)``: one plane per field."""
        return (len(self.fields), 1, *sampling.shape[1:])

    @override
    def check_sampling(self, sampling: Sampling) -> None:
        """Reject a stack whose plane count differs from ``z_positions`` and a grid that cannot hold the optics."""
        expect_resolved_aperture(
            sampling,
            numerical_aperture=_upper(self.detection),
            wavelength=self.detection.wavelength,
            where="ThinPhase2D",
        )
        expect_feasible_grid(sampling, self.upsampling(sampling), where="ThinPhase2D")
        if self.z_positions is not None and len(self.z_positions) != sampling.shape[0]:
            raise IncompatibleModelError(
                f"z_positions lists {len(self.z_positions)} planes, the data have {sampling.shape[0]}"
            )

    def upsampling(self, sampling: Sampling) -> ZYX[int]:
        """Return the lateral oversampling factors (z is never oversampled): 3.x's, for the largest apertures."""
        wavelength = self.illumination.wavelength
        step = transverse_nyquist(wavelength, self.illumination.numerical_aperture_range[1] + _upper(self.detection))
        _, ky, kx = _oversampling(sampling, lateral_step=step, axial_step=math.inf)
        return (1, ky, kx)

    def plane_positions(self, sampling: Sampling) -> Tensor:
        """Return the defocus of each data plane in um, shape ``(Z,)``, data order, signs as imaged."""
        reference = self.detection.numerical_aperture
        nz, dz = sampling.shape[0], sampling.pixel_size[0]
        if self.z_positions is None:
            base = (nz // 2 - torch.arange(nz, device=reference.device, dtype=reference.dtype)) * dz
        else:
            base = torch.tensor(self.z_positions, device=reference.device, dtype=reference.dtype)
        z = base + self.detection.z_focus_offset * dz
        return -z if self.invert_phase_contrast else z

    def plane_transfers(self, sampling: Sampling) -> Tensor:
        """Return the 3.x transfer functions of every plane, before projection.

        Returns
        -------
        Tensor
            Half spectra stored real, shape ``(Z, 2, 1, Y, X // 2 + 1, 2)``: ``[H_phase, H_absorption]`` per plane.
        """
        self.check_sampling(sampling)
        reference = self.detection.numerical_aperture
        dtype, device = reference.dtype, reference.device
        _, ny, nx = sampling.shape
        fine = Sampling(shape=(1, ny, nx), pixel_size=sampling.pixel_size).upsampled(self.upsampling(sampling))
        grid = PupilGrid.from_sampling(fine, device=device, dtype=dtype)
        fy, fx = lateral_frequencies(fine, device=device, dtype=dtype)
        pupil, support = _stored_pupil(self.detection, grid)

        def source_of(illumination: Illumination) -> Tensor:
            return illumination.source(
                fy[:, None], fx[None, :], index_of_refraction_media=self.index_of_refraction_media, spacing=grid.spacing
            )

        if self.clamps_source:  # 3.x: an illumination aperture reaching the detection's acts as 0.9 of it
            na_ill, na_det = self.illumination.numerical_aperture, self.detection.numerical_aperture
            aperture = torch.where(na_ill >= na_det, 0.9 * na_det, na_ill)
            source = evaluate(source_of, self.illumination, {"numerical_aperture": aperture})
        else:
            source = source_of(self.illumination)
        medium_wavelength = self.illumination.wavelength / self.index_of_refraction_media
        oblique = torch.sqrt(torch.clamp(1 - medium_wavelength**2 * grid.radius**2, min=0.0)) / medium_wavelength
        lateral = (-2, -1)
        aperture_pupil = support[..., None] * pupil

        def slab(part: Tensor) -> Tensor:
            """Return the transfers of the planes ``part``, cropped laterally: ``(P, 2, Y, X // 2 + 1, 2)``."""
            detected = complex_multiply(aperture_pupil, _defocus(oblique, part))  # (P, Y', X', 2)
            illuminated = _fft(source[..., None] * detected, dim=lateral)
            transmitted = _fft(detected, dim=lateral)
            direct = (source * abs2(detected)).sum(dim=(-2, -1), keepdim=True)[..., None]
            del detected
            first = _fft(complex_multiply(_conjugate(illuminated), transmitted), dim=lateral, inverse=True)
            first = _crop_full(first, sampling.shape)
            second = _fft(complex_multiply(illuminated, _conjugate(transmitted)), dim=lateral, inverse=True)
            second = _crop_full(second, sampling.shape)
            return torch.stack((_times_i(first - second) / direct, (first + second) / direct), dim=1)

        # In slabs of planes, each cropped at once: the planes are independent.
        slabs = plane_slabs(self.plane_positions(sampling), oblique.numel())
        return torch.cat([slab(part) for part in slabs]).unsqueeze(2)  # (Z, 2, 1, Y, X // 2 + 1, 2)

    @override
    def compute_operator(self, sampling: Sampling) -> ReshapedFourierDiagonal:
        """Return ``H`` on the stack ``sampling``: the plane operator with one channel per focal plane, as a stack."""
        plane = Sampling(shape=(1, *sampling.shape[1:]), pixel_size=sampling.pixel_size)
        planes = FourierDiagonal(self.plane_transfers(sampling), plane)
        return ReshapedFourierDiagonal(planes, sampling, data_shape=self.data_shape(sampling))


def _lower(detection: Detection) -> float:
    na_range = detection.numerical_aperture_range
    if na_range is None:
        raise ParameterError(
            "detection.numerical_aperture must have bounds when free", path="detection.numerical_aperture"
        )
    return na_range[0]
