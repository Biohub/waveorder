"""Fluorescence imaging systems: incoherent emission detected through one pupil.

* :class:`WidefieldFluorescence3D`, a 3D object imaged as a 3D stack (3.x ``isotropic_fluorescent_thick_3d``).
* :class:`ConfocalFluorescence3D`, the same with a pinhole: the widefield point-spread function times its
  pinhole-filtered copy (3.x confocal path).
* :class:`ThinFluorescence2D`, a single plane imaged at several focal planes (3.x ``isotropic_fluorescent_thin_3d``).
"""

import math
from typing import override

import torch
from torch import Tensor, nn

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.functional import (
    axial_nyquist,
    crop_full_lateral,
    defocused_intensity,
    from_half_spectrum,
    half_of_hermitian_part,
    intensity_otf,
    irfftn,
    plane_slabs,
    to_half_spectrum,
    transverse_nyquist,
    widefield_otf,
)
from waveorder.nn.operators import FourierDiagonal, LinearOperator, ReshapedFourierDiagonal
from waveorder.nn.optics import Detection, PupilGrid, pinhole_otf
from waveorder.nn.parameters import Quantity, repr_parts, value_range
from waveorder.nn.sampling import ZYX, Sampling
from waveorder.nn.systems._checks import (
    expect_emission_wavelength,
    expect_feasible_grid,
    expect_index,
    expect_resolved_aperture,
    expect_z_positions,
)
from waveorder.nn.systems.base import CacheMode, ImagingSystem, ShiftInvariantSystem
from waveorder.registry import register


def _reachable_aperture(detection: Detection, *, index_of_refraction_media: float) -> float:
    """Return the largest numerical aperture ``detection`` can reach (the medium's index when unbounded).

    Raises :class:`ParameterError` if it can exceed ``index_of_refraction_media``.
    """
    na_range = detection.numerical_aperture_range
    na_limit = index_of_refraction_media if na_range is None else na_range[1]
    if na_limit > index_of_refraction_media:
        raise ParameterError(
            f"detection.numerical_aperture can reach {na_limit}, above index_of_refraction_media = "
            f"{index_of_refraction_media}",
            path="detection.numerical_aperture",
        )
    return na_limit


@register("widefield_fluorescence_3d", kind="system")
class WidefieldFluorescence3D(ShiftInvariantSystem):
    """Incoherent widefield fluorescence: the OTF is the 3D Fourier transform of ``|IFFT2(pupil * defocus)|^2``.

    The transfer function is computed on a grid fine enough for the largest numerical aperture the detection can
    reach, then cropped to the data's frequencies. Scalar diffraction through an unapodized pupil in one medium: no
    polarization, aplanatic factor or index mismatch. The transfer is stored real when the detection has no pupil
    aberrations and a fixed zero focus offset.

    Parameters
    ----------
    detection : Detection
        Detection arm; its wavelength, the emission wavelength in um, is required. Its ``z_focus_offset`` (slices)
        translates the estimate along z.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    cache : {"off", "auto"}
        See :class:`~waveorder.nn.ShiftInvariantSystem`.

    Raises
    ------
    ParameterError
        If ``detection`` has no wavelength, ``index_of_refraction_media`` is not a positive finite number, or the
        detection's numerical aperture can exceed it.
    """

    detection: Detection
    preserves_nonnegativity = True  # the PSF is an intensity, clamped at 0 after the crop (compute_transfer)

    def __init__(self, detection: Detection, *, index_of_refraction_media: float, cache: CacheMode = "off") -> None:

        expect_emission_wavelength(detection, where=type(self).__name__)
        expect_index(index_of_refraction_media, where="WidefieldFluorescence3D")
        super().__init__(fields=("fluorescence",), channel_count=1, cache=cache)
        self.detection = detection
        self.index_of_refraction_media = index_of_refraction_media
        self.na_limit = _reachable_aperture(detection, index_of_refraction_media=index_of_refraction_media)
        focus_fixed = not isinstance(detection.z_focus_offset, nn.Parameter)
        aberrated = detection.pupil_aberrations is not None
        self.real_transfer = not aberrated and focus_fixed and detection.z_focus_offset_range == (0, 0)

    @override
    def extra_repr(self) -> str:
        """Return the medium's index, then the quantities."""
        return repr_parts(f"index_of_refraction_media={self.index_of_refraction_media:g}", super().extra_repr())

    @override
    def check_sampling(self, sampling: Sampling) -> None:
        """Reject a focus offset beyond half the stack (the defocus would wrap) and a grid that cannot hold the optics.

        Raises
        ------
        IncompatibleModelError
            If the offset can leave the stack, the cut-off ``NA / wavelength`` is below one lateral frequency step
            (a wavelength in nm), or the transfer function grid is too large (a pixel size in nm).
        """
        where = type(self).__name__
        expect_resolved_aperture(
            sampling, numerical_aperture=self.na_limit, wavelength=self.detection.wavelength, where=where
        )
        expect_feasible_grid(sampling, self.upsampling(sampling), where=where)
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
        """Return the per-axis factors that make ``sampling`` fine enough for the reachable numerical aperture."""
        wavelength = self.detection.wavelength
        lateral = transverse_nyquist(wavelength, 2 * self.na_limit)
        axial = axial_nyquist(wavelength, self.na_limit, self.index_of_refraction_media)
        dz, dy, dx = sampling.pixel_size
        return (math.ceil(dz / axial), math.ceil(dy / lateral), math.ceil(dx / lateral))

    @override
    def compute_transfer(self, sampling: Sampling) -> Tensor:
        """Return the OTF cropped to ``sampling``'s frequencies, its point-spread function clamped at 0.

        The crop rings the point-spread function below zero where the data undersample the OTF; clamping and
        renormalizing to 1 at zero frequency keeps it an intensity (3.x ``_enforce_nonnegative_psf``).
        """
        otf = self._cropped_otf(sampling)
        psf = from_half_spectrum(otf, sampling.shape).clamp_min(0)
        otf = to_half_spectrum(psf / psf.sum())
        return (otf[..., 0] if self.real_transfer else otf)[None, None]

    def _cropped_otf(self, sampling: Sampling) -> Tensor:
        """Return the OTF on ``sampling``'s frequencies, 1 at zero frequency, stored real ``(Z, Y, X // 2 + 1, 2)``."""
        pupil, kz, z = self._pupil_planes(sampling)
        return widefield_otf(pupil, kz, z, shape=sampling.shape)

    def _pupil_planes(self, sampling: Sampling) -> tuple[Tensor, Tensor, Tensor]:
        """Return the pupil ``(Y', X', 2)``, its ``k_z`` ``(Y', X')`` and the plane defocus ``(Z',)``, upsampled."""
        reference = self.detection.numerical_aperture
        fine = sampling.upsampled(self.upsampling(sampling))
        grid = PupilGrid.from_sampling(fine, device=reference.device, dtype=reference.dtype)
        pupil = self.detection.pupil(grid)
        n_over_lambda = self.index_of_refraction_media / self.detection.wavelength
        kz = torch.sqrt(torch.clamp(n_over_lambda**2 - grid.radius**2, min=0.0))
        nz, dz = fine.shape[0], fine.pixel_size[0]
        planes = torch.fft.fftfreq(nz, d=1 / nz, device=reference.device, dtype=reference.dtype)
        # z_focus_offset is in data slices: the focal plane sits that many slices from the first plane (index 0).
        return pupil, kz, planes * dz + self.detection.z_focus_offset * sampling.pixel_size[0]


@register("confocal_fluorescence_3d", kind="system")
class ConfocalFluorescence3D(WidefieldFluorescence3D):
    """Confocal fluorescence of a thick object (3.x ``isotropic_fluorescent_thick_3d`` with a pinhole).

    The point-spread function is the widefield one times its pinhole-filtered copy, both at the emission wavelength as
    in 3.x. The pinhole multiplies the lateral OTF by ``(2 J1(pi d k) / (pi d k))^2``
    (:func:`~waveorder.nn.optics.pinhole_otf`); the 3D OTF is normalized to 1 at zero frequency.

    Parameters
    ----------
    detection : Detection
        Detection arm; its wavelength, the emission (and excitation) wavelength in um, is required.
    pinhole_diameter : float or Free
        Effective pinhole diameter ``d`` in um in the object plane (physical diameter over magnification, as in 3.x);
        ``0`` is a point pinhole. The squared jinc makes a hard pinhole of diameter ``D`` act like ``D / sqrt(2)``. A
        free diameter needs a lower bound of at least 0.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    cache : {"off", "auto"}
        See :class:`~waveorder.nn.ShiftInvariantSystem`.

    Raises
    ------
    ParameterError
        As :class:`WidefieldFluorescence3D`, or if ``pinhole_diameter`` can go below 0 or is free without bounds.
    """

    pinhole_diameter: Tensor

    def __init__(
        self,
        detection: Detection,
        *,
        pinhole_diameter: Quantity,
        index_of_refraction_media: float,
        cache: CacheMode = "off",
    ) -> None:
        super().__init__(detection, index_of_refraction_media=index_of_refraction_media, cache=cache)
        self.register_quantity("pinhole_diameter", pinhole_diameter)
        reach = value_range(pinhole_diameter)
        if reach is None or reach[0] < 0:
            raise ParameterError(
                f"ConfocalFluorescence3D pinhole_diameter must stay at least 0 um (give a free one bounds), got "
                f"{pinhole_diameter}",
                path="pinhole_diameter",
            )

    @override
    def upsampling(self, sampling: Sampling) -> ZYX[int]:
        """Return factors for half the widefield Nyquist steps: the confocal OTF reaches twice as far."""
        wavelength = self.detection.wavelength
        lateral = transverse_nyquist(wavelength, 2 * self.na_limit) / 2
        axial = axial_nyquist(wavelength, self.na_limit, self.index_of_refraction_media) / 2
        dz, dy, dx = sampling.pixel_size
        return (math.ceil(dz / axial), math.ceil(dy / lateral), math.ceil(dx / lateral))

    @override
    def _cropped_otf(self, sampling: Sampling) -> Tensor:
        # The pinhole filters laterally only, so the detection PSF of each plane is that plane's excitation PSF (the
        # widefield intensity) filtered in 2D, and the confocal PSF is their product plane by plane. Its sum
        # normalizes, so the widefield intensity needs none.
        pupil, kz, z = self._pupil_planes(sampling)
        lateral = (kz.shape[-2], kz.shape[-1])
        pinhole = pinhole_otf(self.pinhole_diameter, sampling.upsampled(self.upsampling(sampling)))

        def confocal(part: Tensor) -> Tensor:
            excitation = defocused_intensity(pupil, kz, part)
            return excitation * irfftn(torch.fft.rfft2(excitation) * pinhole, lateral)

        return intensity_otf(confocal, z, lateral=lateral, shape=sampling.shape)


@register("thin_fluorescence_2d", kind="system")
class ThinFluorescence2D(ImagingSystem):
    """Fluorescence of a thin object from a defocus stack (3.x ``isotropic_fluorescent_thin_3d``).

    The object is one plane of fluorophore density, ``(1, 1, Y, X)``; the data are one channel at ``Z`` focal planes,
    ``(1, Z, Y, X)``. Plane ``i`` sees the 2D Fourier transform of ``|IFFT2(pupil * exp(2 pi i z_i k_z))|^2`` at
    defocus ``z_i = (Z // 2 + z_focus_offset - i) * dz``, as in 3.x, normalized to 1 at zero frequency.

    Parameters
    ----------
    detection : Detection
        Detection arm; its wavelength, the emission wavelength in um, is required. Its ``z_focus_offset``
        (slices) sets the slice in focus, ``Z // 2 + z_focus_offset``.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    z_positions : tuple[float, ...], optional
        Defocus of each plane in um, in data order, before the focus offset; replaces ``(Z // 2 - i) * dz``.

    Raises
    ------
    ParameterError
        If ``detection`` has no wavelength, ``index_of_refraction_media`` is not a positive finite number, the
        detection's numerical aperture can exceed it, or ``z_positions`` is not a nonempty tuple of finite numbers.
    """

    operator_type: type[LinearOperator] = ReshapedFourierDiagonal
    detection: Detection
    preserves_nonnegativity = True  # each plane's PSF is an intensity

    def __init__(
        self, detection: Detection, *, index_of_refraction_media: float, z_positions: tuple[float, ...] | None = None
    ) -> None:

        expect_emission_wavelength(detection, where="ThinFluorescence2D")
        expect_index(index_of_refraction_media, where="ThinFluorescence2D")
        expect_z_positions(z_positions, where="ThinFluorescence2D")
        super().__init__(fields=("fluorescence",), channel_count=1)
        self.detection = detection
        self.index_of_refraction_media = index_of_refraction_media
        self.na_limit = _reachable_aperture(detection, index_of_refraction_media=index_of_refraction_media)
        self.z_positions = z_positions
        self.real_transfer = detection.pupil_aberrations is None

    @override
    def extra_repr(self) -> str:
        """Return the medium's index, the plane positions when given, then the quantities."""
        positions = "" if self.z_positions is None else f"z_positions={self.z_positions!r}"
        index = f"index_of_refraction_media={self.index_of_refraction_media:g}"
        return repr_parts(index, positions, super().extra_repr())

    @override
    def object_shape(self, sampling: Sampling) -> tuple[int, ...]:
        """Return ``(1, 1, Y, X)``: one plane."""
        return (len(self.fields), 1, *sampling.shape[1:])

    @override
    def check_sampling(self, sampling: Sampling) -> None:
        """Reject a stack whose plane count differs from ``z_positions`` and a grid that cannot hold the optics."""
        expect_resolved_aperture(
            sampling, numerical_aperture=self.na_limit, wavelength=self.detection.wavelength, where="ThinFluorescence2D"
        )
        expect_feasible_grid(sampling, self.upsampling(sampling), where="ThinFluorescence2D")
        if self.z_positions is not None and len(self.z_positions) != sampling.shape[0]:
            raise IncompatibleModelError(
                f"z_positions lists {len(self.z_positions)} planes, the data have {sampling.shape[0]}"
            )

    def upsampling(self, sampling: Sampling) -> ZYX[int]:
        """Return the lateral oversampling factors (z is never oversampled) for the reachable numerical aperture."""
        step = transverse_nyquist(self.detection.wavelength, 2 * self.na_limit)
        _, dy, dx = sampling.pixel_size
        return (1, math.ceil(dy / step), math.ceil(dx / step))

    def plane_positions(self, sampling: Sampling) -> Tensor:
        """Return the defocus of each data plane in um, shape ``(Z,)``, data order."""
        reference = self.detection.numerical_aperture
        nz, dz = sampling.shape[0], sampling.pixel_size[0]
        if self.z_positions is None:
            base = (nz // 2 - torch.arange(nz, device=reference.device, dtype=reference.dtype)) * dz
        else:
            base = torch.tensor(self.z_positions, device=reference.device, dtype=reference.dtype)
        return base + self.detection.z_focus_offset * dz

    def plane_transfers(self, sampling: Sampling) -> Tensor:
        """Return the transfer function of every plane.

        Returns
        -------
        Tensor
            Half spectra, shape ``(Z, 1, 1, Y, X // 2 + 1)`` stored real when :attr:`real_transfer` is set, else
            ``(Z, 1, 1, Y, X // 2 + 1, 2)``: per plane, the half of the Hermitian part of the cropped transfer
            function (:func:`~waveorder.nn.functional.half_of_hermitian_part`).
        """
        self.check_sampling(sampling)
        reference = self.detection.numerical_aperture
        _, ny, nx = sampling.shape
        fine = Sampling(shape=(1, ny, nx), pixel_size=sampling.pixel_size).upsampled(self.upsampling(sampling))
        grid = PupilGrid.from_sampling(fine, device=reference.device, dtype=reference.dtype)
        n_over_lambda = self.index_of_refraction_media / self.detection.wavelength
        kz = torch.sqrt(torch.clamp(n_over_lambda**2 - grid.radius**2, min=0.0))
        pupil = self.detection.pupil(grid)

        def spectra(part: Tensor) -> Tensor:
            intensity = defocused_intensity(pupil, kz, part)  # (P, Y', X')
            # The zero frequency of fft2(I) is sum(I): normalizing I first makes it 1 on every plane.
            spectrum = torch.view_as_real(torch.fft.fft2(intensity / intensity.sum(dim=(-2, -1), keepdim=True)))
            return half_of_hermitian_part(crop_full_lateral(spectrum, sampling.shape), axes=2)

        # In slabs of planes, each cropped at once: the planes are independent.
        slabs = plane_slabs(self.plane_positions(sampling), kz.numel())
        planes = torch.cat([spectra(part) for part in slabs])[:, None, None]  # (Z, 1, 1, Y, X // 2 + 1, 2)
        return planes[..., 0] if self.real_transfer else planes

    @override
    def compute_operator(self, sampling: Sampling) -> ReshapedFourierDiagonal:
        """Return ``H`` on the stack ``sampling``: the plane operator with one channel per focal plane."""
        plane = Sampling(shape=(1, *sampling.shape[1:]), pixel_size=sampling.pixel_size)
        planes = FourierDiagonal(self.plane_transfers(sampling), plane)
        return ReshapedFourierDiagonal(planes, sampling, data_shape=self.data_shape(sampling))
