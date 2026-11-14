"""The models of each contrast: brightfield phase and fluorescence, for a thick and a thin object.

:class:`Phase3D` and :class:`Phase2D` image brightfield phase, :class:`Fluorescence3D` and :class:`Fluorescence2D`
fluorescence.

The phase models turn recorded brightfield intensities into the contrast ``I / mean(I) - 1`` the phase systems
measure, as 3.x normalized them: :class:`Phase3D` by the mean of the whole (z-padded) stack, :class:`Phase2D` plane
by plane. Recorded fluorescence is what the fluorescence systems measure, so it is reconstructed as recorded.
"""

from collections.abc import Sequence
from typing import Literal, Unpack, override

import torch
from torch import Tensor

from waveorder.models._arguments import expect_optics
from waveorder.models._base import ReconstructionModel
from waveorder.models._parts import Parts
from waveorder.nn import (
    ConfocalFluorescence3D,
    Detection,
    Illumination,
    Quantity,
    ThickPhase3D,
    ThinFluorescence2D,
    ThinPhase2D,
    WidefieldFluorescence3D,
    brightfield_contrast,
)
from waveorder.nn.parameters import placement


class Phase3D(ReconstructionModel):
    """Phase of a thick object from a brightfield stack (:class:`~waveorder.nn.ThickPhase3D`).

    The object is the ``phase`` field of each voxel, ``(1, Z, Y, X)``, or phase and absorption with
    ``absorption="independent"``. The recorded stack is divided by its mean and 1 is subtracted
    (:meth:`prepare`); the default solver is the closed-form Wiener filter, the 3.x Tikhonov reconstruction::

        model = Phase3D(
            Illumination(wavelength=0.532, numerical_aperture=0.5),
            Detection(wavelength=0.532, numerical_aperture=1.2),
            index_of_refraction_media=1.33,
        )
        phase = model.reconstruct(stack, pixel_size=(0.25, 0.1, 0.1)).estimate

    Units and assumptions. The data the system measures is the brightfield contrast ``I / mean(I) - 1``, a
    fraction of the mean intensity, so a Gaussian noise ``sigma`` is a fraction too (0.01 is 1 % of the mean). The
    estimate is the ``phase`` field of :class:`~waveorder.nn.ThickPhase3D`: a scattering potential in um^-2, not
    a phase per voxel and not radians as :class:`Phase2D` returns. wo-4.0 returned the same numbers (its transfer
    function is this one) but documented them as cycles per voxel. Multiply the estimate by
    ``dz * wavelength / (4 pi n_media)`` for the phase delay across one voxel in radians (``dz`` the z voxel
    size and ``wavelength`` in um, ``n_media`` the ``index_of_refraction_media``);
    :meth:`waveorder.nn.ThickPhase3D.radians_per_voxel` returns the factor for a sampling, and its docstring says
    how exact the conversion is. The model is the weak-object (first Born) approximation: the
    contrast is linear in the object, which holds while the light the object scatters is weak against the
    unscattered light (a contrast well below 1) and the mean of the stack stands for the unscattered intensity.
    One wavelength and one refractive index of the medium are assumed.

    Parameters
    ----------
    illumination : Illumination
        Condenser aperture and tilt; its wavelength is the imaging wavelength, in um.
    detection : Detection
        Detection arm, with the illumination's wavelength.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    invert_phase_contrast : bool
        Negate the defocus of every plane, for stacks acquired in the opposite z direction.
    absorption : float, Free or "independent"
        Absorption per unit phase (3.x ``absorption_ratio``, default 0: a pure phase object), or ``"independent"``
        for a second object field.
    **parts
        The parts every model takes: ``object_prior``, ``noise``, ``solver``, ``score``, ``search``,
        ``z_padding``, ``field_tiles``, ``pixel_size``, ``apodization_rolloff`` and ``device``
        (:class:`~waveorder.models.ReconstructionModel`).

    Raises
    ------
    ParameterError
        As :class:`~waveorder.nn.ThickPhase3D` for the optics, and as
        :class:`~waveorder.models.ReconstructionModel` for the parts.
    """

    system_types = (ThickPhase3D,)
    prepare_scope = "stack_mean"

    def __init__(
        self,
        illumination: Illumination,
        detection: Detection,
        *,
        index_of_refraction_media: float,
        invert_phase_contrast: bool = False,
        absorption: Quantity | Literal["independent"] = 0.0,
        **parts: Unpack[Parts],
    ) -> None:
        expect_optics(
            type(self).__name__,
            illumination=illumination,
            detection=detection,
            index_of_refraction_media=index_of_refraction_media,
        )
        system = ThickPhase3D(
            illumination,
            detection,
            index_of_refraction_media=index_of_refraction_media,
            invert_phase_contrast=invert_phase_contrast,
            absorption=absorption,
        )
        super().__init__(system, **parts)

    @override
    def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
        """Return the contrast ``I / mean(I) - 1`` of recorded intensities, in the dtype the model computes in.

        The mean is one per channel over the whole stack: :meth:`reconstruct` pads z before it prepares, so there
        it is the mean of the z-padded stack, as 3.x normalized; the tile and position methods compute it once
        for the volume and pass it with each block. Means are clamped at 1e-12.

        Parameters
        ----------
        data : Tensor
            Recorded intensities ``(C, Z, Y, X)``.
        mean : Tensor, optional
            The mean of the whole stack per channel, ``(C,)`` (or one number), when ``data`` is only a part of it (a
            tile); computed from ``data`` by default.
        """
        data = data.to(dtype=placement(self.system)[0])  # in float32 for a float32 model, as 3.x normalized
        if mean is None:
            mean = data.mean(dim=(-3, -2, -1), keepdim=True)
        else:
            mean = torch.as_tensor(mean).to(device=data.device, dtype=data.dtype).reshape(-1, 1, 1, 1)
        return data / mean.clamp_min(1e-12) - 1

    @override
    def output_channel_names(self, channels: Sequence[str]) -> list[str]:
        """Return ``["Phase3D"]``, the 3.x name (with ``"Absorption3D"`` for an independent absorption field)."""
        del channels
        return [f"{field.capitalize()}3D" for field in self.system.fields]


class Phase2D(ReconstructionModel):
    """Phase and absorption of a thin object from a brightfield defocus stack (:class:`~waveorder.nn.ThinPhase2D`).

    The object is one plane, ``(2, 1, Y, X)``: phase and absorption. Each recorded plane is divided by its own mean
    and 1 is subtracted (:meth:`prepare`).

    Units and assumptions. The data the system measures is the contrast ``I / mean(I) - 1`` of each plane, a
    fraction of that plane's mean intensity, and a Gaussian noise ``sigma`` is such a fraction. The estimated
    ``phase`` is in radians and ``absorption`` is the log of the amplitude transmittance, both as
    :class:`~waveorder.nn.ThinPhase2D` defines them. The model is the weak-object approximation of a thin
    object: the contrast is linear in phase and absorption, which holds while the scattered light is weak
    against the unscattered light (a contrast well below 1) and the object is much thinner than the depth of
    field, in a medium of one refractive index. Because every plane is normalized by its own mean, the result
    of :meth:`prepare` depends on the block it is given (``prepare_scope = "block"``), so the model reconstructs
    whole volumes: the tile methods refuse to split one.

    Parameters
    ----------
    illumination : Illumination
        Condenser aperture and tilt; its numerical aperture must stay below the detection's.
    detection : Detection
        Detection arm, with the illumination's wavelength.
    index_of_refraction_media : float
        Refractive index of the medium.
    invert_phase_contrast : bool
        Negate the defocus of every plane.
    z_positions : tuple[float, ...], optional
        Defocus of each plane in um, in data order, for unevenly spaced stacks; by default the planes are evenly
        spaced around the centre slice.
    **parts
        The parts every model takes (:class:`~waveorder.models.ReconstructionModel`).

    Raises
    ------
    ParameterError
        As :class:`~waveorder.nn.ThinPhase2D` for the optics, and as
        :class:`~waveorder.models.ReconstructionModel` for the parts.
    """

    system_types = (ThinPhase2D,)
    prepare_scope = "block"  # each plane by its own mean: a tile's planes would have other means than the volume's

    def __init__(
        self,
        illumination: Illumination,
        detection: Detection,
        *,
        index_of_refraction_media: float,
        invert_phase_contrast: bool = False,
        z_positions: tuple[float, ...] | None = None,
        **parts: Unpack[Parts],
    ) -> None:
        expect_optics(
            type(self).__name__,
            illumination=illumination,
            detection=detection,
            index_of_refraction_media=index_of_refraction_media,
        )
        system = ThinPhase2D(
            illumination,
            detection,
            index_of_refraction_media=index_of_refraction_media,
            invert_phase_contrast=invert_phase_contrast,
            z_positions=z_positions,
        )
        super().__init__(system, **parts)

    @override
    def prepare(self, data: Tensor, *, mean: Tensor | None = None) -> Tensor:
        """Return the contrast of recorded intensities ``(C, Z, Y, X)``, each plane by its own mean, in float64."""
        del mean
        return brightfield_contrast(data.double(), per_plane=True)

    @override
    def output_channel_names(self, channels: Sequence[str]) -> list[str]:
        """Return ``["Phase2D", "Absorption2D"]``, the 3.x names."""
        del channels
        return [f"{field.capitalize()}2D" for field in self.system.fields]


class Fluorescence3D(ReconstructionModel):
    """Fluorophore density of a thick object from a fluorescence stack, widefield or confocal.

    Without a pinhole the system is :class:`~waveorder.nn.WidefieldFluorescence3D`; with a pinhole diameter it is
    :class:`~waveorder.nn.ConfocalFluorescence3D`. The object is the density of each voxel, ``(1, Z, Y, X)``.

    Units and assumptions. The data is reconstructed as recorded (:meth:`prepare` changes nothing), so a Gaussian
    noise ``sigma`` is in the recorded values (camera counts), and a camera offset left in the data is
    reconstructed as object. The transfer function passes the mean unchanged, so the estimated density is in the
    units of the data: the recorded signal each voxel would give in focus. The image is linear in the density (no
    saturation, bleaching or reabsorption), the point-spread function is the same over the field of view (see
    ``field_tiles`` otherwise) and the medium has one refractive index. With a pinhole, the excitation
    point-spread function is taken equal to the detection's, at the emission wavelength (the 3.x model): the
    excitation wavelength and the Stokes shift do not enter, so the true confocal resolution is slightly better
    than the model's when the excitation wavelength is shorter.

    Background and calibration. The default object prior is Gaussian with zero mean, and :meth:`prepare` leaves
    the data as recorded, so a background in the data (a camera offset, a constant haze) is an object the prior
    does not expect. A calibration by the evidence then moves the free quantities to account for it. Subtract
    the background from the data before calibrating by the evidence.

    Parameters
    ----------
    detection : Detection
        Detection arm; its wavelength is the emission wavelength, in um.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    pinhole_diameter : float or Free, optional
        Effective confocal pinhole diameter in um, in the object plane (the physical diameter divided by the
        magnification; ``detection.pinhole.diameter_um`` in a configuration); ``0`` is a point pinhole. ``None``
        (the default) is widefield detection. The pinhole factor is the 3.x squared jinc, not the transform of a
        hard disk, which is the unsquared jinc: to second order a hard pinhole of diameter ``D`` acts like
        ``pinhole_diameter = D / sqrt(2)``, so a calibrated value is about 0.71 times the true diameter (see
        :class:`~waveorder.nn.ConfocalFluorescence3D`).
    **parts
        The parts every model takes: ``object_prior``, ``noise``, ``solver``, ``score``, ``search``,
        ``z_padding``, ``field_tiles``, ``pixel_size``, ``apodization_rolloff`` and ``device``
        (:class:`~waveorder.models.ReconstructionModel`).

    Raises
    ------
    ParameterError
        As :class:`~waveorder.nn.WidefieldFluorescence3D` or :class:`~waveorder.nn.ConfocalFluorescence3D` for the
        optics, and as :class:`~waveorder.models.ReconstructionModel` for the parts.
    """

    system_types = (WidefieldFluorescence3D, ConfocalFluorescence3D)

    def __init__(
        self,
        detection: Detection,
        *,
        index_of_refraction_media: float,
        pinhole_diameter: Quantity | None = None,
        **parts: Unpack[Parts],
    ) -> None:
        expect_optics(type(self).__name__, detection=detection, index_of_refraction_media=index_of_refraction_media)
        system: WidefieldFluorescence3D
        if pinhole_diameter is None:
            system = WidefieldFluorescence3D(detection, index_of_refraction_media=index_of_refraction_media)
        else:
            system = ConfocalFluorescence3D(
                detection, pinhole_diameter=pinhole_diameter, index_of_refraction_media=index_of_refraction_media
            )
        super().__init__(system, **parts)

    @override
    def output_channel_names(self, channels: Sequence[str]) -> list[str]:
        """Return ``["<channel>_Density3D"]``, the 3.x name, for the one input channel."""
        return [f"{_first(channels)}_Density3D"]


class Fluorescence2D(ReconstructionModel):
    """Fluorophore density of a thin object from a defocus stack (:class:`~waveorder.nn.ThinFluorescence2D`).

    The object is one plane, ``(1, 1, Y, X)``.

    Units and assumptions. The data is reconstructed as recorded (:meth:`prepare` changes nothing), so a Gaussian
    noise ``sigma`` is in the recorded values (camera counts), and the estimated density is in the units of the
    data: each plane's transfer function passes the mean unchanged. The object is thin: all of the fluorescence
    comes from one plane, much thinner than the depth of field, which every plane of the stack images at its own
    defocus. There is no term for out-of-focus background: light from fluorophores above or below that plane,
    and a camera offset left in the data, are reconstructed as density in the plane. The image is linear in the
    density, the point-spread function is the same over the field of view, and the medium has one refractive
    index. For a thick object use :class:`Fluorescence3D`.

    Background and calibration. The default object prior is Gaussian with zero mean, and :meth:`prepare` leaves
    the data as recorded, so a background in the data (a camera offset, out-of-focus light) is an object the
    prior does not expect. A calibration by the evidence then moves the free quantities to account for it.
    Subtract the background from the data before calibrating by the evidence.

    Parameters
    ----------
    detection : Detection
        Detection arm; its wavelength is the emission wavelength, in um.
    index_of_refraction_media : float
        Refractive index of the immersion and object medium.
    z_positions : tuple[float, ...], optional
        Defocus of each plane in um, in data order, for unevenly spaced stacks; by default the planes are evenly
        spaced around the centre slice.
    **parts
        The parts every model takes (:class:`~waveorder.models.ReconstructionModel`).

    Raises
    ------
    ParameterError
        As :class:`~waveorder.nn.ThinFluorescence2D` for the optics, and as
        :class:`~waveorder.models.ReconstructionModel` for the parts.
    """

    system_types = (ThinFluorescence2D,)

    def __init__(
        self,
        detection: Detection,
        *,
        index_of_refraction_media: float,
        z_positions: tuple[float, ...] | None = None,
        **parts: Unpack[Parts],
    ) -> None:
        expect_optics(type(self).__name__, detection=detection, index_of_refraction_media=index_of_refraction_media)
        system = ThinFluorescence2D(
            detection, index_of_refraction_media=index_of_refraction_media, z_positions=z_positions
        )
        super().__init__(system, **parts)

    @override
    def output_channel_names(self, channels: Sequence[str]) -> list[str]:
        """Return ``["<channel>_Density2D"]``, the 3.x name, for the one input channel."""
        return [f"{_first(channels)}_Density2D"]


def _first(channels: Sequence[str]) -> str:
    """Return the first input channel name, or the empty string when none is given (as 3.x named the output)."""
    return channels[0] if channels else ""
