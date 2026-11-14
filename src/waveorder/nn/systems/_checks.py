"""Argument checks the imaging systems share."""

import math

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.optics import Detection
from waveorder.nn.parameters import expect_fixed
from waveorder.nn.sampling import ZYX, Sampling

MAX_UPSAMPLING = 100
"""The largest factor by which a transfer function grid may be finer than the data along one axis."""

MAX_TRANSFER_VOXELS = 2**36
"""The most voxels a transfer function grid may hold (512 GiB as one complex64 tensor)."""


def expect_index(value: float, *, where: str) -> None:
    """Raise :class:`ParameterError` unless ``index_of_refraction_media`` is a positive finite number (not free)."""
    expect_fixed(value, what=f"{where} index_of_refraction_media", path="index_of_refraction_media")


def expect_z_positions(value: tuple[float, ...] | None, *, where: str) -> None:
    """Raise :class:`ParameterError` unless ``z_positions`` is ``None`` or a nonempty tuple of finite numbers."""
    if value is None:
        return
    if not value or not all(math.isfinite(z) for z in value):
        raise ParameterError(
            f"{where} z_positions must be finite defocus values in um, at least one, got {value}", path="z_positions"
        )


def expect_emission_wavelength(detection: Detection, *, where: str) -> None:
    """Raise :class:`ParameterError` at ``detection.wavelength`` unless a fluorescence detection has a wavelength."""
    if not detection.has_wavelength:
        raise ParameterError(
            f"{where} needs detection.wavelength, the emission wavelength in um: pass Detection(wavelength=...); "
            "only the phase systems take it from the illumination",
            path="detection.wavelength",
        )


def expect_resolved_aperture(sampling: Sampling, *, numerical_aperture: float, wavelength: float, where: str) -> None:
    """Raise :class:`IncompatibleModelError` if the detection cut-off is below one lateral frequency step of the grid.

    Such a pupil holds only the zero frequency, so a reconstruction would return zeros. The usual cause is a
    wavelength in nm (``532`` for ``0.532``).

    Parameters
    ----------
    sampling : Sampling
        The data's grid; the threshold is its finer lateral frequency step, ``1 / (N * pixel)``.
    numerical_aperture : float
        The largest detection numerical aperture the system can reach.
    wavelength : float
        The detected wavelength in vacuum, um.
    where : str
        The system's class name, which starts the message.
    """
    (_, ny, nx), (_, dy, dx) = sampling.shape, sampling.pixel_size
    step = min(1 / (ny * dy), 1 / (nx * dx))
    cutoff = numerical_aperture / wavelength
    if cutoff < step:
        raise IncompatibleModelError(
            f"{where}: the detection cut-off NA / wavelength = {numerical_aperture:g} / {wavelength:g} = {cutoff:.3g} "
            f"cycles/um is below one frequency step, {step:.3g} cycles/um, of a {ny} x {nx} grid of {dy:g} x {dx:g} um "
            f"pixels, so the pupil holds no frequency but zero. wavelength is in um, got {wavelength:g} (a value in "
            f"nm is 1000 times too large); pixel sizes are in um too, and the field of view must span at least one "
            f"resolution element wavelength / NA"
        )


def expect_feasible_grid(sampling: Sampling, factors: ZYX[int], *, where: str) -> None:
    """Raise :class:`IncompatibleModelError` if the transfer function grid of ``sampling`` could not be computed.

    The grid is the data's grid refined by ``factors``. A factor above :data:`MAX_UPSAMPLING` (usually a pixel size
    in nm) or a grid above :data:`MAX_TRANSFER_VOXELS` voxels is refused.

    Parameters
    ----------
    sampling : Sampling
        The data's grid.
    factors : tuple[int, int, int]
        The system's upsampling factors along z, y and x on ``sampling``.
    where : str
        The system's class name, which starts the message.
    """
    sizes = " x ".join(f"{size:g}" for size in sampling.pixel_size)
    fine = tuple(count * factor for count, factor in zip(sampling.shape, factors, strict=True))
    grid = " x ".join(map(str, fine))
    worst = max(range(3), key=lambda axis: factors[axis])
    if factors[worst] > MAX_UPSAMPLING:
        raise IncompatibleModelError(
            f"{where}: a pixel size of {sizes} um (z, y, x) is {' x '.join(map(str, factors))} times the steps that "
            f"sample these optics, more than the {MAX_UPSAMPLING} times a transfer function grid may be refined "
            f"along one axis (it would be {grid}). pixel_size is in um, got {sampling.pixel_size[worst]:g} along "
            f"{'zyx'[worst]} (a value in nm is 1000 times too large); a camera pixel is divided by the magnification"
        )
    if math.prod(fine) > MAX_TRANSFER_VOXELS:
        raise IncompatibleModelError(
            f"{where}: the transfer function of {' x '.join(map(str, sampling.shape))} voxels of {sizes} um would be "
            f"computed on a {grid} grid, {math.prod(fine):.3g} voxels, more than the {MAX_TRANSFER_VOXELS} that can "
            f"be held. pixel_size is in um, got {sizes}; if it is right, reconstruct the volume tile by tile"
        )
