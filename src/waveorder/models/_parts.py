"""The settings a model takes besides its imaging system: :class:`Parts`, :class:`FieldTiles`, ``PrepareScope``.

Also the z padding ``"auto"`` resolves to (:func:`auto_z_padding`).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch

from waveorder.errors import ParameterError, ShapeError
from waveorder.models._arguments import PixelSize, ints
from waveorder.nn import ImagingSystem, Noise, ObjectPrior, Sampling, Score, SearchStage, Solver
from waveorder.nn.optics import Detection

type PrepareScope = Literal["voxel", "stack_mean", "block"]
"""What a model's ``prepare`` reads to prepare one voxel: the voxel, also the stack's mean, or the whole block."""

type ZPaddingSetting = int | Literal["auto"]
"""A model's ``z_padding``: mirrored slices at each end of z, or ``"auto"`` (:func:`auto_z_padding`)."""

PREPARE_SCOPES: tuple[PrepareScope, ...] = ("voxel", "stack_mean", "block")
"""The values a model class's ``prepare_scope`` may take."""


@dataclass(frozen=True, slots=True, init=False)
class FieldTiles:
    """The tiles a shift-variant system is calibrated and reconstructed on, one parameter set per tile.

    The settings of ``imaging_system.field_dependence: tiled``. Each tile spans z, is treated as shift-invariant and
    is calibrated on its own data alone; neighbouring reconstructions are cross-faded in their overlap.

    Parameters
    ----------
    size_yx : tuple[int, int]
        Tile size in pixels along y and x.
    overlap_yx : tuple[int, int], optional
        Pixels each tile is widened by on each side and cross-faded with its neighbours, at most half of ``size_yx``;
        by default ``min(32, size // 2)`` per axis.
    blend : {"linear", "cosine"}
        The cross-fade in the overlaps.

    Raises
    ------
    ParameterError
        If ``size_yx`` is not two positive ints, ``overlap_yx`` not two non-negative ints at most half of
        ``size_yx``, or ``blend`` is unknown; ``path`` names the field.
    """

    size_yx: tuple[int, int]
    overlap_yx: tuple[int, int]
    blend: Literal["linear", "cosine"]

    def __init__(
        self,
        size_yx: tuple[int, int],
        overlap_yx: tuple[int, int] | None = None,
        blend: Literal["linear", "cosine"] = "linear",
    ) -> None:
        where = "FieldTiles"
        sy, sx = ints(size_yx, where=where, name="size_yx", axes="yx", minimum=1)
        if overlap_yx is None:
            oy, ox = min(32, sy // 2), min(32, sx // 2)
        else:
            oy, ox = ints(overlap_yx, where=where, name="overlap_yx", axes="yx", minimum=0)
        for axis, size, overlap in (("y", sy, oy), ("x", sx, ox)):
            if 2 * overlap > size:
                raise ParameterError(
                    f"{where} overlap_yx.{axis} = {overlap} exceeds half of size_yx.{axis} = {size}: the cross-fades "
                    f"of one tile would meet. Pass overlap_yx with at most {size // 2} along {axis}, or leave it out "
                    "for 32 pixels or half the tile, whichever is less",
                    path=f"overlap_yx.{axis}",
                )
        if blend not in ("linear", "cosine"):
            raise ParameterError(f"{where} blend must be linear or cosine, got {blend!r}", path="blend")
        # Frozen: the fields are set through object.__setattr__.
        object.__setattr__(self, "size_yx", (sy, sx))
        object.__setattr__(self, "overlap_yx", (oy, ox))
        object.__setattr__(self, "blend", blend)


class Parts(TypedDict, total=False):
    """The keyword arguments every model takes besides its system: ``__init__(self, ..., **parts: Unpack[Parts])``.

    A modality's constructor passes them on to :class:`~waveorder.models.ReconstructionModel`, which documents each.
    Import it from :mod:`waveorder.models`.
    """

    object_prior: ObjectPrior | None
    noise: Noise | None
    solver: Solver | None
    score: Score | None
    search: Sequence[SearchStage] | None
    z_padding: ZPaddingSetting
    field_tiles: FieldTiles | None
    pixel_size: PixelSize | Literal["auto"] | None
    apodization_rolloff: float
    device: str | torch.device | None


def axial_extent(system: ImagingSystem) -> float:
    """Return ``n * wavelength / NA**2`` in um, the axial extent of the system's point-spread function.

    ``n`` is the system's ``index_of_refraction_media``; ``wavelength`` and ``NA`` are the detection's, the numerical
    aperture at its nominal value when it is free.

    Raises
    ------
    ParameterError
        At ``z_padding``, for a system without ``index_of_refraction_media`` or a detection with a wavelength.
    """
    detection = getattr(system, "detection", None)
    index = getattr(system, "index_of_refraction_media", None)
    if not isinstance(detection, Detection) or not detection.has_wavelength or not isinstance(index, int | float):
        raise ParameterError(
            "z_padding 'auto' reads index_of_refraction_media and the detection's wavelength and numerical aperture, "
            f"which a {type(system).__name__} does not have; give z_padding as an integer",
            path="z_padding",
        )
    free = detection.free_quantities().get("numerical_aperture")
    aperture = detection.fixed_quantities()["numerical_aperture"] if free is None else free.nominal
    return index * detection.wavelength / aperture**2


def least_z_padding(system: ImagingSystem, z_step: float) -> int:
    """Return :func:`axial_extent` in whole slices of ``z_step`` um, rounded up: the least ``"auto"`` pads.

    Raises
    ------
    ParameterError
        As :func:`axial_extent`.
    """
    return math.ceil(round(axial_extent(system) / z_step, 9))  # rounded first, so float error adds no slice


def auto_z_padding(system: ImagingSystem, sampling: Sampling) -> int:
    """Return the slices ``z_padding="auto"`` mirrors at each end of z of a stack on ``sampling``.

    At least :func:`least_z_padding`; then the fewest more that make the padded length ``Z + 2p`` a fast FFT length
    (no prime factor above 7), while ``p`` stays below ``Z``, the most a mirror pads (the least padding when no such
    ``p`` exists).

    Raises
    ------
    ParameterError
        As :func:`axial_extent`.
    ShapeError
        If the stack has no more slices than that least padding.
    """
    z, step = sampling.shape[0], sampling.pixel_size[0]
    least = least_z_padding(system, step)
    if least >= z:
        raise ShapeError(
            f"z_padding 'auto' pads at least {least} slices (n * wavelength / NA**2 = {axial_extent(system):.4g} um "
            f"at a z step of {step:g} um), but the stack has {z}, and a mirror pads fewer slices than the stack "
            f"holds; give z_padding as an integer below {z}"
        )
    return next((p for p in range(least, z) if _fast(z + 2 * p)), least)


def _fast(length: int) -> bool:
    """Whether ``length`` has no prime factor above 7."""
    for factor in (2, 3, 5, 7):
        while length % factor == 0:
            length //= factor
    return length == 1
