"""The small value types every module and verb shares: the voxel grid, the tensor layouts, and how a solve ended.

The object ``f`` holds one real volume per named field, data one real volume per channel, and a shift-invariant
system one ``(C, F)`` matrix per spatial frequency on the ``rfftn`` half spectrum.
"""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from torch import Tensor

from waveorder.errors import ParameterError, ShapeError

type ZYX[T] = tuple[T, T, T]
"""One value per axis, in array order: z, y, x."""

type Object = Annotated[Tensor, "(F, Z, Y, X) real"]
"""Object fields, one volume per name in ``ImagingSystem.fields``."""

type Data = Annotated[Tensor, "(C, Z, Y, X) real"]
"""Measured or simulated data, one volume per channel."""

type HalfTransfer = Annotated[Tensor, "(C, F, Z, Y, X // 2 + 1, 2) or (C, F, Z, Y, X // 2 + 1) real"]
"""A shift-invariant transfer matrix on the ``rfftn`` half spectrum, FFT order along z and y, stored real.

With a trailing axis of size 2 it holds the real and imaginary parts of a complex ``H`` (``torch.view_as_real``);
without it, ``H`` is real (an even point-spread function).
"""

type Status = Literal["converged", "direct", "budget", "max_iterations", "nonfinite"]
"""How a solve ended, in the style of lineax's ``RESULTS``.

- ``"converged"``: a tolerance was requested, checked and met.
- ``"direct"``: a closed-form or direct solve ran with no requested check. Nothing was read back, so finiteness is
  not known on the host; see the result's ``finite``.
- ``"budget"``: a fixed iteration count ran with no requested criterion. This is also the status wherever a check
  cannot be read back (under ``vmap`` and ``grad``); per-item diagnostics (a residual, a
  finiteness flag) are then tensors on the result.
- ``"max_iterations"``: a criterion was requested and an iterative method did not meet it.
- ``"nonfinite"``: a requested check saw inf or nan.

Only ``"converged"`` counts as converged. With a tolerance, ``"max_iterations"`` and ``"nonfinite"`` each come with a
:class:`~waveorder.errors.ConvergenceWarning`.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampling:
    """Shape and voxel size of the data *g*, along z, y and x.

    Every field is a Python number, never a tensor, so shapes derived from it are static.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Voxel counts along z, y, x: positive Python ints; any sequence is stored as a tuple.
    pixel_size : tuple[float, float, float]
        Voxel size along z, y, x in um: positive finite Python numbers, stored as a tuple.
    z_padding : int
        Slices at each end of z that pad the data (:class:`~waveorder.nn.ZPadding`), included in ``shape``; 0 by
        default. :meth:`upsampled` refines the other slices only, as wo-4.0 computed its transfer functions.

    Raises
    ------
    ShapeError
        If ``shape`` or ``pixel_size`` does not hold three values, ``shape`` holds a count below 1, or
        ``z_padding`` is not an int from 0 to below half of ``shape[0]``.
    ParameterError
        If ``pixel_size`` holds anything but positive finite numbers.
    """

    shape: ZYX[int]
    pixel_size: ZYX[float]
    z_padding: int = 0

    def __post_init__(self) -> None:
        for name in ("shape", "pixel_size"):
            value = tuple(getattr(self, name))
            if len(value) != 3:
                raise ShapeError(f"Sampling.{name} must be three values (z, y, x), got {value!r}")
            # Frozen: the stored tuple replaces the given sequence through object.__setattr__.
            object.__setattr__(self, name, value)
        if min(self.shape) < 1:
            raise ShapeError(f"Sampling.shape must be three positive int voxel counts (z, y, x), got {self.shape}")
        if not all(d > 0 and math.isfinite(d) for d in self.pixel_size):
            raise ParameterError(
                f"Sampling.pixel_size must be three positive finite numbers in um (z, y, x), got {self.pixel_size}"
            )
        if not isinstance(self.z_padding, int) or not 0 <= 2 * self.z_padding < self.shape[0]:
            raise ShapeError(
                f"Sampling.z_padding must be an int from 0 to below half of the {self.shape[0]} slices, got "
                f"{self.z_padding!r}"
            )

    def upsampled(self, factors: ZYX[int]) -> Self:
        """Return the grid sampled ``factors`` times more finely along each axis.

        The measured slices keep their field of view; the ``z_padding`` slices at each end stay as many slices of the
        finer step, as wo-4.0 oversampled a padded stack: ``(Z - 2p) kz + 2p`` slices along z.
        """
        (nz, ny, nx), (dz, dy, dx), (kz, ky, kx) = self.shape, self.pixel_size, factors
        p = self.z_padding
        shape = ((nz - 2 * p) * kz + 2 * p, ny * ky, nx * kx)
        return dataclasses.replace(self, shape=shape, pixel_size=(dz / kz, dy / ky, dx / kx))


def voxel_size(value: Sequence[float], *, where: str) -> ZYX[float]:
    """Return ``value`` as the voxel size ``(z, y, x)`` in um, or raise naming the caller.

    Parameters
    ----------
    value : Sequence[float]
        Three positive finite numbers in any sequence: a tuple, a list, a 1-D tensor.
    where : str
        The caller, for the message.

    Raises
    ------
    ParameterError
        If ``value`` does not hold three positive finite numbers; ``path`` is ``pixel_size``.
    """
    sizes = [float(size) for size in value]
    if len(sizes) != 3 or not all(0 < size < math.inf for size in sizes):
        raise ParameterError(
            f"{where} pixel_size must be three positive numbers (z, y, x) in um, got {value!r}", path="pixel_size"
        )
    z, y, x = sizes
    return (z, y, x)


def expect_shape(tensor: Tensor, shape: tuple[int, ...], *, what: str) -> None:
    """Raise :class:`ShapeError` unless ``tensor`` has exactly ``shape``; broadcasting is never implied."""
    if tuple(tensor.shape) != shape:
        raise ShapeError(
            f"{what} has shape {tuple(tensor.shape)}, expected {shape}; batch with torch.func.vmap, not extra dims"
        )


def expect_real(tensor: Tensor, *, what: str) -> None:
    """Raise :class:`ShapeError` if ``tensor`` is complex: objects and data are real."""
    if tensor.is_complex():
        raise ShapeError(f"{what} must be real, got {tensor.dtype}")
