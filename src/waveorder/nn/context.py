"""Where a solve happens: the position, time point and region within the field that a solver's data covers.

The position methods and the tile driver set it; a solver that needs it (e.g. a network trained with each patch's
place as input) reads it with :func:`current_context`. It is a :class:`contextvars.ContextVar`, so it follows the
thread and task that set it.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from waveorder.errors import ParameterError


@dataclass(frozen=True, slots=True, kw_only=True)
class SolveContext:
    """What is known about where the data of a solve came from; every field is ``None`` when unknown.

    Attributes
    ----------
    position : str or None
        The position's path within its plate, e.g. ``"A/1/002026"``.
    time : int or None
        The time index.
    field_shape : tuple[int, int] or None
        Pixels of the whole field of view along y and x.
    region : tuple[slice, slice] or None
        The y and x extent, within the field, of the data the solver is given: for a tile, its read region, halo
        included. ``None`` means the data is the whole field.
    """

    position: str | None = None
    time: int | None = None
    field_shape: tuple[int, int] | None = None
    region: tuple[slice, slice] | None = None


_UNKNOWN = SolveContext()
_context: ContextVar[SolveContext] = ContextVar("waveorder.nn.solve_context", default=_UNKNOWN)


def current_context() -> SolveContext:
    """Return the solve context set by the enclosing :func:`solve_context` blocks; all ``None`` outside any."""
    return _context.get()


@contextmanager
def solve_context(
    *,
    position: str | None = None,
    time: int | None = None,
    field_shape: tuple[int, int] | None = None,
    region: tuple[slice, slice] | None = None,
) -> Generator[SolveContext]:
    """Set the solve context for the body; fields left ``None`` keep the value of the enclosing block.

    Parameters
    ----------
    position : str, optional
        The position's path within its plate.
    time : int, optional
        The time index.
    field_shape : tuple[int, int], optional
        Pixels of the whole field of view along y and x.
    region : tuple[slice, slice], optional
        The y and x extent of the solver's data within the field, as two slices with integer bounds.

    Yields
    ------
    SolveContext
        The context in force inside the block.

    Raises
    ------
    ParameterError
        If ``field_shape`` is not two positive ints, or ``region`` is not two slices with ``0 <= start < stop`` and
        no step.
    """
    changes: dict[str, object] = {}
    if position is not None:
        changes["position"] = position
    if time is not None:
        changes["time"] = time
    if field_shape is not None:
        changes["field_shape"] = _field_shape(field_shape)
    if region is not None:
        changes["region"] = _region(region)
    context = replace(_context.get(), **changes)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def _field_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) == 2 and min(shape) >= 1:
        return (shape[0], shape[1])
    raise ParameterError(
        f"solve_context field_shape must be (Y, X), two positive ints, got {shape!r}", path="field_shape"
    )


def _region(region: tuple[slice, slice]) -> tuple[slice, slice]:
    if len(region) == 2 and all(_is_extent(extent) for extent in region):
        return (region[0], region[1])
    raise ParameterError(
        f"solve_context region must be two slices (y, x) with 0 <= start < stop and no step, got {region!r}",
        path="region",
    )


def _is_extent(extent: slice) -> bool:
    start, stop = extent.start, extent.stop
    return extent.step in (None, 1) and start is not None and stop is not None and 0 <= start < stop
