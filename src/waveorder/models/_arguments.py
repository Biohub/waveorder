"""The argument checks of a model's methods, run once at method entry: types, a voxel size and tuples of ints."""

import math
import numbers
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from waveorder.errors import ParameterError
from waveorder.nn.optics import Detection, Illumination
from waveorder.nn.sampling import ZYX

if TYPE_CHECKING:  # importing iohub takes a second; the alias below is evaluated only when it is inspected
    from iohub.ngff import Plate, Position

type PixelSize = float | Sequence[float] | Mapping[str, float] | object
"""A voxel size in um: one number (isotropic), ``(z, yx)``, ``(z, y, x)``, ``{"z": ..., "y": ..., "x": ...}`` (the
mapping of a configuration's ``data.pixel_size``), or an object with ``z``, ``y`` and ``x``."""

type Inputs = str | Path | Position | Plate | Sequence[str | Path | Position | Plate]
"""The ``positions`` argument: paths, globs and plates, open iohub ``Position`` and ``Plate`` nodes, or a sequence."""

_FORMS = "one (isotropic), (z, yx), (z, y, x) or {z, y, x}"


def shown_sizes(values: Sequence[float]) -> str:
    """``0.25, 0.1, 0.1``: a voxel size for messages."""
    return ", ".join(f"{value:.6g}" for value in values)


def shown_shape(values: Sequence[int]) -> str:
    """``32x256x256``: a shape for messages."""
    return "x".join(str(int(value)) for value in values)


def pixel_size(value: object, *, where: str, default: object = None) -> ZYX[float]:
    """Return ``value`` (else ``default``, a model's own) as the voxel size ``(z, y, x)`` in um, or raise.

    Raises
    ------
    ParameterError
        At ``pixel_size``, if neither is given, or the value is none of the forms of :data:`PixelSize` with
        positive finite numbers.
    """
    value = default if value is None else value
    if value is None:
        raise ParameterError(
            f"{where} has no voxel size: set data.pixel_size in the configuration, pass pixel_size= to the "
            "model's constructor, or pass pixel_size= to this call",
            path="pixel_size",
        )
    if isinstance(value, Mapping):
        if frozenset(value) != frozenset("zyx"):
            raise ParameterError(
                f"{where} pixel_size must be positive numbers in um: {_FORMS}; got the keys {sorted(map(str, value))}",
                path="pixel_size",
            )
        value = [value[axis] for axis in "zyx"]
    elif all(hasattr(value, axis) for axis in "zyx"):
        value = [getattr(value, axis) for axis in "zyx"]
    listed = value.tolist() if hasattr(value, "tolist") else value  # ty: ignore[call-non-callable] (a tensor, an array)
    if isinstance(listed, numbers.Real):
        listed = [listed]
    plain = isinstance(listed, Iterable) and not isinstance(listed, str | bytes | Mapping | set)
    entries = list(listed) if plain else []
    if len(entries) == 1:
        entries *= 3
    elif len(entries) == 2:
        entries = [entries[0], entries[1], entries[1]]
    if len(entries) != 3 or not all(
        isinstance(size, numbers.Real) and not isinstance(size, bool) and 0 < float(size) < math.inf for size in entries
    ):
        raise ParameterError(
            f"{where} pixel_size must be positive numbers in um: {_FORMS}; got {value!r}",
            path="pixel_size",
        )
    z, y, x = (float(size) for size in entries)
    return (z, y, x)


def zyx(value: object, *, where: str, name: str, minimum: int) -> ZYX[int]:
    """Return ``value`` as a ``(z, y, x)`` tuple of ints at least ``minimum``, or raise :class:`ParameterError`."""
    z, y, x = ints(value, where=where, name=name, axes="zyx", minimum=minimum)
    return (z, y, x)


def ints(value: object, *, where: str, name: str, axes: str, minimum: int) -> tuple[int, ...]:
    """Return ``value`` (a tuple or list) as one int per axis of ``axes``, each at least ``minimum``.

    Raises
    ------
    ParameterError
        At ``name`` or ``name.<axis>``.
    """
    expect_instance(value, (tuple, list), where=where, name=name)
    entries = list(value) if isinstance(value, tuple | list) else []
    if len(entries) != len(axes):
        listed = ", ".join(axes)
        raise ParameterError(f"{where} {name} must have {len(axes)} entries ({listed}), got {len(entries)}", path=name)
    for axis, entry in zip(axes, entries, strict=True):
        expect_instance(entry, int, where=where, name=f"{name}.{axis}")
        if entry < minimum:
            raise ParameterError(
                f"{where} {name}.{axis} must be at least {minimum}, got {entry}", path=f"{name}.{axis}"
            )
    return tuple(entries)


def expect_instance(value: object, kind: type | tuple[type, ...], *, where: str, name: str) -> None:
    """Raise :class:`ParameterError` at ``name`` unless ``value`` is an instance of ``kind``.

    The message reads ``"{where} {name} must be {kinds}, got {type}"``. A ``bool`` is refused where a number is
    expected (unless ``bool`` is listed), and a class passed for an instance of it gets the hint ``"pass an instance,
    FISTA()"``. Include ``type(None)`` in ``kind`` to allow ``None``.
    """
    kinds = kind if isinstance(kind, tuple) else (kind,)
    wanted = " or ".join(_described(accepted) for accepted in kinds)
    if isinstance(value, type) and not any(isinstance(k, type) and issubclass(k, type) for k in kinds):
        hint = f"; pass an instance, {value.__qualname__}()" if any(_is_kind(value, k) for k in kinds) else ""
        raise ParameterError(f"{where} {name} must be {wanted}, got the class {_type_name(value)}{hint}", path=name)
    numeric = any(accepted in (int, float) for accepted in kinds) and bool not in kinds
    if isinstance(value, kinds) and not (numeric and isinstance(value, bool)):
        return
    raise ParameterError(f"{where} {name} must be {wanted}, got {_type_name(type(value))}", path=name)


_ABSENT = object()


def expect_optics(
    where: str, *, detection: object, index_of_refraction_media: object, illumination: object = _ABSENT
) -> None:
    """Raise :class:`ParameterError` at the argument unless the optics a model's constructor takes have their types.

    ``illumination`` is checked when given (a phase model's); the index of refraction is a number.
    """
    if illumination is not _ABSENT:
        expect_instance(illumination, Illumination, where=where, name="illumination")
    expect_instance(detection, Detection, where=where, name="detection")
    expect_instance(index_of_refraction_media, (int, float), where=where, name="index_of_refraction_media")


def _is_kind(cls: type, kind: type) -> bool:
    """Whether instances of ``cls`` would be accepted as ``kind``; a protocol is matched by its member names."""
    members = getattr(kind, "__protocol_attrs__", None)
    if members is not None:
        return all(hasattr(cls, member) for member in members)
    return issubclass(cls, kind)


def _described(kind: type) -> str:
    """``"None"``, ``"an int"``, ``"a torch.Tensor"``: a type's name with its article."""
    if kind is type(None):
        return "None"
    label = _type_name(kind)
    return f"{'an' if label[0].lower() in 'aeiou' else 'a'} {label}"


def _type_name(kind: type) -> str:
    """Return a builtin's bare name, else the shortest imported module path that exports the type."""
    if kind.__module__ == "builtins":
        return kind.__qualname__
    parts = kind.__module__.split(".")
    for end in range(1, len(parts) + 1):
        module = sys.modules.get(".".join(parts[:end]))
        if module is not None and getattr(module, kind.__qualname__, None) is kind:
            return f"{'.'.join(parts[:end])}.{kind.__qualname__}"
    return f"{kind.__module__}.{kind.__qualname__}"
