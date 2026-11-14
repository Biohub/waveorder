"""The registry: a name for every part class, whose constructor arguments are its plain-data form.

A part (system, object prior, noise model, solver, score, search stage, network piece) is registered under a name,
unique across kinds, and a kind. Its spec is ``{name: arguments}``, or the bare name for a part built with its
defaults: :func:`resolve` builds a part from a spec and :func:`describe` writes one. Training parts and bundles use
:func:`class_spec` and :func:`build` instead, which name a class by its import path
(``{"class_path": "package.module.Class", "init_args": {...}}``), so a class written outside waveorder needs no name.
This module imports no torch.
"""

import dataclasses
import functools
import importlib
import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any, cast

from waveorder.errors import ParameterError, RegistrationError

__all__ = [
    "VALUE",
    "Spec",
    "arguments",
    "build",
    "build_with",
    "class_spec",
    "describe",
    "names",
    "register",
    "resolve",
]

type Spec = str | Mapping[str, Mapping[str, object] | None]
"""A part as plain data: a bare name, or ``{name: arguments}``."""

VALUE = "value"
"""The kind of small frozen dataclasses that are arguments of parts, not parts: ``Free``, ``Loop``."""

_classes: dict[str, tuple[str, type]] = {}
"""Name -> (kind, class)."""
_names: dict[type, str] = {}
"""Class -> name."""
_builders: dict[type, Callable[..., Any]] = {}
"""Class -> the function :func:`resolve` calls in place of the class (:func:`build_with`)."""


def register[C: type](name: str, *, kind: str) -> Callable[[C], C]:
    """Return a decorator that registers a class under ``name``, of ``kind``.

    Each instance then keeps the arguments it was built with (a dataclass reports its fields, and a class may define
    ``arguments()`` instead). A part whose state changes after construction reports an argument at its current value
    through ``current_argument(name, given)``.

    Parameters
    ----------
    name : str
        The name specs use; unique across every kind.
    kind : str
        What the part is: ``"system"``, ``"object_prior"``, ``"noise"``, ``"solver"``, ``"score"``, ``"search"``,
        ``"network"``, ... or :data:`VALUE`.

    Raises
    ------
    RegistrationError
        If ``name`` is taken by another class.
    """

    def decorate(cls: C) -> C:
        held = _classes.get(name)
        if held is not None and held[1] is not cls:
            raise RegistrationError(name, f"is taken by {held[1].__name__}; choose another name")
        if not dataclasses.is_dataclass(cls) and not callable(getattr(cls, "arguments", None)):
            _capture(cls)
        _classes[name] = (kind, cls)
        _names[cls] = name
        return cls

    return decorate


def build_with[F: Callable[..., object]](cls: type) -> Callable[[F], F]:
    """Return a decorator that makes :func:`resolve` build ``cls`` by calling the decorated function instead.

    For a part only a higher layer can build, e.g. :class:`~waveorder.nn.Distilled`, whose bundle :mod:`waveorder.io`
    reads.
    """

    def decorate(function: F) -> F:
        _builders[cls] = function
        return function

    return decorate


def _capture(cls: type) -> None:
    """Wrap ``cls.__init__`` so that each instance of exactly ``cls`` keeps the arguments it was built with."""
    init = cls.__init__
    if cls in _captured:
        return
    parameters = list(inspect.signature(init).parameters.values())[1:]
    positional = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    _captured[cls] = (positional, {p.name: p.default for p in parameters if p.default is not p.empty})

    @functools.wraps(init)
    def captured(self: object, *args: object, **kwargs: object) -> None:
        init(self, *args, **kwargs)
        if type(self) is not cls:  # a subclass is another part, under its own name
            return
        # Plain dict work, not inspect.Signature.bind: extra positionals (*args) are dropped.
        names, defaults = _captured[cls]
        given = defaults | dict(zip(names, args, strict=False)) | kwargs
        object.__setattr__(self, "_arguments", given)

    cls.__init__ = captured  # ty: ignore[invalid-assignment] (the wrapper keeps the signature)


_captured: dict[type, tuple[list[str], dict[str, object]]] = {}
"""Positional names and defaults per wrapped class. A class registered again is not wrapped twice."""


def resolve[T](spec: T | Spec, kind: str, expected: type[T] | None = None) -> T:
    """Return the part ``spec`` describes, or ``spec`` itself when it is already an object.

    Nested specs of registered names are built first, and lists become tuples.

    Parameters
    ----------
    spec : object, str or Mapping
        A part, a bare name, or ``{name: arguments}``.
    kind : str
        The kind the part must be registered as.
    expected : type, optional
        A class the result must be an instance of.

    Raises
    ------
    ParameterError
        If no part of ``kind`` has the name, the class does not take the arguments, or the result is not an
        ``expected``; ``path`` is the name, or the constructor's own path for an error it raises.
    """
    if isinstance(spec, str):
        name, given = spec, {}
    elif isinstance(spec, Mapping):
        if len(spec) != 1:
            raise ParameterError(f"a {kind} is a name or one {{name: arguments}}, got {len(spec)} keys")
        ((name, given),) = spec.items()
        given = {} if given is None else given
    else:
        return spec
    held = _classes.get(name)
    if held is None or held[0] != kind:
        raise ParameterError(f"no {kind} is named {name!r}; choose one of {', '.join(names(kind))}", path=name)
    cls = held[1]
    factory = _builders.get(cls, cls)
    try:
        part = factory(**{key: _value(item) for key, item in given.items()})
    except TypeError as error:
        raise ParameterError(f"the {kind} {name!r} does not take these arguments: {error}", path=name) from error
    if expected is not None and not isinstance(part, expected):
        raise ParameterError(f"the {kind} {name!r} is a {type(part).__name__}, not a {expected.__name__}", path=name)
    return part


def _value(item: object) -> object:
    """Rebuild a nested spec ``{name: arguments}`` of a registered name; lists become tuples."""
    if isinstance(item, Mapping):
        if len(item) == 1:
            ((name, fields),) = item.items()
            if name in _classes and (fields is None or isinstance(fields, Mapping)):
                return resolve({name: fields}, _classes[name][0])
        return {key: _value(entry) for key, entry in item.items()}
    if isinstance(item, list | tuple):
        return tuple(_value(entry) for entry in item)
    return item


def arguments(part: object) -> dict[str, object]:
    """Return the constructor arguments of ``part`` as plain data, each at its current value.

    Raises
    ------
    ParameterError
        If ``part`` neither is a dataclass, nor defines ``arguments()``, nor was built by a registered class.
    """
    given = _given(part)
    if given is None:
        raise ParameterError(f"{type(part).__name__} is not a registered part; register its class by name")
    return {name: _plain(item) for name, item in given.items()}


def _given(part: object) -> dict[str, object] | None:
    """Return the constructor arguments of ``part`` at their current values; ``None`` when the registry cannot tell.

    From ``arguments()``, the fields of a dataclass, or what a registered class captured.
    """
    own = getattr(part, "arguments", None)
    if callable(own):
        given = dict(cast(Callable[[], Mapping[str, object]], own)())
    elif dataclasses.is_dataclass(part):
        given = {field.name: getattr(part, field.name) for field in dataclasses.fields(part) if field.init}
    elif "_arguments" in getattr(part, "__dict__", {}):  # a slotted or built-in object has no __dict__
        given = dict(part.__dict__["_arguments"])
    else:
        return None
    current = getattr(part, "current_argument", None)
    if callable(current):
        given = {name: current(name, item) for name, item in given.items()}
    return given


def describe(part: object) -> dict[str, dict[str, object]]:
    """Return ``{name: arguments}``, the spec :func:`resolve` rebuilds ``part`` from.

    Raises
    ------
    ParameterError
        If the exact class of ``part`` is not registered: a subclass is another part, with its own name.
    """
    name = _names.get(type(part))
    if name is None:
        raise ParameterError(
            f"{type(part).__name__} is not registered, so it has no plain-data form; register it: "
            "@waveorder.registry.register(name, kind=...)"
        )
    return {name: arguments(part)}


def _plain(item: object) -> object:
    if type(item) in _names:
        return describe(item)
    if isinstance(item, Mapping):
        return {str(key): _plain(entry) for key, entry in item.items()}
    if isinstance(item, list | tuple):
        return [_plain(entry) for entry in item]
    return os.fspath(item) if isinstance(item, os.PathLike) else item


def names(kind: str) -> tuple[str, ...]:
    """Return the names registered as ``kind``, in registration order."""
    return tuple(name for name, (held, _) in _classes.items() if held == kind)


def class_spec(part: object) -> dict[str, object]:
    """Return ``{"class_path": ..., "init_args": {...}}``, the plain data :func:`build` rebuilds ``part`` from.

    The arguments are those :func:`arguments` reads or, for any other class, the attribute named like each constructor
    parameter. Arguments that are parts are class specs too.

    Raises
    ------
    ParameterError
        If the class is defined inside a function (no path imports it), or a constructor parameter has no attribute of
        its name.
    """
    cls = type(part)
    if "<locals>" in cls.__qualname__:
        raise ParameterError(
            f"{cls.__qualname__} is defined inside a function, so no path imports it; define it at module level"
        )
    given = _given(part)
    if given is None:
        parameters = inspect.signature(cls).parameters.values()
        names = [p.name for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        missing = [name for name in names if not hasattr(part, name)]
        if missing:
            raise ParameterError(
                f"{cls.__name__} keeps no attribute {', '.join(missing)}, so its arguments cannot be written: store "
                "each constructor argument under its own name, or define arguments() that returns them"
            )
        given = {name: getattr(part, name) for name in names}
    path = f"{cls.__module__}.{cls.__qualname__}"
    return {"class_path": path, "init_args": {name: _spec_value(item) for name, item in given.items()}}


def _spec_value(item: object) -> object:
    if item is None or isinstance(item, str | int | float | bool):
        return item
    if isinstance(item, os.PathLike):
        return os.fspath(item)
    if isinstance(item, Mapping):
        return {str(key): _spec_value(entry) for key, entry in item.items()}
    if isinstance(item, list | tuple):
        return [_spec_value(entry) for entry in item]
    return class_spec(item)


def build(spec: object) -> object:
    """Return what a class spec describes: the class imported from ``class_path``, called with ``init_args``.

    Values that are class specs are built first and lists become tuples; other values are passed as they are, so
    ``build(class_spec(part))`` rebuilds ``part``.

    Raises
    ------
    ParameterError
        If ``class_path`` imports nothing, or the class does not take the arguments.
    """
    if isinstance(spec, list | tuple):
        return tuple(build(entry) for entry in spec)
    if not isinstance(spec, Mapping):
        return spec
    if "class_path" not in spec or not set(spec) <= {"class_path", "init_args"}:
        return {key: build(entry) for key, entry in spec.items()}
    path = str(spec["class_path"])
    cls = _imported(path)
    init_args = cast(Mapping[str, object], spec.get("init_args") or {})
    given = {str(key): build(entry) for key, entry in init_args.items()}
    try:
        return cls(**given)
    except TypeError as error:
        raise ParameterError(f"{path} does not take these arguments: {error}", path=path) from error


def _imported(path: str) -> Any:  # noqa: ANN401 (any class)
    """Return the object at the dotted ``path``: the longest importable module, then attributes."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:cut])
        try:
            found: object = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name is not None and not module.startswith(error.name):
                raise  # the module exists, and what it imports is missing
            continue
        for name in parts[cut:]:
            found = getattr(found, name, None)
        if found is not None:
            return found
        break
    raise ParameterError(f"no class is importable as {path!r}; install or put on the path what defines it", path=path)
