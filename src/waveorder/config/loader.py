"""Read a configuration and say what is wrong with it, each problem at its dotted config path.

:func:`load` reads a ``version: 4`` YAML file or a mapping and returns the validated :class:`V4Config`, or raises
:class:`~waveorder.errors.Rejected` with one :class:`~waveorder.errors.Problem` per error. A 3.x file (no
``version``) is refused with the ``wo config migrate`` command that converts it. :func:`problems` turns any error of
:data:`CONFIG_ERRORS` into problems, for the callers that build a model from a valid configuration.
"""

import difflib
import warnings
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from waveorder.config.migrate import bounded
from waveorder.config.schema import V4Config
from waveorder.errors import (
    ConfigurationError,
    ConfigurationWarning,
    ParameterError,
    Problem,
    Rejected,
    UnsupportedOptionError,
)

CONFIG_ERRORS = (ValidationError, yaml.YAMLError, ConfigurationError, UnsupportedOptionError, ParameterError)
"""What reading, validating or building a configuration raises for a configuration that cannot run."""


class _UniqueKeys(yaml.SafeLoader):
    """A YAML loader that refuses a key given twice in one mapping, which ``safe_load`` silently overwrites.

    A key a merge (``<<: *anchor``) brings in may be given again: that overrides it, as YAML means.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:  # noqa: FBT001, FBT002
        seen = set()
        for key, _ in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                continue
            name = self.construct_object(key, deep=deep)
            if name in seen:
                raise ConstructorError(
                    "while reading a mapping", node.start_mark, f"the key {name!r} is given twice", key.start_mark
                )
            seen.add(name)
        return super().construct_mapping(node, deep=deep)


def parse(text: str) -> Any:  # noqa: ANN401
    """Return the YAML document ``text``, refusing a key given twice."""
    return yaml.load(text, Loader=_UniqueKeys)


def read(path: str | Path) -> dict[str, Any]:
    """Return the YAML mapping of the configuration file at ``path``, refusing a 3.x file.

    Raises
    ------
    ConfigurationError
        At the file, if it is empty, not a mapping, or has no ``version`` (a 3.x configuration); the message gives
        the ``wo config new`` or ``wo config migrate`` command.
    OSError, yaml.YAMLError
        If the file cannot be read or parsed, or a key is given twice.
    """
    data = parse(Path(path).read_text())
    expect_version(data, name=str(path))
    return data


def expect_version(data: object, *, name: str) -> None:
    """Refuse an empty document, one that is not a mapping, and a 3.x one (no ``version``), at ``name``.

    Raises
    ------
    ConfigurationError
        At ``name``, with the command that writes a valid configuration.
    """
    if data is None or (isinstance(data, Mapping) and not data):
        raise ConfigurationError(name, "is empty; print a configuration to start from with: wo config new")
    if not isinstance(data, Mapping):
        raise ConfigurationError(name, f"must hold a YAML mapping, got {type(data).__name__}")
    if "version" not in data:
        file = Path(name).is_file()  # a mapping has no file to name
        stem = Path(name).stem if file and name.endswith((".yml", ".yaml")) else None
        then = f", then use {Path(name).with_name(f'{stem}_v4.yml')}" if stem else ""
        raise ConfigurationError(
            name,
            "has no version, so it is a 3.x configuration; convert it with: "
            f"wo config migrate {name if file else 'FILE'}{then}",
        )


def load(source: str | Path | Mapping[str, object] | V4Config) -> V4Config:
    """Return the validated configuration of a YAML file or a mapping, or refuse it with every problem.

    A free aperture or tilt zenith written without bounds, as wo-4.0 took it, gets the range ``wo config migrate``
    writes, with a warning at its key.

    Parameters
    ----------
    source : str, Path, Mapping or V4Config
        The path of a ``version: 4`` YAML file, the mapping of one, or a configuration (returned as it is).

    Raises
    ------
    Rejected
        With one problem per error, each naming the dotted path of the setting, located at the file.
    """
    if isinstance(source, V4Config):
        return source
    name = str(source) if isinstance(source, str | Path) else "the configuration"
    try:
        if isinstance(source, str | Path):
            data: object = read(source)
        else:
            data = source
            expect_version(data, name=name)
    except OSError as error:
        raise Rejected.one(
            f"cannot be read as a configuration file: {error.strerror or error}", location=name
        ) from error
    except CONFIG_ERRORS as error:
        raise Rejected(problems(error, source=name)) from error
    wo40 = _wo40(data)
    migrate = f"wo config migrate {name if isinstance(source, str | Path) else 'FILE'}"
    ranges: tuple[str, ...] = ()
    if isinstance(data, Mapping):
        data, ranges = bounded(data)  # wo-4.0 took free apertures and zeniths without bounds
    try:
        loaded = V4Config.model_validate(data)
    except CONFIG_ERRORS as error:
        found = problems(error, source=name)
        if wo40:
            found.append(Problem(f"{_WO40}; convert it with: {migrate}", name))
        raise Rejected(found) from error
    for note in ranges:
        field, _, message = note.partition(" ")
        warnings.warn(ConfigurationWarning(message, field=field), stacklevel=2)
    if wo40:
        warnings.warn(
            ConfigurationWarning(f"{_WO40}; if wo-4.0 wrote it, convert it with: {migrate}", field=name),
            stacklevel=2,
        )
    return loaded


_WO40 = (
    "data.pixel_size is written {z, yx}, as wo-4.0 writes it, and wo-4.0 files have other defaults "
    "(detection.edge: sigmoid, a 3D z_focus_offset read as 0)"
)


def _wo40(data: object) -> bool:
    """Whether ``data`` spells the pixel size as wo-4.0 files do, ``{z, yx}``."""
    block = data.get("data") if isinstance(data, Mapping) else None
    size = block.get("pixel_size") if isinstance(block, Mapping) else None
    return isinstance(size, Mapping) and "yx" in size


def problems(error: Exception, *, source: str) -> list[Problem]:
    """Return the problems of a configuration error, each at the dotted path of its setting, located at ``source``.

    An unknown key names the known key it most resembles.
    """
    if isinstance(error, ValidationError):
        found = []
        for line in error.errors():
            path = tuple(line["loc"])
            text = str(line["msg"]).removeprefix("Value error, ")
            if line["type"] == "extra_forbidden":
                close = suggest_key(path)
                text = "is not a configuration key" + (f"; did you mean {close}?" if close else "")
            dotted = ".".join(str(part) for part in path)
            found.append(Problem(f"{dotted}: {text}" if dotted else text, source))
        return found
    if isinstance(error, ConfigurationError | UnsupportedOptionError) and error.field == source:
        return [Problem(str(error).removeprefix(f"{error.field}: "), source)]
    if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
        mark = error.problem_mark
        message = " ".join(part for part in (error.context, error.problem) if part)
        return [Problem(message, f"{source}:{mark.line + 1}:{mark.column + 1}")]
    return [Problem(str(error), source)]


_UNCHOSEN = frozenset(
    {
        "simulate",
        "object.source",
        "noise.gaussian",
        "noise.poisson",
        "noise.poisson_gaussian",
        "noise.gaussian.snr_db",
        "noise.poisson.photons",
        "imaging_system.detection.wavelength",
        "imaging_system.detection.pinhole.diameter_au",
        "imaging_system.detection.pinhole.diameter_um",
    }
)
"""Paths whose null says only "not chosen", which :func:`dump` leaves out."""


def dump(config: V4Config) -> dict[str, Any]:
    """Return ``config`` as a plain mapping with every default filled in, without the nulls that say "not chosen".

    The mapping reads back as the same configuration; an empty ``calibrated`` block is left out.
    """

    def chosen(value: object, path: str) -> object:
        if isinstance(value, dict):
            return {
                key: chosen(item, f"{path}.{key}" if path else key)
                for key, item in value.items()
                if not (item is None and (f"{path}.{key}" if path else key) in _UNCHOSEN)
            }
        return [chosen(item, path) for item in value] if isinstance(value, list) else value

    shown = chosen(config.model_dump(mode="json"), "")
    assert isinstance(shown, dict)
    if not shown.get("calibrated"):
        shown.pop("calibrated", None)
    return shown


def dump_yaml(config: V4Config) -> str:
    """Return :func:`dump` as YAML, in the order of the schema (what ``wo config show`` prints)."""
    return yaml.safe_dump(dump(config), sort_keys=False)


@cache
def _json_schema() -> dict[str, Any]:
    return V4Config.model_json_schema()


def _branches(node: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the object schemas ``node`` stands for, through ``$ref`` and ``anyOf``/``oneOf``/``allOf``."""
    if "$ref" in node:
        return _branches(root["$defs"][node["$ref"].rsplit("/", 1)[-1]], root)
    found = [node]
    for key in ("anyOf", "oneOf", "allOf"):
        found += [branch for item in node.get(key, ()) for branch in _branches(item, root)]
    return found


def suggest_key(path: tuple[int | str, ...]) -> str | None:
    """Return the dotted path of the key the schema takes beside ``path[-1]`` that is closest to it, or ``None``."""
    if not path or not isinstance(path[-1], str):
        return None
    root = _json_schema()
    nodes = _branches(root, root)
    for part in path[:-1]:
        below = []
        for node in nodes:
            child = node.get("items") if isinstance(part, int) else node.get("properties", {}).get(part)
            if child is None and isinstance(node.get("additionalProperties"), dict):
                return None
            if child is not None:
                below += _branches(child, root)
        nodes = below
    known = sorted({name for node in nodes for name in node.get("properties", {})})
    close = difflib.get_close_matches(path[-1], known, n=1)
    return ".".join(str(part) for part in (*path[:-1], close[0])) if close else None
