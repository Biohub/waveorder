"""A model and its configuration: build a model from one, describe a model, and save and load a model as a folder.

A model built from a configuration keeps it, with its calibrated values in the ``calibrated`` block. A model built
in Python is described part by part by :func:`waveorder.registry.describe`, which is what :func:`save` writes.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from waveorder import registry
from waveorder.config import CONFIG_ERRORS, DataSettings, V4Config, problems
from waveorder.config import load as load_config
from waveorder.config.build import build
from waveorder.errors import ConfigurationError, IncompatibleModelError, ParameterError, Rejected, named
from waveorder.models._arguments import PixelSize as VoxelSize
from waveorder.models._arguments import expect_instance
from waveorder.models._arguments import pixel_size as voxel_size
from waveorder.models._base import ReconstructionModel
from waveorder.models._parts import FieldTiles, Parts
from waveorder.nn import free_specs, load_parameters
from waveorder.nn.parameters import quantity_number

SCHEMA = "waveorder.model/1"
"""The schema of the index ``model.json`` of a saved model."""

_TILE_PARAMETERS = "tile_parameters.pt"
"""The file of a saved model that holds its per-field-tile values, ``{name: (tiles, ...)}``."""


def from_config[M: ReconstructionModel](cls: type[M], config: object) -> M:
    """Build the model ``config`` describes, keep the configuration, and apply its ``calibrated`` block.

    Raises
    ------
    Rejected
        With the problems of a configuration that is not valid, or whose parts refuse each other: each at the
        config path of the part that refuses (``solver.reconstruction``, ``solver.calibration.score``), with the
        part's own reason.
    """
    where = f"{cls.__name__}.from_config"
    expect_instance(config, (V4Config, Mapping, str, Path), where=where, name="config")
    loaded = load_config(config)  # ty: ignore[invalid-argument-type] (one of the four, checked above)
    source = str(config) if isinstance(config, str | Path) else "the configuration"
    try:
        system, arguments = build(loaded)
    except CONFIG_ERRORS as error:
        raise Rejected(problems(error, source=source)) from error
    if cls.system_types and not isinstance(system, cls.system_types):
        raise Rejected.one(
            f"imaging_system.contrast: builds a {type(system).__name__}, which {cls.__name__} does not wrap; call "
            "ReconstructionModel.from_config",
            location=source,
        )
    tiles = _field_tiles(arguments.pop("field_tiles"), source=source)
    parts = Parts(
        **arguments,
        field_tiles=tiles,
        pixel_size=loaded.data.pixel_size,
        apodization_rolloff=loaded.solver.apodization_rolloff,
    )
    left = _left_at_default(loaded)
    try:
        model = cls.for_system(system, **parts)
    except IncompatibleModelError as error:
        raise Rejected.one(f"solver.reconstruction: {error}", location=source) from error
    except ConfigurationError as error:
        located = error
        if error.field == "device":  # the model's own argument: the configuration writes it as solver.device
            located = ConfigurationError("solver.device", str(error).removeprefix("device: "))
        raise Rejected(problems(located, source=source)) from error
    score = model.score
    if score is not None:
        try:
            score.check(model.imaging_model)
        except IncompatibleModelError as error:
            raise Rejected.one(f"solver.calibration.score: {error}", location=source) from error
    if loaded.calibrated:
        try:
            load_parameters(model.imaging_model, dict(loaded.calibrated))
        except ParameterError as error:
            raise Rejected.one(f"calibrated: {error}", location=source) from error
    model._remember(loaded, left)  # noqa: SLF001 (the model's own bookkeeping, set by its constructor helper)
    return model


def _field_tiles(tiles: Mapping[str, Any] | None, *, source: str) -> FieldTiles | None:
    """Build the configured :class:`FieldTiles`; an error is located at its key of the ``tiled`` block."""
    if tiles is None:
        return None
    try:
        return FieldTiles(**tiles)
    except ParameterError as error:
        at = "imaging_system.field_dependence.tiled"
        argument, _, rest = (error.path or "").partition(".")
        key = "tile_blend" if argument == "blend" else argument
        located = ConfigurationError(".".join(part for part in (at, key, rest) if part), str(error))
        raise Rejected(problems(located, source=source)) from error


def _left_at_default(config: V4Config) -> tuple[str, ...]:
    """Return which of ``object_prior`` and ``noise`` the configuration does not set: no weight, no noise level."""
    names = []
    if isinstance(config.object.prior, str) and "weight" not in config.object.model_fields_set:
        names.append("object_prior")
    noise = config.noise
    level = noise.gaussian is not None and bool({"sigma", "snr_db"} & noise.gaussian.model_fields_set)
    if noise.poisson is None and noise.poisson_gaussian is None and not level:
        names.append("noise")
    return tuple(names)


def calibrated(model: ReconstructionModel) -> dict[str, float]:
    """Return the scalar free quantities of the model that moved from their nominal value, by parameter name."""
    parameters = dict(model.imaging_model.named_parameters())
    moved = {}
    for name, spec in free_specs(model.imaging_model).items():
        tensor = parameters.get(name)
        if tensor is not None and tensor.numel() == 1:
            value = quantity_number(tensor, spec.nominal)
            if value != spec.nominal:
                moved[name] = value
    return moved


def to_config(
    model: ReconstructionModel,
    *,
    channels: Sequence[str] | None,
    pixel_size: VoxelSize | None,
    times: Sequence[int] | int | None,
) -> V4Config | dict[str, object]:
    """Return the configuration the model was built from with its ``calibrated`` block, or the model's description.

    The ``data`` block is replaced by the arguments given; the description of a model built in Python carries them
    as its ``data`` entry.

    Raises
    ------
    ParameterError
        At ``pixel_size`` or ``times``, if given and not valid.
    ConfigurationError
        At the model argument of a part whose class is not registered, for a model built in Python.
    """
    where = f"{type(model).__name__}.to_config"
    data: dict[str, object] = {}
    if channels is not None:
        data["input_channel_names"] = [channels] if isinstance(channels, str) else list(channels)
    size = model.pixel_size if pixel_size is None else pixel_size
    if size is not None:
        data["pixel_size"] = dict(zip("zyx", voxel_size(size, where=where), strict=True))
    if times is not None:
        _expect_times(times, where=where)
        data["time_indices"] = times if isinstance(times, int) else list(times)
    stored = model.config
    if stored is not None:
        update: dict[str, object] = {"calibrated": calibrated(model)}
        if data:
            update["data"] = DataSettings.model_validate(stored.data.model_dump() | data)
        return stored.model_copy(update=update)
    described = describe(model)
    return described | ({"data": data} if data else {})


def describe(model: ReconstructionModel) -> dict[str, Any]:
    """Return the model as plain data: its class, each part as ``{name: arguments}``, and its settings.

    A solver the model chose itself is written as ``None``, so the loaded model chooses it again from its parts.

    Raises
    ------
    ConfigurationError
        At the model argument of a part whose class is not registered.
    """
    held = {
        "system": model.system,
        "object_prior": model.object_prior,
        "noise": model.noise,
        "solver": None if model._solver_is_default else model.solver,  # noqa: SLF001
        "score": model.score,
    }
    parts: dict[str, object] = {}
    for slot, part in held.items():
        parts[slot] = None if part is None else _described(part, slot)
    stages = model.search
    parts["search"] = None if stages is None else [_described(stage, f"search[{i}]") for i, stage in enumerate(stages)]
    tiles = model.field_tiles
    settings = {
        "z_padding": model.z_padding,
        "field_tiles": None
        if tiles is None
        else {"size_yx": list(tiles.size_yx), "overlap_yx": list(tiles.overlap_yx), "blend": tiles.blend},
        "pixel_size": None if model.pixel_size is None else list(model.pixel_size),
        "apodization_rolloff": model.apodization_rolloff,
    }
    return {"schema": SCHEMA, "class": type(model).__name__, "parts": parts, "settings": settings}


def _described(part: object, slot: str) -> object:
    try:
        return registry.describe(part)
    except ParameterError as error:
        raise ConfigurationError(slot, str(error)) from error


def save(model: ReconstructionModel, path: str | Path, *, overwrite: bool) -> Path:
    """Write the model as a folder that :func:`load` reads; return the folder.

    The folder holds ``model.json``, the description, and with :attr:`tile_parameters` their tensors
    (``tile_parameters.pt``).

    Raises
    ------
    Rejected
        At ``path``, with nothing written: it exists and ``overwrite`` is false, or it is a file.
    ConfigurationError
        As :func:`describe`.
    """
    expect_instance(overwrite, bool, where=f"{type(model).__name__}.save", name="overwrite")
    folder = Path(path)
    index = folder / "model.json"
    if folder.exists() and (not folder.is_dir() or (index.exists() and not overwrite)):
        raise Rejected.one(f"exists; pass {named('overwrite')} to replace it", location=str(folder))
    described = describe(model)
    folder.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(described, indent=1))
    stored = folder / _TILE_PARAMETERS
    table = model.tile_parameters
    if table is None:
        stored.unlink(missing_ok=True)
    else:
        torch.save({name: values.detach().cpu() for name, values in table.items()}, stored)
    return folder


def load[M: ReconstructionModel](cls: type[M], source: str | Path | Mapping[str, Any]) -> M:
    """Return the model saved in the folder ``source``, or the one a description (:func:`describe`) holds.

    Raises
    ------
    Rejected
        At ``source``, if it holds no saved model of this schema, or a part cannot be rebuilt (a class that is not
        registered in this process).
    """
    table = None
    if isinstance(source, Mapping):
        described: object = dict(source)
        location = "the description"
    else:
        location = str(source)
        try:
            described = json.loads((Path(source) / "model.json").read_text())
            stored = Path(source) / _TILE_PARAMETERS
            table = torch.load(stored, weights_only=True) if stored.exists() else None
        except (OSError, ValueError, RuntimeError) as error:
            raise Rejected.one(f"holds no saved model: {error}", location=location) from error
    if not isinstance(described, dict) or described.get("schema") != SCHEMA:
        raise Rejected.one(f"holds no saved model of schema {SCHEMA}", location=location)
    held, settings = described["parts"], described["settings"]
    kinds = {"system": "system", "object_prior": "object_prior", "noise": "noise", "solver": "solver", "score": "score"}
    try:
        system = registry.resolve(held["system"], "system")
        parts: dict[str, object] = {
            slot: None if held[slot] is None else registry.resolve(held[slot], kind)
            for slot, kind in kinds.items()
            if slot != "system"
        }
        stages = held["search"]
        parts["search"] = None if stages is None else [registry.resolve(stage, "search") for stage in stages]
    except ParameterError as error:
        raise Rejected.one(f"cannot rebuild the model: {error}", location=location) from error
    tiles = settings["field_tiles"]
    model = cls.for_system(
        system,
        **parts,  # ty: ignore[invalid-argument-type]
        z_padding=settings["z_padding"],
        field_tiles=None if tiles is None else FieldTiles(**tiles),
        pixel_size=settings.get("pixel_size"),
        apodization_rolloff=settings.get("apodization_rolloff", 0.0),
    )
    model.tile_parameters = table
    return model


def _expect_times(times: object, *, where: str) -> None:
    """Raise unless ``times`` is a non-negative int or a list or tuple of them.

    Raises
    ------
    ParameterError
        Naming the method; ``path`` is ``times``.
    """
    entries = list(times) if isinstance(times, list | tuple | range) else [times]
    if all(isinstance(entry, int) and not isinstance(entry, bool) and entry >= 0 for entry in entries):
        return
    raise ParameterError(
        f"{where} times must be None (every time point), a time index or a list of time indices (ints from 0), "
        f"got {times!r}",
        path="times",
    )
