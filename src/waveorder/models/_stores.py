"""What a transfer-function store means to a model: whether it fits, its prior and start, its transfer functions."""

import dataclasses
import hashlib
import logging
import math
import os
import socket
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from waveorder.errors import ParameterError, Problems, Rejected, StoreWarning, WaveorderError, named
from waveorder.io.tf_store import FORMAT, TfStore, TransferArray, canonical_json, parameters_sha256, read_transfer
from waveorder.models._arguments import shown_shape, zyx
from waveorder.models._base import ReconstructionModel
from waveorder.models._parts import FieldTiles
from waveorder.nn import Free, ParameterPrior, Sampling, ZPadding, free_specs
from waveorder.nn.parameters import shortest_decimal
from waveorder.nn.systems.base import ShiftInvariantSystem
from waveorder.tiling import TilePlan

log = logging.getLogger(__name__)

type ZYX = tuple[int, int, int]
type PixelSize = tuple[float, float, float]
type StoreSource = TfStore | str | Path

SYSTEM_PREFIX = "system."
"""Prefix of the dotted names of imaging-system quantities (``free_specs`` names)."""

_FREE = "free"
"""How a free quantity's place in an imaging-system block is shown in a mismatch."""


def warn(message: str, *, location: str | None) -> None:
    """Emit a :class:`~waveorder.errors.StoreWarning`."""
    warnings.warn(StoreWarning(message, location=location), stacklevel=2)


def where(argument: str, path: object) -> str:
    """Return the location of a store an argument gave: the argument as the caller spells it, then the path."""
    return named(argument) if path is None else f"{named(argument)} {path}"


def _shown_store(store: TfStore) -> str:
    return "the store" if store.path is None else str(store.path)


def read_store(source: StoreSource, argument: str) -> TfStore:
    """Return the store the argument ``argument`` gave, reading it when it is a path.

    Raises Rejected at the argument if ``source`` is no store or path (a prior object is told where it goes), or
    cannot be read.
    """
    if isinstance(source, TfStore):
        return source
    if not isinstance(source, str | Path):
        message = f"must be a TfStore or the path of a tf.zarr store, got {type(source).__name__}"
        if isinstance(source, ParameterPrior):
            message += (
                ": a method on a position takes its prior as a store (its values become the prior centres); a "
                "waveorder.nn.ParameterPrior goes to model.calibrate or model.calibrate_tiles, and "
                "TfStore.to_prior(specs) turns a store into one"
            )
        raise Rejected.one(message, location=named(argument))
    try:
        return TfStore.load(source)
    except Rejected as error:
        location = where(argument, source)
        raise Rejected([dataclasses.replace(problem, location=location) for problem in error.problems]) from None


# --- the imaging system a store was made for -----------------------------------------------------------------------


def _system_block(model: ReconstructionModel) -> dict[str, object] | None:
    """Return :func:`waveorder.registry.describe` of the model's system, free quantities as ``free``, and its tiles.

    The system's ``cache`` argument says how transfers are kept, not what they are, and is left out, as is a
    ``pupil_aberrations`` of ``None`` (an aberration-free pupil), so the hash is the one stores recorded before
    detections took that argument. ``None`` when a part of the system has no plain-data form (it is not registered).
    """
    from waveorder import registry

    def shown(node: object) -> object:
        if isinstance(node, dict):
            if set(node) == {"free"}:  # the registry's spec of a Free
                return _FREE
            return {
                key: shown(value) for key, value in node.items() if not (key == "pupil_aberrations" and value is None)
            }
        if isinstance(node, list):
            return [shown(value) for value in node]
        if node is None or isinstance(node, str | int | float):
            return node
        raise TypeError(f"{type(node).__name__} has no plain-data form")

    try:
        ((name, arguments),) = registry.describe(model.system).items()
        block: dict[str, object] = {name: shown({key: value for key, value in arguments.items() if key != "cache"})}
    except (ParameterError, TypeError):
        return None
    tiles = model.field_tiles
    if tiles is not None:
        block["field_tiles"] = {
            "size_yx": list(tiles.size_yx),
            "overlap_yx": list(tiles.overlap_yx),
            "blend": tiles.blend,
        }
    return block


def system_record(model: ReconstructionModel) -> dict[str, object]:
    """Return ``imaging_system`` and ``imaging_system_sha256`` of a store's provenance; empty with no such form.

    The block is the registry's description of the system, each free quantity as ``free``, so models that differ
    only in their free quantities' priors or values share the hash.
    """
    block = _system_block(model)
    if block is None:
        return {}
    return {
        "imaging_system_sha256": hashlib.sha256(canonical_json(block).encode()).hexdigest(),
        "imaging_system": block,
    }


def system_sha256(model: ReconstructionModel) -> str | None:
    """Return the sha256 of the model's live imaging system and field tiles, free quantities as ``free``."""
    digest = system_record(model).get("imaging_system_sha256")
    return digest if isinstance(digest, str) else None


def _system_problems(store: TfStore, model: ReconstructionModel) -> list[str]:
    record = system_record(model)
    if not record:  # the model has no configuration form to compare
        return []
    recorded = store.provenance.get("imaging_system_sha256")
    if recorded == record["imaging_system_sha256"]:
        return []
    fix = "recalibrate for this system"
    if store.origin == "#592":
        return [f"was written by #592, which records no imaging_system_sha256; {fix}"]
    if recorded is None:
        return [f"records no imaging_system_sha256; {fix}"]
    stored, own = store.provenance.get("imaging_system"), record["imaging_system"]
    fields: list[str] = []
    if isinstance(stored, Mapping) and isinstance(own, Mapping):
        _diff(stored, own, prefix="", found=fields)
    shown = [f"imaging_system.{line}".replace(f'"{_FREE}"', _FREE) for line in fields]
    detail = f" ({'; '.join(shown)})" if shown else ""
    return [f"was made for another imaging_system{detail}; {fix}"]


def _diff(old: object, new: object, *, prefix: str, found: list[str]) -> None:
    """Add ``"path: old -> new"`` to ``found`` for each value that differs between two nested mappings."""
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in sorted(set(old) | set(new), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old:
                found.append(f"{path}: (unset) -> {canonical_json(new[key])}")
            elif key not in new:
                found.append(f"{path}: {canonical_json(old[key])} -> (unset)")
            else:
                _diff(old[key], new[key], prefix=path, found=found)
    elif canonical_json(old) != canonical_json(new):
        found.append(f"{prefix}: {canonical_json(old)} -> {canonical_json(new)}")


# --- the transfer functions a store holds ---------------------------------------------------------------------------

_REUSED = ("pixel_size_zyx_um", "imaging_system_sha256", "parameters_sha256", "dtype", "device")
"""What a stored transfer function was computed for besides its grid (:func:`transfer_record`), all to match."""

_HASHED = {"imaging_system_sha256": "another imaging system", "parameters_sha256": "other parameter values"}


def float64(values: Tensor) -> Tensor:
    """Widen to float64 on the CPU; float32 values become their shortest decimal."""
    values = values.detach().cpu()
    if values.dtype != torch.float32:
        return values.double()
    shortest = [shortest_decimal(value, torch.float32) for value in values.double().flatten().tolist()]
    return torch.tensor(shortest, dtype=torch.float64).reshape(values.shape)


def current_values(model: ReconstructionModel) -> tuple[tuple[str, ...], Tensor, bool]:
    """Return the free quantities' names, their values as float64 rows, and whether they are the nominal ones."""
    specs = free_specs(model.imaging_model)
    names = tuple(specs)
    table = model.tile_parameters
    if table is not None:
        return names, torch.stack([float64(table[name]) for name in names], dim=1), False
    values = dict(model.imaging_model.named_parameters())
    row = [float(float64(values[name].reshape(()))) for name in names]
    nominal = all(
        math.isclose(value, float(specs[name].nominal), rel_tol=1e-6, abs_tol=1e-12)
        for name, value in zip(names, row, strict=True)
    )
    return names, torch.tensor(row, dtype=torch.float64).reshape(1, len(names)), nominal


def transfer_record(model: ReconstructionModel, sampling: Sampling) -> dict[str, object]:
    """Return what the model's transfer function on ``sampling`` is computed for, as a store records it.

    ``sampling`` is the padded grid; :func:`_transfer_at` adds the ``z_padding`` it was padded by.
    """
    names, values, _ = current_values(model)
    return {
        "shape_zyx": list(sampling.shape),
        "pixel_size_zyx_um": list(sampling.pixel_size),
        "parameters_sha256": parameters_sha256(names, values),
        "imaging_system_sha256": system_sha256(model),
        "dtype": str(model.dtype).removeprefix("torch."),
        "device": model.device.type,
    }


@dataclass(kw_only=True, eq=False)
class StoredTransfers:
    """The transfers of the saved store a model loaded, by grid; workers get it pickled, without ``last``."""

    path: Path
    listed: dict[ZYX, TransferArray]
    said: set[str] = field(default_factory=set)
    last: tuple[ZYX, Tensor] | None = None

    def __getstate__(self) -> dict[str, object]:
        return {**self.__dict__, "last": None}


def stored_transfers(store: TfStore, model: ReconstructionModel) -> StoredTransfers | None:
    """Return what of a saved ``waveorder.tf/2`` store a shift-invariant model's binds may use; else ``None``."""
    if store.path is None or store.origin != FORMAT or not store.transfers or model.field_tiles is not None:
        return None
    listed = {item.shape: item for item in store.transfers}
    return StoredTransfers(path=store.path, listed=listed) if isinstance(model.system, ShiftInvariantSystem) else None


def seed_stored_transfer(model: ReconstructionModel, sampling: Sampling) -> None:
    """Hand the system the loaded store's transfer on ``sampling`` if it fits (:data:`_REUSED`); log once which, why."""
    stored, system = model._stored_transfers, model.system  # noqa: SLF001 (set by load_calibration)
    if stored is None or not isinstance(system, ShiftInvariantSystem) or system.state_key() is None:
        return  # nothing stored, or a transfer that carries gradients
    item = stored.listed.get(sampling.shape)
    reasons = _differences(stored, item, model=model, sampling=sampling)
    transfer = _read(stored, item, model) if item is not None and not reasons else None
    grid = shown_shape(sampling.shape)
    message = f"the transfer function on {grid} is read from {stored.path}, not computed"
    if not isinstance(transfer, Tensor) or not system.seed_transfer(sampling, transfer):
        why = "; ".join(reasons or [str(transfer)])
        message = f"the transfer function on {grid} is computed, not read from {stored.path}: {why}"
    if message not in stored.said:
        stored.said.add(message)
        log.info(message)


def _differences(
    stored: StoredTransfers, item: TransferArray | None, *, model: ReconstructionModel, sampling: Sampling
) -> list[str]:
    """Return how what ``item`` was computed for differs from the model on ``sampling``, or nothing."""
    if item is None:
        return [f"it holds them on {', '.join(map(shown_shape, stored.listed))} only"]
    own = transfer_record(model, sampling)
    if own["imaging_system_sha256"] is None:
        return ["the imaging system has no plain-data form to compare"]
    found = [key for key in _REUSED if item.attrs.get(key) != own[key]]
    return [_HASHED.get(key, f"{key} {item.attrs.get(key)} (the model: {own[key]})") for key in found]


def _read(stored: StoredTransfers, item: TransferArray, model: ReconstructionModel) -> Tensor | str:
    """Return the transfer ``item`` lists on the model's device as the system takes it, else why it cannot be read."""
    if stored.last is None or stored.last[0] != item.shape:
        try:
            read = read_transfer(stored.path, item.shape)
        except Rejected as error:
            return f"it cannot be read ({'; '.join(problem.message for problem in error.problems)})"
        if read is None or read.transfer is None or read.attrs != item.attrs:
            return "the store was replaced since it was loaded"
        transfer = torch.view_as_real(read.transfer) if read.transfer.is_complex() else read.transfer
        stored.last = (item.shape, transfer.to(model.device))
    return stored.last[1].to(model.device)


# --- tiles ----------------------------------------------------------------------------------------------------------


def tile_grid(
    shape: ZYX, pixel_size: PixelSize, *, plan: TilePlan | None = None, field_tiles: FieldTiles | None = None
) -> dict[str, object]:
    """Return a store's ``tile_grid`` (centres in um): one centred tile without ``plan``, else the field tiles'."""
    _, py, px = pixel_size
    if plan is None:
        _, y, x = shape
        centers = [[(y - 1) / 2 * py, (x - 1) / 2 * px]]
        return {
            "size_yx": None,
            "overlap_yx": [0, 0],
            "centers_yx": centers,
            "shape_zyx": list(shape),
            "blend": None,
            "units": "um",
        }
    centers = []
    for region in plan.regions:
        _, (y0, y1), (x0, x1) = region.core
        centers.append([(y0 + y1 - 1) / 2 * py, (x0 + x1 - 1) / 2 * px])
    return {
        "size_yx": None if field_tiles is None else list(field_tiles.size_yx),
        "overlap_yx": None if field_tiles is None else list(field_tiles.overlap_yx),
        "centers_yx": centers,
        "shape_zyx": list(shape),
        "blend": plan.blend,
        "units": "um",
    }


def grid_problems(store: TfStore, plan_grid: Mapping[str, object] | None) -> list[str]:
    """Return why the tiles of ``store`` do not fit ``plan_grid`` (``None`` for a shift-invariant model), or nothing."""
    stored = store.tile_grid
    if plan_grid is None:
        return [f"holds {store.tiles} field tiles, but the model is shift-invariant"] if store.tiles != 1 else []
    if stored.get("shape_zyx") is None:
        return [_shapeless(store)]
    differences = [
        f"{name} {_shown(stored.get(name))} (model: {_shown(plan_grid.get(name))})"
        for name in ("shape_zyx", "size_yx", "overlap_yx")
        if _listed(stored.get(name)) != _listed(plan_grid.get(name))
    ]
    if not _same_centres(stored.get("centers_yx"), plan_grid.get("centers_yx")):
        differences.append(
            f"tile centres {_shown(stored.get('centers_yx'))} (model: {_shown(plan_grid.get('centers_yx'))})"
        )
    return [f"was made on another tile grid: {'; '.join(differences)}"] if differences else []


def _shapeless(store: TfStore) -> str:
    writer = "#592, which" if store.origin == "#592" else "a writer that"
    return (
        f"was written by {writer} recorded no volume shape for its tile grid (tile_grid.shape_zyx), so its tiles "
        "cannot be placed on this data; recalibrate with calibrate_position (wo calibrate-tf)"
    )


def _listed(value: object) -> object:
    return list(value) if isinstance(value, list | tuple) else value


def _shown(value: object) -> str:
    return canonical_json(_listed(value))


def _same_centres(stored: object, planned: object) -> bool:
    try:
        a, b = (torch.tensor(centres, dtype=torch.float64) for centres in (stored, planned))
        return a.shape == b.shape and torch.allclose(a, b, rtol=1e-9, atol=1e-9)
    except (TypeError, ValueError, RuntimeError):
        return stored == planned


def store_problems(store: TfStore, model: ReconstructionModel, *, free: Sequence[str]) -> list[str]:
    """Return why ``store`` cannot serve ``model`` (whose free quantities are ``free``), or nothing.

    The imaging-system check is left out when the names do not cover the model or the tiles cannot be placed.
    Without a volume, field tiles are compared by size and overlap only.
    """
    problems = []
    missing = [name for name in free if name not in store.parameter_names]
    if missing:
        problems.append(f"has no value for the free quantities {', '.join(missing)}")
    fixed = [name for name in store.parameter_names if name not in free and name.startswith(SYSTEM_PREFIX)]
    if fixed:
        problems.append(
            f"holds {', '.join(fixed)}, which the model fixes in its imaging system; an imaging-system value the "
            "model fixes never comes from a store: make them free, or use a store made for this system"
        )
    field_tiles = model.field_tiles
    grid: list[str] = []
    if field_tiles is None:
        grid = grid_problems(store, None)
    elif store.tile_grid.get("shape_zyx") is None:
        return [*problems, _shapeless(store)]
    else:
        own = {"size_yx": list(field_tiles.size_yx), "overlap_yx": list(field_tiles.overlap_yx)}
        differences = [
            f"{name} {_shown(store.tile_grid.get(name))} (model: {_shown(value)})"
            for name, value in own.items()
            if _listed(store.tile_grid.get(name)) != value
        ]
        grid = [f"was made on another tile grid: {'; '.join(differences)}"] if differences else []
    return [*problems, *([] if problems else _system_problems(store, model)), *grid]


# --- the prior and the start a store gives a calibration ------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Start:
    """A search start: values by name, ``(1, N)`` (shared by every tile) or ``(T, N)`` float64."""

    names: tuple[str, ...]
    values: Tensor

    def row(self) -> dict[str, float]:
        """Return the first row's values by name."""
        return dict(zip(self.names, self.values[0].tolist(), strict=True))


@dataclass(frozen=True, kw_only=True)
class SearchInputs:
    """What the ``prior`` and ``start`` stores give a calibration: the prior store, the start and their paths."""

    prior: TfStore | None = None
    start: Start | None = None
    prior_path: str | None = None
    start_path: str | None = None


def _checked_store(
    problems: Problems,
    source: StoreSource | None,
    argument: str,
    *,
    specs: Mapping[str, Free],
    tiles: int | None,
    plan_grid: Mapping[str, object] | None,
) -> TfStore | None:
    """Read a ``prior``/``start`` store and check its tiles and names; ``None`` without one or when refused."""
    if source is None:
        return None
    with problems.collect():
        store = read_store(source, argument)
        location = where(argument, store.path)
        if tiles is not None and store.tiles not in (1, tiles):
            raise Rejected.one(
                f"{_shown_store(store)} holds {store.tiles} tiles, but the model calibrates {tiles}; give a store of "
                "one row or one per tile",
                location=location,
            )
        if tiles is not None and store.tiles > 1 and (refused := grid_problems(store, plan_grid)):
            raise Rejected.one(refused[0], location=location)
        if specs and not any(name in specs for name in store.parameter_names):
            raise Rejected.one(
                f"holds {', '.join(store.parameter_names)}, none of which is free in the model (free: "
                f"{', '.join(specs)})",
                location=location,
            )
        for name in store.parameter_names if specs else ():
            if name not in specs:
                warn(f"{name} is not free in the model; its stored value is ignored", location=location)
        return store
    return None


def _warn_prior(store: TfStore, specs: Mapping[str, Free]) -> None:
    """Warn of a prior mean on its quantity's bound, and of named quantities without a tolerance (a flat prior)."""
    location = where("prior", store.path)
    at_bound = store.provenance.get("at_bound")
    ended = set()
    if isinstance(at_bound, list):
        ended = {name for names in at_bound if isinstance(names, list) for name in names}
    for name in specs:
        if name not in store.parameter_names:
            continue
        values = store.parameters[:, store.parameter_names.index(name)].tolist()
        bounds = specs[name].bounds
        if name in ended or (bounds is not None and any(value <= bounds[0] or value >= bounds[1] for value in values)):
            warn(
                f"the prior of {name} is centred on {_shown_values(values)}, at a bound"
                f"{'' if bounds is None else f' of {_shown_bounds(bounds)}'}; the calibration that made it ended "
                "there, so the prior may hold the search on the bound",
                location=location,
            )
    flat = [name for name, spec in specs.items() if name in store.parameter_names and spec.tolerance is None]
    if flat:
        warn(
            f"sets no prior on {', '.join(flat)}: the model gives them no tolerance, so their prior stays flat and "
            "only the search starts at the stored values; give them a tolerance",
            location=location,
        )


def _shown_values(values: Sequence[float]) -> str:
    return canonical_json(values[0] if len(values) == 1 else list(values))


def _shown_bounds(bounds: Sequence[float]) -> str:
    return f"{bounds[0]:.6g} to {bounds[1]:.6g}"


def _start(specs: Mapping[str, Free], *, prior: TfStore | None, start: TfStore | None) -> Start | None:
    """Return where the search starts: the ``start`` values, else the ``prior`` means clamped to the bounds."""
    columns: dict[str, Tensor] = {}
    for store, argument in ((prior, "prior"), (start, "start")):
        if store is None:
            continue
        for name in specs:
            if name not in store.parameter_names:
                continue
            values = store.parameters[:, store.parameter_names.index(name)]
            bounds = specs[name].bounds
            if bounds is not None and argument == "start" and not ((values >= bounds[0]) & (values <= bounds[1])).all():
                raise Rejected.one(
                    f"{_shown_store(store)} starts {name} at {_shown_values(values.tolist())}, outside its bounds "
                    f"{_shown_bounds(bounds)}",
                    location=where(argument, store.path),
                )
            columns[name] = values.clamp(*bounds) if bounds is not None else values
    if not columns:
        return None
    rows = max(values.shape[0] for values in columns.values())
    return Start(names=tuple(columns), values=torch.stack([values.expand(rows) for values in columns.values()], dim=1))


def _path_text(source: object) -> str | None:
    """Return the absolute path of a store argument; ``None`` for a store in memory."""
    path = source.path if isinstance(source, TfStore) else source
    return str(Path(path).absolute()) if isinstance(path, str | Path) else None


def search_inputs(
    problems: Problems,
    model: ReconstructionModel,
    *,
    prior: StoreSource | None,
    start: StoreSource | None,
    tiles: int | None,
    plan_grid: Mapping[str, object] | None,
) -> SearchInputs:
    """Check the ``prior`` and ``start`` stores of a calibration and return what they give its search.

    ``tiles`` is how many tiles the model calibrates (``None`` when unknown leaves the tile checks out) and
    ``plan_grid`` the :func:`tile_grid` of its field tiles. Every problem goes into ``problems``.
    """
    specs = free_specs(model.imaging_model)
    prior_store = _checked_store(problems, prior, "prior", specs=specs, tiles=tiles, plan_grid=plan_grid)
    start_store = _checked_store(problems, start, "start", specs=specs, tiles=tiles, plan_grid=plan_grid)
    values = None
    if specs and prior_store is not None:
        with problems.collect(location=where("prior", prior_store.path)):
            prior_store.to_prior(specs)
            _warn_prior(prior_store, specs)
    if specs:
        with problems.collect():
            values = _start(specs, prior=prior_store, start=start_store)
    return SearchInputs(prior=prior_store, start=values, prior_path=_path_text(prior), start_path=_path_text(start))


# --- a model's current values as a store ---------------------------------------------------------------------------


def made_by(model: ReconstructionModel) -> dict[str, object]:
    """Return the provenance every store carries: the package versions, the host, the time and the device."""
    from waveorder.io.plate import software_versions

    versions = software_versions()
    return {
        "waveorder_version": versions["waveorder"],
        "torch_version": torch.__version__,
        "iohub_version": versions["iohub"],
        "host": {"host": socket.gethostname(), "pid": os.getpid()},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(model.device),
    }


def _transfer_at(model: ReconstructionModel, shape: ZYX, pixel_size: PixelSize) -> TransferArray | None:
    """Return the model's transfer on ``shape`` after z padding as reconstructions compute it; ``None`` for no such."""
    system = model.system
    if not isinstance(system, ShiftInvariantSystem):
        log.info("%s has no single transfer function; none is stored", type(system).__name__)
        return None
    slices = model.z_padding_for(shape, pixel_size=pixel_size)
    sampling = ZPadding(slices).sampling(Sampling(shape=shape, pixel_size=pixel_size))
    with torch.no_grad():
        values = system.transfer(sampling).detach().cpu()
    attrs: dict[str, object] = {
        "layout": ["channel", "field", "z", "y", "x_half"],
        "fft": "data = irfftn(sum over fields of H[c, f] * rfftn(object[f])) over (z, y, x); x keeps X // 2 + 1",
        "real": system.real_transfer,
        **transfer_record(model, sampling),
        "z_padding": slices,
    }
    transfer = values if system.real_transfer else torch.view_as_complex(values.contiguous())
    return TransferArray(shape=sampling.shape, attrs=attrs, transfer=transfer)


def _listed_shapes(shapes: object) -> tuple[ZYX, ...]:
    """Return the ``shapes`` of ``to_tf_store`` once each; raise ParameterError at ``shapes`` for anything else."""
    if isinstance(shapes, Sequence) and not isinstance(shapes, str):
        try:
            return tuple(dict.fromkeys(zyx(entry, where="to_tf_store", name="shapes", minimum=1) for entry in shapes))
        except ParameterError:
            pass
    raise ParameterError(
        f"{named('shapes')} must be a sequence of (z, y, x) volume shapes of positive ints, got {shapes!r}; for one "
        "shape pass [(z, y, x)]",
        path="shapes",
    )


def current_store(model: ReconstructionModel, *, shapes: object, pixel_size: PixelSize | None) -> TfStore:
    """Return ``model.to_tf_store``: the free quantities' current values, with the transfer on each shape."""
    listed = _listed_shapes(shapes)
    tiled = model.field_tiles is not None
    if tiled and len(listed) != 1:
        raise Rejected.one(
            f"a model with field tiles takes one volume shape, which places its tiles, with {named('pixel_size')}, "
            f"but {named('shapes')} has {len(listed)} (ZxYxX): {', '.join(map(shown_shape, listed)) or 'none'}",
            location=named("shapes"),
        )
    plan = None
    try:
        for shape in listed:
            model.check(shape, pixel_size=pixel_size)
        if tiled:
            plan = model.field_tile_plan(listed[0], pixel_size=pixel_size)
    except WaveorderError as error:
        raise Rejected.one(str(error), location=named("shapes")) from error
    names, parameters, nominal = current_values(model)
    tiles = 1 if plan is None else len(plan.regions)
    if parameters.shape[0] == 1:
        parameters = parameters.expand(tiles, len(names)).clone()
    elif parameters.shape[0] != tiles:
        raise Rejected.one(
            f"the model holds values for {parameters.shape[0]} field tiles, but {shown_shape(listed[0])} (ZxYxX) "
            f"has {tiles}",
            location=named("shapes"),
        )
    grid: dict[str, object] = {}
    transfers: list[TransferArray] = []
    if listed and pixel_size is not None:
        grid = tile_grid(listed[0], pixel_size, plan=plan, field_tiles=model.field_tiles)
        if plan is None:
            transfers = [item for shape in listed if (item := _transfer_at(model, shape, pixel_size)) is not None]
    return TfStore(
        parameter_names=names,
        parameters=parameters,
        tile_grid=grid,
        provenance={
            "kind": "nominal" if nominal else "parameters",
            "shapes_zyx": [list(shape) for shape in listed],
            **made_by(model),
            **system_record(model),
        },
        transfers=tuple(transfers),
    )
