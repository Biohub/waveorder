"""The pieces of a model's ``calibrate_position``: its checks, its calibration, and the store of what it found.

The volume is calibrated whole, or each field tile on its own; the ``tf.zarr`` store carries the values found.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from iohub.ngff import Position
from torch import Tensor

from waveorder.config import V4Config, dump
from waveorder.config.schema import LegacyAutoWeight, WeightAuto
from waveorder.errors import ConfigurationError, Problems, Rejected, WaveorderError, named
from waveorder.io.positions import channel_indices, read_position_meta, store_path
from waveorder.io.sweep_report import save_sweep_montage, save_sweep_report
from waveorder.io.tf_store import SCORES, TfStore
from waveorder.models._arguments import shown_shape
from waveorder.models._base import ReconstructionModel
from waveorder.models._checks import ZYX, GivenPixelSize, PixelSize, position_pixel_size
from waveorder.models._stores import (
    SearchInputs,
    StoreSource,
    float64,
    made_by,
    search_inputs,
    system_record,
    tile_grid,
    warn,
)
from waveorder.nn import Sampling, free_specs, load_parameters
from waveorder.nn.parameters import placement, shortest_decimal
from waveorder.nn.priors import Beads
from waveorder.nn.search import BeadCalibration, SweepResult
from waveorder.tiling import TilePlan

log = logging.getLogger(__name__)

_SIGMA = "noise.sigma"
"""The Gaussian noise level, which a bead fit estimates beside the system's quantities."""


@dataclass(frozen=True, kw_only=True)
class Plan:
    """What the checks decided: the volume, its free quantities, field tiles and search inputs."""

    path: Path
    name: str
    shape: ZYX
    channel_names: tuple[str, ...]
    channels: tuple[int, ...]
    pixel_size: PixelSize
    time: int
    names: tuple[str, ...]
    tile_plan: TilePlan | None
    search: SearchInputs


def _fits(problems: Problems, model: ReconstructionModel, plan: dict) -> tuple[TilePlan | None, bool]:
    """Check that the model can calibrate on the volume's grid; return its field-tile plan and whether it can."""
    shape, pixel_size, location = plan["shape"], plan["pixel_size"], str(plan["path"])
    measured = (len(plan["channels"]), *shape)
    try:
        expected = tuple(model.system.data_shape(Sampling(shape=shape, pixel_size=pixel_size)))
        if expected != measured:
            problems.add(
                f"{location} holds {shown_shape(measured)} (CxZxYxX) for {named('channels')}, but the system "
                f"measures {shown_shape(expected)} on that grid",
                location,
            )
            return None, False
        model.check(shape, pixel_size=pixel_size, calibration=True)
        if model.field_tiles is None:
            return None, True
        if model.z_padding:
            problems.add(
                f"z_padding is {model.z_padding}, but field tiles are calibrated unpadded (their halos treat the "
                "boundaries); set it to 0"
            )
            return None, False
        return model.field_tile_plan(shape, pixel_size=pixel_size), True
    except Rejected as error:
        problems.take(error, location=location)
    except WaveorderError as error:
        problems.add(str(error), location)
    return None, False


def checked_plan(  # noqa: PLR0913 (the arguments of the method that the checks read)
    model: ReconstructionModel,
    position: Position | None,
    *,
    problems: Problems,
    channels: Sequence[str],
    pixel_size: GivenPixelSize,
    time: int | None,
    prior: StoreSource | None,
    start: StoreSource | None,
) -> Plan:
    """Run every check of ``calibrate_position`` on the opened position (``None``: refused in ``problems``)."""
    names = tuple(free_specs(model.imaging_model))
    if not names:
        problems.add(
            "nothing to calibrate: calibration estimates the quantities the model frees, and it frees none; give "
            "one as waveorder.nn.Free(nominal, tolerance=t, bounds=(lo, hi)) instead of a number, for example "
            "Illumination(numerical_aperture=Free(0.5, tolerance=0.05, bounds=(0.3, 0.7))); in a configuration "
            "file, write it as {nominal: x, tolerance: t, bounds: [lo, hi]}"
        )
    if isinstance(model.object_prior, Beads) and (prior is not None or start is not None):
        problems.add(f"{named('prior')} and {named('start')} stores with a Beads object prior are not supported yet")
    path = None if position is None else store_path(position)
    found: dict = {"path": path, "channel_names": (channels,) if isinstance(channels, str) else tuple(channels)}
    meta = None
    with problems.collect():
        meta = None if position is None else read_position_meta(position)
    if meta is not None:
        found |= {"name": meta.name, "shape": meta.zyx, "time": 0 if time is None else time}
        with problems.collect():
            found["channels"] = channel_indices(meta, found["channel_names"])
        with problems.collect():
            found["pixel_size"] = position_pixel_size(model, meta, pixel_size)
        index = found["time"]
        if not (isinstance(index, int) and not isinstance(index, bool) and 0 <= index < meta.times):
            problems.add(
                f"{named('time')} is {time!r}, but {path} has {meta.times} time points (0 to {meta.times - 1})",
                str(path),
            )
            meta = None
    complete = meta is not None and {"channels", "pixel_size"} <= set(found)
    tile_plan, tiles, grid = None, None, None
    if complete and names:
        tile_plan, fits = _fits(problems, model, found)
        if fits:
            tiles = 1 if tile_plan is None else len(tile_plan.regions)
        if tile_plan is not None:
            grid = tile_grid(found["shape"], found["pixel_size"], plan=tile_plan, field_tiles=model.field_tiles)
    search = search_inputs(problems, model, prior=prior, start=start, tiles=tiles, plan_grid=grid)
    problems.raise_if_any()
    return Plan(**found, names=names, tile_plan=tile_plan, search=search)


# --- calibrating ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class _Estimates:
    """What a calibration found, in the store's form: the quantities' names, and one row per tile."""

    names: tuple[str, ...]
    parameters: Tensor
    at_bound: list[list[str]]
    objective: list[float]
    scores: dict[str, Tensor]
    grid_columns: list[str] | None = None


def _calibrate_whole(model: ReconstructionModel, volume: Tensor, plan: Plan) -> _Estimates:
    """Calibrate on the whole volume; the model takes the estimates, and a bead fit's noise level when it is free."""
    search = plan.search
    if search.start is not None:
        load_parameters(model.imaging_model, search.start.row())
    config = model.config
    wanted = [name for name in SCORES if config is not None and (config.outputs == "all" or name in config.outputs)]
    weight = None if config is None else config.object.weight
    auto = weight.auto if isinstance(weight, WeightAuto) else None
    written = auto is not None and (auto.report_path is not None or auto.plot_path is not None)
    result = model.calibrate(volume, pixel_size=plan.pixel_size, prior=search.prior, history=bool(wanted) or written)
    found = dict(result.parameters)
    scores: dict[str, Tensor] = {}
    columns = None
    if isinstance(result, BeadCalibration):  # an objective trace, and no bounds but the noise level's
        at_bound, objective = [], result.trace[-1]
        if "score_trace" in wanted:
            scores["score_trace"] = result.trace.detach().cpu().double()
        sigma = float(result.noise_sigma.detach())
        log.info("the bead fit's noise level is %s", shortest_decimal(sigma, torch.float32))
        spec = free_specs(model.imaging_model).get(_SIGMA)
        if spec is not None:
            low, high = spec.bounds or (sigma, sigma)
            held = min(max(sigma, low), high)
            if held != sigma:
                at_bound.append(_SIGMA)
            load_parameters(model.imaging_model, {_SIGMA: held})
            found[_SIGMA] = torch.tensor(held, dtype=torch.float64)
    else:
        at_bound, objective = list(result.at_bound), result.objective
        if "score_trace" in wanted and result.steps:
            scores["score_trace"] = torch.tensor(result.steps, dtype=torch.float64)
        if "score_grid" in wanted and result.grid is not None:
            scores["score_grid"], columns = result.grid.values, [*result.grid.names, "objective"]
        if auto is not None and result.sweep is not None:
            _write_sweep(result.sweep, auto, title=f"{plan.name}   t={plan.time}")
    names = tuple(name for name in plan.names if name in found)
    parameters = float64(torch.stack([torch.as_tensor(found[name]).reshape(()) for name in names]))
    return _Estimates(
        names=names,
        parameters=parameters[None],
        at_bound=[at_bound],
        objective=[shortest_decimal(float(objective.detach()), torch.float32)],
        scores=scores,
        grid_columns=columns,
    )


def _write_sweep(result: SweepResult, auto: LegacyAutoWeight, *, title: str) -> None:
    """Write the sweep's JSON report and PNG montage where ``object.weight.auto`` says (each when set)."""
    if auto.report_path is not None:
        log.info("wrote the weight sweep's report to %s", save_sweep_report(result, auto.report_path))
    if auto.plot_path is not None:
        log.info("wrote the weight sweep's montage to %s", save_sweep_montage(result, auto.plot_path, title=title))


def _calibrate_field_tiles(model: ReconstructionModel, volume: Tensor, plan: Plan) -> _Estimates:
    """Calibrate every field tile on the volume prepared whole; the model takes the table as ``tile_parameters``."""
    assert plan.tile_plan is not None
    search = plan.search
    dtype, _ = placement(model.system)
    data = model.prepare(volume).to(dtype=dtype, device=model.device)
    tiles = len(plan.tile_plan.regions)
    start = None
    if search.start is not None:
        start = {
            name: search.start.values[:, index].expand(tiles).to(dtype=dtype, device=model.device)
            for index, name in enumerate(search.start.names)
        }
    tiled = model.calibrate_tiles(
        data, pixel_size=plan.pixel_size, plan=plan.tile_plan, start=start, prior=search.prior, prepared=True
    )
    model.tile_parameters = dict(tiled.parameters)
    return _Estimates(
        names=plan.names,
        parameters=torch.stack([float64(tiled.parameters[name]) for name in plan.names], dim=1),
        at_bound=[list(tile.at_bound) for tile in tiled.tiles],
        objective=float64(tiled.objective).tolist(),
        scores={},
    )


def _provenance(model: ReconstructionModel, plan: Plan, found: _Estimates) -> dict[str, object]:
    record = system_record(model)
    try:
        config = model.to_config(channels=plan.channel_names, pixel_size=plan.pixel_size, times=[plan.time])
        record = {"config": dump(config) if isinstance(config, V4Config) else config, **record}
    except ConfigurationError:  # a part with no plain-data form: the imaging-system record alone
        pass
    return {
        "kind": "calibrated",
        "sources": [str(plan.path)],
        "position": str(plan.path),
        "position_name": plan.name,
        "time_index": plan.time,
        **made_by(model),
        **record,
        "score": "Evidence" if model.score is None else type(model.score).__name__,
        "objective": found.objective,
        "at_bound": found.at_bound,
        **({} if found.grid_columns is None else {"score_grid_columns": found.grid_columns}),
        "prior": plan.search.prior_path,
        "start": plan.search.start_path,
    }


def calibrated_store(model: ReconstructionModel, volume: Tensor, plan: Plan) -> TfStore:
    """Calibrate ``model`` on the read ``volume`` as ``plan`` says; return the store of what it found, unsaved.

    Warns (:class:`~waveorder.errors.StoreWarning`) of each quantity that ended on a bound.
    """
    whole = plan.tile_plan is None
    found = _calibrate_whole(model, volume, plan) if whole else _calibrate_field_tiles(model, volume, plan)
    store = TfStore(
        parameter_names=found.names,
        parameters=found.parameters,
        tile_grid=tile_grid(plan.shape, plan.pixel_size, plan=plan.tile_plan, field_tiles=model.field_tiles),
        provenance=_provenance(model, plan, found),
        scores=found.scores,
    )
    for tile, names in enumerate(found.at_bound):
        place = f" in tile {tile}" if store.tiles > 1 else ""
        for name in names:
            value = shortest_decimal(float(store.parameters[tile, found.names.index(name)]), torch.float32)
            warn(f"{name} ended on a bound{place} ({value}); it is the bound, not an estimate", location=str(plan.path))
    return store
