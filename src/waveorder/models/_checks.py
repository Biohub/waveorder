"""The checks a position method runs before it computes, which raise all problems together as one ``Rejected``."""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from waveorder.errors import ParameterError, PositionWarning, Problems, Rejected, WaveorderError, named
from waveorder.io.plate import DEFAULT_VERSION, OutputPosition, choose_geometry, is_single, name_problems, read_plate
from waveorder.io.positions import (
    SINGLE_POSITION,
    PositionMeta,
    channel_indices,
    listed_inputs,
    open_positions,
    read_position_meta,
)
from waveorder.models._arguments import PixelSize as AnyPixelSize
from waveorder.models._arguments import pixel_size as voxel_size
from waveorder.models._arguments import shown_shape, shown_sizes
from waveorder.models._base import ReconstructionModel
from waveorder.nn import Sampling
from waveorder.tiling import TilePlan

type ZYX = tuple[int, int, int]
type PixelSize = tuple[float, float, float]
type GivenPixelSize = AnyPixelSize | Literal["auto"] | None
type Times = Sequence[int] | int | None

MISMATCH_TOLERANCE = 0.05
"""Relative difference between a stated and a recorded voxel size above which the stated one is warned about."""


def stated(model: ReconstructionModel, pixel_size: GivenPixelSize) -> PixelSize | None:
    """Return the voxel size the argument states, else the model's own; ``None`` for each position's recorded one.

    Raises Rejected at ``pixel_size`` for any other form.
    """
    if pixel_size is None:
        return model.pixel_size
    if isinstance(pixel_size, str) and pixel_size == "auto":
        return None
    try:
        return voxel_size(pixel_size, where=type(model).__name__)
    except ParameterError as error:
        raise Rejected.one(str(error), location=named("pixel_size")) from None


def _differs(given: Sequence[float], recorded: Sequence[float]) -> bool:
    return not all(math.isclose(a, b, rel_tol=MISMATCH_TOLERANCE) for a, b in zip(given, recorded, strict=True))


def _warn_mismatch(given: PixelSize, recorded: Sequence[float], *, where: str) -> None:
    argument = named("pixel_size")
    warnings.warn(
        PositionWarning(
            f"{argument} is (z, y, x) = ({shown_sizes(given)}) um, but {where} ({shown_sizes(recorded)}) um; "
            f"reconstructing at {argument}. Set it to the data's voxel size, or to auto to use the recorded one",
        ),
        stacklevel=4,
    )


def _unrecorded(meta: PositionMeta) -> str | None:
    """Return why a position's own voxel size cannot be used (a scale of ones), else ``None``."""
    if any(value != 1.0 for value in meta.pixel_size):
        return None
    return (
        f"{meta.path} records no voxel size (its scale is 1, 1, 1); set {named('pixel_size')} to the voxel size "
        "(z, y, x) in um"
    )


def _shown(meta: PositionMeta) -> str:
    """Return how a message names a position: its plate name, or its store's path for a single-position store."""
    return str(meta.path) if meta.name == SINGLE_POSITION else meta.name


def position_pixel_size(model: ReconstructionModel, meta: PositionMeta, pixel_size: GivenPixelSize) -> PixelSize:
    """Return the voxel size of a position: stated (warned of where it differs from the recorded one), else recorded.

    Raises Rejected as :func:`stated`, and at the position if no size is stated and it records none.
    """
    given = stated(model, pixel_size)
    if given is None:
        problem = _unrecorded(meta)
        if problem is not None:
            raise Rejected.one(problem, location=str(meta.path))
        return meta.pixel_size
    if _differs(given, meta.pixel_size):
        _warn_mismatch(given, meta.pixel_size, where=f"{meta.path} records")
    return given


def selected_times(times: Times, count: int) -> list[int]:
    """Return the absolute time indices ``times`` selects among ``count``, in the order given, each once.

    Raises Rejected for an index that is no integer or outside ``range(count)``.
    """
    if times is None:
        return list(range(count))
    argument = named("times")
    chosen = [times] if isinstance(times, int) else list(times)
    if any(isinstance(t, bool) or not isinstance(t, int) for t in chosen):
        raise Rejected.one(f"{argument} must be time indices (0, 1, ...), got {times!r}", location=argument)
    outside = [t for t in chosen if not 0 <= t < count]
    if outside:
        raise Rejected.one(
            f"{argument} selects {', '.join(map(str, outside))}, outside the {count} time points of the data"
        )
    return list(dict.fromkeys(chosen))


# --- the checks of reconstruct_positions ------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Arguments:
    """What ``reconstruct_positions`` was asked for, as far as the checks read it."""

    positions: object
    output: str | Path
    channels: Sequence[str]
    pixel_size: GivenPixelSize = None
    times: Times = None
    overwrite: bool = False
    resume: bool = False
    write_pixel_size: bool = False
    tile: Sequence[int] | None = None
    tile_overlap: Sequence[int] | None = None


@dataclass(frozen=True, kw_only=True)
class Grid:
    """How one input grid reconstructs: path ``"whole"`` (iohub), ``"tiles"`` (of ``tile``) or ``"field_tiles"``."""

    output_z: int
    path: str
    tiles: TilePlan | None = None


@dataclass(kw_only=True)
class Plan:
    """What :func:`check_positions` decided: what is read, how it reconstructs and what is written.

    ``pixel_sizes``, ``times`` and ``grids`` are by position name; ``version`` is an existing plate's OME-Zarr
    version, else iohub's ``version`` of the first input; ``replaces`` deletes an existing store first.
    """

    output: Path
    version: str
    creates: bool
    replaces: bool
    positions: tuple[PositionMeta, ...]
    channels: tuple[str, ...]
    names: tuple[str, ...]
    pixel_sizes: dict[str, PixelSize]
    times: dict[str, tuple[int, ...]]
    grids: dict[str, Grid] = field(default_factory=dict)
    outputs: tuple[OutputPosition, ...] = ()
    tile: ZYX | None = None
    tile_overlap: ZYX | None = None


def check_positions(model: ReconstructionModel, arguments: Arguments, *, problems: Problems) -> Plan:
    """Check a reconstruction of positions against the data's metadata, the model and the output; write nothing.

    Every problem goes into ``problems``; the plan is complete only when there is none. The calibration score is
    not checked: nothing calibrates here.
    """
    metas = _inputs(arguments.positions, problems=problems)
    channels = _channel_names(arguments.channels, problems=problems)
    names = _output_names(model, channels, problems=problems)
    output = Path(arguments.output).absolute()
    plan = Plan(
        output=output,
        version=metas[0].version if metas else DEFAULT_VERSION,
        creates=not output.exists(),
        replaces=arguments.overwrite and output.exists(),
        positions=tuple(metas),
        channels=channels,
        names=names,
        pixel_sizes=_pixel_sizes(model, metas, arguments.pixel_size, problems=problems),
        times=_times(metas, channels, arguments.times, problems=problems),
        tile=_triple(arguments.tile, argument="tile", minimum=1, problems=problems),
        tile_overlap=_triple(arguments.tile_overlap, argument="tile_overlap", minimum=0, problems=problems),
    )
    if arguments.resume and arguments.overwrite:
        problems.add(f"{named('resume')} continues the output that {named('overwrite')} deletes; give one of them")
    _names_and_places(plan, arguments, problems=problems)
    _model_checks(model, plan, problems=problems)
    _tile_rules(model, plan, arguments, problems=problems)
    _output(plan, arguments, problems=problems)
    return plan


def _inputs(positions: object, *, problems: Problems) -> list[PositionMeta]:
    """Open the inputs as iohub positions (each pattern matching nothing is reported) and read every one's metadata."""
    given = listed_inputs(positions)
    if not given:
        problems.add("names no position", named("positions"))
    metas: list[PositionMeta] = []
    with open_positions(given, problems=problems) as opened:
        for position in opened:
            with problems.collect():
                metas.append(read_position_meta(position))
    return metas


def _channel_names(channels: object, *, problems: Problems) -> tuple[str, ...]:
    argument = named("channels")
    listed = [channels] if isinstance(channels, str) else list(channels)  # ty: ignore[invalid-argument-type]
    if not listed or not all(isinstance(name, str) for name in listed):
        problems.add(f"{argument} must list the input channel names, got {channels!r}", argument)
        return ()
    if len(set(listed)) != len(listed):
        problems.add(f"{argument} names a channel twice: {', '.join(listed)}", argument)
        return ()
    return tuple(listed)


def _output_names(model: ReconstructionModel, channels: Sequence[str], *, problems: Problems) -> tuple[str, ...]:
    if not channels:
        return ()
    names = tuple(model.output_channel_names(channels))
    if not names or len(set(names)) != len(names):
        problems.add(f"the model's output_channel_names are {list(names)}; they must be distinct and at least one")
        return ()
    return names


def _pixel_sizes(
    model: ReconstructionModel, metas: Sequence[PositionMeta], pixel_size: GivenPixelSize, *, problems: Problems
) -> dict[str, PixelSize]:
    """Return each position's voxel size; warn of recorded sizes the stated one differs from, once per size."""
    given = None
    with problems.collect():
        given = stated(model, pixel_size)
    sizes: dict[str, PixelSize] = {}
    differing: dict[PixelSize, list[str]] = {}
    for meta in metas:
        if given is not None:
            sizes[meta.name] = given
            if _differs(given, meta.pixel_size):
                differing.setdefault(meta.pixel_size, []).append(_shown(meta))
        elif (problem := _unrecorded(meta)) is not None:
            problems.add(problem, str(meta.path))
        else:
            sizes[meta.name] = meta.pixel_size
    for recorded, where in differing.items():
        assert given is not None
        shown = ", ".join(where[:3]) + (f" and {len(where) - 3} more" if len(where) > 3 else "")
        _warn_mismatch(given, recorded, where=f"{shown} {'records' if len(where) == 1 else 'record'}")
    return sizes


def _times(
    metas: Sequence[PositionMeta], channels: Sequence[str], times: Times, *, problems: Problems
) -> dict[str, tuple[int, ...]]:
    """Return the selected times of each position; report the channels and the time indices a position lacks."""
    selected: dict[str, tuple[int, ...]] = {}
    for meta in metas:
        with problems.collect():
            channel_indices(meta, channels)
        with problems.collect(location=str(meta.path)):
            selected[meta.name] = tuple(selected_times(times, meta.times))
    return selected


def _triple(value: Sequence[int] | None, *, argument: str, minimum: int, problems: Problems) -> ZYX | None:
    if value is None:
        return None
    entries = list(value) if isinstance(value, Sequence) and not isinstance(value, str) else []
    if len(entries) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < minimum for n in entries):
        shown = named(argument)
        problems.add(f"{shown} must be three integers >= {minimum} (z, y, x), got {value!r}", shown)
        return None
    z, y, x = entries
    return (z, y, x)


def _names_and_places(plan: Plan, arguments: Arguments, *, problems: Problems) -> None:
    """Refuse positions of several plates that share a name, and an output that is, lies inside or holds an input."""
    metas = plan.positions
    duplicated = sorted(
        {
            meta.name
            for meta in metas
            if meta.name != SINGLE_POSITION and sum(other.name == meta.name for other in metas) > 1
        }
    )
    if duplicated:
        problems.add(
            f"positions of other plates share the names {duplicated}; one output plate keeps one each",
            named("positions"),
        )
    target = Path(arguments.output).resolve()
    location = f"{named('output')} {arguments.output}"
    inside = [_shown(meta) for meta in metas if target == meta.path or meta.path in target.parents]
    if inside:
        problems.add(f"is the input position {inside[0]} or lies inside it; write to another path", location)
    held = [_shown(meta) for meta in metas if target in meta.path.parents]
    if held and arguments.overwrite:
        problems.add(
            f"holds the input position {held[0]}, which {named('overwrite')} would delete; write to another "
            f"{named('output')}",
            location,
        )
    elif held:
        problems.add(
            f"holds the input position {held[0]}: a reconstruction adds its channels to the positions of its output "
            f"plate, which would change the recorded data; write to another {named('output')}",
            location,
        )


# --- the model against each grid -------------------------------------------------------------------------------------


def _model_checks(model: ReconstructionModel, plan: Plan, *, problems: Problems) -> None:
    """Check the model at each distinct input grid and pick how it reconstructs; fill ``plan.grids``."""
    checked: dict[tuple[ZYX, PixelSize], Grid | None] = {}
    for meta in plan.positions:
        if meta.name not in plan.pixel_sizes:
            continue
        key = (meta.zyx, plan.pixel_sizes[meta.name])
        if key not in checked:
            checked[key] = _check_grid(model, meta, plan, problems=problems)
        grid = checked[key]
        if grid is not None:
            plan.grids[meta.name] = grid


def _check_grid(model: ReconstructionModel, meta: PositionMeta, plan: Plan, *, problems: Problems) -> Grid | None:
    """Check one input grid; ``None`` when the system refuses it."""
    system = model.system
    zyx = meta.zyx
    pixel_size = plan.pixel_sizes[meta.name]
    where = f"{_shown(meta)} {shown_shape(zyx)}"
    try:
        sampling = Sampling(shape=zyx, pixel_size=pixel_size)
        measured = system.data_shape(sampling)
        fields, z, *_ = system.object_shape(sampling)
    except WaveorderError as error:
        problems.add(f"{where}: {error}")
        return None
    channels = named("channels")
    if plan.channels and measured[0] != len(plan.channels):
        problems.add(
            f"{channels} lists {len(plan.channels)} ({', '.join(plan.channels)}), but the model's "
            f"{type(system).__name__} measures {measured[0]}",
            channels,
        )
    if tuple(measured[1:]) != zyx:
        problems.add(f"{_shown(meta)}: the system measures {shown_shape(measured)} on {shown_shape(zyx)}")
    if plan.names and fields != len(plan.names):
        problems.add(
            f"the model reconstructs {fields} fields ({', '.join(system.fields)}), but the output channels are "
            f"{', '.join(plan.names)}"
        )
    path = "whole"
    if model.field_tiles is not None:
        path = "field_tiles"
    elif plan.tile is not None and z == zyx[0]:
        path = "tiles"
    tiles = _tiling(model, meta, plan, path=path, problems=problems)
    try:
        model.check(zyx if tiles is None else tiles.read_shape, pixel_size=pixel_size)
    except WaveorderError as error:
        problems.add(f"{where}: {error}")
    return Grid(output_z=z, path=path, tiles=tiles)


def _tiling(
    model: ReconstructionModel, meta: PositionMeta, plan: Plan, *, path: str, problems: Problems
) -> TilePlan | None:
    """Plan the tiles of a grid on a path that splits the volume; report a tiling the model refuses."""
    pixel_size = plan.pixel_sizes[meta.name]
    where = f"{_shown(meta)} {shown_shape(meta.zyx)}"
    try:
        if path == "field_tiles":
            tiles = model.field_tile_plan(meta.zyx, pixel_size=pixel_size)
            count = len(tiles.regions)
            for name, values in (model.tile_parameters or {}).items():
                if values.shape[0] != count:
                    problems.add(
                        f"{where}: the model's tile_parameters hold {values.shape[0]} tiles of {name}, but its "
                        f"field tiles split this volume into {count}; load a calibration of a volume of this shape"
                    )
                    break
            return tiles
        if path == "tiles" and plan.tile is not None:
            return model.tile_plan(
                meta.zyx, pixel_size=pixel_size, tile=plan.tile, overlap=plan.tile_overlap or (0, 0, 0)
            )
    except (WaveorderError, ValueError) as error:
        problems.add(f"{where}: {error}", named("tile") if path == "tiles" else None)
    return None


def _tile_rules(model: ReconstructionModel, plan: Plan, arguments: Arguments, *, problems: Problems) -> None:
    """Refuse tile arguments where tiling cannot apply, and a tiling the model's ``prepare`` cannot share."""
    tile, overlap = named("tile"), named("tile_overlap")
    tiled = arguments.tile is not None
    paths = {grid.path for grid in plan.grids.values()}
    thin = any(plan.grids[meta.name].output_z != meta.zyx[0] for meta in plan.positions if meta.name in plan.grids)
    if "field_tiles" in paths:
        if thin:
            problems.add("thin systems with field-tiled dependence are not in waveorder 4.0")
        if tiled:
            problems.add(f"field tiles come from the model's {named('field_tiles')}; drop {tile}", tile)
    elif tiled and thin:
        problems.add(f"{tile} cannot split these volumes: a thin object reconstructs one plane", tile)
    elif tiled and model.z_padding and plan.tile is not None:
        split = [meta.zyx[0] for meta in plan.positions if meta.name in plan.grids and plan.tile[0] < meta.zyx[0]]
        if split:
            problems.add(
                f"{tile} splits z ({plan.tile[0]} of {split[0]} slices), but {named('z_padding')} pads whole stacks "
                f"along z: give {tile} a z of at least the stack's",
                tile,
            )
    if arguments.tile_overlap is not None and not tiled:
        problems.add(f"{overlap} widens {tile} cores; give {tile} too", overlap)
    splits = any(grid.tiles is not None for grid in plan.grids.values())
    if splits and type(model).prepare_scope == "block":
        problems.add(
            f"{type(model).__name__}.prepare is given one block at a time, and its prepare_scope is 'block' (the "
            f"result depends on the block), so tiles would not share one normalization: drop {tile} and field "
            "tiles, or set the class attribute prepare_scope to 'voxel' when prepare acts voxel by voxel, or to "
            "'stack_mean' when it normalizes by the mean it is given"
        )


# --- the output plate ------------------------------------------------------------------------------------------------


def _output(plan: Plan, arguments: Arguments, *, problems: Problems) -> None:
    """Check the output plate's names and its existing positions' shapes; fill ``plan.outputs``."""
    outputs = []
    for meta in plan.positions:
        grid = plan.grids.get(meta.name)
        if grid is None or not plan.names:
            continue
        t, _, z, y, x = meta.shape
        z_size, y_size, x_size = plan.pixel_sizes[meta.name] if arguments.write_pixel_size else meta.scale[2:]
        core = None if grid.tiles is None else (min(grid.tiles.tile[0], z), min(grid.tiles.tile[1], y), x)
        outputs.append(
            OutputPosition(
                name=meta.name,
                shape=(t, len(plan.names), grid.output_z, y, x),
                scale=(meta.scale[0], 1.0, z_size, y_size, x_size),
                geometry=choose_geometry(core, x=x),
            )
        )
    plan.outputs = tuple(outputs)
    problems.extend(name_problems([position.name for position in outputs]))
    existing = None
    with problems.collect():
        existing = read_plate(plan.output, single=is_single([position.name for position in outputs]))
    if existing is not None and not plan.replaces:
        plan.version = existing.version
        for position in outputs:
            held = existing.shapes.get(position.name)
            t, _, z, y, x = position.shape
            if held is not None and (held[0], *held[2:]) != (t, z, y, x):
                problems.add(
                    f"{plan.output / position.name} holds (T, Z, Y, X) = {shown_shape((held[0], *held[2:]))}, but "
                    f"this reconstruction writes {shown_shape((t, z, y, x))}; use another {named('output')}",
                    str(plan.output),
                )
    if plan.version == "0.4" and any(grid.tiles is not None for grid in plan.grids.values()):
        problems.add(
            f"v0.4 outputs have no shards, so the tile merger cannot give each tile row blocks of its own: drop "
            f"{named('tile')} and field tiles, or write to a new output from v0.5 inputs",
            str(plan.output),
        )
