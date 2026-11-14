"""The runs of ``wo rec``, ``wo apply-inv-tf`` and ``wo compute-tf``, which the commands import when they run.

This module imports the models and torch; :mod:`waveorder.cli.reconstruct` holds the commands and their flags.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import typer

from waveorder.cli._configured import ResolvedConfig, build_model, collecting, located, resolve_config
from waveorder.cli._observation import Observation, check_wandb, observing, parameter_table
from waveorder.cli.options import devices_argument, expect_store_output
from waveorder.cli.render import status
from waveorder.errors import Problems, ReconstructionFailed, ReconstructionReport, named
from waveorder.io.positions import SINGLE_POSITION, expand_positions, plate_name, read_position_meta
from waveorder.models import ReconstructionModel
from waveorder.models._checks import position_pixel_size
from waveorder.nn import free_specs

FAILED_UNITS = 3
"""The exit code of a reconstruction that ``--keep-going`` took past failed units."""


@dataclass(frozen=True, kw_only=True)
class Flags:
    """The flags of a run command, parsed: one field per argument of the model's methods, or per flag behind one."""

    command: str
    inputs: tuple[str, ...]
    config: Path
    output: Path
    store: Path | None
    device: str | None
    tile: tuple[int, int, int] | None
    tile_overlap: tuple[int, int, int] | None
    num_processes: int | None
    overwrite: bool
    resume: bool
    keep_going: bool
    write_config_scale_to_output: bool
    wandb: str | None


def _positions(
    model: ReconstructionModel, flags: Flags, resolved: ResolvedConfig, *, dry_run: bool
) -> ReconstructionReport:
    """Call ``model.reconstruct_positions`` with one argument per flag; the ``data`` block gives the rest."""
    return model.reconstruct_positions(
        list(flags.inputs),
        output=flags.output,
        **resolved.data,
        devices=devices_argument(flags.device, configured=resolved.config.solver.device),
        workers="auto" if flags.num_processes is None else flags.num_processes,
        tile=flags.tile,
        tile_overlap=flags.tile_overlap,
        keep_going=flags.keep_going,
        dry_run=dry_run,
        overwrite=flags.overwrite,
        resume=flags.resume,
        write_pixel_size=flags.write_config_scale_to_output,
    )


@dataclass(frozen=True, kw_only=True)
class _Calibration:
    """The calibration ``wo rec`` runs first: the position and time, and where its store is published."""

    position: Path
    name: str
    time: int
    path: Path


def _calibration(flags: Flags, resolved: ResolvedConfig, *, problems: Problems) -> _Calibration | None:
    """Return what ``wo rec`` calibrates on (the first position and selected time) and where the store goes.

    The store is ``<output stem>_transfer_function_<config stem>.zarr`` beside the output plate, so two outputs
    reconstructed with one configuration never share a store; it replaces the one an earlier run of the command
    published there, and nothing else. ``None`` when a part is refused.
    """
    positions: list[Path] = []
    with collecting(problems, resolved):
        positions = expand_positions(flags.inputs)
    output = flags.output.absolute()
    path = output.parent / f"{output.stem}_transfer_function_{flags.config.stem}.zarr"
    with collecting(problems, resolved):
        expect_store_output(path, overwrite=True, inputs=positions, location=str(path))
    if not positions:
        return None
    times = resolved.times
    name = plate_name(positions[0])
    name = str(positions[0]) if name == SINGLE_POSITION else name
    return _Calibration(position=positions[0], name=name, time=times[0] if times else 0, path=path)


def _checked(model: ReconstructionModel, flags: Flags, resolved: ResolvedConfig) -> _Calibration | None:
    """Run every check that needs no computation and refuse the request with all the problems found.

    The ``-t`` store is loaded into the model, or the calibration the command runs first is resolved; ``--resume``
    loads the store an interrupted run published beside the output instead, so the run it resumes keeps its
    values (and their fingerprint). The checks of the reconstruction (``dry_run=True``) then run when something
    was refused already, so its problems are reported too, and before a calibration, so nothing is computed for a
    request that would be refused. Otherwise the reconstruction itself runs them, once.

    Returns
    -------
    _Calibration or None
        The calibration to run first; ``None``: none.

    Raises
    ------
    Rejected
        With every problem found, in the order found.
    """
    problems = Problems()
    calibration = None
    if flags.store is not None:
        with collecting(problems, resolved):
            model.load_calibration(flags.store)
    elif free_specs(model.imaging_model):
        calibration = _calibration(flags, resolved, problems=problems)
        if calibration is not None and flags.resume and calibration.path.exists():
            with collecting(problems, resolved):
                model.load_calibration(calibration.path)
            calibration = None
    if problems.items or calibration is not None:
        with collecting(problems, resolved):
            _positions(model, flags, resolved, dry_run=True)
    problems.raise_if_any()
    return calibration


def _run(
    model: ReconstructionModel,
    flags: Flags,
    resolved: ResolvedConfig,
    *,
    calibration: _Calibration | None,
    observed: Observation,
) -> ReconstructionReport:
    """Calibrate first when the command has to, publishing the store beside the output; then reconstruct."""
    with located(resolved):
        if calibration is not None:
            observed.calibrating(position=calibration.name, time_index=calibration.time)
            store = model.calibrate_position(
                calibration.position,
                channels=resolved.channels,
                pixel_size=resolved.pixel_size,
                time=calibration.time,
            )
            assert store is not None
            store.save(calibration.path, overwrite=True)
            observed.calibrated(store, path=calibration.path)
        return _positions(model, flags, resolved, dry_run=False)


def _summary(report: ReconstructionReport) -> list[str]:
    """Return the summary lines: the counts, then one line per failed unit.

    ``out.zarr: finished; 4 units: 4 reconstructed, 0 failed, 0 left``, with the empty and the skipped units when
    there are any; a run a unit stopped has units left.
    """
    counts = [f"{report.count('complete')} reconstructed"]
    counts += [f"{report.count(status)} {status}" for status in ("empty", "skipped") if report.count(status)]
    counts += [f"{report.count('failed')} failed", f"{len(report.pending)} left"]
    noun = "unit" if report.total == 1 else "units"
    lines = [f"{report.output}: {report.status}; {report.total} {noun}: {', '.join(counts)}"]
    lines += [f"  failed {result.unit}: {result.error}" for result in report.results if result.status == "failed"]
    return lines


def reconstruct(flags: Flags) -> None:
    """Build the model, check the request, give the model its calibration and reconstruct the positions.

    Prints the summary, and exits 3 when ``--keep-going`` finished with failed units. A reconstruction a unit
    stopped prints its counts, then the error boundary reports the failure (exit 1).
    """
    check_wandb(flags.wandb)
    resolved = resolve_config(flags.config, device=flags.device)
    model = build_model(resolved)
    try:
        with observing(flags.wandb, job_type=flags.command, store=flags.output.name) as observed:
            calibration = _checked(model, flags, resolved)
            report = _run(model, flags, resolved, calibration=calibration, observed=observed)
            observed.exit_code = FAILED_UNITS if report.count("failed") else 0
    except ReconstructionFailed as failed:
        typer.echo(_summary(failed.report)[0])  # the failed unit's error follows as the error line
        raise
    for line in _summary(report):
        typer.echo(line)
    if report.count("failed"):
        raise typer.Exit(FAILED_UNITS)


def _grid(
    resolved: ResolvedConfig, model: ReconstructionModel, inputs: Sequence[Path], *, output: Path, overwrite: bool
) -> tuple[list[tuple[int, int, int]], tuple[float, float, float] | None]:
    """Check ``-i`` and ``-o``, each whatever the other is; return the positions' distinct shapes and voxel size.

    The voxel size is ``data.pixel_size``, or with ``auto`` the one the positions record, which must be one.

    Raises
    ------
    Rejected
        With every problem found, in the order found.
    """
    problems = Problems()
    positions: list[Path] = []
    shapes: list[tuple[int, int, int]] = []
    sizes: list[tuple[float, float, float]] = []
    with collecting(problems, resolved):
        positions = expand_positions([str(path) for path in inputs])
        metas = [read_position_meta(path) for path in positions]
        shapes = list(dict.fromkeys(meta.zyx for meta in metas))
        given = "auto" if resolved.pixel_size is None else resolved.pixel_size
        sizes = list(dict.fromkeys(tuple(position_pixel_size(model, meta, given)) for meta in metas))
    if len(sizes) > 1:
        problems.add(
            f"{named('pixel_size')} is auto, but the positions record {len(sizes)} voxel sizes "
            f"({'; '.join(', '.join(map(str, size)) for size in sizes)} um); a store holds one: set it, or compute "
            "a store per voxel size",
            str(resolved.path),
        )
    with collecting(problems, resolved):
        expect_store_output(output, overwrite=overwrite, inputs=positions)
    problems.raise_if_any()
    return shapes, sizes[0] if sizes else None


def compute_transfer_function(
    config_filepath: Path, inputs: Sequence[Path], output: Path, *, device: str | None, overwrite: bool
) -> None:
    """Write the store of the configuration's nominal parameters on the inputs' shapes (``wo compute-tf``)."""
    resolved = resolve_config(config_filepath, device=device)
    model = build_model(resolved)
    shapes, pixel_size = _grid(resolved, model, inputs, output=output, overwrite=overwrite)
    with located(resolved):
        placed = resolved.wants_transfer or model.field_tiles is not None
        store = model.to_tf_store(shapes=shapes if placed else (), pixel_size=pixel_size)
        store.save(output, overwrite=overwrite)
    status(f"wrote {output}")
    said = "nominal parameters"
    if store.transfers:
        said += f"; transfer functions at {', '.join('x'.join(map(str, item.shape)) for item in store.transfers)}"
    elif store.tiles > 1:
        said += f"; {store.tiles} field tiles (no transfer function is stored for a field-tiled system)"
    typer.echo(f"{output}: {said}")
    for line in parameter_table(store):
        typer.echo(line)
