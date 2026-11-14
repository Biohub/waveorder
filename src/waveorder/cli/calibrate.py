"""``wo calibrate-tf``: estimate the configuration's free quantities from one position and write a ``tf.zarr``.

The command builds the model of the configuration (:meth:`waveorder.models.ReconstructionModel.from_config`), calls
``model.calibrate_position`` on the first position at its first selected time and saves the store it returns. A
request the model's checks refuse is exit 2, with configuration errors located at the file and their setting; a
computation that fails is exit 1.
"""

from pathlib import Path
from typing import Annotated

import typer

from waveorder.cli.options import (
    ConfigFilepath,
    Device,
    InputPositions,
    OutputDirpath,
    Overwrite,
    Wandb,
    expect_store_output,
)
from waveorder.cli.render import status
from waveorder.errors import Problems

_Prior = Annotated[
    Path | None,
    typer.Option(
        "--prior",
        "-t",
        "--transfer-function-dirpath",
        metavar="STORE",
        rich_help_panel="Inputs",
        help="A transfer-function store whose parameters centre the prior.",
    ),
]
_Start = Annotated[
    Path | None,
    typer.Option(
        "--start",
        metavar="STORE",
        rich_help_panel="Inputs",
        help="A transfer-function store whose parameters are the starting point of the search only.",
    ),
]


def calibrate_transfer_function(  # noqa: PLR0913 (one keyword per flag)
    *,
    input_position_dirpaths: InputPositions,
    config_filepath: ConfigFilepath,
    output_dirpath: OutputDirpath,
    prior: _Prior = None,
    start: _Start = None,
    device: Device = None,
    overwrite: Overwrite = False,
    wandb: Wandb = None,
) -> None:
    """Estimate the configuration's free quantities from one position and write a tf.zarr.

    Every imaging-system number written as {nominal, tolerance} (and object.weight: auto) is estimated on the first
    position at its first selected time. solver.calibration chooses the score and the search; without it, the
    evidence and the default search. -i takes positions, plates or globs as wo rec does. The store holds the values
    and their provenance. An existing -o is refused without --overwrite, which replaces a transfer-function store
    and never anything else.

    Example: wo calibrate-tf -i in.zarr -c phase.yml -o tf.zarr
    """
    from waveorder.cli._configured import build_model, collecting, load_store, located, resolve_config
    from waveorder.cli._observation import check_wandb, observing, parameter_table
    from waveorder.io.positions import SINGLE_POSITION, expand_positions, plate_name
    from waveorder.io.tf_store import TfStore

    check_wandb(wandb)
    resolved = resolve_config(config_filepath, device=device)
    model = build_model(resolved)
    problems = Problems()
    positions: list[Path] = []
    with collecting(problems, resolved):
        positions = expand_positions([str(path) for path in input_position_dirpaths])
    with collecting(problems, resolved):
        expect_store_output(output_dirpath, overwrite=overwrite, inputs=positions)
    time = resolved.times[0] if resolved.times else 0
    prior_store = load_store(prior, flag="--prior", problems=problems)
    start_store = load_store(start, flag="--start", problems=problems)

    def calibrate(*, dry_run: bool) -> TfStore | None:
        return model.calibrate_position(
            positions[0],
            channels=resolved.channels,
            pixel_size=resolved.pixel_size,
            time=time,
            prior=prior_store,
            start=start_store,
            dry_run=dry_run,
        )

    if positions and problems.items:  # the calibration's own checks, so that their problems are reported too
        with collecting(problems, resolved):
            calibrate(dry_run=True)
    problems.raise_if_any()
    name = plate_name(positions[0])
    name = str(positions[0]) if name == SINGLE_POSITION else name
    watching = observing(wandb, job_type="calibrate-transfer-function", store=output_dirpath.name)
    with located(resolved), watching as observed:
        observed.calibrating(position=name, time_index=time)
        store = calibrate(dry_run=False)
        assert store is not None
        store.save(output_dirpath, overwrite=overwrite)
    status(f"wrote {output_dirpath}")
    typer.echo(f"calibrated on {name} at t={time}: {output_dirpath}")
    for line in parameter_table(store):
        typer.echo(line)
