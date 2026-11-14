"""``wo config``: write, check, show and migrate WaveOrder 4 configuration files.

Command bodies import :mod:`waveorder.config` when they run, so ``wo --help`` loads neither pydantic nor torch, and
``config new`` never imports torch. ``config check``, ``config show`` and ``config migrate`` build the model of the
configuration on the CPU (:func:`waveorder.cli._configured.build_model`).
"""

import textwrap
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer
from typer._click import Context, HelpFormatter

from waveorder.cli.options import ConfigFile, ContrastName, OutputFilepath, Overwrite, WoCommand
from waveorder.cli.render import status
from waveorder.errors import Rejected

app = typer.Typer(
    name="config", no_args_is_help=True, help="Write, check, show and migrate WaveOrder 4 configurations."
)


def template(contrast: str, dimension: int) -> str:
    """Return the commented configuration template for ``contrast`` and ``dimension`` (``2`` or ``3``)."""
    return (resources.files("waveorder.cli") / "templates" / f"{contrast}_{dimension}d.yml").read_text()


def write_text(
    text: str, output: Path | None, *, overwrite: bool, reads: Path | None = None, location: str | None = None
) -> None:
    """Print ``text`` on stdout, or write it to ``output`` (``-`` is stdout).

    Raises
    ------
    Rejected
        At ``location`` (default ``-o <output>``), if ``output`` is ``reads`` (the file the command reads), its
        directory does not exist, or it exists and ``overwrite`` is false.
    """
    if output is None or str(output) == "-":
        typer.echo(text, nl=False)
        return
    where = location or f"-o {output}"
    if reads is not None and output.resolve() == reads.resolve():
        raise Rejected.one(f"is the file being read, {reads}; write to another file", location=where)
    if not output.parent.is_dir():
        raise Rejected.one(f"directory {output.parent} does not exist", location=where)
    if output.exists() and not overwrite:
        raise Rejected.one("exists; pass --overwrite to replace it", location=where)
    output.write_text(text)
    status(f"wrote {output}")


def _dimension(value: int) -> int:
    if value not in (2, 3):
        raise typer.BadParameter(f"{value} must be 2 or 3")
    return value


@app.command("new", cls=WoCommand)
def new(
    *,
    contrast: Annotated[
        ContrastName,
        typer.Option(
            "--contrast",
            metavar="phase|fluorescence",
            rich_help_panel="Template",
            help="Contrast mechanism of the imaging system.",
        ),
    ] = ContrastName.phase,
    dimension: Annotated[
        int,
        typer.Option(
            "--dimension",
            metavar="2|3",
            callback=_dimension,
            rich_help_panel="Template",
            help="3 for a thick object, 2 for a thin one.",
        ),
    ] = 3,
    output: OutputFilepath = None,
    overwrite: Overwrite = False,
) -> None:
    """Print a commented configuration to start from.

    Example: wo config new --contrast phase --dimension 3 -o phase.yml
    """
    write_text(template(contrast.value, dimension), output, overwrite=overwrite)


@app.command("check", cls=WoCommand, no_args_is_help=True)
def check(*, file: ConfigFile) -> None:
    """Check that a configuration file is valid and builds into modules that can run.

    The model is built on the CPU. A solver.device this machine does not have is a warning, since the run checks it
    where it executes. For solver.padding.z auto, it prints the slices auto pads at least at data.pixel_size.

    Example: wo config check phase.yml
    """
    from waveorder.cli._configured import build_model, resolve_config
    from waveorder.models._parts import axial_extent, least_z_padding

    model = build_model(resolve_config(file), on_cpu=True)
    if model.z_padding == "auto":
        rule = f"n * wavelength / NA**2 = {axial_extent(model.system):.4g} um"
        if model.pixel_size is None:
            typer.echo(
                f"solver.padding.z: auto pads at least {rule} at each end of z; the slices are resolved per position "
                "from its z voxel size and stack"
            )
        else:
            step = model.pixel_size[0]
            typer.echo(
                f"solver.padding.z: auto pads at least {least_z_padding(model.system, step)} slices at each end "
                f"({rule} at a z voxel size of {step:g} um), then more for a fast FFT length of each stack"
            )
    typer.echo(f"ok: {file}")


@app.command("show", cls=WoCommand, no_args_is_help=True)
def show(
    *,
    file: ConfigFile,
    transfer_function_dirpath: Annotated[
        Path | None,
        typer.Option(
            "-t",
            "--transfer-function-dirpath",
            metavar="STORE",
            rich_help_panel="Inputs",
            help="A transfer-function store whose values fill the calibrated block.",
        ),
    ] = None,
    output: OutputFilepath = None,
    overwrite: Overwrite = False,
) -> None:
    """Print the configuration of the model it builds, every default filled in.

    With -t, the calibrated block holds the value the store holds for every free quantity: the configuration a
    calibration found; closing comments name the score arrays and the grids of the transfer functions the store
    holds. The configuration is checked as wo config check does.

    Example: wo config show phase.yml -t tf.zarr
    """
    from waveorder.cli._configured import build_model, located, resolve_config
    from waveorder.config import dump_yaml
    from waveorder.io.tf_store import TfStore

    resolved = resolve_config(file)
    model = build_model(resolved, on_cpu=True)
    notes: list[str] = []
    with located(resolved):
        if transfer_function_dirpath is not None:
            model.load_calibration(transfer_function_dirpath)
            store = TfStore.load(transfer_function_dirpath)
            notes = [f"the store holds {name}, shape {list(values.shape)}" for name, values in store.scores.items()]
            grids = [(item.shape, item.attrs.get("pixel_size_zyx_um")) for item in store.transfers]
            notes += [f"the store holds a transfer function on {'x'.join(map(str, z))} ({um} um)" for z, um in grids]
            table = model.tile_parameters
            if table is not None:
                tiles = max(int(values.shape[0]) for values in table.values())
                raise Rejected.one(
                    f"holds {tiles} tiles of a field-tiled calibration; per-tile values have no single config",
                    location=f"-t {transfer_function_dirpath}",
                )
        shown = model.to_config()
    assert not isinstance(shown, dict)  # a model built from a configuration returns one
    configured = resolved.config.solver  # the model was built on the CPU
    solver = shown.solver.model_copy(update={"device": configured.device})
    text = dump_yaml(shown.model_copy(update={"solver": solver}))
    text += "".join(f"# {note}\n" for note in notes)
    write_text(text, output, overwrite=overwrite, reads=file)


class _MigrateCommand(WoCommand):
    """``config migrate``: its help ends with the per-key table, read from :mod:`waveorder.config.migrate`.

    The table is built when the help is shown, so ``wo --help`` does not import the configuration package.
    """

    def format_help(self, ctx: Context, formatter: HelpFormatter) -> None:
        """Set the epilog to the migration table, each row wrapped under a hanging indent, then format the help."""
        from rich.console import Console
        from typer import rich_utils

        from waveorder.config.migrate import table_help

        width = Console(width=rich_utils.MAX_WIDTH).width - 2  # the epilog is padded by one column on each side
        rows = [
            textwrap.fill(row, width=width, subsequent_indent="    ", break_long_words=False, break_on_hyphens=False)
            for row in table_help().splitlines()
        ]
        self.epilog = "3.x key (3.x default): what it becomes\n\n" + "\n".join(rows)
        super().format_help(ctx, formatter)


MIGRATE_COMMAND: dict[str, object] = {"cls": _MigrateCommand, "no_args_is_help": True}
"""Settings of ``config migrate`` and of its hidden root spellings ``migrate-config`` and ``mc``."""


@app.command("migrate", **MIGRATE_COMMAND)  # ty: ignore[invalid-argument-type]
def migrate(
    *,
    file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, metavar="FILE", help="waveorder 3.x configuration, or a #592 version-4 file."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-filepath",
            dir_okay=False,
            metavar="FILE",
            rich_help_panel="Outputs",
            help="Write to this file; - is stdout. Default: <input stem>_v4.yml beside the input.",
        ),
    ] = None,
    overwrite: Overwrite = False,
) -> None:
    """Convert a waveorder 3.x configuration to WaveOrder 4.

    Each 3.x key is converted, dropped with a note in the output's header comment, or rejected (exit 2, nothing
    written) with its key; the table below lists them. Keys the file leaves out take their 3.x defaults, written out
    explicitly. The result is checked as wo config check does before it is written. A version-4 file of the #592
    preview (pixel_size.yx) is upgraded; a current WaveOrder 4 file is left as it is.

    Example: wo config migrate phase_3x.yml -o phase.yml
    """
    from waveorder.cli._configured import ResolvedConfig, build_model
    from waveorder.config import CONFIG_ERRORS, load, problems
    from waveorder.config.loader import parse
    from waveorder.config.migrate import migrate as translate
    from waveorder.config.migrate import render
    from waveorder.errors import Problem

    try:
        result = translate(parse(file.read_text()))
    except (*CONFIG_ERRORS, OSError) as error:
        raise Rejected(problems(error, source=str(file))) from error
    if result.rejections:
        raise Rejected([Problem(f"cannot migrate: {rejection}", str(file)) for rejection in result.rejections])
    if result.dialect == "4":
        status(f"{file} is already a WaveOrder 4 configuration; nothing to migrate")
        return
    text = render(result, source=file.name)
    try:
        config = load(parse(text))
        build_model(ResolvedConfig(path=file, config=config), on_cpu=True)
    except Rejected as error:
        raise Rejected([Problem(problem.message, f"{file} (migrated)") for problem in error.problems]) from error
    if output is None:
        target = file.with_name(f"{file.stem}_v4.yml")
        write_text(text, target, overwrite=overwrite, location=str(target))
    else:
        write_text(text, output, overwrite=overwrite, reads=file)
    for note in result.notes:
        status(f"note: {note}")
