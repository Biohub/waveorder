"""The ``waveorder`` and ``wo`` command line.

The command line is an implementation of the Python API: a command builds a model from its configuration
(:meth:`waveorder.models.ReconstructionModel.from_config`) and calls the model's methods, one flag per argument.
Commands import :mod:`waveorder.config`, :mod:`waveorder.models` and :mod:`waveorder.io` from inside their bodies,
so ``wo --help``, ``wo --version`` and ``config new`` never import torch. Errors go through one boundary
(:class:`waveorder.cli.render.RootGroup`).
"""

import inspect
from importlib.metadata import version
from typing import Annotated

import typer

from waveorder.cli import bench, calibrate, config, reconstruct, simulate, train
from waveorder.cli.options import WoCommand
from waveorder.cli.render import RootGroup

__all__ = ["app", "main"]

app = typer.Typer(
    name="wo",
    cls=RootGroup,
    help=(
        "waveorder \\[[green]wo[/green]]: Wave-optical simulation and reconstruction\n\n"
        "Start with [green]wo config new[/green] to write a configuration file."
    ),
    epilog="Shell completion: [green]wo --install-completion[/green] or [green]wo --show-completion[/green]",
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "token_normalize_func": lambda token: token.replace("_", "-"),
    },
    pretty_exceptions_enable=False,
)


def _version(value: bool) -> None:  # noqa: FBT001 (Typer passes the value positionally)
    if value:
        typer.echo(f"waveorder {version('waveorder')}")
        raise typer.Exit


@app.callback()
def _root(
    *,
    show_version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """Wave-optical simulation and reconstruction."""
    del show_version


_COMMANDS = {
    "reconstruct": ("rec", reconstruct.reconstruct, "Reconstruct"),
    "apply-inverse-transfer-function": ("apply-inv-tf", reconstruct.apply_inverse_transfer_function, "Reconstruct"),
    "compute-transfer-function": ("compute-tf", reconstruct.compute_transfer_function, "Calibrate"),
    "calibrate-transfer-function": ("calibrate-tf", calibrate.calibrate_transfer_function, "Calibrate"),
    "simulate": ("sim", simulate.simulate, "Simulate"),
}
"""Each command with its shortcut (3.x), its function and its help panel."""

for _name, (_alias, _command, _panel) in _COMMANDS.items():
    _summary = " ".join(inspect.cleandoc(_command.__doc__ or "").split("\n\n")[0].split())
    app.command(
        _name,
        cls=WoCommand,
        no_args_is_help=True,
        short_help=f"[green]\\[{_alias}][/green] {_summary}",
        rich_help_panel=_panel,
    )(_command)
    app.command(_alias, cls=WoCommand, no_args_is_help=True, hidden=True)(_command)

app.add_typer(config.app, name="config", rich_help_panel="Configure")
for _alias in ("migrate-config", "mc"):  # 3.x spellings of wo config migrate
    app.command(_alias, hidden=True, **config.MIGRATE_COMMAND)(config.migrate)  # ty: ignore[invalid-argument-type]
app.add_typer(train.app, name="distill", rich_help_panel="Distill")
app.add_typer(
    bench.app, name="benchmark", rich_help_panel="Benchmark", short_help="[green]\\[bm][/green] Run benchmarks"
)
app.add_typer(bench.app, name="bm", hidden=True)


def main() -> None:
    """Run ``wo`` (``python -m waveorder.cli``)."""
    app(prog_name="wo")
