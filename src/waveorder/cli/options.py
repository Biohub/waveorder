"""What the ``wo`` commands share: their flags, the command class and logging.

A flag is one argument of a method of :class:`waveorder.models.ReconstructionModel`. The model's checks name the
arguments in their messages; while a command runs they read the flags instead (:data:`ARGUMENT_FLAGS`). Importing
this module imports no torch; the step from a configuration file to a model is :mod:`waveorder.cli._configured`,
which a command imports when it runs.
"""

import logging
import os
import sys
import warnings
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

import typer
from iohub.cli import OptionEatAll
from typer._click import Context
from typer.core import TyperCommand, TyperOption

from waveorder.errors import Rejected, WaveorderWarning, argument_names

# --- logging and help -----------------------------------------------------------------------------------------------

PROGRESS_LOGGER = "waveorder.cli.progress"
"""The logger of the run commands' progress lines: units as they end, calibration stages, the periodic summary."""


class _LogToStderr(logging.Handler):
    """Write records to the current ``sys.stderr``, which a test runner or a Rich progress bar may swap."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except (OSError, ValueError):
            self.handleError(record)


class _Formatter(logging.Formatter):
    """The one layout of stderr's log lines: ``<time> <message>``, with the level first above or below INFO."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802 (logging's name)
        """Format one record in the one layout."""
        level = "" if record.levelno == logging.INFO else f"{record.levelname.lower()}: "
        name = "" if record.name.split(".")[0] == "waveorder" else f"{record.name}: "
        return f"{record.asctime} {level}{name}{record.message}"


def _configure_logging() -> None:
    """Route log records to stderr through one handler: waveorder's warnings, progress and cache lines, others' errors.

    The cache lines (``waveorder.models._stores``) say which transfer functions a store gave. iohub's own stderr
    handler (added when it is imported) is removed, so its records take this layout too.
    """
    root = logging.getLogger()
    if not any(isinstance(handler, _LogToStderr) for handler in root.handlers):
        handler = _LogToStderr()
        handler.setFormatter(_Formatter())
        root.addHandler(handler)
    iohub = logging.getLogger("iohub")
    iohub.handlers.clear()
    iohub.propagate = True
    logging.getLogger("waveorder").propagate = True
    logging.getLogger("waveorder").setLevel(logging.WARNING)
    root.setLevel(logging.ERROR)
    iohub.setLevel(logging.ERROR)
    logging.getLogger(PROGRESS_LOGGER).setLevel(logging.INFO)
    logging.getLogger("waveorder.models._stores").setLevel(logging.INFO)


ARGUMENT_FLAGS = MappingProxyType(
    {
        "positions": "-i",
        "output": "-o",
        "overwrite": "--overwrite",
        "store": "-t",
        "prior": "--prior",
        "start": "--start",
        "write_pixel_size": "--write-config-scale-to-output",
        "resume": "--resume",
        "tile": "--tile",
        "tile_overlap": "--tile-overlap",
        "devices": "--device",
        "workers": "-j",
        "shapes": "-i",
        "time": "the calibration time",
        "channels": "data.input_channel_names",
        "pixel_size": "data.pixel_size",
        "times": "data.time_indices",
        "field_tiles": "imaging_system.field_dependence",
        "z_padding": "solver.padding.z",
    }
)
"""How the command line spells the arguments the model's checks name (:func:`waveorder.errors.named`).

An argument a flag gives is the flag; one the configuration gives is its dotted path there, which
:func:`waveorder.cli._configured.located` moves to the configuration file.
"""


class WoCommand(TyperCommand):
    """A command with a greedy ``-i`` that sets up logging and the flag names before it runs.

    ``-i`` takes every value up to the next option (3.x, iohub's ``OptionEatAll``), so ``-i a b -o out -i c`` gives
    ``a``, ``b`` and ``c``, and an unquoted shell glob passes all its matches. A subclass makes other options greedy
    by naming their parameters in :attr:`greedy`.
    """

    greedy: frozenset[str] = frozenset({"input_position_dirpaths"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 (click's arguments, passed through)
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, TyperOption) and param.name in self.greedy:
                param.__class__ = OptionEatAll

    def collect_usage_pieces(self, ctx: Context) -> list[str]:
        """Return the usage line's pieces with a required argument as ``FILE``, not Typer's ``{FILE}``."""
        return [piece.removeprefix("{").removesuffix("}") for piece in super().collect_usage_pieces(ctx)]

    def invoke(self, ctx: Context) -> object:
        """Configure logging, then run with the model's checks naming the flags (:data:`ARGUMENT_FLAGS`)."""
        _configure_logging()
        token = argument_names.set(ARGUMENT_FLAGS)
        try:
            return super().invoke(ctx)
        finally:
            argument_names.reset(token)


# --- flag values ----------------------------------------------------------------------------------------------------


class FlagWarning(WaveorderWarning):
    """A warning about a flag's value; ``location`` is the flag as given, e.g. ``-j 64``."""

    def __init__(self, message: str, *, location: str) -> None:
        super().__init__(message)
        self.location = location


def _num_processes(value: int | None) -> int | None:
    """Refuse fewer than 1 or more processes than CPUs (3.x's limit); warn above the CPUs this process may use."""
    if value is None:
        return None
    if value < 1:
        raise typer.BadParameter(f"{value} must be at least 1")
    cpus = os.cpu_count() or 1
    if value > cpus:
        raise typer.BadParameter(f"{value} is more than the {cpus} CPUs of this machine")
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpus
    if value > usable:
        message = f"is more than the {usable} CPUs this process may use"
        warnings.warn(FlagWarning(message, location=f"-j {value}"), stacklevel=2)
    return value


def split_devices(value: str | None) -> list[str]:
    """Return the devices of ``--device``: ``cuda:0,cuda:1`` is two, ``cuda`` one, no flag none."""
    return [] if value is None else [part.strip() for part in value.split(",")]


def _check_devices(value: str | None) -> str | None:
    if value is not None and not all(split_devices(value)):
        raise typer.BadParameter(f"{value!r} must be a device or a comma-separated list of devices, e.g. cuda:0,cuda:1")
    return value


def _check_device(value: str | None) -> str | None:
    if value is not None and len(split_devices(_check_devices(value))) > 1:
        raise typer.BadParameter(f"{value!r} is a list; this command computes on one device, e.g. cuda:0")
    return value


def devices_argument(device: str | None, *, configured: str | None) -> list[str] | Literal["auto"]:
    """Return the ``devices`` a reconstruction of positions is spread over.

    Parameters
    ----------
    device : str, optional
        ``--device`` as given.
    configured : str, optional
        ``solver.device`` of the resolved configuration: the first device of ``--device`` when that was given.

    Returns
    -------
    list[str] or "auto"
        The devices of a list as given; a device with an index (``cuda:1``) alone; else ``"auto"``, which is every
        visible GPU for ``cuda`` or ``auto`` on a machine with CUDA, and the model's device otherwise.
    """
    listed = split_devices(device)
    if len(listed) > 1:
        return listed
    return "auto" if configured is None or ":" not in configured else [configured]


def parse_zyx(value: str | None, *, flag: str, minimum: int = 1) -> tuple[int, int, int] | None:
    """Turn ``Z,Y,X`` into three ints of at least ``minimum`` (``--tile``, ``--tile-overlap``).

    Raises
    ------
    typer.BadParameter
        If the text is not three such integers.
    """
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3 or not all(part.isdigit() and int(part) >= minimum for part in parts):
        kind = "positive" if minimum > 0 else "non-negative"
        raise typer.BadParameter(f"{value!r} must be three {kind} voxel counts Z,Y,X, e.g. 16,512,512", param_hint=flag)
    z, y, x = (int(part) for part in parts)
    return (z, y, x)


# --- flags ----------------------------------------------------------------------------------------------------------

InputPositions = Annotated[
    list[Path],
    typer.Option(
        "-i",
        "--input-position-dirpaths",
        metavar="PATH",
        rich_help_panel="Inputs",
        help="Input positions or plates, e.g. -i 'in.zarr/*/*/*'; files among them are skipped.",
    ),
]
ConfigFilepath = Annotated[
    Path,
    typer.Option(
        "-c",
        "--config-filepath",
        exists=True,
        dir_okay=False,
        metavar="FILE",
        rich_help_panel="Inputs",
        help="WaveOrder 4 configuration file (YAML).",
    ),
]
ConfigFile = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, metavar="FILE", help="WaveOrder 4 configuration file (YAML)."),
]
TransferFunctionDirpath = Annotated[
    Path,
    typer.Option(
        "-t",
        "--transfer-function-dirpath",
        metavar="STORE",
        rich_help_panel="Inputs",
        help="A transfer-function store (tf.zarr) whose parameters are used as they are.",
    ),
]
OutputDirpath = Annotated[
    Path,
    typer.Option("-o", "--output-dirpath", metavar="PATH", rich_help_panel="Outputs", help="Output store."),
]
OutputFilepath = Annotated[
    Path | None,
    typer.Option(
        "-o",
        "--output-filepath",
        dir_okay=False,
        metavar="FILE",
        rich_help_panel="Outputs",
        help="Write to this file instead of stdout; - is stdout.",
    ),
]
Overwrite = Annotated[
    bool,
    typer.Option("--overwrite", rich_help_panel="Outputs", help="Replace an existing output."),
]
OutputOverwrite = Annotated[
    bool,
    typer.Option(
        "--overwrite",
        rich_help_panel="Outputs",
        help="Delete an existing output plate and write it anew; without it, units are written into the plate.",
    ),
]
Resume = Annotated[
    bool,
    typer.Option(
        "--resume",
        rich_help_panel="Outputs",
        help="Skip timepoints that an interrupted previous run into the same output already finished. Changed "
        "reconstruction settings or a recomputed transfer function are detected and recomputed; changed input data "
        "is not. Needs an OME-Zarr v0.5 output.",
    ),
]
WriteConfigScaleToOutput = Annotated[
    bool,
    typer.Option(
        "--write-config-scale-to-output",
        rich_help_panel="Outputs",
        help="Write data.pixel_size to the output's scale instead of the input's (a data.pixel_size that differs "
        "from the input's is used for the reconstruction either way, with a warning).",
    ),
]
UniqueId = Annotated[
    str | None,
    typer.Option("--unique-id", "-uid", hidden=True, help="Accepted for wo-4.0 command lines and ignored."),
]
Device = Annotated[
    str | None,
    typer.Option(
        "--device",
        metavar="DEVICE",
        callback=_check_device,
        rich_help_panel="Run",
        help="Compute device, e.g. cpu, cuda, cuda:1 or auto; sets solver.device.",
    ),
]
Devices = Annotated[
    str | None,
    typer.Option(
        "--device",
        metavar="DEVICE[,...]",
        callback=_check_devices,
        rich_help_panel="Run",
        help="Compute device, e.g. cpu, cuda:1 or auto, or a comma-separated list the units are spread over, e.g. "
        "cuda:0,cuda:1. The first is the model's device (solver.device); cuda or auto alone uses every visible GPU.",
    ),
]
NumProcesses = Annotated[
    int | None,
    typer.Option(
        "-j",
        "--num-processes",
        metavar="N",
        callback=_num_processes,
        rich_help_panel="Run",
        help="Worker processes over (position, time) units, shared among the devices in turn; default one per "
        "device; at most the CPUs of this machine.",
    ),
]
Tile = Annotated[
    str | None,
    typer.Option(
        "--tile",
        metavar="Z,Y,X",
        rich_help_panel="Run",
        help="Reconstruct in tiles of this core size; default one tile per volume.",
    ),
]
TileOverlap = Annotated[
    str | None,
    typer.Option(
        "--tile-overlap",
        metavar="Z,Y,X",
        rich_help_panel="Run",
        help="Voxels each tile core is widened by and cross-faded with its neighbours (with --tile).",
    ),
]
KeepGoing = Annotated[
    bool,
    typer.Option(
        "--keep-going", rich_help_panel="Run", help="Report a failed unit and continue with the others (exit 3)."
    ),
]
Wandb = Annotated[
    str | None,
    typer.Option(
        "--wandb",
        metavar="PROJECT",
        rich_help_panel="Run",
        help="Log the run to this Weights & Biases project (pip install waveorder\\[wandb]), grouped by the output "
        "store's name.",
    ),
]


class ContrastName(StrEnum):
    """Contrasts ``wo config new`` has a template for."""

    phase = "phase"
    fluorescence = "fluorescence"


# --- the output store -----------------------------------------------------------------------------------------------


def expect_store_output(
    path: Path, *, overwrite: bool, inputs: Sequence[Path] = (), location: str | None = None
) -> None:
    """Refuse a path a transfer-function store cannot be published at, before anything is computed.

    Parameters
    ----------
    path : Path
        Where the store goes.
    overwrite : bool
        Whether a transfer-function store there is replaced.
    inputs : Sequence[Path]
        The input positions, resolved, which the store must not be, hold or lie in.
    location : str, optional
        Where the problem is reported; ``-o <path>`` by default.

    Raises
    ------
    Rejected
        If ``path`` is, is inside or contains an input position; exists without ``overwrite``; or exists and is
        not a transfer-function store.
    """
    from waveorder.io.tf_store import store_layout

    where = location or f"-o {path}"
    target = path.absolute().resolve()
    for position in inputs:
        relation = (
            "is"
            if target == position
            else "is inside"
            if position in target.parents
            else "contains"
            if target in position.parents
            else None
        )
        if relation is not None:
            raise Rejected.one(f"{relation} the input position {position}; use another -o", location=where)
    if not path.exists():
        return
    if not overwrite:
        raise Rejected.one("exists; pass --overwrite to replace it", location=where)
    if store_layout(path) is None:
        raise Rejected.one("is not a transfer-function store; use another -o", location=where)
