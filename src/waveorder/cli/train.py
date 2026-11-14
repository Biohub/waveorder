"""``wo distill``: index plate pairs, train a distilled model, and score and export its bundle.

A distilled model is a network (the student) trained on pairs of a raw plate and its reconstruction, made beforehand
with ``wo rec``, then run as the reconstruction model's solver (``solver.reconstruction: {distilled: {bundle: ...}}``).
``index`` is ``index_from_plates`` and ``FovIndex.save``; ``fit``, ``validate`` and ``test`` are Lightning's command
line (:class:`waveorder.models.DistillCLI`): configuration files (``--config``, repeatable), flags and
``--ckpt_path``; ``evaluate`` and ``export`` take a bundle. Command bodies import those packages, so ``wo --help``
imports no torch.
"""

import contextlib
from collections.abc import Generator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from waveorder.cli.options import Overwrite, WoCommand
from waveorder.errors import ParameterError, Rejected, argument_names

app = typer.Typer(
    name="distill",
    no_args_is_help=True,
    help="Train, score and export distilled models: networks trained on pairs of raw and reconstructed plates, run "
    "as the model's solver.",
)

_INPUTS, _OUTPUTS, _RUN = "Inputs", "Outputs", "Run"


class SplitLevel(StrEnum):
    """What ``wo distill index --split-by`` assigns to a split as a whole."""

    position = "position"
    well = "well"
    plate = "plate"


BundleDirpath = Annotated[
    Path,
    typer.Argument(
        exists=True, file_okay=False, metavar="BUNDLE", help="A bundle directory, such as RUN/best of a fit."
    ),
]


@contextlib.contextmanager
def _flagged(flags: Mapping[str, str]) -> Generator[None]:
    """Refuse what a function refuses about an argument at the flag that gave it (exit 2)."""
    try:
        yield
    except ParameterError as error:
        raise Rejected.one(str(error), location=flags.get(error.path or "")) from error


# --- index ----------------------------------------------------------------------------------------------------------


class _IndexCommand(WoCommand):
    """``distill index``: ``--targets`` takes every value up to the next option, as ``-i`` does."""

    greedy = frozenset({"targets"})


def _fractions(value: str, *, flag: str = "--fractions") -> dict[str, float]:
    """Turn ``train=0.7,val=0.1,test=0.2`` into the fractions of a ``GroupSplit``."""
    shares: dict[str, float] = {}
    for part in value.split(","):
        name, separator, number = part.partition("=")
        try:
            shares[name.strip()] = float(number)
        except ValueError:
            separator = ""
        if not separator or not name.strip():
            raise typer.BadParameter(
                f"{value!r} must read NAME=FRACTION,..., e.g. train=0.7,val=0.1,test=0.2", param_hint=flag
            )
    return shares


def _named_split(value: str) -> tuple[str, SplitLevel, dict[str, float]]:
    """Turn ``plates=plate:train=0.8,val=0.1,test=0.1`` into a split name, its level and its fractions."""
    name, _, rest = value.partition("=")
    level, _, fractions = rest.partition(":")
    if not name or level not in SplitLevel.__members__ or not fractions:
        raise typer.BadParameter(
            f"{value!r} must read NAME=LEVEL:NAME=FRACTION,..., e.g. plates=plate:train=0.8,val=0.1,test=0.1",
            param_hint="--named-split",
        )
    return name, SplitLevel(level), _fractions(fractions, flag="--named-split")


@app.command("index", cls=_IndexCommand, no_args_is_help=True)
def index(  # noqa: PLR0913 (one keyword per flag)
    *,
    inputs: Annotated[
        list[Path],
        typer.Argument(metavar="INPUT...", help="Raw plates, positions or quoted globs, e.g. 'raw.zarr/*/*/*'."),
    ],
    channel: Annotated[
        list[str],
        typer.Option("--channel", metavar="NAME", rich_help_panel=_INPUTS, help="Input channel to read; repeatable."),
    ],
    targets: Annotated[
        list[Path],
        typer.Option(
            "--targets",
            metavar="PLATE",
            rich_help_panel=_INPUTS,
            help="One reconstructed plate per input (wo rec), holding each position's target under the same name.",
        ),
    ],
    target_channel: Annotated[
        list[str],
        typer.Option(
            "--target-channel",
            metavar="NAME",
            rich_help_panel=_INPUTS,
            help="Target channel to read from the targets, e.g. Phase2D; repeatable.",
        ),
    ],
    split_by: Annotated[
        SplitLevel,
        typer.Option(
            "--split-by",
            metavar="position|well|plate",
            rich_help_panel="Split",
            help="What goes to one split as a whole; position splits the positions within each well.",
        ),
    ] = SplitLevel.well,
    fractions: Annotated[
        str,
        typer.Option(
            "--fractions", metavar="NAME=F,...", rich_help_panel="Split", help="Split names and their fractions."
        ),
    ] = "train=0.7,val=0.1,test=0.2",
    named_split: Annotated[
        list[str] | None,
        typer.Option(
            "--named-split",
            metavar="NAME=LEVEL:NAME=F,...",
            rich_help_panel="Split",
            help="Another split, written as the column split_NAME, which a run picks with data.split; repeatable.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dirpath",
            file_okay=False,
            metavar="DIR",
            rich_help_panel=_OUTPUTS,
            help="The index directory: one parquet file per plate and manifest.json.",
        ),
    ],
    append: Annotated[
        bool,
        typer.Option("--append", rich_help_panel=_OUTPUTS, help="Add the plates to the index at DIR."),
    ] = False,
    overwrite: Overwrite = False,
) -> None:
    """Index the fields of view of raw plates and their reconstructions, split by group.

    One row per position and time point, with the plate, the well, its target and a split column per split. Plates
    added with --append keep the index's recorded splits and plate names: a split by position gives them the splits
    they would get together, and a split by well or plate sends each new group to the split furthest below its
    fraction.

    Example: wo distill index raw.zarr --channel BF --targets phase.zarr --target-channel Phase2D -o index
    """
    from waveorder.data import GroupSplit, index_from_plates

    shares = _fractions(fractions)
    with _flagged({"by": "--split-by", "fractions": "--fractions"}):
        split = GroupSplit(by=split_by.value, fractions=shares)
    named = {}
    for name, level, shares_of in map(_named_split, named_split or []):
        with _flagged({"by": "--named-split", "fractions": "--named-split"}):
            named[name] = GroupSplit(by=level.value, fractions=shares_of)
    token = argument_names.set({**argument_names.get(), "positions": "INPUT"})
    try:
        with _flagged({"attributes": "INPUT", "targets": "--targets", "splits": "--named-split"}):
            found = index_from_plates(
                [str(path) for path in inputs],
                channels=channel,
                targets=[str(path) for path in targets],
                target_channels=target_channel,
                split=split,
                splits=named,
                append_to=output if append else None,
            )
    finally:
        argument_names.reset(token)
    found.save(output, overwrite=overwrite, append=append)
    shown = ", ".join(f"{name} {count}" for name, count in found.counts.items())
    typer.echo(f"{output}: {len(found)} fields: {shown}")


# --- fit, validate, test: Lightning's command line --------------------------------------------------------------------

_LIGHTNING = {
    "context_settings": {"allow_extra_args": True, "ignore_unknown_options": True, "help_option_names": []},
    "add_help_option": False,
}
"""Every argument, ``--help`` included, goes to Lightning's parser."""


@app.command("fit", **_LIGHTNING)  # ty: ignore[invalid-argument-type]
def fit(ctx: typer.Context) -> None:
    """Train a student: --config FILE (repeatable, later files override), flags, --ckpt_path to resume.

    The run folder (the logger's, runs/lightning_logs/version_N by default) gets config.yaml (the resolved
    configuration, absolute paths), run.json (the inputs' hashes, the split, the software and commit), best/ and
    last/ (bundles) and checkpoints/last.ckpt. wo distill fit --help lists every key.

    Example: wo distill fit --config base.yml --config cytoland.yml --trainer.max_epochs 20
    """
    _lightning("fit", ctx.args)


@app.command("validate", **_LIGHTNING)  # ty: ignore[invalid-argument-type]
def validate(ctx: typer.Context) -> None:
    """Score a checkpoint on the val split: the run's configuration and --ckpt_path.

    Example: wo distill validate --config RUN/config.yaml --ckpt_path RUN/checkpoints/last.ckpt
    """
    _lightning("validate", ctx.args)


@app.command("test", **_LIGHTNING)  # ty: ignore[invalid-argument-type]
def test(ctx: typer.Context) -> None:
    """Score a checkpoint on patches of the test split: the run's configuration and --ckpt_path.

    Example: wo distill test --config RUN/config.yaml --ckpt_path RUN/checkpoints/last.ckpt
    """
    _lightning("test", ctx.args)


def _lightning(subcommand: str, args: list[str]) -> None:
    from waveorder.models._lightning import main

    main([subcommand, *args])


# --- bundles --------------------------------------------------------------------------------------------------------


def _table(result: Mapping[str, Any]) -> list[str]:
    """Return the lines of an evaluation's summary: one row per metric, its mean and percentiles."""
    count, seconds = len(result["fields"]), result["seconds_per_field"]
    lines = [
        f"{result['split']}: {count} fields, {seconds:.3g} s per field",
        f"{'metric':<18}" + "".join(f"{name:>12}" for name in ("mean", "p10", "p50", "p90")),
    ]
    for metric, summary in result["summary"].items():
        lines.append(f"{metric:<18}" + "".join(f"{summary[name]:>12.4g}" for name in ("mean", "p10", "p50", "p90")))
    return lines


@app.command("evaluate", cls=WoCommand, no_args_is_help=True)
def evaluate(
    *,
    bundle: BundleDirpath,
    index_dir: Annotated[
        Path,
        typer.Option(
            "--index",
            exists=True,
            file_okay=False,
            metavar="DIR",
            rich_help_panel=_INPUTS,
            help="The index directory whose test split is scored, as wo distill index writes it.",
        ),
    ],
    model: Annotated[
        Path | None,
        typer.Option(
            "--model",
            exists=True,
            dir_okay=False,
            metavar="FILE",
            rich_help_panel=_INPUTS,
            help="The waveorder 4 configuration of the model: the student then runs as its solver, as in "
            "production. By default the student runs alone, on data prepared by the model the bundle records.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            metavar="DEVICE",
            rich_help_panel=_RUN,
            help="Where to run, e.g. cpu or cuda:0; by default where the model is, else the CPU.",
        ),
    ] = None,
) -> None:
    """Score a bundle's student on the whole fields of an index's test split against their recorded targets.

    Prints the mean and the 10th, 50th and 90th percentiles of the SSIM, the dynamic-range ratio and the student's
    seconds per field, and with --model the error inside and outside the model's passband.

    Example: wo distill evaluate run/best --index index --model phase_2d.yml
    """
    from waveorder.data import FovIndex, PatchData, RandomPatches
    from waveorder.io.bundle import load_bundle
    from waveorder.models import ReconstructionModel, _distill
    from waveorder.train import evaluate as score

    loaded = load_bundle(bundle)
    reconstruction = None if model is None else ReconstructionModel.from_config(model)
    preparing = _distill.recorded_model(loaded) if reconstruction is None else reconstruction
    fields = FovIndex.load(index_dir)
    size = (min(fields.column("height"), default=1), min(fields.column("width"), default=1))
    data = PatchData(
        fields,
        patches=RandomPatches(size, per_field=1),  # unused: whole fields are read
        workers=0,
        conditions=loaded.conditions,
        prepare=_distill.prepare(preparing),
    )
    with _flagged({"split": "--index"}):
        if reconstruction is None:
            result = score(loaded, data, device=device)
        else:
            result = _distill.evaluate(loaded, data, model=reconstruction, device=device)
    for line in _table(result):
        typer.echo(line)


@app.command("export", cls=WoCommand, no_args_is_help=True)
def export(
    *,
    bundle: BundleDirpath,
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-filepath",
            dir_okay=False,
            metavar="FILE",
            rich_help_panel=_OUTPUTS,
            help="The ONNX file; by default model.onnx inside the bundle, whose hash the bundle then records.",
        ),
    ] = None,
) -> None:
    r"""Write a bundle's student as an ONNX file, its normalizers folded in, checked against the bundle's test tensors.

    The graph takes the model's prepared data (B, C, Z, Y, X) and one (B, K) input per condition, and returns the
    estimate in the model's units; the batch size and Y, X are free, the plane count that of the bundle's test
    input. It needs pip install "waveorder\[export]".

    Example: wo distill export run/best -o model.onnx
    """
    from waveorder.models._distill import export as export_onnx

    written = export_onnx(bundle, output)
    typer.echo(f"{written}: checked against the test tensors of {bundle}")
