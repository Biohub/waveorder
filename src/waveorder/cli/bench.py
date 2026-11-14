"""``wo bm`` (``wo benchmark``): run the regression suite of ``benchmarks/`` and read its runs.

The suite lives in the source checkout, beside ``src/``, and is not part of the installed package: the commands
import it lazily from the checkout this ``waveorder`` was installed from (editable install).
"""

import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

app = typer.Typer(
    name="benchmark",
    help="Run and inspect the reconstruction benchmarks of a source checkout.",
    no_args_is_help=True,
)


def _checkout() -> Path:
    """Return the source checkout holding ``benchmarks/``, put on ``sys.path`` so ``import benchmarks`` works."""
    import waveorder

    root = Path(waveorder.__file__).resolve().parents[2]
    if not (root / "benchmarks" / "runner.py").is_file():
        raise typer.BadParameter(
            f"the benchmarks need a source checkout of waveorder: no benchmarks/ beside {root / 'src'}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve_output_dir(cli_value: Path | None) -> Path:
    """Return ``-o``, else ``$WAVEORDER_BENCH_OUTPUT``, else the current directory."""
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get("WAVEORDER_BENCH_OUTPUT")
    return Path(env_value) if env_value else Path()


OutputDir = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Root output directory; default $WAVEORDER_BENCH_OUTPUT if set, else the current directory.",
    ),
]


@app.command()
def run(
    *,
    experiment: Annotated[
        Path | None,
        typer.Option("--experiment", "-e", exists=True, help="Experiment YAML. Default: the regression suite."),
    ] = None,
    scope: Annotated[Literal["synthetic", "all"], typer.Option("--scope", help="Which cases to run.")] = "synthetic",
    output_dir: OutputDir = None,
    save_all: Annotated[
        bool,
        typer.Option(
            "--save-all",
            help=(
                "Keep every intermediate output. By default transfer function stores are deleted, and "
                "reconstruction and simulated zarrs larger than 25 MB, after the metrics are computed."
            ),
        ),
    ] = False,
) -> None:
    """Run benchmark cases.

    Each case configuration (waveorder 3.x) is migrated with wo config migrate, reconstructed with wo rec and
    scored. A run writes <output>/<timestamp>_<experiment>/ with metadata.json, experiment.yml, summary.json and one
    folder per case.

    Example: wo bm run --scope synthetic -o /tmp/bench
    """
    root = _checkout()
    from benchmarks.config import infer_modality, load_experiment, resolve_recon_config
    from benchmarks.runner import run_hpc_case, run_synthetic_case
    from benchmarks.utils import collect_metadata

    experiment_path = experiment or root / "benchmarks" / "experiments" / "regression.yml"
    exp = load_experiment(experiment_path)
    selected = {name: case for name, case in exp.cases.items() if scope == "all" or case.type != "hpc"}
    n_skipped_hpc = len(exp.cases) - len(selected)

    metadata = collect_metadata()
    run_dir = _resolve_output_dir(output_dir) / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{exp.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.copy2(experiment_path, run_dir / "experiment.yml")

    summary = f"{len(selected)} case{'s' if len(selected) != 1 else ''}"
    if n_skipped_hpc:
        summary += f"; skipping {n_skipped_hpc} hpc"
    typer.echo(typer.style(f"WaveOrder Benchmark — {exp.name}", fg="green", bold=True))
    dirty = " dirty" if metadata["git_dirty"] else ""
    typer.echo(f"  Git: {metadata['git_hash']} ({metadata['git_branch']}){dirty}")
    typer.echo(f"  Experiment: {exp.name} ({summary})")
    typer.echo(f"  Output: {run_dir}")
    typer.echo()
    _print_header()

    results: dict[str, dict[str, Any]] = {}
    for case_name, case in selected.items():
        case_dir = run_dir / "cases" / case_name
        recon_config = resolve_recon_config(case, experiment_path.parent)
        typer.echo(f"  {case_name:21s} running...", nl=False)
        try:
            if case.type == "synthetic":
                if case.phantom is None:
                    raise ValueError(f"synthetic case {case_name!r} has no phantom")
                metrics = run_synthetic_case(
                    phantom_config=case.phantom,
                    recon_config=recon_config,
                    case_dir=case_dir,
                    modality=infer_modality(recon_config),
                    save_all=save_all,
                    reference=case.reference,
                    simulation=case.simulation,
                    recovery=case.recovery,
                )
            else:
                if case.input is None or case.position is None:
                    raise ValueError(f"hpc case {case_name!r} needs input and position")
                metrics = run_hpc_case(
                    input_path=case.input,
                    position=case.position,
                    recon_config=recon_config,
                    case_dir=case_dir,
                    save_all=save_all,
                    reference=case.reference,
                    crop=case.crop,
                    tile=case.tile,
                    tile_overlap=case.tile_overlap,
                )
            elapsed = json.loads((case_dir / "timing.json").read_text()).get("elapsed_s", 0)
            metrics["elapsed_s"] = elapsed
            results[case_name] = metrics
            typer.echo("\033[2K\r", nl=False)
            _print_row(case_name, metrics, elapsed=elapsed)
        except Exception as error:  # noqa: BLE001 (a failed case is recorded and the run goes on)
            typer.echo("\033[2K\r", nl=False)
            typer.echo(f"  {case_name:21s} " + typer.style(f"FAILED: {error}", fg="red"))
            tb = traceback.format_exc()
            typer.echo(tb, err=True)
            results[case_name] = {"error": str(error), "traceback": tb}
        # Written after every case, so an interrupted run keeps what it measured.
        (run_dir / "summary.json").write_text(json.dumps(results, indent=2))


@app.command()
def latest(output_dir: OutputDir = None) -> None:
    """Show the summary of the most recent benchmark run."""
    run_dirs = _list_run_dirs(_resolve_output_dir(output_dir))
    if not run_dirs:
        typer.echo("No benchmark runs found.")
        return
    run_dir = run_dirs[-1]
    typer.echo(typer.style(f"Latest run: {run_dir.name}", fg="green", bold=True))
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        typer.echo(f"  Git: {meta['git_hash']} ({meta['git_branch']})")
    results = _load_summary(run_dir)
    if results:
        typer.echo()
        _print_header()
        for case_name, metrics in results.items():
            if "error" in metrics:
                typer.echo(f"  {case_name:21s} " + typer.style("ERROR", fg="red"))
            else:
                _print_row(case_name, metrics, elapsed=metrics.get("elapsed_s"))


@app.command()
def history(
    output_dir: OutputDir = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of runs to show.")] = 10,
) -> None:
    """List recent benchmark runs."""
    run_dirs = _list_run_dirs(_resolve_output_dir(output_dir))
    if not run_dirs:
        typer.echo("No benchmark runs found.")
        return
    for run_dir in run_dirs[-limit:]:
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        n_cases = len(_load_summary(run_dir))
        typer.echo(f"  {run_dir.name:50s} {n_cases} cases  git={meta.get('git_hash', '?')}")


@app.command()
def compare(
    run_a: Annotated[Path | None, typer.Argument(help="Older run directory name.")] = None,
    run_b: Annotated[Path | None, typer.Argument(help="Newer run directory name.")] = None,
    output_dir: OutputDir = None,
) -> None:
    """Compare midband power, mse and ssim of two benchmark runs; by default the two most recent.

    Example: wo bm compare -o /tmp/bench
    """
    root = _resolve_output_dir(output_dir)
    dirs = [root / run_a, root / run_b] if run_a and run_b else _list_run_dirs(root)[-2:]
    if len(dirs) < 2:
        typer.echo("Need at least 2 runs to compare.")
        return
    dir_a, dir_b = dirs
    summary_a, summary_b = _load_summary(dir_a), _load_summary(dir_b)
    typer.echo(typer.style("Benchmark comparison", fg="green", bold=True))
    typer.echo(f"  A: {dir_a.name}")
    typer.echo(f"  B: {dir_b.name}")
    typer.echo()
    common = sorted(set(summary_a) & set(summary_b))
    if not common:
        typer.echo("No common cases between runs.")
        return
    typer.echo(f"  {'case':21s} {'metric':>10s} {'A':>10s} {'B':>10s} {'delta':>10s}")
    typer.echo("  " + "─" * 66)
    for case_name in common:
        ma, mb = summary_a[case_name], summary_b[case_name]
        if "error" in ma or "error" in mb:
            continue
        pairs = []
        iq_a, iq_b = ma.get("image_quality", {}), mb.get("image_quality", {})
        if "midband_power" in iq_a and "midband_power" in iq_b:
            pairs.append(("midband", iq_a["midband_power"], iq_b["midband_power"]))
        for group in ("with_phantom", "with_reference"):
            ga, gb = ma.get(group, {}), mb.get(group, {})
            pairs += [(key, ga[key], gb[key]) for key in ("mse", "ssim") if key in ga and key in gb]
        label = case_name
        for metric, va, vb in pairs:
            typer.echo(f"  {label:21s} {metric:>10s} {_fmt(va)} {_fmt(vb)} {_fmt(vb - va)}")
            label = ""


def _list_run_dirs(output_dir: Path) -> list[Path]:
    """Return the run directories under ``output_dir``, oldest first."""
    if not output_dir.exists():
        return []
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _load_summary(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Return a run's ``summary.json``, or an empty dict."""
    path = run_dir / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(value: float, width: int = 10) -> str:
    """Format a float as fixed-width scientific notation with an explicit sign."""
    if value == 0:
        return f"{' 0':>{width}s}"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1e}".rjust(width)


def _sparkline(hist: dict[str, list[float]], n_bins: int = 10) -> str:
    """Render a histogram's counts as ``n_bins`` block characters."""
    counts = [float(c) for c in hist["counts"]]
    if len(counts) > n_bins:
        factor = len(counts) // n_bins
        counts = [sum(counts[i * factor : (i + 1) * factor]) for i in range(n_bins)]
    peak = max(counts, default=0.0)
    if peak == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[int(c / peak * 7)] for c in counts)


_W = 10  # metric column width
_H = 10  # min/max column width


def _print_header() -> None:
    """Print the summary table header."""
    header = (
        f"  {'case':21s} {'ref':>3s} {'time':>6s}"
        f" {'midband':>{_W}s} {'mse':>{_W}s} {'ssim':>{_W}s}"
        f"  {'min':>{_H}s} {'histogram':^12s} {'max':>{_H}s}"
    )
    typer.echo(header)
    typer.echo("  " + "─" * (len(header) - 2))


def _ref_badge(metrics: dict[str, Any]) -> str:
    """Return the reference check's badge: pass, fail, or none."""
    check = metrics.get("reference_check")
    if not check:
        return " — "
    return " ✅" if check.get("all_pass") else " ❌"


def _print_reference_failures(case_name: str, metrics: dict[str, Any]) -> None:
    """Print each bound the case's reference check failed."""
    check = metrics.get("reference_check")
    if not check or check.get("all_pass"):
        return
    for path, entry in check.get("per_ref", {}).items():
        if entry.get("pass"):
            continue
        bounds = [f"{key}={entry[key]}" for key in ("min", "max") if entry.get(key) is not None]
        msg = f"    ❌ {case_name}.{path}: value={entry['value']:.4g}  ({', '.join(bounds)})"
        typer.echo(typer.style(msg, fg="red"))


def _print_row(case_name: str, metrics: dict[str, Any], elapsed: float | None = None) -> None:
    """Print one summary table row."""
    iq = metrics.get("image_quality", {})
    phantom = metrics.get("with_phantom", {})
    ssim_str = f"{phantom['ssim']:>{_W}.3f}" if "ssim" in phantom else f"{'—':>{_W}s}"
    mse_str = _fmt(phantom["mse"], _W) if "mse" in phantom else f"{'—':>{_W}s}"
    hist = iq.get("histogram", {})
    if hist:
        min_str, max_str = _fmt(hist["bin_edges"][0], _H), _fmt(hist["bin_edges"][-1], _H)
        spark = _sparkline(hist)
    else:
        min_str = max_str = f"{'—':>{_H}s}"
        spark = ""
    time_str = f"{elapsed:5.1f}s" if elapsed is not None else f"{'—':>6s}"
    typer.echo(
        f"  {case_name:21s} {_ref_badge(metrics)} {time_str} {_fmt(iq.get('midband_power', 0), _W)} {mse_str} "
        f"{ssim_str}  {min_str} {spark:^12s} {max_str}"
    )
    _print_reference_failures(case_name, metrics)
