"""The ``wo`` boundary: the command set, exit codes, error lines, flag normalization and the removed flags."""

import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from waveorder.cli import app

runner = CliRunner()

PHASE = """\
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: phase
  illumination: {wavelength: 0.532, numerical_aperture: 0.9}
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "phase.yml"
    path.write_text(PHASE)
    return path


def test_no_arguments_shows_help_and_exits_2():
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.output
    assert "wo config new" in result.output


def test_help_and_version():
    assert runner.invoke(app, ["-h"]).exit_code == 0
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("waveorder ")


@pytest.mark.parametrize(
    "command",
    [
        ["rec"], ["reconstruct"], ["compute-tf"], ["compute-transfer-function"], ["apply-inv-tf"],
        ["apply-inverse-transfer-function"], ["calibrate-tf"], ["calibrate-transfer-function"], ["simulate"], ["sim"],
        ["config", "new"], ["config", "check"], ["config", "show"], ["config", "migrate"], ["migrate-config"], ["mc"],
        ["distill"], ["distill", "index"], ["distill", "fit"], ["distill", "validate"], ["distill", "test"],
        ["distill", "evaluate"], ["distill", "export"],
        ["benchmark"], ["bm"],
    ],
    ids=" ".join,
)  # fmt: skip
def test_every_command_has_help(command):
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output


def test_rec_help_shows_resume_and_tile():
    shown = runner.invoke(app, ["rec", "--help"]).stdout
    assert all(flag in shown for flag in ("--resume", "--tile", "--tile-overlap", "--wandb", "--num-processes"))


@pytest.mark.parametrize(
    "command", [["view"], ["interactive"], ["tile-stitch"], ["create-output"], ["config", "schema"], ["distill", "new"]]
)
def test_commands_not_in_4_are_unknown(command):
    result = runner.invoke(app, [*command, "-h"])
    assert result.exit_code == 2
    assert "No such command" in result.stderr


def test_usage_error_is_plain_text_when_piped(config):
    result = runner.invoke(app, ["config", "check", str(config), "--bogus"])
    assert result.exit_code == 2
    assert "wo config check: error: No such option: --bogus" in result.stderr
    assert "Usage: wo config check" in result.stderr
    assert "\x1b[" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("reconstruct", ["-o", "out.zarr"]),
        ("rec", ["-o", "out.zarr", "--num_processes", "1"]),
        ("apply-inv-tf", ["-t", "tf.zarr", "-o", "out.zarr"]),
        ("rec", ["-o", "out.zarr", "--write-config-scale-to-output", "--resume"]),
    ],
)
def test_run_commands_take_3x_flags_and_preflight_the_request(config, command, arguments):
    result = runner.invoke(app, [command, "-i", "in.zarr", "-c", str(config), *arguments])
    assert result.exit_code == 2
    assert "in.zarr matches nothing on disk" in result.stderr


_REMOVED = {
    "rec": [
        "--json", "--dry-run", "--events", "--torch-trace", "--wandb-project", "--wandb-tag", "--wandb-run-name",
        "--memory-budget", "--calibrate-on", "--time-indices", "--output-version", "--allow-system-mismatch",
        "-t", "-v", "-q", "--traceback",
    ],
    "apply-inv-tf": ["--json", "--dry-run", "--allow-system-mismatch", "--output-version", "--time-indices"],
    "compute-tf": ["--shape", "--dry-run", "-t", "--allow-system-mismatch", "--json"],
    "calibrate-tf": ["--calibrate-on", "--time-indices", "--dry-run", "--json"],
}  # fmt: skip


@pytest.mark.parametrize(
    ("command", "flag"), [(command, flag) for command, flags in _REMOVED.items() for flag in flags]
)
def test_a_removed_flag_is_an_unknown_option(config, command, flag):
    arguments = [command, "-i", "in.zarr", "-c", str(config), "-o", "out.zarr", flag, "x"]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2
    assert f"No such option: {flag}" in result.stderr
    assert re.search(rf"{re.escape(flag)}[\s,]", runner.invoke(app, [command, "--help"]).stdout) is None


def test_device_takes_one_device_or_a_list(config):
    from waveorder.cli._configured import resolve_config
    from waveorder.cli.options import devices_argument, split_devices

    assert split_devices(None) == []
    assert split_devices("cuda:0, cuda:1") == ["cuda:0", "cuda:1"]
    assert resolve_config(config, device="cuda:0,cuda:1").config.solver.device == "cuda:0"
    # A list is the devices as given; one indexed device is itself; cuda, auto and cpu leave the choice to the model.
    assert devices_argument("cuda:0,cuda:1", configured="cuda:0") == ["cuda:0", "cuda:1"]
    assert devices_argument("cuda:1", configured="cuda:1") == ["cuda:1"]
    assert devices_argument(None, configured="cuda:1") == ["cuda:1"]
    for name in ("cuda", "auto", "cpu", None):
        assert devices_argument(name, configured=name) == "auto"


def test_a_device_list_is_refused_where_one_device_computes_and_an_empty_entry_everywhere(config):
    listed = runner.invoke(
        app, ["calibrate-tf", "-i", "in.zarr", "-c", str(config), "-o", "tf.zarr", "--device", "cuda:0,cuda:1"]
    )
    assert listed.exit_code == 2
    assert "'cuda:0,cuda:1' is a list; this command computes on one device" in listed.stderr
    empty = runner.invoke(app, ["rec", "-i", "in.zarr", "-c", str(config), "-o", "out.zarr", "--device", "cuda:0,"])
    assert empty.exit_code == 2
    assert "must be a device or a comma-separated list of devices" in empty.stderr


def test_every_argument_the_library_names_in_a_message_has_a_flag():
    import waveorder
    from waveorder.cli.options import ARGUMENT_FLAGS

    root = Path(waveorder.__file__).parent
    call = re.compile(r"""\b(?:named|where)\(\s*["']([a-z_]+)["']""")  # named("tile"), where("prior", path)
    sources = [path for path in root.rglob("*.py") if path.relative_to(root).parts[0] != "cli"]
    names = {name for path in sources for name in call.findall(path.read_text())}
    assert {"positions", "output", "store", "tile", "pixel_size"} <= names
    assert names - set(ARGUMENT_FLAGS) == set()
    assert "start" in ARGUMENT_FLAGS  # the --start store's argument reaches where() through a variable


def test_every_command_the_library_names_in_a_message_exists():
    import typer
    from typer.core import TyperGroup

    import waveorder

    root = Path(waveorder.__file__).parent
    group = typer.main.get_command(app)
    assert isinstance(group, TyperGroup)
    mention = re.compile(r"\bwo ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?")  # wo rec, wo config check
    missing = set()
    for path in root.rglob("*.py"):
        for first, second in mention.findall(path.read_text()):
            command = group.commands.get(first)
            if command is None:
                missing.add(f"{path.name}: wo {first}")
            elif isinstance(command, TyperGroup) and second and second not in command.commands:
                missing.add(f"{path.name}: wo {first} {second}")
    assert missing == set()


def test_a_problem_the_build_finds_is_located_at_the_file_with_its_config_path(tmp_path):
    path = tmp_path / "calibrated.yml"
    path.write_text(PHASE + "calibrated: {system.bogus: 1.0}\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert f"{path}: calibrated: unknown parameter path(s) ['system.bogus']" in result.stderr
    assert "the configuration" not in result.stderr


def test_underscores_normalize_to_hyphens(config):
    good = runner.invoke(app, ["rec", "-i", "in.zarr", "-c", str(config), "-o", "out", "--num_processes", "1"])
    assert "matches nothing on disk" in good.stderr
    bad = runner.invoke(app, ["calibrate-tf", "-i", "in.zarr", "-c", str(config), "-o", "out", "--time_idx", "1"])
    assert bad.exit_code == 2
    assert "No such option: --time-idx" in bad.stderr


def test_input_positions_are_greedy_and_repeatable():
    import typer

    from waveorder.cli.options import InputPositions, Tile, WoCommand

    probe = typer.Typer()

    @probe.command(cls=WoCommand)
    def show(*, input_position_dirpaths: InputPositions, tile: Tile = None) -> None:
        typer.echo(" ".join(path.name for path in input_position_dirpaths) + f" | {tile}")

    result = runner.invoke(probe, ["-i", "a", "b", "--tile", "1,2,3", "-i", "c"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "a b c | 1,2,3"


def test_num_processes_limit(config):
    too_many = str((os.cpu_count() or 1) + 1)
    result = runner.invoke(app, ["rec", "-i", "in.zarr", "-c", str(config), "-o", "out", "-j", too_many])
    assert result.exit_code == 2
    assert "CPUs of this machine" in result.stderr
    none = runner.invoke(app, ["rec", "-i", "in.zarr", "-c", str(config), "-o", "out", "-j", "0"])
    assert none.exit_code == 2
    assert "at least 1" in none.stderr


def test_an_error_at_a_flag_does_not_repeat_the_flag(capsys):
    from waveorder.cli.render import Failure, render_failure
    from waveorder.errors import Problem

    problems = (
        Problem("--tile cannot split these volumes", "--tile"),
        Problem("--tile-overlap widens --tile cores", "--tile"),
        Problem("weight: must be positive", "phase.yml"),
    )
    render_failure(Failure(2, problems), command="wo rec")
    assert capsys.readouterr().err.splitlines() == [
        "wo rec: error: --tile: cannot split these volumes",
        "wo rec: error: --tile: --tile-overlap widens --tile cores",
        "wo rec: error: phase.yml: weight: must be positive",
    ]


def test_a_bug_prints_its_stack_and_a_waveorder_error_one_line(config, monkeypatch):
    from waveorder.cli import _configured as commands
    from waveorder.errors import ShapeError

    def broken(*_: object, **__: object) -> None:
        raise KeyError("lost")

    monkeypatch.setattr(commands, "build_model", broken)
    bug = runner.invoke(app, ["config", "check", str(config)])
    assert bug.exit_code == 1
    assert "Traceback (most recent call last)" in bug.stderr
    assert "wo config check: error: KeyError: 'lost'" in bug.stderr
    assert "this is a bug in waveorder" in bug.stderr

    def refused(*_: object, **__: object) -> None:
        raise ShapeError("too small")

    monkeypatch.setattr(commands, "build_model", refused)
    known = runner.invoke(app, ["config", "check", str(config)])
    assert known.exit_code == 1
    assert known.stderr.splitlines() == ["wo config check: error: ShapeError: too small"]


def test_input_files_are_skipped(config, tmp_path):
    (tmp_path / "zarr.json").write_text("{}")
    result = runner.invoke(
        app, ["rec", "-i", str(tmp_path / "zarr.json"), "-i", "in.zarr", "-c", str(config), "-o", "out"]
    )
    assert result.exit_code == 2
    assert "in.zarr matches nothing on disk" in result.stderr
    assert "zarr.json" not in result.stderr


def test_wandb_needs_the_package(config, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "wandb", None)
    result = runner.invoke(app, ["rec", "-i", "in.zarr", "-c", str(config), "-o", "out.zarr", "--wandb", "p"])
    assert result.exit_code == 2
    assert "--wandb: needs the wandb package: pip install waveorder[wandb]" in result.stderr
