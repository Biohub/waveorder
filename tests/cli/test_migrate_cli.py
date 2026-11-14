"""``wo config migrate`` and its hidden spellings ``migrate-config`` and ``mc``: output paths, exit codes, help."""

import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from waveorder.cli import app
from waveorder.config import load

runner = CliRunner()

EXAMPLES = Path(__file__).parents[1] / "config" / "data" / "3x"
PREVIEW = Path(__file__).parents[1] / "config" / "data" / "592" / "phase_3d_optimized.yml"
SPELLINGS = [["config", "migrate"], ["migrate-config"], ["mc"]]


@pytest.fixture
def phase_3x(tmp_path):
    return Path(shutil.copy(EXAMPLES / "phase_3d.yml", tmp_path / "phase.yml"))


@pytest.mark.parametrize("command", SPELLINGS)
def test_default_output_is_beside_the_input(phase_3x, command):
    result = runner.invoke(app, [*command, str(phase_3x)])
    assert result.exit_code == 0, result.output
    target = phase_3x.with_name("phase_v4.yml")
    assert result.stdout == ""
    assert f"wrote {target}" in result.stderr
    text = target.read_text()
    assert text.startswith("# WaveOrder 4 configuration migrated by wo config migrate from waveorder 3.x: phase.yml")
    assert load(yaml.safe_load(text)).imaging_system.contrast == "phase"
    notes = [line.removeprefix("note: ") for line in result.stderr.splitlines() if line.startswith("note: ")]
    assert notes  # what changed in meaning is said on stderr too, not only in the file's header
    header = " ".join(line.removeprefix("# - ").removeprefix("#   ") for line in text.splitlines() if line[:1] == "#")
    assert all(note in header for note in notes)


def test_dash_prints_to_stdout(phase_3x):
    result = runner.invoke(app, ["config", "migrate", str(phase_3x), "-o", "-"])
    assert result.exit_code == 0, result.output
    assert "edge: sigmoid" in result.stdout
    assert "wrote" not in result.stderr
    assert not phase_3x.with_name("phase_v4.yml").exists()


def test_existing_output_needs_overwrite(phase_3x, tmp_path):
    target = tmp_path / "out.yml"
    target.write_text("keep\n")
    refused = runner.invoke(app, ["config", "migrate", str(phase_3x), "-o", str(target)])
    assert refused.exit_code == 2
    assert "pass --overwrite" in refused.stderr
    assert target.read_text() == "keep\n"
    replaced = runner.invoke(app, ["config", "migrate", str(phase_3x), "-o", str(target), "--overwrite"])
    assert replaced.exit_code == 0, replaced.output
    assert "version: 4" in target.read_text()


def test_already_v4_exits_zero_without_writing(phase_3x):
    runner.invoke(app, ["config", "migrate", str(phase_3x)])
    current = phase_3x.with_name("phase_v4.yml")
    result = runner.invoke(app, ["mc", str(current)])
    assert result.exit_code == 0, result.output
    assert "is already a WaveOrder 4 configuration" in result.stderr
    assert not current.with_name("phase_v4_v4.yml").exists()


def test_preview_file_is_upgraded(tmp_path):
    source = Path(shutil.copy(PREVIEW, tmp_path / "old.yml"))
    result = runner.invoke(app, ["config", "migrate", str(source)])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "old_v4.yml").read_text()
    migrated = yaml.safe_load(text)
    assert "object" in migrated
    assert "yx" not in migrated["data"]["pixel_size"]
    load(migrated)


@pytest.mark.parametrize("name", ["phase_2d", "phase_3d", "fluorescence_2d", "fluorescence_3d"])
def test_a_migrated_file_passes_check(tmp_path, name):
    source = Path(shutil.copy(EXAMPLES / f"{name}.yml", tmp_path / f"{name}.yml"))
    migrated = runner.invoke(app, ["config", "migrate", str(source)])
    assert migrated.exit_code == 0, migrated.output
    checked = runner.invoke(app, ["config", "check", str(tmp_path / f"{name}_v4.yml")])
    assert checked.exit_code == 0, checked.output


def test_a_migration_that_does_not_build_is_refused_and_not_written(tmp_path):
    # 2D TV migrates to ADMM, which the thin system refuses when the model is built, not when the schema validates.
    source = tmp_path / "tv_2d.yml"
    algorithm = "reconstruction_algorithm: "
    source.write_text((EXAMPLES / "phase_2d.yml").read_text().replace(f"{algorithm}Tikhonov", f"{algorithm}TV"))
    result = runner.invoke(app, ["config", "migrate", str(source)])
    assert result.exit_code == 2
    assert f"{source} (migrated): solver.reconstruction:" in result.stderr
    assert "the configuration" not in result.stderr
    assert not (tmp_path / "tv_2d_v4.yml").exists()


def test_a_preview_file_passes_check_with_the_migrate_command_in_its_warning(tmp_path):
    # wo-4.0 wrote it; it loads as wo-4.0 read it, and the warning names the command that converts it.
    source = Path(shutil.copy(PREVIEW, tmp_path / "old.yml"))
    result = runner.invoke(app, ["config", "check", str(source)])
    assert result.exit_code == 0, result.output
    assert f"if wo-4.0 wrote it, convert it with: wo config migrate {source}" in result.stderr


@pytest.mark.parametrize("command", [["config", "check"], ["config", "show"]])
def test_a_3x_file_is_refused_with_its_path_once(phase_3x, command):
    result = runner.invoke(app, [*command, str(phase_3x)])
    assert result.exit_code == 2
    (line,) = result.stderr.splitlines()
    assert line.startswith(f"wo {' '.join(command)}: error: {phase_3x}: has no version, so it is a 3.x ")
    assert f"wo config migrate {phase_3x}" in line
    assert str(phase_3x.with_name("phase_v4.yml")) in line  # the file to pass to -c next


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("birefringence_3d", "birefringence"),
        ("tile-stitch_phase_3d", "tile"),
    ],
)
def test_rejections_exit_2_with_their_key(tmp_path, name, key):
    source = Path(shutil.copy(EXAMPLES / f"{name}.yml", tmp_path / f"{name}.yml"))
    result = runner.invoke(app, ["config", "migrate", str(source)])
    assert result.exit_code == 2
    assert f"wo config migrate: error: {source}: cannot migrate: {key}:" in result.stderr
    assert not list(tmp_path.glob("*_v4.yml"))


def test_birefringence_message(tmp_path):
    source = Path(shutil.copy(EXAMPLES / "birefringence_3d.yml", tmp_path / "b.yml"))
    result = runner.invoke(app, ["config", "migrate", str(source)])
    assert "birefringence is not in waveorder 4; use waveorder 3.x" in result.stderr


def test_tile_stitch_is_rejected_once_with_the_tile_hint(tmp_path):
    source = Path(shutil.copy(EXAMPLES / "tile-stitch_phase_3d.yml", tmp_path / "ts.yml"))
    result = runner.invoke(app, ["config", "migrate", str(source)])
    assert result.exit_code == 2
    (line,) = result.stderr.splitlines()
    assert line.startswith(f"wo config migrate: error: {source}: cannot migrate: tile: tile-stitch (tile, ")
    assert "wo rec --tile" in line
    assert "the output becomes a plate" in line


def test_yaml_error_is_exit_2(tmp_path):
    source = tmp_path / "bad.yml"
    source.write_text("phase: [\n")
    result = runner.invoke(app, ["config", "migrate", str(source)])
    assert result.exit_code == 2
    assert "bad.yml:" in result.stderr


def test_missing_input_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["config", "migrate", str(tmp_path / "nope.yml")])
    assert result.exit_code == 2


def test_no_arguments_print_help():
    result = runner.invoke(app, ["config", "migrate"])
    assert result.exit_code == 2
    assert "Usage" in result.output


@pytest.mark.parametrize("command", SPELLINGS)
def test_json_is_not_offered(phase_3x, command):
    result = runner.invoke(app, [*command, str(phase_3x), "--json"])
    assert result.exit_code == 2
    assert "--json" in result.output


@pytest.mark.parametrize("command", SPELLINGS)
def test_help_ends_with_the_table(command):
    result = runner.invoke(app, [*command, "--help"], terminal_width=200)
    assert result.exit_code == 0
    assert "3.x key (3.x default): what it becomes" in result.stdout
    assert "optimization.convergence_patience (5)" in result.stdout
    assert "--overwrite" in result.stdout


def test_help_table_rows_wrap_under_a_hanging_indent(monkeypatch):
    from waveorder.config.migrate import TABLE

    monkeypatch.setattr("typer.rich_utils.MAX_WIDTH", 80)
    result = runner.invoke(app, ["config", "migrate", "--help"], env={"COLUMNS": "80"})
    assert result.exit_code == 0
    table = result.stdout.split("3.x key (3.x default): what it becomes")[1].strip("\n").splitlines()
    rows = [line for line in table if line.strip()]
    starts = [line for line in rows if not line.startswith("  ")]
    assert len(starts) == len(TABLE)  # every wrapped continuation is indented, so each key starts one line
    assert all(line.startswith(f" {row.key}") for line, row in zip(starts, TABLE, strict=True))
    assert max(len(line.rstrip()) for line in rows) <= 80
