"""``wo config new | check | show``: templates, located errors and output files."""

import pytest
import yaml
from typer.testing import CliRunner

from tests.cli.plates import edited
from waveorder.cli import app
from waveorder.cli.config import template
from waveorder.config import load

runner = CliRunner()

TEMPLATES = [(contrast, dimension) for contrast in ("phase", "fluorescence") for dimension in (2, 3)]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "phase.yml"
    path.write_text(template("phase", 3))
    return path


@pytest.mark.parametrize(("contrast", "dimension"), TEMPLATES)
def test_every_template_validates(tmp_path, contrast, dimension):
    result = runner.invoke(app, ["config", "new", "--contrast", contrast, "--dimension", str(dimension)])
    assert result.exit_code == 0, result.output
    path = tmp_path / "new.yml"
    path.write_text(result.stdout)
    config = load(yaml.safe_load(result.stdout))
    assert config.imaging_system.contrast == contrast
    assert config.object.dimension == dimension
    checked = runner.invoke(app, ["config", "check", str(path)])
    assert checked.exit_code == 0, checked.output
    assert checked.stdout == f"ok: {path}\n"


def test_new_writes_a_file_once(tmp_path):
    path = tmp_path / "phase.yml"
    first = runner.invoke(app, ["config", "new", "-o", str(path)])
    assert first.exit_code == 0
    assert first.stdout == ""
    assert f"wrote {path}" in first.stderr
    again = runner.invoke(app, ["config", "new", "--contrast", "fluorescence", "-o", str(path)])
    assert again.exit_code == 2
    assert "pass --overwrite" in again.stderr
    replaced = runner.invoke(app, ["config", "new", "--contrast", "fluorescence", "-o", str(path), "--overwrite"])
    assert replaced.exit_code == 0
    assert "contrast: fluorescence" in path.read_text()


def test_show_fills_in_defaults(config):
    result = runner.invoke(app, ["config", "show", str(config)])
    assert result.exit_code == 0, result.output
    resolved = yaml.safe_load(result.stdout)
    assert "calibration" not in yaml.safe_load(config.read_text())["solver"]
    assert resolved["solver"]["calibration"] is None
    assert load(resolved).model_dump() == load(yaml.safe_load(config.read_text())).model_dump()


@pytest.mark.parametrize(("contrast", "dimension"), TEMPLATES)
def test_show_leaves_out_unchosen_nulls_and_reads_back_the_same(tmp_path, contrast, dimension):
    path = tmp_path / "c.yml"
    path.write_text(template(contrast, dimension))
    result = runner.invoke(app, ["config", "show", str(path)])
    assert result.exit_code == 0, result.output
    resolved = yaml.safe_load(result.stdout)
    assert "simulate" not in resolved
    assert all(value is not None for value in resolved["noise"].values())  # only the chosen noise model
    assert resolved["solver"]["device"] is None  # a null that means something: the CPU
    assert "calibration" in resolved["solver"]
    assert load(resolved).model_dump() == load(yaml.safe_load(path.read_text())).model_dump()


def test_a_refused_value_is_reported_at_the_file_and_its_key(config):
    path = edited(config, "object.weight=-1")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert f"wo config check: error: {path}: object.weight:" in result.stderr


def test_an_unknown_key_suggests(config):
    path = edited(config, "solver.reconstrucion=fista")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert "did you mean solver.reconstruction?" in result.stderr


def test_3x_file_names_the_migrate_command(tmp_path):
    path = tmp_path / "old.yml"
    path.write_text("input_channel_names: [BF]\nreconstruction_dimension: 3\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert f"wo config migrate {path}" in result.stderr
    assert f"wo config check: error: {path}: has no version" in result.stderr


def test_an_aliased_value_is_refused_at_its_dotted_path(tmp_path):
    path = tmp_path / "alias.yml"
    path.write_text(
        "version: 4\n"
        "data: {input_channel_names: [BF], pixel_size: {z: 0.25, y: 0.1, x: 0.1}}\n"
        "imaging_system:\n"
        "  contrast: phase\n"
        "  detection: &arm {wavelength: 0.532, numerical_aperture: 1.6}\n"
        "  illumination: *arm\n"
    )
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    expected = f"{path}: imaging_system.illumination.numerical_aperture: illumination.numerical_aperture can reach 1.6"
    assert expected in result.stderr


def test_check_without_file_shows_help():
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 2
    assert "Usage" in result.output
