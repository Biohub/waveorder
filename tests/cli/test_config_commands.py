"""``wo config new | check | show``: templates, located errors and output files."""

import shutil
from pathlib import Path

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


REPOSITORY = Path(__file__).parents[2]
WO40 = REPOSITORY / "tests" / "config" / "data" / "wo40"


def _shown_then_checked(path: Path) -> None:
    shown = path.with_name(f"{path.stem}.shown.yml")
    result = runner.invoke(app, ["config", "show", str(path), "-o", str(shown)])
    assert result.exit_code == 0, result.output
    checked = runner.invoke(app, ["config", "check", str(shown)])
    assert checked.exit_code == 0, checked.output


@pytest.mark.parametrize("name", sorted(path.stem for path in (REPOSITORY / "benchmarks" / "configs").glob("*.yml")))
def test_show_of_every_migrated_benchmark_config_passes_check(tmp_path, name):
    source = REPOSITORY / "benchmarks" / "configs" / f"{name}.yml"
    migrated = tmp_path / f"{name}.yml"
    result = runner.invoke(app, ["config", "migrate", str(source), "-o", str(migrated)])
    assert result.exit_code == 0, result.output
    _shown_then_checked(migrated)


@pytest.mark.parametrize(("contrast", "dimension"), TEMPLATES)
def test_show_of_every_template_passes_check(tmp_path, contrast, dimension):
    path = tmp_path / f"{contrast}_{dimension}d.yml"
    path.write_text(template(contrast, dimension))
    _shown_then_checked(path)


@pytest.mark.parametrize("name", sorted(path.stem for path in WO40.glob("*.yml")))
def test_wo40_written_files_pass_check_and_their_show_passes_check(tmp_path, name):
    # wo-4.0 writes solver.reconstruction: closed_form in every file, which a bead fit or a simulation never solves.
    path = Path(shutil.copy(WO40 / f"{name}.yml", tmp_path / f"{name}.yml"))
    checked = runner.invoke(app, ["config", "check", str(path)])
    assert checked.exit_code == 0, checked.output
    _shown_then_checked(path)


def test_show_leaves_out_the_parts_the_model_chooses_so_its_output_builds_the_same_model(tmp_path):
    from waveorder.errors import ConfigurationWarning
    from waveorder.models import ReconstructionModel

    path = Path(shutil.copy(WO40 / "sim_poisson_short.yml", tmp_path / "sim.yml"))
    result = runner.invoke(app, ["config", "show", str(path)])
    assert result.exit_code == 0, result.output
    shown = yaml.safe_load(result.stdout)
    assert "reconstruction" not in shown["solver"]
    assert {"prior", "weight", "constraints"}.isdisjoint(shown["object"])
    assert shown["noise"] == {"poisson": {"gain": 1.0, "offset": 0.0, "photons": 50.0}}
    parts = ("object_prior", "noise", "solver")
    with pytest.warns(ConfigurationWarning, match=r"written \{z, yx\}"):  # the source is a wo-4.0 file
        model = ReconstructionModel.from_config(path)
    again = ReconstructionModel.from_config(shown)
    assert [type(getattr(model, name)) for name in parts] == [type(getattr(again, name)) for name in parts]


def test_check_accepts_the_denoiser_prior_with_a_note_and_rec_refuses_it(tmp_path, phase_plate):
    path = tmp_path / "denoiser.yml"
    path.write_text(template("phase", 3).replace("prior: tikhonov", "prior: {denoiser: {checkpoint: x.pt}}"))
    checked = runner.invoke(app, ["config", "check", str(path)])
    assert checked.exit_code == 0, checked.output
    assert "object.prior.denoiser: the denoiser prior is not in this version" in checked.stderr
    output = tmp_path / "o.zarr"
    run = runner.invoke(app, ["rec", "-i", str(phase_plate / "A/1/0"), "-c", str(path), "-o", str(output)])
    assert run.exit_code == 2
    assert "object.prior.denoiser: the denoiser prior is not in this version" in run.stderr
    assert not output.exists()


def test_device_keeps_the_file_in_the_wo40_pixel_size_warning(tmp_path):
    from waveorder.cli._configured import resolve_config
    from waveorder.errors import ConfigurationWarning

    path = Path(shutil.copy(WO40 / "bead_short.yml", tmp_path / "beads.yml"))
    with pytest.warns(ConfigurationWarning) as caught:
        resolve_config(path, device="cpu")
    found = [w.message for w in caught if isinstance(w.message, ConfigurationWarning)]
    (warning,) = [message for message in found if "{z, yx}" in str(message)]
    assert warning.location == str(path)
    assert f"wo config migrate {path}" in str(warning)
