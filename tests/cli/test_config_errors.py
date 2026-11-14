"""``wo config check | show``: the data-free checks, located errors and warnings, output files, and ``show -t``."""

import pytest
import torch
import yaml
from typer.testing import CliRunner

from tests.cli.plates import FIELD_TILED, NA, PHASE, edited
from waveorder.cli import app
from waveorder.cli.config import template
from waveorder.config import load
from waveorder.io import TfStore
from waveorder.models import ReconstructionModel

runner = CliRunner()


def _file(tmp_path, text, name="phase.yml"):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def config(tmp_path):
    return _file(tmp_path, template("phase", 3))


def _errors(result, command: str) -> list[str]:
    """The error lines of ``wo config <command>``, without their prefix."""
    prefix = f"wo config {command}: error: "
    return [line.removeprefix(prefix) for line in result.stderr.splitlines() if line.startswith(prefix)]


# --- check means runnable -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dimension", "setting", "key"),
    [
        (2, "solver.padding.z=4", "solver.padding.z"),
        (3, "solver.device=gpu0", "solver.device"),
    ],
)
@pytest.mark.parametrize("command", ["check", "show"])
def test_check_and_show_refuse_what_the_build_refuses(tmp_path, command, dimension, setting, key):
    path = edited(_file(tmp_path, template("phase", dimension)), setting)
    result = runner.invoke(app, ["config", command, str(path)])
    assert result.exit_code == 2, result.output
    (error,) = _errors(result, command)
    assert error.startswith(f"{path}: {key}: ")


def test_a_refused_value_in_the_file_is_located_at_the_file_and_its_key(tmp_path, config):
    text = config.read_text().replace("apodization_rolloff: 0.0", "apodization_rolloff: 1.5")
    path = _file(tmp_path, text, "rolloff.yml")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert f"wo config check: error: {path}: solver.apodization_rolloff" in result.stderr


def test_an_absent_device_is_a_warning_at_its_key(config):
    path = edited(config, "solver.device=cuda:99")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"ok: {path}\n"
    assert "wo config check: warning: solver.device: " in result.stderr


def test_check_reports_a_missing_bundle_at_its_key(tmp_path, config):
    missing = tmp_path / "absent"
    text = config.read_text().replace(
        "reconstruction: closed_form", f"reconstruction: {{distilled: {{bundle: {missing}}}}}"
    )
    path = _file(tmp_path, text, "distilled.yml")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2, result.output
    (error,) = _errors(result, "check")
    assert error.startswith(f"{path}: solver.reconstruction.distilled.bundle: {missing} is not a directory")


# --- warnings -------------------------------------------------------------------------------------------------------

HUGE_GRID = PHASE.replace("{grid: {points: 17}}", "{grid: {points: 20001}}")


def test_a_stage_warning_is_located_at_its_stage_once(tmp_path):
    path = _file(tmp_path, HUGE_GRID)
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.stderr.splitlines() if "warning:" in line]
    assert len(lines) == 1, result.stderr
    assert lines[0].startswith("wo config check: warning: solver.calibration.search.0: ")
    assert "evaluates 20001 points" in lines[0]


def test_an_interrupt_exits_130_with_one_line(config, monkeypatch):
    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("waveorder.cli._configured.build_model", interrupted)
    result = runner.invoke(app, ["config", "check", str(config)])
    assert result.exit_code == 130
    assert result.stdout == ""
    assert result.stderr == "wo config check: interrupted\n"


# --- located errors -------------------------------------------------------------------------------------------------

BASE = """\
version: 4
data: {input_channel_names: [BF], pixel_size: {z: 0.25, y: 0.1, x: 0.1}}
imaging_system:
  contrast: phase
  illumination: {wavelength: 0.532, numerical_aperture: 0.9}
"""


@pytest.mark.parametrize("key", ["my-key", "My Key", "MyKey", "float"])
def test_an_unknown_key_keeps_its_name(tmp_path, key):
    path = _file(tmp_path, BASE + f"  detection: {{numerical_aperture: 1.2, {key}: 1}}\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    (error,) = _errors(result, "check")
    assert error.startswith(f"{path}: imaging_system.detection.{key}: is not a configuration key")


def test_an_unknown_key_in_the_file_suggests_the_known_one(tmp_path):
    path = _file(tmp_path, BASE + "  detection: {numerical_apertur: 1.2}\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert "did you mean imaging_system.detection.numerical_aperture?" in result.stderr


def test_a_union_reports_the_error_that_explains_the_input(tmp_path):
    path = _file(tmp_path, BASE + "object: {weight: -1}\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    (error,) = _errors(result, "check")
    assert error.startswith(f"{path}: object.weight: ")
    assert "auto" not in error


def test_value_errors_carry_no_pydantic_prefix(tmp_path):
    path = _file(tmp_path, BASE + "  detection: {pinhole: {diameter_au: 1, diameter_um: 1}}\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert "exactly one of 'diameter_au' or 'diameter_um'" in result.stderr
    assert "Value error" not in result.stderr


def test_a_nonpositive_detection_aperture_is_refused_at_its_key(tmp_path):
    path = _file(tmp_path, BASE + "  detection:\n    numerical_aperture: -0.5\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    expected = f"{path}: imaging_system.detection.numerical_aperture: Detection.numerical_aperture must be positive"
    assert expected in result.stderr


def test_a_duplicate_key_is_refused_at_its_line(tmp_path):
    path = _file(tmp_path, BASE + "object:\n  weight: 1e-3\n  prior: tikhonov\n  weight: 1e-2\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert f"{path}:9:3: while reading a mapping the key 'weight' is given twice" in result.stderr


def test_an_empty_file_says_it_is_empty(tmp_path):
    path = _file(tmp_path, "# nothing yet\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert (
        result.stderr
        == f"wo config check: error: {path}: is empty; print a configuration to start from with: wo config new\n"
    )


def test_a_root_that_is_not_a_mapping_names_the_file_once(tmp_path):
    path = _file(tmp_path, "- a\n- b\n")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 2
    assert result.stderr == f"wo config check: error: {path}: must hold a YAML mapping, got list\n"


# --- output files ---------------------------------------------------------------------------------------------------


def test_an_existing_output_needs_overwrite(config, tmp_path):
    output = _file(tmp_path, "keep\n", "out.txt")
    refused = runner.invoke(app, ["config", "show", str(config), "-o", str(output)])
    assert refused.exit_code == 2
    assert f"-o {output}: exists; pass --overwrite" in refused.stderr
    assert output.read_text() == "keep\n"
    replaced = runner.invoke(app, ["config", "show", str(config), "-o", str(output), "--overwrite"])
    assert replaced.exit_code == 0, replaced.output
    assert load(yaml.safe_load(output.read_text())).imaging_system.contrast == "phase"


def test_show_never_writes_over_its_input(config):
    original = config.read_text()
    result = runner.invoke(app, ["config", "show", str(config), "-o", str(config), "--overwrite"])
    assert result.exit_code == 2
    assert config.read_text() == original


def test_an_output_in_a_missing_directory_is_refused_with_its_path(config, tmp_path):
    output = tmp_path / "missing" / "out.yml"
    result = runner.invoke(app, ["config", "show", str(config), "-o", str(output)])
    assert result.exit_code == 2
    assert f"-o {output}: directory {output.parent} does not exist" in result.stderr


# --- show -t --------------------------------------------------------------------------------------------------------


def _store(tmp_path, *, rows=1, names=(NA,), sha="match"):
    """Write a store of 0.52 for ``names``; ``sha`` is the imaging system it records: PHASE's, another, or none."""
    made = ReconstructionModel.from_config(load(yaml.safe_load(PHASE))).to_tf_store()
    provenance: dict[str, object] = {"kind": "calibrated"}
    if sha is not None:
        provenance["imaging_system_sha256"] = made.provenance["imaging_system_sha256"] if sha == "match" else "0" * 64
    store = TfStore(
        parameter_names=names,
        parameters=torch.full((rows, len(names)), 0.52, dtype=torch.float64),
        tile_grid={
            "size_yx": [32, 32],
            "overlap_yx": [0, 0],
            "shape_zyx": [8, 32, 32],
            "blend": "linear",
            "units": "um",
        },
        provenance=provenance,
    )
    return store.save(tmp_path / "tf.zarr")


def test_show_with_a_store_writes_the_stored_values_in_the_calibrated_block(tmp_path):
    path = edited(_file(tmp_path, PHASE), "solver.device=cuda:99")
    store = _store(tmp_path)
    result = runner.invoke(app, ["config", "show", str(path), "-t", str(store)])
    assert result.exit_code == 0, result.output
    resolved = yaml.safe_load(result.stdout)
    assert resolved["calibrated"] == {NA: pytest.approx(0.52)}
    aperture = resolved["imaging_system"]["illumination"]["numerical_aperture"]
    assert aperture == {"nominal": 0.45, "tolerance": "free", "bounds": [0.3, 0.7]}  # the prior is kept as written
    assert load(resolved).calibrated == {NA: pytest.approx(0.52)}
    assert resolved["solver"]["device"] == "cuda:99"  # the model is built on the CPU; the configured device is shown


@pytest.mark.parametrize(
    ("store", "message"),
    [
        ({"rows": 2}, "holds 2 field tiles, but the model is shift-invariant"),
        ({"sha": "other"}, "was made for another imaging_system"),
        ({"sha": None}, "records no imaging_system_sha256"),
        ({"names": (NA, "system.detection.numerical_aperture")}, "which the model fixes in its imaging system"),
    ],
)
def test_show_refuses_a_store_that_does_not_fit(tmp_path, store, message):
    path = _file(tmp_path, PHASE)
    written = _store(tmp_path, **store)
    result = runner.invoke(app, ["config", "show", str(path), "-t", str(written)])
    assert result.exit_code == 2
    assert f"wo config show: error: -t {written}: " in result.stderr
    assert message in result.stderr


def test_show_refuses_the_per_tile_values_of_a_field_tiled_calibration(tmp_path, phase_plate):
    path = _file(tmp_path, FIELD_TILED, "tiled.yml")
    store = tmp_path / "tiled.zarr"
    computed = runner.invoke(app, ["compute-tf", "-i", str(phase_plate), "-c", str(path), "-o", str(store)])
    assert computed.exit_code == 0, computed.output
    result = runner.invoke(app, ["config", "show", str(path), "-t", str(store)])
    assert result.exit_code == 2
    expected = f"-t {store}: holds 4 tiles of a field-tiled calibration; per-tile values have no single config"
    assert result.stderr == f"wo config show: error: {expected}\n"


def test_show_keeps_the_configured_value_of_a_stored_quantity_it_fixes(tmp_path):
    path = _file(tmp_path, PHASE)
    store = _store(tmp_path, names=(NA, "object_prior.weight"))
    result = runner.invoke(app, ["config", "show", str(path), "-t", str(store)])
    assert result.exit_code == 0, result.output
    warning = f"-t {store}: holds object_prior.weight, which the model fixes; the model's values are used"
    assert f"wo config show: warning: {warning}" in result.stderr.splitlines()
    resolved = yaml.safe_load(result.stdout)
    assert resolved["object"]["weight"] == 4e-4
    assert resolved["calibrated"] == {NA: pytest.approx(0.52)}


def test_show_refuses_a_store_that_is_not_one(tmp_path):
    path = _file(tmp_path, PHASE)
    result = runner.invoke(app, ["config", "show", str(path), "-t", str(tmp_path / "nothing.zarr")])
    assert result.exit_code == 2
    assert f"-t {tmp_path / 'nothing.zarr'}: " in result.stderr
