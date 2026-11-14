"""``wo calibrate-tf`` end to end on a tiny CPU plate, with stores ``wo compute-tf`` writes."""

import json

import pytest
from typer.testing import CliRunner

from tests.cli.plates import (
    FIELD_TILED,
    FIXED_PHASE,
    FLUORESCENCE,
    NA,
    PHASE,
    SHAPE,
    TRUE_NA,
    edited,
    intensities,
    simulate_phase,
    write_plate,
)
from waveorder.cli import app
from waveorder.io import TfStore

runner = CliRunner()

SHORT_SEARCH = "solver.calibration.search=[{grid: {points: 5}}]"


def _calibrate(plate, config, output, *extra: str):
    return runner.invoke(
        app, ["calibrate-tf", "-i", str(plate / "A" / "1" / "0"), "-c", str(config), "-o", str(output), *extra]
    )


def _compute(plate, config, output, *extra: str):
    return runner.invoke(app, ["compute-tf", "-i", str(plate), "-c", str(config), "-o", str(output), *extra])


def test_calibrate_tf_recovers_the_aperture_and_never_reuses_a_store(phase_plate, phase_config, tmp_path):
    output = tmp_path / "tf.zarr"
    result = _calibrate(phase_plate, phase_config, output)
    assert result.exit_code == 0, result.output
    assert f"wrote {output}" in result.stderr
    assert " calibrating on A/1/0 at t=0 ..." in result.stderr
    assert result.stdout.startswith(f"calibrated on A/1/0 at t=0: {output}\n")
    assert NA in result.stdout
    assert "at a bound: none" in result.stdout
    store = TfStore.load(output)
    assert store.row()[NA] == pytest.approx(TRUE_NA, abs=0.04)
    assert store.provenance["position_name"] == "A/1/0"
    assert store.transfers == ()  # a calibration store holds parameters only, whatever outputs asks for

    existing = _calibrate(phase_plate, phase_config, output)
    assert existing.exit_code == 2
    assert existing.stderr.splitlines() == [
        f"wo calibrate-tf: error: -o {output}: exists; pass --overwrite to replace it"
    ]
    again = _calibrate(phase_plate, edited(phase_config, SHORT_SEARCH), output, "--overwrite")
    assert again.exit_code == 0, again.output
    assert TfStore.load(output).row()[NA] == pytest.approx(TRUE_NA, abs=0.04)


def test_calibrate_tf_checks_the_prior_and_start_stores_against_the_model(phase_plate, phase_config, tmp_path):
    fluorescence = tmp_path / "fluor.yml"
    fluorescence.write_text(FLUORESCENCE)
    other = tmp_path / "other.zarr"
    computed = _compute(phase_plate, fluorescence, other)
    assert computed.exit_code == 0, computed.output
    for flag in ("--prior", "-t", "--start"):
        unfit = _calibrate(phase_plate, phase_config, tmp_path / "x.zarr", flag, str(other))
        assert unfit.exit_code == 2
        shown = "--start" if flag == "--start" else "--prior"
        assert f"wo calibrate-tf: error: {shown} {other}: " in unfit.stderr
        assert "none of which is free in the model" in unfit.stderr
        assert not (tmp_path / "x.zarr").exists()


def test_calibrate_tf_calibrates_on_the_first_position_and_never_writes_over_an_input(
    phase_volume, phase_config, tmp_path
):
    plate = tmp_path / "two.zarr"
    volumes = [intensities(phase_volume), intensities(simulate_phase(aperture=0.65, seed=1))]
    write_plate(plate, [volume[None] for volume in volumes])
    two = plate / "A" / "1"
    output = tmp_path / "tf.zarr"
    command = ["calibrate-tf", "-i", str(two / "0"), str(two / "1"), "-c", str(phase_config), "-o", str(output)]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert "calibrated on A/1/0 at t=0" in result.stdout
    assert TfStore.load(output).row()[NA] == pytest.approx(TRUE_NA, abs=0.04)  # A/1/1 was imaged at 0.65

    # --overwrite never replaces an input plate, nor anything that is not a transfer-function store.
    onto = runner.invoke(
        app, ["calibrate-tf", "-i", str(plate), "-c", str(phase_config), "-o", str(plate), "--overwrite"]
    )
    assert onto.exit_code == 2
    assert f"-o {plate}: contains the input position {two / '0'}; use another -o" in onto.stderr
    (tmp_path / "notes").mkdir()
    foreign = runner.invoke(
        app, ["calibrate-tf", "-i", str(plate), "-c", str(phase_config), "-o", str(tmp_path / "notes"), "--overwrite"]
    )
    assert foreign.exit_code == 2
    assert "is not a transfer-function store; use another -o" in foreign.stderr


def test_calibrate_tf_starts_from_a_start_store_and_locates_a_bad_prior_at_its_flag(
    phase_plate, phase_config, tmp_path
):
    start = tmp_path / "start.zarr"
    computed = _compute(phase_plate, phase_config, start)
    assert computed.exit_code == 0, computed.output
    output = tmp_path / "tf.zarr"
    result = _calibrate(phase_plate, edited(phase_config, SHORT_SEARCH), output, "--start", str(start))
    assert result.exit_code == 0, result.output
    assert TfStore.load(output).provenance["start"] == str(start)
    missing = tmp_path / "missing.zarr"
    refused = _calibrate(phase_plate, phase_config, tmp_path / "x.zarr", "--prior", str(missing))
    assert refused.exit_code == 2
    assert f"wo calibrate-tf: error: --prior {missing}: does not exist" in refused.stderr.splitlines()
    assert not (tmp_path / "x.zarr").exists()


def test_calibrate_tf_and_compute_tf_report_every_refused_input_together(phase_plate, phase_config, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    prior, start, missing = (tmp_path / name for name in ("p.zarr", "s.zarr", "none.zarr"))
    wrong = ["-o", str(notes), "--prior", str(prior), "--start", str(start)]
    result = runner.invoke(app, ["calibrate-tf", "-i", str(phase_plate), "-c", str(phase_config), *wrong])
    assert result.exit_code == 2
    assert result.stderr.splitlines() == [
        f"wo calibrate-tf: error: -o {notes}: exists; pass --overwrite to replace it",
        f"wo calibrate-tf: error: --prior {prior}: does not exist",
        f"wo calibrate-tf: error: --start {start}: does not exist",
    ]
    computed = _compute(missing, phase_config, notes)
    assert computed.exit_code == 2
    assert computed.stderr.splitlines() == [
        f"wo compute-tf: error: -i {missing}: {missing} matches nothing on disk",
        f"wo compute-tf: error: -o {notes}: exists; pass --overwrite to replace it",
    ]


def test_compute_tf_output_without_free_quantities_or_with_field_tiles(phase_plate, phase_config, tmp_path):
    config = edited(phase_config, "imaging_system.illumination.numerical_aperture=0.5")
    fixed = _compute(phase_plate, config, tmp_path / "a.zarr")
    assert fixed.exit_code == 0, fixed.output
    assert "no free quantities" in fixed.stdout
    assert TfStore.load(tmp_path / "a.zarr").provenance["kind"] == "nominal"
    tiled_config = tmp_path / "tiled.yml"
    tiled_config.write_text(FIELD_TILED)
    tiled = _compute(phase_plate, tiled_config, tmp_path / "t.zarr")
    assert tiled.exit_code == 0, tiled.output
    assert "4 field tiles" in tiled.stdout
    assert TfStore.load(tmp_path / "t.zarr").tiles == 4
    no_transfer = edited(phase_config, "outputs=[reconstruction]")
    without = _compute(phase_plate, no_transfer, tmp_path / "n.zarr")
    assert without.exit_code == 0, without.output
    assert TfStore.load(tmp_path / "n.zarr").transfers == ()  # outputs decides whether transfers are stored


def test_calibrate_tf_refuses_inputs_matching_nothing(phase_config, tmp_path):
    missing = runner.invoke(
        app, ["calibrate-tf", "-i", str(tmp_path / "none.zarr"), "-c", str(phase_config), "-o", str(tmp_path / "t")]
    )
    assert missing.exit_code == 2
    assert "matches nothing on disk" in missing.stderr


def test_calibrate_tf_refuses_a_configuration_with_nothing_free(phase_plate, phase_config, tmp_path):
    fixed = edited(phase_config, "imaging_system.illumination.numerical_aperture=0.5")
    result = _calibrate(phase_plate, fixed, tmp_path / "tf.zarr")
    assert result.exit_code == 2
    assert "wo calibrate-tf: error: nothing to calibrate: " in result.stderr
    assert not (tmp_path / "tf.zarr").exists()


def test_calibrate_tf_locates_a_configuration_error_at_the_file_and_its_key(phase_plate, phase_config, tmp_path):
    config = edited(phase_config, "object.weight=-1")
    result = _calibrate(phase_plate, config, tmp_path / "tf.zarr")
    assert result.exit_code == 2
    lines = result.stderr.splitlines()
    assert lines
    assert all(line.startswith(f"wo calibrate-tf: error: {config}: object.weight") for line in lines)


def test_free_quantities_without_solver_calibration_calibrate_with_the_defaults(phase_plate, tmp_path):
    config = tmp_path / "default.yml"
    config.write_text(PHASE.split("solver:")[0])  # a free aperture and no solver.calibration, as in Python
    output = tmp_path / "tf.zarr"
    result = _calibrate(phase_plate, config, output)
    assert result.exit_code == 0, result.output
    assert 0.3 <= TfStore.load(output).row()[NA] <= 0.7


def test_calibrate_tf_stores_the_score_trace_and_grid_and_config_show_names_them(phase_plate, phase_config, tmp_path):
    config = tmp_path / "scores.yml"
    config.write_text(PHASE + "outputs: [reconstruction, score_trace, score_grid]\n")
    output = tmp_path / "tf.zarr"
    result = _calibrate(phase_plate, config, output)
    assert result.exit_code == 0, result.output
    store = TfStore.load(output)
    assert store.transfers == ()  # outputs names no transfer function
    assert store.scores["score_trace"].shape == (31,)  # the 30 gradient steps and the evaluation after the last
    assert store.scores["score_grid"].shape == (17, 2)  # the 17 grid points: the aperture, then the objective
    assert store.provenance["score_grid_columns"] == [NA, "objective"]
    assert store.scores["score_grid"][:, 1].min() >= store.scores["score_trace"].min()
    shown = runner.invoke(app, ["config", "show", str(config), "-t", str(output)])
    assert shown.exit_code == 0, shown.output
    holds = "# the store holds score_trace, shape [31]\n# the store holds score_grid, shape [17, 2]\n"
    assert shown.stdout.endswith(holds)
    plain = _calibrate(phase_plate, phase_config, output, "--overwrite")  # the default outputs: no score arrays
    assert plain.exit_code == 0, plain.output
    assert TfStore.load(output).scores == {}
    assert "# the store holds" not in runner.invoke(app, ["config", "show", str(config), "-t", str(output)]).stdout


@pytest.mark.parametrize("rule", ["otsu_cnr", "l_curve"])
def test_a_weight_sweep_writes_its_report_and_montage(phase_plate, tmp_path, rule):
    report, plot = tmp_path / "out" / "sweep.json", tmp_path / "out" / "sweep.png"
    auto = f"{{auto: {{rule: {rule}, num_samples: 9, report_path: {report}, plot_path: {plot}}}}}"
    config = tmp_path / "auto.yml"
    config.write_text(FIXED_PHASE.replace("weight: 4e-4", f"weight: {auto}"))
    result = _calibrate(phase_plate, config, tmp_path / "tf.zarr")
    assert result.exit_code == 0, result.output
    written = json.loads(report.read_text())
    assert written["rule"] == rule
    assert written["crop"] == {"y0": 0, "x0": 0, "size": SHAPE[1]}  # the whole 32 x 32 frame: under 256 pixels
    assert len(written["regularization_strengths"]) == 9
    stored = TfStore.load(tmp_path / "tf.zarr").row()["object_prior.weight"]
    assert stored == pytest.approx(written["regularization_strength"], rel=1e-6)
    assert plot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
