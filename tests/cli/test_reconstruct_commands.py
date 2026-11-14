"""``wo rec``, ``wo apply-inv-tf`` and ``wo compute-tf`` end to end on tiny CPU plates."""

import re
from pathlib import Path

import iohub
import pytest
import torch
from typer.testing import CliRunner

from tests.cli.plates import (
    FIELD_TILED,
    FIXED_PHASE,
    FLUORESCENCE,
    FREE_PHASE,
    NA,
    THIN_PHASE,
    TRUE_NA,
    read_output,
    series,
    write_plate,
)
from tests.models.bundles import phase_bundle
from waveorder.cli import app
from waveorder.io import TfStore
from waveorder.io.bundle import load_bundle

runner = CliRunner()


@pytest.fixture
def plate(tmp_path, phase_volume):
    path = tmp_path / "in.zarr"
    write_plate(path, [series(phase_volume, 2), series(-phase_volume, 2)])
    return path


def _config(tmp_path, text: str, name: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def _rec(plate, config, output, *extra: str, command: str = "rec"):
    return runner.invoke(app, [command, "-i", f"{plate}/*/*/*", "-c", str(config), "-o", str(output), *extra])


def test_rec_thick_phase(plate, tmp_path):
    output = tmp_path / "out.zarr"
    result = _rec(plate, _config(tmp_path, FIXED_PHASE, "phase.yml"), output)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{output}: finished; 4 units: 4 reconstructed, 0 failed, 0 left\n"
    names, image, attrs = read_output(output, "A/1/1")
    assert names == ["Phase3D"]
    assert image.shape == (2, 1, 8, 32, 32)
    assert "waveorder-Phase3D" in attrs
    assert not (tmp_path / "out_transfer_function_phase.zarr").exists()  # nothing free: no store
    again = _rec(plate, tmp_path / "phase.yml", output)  # no --resume: every unit is written again, in place
    assert again.exit_code == 0, again.output
    assert again.stdout == f"{output}: finished; 4 units: 4 reconstructed, 0 failed, 0 left\n"
    assert torch.equal(read_output(output, "A/1/1")[1], image)


def test_rec_resume_skips_the_units_an_earlier_run_finished_unless_the_reconstruction_changed(plate, tmp_path):
    config = _config(tmp_path, FIXED_PHASE, "phase.yml")
    first_time = _config(
        tmp_path,
        FIXED_PHASE.replace("  input_channel_names: [BF]\n", "  input_channel_names: [BF]\n  time_indices: [0]\n"),
        "t0.yml",
    )
    output = tmp_path / "out.zarr"
    first = _rec(plate, first_time, output)  # as a run stopped after its first time point
    assert first.exit_code == 0, first.output
    resumed = _rec(plate, config, output, "--resume")
    assert resumed.exit_code == 0, resumed.output
    assert resumed.stdout == f"{output}: finished; 4 units: 2 reconstructed, 2 skipped, 0 failed, 0 left\n"
    whole = tmp_path / "whole.zarr"
    assert _rec(plate, config, whole).exit_code == 0
    assert torch.equal(read_output(output, "A/1/1")[1], read_output(whole, "A/1/1")[1])
    changed = _config(tmp_path, FIXED_PHASE.replace("weight: 4e-4", "weight: 1e-3"), "changed.yml")
    again = _rec(plate, changed, output, "--resume")
    assert again.exit_code == 0, again.output
    assert "4 reconstructed" in again.stdout  # another reconstruction: nothing is finished
    both = _rec(plate, config, output, "--resume", "--overwrite")
    assert both.exit_code == 2
    assert "--resume continues the output that --overwrite deletes; give one of them" in both.stderr
    help_text = re.sub(
        r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m|[│╭╮╰╯─]", "", runner.invoke(app, ["rec", "--help"]).stdout)
    )
    assert "--resume" in help_text
    assert "Needs an OME-Zarr v0.5 output." in help_text


def test_rec_thin_phase_writes_one_plane_with_phase_and_absorption(plate, tmp_path):
    output = tmp_path / "thin.zarr"
    result = _rec(plate, _config(tmp_path, THIN_PHASE, "thin.yml"), output)
    assert result.exit_code == 0, result.output
    names, image, attrs = read_output(output)
    assert names == ["Phase2D", "Absorption2D"]
    assert image.shape == (2, 2, 1, 32, 32)
    assert "waveorder-Phase2D" in attrs


def test_rec_with_a_distilled_student_writes_what_the_student_returns(plate, tmp_path):
    bundle = phase_bundle(tmp_path / "student")  # trained for the system of THIN_PHASE
    text = THIN_PHASE.replace("reconstruction: closed_form", f"reconstruction: {{distilled: {{bundle: {bundle}}}}}")
    output = tmp_path / "distilled.zarr"
    result = _rec(plate, _config(tmp_path, text, "distilled.yml"), output)
    assert result.exit_code == 0, result.output
    assert "4 units: 4 reconstructed" in result.stdout
    names, image, attrs = read_output(output, "A/1/1")
    assert names == ["Phase2D", "Absorption2D"]
    assert image.shape == (2, 2, 1, 32, 32)
    assert image[:, 0].abs().max() > 0
    assert image[:, 1].abs().max() == 0  # the student returns phase only
    assert attrs["waveorder-Phase2D"]["bundle"]["sha256"] == load_bundle(bundle).sha256


def test_apply_inverse_transfer_function_uses_a_store_calibrate_tf_wrote(plate, tmp_path):
    config = _config(tmp_path, FLUORESCENCE, "fluor.yml")
    store = tmp_path / "tf.zarr"
    calibrated = runner.invoke(app, ["calibrate-tf", "-i", str(plate / "A/1/0"), "-c", str(config), "-o", str(store)])
    assert calibrated.exit_code == 0, calibrated.output
    ((name, aperture),) = TfStore.load(store).row().items()
    applied = _rec(plate, config, tmp_path / "applied.zarr", "-t", str(store), command="apply-inv-tf")
    assert applied.exit_code == 0, applied.output
    names, image, attrs = read_output(tmp_path / "applied.zarr", "A/1/1")
    assert names == ["BF_Density3D"]
    assert image.abs().max() > 0
    # The output records the model as it reconstructed: the configuration with the store's value calibrated.
    assert attrs["waveorder-BF_Density3D"]["config"]["calibrated"] == {name: pytest.approx(aperture)}
    assert not list(tmp_path.glob("*_transfer_function_*"))  # -t: nothing is calibrated or published
    assert "transfer function on" not in applied.stderr  # a calibration store holds no transfer function
    missing = runner.invoke(app, ["apply-inv-tf", "-i", str(plate), "-c", str(config), "-o", "x.zarr"])
    assert missing.exit_code == 2
    assert "Missing option '-t'" in missing.stderr


def test_rec_without_a_store_calibrates_publishes_the_store_beside_the_output_and_reconstructs(plate, tmp_path):
    config = _config(tmp_path, FREE_PHASE, "free.yml")
    output = tmp_path / "out.zarr"
    store = tmp_path / "out_transfer_function_free.zarr"
    result = _rec(plate, config, output)
    assert result.exit_code == 0, result.output
    assert "calibrating on A/1/0 at t=0 ..." in result.stderr
    assert "4 reconstructed" in result.stdout
    assert "transfer function on" not in result.stderr  # a run that estimates says nothing of stored transfers
    aperture = TfStore.load(store).row()[NA]
    assert aperture == pytest.approx(TRUE_NA, abs=0.04)
    assert TfStore.load(store).provenance["position_name"] == "A/1/0"
    names, image, attrs = read_output(output)
    assert names == ["Phase3D"]
    assert image.abs().max() > 0
    assert attrs["waveorder-Phase3D"]["config"]["calibrated"] == {NA: pytest.approx(aperture)}
    again = _rec(plate, config, output)  # a rerun calibrates again and replaces the store this command wrote
    assert again.exit_code == 0, again.output
    assert "calibrating on A/1/0" in again.stderr


def test_compute_tf_then_apply_inv_tf_on_another_plate_of_the_grid_reads_the_transfer_function(
    plate, phase_volume, tmp_path
):
    config = _config(tmp_path, FIXED_PHASE, "phase.yml")
    store = tmp_path / "tf.zarr"
    computed = runner.invoke(app, ["compute-tf", "-i", str(plate / "A/1/0"), "-c", str(config), "-o", str(store)])
    assert computed.exit_code == 0, computed.output
    assert computed.stdout.splitlines()[0] == f"{store}: nominal parameters; transfer functions at 8x32x32"
    other = tmp_path / "other.zarr"
    write_plate(other, [series(0.5 * phase_volume, 1), series(-0.7 * phase_volume, 1)])  # other data, the same grid
    applied = _rec(other, config, tmp_path / "applied.zarr", "-t", str(store), "-j", "2", command="apply-inv-tf")
    assert applied.exit_code == 0, applied.output
    assert applied.stderr.count(f"the transfer function on 8x32x32 is read from {store}, not computed") == 1
    direct = _rec(other, config, tmp_path / "direct.zarr")
    assert direct.exit_code == 0, direct.output
    assert "transfer function on" not in direct.stderr
    for position in ("A/1/0", "A/1/1"):  # the workers read the store themselves: the same bits as computing it
        assert torch.equal(
            read_output(tmp_path / "applied.zarr", position)[1], read_output(tmp_path / "direct.zarr", position)[1]
        )
    shown = runner.invoke(app, ["config", "show", str(config), "-t", str(store)])
    assert shown.exit_code == 0, shown.output
    assert "# the store holds a transfer function on 8x32x32 ([0.4, 0.15, 0.15] um)" in shown.stdout


def test_rec_refuses_a_3x_configuration_naming_wo_config_migrate(plate, tmp_path):
    old = _config(tmp_path, "input_channel_names: [BF]\nreconstruction_dimension: 3\n", "old.yml")
    result = _rec(plate, old, tmp_path / "out.zarr")
    assert result.exit_code == 2
    assert f"wo rec: error: {old}: has no version, so it is a 3.x configuration" in result.stderr
    assert f"wo config migrate {old}" in result.stderr
    assert not list(tmp_path.glob("out*"))


def test_every_mistake_of_a_request_is_reported_together(tmp_path):
    fixed = _config(tmp_path, FIXED_PHASE, "phase.yml")
    missing, store, output = tmp_path / "none.zarr", tmp_path / "tf.zarr", tmp_path / "out.zarr"
    arguments = ["apply-inv-tf", "-i", str(missing), "-c", str(fixed), "-t", str(store), "-o", str(output)]
    result = runner.invoke(app, [*arguments, "--tile-overlap", "1,1,1"])
    assert result.exit_code == 2
    assert result.stderr.splitlines() == [  # the store, the positions and the tile flags are each checked
        f"wo apply-inv-tf: error: -t {store}: does not exist",
        f"wo apply-inv-tf: error: -i {missing}: {missing} matches nothing on disk",
        "wo apply-inv-tf: error: --tile-overlap: widens --tile cores; give --tile too",
    ]
    assert not list(tmp_path.glob("out*"))


def test_rec_refuses_the_reconstruction_before_it_calibrates(plate, tmp_path):
    result = _rec(plate, _config(tmp_path, FREE_PHASE, "free.yml"), tmp_path / "out.zarr", "--tile-overlap", "1,1,1")
    assert result.exit_code == 2
    assert "wo rec: error: --tile-overlap: widens --tile cores; give --tile too" in result.stderr.splitlines()
    assert "calibrating" not in result.stderr
    assert not list(tmp_path.glob("out*"))


def test_apply_inv_tf_field_tiled_with_its_calibrated_store(tmp_path):
    generator = torch.Generator().manual_seed(0)
    plate = tmp_path / "in.zarr"
    write_plate(plate, [1 + 0.05 * torch.randn(1, 8, 32, 32, generator=generator)])
    config = _config(tmp_path, FIELD_TILED, "tiled.yml")
    store = tmp_path / "tf.zarr"
    calibrated = runner.invoke(app, ["calibrate-tf", "-i", str(plate / "A/1/0"), "-c", str(config), "-o", str(store)])
    assert calibrated.exit_code == 0, calibrated.output
    assert TfStore.load(store).tiles == 4
    output = tmp_path / "out.zarr"
    result = _rec(plate, config, output, "-t", str(store), command="apply-inv-tf")
    assert result.exit_code == 0, result.output
    names, image, _ = read_output(output)
    assert names == ["BF_Density3D"]
    assert image.abs().amax(dim=(0, 1, 2, 4)).min() > 0
    with iohub.open_ome_zarr(output / "A/1/0", mode="r") as position:
        assert tuple(position["0"].shards)[2:4] == (8, 16)
    refused = _rec(plate, config, tmp_path / "x.zarr", "-t", str(store), "--tile", "8,16,16", command="apply-inv-tf")
    assert refused.exit_code == 2
    message = "field tiles come from the model's imaging_system.field_dependence; drop --tile"
    assert f"wo apply-inv-tf: error: --tile: {message}" in refused.stderr.splitlines()


def test_write_config_scale_to_output_with_and_without_a_mismatch(tmp_path, phase_volume):
    config = _config(tmp_path, FIXED_PHASE, "phase.yml")
    other = tmp_path / "other.zarr"
    write_plate(other, [series(phase_volume, 1)], pixel_size=(0.5, 0.15, 0.15))
    used = _rec(other, config, tmp_path / "used.zarr")  # the configuration's size is used, with a warning
    assert used.exit_code == 0, used.output
    (line,) = [line for line in used.stderr.splitlines() if "warning" in line]
    assert "data.pixel_size is (z, y, x) = (0.4, 0.15, 0.15) um, but A/1/0 records (0.5, 0.15, 0.15) um" in line
    written = _rec(other, config, tmp_path / "scaled.zarr", "--write-config-scale-to-output")
    assert written.exit_code == 0, written.output
    same = tmp_path / "same.zarr"
    write_plate(same, [series(phase_volume, 1)])
    plain = _rec(same, config, tmp_path / "plain.zarr", "--write-config-scale-to-output")
    assert plain.exit_code == 0, plain.output
    assert "warning" not in plain.stderr
    assert torch.equal(read_output(tmp_path / "used.zarr")[1], read_output(tmp_path / "plain.zarr")[1])
    for name, scale in (("used", 0.5), ("scaled", 0.4), ("plain", 0.4)):
        with iohub.open_ome_zarr(tmp_path / f"{name}.zarr" / "A/1/0", mode="r") as position:
            assert list(position.scale[2:]) == pytest.approx([scale, 0.15, 0.15])


def test_check_messages_name_the_flags_not_the_python_arguments(plate, tmp_path):
    config = _config(tmp_path, FIXED_PHASE, "phase.yml")
    output = tmp_path / "out.zarr"
    late = _config(
        tmp_path,
        FIXED_PHASE.replace("  input_channel_names: [BF]\n", "  input_channel_names: [BF]\n  time_indices: [7]\n"),
        "late.yml",
    )
    result = _rec(plate, late, output, "--tile-overlap", "1,1,1")
    assert result.exit_code == 2
    lines = result.stderr.splitlines()  # a flag that is the location is not repeated in front of the message
    assert "wo rec: error: --tile-overlap: widens --tile cores; give --tile too" in lines
    selects = "data.time_indices selects 7, outside the 2 time points of the data"
    assert f"wo rec: error: {plate / 'A/1/0'}: {selects}" in lines
    for argument in ("tile_overlap", "times ", "positions", "pixel_size="):
        assert argument not in result.stderr
    none = tmp_path / "none.zarr"
    missing = runner.invoke(app, ["rec", "-i", str(none), "-c", str(config), "-o", str(output)])
    assert missing.exit_code == 2
    assert missing.stderr.splitlines() == [f"wo rec: error: -i {none}: {none} matches nothing on disk"]
    target = plate / "A/1/0" / "out.zarr"
    inside = _rec(plate, config, target)
    assert inside.exit_code == 2
    assert f"wo rec: error: -o {target}: is the input position A/1/0 or lies inside it" in inside.stderr


def test_rec_with_two_worker_processes(plate, tmp_path):
    output = tmp_path / "out.zarr"
    result = _rec(plate, _config(tmp_path, FIXED_PHASE, "phase.yml"), output, "-j", "2")
    assert result.exit_code == 0, result.output
    assert "4 reconstructed" in result.stdout
    assert " out.zarr: 4 units on cpu (2 workers)" in result.stderr


def test_a_device_list_spreads_the_units_over_one_worker_per_device(plate, tmp_path):
    config = _config(tmp_path, FIXED_PHASE, "phase.yml")
    result = _rec(plate, config, tmp_path / "out.zarr", "--device", "cpu,cpu")
    assert result.exit_code == 0, result.output
    assert "4 reconstructed" in result.stdout
    assert " out.zarr: 4 units on cpu (2 workers)" in result.stderr  # one worker per device of the list
    absent = _rec(plate, config, tmp_path / "x.zarr", "--device", "cpu,cuda:99")
    assert absent.exit_code == 2
    assert "wo rec: error: --device: 'cuda:99' is not available here" in absent.stderr
    assert not (tmp_path / "x.zarr").exists()


def _failing_every_second_unit(monkeypatch) -> list[str]:
    from waveorder.models import _positions

    process = _positions.process_time
    calls: list[str] = []

    def failing(solve, **arguments):
        calls.append("/".join(Path(arguments["source"]).parts[-3:]) + f"/t{arguments['time']}")
        if len(calls) % 2 == 0:
            raise OSError("blosc decompression failed")
        return process(solve, **arguments)

    monkeypatch.setattr(_positions, "process_time", failing)
    return calls


def test_a_failed_unit_stops_the_run_with_its_counts_and_the_hint(plate, tmp_path, monkeypatch):
    calls = _failing_every_second_unit(monkeypatch)
    output = tmp_path / "out.zarr"
    result = _rec(plate, _config(tmp_path, FIXED_PHASE, "phase.yml"), output)
    assert result.exit_code == 1, result.output
    assert result.stdout == f"{output}: failed; 4 units: 1 reconstructed, 1 failed, 2 left\n"
    error = f"wo rec: error: unit {calls[1]} failed: OSError: blosc decompression failed"
    assert error in result.stderr.splitlines()
    hint = (
        "wo rec: fix the cause and rerun with --resume to skip the finished units; --keep-going runs past a failed unit"
    )
    assert hint in result.stderr.splitlines()
    assert "Traceback" not in result.stderr


def test_a_single_position_store_among_other_inputs_is_refused_before_any_write(plate, tmp_path, phase_volume):
    single = tmp_path / "my-run" / "cli4" / "fov0"  # no plate lists it: a single-position store
    single.parent.mkdir(parents=True)
    with iohub.open_ome_zarr(single, layout="fov", mode="w-", channel_names=["BF"]) as node:
        node.create_image("0", series(phase_volume, 1)[:, None].to(torch.float32).numpy())
    config = _config(tmp_path, FIXED_PHASE, "phase.yml")
    output = tmp_path / "out.zarr"
    arguments = ["rec", "-i", str(single), "-i", str(plate / "A" / "1" / "0"), "-c", str(config), "-o", str(output)]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2, result.output
    assert "wo rec: error: -i: " in result.stderr
    assert "one is a single-position store" in result.stderr
    assert "reconstruct each single-position store in a run of its own" in result.stderr
    assert not output.exists()


def test_keep_going_runs_past_failed_units_and_exits_3(plate, tmp_path, monkeypatch):
    _failing_every_second_unit(monkeypatch)
    output = tmp_path / "out.zarr"
    result = _rec(plate, _config(tmp_path, FIXED_PHASE, "phase.yml"), output, "--keep-going")
    assert result.exit_code == 3, result.output
    assert result.stdout.splitlines() == [
        f"{output}: failed; 4 units: 2 reconstructed, 2 failed, 0 left",
        "  failed A/1/0/t1: OSError: blosc decompression failed",
        "  failed A/1/1/t1: OSError: blosc decompression failed",
    ]


def test_rec_help_fits_80_columns():
    result = runner.invoke(app, ["rec", "--help"], terminal_width=80, env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    for panel in ("Inputs", "Outputs", "Run"):
        assert panel in text
    assert "Example: wo rec" in text
    assert "DEVICE[,...]" in text


def test_compute_tf_writes_nominal_transfer_functions_per_shape(plate, phase_volume, tmp_path):
    config = _config(tmp_path, FLUORESCENCE, "fluor.yml")
    other = tmp_path / "other.zarr"
    write_plate(other, [series(phase_volume[:4], 1)])
    output = tmp_path / "tf.zarr"
    arguments = ["compute-tf", "-i", str(plate / "A/1/0"), str(other / "A/1/0"), "-c", str(config), "-o", str(output)]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == f"{output}: nominal parameters; transfer functions at 8x32x32, 4x32x32"
    store = TfStore.load(output)
    assert store.row() == {"system.detection.numerical_aperture": pytest.approx(0.8)}
    again = runner.invoke(app, arguments)
    assert again.exit_code == 2
    assert f"wo compute-tf: error: -o {output}: exists; pass --overwrite to replace it" in again.stderr
    assert runner.invoke(app, [*arguments, "--overwrite"]).exit_code == 0


def test_compute_tf_takes_the_voxel_size_the_positions_record_when_the_configuration_says_auto(
    plate, phase_volume, tmp_path
):
    config = _config(
        tmp_path, FLUORESCENCE.replace("pixel_size: {z: 0.4, y: 0.15, x: 0.15}", "pixel_size: auto"), "f.yml"
    )
    output = tmp_path / "tf.zarr"
    result = runner.invoke(app, ["compute-tf", "-i", str(plate / "A/1/0"), "-c", str(config), "-o", str(output)])
    assert result.exit_code == 0, result.output
    (transfer,) = TfStore.load(output).transfers
    assert transfer.attrs["pixel_size_zyx_um"] == pytest.approx([0.4, 0.15, 0.15])
    other = tmp_path / "other.zarr"
    write_plate(other, [series(phase_volume, 1)], pixel_size=(0.5, 0.15, 0.15))
    arguments = ["compute-tf", "-i", str(plate / "A/1/0"), str(other / "A/1/0"), "-c", str(config), "-o", "two.zarr"]
    mixed = runner.invoke(app, arguments)
    assert mixed.exit_code == 2
    assert f"{config}: data.pixel_size is auto, but the positions record 2 voxel sizes" in mixed.stderr
