"""What the run commands show while they run, and the log levels."""

import logging
import subprocess
import sys
import time
import warnings

import pytest
from typer.testing import CliRunner

from tests.cli.plates import FIXED_PHASE, FREE_PHASE, NA, series, write_plate
from waveorder.cli import app
from waveorder.errors import WaveorderWarning

runner = CliRunner()


@pytest.fixture
def plate(tmp_path, phase_volume):
    path = tmp_path / "in.zarr"
    write_plate(path, [series(phase_volume, 2), series(-phase_volume, 2)])
    return path


def _config(tmp_path, text: str, name: str = "phase.yml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _rec(plate, config, output, *extra: str):
    return runner.invoke(app, ["rec", "-i", f"{plate}/*/*/*", "-c", str(config), "-o", str(output), *extra])


def test_unit_lines_when_stderr_is_not_a_terminal(plate, tmp_path):
    result = _rec(plate, _config(tmp_path, FIXED_PHASE), tmp_path / "out.zarr")
    assert result.exit_code == 0, result.output
    lines = [line for line in result.stderr.splitlines() if " unit A/1/" in line]
    assert len(lines) == 4
    assert lines[-1].endswith("; 4/4 units")
    assert " reconstructed in " in lines[0]
    assert "out.zarr: 4 units on cpu" in result.stderr


def test_rec_says_what_it_calibrates_then_the_stages_and_the_values(plate, tmp_path):
    result = _rec(plate, _config(tmp_path, FREE_PHASE), tmp_path / "out.zarr")
    assert result.exit_code == 0, result.output
    lines = result.stderr.splitlines()
    calibrating = next(index for index, line in enumerate(lines) if " calibrating on A/1/0 at t=0 ..." in line)
    stage = next(index for index, line in enumerate(lines) if " calibration stage 0 " in line)
    table = next(index for index, line in enumerate(lines) if line.split()[:1] == ["quantity"])
    unit = next(index for index, line in enumerate(lines) if " unit A/1/" in line)
    assert calibrating < stage < table < unit
    assert lines[table + 1].split()[0] == NA
    assert f" calibrated {tmp_path / 'out_transfer_function_phase.zarr'}:" in lines[table - 1]


def test_the_periodic_summary_of_a_run_on_a_pipe(caplog):
    from waveorder.cli.options import PROGRESS_LOGGER
    from waveorder.cli.render import RunDisplay
    from waveorder.models import UnitResult

    display = RunDisplay(bar=False, interval=0.05)
    failed = UnitResult(unit="A/1/0/t1", position="A/1/0", time=1, status="failed", error="E: x")
    with caplog.at_level(logging.INFO, logger=PROGRESS_LOGGER):
        display.start("/x/out.zarr", 3)
        display.unit(failed)
        deadline = time.monotonic() + 10
        while "1 failed" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
        display.close()
    messages = [record.getMessage() for record in caplog.records]
    assert not any("A/1/0/t1" in message for message in messages)  # the run logs a failed unit itself
    assert "1/3 units: 0 reconstructed, 1 failed, 2 left; 0 s" in messages


def _fake_reconstruct_positions(self, positions, *, output, on_start=None, on_unit=None, **arguments):
    """What a reconstruction logs and warns through every channel, at every level, then one unit."""
    from waveorder.models import ReconstructionReport, UnitResult

    del self, positions, arguments
    nodes = logging.getLogger("iohub.ngff.nodes")
    nodes.warning("iohub warning")
    ours = logging.getLogger("waveorder.models._positions")
    ours.warning("waveorder warning")
    ours.info("waveorder info")
    logging.getLogger("zarr.core").warning("zarr warning")
    warnings.warn("waveorder python warning", WaveorderWarning, stacklevel=1)
    warnings.warn("other python warning", UserWarning, stacklevel=1)
    if on_start is not None:
        on_start(1)
    result = UnitResult(
        unit="A/1/0/t0", position="A/1/0", time=0, status="complete", wall_s=1.0, read_s=0.25, solve_s=0.5,
        write_s=0.125, host="h", device="d",
    )  # fmt: skip
    if on_unit is not None:
        on_unit(result)
    return ReconstructionReport(output=str(output), status="finished", total=1, results=[result])


def test_stderr_shows_waveorder_warnings_and_the_progress_only(monkeypatch, tmp_path):
    from waveorder.models import ReconstructionModel

    monkeypatch.setattr(ReconstructionModel, "reconstruct_positions", _fake_reconstruct_positions)
    result = _rec(tmp_path / "in.zarr", _config(tmp_path, FIXED_PHASE), tmp_path / "out.zarr")
    assert result.exit_code == 0, result.output
    assert "finished; 1 unit:" in result.stdout
    for message in ("waveorder warning", "waveorder python warning", "unit A/1/0/t0 reconstructed in 1.0 s on d"):
        assert message in result.stderr
    for message in ("waveorder info", "zarr warning", "other python warning", "iohub warning"):
        assert message not in result.stderr


def test_iohub_stays_silent_in_a_real_run(plate, tmp_path):
    config = _config(tmp_path, FIXED_PHASE)
    command = [sys.executable, "-m", "waveorder.cli", "rec", "-i", str(plate), "-c", str(config)]
    default = subprocess.run([*command, "-o", str(tmp_path / "a.zarr")], capture_output=True, text=True, check=True)
    assert "iohub" not in default.stderr, default.stderr
    assert len([line for line in default.stderr.splitlines() if "unit A/1/1/t0" in line]) == 1  # each unit once


def test_calibrate_tf_prints_its_stages(phase_plate, phase_config, tmp_path):
    arguments = ["calibrate-tf", "-i", str(phase_plate / "A/1/0"), "-c", str(phase_config)]
    result = runner.invoke(app, [*arguments, "-o", str(tmp_path / "tf.zarr")])
    assert result.exit_code == 0, result.output
    stages = [line for line in result.stderr.splitlines() if " calibration stage " in line]
    assert [line.split(" calibration stage ")[1].split(":")[0] for line in stages] == [
        "0 GridSearch",
        "1 GradientSearch",
    ]
    assert all("objective " in line and not line.endswith(" s") for line in stages)  # no per-stage seconds
    assert " calibrating on A/1/0 at t=0 ..." in result.stderr.splitlines()[0]  # before 0 GridSearch ends
