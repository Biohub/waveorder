"""What the run commands show while they run, the log levels, and ``--wandb``."""

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
    assert " out.zarr: 4 units on cpu" in result.stderr


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

    display = RunDisplay(bar=False, interval=0.05)
    start = {"output": "/x/out.zarr", "total": 3, "positions": 1, "devices": ["cuda:0", "cuda:1"], "workers": 2}
    failed = {"unit": "A/1/0/t1", "position": "A/1/0", "time": 1, "status": "failed", "error": "E: x"}
    with caplog.at_level(logging.INFO, logger=PROGRESS_LOGGER):
        display.event("units.start", start)
        display.event("unit.end", failed)
        deadline = time.monotonic() + 10
        while "1 failed" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
        display.close()
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "out.zarr: 3 units on cuda:0, cuda:1 (2 workers)"
    assert not any("A/1/0/t1" in message for message in messages)  # the run logs a failed unit itself
    assert "1/3 units: 0 reconstructed, 1 failed, 2 left; 0 s" in messages


def _fake_reconstruct_positions(self, positions, *, output, **arguments):
    """What a reconstruction logs, warns and emits through every channel, at every level, then one unit."""
    from waveorder.models import ReconstructionReport, UnitResult
    from waveorder.observe import emit

    del self, positions, arguments
    nodes = logging.getLogger("iohub.ngff.nodes")
    nodes.warning("iohub warning")
    ours = logging.getLogger("waveorder.models._positions")
    ours.warning("waveorder warning")
    ours.info("waveorder info")
    logging.getLogger("zarr.core").warning("zarr warning")
    warnings.warn("waveorder python warning", WaveorderWarning, stacklevel=1)
    warnings.warn("other python warning", UserWarning, stacklevel=1)
    emit("units.start", {"output": str(output), "total": 1, "positions": 1, "devices": ["d"], "workers": 1})
    result = UnitResult(
        unit="A/1/0/t0", position="A/1/0", time=0, status="complete", wall_s=1.0, read_s=0.25, solve_s=0.5,
        write_s=0.125, host="h", device="d",
    )  # fmt: skip
    emit("unit.end", result.to_json())
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


class _Recorded:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.values: dict = {}

    def update(self, d, allow_val_change=None) -> None:
        del allow_val_change
        self.updates.append(dict(d))
        self.values.update(d)


class _FakeRun:
    def __init__(self) -> None:
        self.config = _Recorded()
        self.summary = _Recorded()
        self.logged: list[dict] = []
        self.finished: list[int] = []

    def log(self, data: dict) -> None:
        self.logged.append(data)

    def finish(self, exit_code: int = 0) -> None:
        self.finished.append(exit_code)


def test_wandb_logs_the_command_to_one_run_grouped_by_its_store(plate, tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    runs, calls = [], []

    def init(**kwargs):
        calls.append(kwargs)
        runs.append(_FakeRun())
        return runs[-1]

    monkeypatch.setattr(wandb, "init", init)
    config = _config(tmp_path, FREE_PHASE)
    result = _rec(plate, config, tmp_path / "out.zarr", "--wandb", "p")
    assert result.exit_code == 0, result.output
    ((kwargs,), (run,)) = calls, runs
    assert kwargs == {
        "project": "p",
        "group": "out.zarr",
        "job_type": "reconstruct",
        "config": {"waveorder": {"command": "reconstruct", "store": "out.zarr"}},
    }
    methods = [update["method"] for update in run.config.updates]
    assert methods[0] == "model.calibrate"  # the calibration the command runs first
    assert methods.count("model.reconstruct") == 4
    assert any("calibration/objective" in data for data in run.logged)
    assert run.summary.values["run/model.reconstruct/status"] == "ok"
    assert f"calibration/parameters/{NA}" in run.summary.values
    assert run.finished == [0]
    refused = _rec(plate, config, tmp_path / "x.zarr", "--wandb", "p", "--tile-overlap", "1,1,1")
    assert refused.exit_code == 2
    assert len(calls) == 1  # a refused request computes nothing, so it starts no W&B run


def test_calibrate_tf_prints_its_stages(phase_plate, phase_config, tmp_path):
    arguments = ["calibrate-tf", "-i", str(phase_plate / "A/1/0"), "-c", str(phase_config)]
    result = runner.invoke(app, [*arguments, "-o", str(tmp_path / "tf.zarr")])
    assert result.exit_code == 0, result.output
    stages = [line for line in result.stderr.splitlines() if " calibration stage " in line]
    assert [line.split(" calibration stage ")[1].split(":")[0] for line in stages] == [
        "0 GridSearch",
        "1 GradientSearch",
    ]
    assert all(line.endswith(" s") and "objective " in line for line in stages)
    assert " calibrating on A/1/0 at t=0 ..." in result.stderr.splitlines()[0]  # before 0 GridSearch ends
