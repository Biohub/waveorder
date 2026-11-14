"""The W&B observer maps waveorder's events onto a run's config, log and summary, offline with a fake run."""

from collections.abc import Mapping

import pytest
import torch
import yaml

from tests.test_observe import BEADS, SAMPLING, _data, _model
from tests.tiling.conftest import blur_model
from waveorder import observe
from waveorder.config import V4Config
from waveorder.integrations.wandb import WandbObserver
from waveorder.models import ReconstructionModel
from waveorder.nn import FISTA, Evidence, GradientSearch, GridSearch, Loop, calibrate
from waveorder.tiling import TilePlan, reconstruct_series

wandb = pytest.importorskip("wandb")
NA = "system.detection.numerical_aperture"


class _Config:
    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, object], bool | None]] = []

    def update(self, d: Mapping[str, object], allow_val_change: bool | None = None) -> None:  # noqa: FBT001
        self.updates.append((dict(d), allow_val_change))


class _Summary:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, d: Mapping[str, object]) -> None:
        self.values.update(d)


class FakeRun:
    """What WandbObserver touches on a ``wandb.Run``, recorded instead of sent."""

    def __init__(self) -> None:
        self.config = _Config()
        self.summary = _Summary()
        self.logged: list[dict[str, object]] = []

    def log(self, data: dict[str, object]) -> None:
        self.logged.append(data)


def test_run_start_updates_the_config_and_the_rest_reaches_the_summary():
    model, run = ReconstructionModel.from_config(V4Config(**yaml.safe_load(BEADS))), FakeRun()
    with observe(WandbObserver(run)):
        model.reconstruct(_data(), pixel_size=SAMPLING.pixel_size, prepared=True)
    ((config, allow_change),) = run.config.updates
    assert allow_change is True
    assert config["method"] == "model.reconstruct"
    assert config["config"] == V4Config(**yaml.safe_load(BEADS)).model_dump(mode="json")
    summary = run.summary.values
    assert summary["run/model.reconstruct/status"] == "ok"
    assert summary["solve/status"] == "direct"
    assert run.logged == []  # a closed-form solve has no history to plot


def test_each_model_method_keeps_its_own_run_summary():
    run = FakeRun()
    observer = WandbObserver(run)
    observer.event("run.end", {"method": "calibrate", "status": "ok", "wall_s": 2.0})
    observer.event("run.end", {"method": "reconstruct", "status": "error", "error": "broken", "wall_s": 1.0})
    summary = run.summary.values
    assert (summary["run/calibrate/status"], summary["run/calibrate/wall_s"]) == ("ok", 2.0)
    assert (summary["run/reconstruct/status"], summary["run/reconstruct/error"]) == ("error", "broken")


def test_a_solve_history_is_one_line_plot_log():
    model, run = _model(), FakeRun()
    with observe(WandbObserver(run), iterations=True):
        result = FISTA(loop=Loop(4))(_data(), model, SAMPLING)
    ((logged,),) = [list(entry.items()) for entry in run.logged]
    key, chart = logged
    assert key == "solve/history"
    assert isinstance(chart, wandb.plot.CustomChart)
    assert result.history is not None
    expected = [[index, value] for index, value in enumerate(result.history.tolist(), start=1)]
    assert chart.table.data == expected  # float32 values survive the float64 read-back exactly
    summary = run.summary.values
    assert (summary["solve/solver"], summary["solve/iterations"]) == ("FISTA", 4)
    assert "solve/history" not in summary


def test_calibration_stages_log_in_order_and_the_end_is_summarized():
    model, run = _model(), FakeRun()
    stages = (GridSearch(points=3), GradientSearch(iterations=2))
    with observe(WandbObserver(run), iterations=True):
        result = calibrate(_data(), model, SAMPLING, score=Evidence(), search=stages)
    assert [entry["calibration/stage"] for entry in run.logged] == [0, 1]
    assert [entry["calibration/objective"] for entry in run.logged] == [level for _, level in result.trace]
    assert isinstance(run.logged[1]["calibration/steps/1.GradientSearch"], wandb.plot.CustomChart)
    summary = run.summary.values
    assert summary[f"calibration/parameters/{NA}"] == pytest.approx(result.parameters[NA].item())


def test_tiles_end_reaches_the_summary():
    run, shape = FakeRun(), (8, 48, 40)
    plan = TilePlan(shape=shape, tile=(8, 24, 20))
    with observe(WandbObserver(run)):
        reconstruct_series(
            [(torch.randn(1, *shape), torch.zeros(1, *shape))],
            model=blur_model(0.5),
            solver=FISTA(loop=Loop(2)),
            pixel_size=(1.0, 1.0, 1.0),
            plan=plan,
            device="cpu",
        )
    assert run.summary.values["tiles/tiles"] == len(plan.regions)
    assert run.logged == []
