"""Observer events at waveorder's public boundaries."""

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, override

import pytest
import torch
import yaml
from torch import Tensor

from tests.nn.conftest import FREE_NA, make_model, make_system, simulate
from tests.tiling.conftest import blur_model
from waveorder import observe
from waveorder.config import V4Config
from waveorder.models import ReconstructionModel
from waveorder.nn import (
    FISTA,
    Evidence,
    Gaussian,
    GradientSearch,
    GridSearch,
    ImagingModel,
    Loop,
    RandomSearch,
    Sampling,
    Score,
    Solver,
    SolveResult,
    Tikhonov,
    Wiener,
    calibrate,
    calibrate_tiles,
)
from waveorder.nn.verbs import InverseProblem
from waveorder.observe import context, emit
from waveorder.testing import assert_no_host_sync
from waveorder.tiling import TilePlan, halo_for, reconstruct_series

SAMPLING = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.1))
NA = "system.detection.numerical_aperture"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HOST = (str, int, float, bool, type(None))


class Recorder:
    """An observer that keeps every event and checks that its payload holds host values only."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, name: str, payload: Mapping[str, object]) -> None:
        _expect_host(payload, path=name)
        self.events.append((name, dict(payload)))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payloads(self, name: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]


def _expect_host(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert isinstance(key, str), f"{path} has a non-string key {key!r}"
            _expect_host(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _expect_host(item, path=f"{path}[{index}]")
    else:
        assert isinstance(value, HOST), f"{path} is a {type(value).__name__}, not a host value"


def _model() -> ImagingModel:
    return ImagingModel(make_system(FREE_NA), Tikhonov(0.05**2 / 9), Gaussian(0.05))


def _data() -> Tensor:
    return simulate(ImagingModel(make_system(1.0), Tikhonov(0.05**2 / 9), Gaussian(0.05)), SAMPLING, scale=3.0)[1]


# --- events -------------------------------------------------------------------------------------------------------


def test_a_direct_solve_reports_one_solve_end():
    model, data, recorder = make_model(1.0), _data(), Recorder()
    with observe(recorder):
        FISTA(loop=Loop(3))(data, model, SAMPLING)
    assert recorder.events == [
        ("solve.end", {"solver": "FISTA", "system": "WidefieldFluorescence3D", "status": "budget", "iterations": 3})
    ]


def test_a_recorded_history_is_delivered_when_the_block_exits():
    model, data, recorder = make_model(1.0), _data(), Recorder()
    with observe(recorder, iterations=True):
        result = FISTA(loop=Loop(4))(data, model, SAMPLING)
        assert recorder.events == []  # waiting for its device values, which are read only after the operation
    assert recorder.names == ["solve.end"]
    (payload,) = recorder.payloads("solve.end")
    assert result.history is not None
    assert payload["history"] == pytest.approx(result.history.tolist())
    assert len(payload["history"]) == 4


def test_every_observer_of_stacked_blocks_gets_every_event():
    model, data = make_model(1.0), _data()
    outer, inner = Recorder(), Recorder()
    with observe(outer):
        Wiener()(data, model, SAMPLING)
        with observe(inner):
            Wiener()(data, model, SAMPLING)
    assert outer.names == ["solve.end", "solve.end"]
    assert inner.names == ["solve.end"]


def test_a_context_adds_its_values_to_the_events_emitted_in_its_block():
    model, data, recorder = make_model(1.0), _data(), Recorder()
    with context(unit="A/1/0/t0"):
        emit("probe", {})  # no observer: nothing is sent, now or later
    with observe(recorder, iterations=True):
        Wiener()(data, model, SAMPLING)
        with context(unit="A/1/0/t0", worker=2):
            Wiener()(data, model, SAMPLING)
            FISTA(loop=Loop(3))(data, model, SAMPLING)  # delivered when the observe block exits: it keeps its unit
            with context(worker=3, solver="outer"):
                emit("probe", {"solver": "explicit"})
        emit("probe", {})
    before, inside, deferred = recorder.payloads("solve.end")
    assert "unit" not in before
    assert (inside["unit"], inside["worker"]) == ("A/1/0/t0", 2)
    assert (deferred["unit"], deferred["worker"], len(deferred["history"])) == ("A/1/0/t0", 2, 3)
    nested, after = recorder.payloads("probe")
    assert nested == {"unit": "A/1/0/t0", "worker": 3, "solver": "explicit"}  # inner over outer, the event's own first
    assert after == {}


def test_nothing_is_emitted_without_an_observer_or_from_inside_transforms():
    model, data, recorder = make_model(1.0), _data(), Recorder()
    solver = FISTA(loop=Loop(2))
    solver(data, model, SAMPLING)  # no observer: nothing is queued for a later block either
    with observe(recorder, iterations=True):
        history = torch.func.vmap(lambda d: solver(d, model, SAMPLING).history)(data[None])
        torch.func.grad(lambda d: solver(d, model, SAMPLING).objective)(data)
    assert recorder.events == []
    assert history is not None
    assert history.shape == (1, 2)


class _Solving(Score):
    """A score that runs a solver verb: its solves must not report inside a calibration."""

    is_likelihood: ClassVar[bool] = False

    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> Tensor:
        solved = FISTA(loop=Loop(2))(problem.data, problem.model, problem.sampling)
        return problem.objective(solved.estimate)


def test_calibrate_reports_its_stages_and_end_only():
    model, data, recorder = _model(), _data(), Recorder()
    stages = (GridSearch(points=3), GradientSearch(iterations=3))
    with observe(recorder, iterations=True):
        result = calibrate(data, model, SAMPLING, score=_Solving(), search=stages)
    assert recorder.names == ["calibration.stage.end", "calibration.stage.end", "calibration.end"]
    grid, gradient = recorder.payloads("calibration.stage.end")
    assert (grid["stage"], grid["index"], "steps" in grid) == ("GridSearch", 0, False)
    assert (gradient["stage"], gradient["index"]) == ("GradientSearch", 1)
    assert len(gradient["steps"]) == 3 + 1  # one evaluation per step and one at the end
    assert [grid["objective"], gradient["objective"]] == [level for _, level in result.trace]
    (end,) = recorder.payloads("calibration.end")
    assert end["parameters"] == {NA: pytest.approx(result.parameters[NA].item())}
    assert end["at_bound"] == []
    assert end["trace"] == [list(entry) for entry in result.trace]


def test_step_histories_are_recorded_only_with_iterations():
    model, data, recorder = _model(), _data(), Recorder()
    with observe(recorder):
        calibrate(data, model, SAMPLING, score=Evidence(), search=(GridSearch(points=5),))
    (end,) = recorder.payloads("calibration.end")
    assert end["at_bound"] == []
    assert "steps" not in recorder.payloads("calibration.stage.end")[0]  # not recorded without iterations=True


def test_calibrate_tiles_reports_nothing_from_its_per_tile_calibrations():
    model, recorder = _model(), Recorder()
    tiles = torch.stack([_data(), _data()])
    with observe(recorder):
        calibrate_tiles(tiles, model, SAMPLING, score=Evidence(), search=(GridSearch(points=3),))
    assert recorder.events == []


BEADS = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection: {wavelength: 0.52, numerical_aperture: 1.0}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.05}}
"""


def test_model_reconstruct_reports_its_run_and_solve():
    model, recorder = ReconstructionModel.from_config(V4Config(**yaml.safe_load(BEADS))), Recorder()
    data = _data()
    with observe(recorder):
        model.reconstruct(data, pixel_size=SAMPLING.pixel_size, prepared=True)
    assert recorder.names == ["run.start", "solve.end", "run.end"]
    start, end = recorder.payloads("run.start")[0], recorder.payloads("run.end")[0]
    assert start["method"] == "model.reconstruct"
    assert start["config"] == V4Config(**yaml.safe_load(BEADS)).model_dump(mode="json")
    assert start["device"] == "cpu"
    versions = start["versions"]
    assert isinstance(versions, dict)
    assert versions["torch"] is not None
    assert "cuda" in versions
    assert (end["method"], end["status"]) == ("model.reconstruct", "ok")


class _Failing(Solver):
    @override
    def check(self, model: ImagingModel) -> None:
        del model

    @override
    def run(self, problem: InverseProblem) -> SolveResult:
        raise ValueError("broken solve")


def test_a_failing_model_method_reports_its_error():
    model, recorder = ReconstructionModel.from_config(V4Config(**yaml.safe_load(BEADS))), Recorder()
    model.solver = _Failing()
    with observe(recorder), pytest.raises(ValueError, match="broken"):
        model.reconstruct(_data(), pixel_size=SAMPLING.pixel_size, prepared=True)
    assert recorder.names == ["run.start", "run.end"]
    assert recorder.payloads("run.end")[0]["status"] == "error"
    assert recorder.payloads("run.end")[0]["error"] == "ValueError: broken solve"


TILE_SHAPE = (8, 48, 40)


def test_a_tiled_reconstruction_reports_one_tiles_end():
    data, model, recorder = torch.randn(1, *TILE_SHAPE), blur_model(0.5), Recorder()
    plan = TilePlan(shape=TILE_SHAPE, tile=(8, 24, 20), overlap=(0, 4, 4))
    sink = torch.zeros(1, *TILE_SHAPE)

    def run() -> None:
        stats = reconstruct_series(
            [(data, sink)], model=model, solver=FISTA(loop=Loop(2)), pixel_size=(1.0, 1.0, 1.0), plan=plan, device="cpu"
        )
        assert stats.tiles == len(plan.regions)

    with observe(recorder, iterations=True):
        run()
    assert recorder.names == ["tiles.end"]
    (stats,) = recorder.payloads("tiles.end")
    assert stats["tiles"] == len(plan.regions)
    assert {"read_s", "gpu_s", "bytes_written", "nonfinite", "wall_s"} <= set(stats)


def test_halo_probe_solves_report_nothing():
    model, recorder = blur_model(0.5), Recorder()
    with observe(recorder, iterations=True):
        halo_for(model, Sampling(shape=(8, 24, 20), pixel_size=(1.0, 1.0, 1.0)), solver=FISTA(loop=Loop(2)))
    assert recorder.names == []


# --- iteration histories ------------------------------------------------------------------------------------------


def test_a_history_is_recorded_only_under_observe_iterations():
    model, data = make_model(1.0).double(), _data().double()
    solver = FISTA(loop=Loop(5))
    assert solver(data, model, SAMPLING).history is None  # not recorded outside observe(iterations=True)
    with observe(iterations=True):
        history = solver(data, model, SAMPLING).history
    assert history is not None
    assert history.shape == (5,)
    assert solver(data, model, SAMPLING).history is None


def test_a_tolerance_history_stops_with_the_solve():
    model, data = make_model(1.0).double(), _data().double()
    with observe(iterations=True):
        result = FISTA(loop=Loop(200, tolerance=1e-3, check_every=5))(data, model, SAMPLING)
    assert result.history is not None
    assert result.history.shape == (result.iterations,)
    assert result.history[-1] < result.history[0]


# --- CUDA: observing adds no sync ----------------------------------------------------------------------------------


@CUDA
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
def test_an_observed_solve_with_histories_stays_sync_free_on_cuda():
    model = ImagingModel(make_system(1.0), Tikhonov(1e-2, nonnegative=True), Gaussian(0.05)).double().cuda()
    data = _data().double().cuda()
    recorder = Recorder()
    solver = FISTA(loop=Loop(5))
    solver(data, model, SAMPLING)  # warm the norm estimate and plans outside the check
    torch.cuda.synchronize()
    with observe(recorder, iterations=True):
        result = assert_no_host_sync(lambda: solver(data, model, SAMPLING), what="an observed FISTA solve")
    assert result.history is not None
    assert recorder.names == ["solve.end"]
    assert len(recorder.payloads("solve.end")[0]["history"]) == 5


@CUDA
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
def test_an_observed_calibration_reads_back_once_on_cuda():
    data, model, recorder = _data().cuda(), _model().cuda(), Recorder()
    stages = (GridSearch(points=5, chunk_size=2), RandomSearch(samples=4), GradientSearch(iterations=3))
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with observe(recorder, iterations=True), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibrate(data, model, SAMPLING, score=Evidence(), search=stages)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    syncs = [str(warning.message) for warning in caught if "synchronizing CUDA operation" in str(warning.message)]
    assert len(syncs) == 1, syncs
    assert recorder.names == ["calibration.stage.end"] * 3 + ["calibration.end"]
    assert len(recorder.payloads("calibration.stage.end")[2]["steps"]) == 4
    (end,) = recorder.payloads("calibration.end")
    assert isinstance(end["parameters"], dict)
