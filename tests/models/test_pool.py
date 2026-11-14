"""Units across devices and worker processes: the plan of the workers, forwarding, failures and a dead worker."""

import logging
import multiprocessing
import os
import sys
from pathlib import Path

import pytest
import torch

from tests.models.calibration_data import PIXEL_SIZE, SHAPE, intensities, model_of, simulate_phase
from tests.models.plates import contrast, fluorescence_model, phase_model, read_output, series, write_plate
from waveorder.errors import Rejected
from waveorder.models import Fluorescence3D, ReconstructionFailed
from waveorder.models._pool import Workers, affinity_threads, pickled_model, plan_workers
from waveorder.nn import Detection, Gaussian, ShiftInvariantSystem, Tikhonov
from waveorder.observe import observe

BRIGHT = 1e6
ROOT = Path(__file__).resolve().parents[2]


class Fragile(Fluorescence3D):
    """Fluorescence that refuses data brighter than ``BRIGHT`` (a unit that fails in whichever process runs it)."""

    prepare_scope = "block"  # it looks at the whole block it is given

    def prepare(self, data, *, mean=None):
        del mean
        if float(data.max()) > BRIGHT:
            logging.getLogger("waveorder.test").warning("refusing a bright block")
            raise ValueError("too bright")
        return data


class Dying(Fluorescence3D):
    """Fluorescence whose process exits on data brighter than ``BRIGHT`` (a worker that dies mid-unit)."""

    prepare_scope = "block"

    def prepare(self, data, *, mean=None):
        del mean
        if float(data.max()) > BRIGHT:
            os._exit(7)
        return data


class Stillborn(Fluorescence3D):
    """Fluorescence whose worker process exits while it rebuilds the model (a worker that dies before any unit)."""

    def __setstate__(self, state: dict) -> None:
        if multiprocessing.parent_process() is not None:  # in a worker
            os._exit(3)
        vars(self).update(state)


def _model(kind: type[Fluorescence3D]) -> Fluorescence3D:
    return kind(
        Detection(wavelength=0.52, numerical_aperture=0.8),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(weight=1e-3),
        noise=Gaussian(sigma=0.01),
    )


class _Events:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    def event(self, name, payload) -> None:
        self.seen.append((name, dict(payload)))


@pytest.fixture(scope="module")
def volume() -> torch.Tensor:
    return contrast()


@pytest.fixture
def plate(tmp_path, volume):
    """One position of four time points."""
    path = tmp_path / "in.zarr"
    write_plate(path, [series(volume, 4)])
    return path


@pytest.fixture
def bright_plate(tmp_path, volume, monkeypatch):
    """One position of four time points whose time 1 is brighter than ``BRIGHT``, for models defined in this module.

    Those models are unpickled by reference in each worker, so the repository root goes on ``sys.path``, which
    spawned workers inherit: under pytest's importlib import mode the test process imports ``tests`` without it.
    """
    monkeypatch.syspath_prepend(str(ROOT))
    stacks = series(volume, 4)
    stacks[1] *= 1e4
    path = tmp_path / "bright.zarr"
    write_plate(path, [stacks])
    return path


def test_workers_are_planned_from_the_devices_and_the_count():
    model = phase_model()
    assert plan_workers(model, "auto", "auto", units=6) == Workers(devices=("cpu",), in_process=True)
    assert plan_workers(model, ["cpu"], 1, units=6).in_process
    assert plan_workers(model, torch.device("cpu"), "auto", units=6).in_process
    assert plan_workers(model, ["cpu"], 3, units=6) == Workers(devices=("cpu", "cpu", "cpu"), in_process=False)
    assert plan_workers(model, ["cpu"], 3, units=2).devices == ("cpu", "cpu")  # never more workers than units
    assert plan_workers(model, ["cpu"], 3, units=0) == Workers(devices=("cpu",), in_process=True)
    for workers in (0, -1, 1.5, True, "two"):
        with pytest.raises(Rejected, match="workers must be auto or a positive number") as error:
            plan_workers(model, "auto", workers, units=6)
        assert error.value.problems[0].location == "workers"
    for devices in ([], [0], "cuda:99", ["cpu", "nowhere"]):
        with pytest.raises(Rejected) as error:
            plan_workers(model, devices, "auto", units=6)
        assert error.value.problems[0].location == "devices"
        assert str(error.value).count("devices:") == 1  # the location, not repeated by the message
    assert affinity_threads(10**6) == 1
    assert affinity_threads(1) >= 1
    assert affinity_threads(1, limit=1) == 1  # never more than the calling process's own threads
    assert affinity_threads(1, limit=10**6) == affinity_threads(1)
    assert affinity_threads(2, limit=0) == 1


class Counting(Fluorescence3D):
    """Fluorescence that logs the torch threads of the process that prepares its data."""

    prepare_scope = "voxel"

    def prepare(self, data, *, mean=None):
        del mean
        logging.getLogger("waveorder.test").warning("torch threads: %d", torch.get_num_threads())
        return data


def test_workers_take_no_more_threads_than_the_calling_process(plate, tmp_path, caplog, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))  # the workers import this module for ``Counting``
    before = torch.get_num_threads()
    torch.set_num_threads(1)  # as OMP_NUM_THREADS=1 would
    try:
        with caplog.at_level(logging.WARNING, logger="waveorder"):
            report = _model(Counting).reconstruct_positions(
                plate, output=tmp_path / "out.zarr", channels=["BF"], workers=2
            )
    finally:
        torch.set_num_threads(before)
    assert report.count("complete") == 4
    counted = {record.getMessage() for record in caplog.records if "torch threads" in record.getMessage()}
    assert counted == {"torch threads: 1"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_auto_devices_of_a_cuda_model_are_every_visible_gpu():
    model = phase_model(device="cuda")
    count = torch.cuda.device_count()
    planned = plan_workers(model, "auto", "auto", units=64)
    assert planned.devices == tuple(f"cuda:{index}" for index in range(count))
    assert planned.in_process == (count == 1)
    assert plan_workers(model, "auto", count + 1, units=64).devices[-1] == "cuda:0"  # round-robin
    assert not plan_workers(phase_model(), ["cuda:0"], "auto", units=64).in_process  # not the model's device


def test_two_cpu_workers_write_what_one_process_writes(plate, tmp_path):
    model = phase_model()
    parallel = model.reconstruct_positions(
        plate, output=tmp_path / "two.zarr", channels=["BF"], devices=["cpu"], workers=2
    )
    assert (parallel.status, parallel.count("complete")) == ("finished", 4)
    assert sorted(result.unit for result in parallel.results) == [f"A/1/0/t{t}" for t in range(4)]
    assert all(result.wall_s > 0 and result.host and result.device == "cpu" for result in parallel.results)
    model.reconstruct_positions(plate, output=tmp_path / "one.zarr", channels=["BF"])
    _, a, _ = read_output(tmp_path / "two.zarr")
    _, b, _ = read_output(tmp_path / "one.zarr")
    assert torch.allclose(a, b, atol=1e-6)
    assert model.device == torch.device("cpu")  # pickling for the workers left the model where it was


def test_a_model_that_cannot_be_pickled_is_refused_before_anything_is_written(plate, tmp_path):
    model = fluorescence_model()
    model.solver.__dict__["note"] = lambda: None  # a local function: nothing a worker process could receive
    output = tmp_path / "out.zarr"
    with pytest.raises(Rejected, match="cannot be pickled") as error:
        model.reconstruct_positions(plate, output=output, channels=["BF"], workers=2)
    (problem,) = error.value.problems
    assert problem.location == "workers"
    assert "worker processes receive the model pickled, and this Fluorescence3D cannot be pickled (" in problem.message
    assert "run one worker on the model's device" in problem.message
    assert not output.exists()
    assert model.reconstruct_positions(plate, output=output, channels=["BF"]).count("complete") == 4  # in process
    with pytest.raises(Rejected, match="cannot be pickled"):
        pickled_model(model)


def test_a_configuration_that_cannot_be_pickled_is_named_as_the_cause():
    model = model_of()
    assert pickled_model(model)
    object.__setattr__(model.config, "note", lambda: None)  # what a worker could not rebuild, inside the config
    with pytest.raises(Rejected, match=r"its configuration, model\.config, cannot be pickled") as error:
        pickled_model(model)
    (problem,) = error.value.problems
    assert problem.location == "workers"
    assert "build the model from its parts in Python, or run one worker on the model's device" in problem.message


def _cached_model(device: str, store: Path | None = None):
    """The phase model on ``device`` after a reconstruction with the transfer cache on (and ``store`` loaded)."""
    model = model_of().to(device)
    if store is not None:
        model.load_calibration(store)
    system = model.system
    assert isinstance(system, ShiftInvariantSystem)
    system.cache = "auto"
    model.reconstruct(intensities(simulate_phase()).to(device))
    assert system._cache_entry is not None  # noqa: SLF001
    return model


def test_a_pickled_model_keeps_no_transfer_function(tmp_path):
    store = model_of().to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE).save(tmp_path / "tf.zarr")
    model = _cached_model("cpu", store)
    stored = model._stored_transfers  # noqa: SLF001
    assert stored is not None
    assert stored.last is not None  # the transfer the store gave
    assert pickled_model(model)
    assert model.system._cache_entry is None  # noqa: SLF001
    assert stored.last is None
    assert model.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_pickled_cuda_model_gives_its_cached_device_memory_back():
    model = _cached_model("cuda")
    assert torch.cuda.memory_reserved() > 0
    assert pickled_model(model)
    assert model.device.type == "cuda"
    assert model.system._cache_entry is None  # noqa: SLF001
    assert torch.cuda.memory_reserved() < 64 * 2**20


def test_workers_forward_their_events_logs_and_failures_to_the_parent(bright_plate, tmp_path, caplog):
    model = _model(Fragile)
    events = _Events()
    with observe(events), caplog.at_level(logging.WARNING, logger="waveorder"):
        report = model.reconstruct_positions(
            bright_plate, output=tmp_path / "out.zarr", channels=["BF"], workers=2, keep_going=True
        )
    assert (report.status, report.count("complete"), report.count("failed")) == ("failed", 3, 1)
    (failed,) = [result for result in report.results if result.status == "failed"]
    assert (failed.unit, failed.error) == ("A/1/0/t1", "ValueError: too bright")
    assert "refusing a bright block" in caplog.text  # the worker's log record, re-emitted here
    assert "unit A/1/0/t1 failed: ValueError: too bright" in caplog.text
    start = dict(events.seen)["units.start"]
    assert (start["total"], start["workers"], start["devices"]) == (4, 2, ["cpu"])
    ends = [payload for name, payload in events.seen if name == "unit.end"]
    assert sorted(payload["unit"] for payload in ends) == [f"A/1/0/t{t}" for t in range(4)]
    forwarded = [payload for name, payload in events.seen if name == "solve.end"]
    assert sorted(payload["unit"] for payload in forwarded) == ["A/1/0/t0", "A/1/0/t2", "A/1/0/t3"]
    assert {payload["worker"] for payload in forwarded} <= {0, 1}
    _, image, _ = read_output(tmp_path / "out.zarr")
    assert image[1].abs().max() == 0
    assert image[3].abs().max() > 0


def test_a_failed_unit_in_a_worker_stops_the_run_with_the_report(bright_plate, tmp_path):
    with pytest.raises(ReconstructionFailed, match="unit A/1/0/t1 failed: ValueError: too bright") as error:
        _model(Fragile).reconstruct_positions(bright_plate, output=tmp_path / "out.zarr", channels=["BF"], workers=2)
    report = error.value.report
    assert report.status == "failed"
    assert report.count("failed") == 1
    assert 1 <= len(report.results) <= 4  # the units being reconstructed when it failed were finished, no more
    (failed,) = [result for result in report.results if result.status == "failed"]
    assert failed.error_type == "ValueError"
    ended = {result.unit for result in report.results}
    assert set(report.pending) == {f"A/1/0/t{t}" for t in range(4)} - ended  # the units that were not run


def test_a_model_class_a_worker_cannot_import_is_named_with_the_fix(plate, tmp_path, monkeypatch):
    elsewhere = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
    monkeypatch.setattr(sys, "path", elsewhere)  # the workers inherit it: ``tests`` is not importable there
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ReconstructionFailed, match="could not start") as error:
        _model(Fragile).reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"], workers=2)
    message = str(error.value)
    assert "the model tests.models.test_pool.Fragile cannot be rebuilt in a worker process" in message
    assert "No module named 'tests'" in message
    assert "must be importable there" in message
    assert error.value.report.results == []


def test_a_worker_that_dies_before_its_first_unit_is_reported_with_the_usual_causes(bright_plate, tmp_path):
    refused = r"a worker process exited while reconstructing, which stops the run, before any unit ended"
    with pytest.raises(ReconstructionFailed, match=refused) as error:
        _model(Stillborn).reconstruct_positions(bright_plate, output=tmp_path / "out.zarr", channels=["BF"], workers=2)
    message = str(error.value)
    assert 'guard the script\'s entry point with `if __name__ == "__main__":`' in message
    assert "keep the model's classes in an importable module" in message
    assert "workers=1" in message
    report = error.value.report
    assert (report.results, report.status) == ([], "failed")
    assert report.pending == tuple(f"A/1/0/t{t}" for t in range(4))  # nothing ran: every unit is to resubmit


def test_a_dead_worker_stops_the_run_even_with_keep_going(bright_plate, tmp_path):
    for keep_going in (False, True):
        with pytest.raises(ReconstructionFailed, match=r"a worker process exited while reconstructing") as error:
            _model(Dying).reconstruct_positions(
                bright_plate, output=tmp_path / "out.zarr", channels=["BF"], workers=2, keep_going=keep_going
            )
        report = error.value.report
        assert report.status == "failed"
        assert "A/1/0/t1" in report.pending  # the unit the dead worker held is to resubmit
