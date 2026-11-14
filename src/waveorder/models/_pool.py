"""Reconstructing (position, time) units on devices: in the calling process, or in ``spawn`` worker processes.

Each worker unpickles the model once, takes the next device and its share of the CPUs, ignores Ctrl-C and dies with
its parent (Linux). What a unit emits in a worker (events, warnings, log records) is re-emitted by the parent.
"""

import contextlib
import ctypes
import logging
import multiprocessing
import multiprocessing.queues
import os
import pickle
import signal
import sys
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import torch

from waveorder.errors import ConfigurationError, ReconstructionFailed, ReconstructionReport, Rejected, UnitResult, named
from waveorder.models._base import ReconstructionModel
from waveorder.models._positions import Unit, UnitRunner, Work

log = logging.getLogger(__name__)

_LOGGERS = ("", "waveorder", "iohub")
"""Loggers whose levels a worker takes from the parent (``""`` is the root)."""

_PR_SET_PDEATHSIG = 1

_UNPICKLABLE = (pickle.PicklingError, TypeError, AttributeError)

_warned: dict[str | tuple[str, type[Warning], int], int] = {}
"""The parent's registry of warnings relayed from workers, so each is shown once per location across workers."""


@dataclass(frozen=True)
class Workers:
    """The device of each worker, and whether the units run in the calling process."""

    devices: tuple[str, ...]
    in_process: bool


def plan_workers(model: ReconstructionModel, devices: object, workers: object, *, units: int) -> Workers:
    """Turn the ``devices`` and ``workers`` arguments into one device per worker, never more workers than units.

    Raises Rejected at the argument for a device not here, no device, or a worker count that is no positive integer.
    """
    from waveorder.nn.parameters import resolve_device, same_device

    argument = named("devices")
    if isinstance(devices, str) and devices == "auto":
        count = torch.cuda.device_count() if model.device.type == "cuda" else 0
        names = [f"cuda:{index}" for index in range(count)] or [str(model.device)]
    else:
        listed = [devices] if isinstance(devices, str | torch.device) else list(devices)  # ty: ignore[invalid-argument-type]
        if not listed or not all(isinstance(device, str | torch.device) for device in listed):
            raise Rejected.one(
                f"{argument} must be auto or a list of devices such as cuda:0, got {devices!r}", location=argument
            )
        names = [str(device) for device in listed]
        for name in names:
            try:
                resolve_device(name, field=argument)
            except ConfigurationError as error:
                raise Rejected.one(str(error).removeprefix(f"{error.field}: "), location=argument) from None
    if workers == "auto":
        count = len(names)
    elif isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1:
        count = workers
    else:
        argument = named("workers")
        raise Rejected.one(
            f"{argument} must be auto or a positive number of workers, got {workers!r}", location=argument
        )
    count = max(1, min(count, units))
    assigned = tuple(names[index % len(names)] for index in range(count))
    return Workers(devices=assigned, in_process=count == 1 and same_device(assigned[0], model.device))


def pickled_model(model: ReconstructionModel) -> bytes:
    """Return ``(class name, pickled model)`` pickled on the CPU, as worker processes receive it; the model stays put.

    The model keeps no transfer function afterwards, and on CUDA this process gives its cached device memory back,
    so the workers find their devices free. Raises Rejected at ``workers``, naming the cause (the model's
    configuration when that fails), if it cannot pickle.
    """
    device = model.device
    try:
        model.to("cpu")
        name = f"{type(model).__module__}.{type(model).__qualname__}"
        return pickle.dumps((name, pickle.dumps(model)))
    except _UNPICKLABLE as error:
        cause = f"{type(error).__name__}: {error}"
        fix = "define its parts at module level (no lambdas or local classes)"
        if not _pickles(getattr(model, "config", None)):
            cause = f"its configuration, model.config, cannot be pickled: {cause}"
            fix = (
                "the configuration names something a worker cannot rebuild by name (an option class made while the "
                "configuration was read): build the model from its parts in Python"
            )
        raise Rejected.one(
            f"worker processes receive the model pickled, and this {type(model).__name__} cannot be pickled "
            f"({cause}); {fix}, or run one worker on the model's device",
            location=named("workers"),
        ) from None
    finally:
        model.to(device)  # moving the system drops its cached transfer function
        if (stored := model._stored_transfers) is not None:  # noqa: SLF001 (the transfer a loaded store last gave)
            stored.last = None
        if device.type == "cuda":  # the checks' freed transfer stays reserved here unless the allocator gives it back
            torch.cuda.empty_cache()


def _pickles(value: object) -> bool:
    try:
        pickle.dumps(value)
    except _UNPICKLABLE:
        return False
    return True


def affinity_threads(processes: int, *, limit: int | None = None) -> int:
    """Torch threads per process: the CPUs this process may use, shared among ``processes``, at most ``limit``."""
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    share = max(1, usable // max(1, processes))
    return share if limit is None else max(1, min(share, limit))


# --- running the units -----------------------------------------------------------------------------------------------


def run_units(  # noqa: PLR0913 (the model, the units and what a run shares)
    model: ReconstructionModel,
    units: Sequence[Unit],
    *,
    work: Work,
    workers: Workers,
    payload: bytes | None,
    keep_going: bool,
    report: ReconstructionReport,
) -> None:
    """Reconstruct ``units`` here or in worker processes; add each result to ``report`` as it ends.

    Emits ``units.start`` and one ``unit.end`` per unit to the observers attached in this process. Raises
    ReconstructionFailed for a unit's failure without ``keep_going``, or a worker that could not start or died.
    """
    from waveorder.observe import context, emit, observing

    if observing():
        emit(
            "units.start",
            {
                "output": work.output,
                "total": len(units),
                "positions": len({unit.position for unit in units}),
                "devices": list(dict.fromkeys(workers.devices)),
                "workers": len(workers.devices),
            },
        )
    if not units:
        return
    if not workers.in_process:
        assert payload is not None
        _run_workers(units, work=work, workers=workers, payload=payload, keep_going=keep_going, report=report)
        return
    runner = UnitRunner(model, work)
    for unit in units:
        with context(unit=unit.name):
            result = runner.run(unit)
        finish(report, result)
        if result.status == "failed" and not keep_going:
            raise _stop(report, f"unit {unit.name} failed: {result.error}")


def finish(report: ReconstructionReport, result: UnitResult) -> None:
    """Record a unit that ended in ``report``, log a failure and emit ``unit.end``."""
    from waveorder.observe import emit, observing

    report.results.append(result)
    if result.status == "failed":
        log.error("unit %s failed: %s", result.unit, result.error)
    if observing():
        emit("unit.end", result.to_json())


def _stop(report: ReconstructionReport, message: str) -> ReconstructionFailed:
    report.status = "failed"
    return ReconstructionFailed(message, report=report)


# --- the worker process ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Setup:
    """What every worker needs: the model, the work, its threads, what to forward, the log levels, the parent pid."""

    model: bytes
    work: Work
    threads: int
    events: bool
    iterations: bool
    levels: dict[str, int]
    parent: int


_runner: UnitRunner | None = None
_worker = 0
_fatal: str | None = None
_events = (False, False)


def _start(setup: _Setup, slots: multiprocessing.queues.Queue[tuple[int, str]]) -> None:
    """Worker initializer: die with the parent, ignore Ctrl-C, take a device and build the runner on it.

    A failure is kept and reported with every unit, since a pool says nothing of an initializer's error.
    """
    global _runner, _worker, _fatal, _events
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if os.getppid() != setup.parent:  # the parent died before the request took effect
        os._exit(0)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import iohub  # noqa: F401 (configures its logger on import, which is undone here)

    logging.getLogger("iohub").handlers.clear()
    for name, level in setup.levels.items():
        logging.getLogger(name).setLevel(level)
    _events = (setup.events, setup.iterations)
    _worker, device = slots.get()
    try:
        _runner = _build(setup, device)
    except BaseException as error:  # noqa: BLE001 (reported to the parent with every unit)
        _fatal = f"worker {_worker} on {device} could not start: {type(error).__name__}: {error}"


def _build(setup: _Setup, device: str) -> UnitRunner:
    """Unpickle the model onto ``device``; raise RuntimeError naming a class this process cannot import."""
    name, pickled = pickle.loads(setup.model)  # the parent's own model, sent to its own child
    try:
        model = pickle.loads(pickled)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"the model {name} cannot be rebuilt in a worker process ({type(error).__name__}: {error}); a worker is "
            "a new Python process, so the module of the model's class and of each of its parts must be importable "
            "there: put it in an installed package or on sys.path (not in a script's __main__ or a notebook cell), "
            "or run one worker on the model's device"
        ) from None
    model.to(device)
    torch.set_num_threads(setup.threads)
    return UnitRunner(model, setup.work)


class _Captured(logging.Handler):
    """What one unit emits in a worker: its events and log records, to be re-emitted by the parent."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, object]]] = []
        self.records: list[dict[str, object]] = []

    def event(self, name: str, payload: Mapping[str, object]) -> None:
        """Keep one :mod:`waveorder.observe` event."""
        self.events.append((name, dict(payload)))

    def emit(self, record: logging.LogRecord) -> None:
        """Keep one log record as the fields :func:`logging.makeLogRecord` takes."""
        fields = dict(record.__dict__) | {"msg": record.getMessage(), "args": None, "exc_info": None}
        if record.exc_info:
            fields["exc_text"] = logging.Formatter().formatException(record.exc_info)
        self.records.append(fields | {"stack_info": None})


type _Shown = list[tuple[Warning, str, int]]
type _Answer = tuple[dict[str, object], list[tuple[str, dict[str, object]]], _Shown, list[dict[str, object]]]


def _unit(position: str, time: int) -> _Answer | str:
    """Reconstruct one unit in a worker; return its result with what it emitted, or why the worker cannot run."""
    from waveorder.observe import context, observe

    if _fatal is not None or _runner is None:
        return _fatal or "the worker has no model"
    unit = Unit(position=position, time=time)
    captured = _Captured()
    root = logging.getLogger()
    root.addHandler(captured)
    events, iterations = _events
    try:
        with warnings.catch_warnings(record=True) as seen, contextlib.ExitStack() as stack:
            warnings.simplefilter("default")
            if events:
                stack.enter_context(observe(captured, iterations=iterations))
            with context(unit=unit.name, worker=_worker):
                result = _runner.run(unit)
    finally:
        root.removeHandler(captured)
    shown = [
        (item.message, item.filename, item.lineno)
        for item in seen
        if isinstance(item.message, Warning) and _pickles(item.message)
    ]
    return result.to_json(), captured.events, shown, captured.records


# --- the parent's side -----------------------------------------------------------------------------------------------


def _run_workers(
    units: Sequence[Unit],
    *,
    work: Work,
    workers: Workers,
    payload: bytes,
    keep_going: bool,
    report: ReconstructionReport,
) -> None:
    """Run ``units`` on spawned workers, each taking the next unit when free; stop once the running units end."""
    from waveorder.observe import observing, recording

    context = multiprocessing.get_context("spawn")
    slots = context.Queue()
    for index, device in enumerate(workers.devices):
        slots.put((index, device))
    setup = _Setup(
        model=payload,
        work=work,
        threads=affinity_threads(len(workers.devices), limit=torch.get_num_threads()),
        events=observing(),
        iterations=recording(),
        levels={name: logging.getLogger(name).level for name in _LOGGERS},
        parent=os.getpid(),
    )
    pool = ProcessPoolExecutor(len(workers.devices), mp_context=context, initializer=_start, initargs=(setup, slots))
    failure: str | None = None
    try:
        futures: dict[Future[_Answer | str], Unit] = {pool.submit(_unit, u.position, u.time): u for u in units}
        for future in as_completed(futures):
            if not future.cancelled():
                failure = failure or _answered(future, report, keep_going=keep_going)
            if failure is not None:
                for pending in futures:
                    pending.cancel()
    except BaseException:
        for process in list(getattr(pool, "_processes", {}).values()):
            process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    if failure is not None:
        raise _stop(report, failure)


def _answered(future: Future[_Answer | str], report: ReconstructionReport, *, keep_going: bool) -> str | None:
    """Re-emit what a worker's unit emitted and take its result; return why the run stops, if it does."""
    from waveorder.observe import emit

    try:
        answer = future.result()
    except BrokenProcessPool:
        return _died(report)
    if isinstance(answer, str):
        return answer
    fields, events, shown, records = answer
    for entry in records:
        record = logging.makeLogRecord(entry)
        logging.getLogger(record.name).handle(record)
    for warning, filename, lineno in shown:
        warnings.warn_explicit(warning, type(warning), filename, lineno, registry=_warned)
    for name, payload in events:
        emit(name, payload)
    result = UnitResult(**fields)  # ty: ignore[invalid-argument-type]
    finish(report, result)
    if result.status == "failed" and not keep_going:
        return f"unit {result.unit} failed: {result.error}"
    return None


def _died(report: ReconstructionReport) -> str:
    """Return what to say about a worker that exited, with the usual causes when no unit had ended yet."""
    text = "a worker process exited while reconstructing, which stops the run"
    if report.results:
        return text
    return (
        f"{text}, before any unit ended. A worker is a new Python process that imports the calling script and the "
        "modules of the model's classes: guard the script's entry point with `if __name__ == \"__main__\":` (an "
        "unguarded script runs again in every worker, which Python refuses), and keep the model's classes in an "
        "importable module (an installed package or one on sys.path, not the script itself or a notebook cell); "
        f"or run one worker on the model's device ({named('workers')}=1)"
    )
