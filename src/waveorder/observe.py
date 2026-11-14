"""Structured events from waveorder's operations, for experiment tracking: :class:`Observer` and :func:`observe`.

An observer is any object with ``event(name, payload)``; inside ``with observe(observer):`` each operation reports
what it did. Events: ``run.start`` and ``run.end`` around each verb of a model (resolved configuration, method,
device, versions); ``solve.end`` (solver, system, status, iterations); ``calibration.stage.end`` and
``calibration.end`` from :func:`~waveorder.nn.calibrate`; ``tiles.end`` (the :class:`~waveorder.tiling.TileStats`
fields); ``units.start`` and ``unit.end`` (the :class:`~waveorder.models.UnitResult` fields) from
:meth:`~waveorder.models.ReconstructionModel.reconstruct_positions`.

Payloads hold host values only, never tensors, and observing adds no device sync: device values are read once, after
the operation. Only the outermost operation reports (the solves inside a calibration or a tiled reconstruction report
nothing), and nothing reports inside a ``torch.func`` transform.
"""

import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = ["Observer", "observe"]


def capturing() -> bool:
    """Whether a ``torch.func`` transform is running here, so Python side effects may run several times or never."""
    return torch._C._functorch.peek_interpreter_stack() is not None  # noqa: SLF001 (torch has no public query)


@runtime_checkable
class Observer(Protocol):
    """Receives waveorder's events; see :mod:`waveorder.observe` for their names and payloads."""

    def event(self, name: str, payload: Mapping[str, object]) -> None:
        """Handle one event; ``payload`` holds host values only."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class _Scope:
    observers: tuple[Observer, ...]
    iterations: bool


@dataclass(frozen=True, slots=True)
class _Queued:
    name: str
    payload: dict[str, object]
    deferred: Mapping[str, Tensor]
    observers: tuple[Observer, ...]


_lock = threading.Lock()
_scopes: list[_Scope] = []
_observers: tuple[Observer, ...] = ()
_queue: list[_Queued] = []
_depth: ContextVar[int] = ContextVar("waveorder.observe.depth", default=0)
_steps: ContextVar[list[Tensor] | None] = ContextVar("waveorder.observe.steps", default=None)
_context: ContextVar[Mapping[str, object]] = ContextVar("waveorder.observe.context", default=MappingProxyType({}))

_recording = False
"""Whether an ``observe(..., iterations=True)`` block is open."""


@contextmanager
def observe(*observers: Observer, iterations: bool = False) -> Generator[None]:
    """Send waveorder's events to ``observers`` while the block runs.

    Observers are process-wide: nested blocks stack, and work on other threads inside the block is observed too. On
    exit, events that wait for device values are read back and delivered, in order.

    Parameters
    ----------
    *observers : Observer
        Objects with ``event(name, payload)``.
    iterations : bool
        Record the per-iteration history of every iterative solve and gradient stage on the device:
        ``SolveResult.history``, the ``"history"`` of ``solve.end`` and the ``"steps"`` of ``calibration.stage.end``.
    """
    scope = _Scope(tuple(observers), iterations)
    with _lock:
        _scopes.append(scope)
        _refresh()
    try:
        yield
    finally:
        with _lock:
            _scopes.remove(scope)
            _refresh()
        flush()


def _refresh() -> None:
    """Under ``_lock``: recompute the observers and the iteration flag from the open scopes."""
    global _observers, _recording
    _observers = tuple(observer for scope in _scopes for observer in scope.observers)
    _recording = any(scope.iterations for scope in _scopes)


def recording() -> bool:
    """Whether iterative solves and gradient stages record their per-iteration history."""
    return _recording


def observing() -> bool:
    """Whether an event emitted here reaches an observer: one is attached, nothing captures, and no operation nests."""
    return bool(_observers) and _depth.get() <= 1 and not capturing()


@contextmanager
def operation() -> Generator[None]:
    """Mark the body as one operation, so that operations run inside it report nothing of their own."""
    token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _depth.reset(token)


@contextmanager
def context(**values: object) -> Generator[None]:
    """Add ``values`` (host values, e.g. ``unit="A/1/0/t3"``) to every event this thread emits in the block.

    A key the event sets itself wins; nested blocks add to the outer block's values.
    """
    token = _context.set({**_context.get(), **values})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def collect_steps(*, on: bool) -> Generator[list[Tensor]]:
    """Collect what :func:`record_step` reports in the body into the yielded list, when ``on``; else nothing."""
    collected: list[Tensor] = []
    token = _steps.set(collected if on else None)
    try:
        yield collected
    finally:
        _steps.reset(token)


def record_step(objective: Tensor) -> None:
    """Add one evaluation's objective (detached, on the device) to the open :func:`collect_steps` list, if any."""
    collected = _steps.get()
    if collected is not None and not capturing():
        collected.append(objective.detach())


def emit(name: str, payload: Mapping[str, object], *, deferred: Mapping[str, Tensor] | None = None) -> None:
    """Send an event to every attached observer, if :func:`observing`.

    Parameters
    ----------
    name : str
        The event's name.
    payload : Mapping[str, object]
        Host values only; the open :func:`context` values are added under the keys it does not set.
    deferred : Mapping[str, Tensor], optional
        Device tensors added to the payload as lists by :func:`flush`; this event and every later one wait until then.
    """
    if not observing():
        return
    queued = _Queued(name, {**_context.get(), **payload}, deferred or {}, _observers)
    with _lock:
        if deferred or _queue:
            _queue.append(queued)
            return
    _deliver(queued)


def flush() -> None:
    """Read back the device values of waiting events, in one transfer per device, and deliver them in order."""
    with _lock:
        waiting = list(_queue)
        _queue.clear()
    if not waiting:
        return
    values = iter(_read_back([tensor for queued in waiting for tensor in queued.deferred.values()]))
    for queued in waiting:
        for key in queued.deferred:
            queued.payload[key] = next(values)
        _deliver(queued)


def _read_back(tensors: list[Tensor]) -> list[object]:
    """Return each tensor as a host list (a float for a 0-d tensor), reading each device once."""
    by_device: dict[torch.device, list[int]] = {}
    for index, tensor in enumerate(tensors):
        by_device.setdefault(tensor.device, []).append(index)
    host: list[object] = [None] * len(tensors)
    for indices in by_device.values():
        parts = [tensors[index].detach().reshape(-1).to(torch.float64) for index in indices]
        flat = torch.cat(parts).tolist()
        start = 0
        for index, part in zip(indices, parts, strict=True):
            size = part.numel()
            host[index] = flat[start] if tensors[index].dim() == 0 else flat[start : start + size]
            start += size
    return host


def _deliver(queued: _Queued) -> None:
    for observer in queued.observers:
        observer.event(queued.name, queued.payload)
