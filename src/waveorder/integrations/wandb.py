"""Log waveorder's events to a Weights & Biases run: :class:`WandbObserver`.

::

    run = wandb.init(project="reconstruction")
    with waveorder.observe(WandbObserver(run), iterations=True):
        model.reconstruct(data, pixel_size=pixel_size)

``run.start`` updates the run's config. ``calibration.stage.end`` and a recorded ``solve.end`` history go to
``run.log`` (``calibration/objective``, ``calibration/steps/<index>.<stage>``, ``solve/history``); ``solve.end``,
``calibration.end``, ``tiles.end`` and ``run.end`` go to the summary under ``solve/``, ``calibration/``, ``tiles/``
and ``run/<method>/``, nested keys joined by ``/``. ``run.log`` never passes ``step``, so the run's step stays
monotone. ``wandb`` is imported only to build a plot; install it with ``pip install waveorder[wandb]``.
"""

from collections.abc import Mapping
from typing import Protocol

__all__ = ["WandbObserver"]


class _Config(Protocol):
    def update(self, d: Mapping[str, object], allow_val_change: bool | None = None) -> None: ...  # noqa: FBT001


class _Summary(Protocol):
    def update(self, d: dict[str, object]) -> None: ...


class RunLike(Protocol):
    """The part of a ``wandb`` run the observer uses: ``config.update``, ``summary.update`` and ``log``."""

    @property
    def config(self) -> _Config:
        """The run's configuration."""
        ...

    @property
    def summary(self) -> _Summary:
        """The run's summary."""
        ...

    def log(self, data: dict[str, object]) -> None:
        """Log one step of values."""
        ...


_SUMMARIZED = {"calibration.end": "calibration", "tiles.end": "tiles"}


class WandbObserver:
    """An :class:`~waveorder.observe.Observer` that logs to a W&B run (see the module docstring for the mapping).

    Parameters
    ----------
    run : RunLike
        The run, e.g. from ``wandb.init()``.
    """

    def __init__(self, run: RunLike) -> None:
        self.run = run

    def event(self, name: str, payload: Mapping[str, object]) -> None:
        """Log one waveorder event to the run."""
        if name == "run.start":
            self.run.config.update(dict(payload), allow_val_change=True)
        elif name == "calibration.stage.end":
            self._stage(payload)
        elif name == "solve.end":
            self._solve(payload)
        elif name == "run.end":
            fields = {key: value for key, value in payload.items() if key != "method"}
            self.run.summary.update(_flattened(f"run/{payload.get('method')}", fields))
        elif name in _SUMMARIZED:
            self.run.summary.update(_flattened(_SUMMARIZED[name], payload))

    def _stage(self, payload: Mapping[str, object]) -> None:
        data: dict[str, object] = {
            "calibration/objective": payload.get("objective"),
            "calibration/stage": payload.get("index"),
        }
        steps = payload.get("steps")
        if isinstance(steps, list):
            key = f"calibration/steps/{payload.get('index')}.{payload.get('stage')}"
            data[key] = _line(steps, x="evaluation", y="objective", title=key)
        self.run.log(data)

    def _solve(self, payload: Mapping[str, object]) -> None:
        fields = {key: value for key, value in payload.items() if key != "history"}
        self.run.summary.update(_flattened("solve", fields))
        history = payload.get("history")
        if isinstance(history, list):
            title = f"{payload.get('solver')} relative change per iteration"
            self.run.log({"solve/history": _line(history, x="iteration", y="relative_change", title=title)})


def _line(values: list[object], *, x: str, y: str, title: str) -> object:
    """Return a W&B line plot of ``values`` against their 1-based index."""
    import wandb

    table = wandb.Table(columns=[x, y], data=[[index, value] for index, value in enumerate(values, start=1)])
    return wandb.plot.line(table, x, y, title=title)


def _flattened(prefix: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Return ``payload`` with ``prefix/`` keys, nested mappings joined by ``/``."""
    flat: dict[str, object] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}"
        if isinstance(value, Mapping):
            flat |= _flattened(name, value)
        else:
            flat[name] = value
    return flat
