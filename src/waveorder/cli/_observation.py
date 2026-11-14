"""What a running command shows of its model: the progress display, the W&B run and the table of a store.

The commands import this module in their bodies: it imports the store, and so torch.
"""

import logging
import sys
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from waveorder.cli.options import PROGRESS_LOGGER
from waveorder.cli.render import RunDisplay
from waveorder.errors import Rejected
from waveorder.integrations.wandb import WandbObserver
from waveorder.io.tf_store import TfStore
from waveorder.observe import observe

log = logging.getLogger(PROGRESS_LOGGER)


def parameter_table(store: TfStore) -> list[str]:
    """Return a store's parameters as rows of name and value per tile, then the quantities on a bound."""
    if not store.parameter_names:
        return ["no free quantities"]
    tiled = store.tiles > 1
    width = max([len("quantity"), *(len(name) for name in store.parameter_names)])
    head = f"{'tile':>4}  " if tiled else ""
    lines = [f"{head}{'quantity':<{width}}  {'value':>12}"]
    for tile in range(store.tiles):
        prefix = f"{tile:>4}  " if tiled else ""
        for index, name in enumerate(store.parameter_names):
            lines.append(f"{prefix}{name:<{width}}  {float(store.parameters[tile, index]):>12.6g}")
    at_bound = store.provenance.get("at_bound")
    if isinstance(at_bound, list):
        for tile, names in enumerate(at_bound):
            where = f"tile {tile} " if tiled else ""
            lines.append(f"{where}at a bound: {', '.join(names) if names else 'none'}")
    return lines


class Observation:
    """The sinks of one command's events, an observer of :func:`waveorder.observe`: the display and the W&B run.

    The W&B run starts at the first event, so a refused request starts none; its group is the output store's name.
    Safe to call from several threads (a unit's reader and writer threads may emit). ``exit_code`` is what the W&B
    run finishes with; the command sets it from its report.
    """

    def __init__(self, *, project: str | None, job_type: str, store: str) -> None:
        self.display = RunDisplay(bar=sys.stderr.isatty())
        self.project = project
        self.job_type = job_type
        self.store_name = store
        self.exit_code = 0
        self._run: Any = None
        self._wandb: WandbObserver | None = None
        self._lock = threading.Lock()
        self._calibrating: dict[str, object] | None = None

    def event(self, name: str, payload: Mapping[str, object]) -> None:
        """Show one event of the model and log it to the W&B run."""
        with self._lock:
            if self._calibrating is not None:  # the calibration started: its checks passed
                self.display.event("tf.store", self._calibrating)
                self._calibrating = None
            self.display.event(name, payload)
            if self.project is not None:
                if self._wandb is None:
                    self._wandb = self._start()
                self._wandb.event(name, payload)

    def _start(self) -> WandbObserver:
        import wandb

        self._run = wandb.init(
            project=self.project,
            group=self.store_name,
            job_type=self.job_type,
            config={"waveorder": {"command": self.job_type, "store": self.store_name}},
        )
        return WandbObserver(self._run)

    def calibrating(self, *, position: str, time_index: int) -> None:
        """Say that the command calibrates on ``position`` at ``time_index``, once the model's first event comes."""
        with self._lock:
            self._calibrating = {"position": position, "time": time_index}

    def calibrated(self, store: TfStore, *, path: Path) -> None:
        """Show the values a calibration found."""
        table = "\n".join(f"  {line}" for line in parameter_table(store))
        log.info("calibrated %s:\n%s", path, table)

    def close(self, exit_code: int) -> None:
        """Stop the display and finish the W&B run with ``exit_code``."""
        self.display.close()
        finish = getattr(self._run, "finish", None)
        if callable(finish):
            finish(exit_code=exit_code)


def check_wandb(project: str | None) -> None:
    """Refuse ``--wandb`` without the ``wandb`` package (exit 2, before anything runs)."""
    if project is None:
        return
    try:
        import wandb  # noqa: F401 (only whether it imports)
    except ImportError:
        raise Rejected.one("needs the wandb package: pip install waveorder[wandb]", location="--wandb") from None


@contextmanager
def observing(project: str | None, *, job_type: str, store: str) -> Generator[Observation]:
    """Observe the block: progress on stderr and, with ``project`` (``--wandb``), a W&B run.

    On an exception the W&B run finishes with exit code 1; otherwise with :attr:`Observation.exit_code`.

    Parameters
    ----------
    project : str, optional
        The W&B project of ``--wandb``.
    job_type : str
        The command, the W&B job type.
    store : str
        The output store's name, the W&B group.
    """
    observation = Observation(project=project, job_type=job_type, store=store)
    code = 1
    try:
        with observe(observation):
            yield observation
        code = observation.exit_code
    finally:
        observation.close(code)
