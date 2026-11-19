"""What a running command shows of its model: the progress display, the calibration lines and the table of a store.

The commands import this module in their bodies: it imports the store, and so torch.
"""

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from waveorder.cli.options import PROGRESS_LOGGER
from waveorder.cli.render import RunDisplay
from waveorder.io.tf_store import TfStore

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


def calibrating(*, position: str, time_index: int) -> None:
    """Say that the command calibrates on ``position`` at ``time_index``."""
    log.info("calibrating on %s at t=%s ...", position, time_index)


def calibrated(store: TfStore, *, path: Path) -> None:
    """Show the values a calibration found."""
    table = "\n".join(f"  {line}" for line in parameter_table(store))
    log.info("calibrated %s:\n%s", path, table)


@contextmanager
def progress() -> Generator[RunDisplay]:
    """Show a run's progress on stderr (a bar on a terminal, else lines) while the block runs."""
    display = RunDisplay(bar=sys.stderr.isatty())
    try:
        yield display
    finally:
        display.close()
