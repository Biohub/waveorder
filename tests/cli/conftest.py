import logging
from pathlib import Path

import lightning.pytorch  # noqa: F401, TID253 (its log handlers keep the sys.stderr of their import: not a CliRunner's)
import pytest
import torch

from tests.cli.plates import PHASE, intensities, simulate_phase, write_plate
from waveorder.cli.options import PROGRESS_LOGGER


@pytest.fixture(autouse=True)
def _restore_logging():
    """Commands configure the root, ``waveorder``, progress and ``iohub`` loggers; undo that after each test."""
    loggers = [logging.getLogger(name) for name in ("", "waveorder", PROGRESS_LOGGER, "iohub")]
    saved = [(logger.level, logger.propagate, list(logger.handlers)) for logger in loggers]
    yield
    for logger, (level, propagate, handlers) in zip(loggers, saved, strict=True):
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers


@pytest.fixture(scope="session")
def phase_volume() -> torch.Tensor:
    """Contrast of the simulated object (what the model measures), ``(Z, Y, X)``."""
    return simulate_phase()


@pytest.fixture
def phase_plate(tmp_path, phase_volume) -> Path:
    """A plate of one position holding brightfield intensities, as a camera records them (not contrast)."""
    write_plate(tmp_path / "in.zarr", [intensities(phase_volume)[None]])
    return tmp_path / "in.zarr"


@pytest.fixture
def phase_config(tmp_path) -> Path:
    path = tmp_path / "phase.yml"
    path.write_text(PHASE)
    return path
