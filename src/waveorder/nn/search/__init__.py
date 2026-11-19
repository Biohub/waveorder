"""Calibration: :func:`calibrate`, the objective it minimizes, and the search stages that minimize it.

:mod:`~waveorder.nn.search.base` holds :func:`calibrate` and the :class:`SearchStage` protocol. The stages are
:class:`GridSearch` and :class:`RandomSearch` (batched candidates), :class:`GradientSearch` and :class:`LBFGSSearch`
(descent), :class:`NelderMeadSearch` (gradient-free, on the host) and :class:`WeightSweep` (object weight by a rule
on reconstructions). :func:`calibrate_beads` fits the system to a volume of beads.
"""

from waveorder.nn.search.base import (
    CalibrationObjective,
    CalibrationResult,
    Estimate,
    SearchStage,
    SweepResult,
    calibrate,
    expect_known_names,
    on_bounds,
    score_candidates,
)
from waveorder.nn.search.beads import BeadCalibration, calibrate_beads
from waveorder.nn.search.descent import GradientSearch, LBFGSSearch, Optimizer
from waveorder.nn.search.grid import GridSearch, RandomSearch, keep_best, search_spans
from waveorder.nn.search.nelder_mead import NelderMeadSearch
from waveorder.nn.search.sweep import WeightSweep

__all__ = [
    "BeadCalibration",
    "CalibrationObjective",
    "CalibrationResult",
    "Estimate",
    "GradientSearch",
    "GridSearch",
    "LBFGSSearch",
    "NelderMeadSearch",
    "Optimizer",
    "RandomSearch",
    "SearchStage",
    "SweepResult",
    "WeightSweep",
    "calibrate",
    "calibrate_beads",
    "expect_known_names",
    "keep_best",
    "on_bounds",
    "score_candidates",
    "search_spans",
]
