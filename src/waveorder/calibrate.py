"""Calibration: the scores, the search stages and the stores a calibration of a model's free quantities uses.

``model.calibrate(data)`` and ``model.calibrate_position(position, channels=...)`` run it on a model; these are the
parts to build one with and the functions under it.
"""

from waveorder.io.tf_store import TfStore
from waveorder.nn import (
    GCV,
    CalibrationResult,
    Evidence,
    Free,
    GradientSearch,
    GridSearch,
    LaplacianVarianceScore,
    LBFGSSearch,
    MidbandPowerScore,
    NelderMeadSearch,
    NormalizedVarianceScore,
    ParameterPrior,
    RandomSearch,
    SpectralFlatnessScore,
    TotalVariationScore,
    calibrate,
)
from waveorder.nn.priors import Beads
from waveorder.nn.scores import CentreSliceMidbandScore
from waveorder.nn.search import BeadCalibration, SweepResult, WeightSweep, calibrate_beads

__all__ = [
    "GCV",
    "BeadCalibration",
    "Beads",
    "CalibrationResult",
    "CentreSliceMidbandScore",
    "Evidence",
    "Free",
    "GradientSearch",
    "GridSearch",
    "LBFGSSearch",
    "LaplacianVarianceScore",
    "MidbandPowerScore",
    "NelderMeadSearch",
    "NormalizedVarianceScore",
    "ParameterPrior",
    "RandomSearch",
    "SpectralFlatnessScore",
    "SweepResult",
    "TfStore",
    "TotalVariationScore",
    "WeightSweep",
    "calibrate",
    "calibrate_beads",
]
