"""The module layer: ``g = H f + n`` as ``torch.nn`` modules, plus the solvers and scores that use them.

Every module takes its sampling per call, so one model serves tiles and volumes of any shape. Free quantities are
``nn.Parameter`` objects named by dotted path (``system.detection.numerical_aperture``). A system bound to a sampling
is a :class:`LinearOperator`.

The names a user needs are importable from here; helpers for writing a new system, prior, noise model, solver, score
or search stage are imported from the file that defines them. :mod:`waveorder.models` builds on this package; nothing
here imports it.
"""

from waveorder.nn.context import SolveContext, current_context, solve_context
from waveorder.nn.model import ImagingModel
from waveorder.nn.noise import Gaussian, Noise, Poisson, PoissonGaussian
from waveorder.nn.operators import FourierDiagonal, LinearOperator
from waveorder.nn.optics import Detection, Illumination, IlluminationTilt, PupilEdge, ZernikeAberration
from waveorder.nn.parameters import (
    CalibratableModule,
    Free,
    ParameterPrior,
    Quantity,
    evaluate,
    free_specs,
    load_parameters,
)
from waveorder.nn.preprocess import Focus, TransverseBand, ZPadding, brightfield_contrast, focus_from_transverse_band
from waveorder.nn.priors import GaussianObjectPrior, NoPrior, ObjectPrior, Tikhonov, TotalVariation
from waveorder.nn.sampling import ZYX, Sampling, Status
from waveorder.nn.scores import (
    GCV,
    Evidence,
    LaplacianVarianceScore,
    MidbandPowerScore,
    NormalizedVarianceScore,
    Score,
    SpectralFlatnessScore,
    TotalVariationScore,
)
from waveorder.nn.search import (
    CalibrationObjective,
    CalibrationResult,
    GradientSearch,
    GridSearch,
    LBFGSSearch,
    NelderMeadSearch,
    RandomSearch,
    SearchStage,
    calibrate,
    score_candidates,
)
from waveorder.nn.solvers import (
    ADMM,
    FISTA,
    Distilled,
    IterativeSolver,
    Loop,
    RichardsonLucy,
    Solver,
    SolveResult,
    Wiener,
)
from waveorder.nn.solvers.wiener import GaussianPosterior
from waveorder.nn.systems.base import CacheMode, ImagingSystem, ShiftInvariantSystem
from waveorder.nn.systems.fluorescence import ConfocalFluorescence3D, ThinFluorescence2D, WidefieldFluorescence3D
from waveorder.nn.systems.phase import ThickPhase3D, ThinPhase2D
from waveorder.nn.systems.tiled import TileCalibration, calibrate_tiles
from waveorder.nn.verbs import InverseProblem, Verb

__all__ = [
    "ADMM",
    "FISTA",
    "GCV",
    "ZYX",
    "CacheMode",
    "CalibratableModule",
    "CalibrationObjective",
    "CalibrationResult",
    "ConfocalFluorescence3D",
    "Detection",
    "Distilled",
    "Evidence",
    "Focus",
    "FourierDiagonal",
    "Free",
    "Gaussian",
    "GaussianObjectPrior",
    "GaussianPosterior",
    "GradientSearch",
    "GridSearch",
    "Illumination",
    "IlluminationTilt",
    "ImagingModel",
    "ImagingSystem",
    "InverseProblem",
    "IterativeSolver",
    "LBFGSSearch",
    "LaplacianVarianceScore",
    "LinearOperator",
    "Loop",
    "MidbandPowerScore",
    "NelderMeadSearch",
    "NoPrior",
    "Noise",
    "NormalizedVarianceScore",
    "ObjectPrior",
    "ParameterPrior",
    "Poisson",
    "PoissonGaussian",
    "PupilEdge",
    "Quantity",
    "RandomSearch",
    "RichardsonLucy",
    "Sampling",
    "Score",
    "SearchStage",
    "ShiftInvariantSystem",
    "SolveContext",
    "SolveResult",
    "Solver",
    "SpectralFlatnessScore",
    "Status",
    "ThickPhase3D",
    "ThinFluorescence2D",
    "ThinPhase2D",
    "Tikhonov",
    "TileCalibration",
    "TotalVariation",
    "TotalVariationScore",
    "TransverseBand",
    "Verb",
    "WidefieldFluorescence3D",
    "Wiener",
    "ZPadding",
    "ZernikeAberration",
    "brightfield_contrast",
    "calibrate",
    "calibrate_tiles",
    "current_context",
    "evaluate",
    "focus_from_transverse_band",
    "free_specs",
    "load_parameters",
    "score_candidates",
    "solve_context",
]
