"""Simulation: the models whose ``simulate`` images an object, and the noise models that draw the data.

``model.simulate(obj, noise=True, seed=0)`` returns ``H f + n`` and ``f``; without an object it draws a power-law
Gaussian field.
"""

from waveorder.models import Fluorescence2D, Fluorescence3D, Phase2D, Phase3D, ReconstructionModel
from waveorder.nn import Gaussian, Noise, Poisson, PoissonGaussian

__all__ = [
    "Fluorescence2D",
    "Fluorescence3D",
    "Gaussian",
    "Noise",
    "Phase2D",
    "Phase3D",
    "Poisson",
    "PoissonGaussian",
    "ReconstructionModel",
]
