"""The user-facing models: one object per modality, wrapping the torch modules of :mod:`waveorder.nn`.

:class:`Phase3D` and :class:`Phase2D` reconstruct brightfield stacks, :class:`Fluorescence3D` and
:class:`Fluorescence2D` fluorescence stacks. :class:`ReconstructionModel` is their base and wraps any other imaging
system (``ReconstructionModel.for_system(system)``, or ``ReconstructionModel.from_config(path)`` for a YAML file).
"""

from waveorder.errors import ReconstructionFailed, ReconstructionReport, UnitResult
from waveorder.models._base import ReconstructionModel
from waveorder.models._parts import FieldTiles, Parts
from waveorder.models.contrasts import Fluorescence2D, Fluorescence3D, Phase2D, Phase3D

__all__ = [
    "FieldTiles",
    "Fluorescence2D",
    "Fluorescence3D",
    "Parts",
    "Phase2D",
    "Phase3D",
    "ReconstructionFailed",
    "ReconstructionModel",
    "ReconstructionReport",
    "UnitResult",
]
