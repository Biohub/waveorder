"""The user-facing models: one object per modality, wrapping the torch modules of :mod:`waveorder.nn`.

:class:`Phase3D` and :class:`Phase2D` reconstruct brightfield stacks, :class:`Fluorescence3D` and
:class:`Fluorescence2D` fluorescence stacks. :class:`ReconstructionModel` is their base and wraps any other imaging
system (``ReconstructionModel.for_system(system)``, or ``ReconstructionModel.from_config(path)`` for a YAML file).
:class:`DistillationModule` and :class:`DistillCLI` train a network; they import Lightning on first use.
"""

from typing import TYPE_CHECKING

from waveorder.errors import ReconstructionFailed, ReconstructionReport, UnitResult
from waveorder.models._base import ReconstructionModel, TrainResult
from waveorder.models._parts import FieldTiles, Parts
from waveorder.models.contrasts import Fluorescence2D, Fluorescence3D, Phase2D, Phase3D

if TYPE_CHECKING:
    from waveorder.models._lightning import DistillationModule, DistillCLI

__all__ = [
    "DistillCLI",
    "DistillationModule",
    "FieldTiles",
    "Fluorescence2D",
    "Fluorescence3D",
    "Parts",
    "Phase2D",
    "Phase3D",
    "ReconstructionFailed",
    "ReconstructionModel",
    "ReconstructionReport",
    "TrainResult",
    "UnitResult",
]


def __getattr__(name: str) -> object:
    """Import the Lightning classes on first use: reconstruction never imports Lightning."""
    if name in ("DistillCLI", "DistillationModule"):
        from waveorder.models import _lightning

        return getattr(_lightning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
