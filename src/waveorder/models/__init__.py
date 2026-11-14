"""The user-facing models: one object per modality, wrapping the torch modules of :mod:`waveorder.nn`.

A model holds an imaging system with the solver that inverts it and the settings that calibrate it::

    import waveorder as wo
    from waveorder.nn import Detection, Illumination

    model = wo.models.Phase3D(
        Illumination(wavelength=0.532, numerical_aperture=0.5),
        Detection(wavelength=0.532, numerical_aperture=1.2),
        index_of_refraction_media=1.33,
        pixel_size=(0.25, 0.1),
    )
    phase = model.reconstruct(stack).estimate

:class:`Phase3D` and :class:`Phase2D` reconstruct brightfield stacks, :class:`Fluorescence3D` and
:class:`Fluorescence2D` fluorescence stacks; :class:`ReconstructionModel` is their base and wraps any other imaging
system (``ReconstructionModel.for_system(system)``, or ``ReconstructionModel.from_config(path)`` for a YAML file).

Public names come from ``waveorder.models``. ``contrasts.py`` holds the models of each contrast; the underscore files
what every model shares: ``_base.py`` the class and its verbs, ``_arguments.py`` the argument checks, ``_parts.py``
the parts a model takes, ``_run.py`` the events and the bind every method runs through, ``_tiles.py`` and
``_prepare.py`` the pieces of the tile methods, ``_config.py`` the configuration and the saved folder,
``_distill.py`` the distillation task, ``_lightning.py`` its Lightning module and command line
(:class:`DistillationModule`, :class:`DistillCLI`, imported on first use, with Lightning), and ``_positions.py``,
``_calibration.py``, ``_stores.py``, ``_checks.py`` and ``_pool.py`` the pieces of the OME-Zarr methods.
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
