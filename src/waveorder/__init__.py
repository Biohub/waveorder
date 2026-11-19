"""Wave-optical simulation and reconstruction of optical properties, built on torch modules.

:mod:`waveorder.models` holds the user-facing models (``waveorder.models.Phase3D``, ...), each wrapping the torch
modules of :mod:`waveorder.nn`; :mod:`waveorder.config` reads the YAML form of a model and :mod:`waveorder.io` reads
and writes files; :mod:`waveorder.tiling` plans and runs tiled work, :mod:`waveorder.errors` holds every exception and
warning, and :mod:`waveorder.testing` the contract checks for new parts.

The package imports nothing up front, so ``import waveorder`` (and the command line, ``waveorder.cli``) stays free of
torch; each public name imports its module on first use.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waveorder import config, errors, io, models, nn, testing, tiling

__all__ = [
    "config",
    "errors",
    "io",
    "models",
    "nn",
    "testing",
    "tiling",
]

_SUBMODULES = frozenset(__all__)
"""The subpackages and modules that are attributes of the package, each imported on first use."""


def __getattr__(name: str) -> object:
    """Import the submodules of :data:`__all__` (``models``, ``errors``, ``tiling``, ...) on first use.

    PEP 562: ``import waveorder as wo; wo.models.Phase3D`` works without ``import waveorder.models``.
    """
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
