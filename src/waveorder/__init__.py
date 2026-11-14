"""Wave-optical simulation and reconstruction of optical properties, built on torch modules.

:mod:`waveorder.models` holds the user-facing models (``waveorder.models.Phase3D``, ...), each wrapping the torch
modules of :mod:`waveorder.nn`; :mod:`waveorder.config` reads the YAML form of a model and :mod:`waveorder.io` reads
and writes files; :mod:`waveorder.tiling` plans and runs tiled work, :mod:`waveorder.errors` holds every exception and
warning, and :mod:`waveorder.testing` the contract checks for new parts. :func:`observe` sends structured events of
every solve, calibration and tiled reconstruction to an :class:`Observer` (:mod:`waveorder.observe`).

The package imports nothing up front, so ``import waveorder`` (and the command line, ``waveorder.cli``) stays free of
torch; each public name imports its module on first use.
"""

import importlib
import sys
import types
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waveorder import config, data, errors, io, models, nn, testing, tiling, train
    from waveorder.observe import Observer, observe

__all__ = [
    "Observer",
    "config",
    "data",
    "errors",
    "io",
    "models",
    "nn",
    "observe",
    "testing",
    "tiling",
    "train",
]

_SUBMODULES = frozenset({"config", "data", "errors", "io", "models", "nn", "testing", "tiling", "train"})
"""The subpackages and modules that are attributes of the package, each imported on first use."""


def __getattr__(name: str) -> object:
    """Import ``Observer`` and the submodules of :data:`__all__` (``models``, ``errors``, ``tiling``, ...) on first use.

    PEP 562: ``import waveorder as wo; wo.models.Phase3D`` works without ``import waveorder.models``.
    """
    if name == "Observer":
        from waveorder.observe import Observer

        return Observer
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


class _Package(types.ModuleType):
    """The package module, whose ``observe`` stays the function after ``import waveorder.observe`` binds the module."""

    @property
    def observe(self) -> Callable[..., object]:
        """:func:`waveorder.observe.observe`, or the value last assigned to ``waveorder.observe`` (even ``None``)."""
        if "observe" in self.__dict__:
            return self.__dict__["observe"]
        from waveorder.observe import observe

        return observe

    @observe.setter
    def observe(self, value: object) -> None:
        if isinstance(value, types.ModuleType) and value.__name__ == f"{self.__name__}.observe":
            return
        self.__dict__["observe"] = value

    @observe.deleter
    def observe(self) -> None:
        self.__dict__.pop("observe", None)


sys.modules[__name__].__class__ = _Package
