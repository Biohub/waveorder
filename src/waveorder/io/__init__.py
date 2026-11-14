"""Reading and writing files.

* :mod:`~waveorder.io.tf_store`: :class:`TfStore`, the ``tf.zarr`` transfer-function store.
* :mod:`~waveorder.io.positions`: input OME-Zarr positions, their metadata and volumes.
* :mod:`~waveorder.io.plate`: the output plate, and :func:`estimate_resources` for a whole-volume reconstruction.
* :mod:`~waveorder.io.bundle`: a student bundle directory, :func:`save_bundle` and :func:`load_bundle`.
* :mod:`~waveorder.io.staging`: :class:`Prefetcher`, background reads handed to a device in order.

:class:`TfStore`, :class:`Prefetcher` and the bundle functions import torch on first use.
"""

import importlib
from typing import TYPE_CHECKING

from waveorder.io.plate import Resources, estimate_resources

if TYPE_CHECKING:
    from waveorder.io.bundle import load_bundle, save_bundle
    from waveorder.io.staging import Prefetcher
    from waveorder.io.tf_store import TfStore

__all__ = ["Prefetcher", "Resources", "TfStore", "estimate_resources", "load_bundle", "save_bundle"]

_LAZY = {
    "Prefetcher": "waveorder.io.staging",
    "TfStore": "waveorder.io.tf_store",
    "load_bundle": "waveorder.io.bundle",
    "save_bundle": "waveorder.io.bundle",
}


def __getattr__(name: str) -> object:
    """Import :class:`TfStore`, :class:`Prefetcher` and the bundle functions on first use (PEP 562)."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
