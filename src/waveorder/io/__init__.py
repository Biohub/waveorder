"""Reading and writing files.

* :mod:`~waveorder.io.tf_store`: :class:`TfStore`, the ``tf.zarr`` transfer-function store.
* :mod:`~waveorder.io.positions`: input OME-Zarr positions, their metadata and volumes.
* :mod:`~waveorder.io.plate`: the output plate, and :func:`estimate_resources` for a whole-volume reconstruction.
* :mod:`~waveorder.io.staging`: :class:`Prefetcher`, background reads handed to a device in order.

:class:`TfStore` and :class:`Prefetcher` import torch on first use.
"""

import importlib
from typing import TYPE_CHECKING

from waveorder.io.plate import Resources, estimate_resources

if TYPE_CHECKING:
    from waveorder.io.staging import Prefetcher
    from waveorder.io.tf_store import TfStore

__all__ = ["Prefetcher", "Resources", "TfStore", "estimate_resources"]

_LAZY = {
    "Prefetcher": "waveorder.io.staging",
    "TfStore": "waveorder.io.tf_store",
}


def __getattr__(name: str) -> object:
    """Import :class:`TfStore` and :class:`Prefetcher` on first use (PEP 562)."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
