"""Training data: an index of recorded pairs of fields, and the loader that cuts them into batches.

An index of fields (:func:`index_from_plates`, :class:`FovIndex`, split by :class:`GroupSplit`, held out by
:class:`Holdout`) is what :class:`PatchData` loads into :class:`Batch` es of patches (:class:`RandomPatches`,
:class:`GridPatches`). Every field has a recorded target, a reconstruction made beforehand; a :class:`PairSource`
reads both (:class:`Recorded` plates by default). :class:`Transform` s augment a batch on its device. The package
knows nothing about training (:mod:`waveorder.train`).
"""

from waveorder.data.index import FovIndex, FovRow, GroupSplit, Holdout, index_from_plates
from waveorder.data.loader import Batch, DeviceLoader, Field, PatchData, distributed_shard
from waveorder.data.samplers import GridPatches, RandomPatches
from waveorder.data.sources import PairSource, Recorded
from waveorder.data.transforms import Flip, FocusShift, PlaneCrop, Transform

__all__ = [
    "Batch",
    "DeviceLoader",
    "Field",
    "Flip",
    "FocusShift",
    "FovIndex",
    "FovRow",
    "GridPatches",
    "GroupSplit",
    "Holdout",
    "PairSource",
    "PatchData",
    "PlaneCrop",
    "RandomPatches",
    "Recorded",
    "Transform",
    "distributed_shard",
    "index_from_plates",
]
