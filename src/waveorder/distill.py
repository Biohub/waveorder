"""Distilled models: a network (the student) trained on recorded reconstructions to give what a physics solver gives.

The front door of distillation: every name a distillation needs, from :mod:`waveorder.data`, :mod:`waveorder.train`
and :mod:`waveorder.models`. :func:`distillation` builds the task of a reconstruction model, :func:`prepare` the
callable that prepares recorded data as the model does, and :func:`evaluate_model` scores a student as that model's
solver. Training runs through ``wo distill fit`` or ``model.train`` (:class:`waveorder.models.DistillationModule`).
"""

from waveorder.data import (
    Batch,
    Field,
    Flip,
    FocusShift,
    FovIndex,
    GridPatches,
    GroupSplit,
    Holdout,
    PairSource,
    PatchData,
    PlaneCrop,
    RandomPatches,
    Recorded,
    Transform,
    index_from_plates,
)
from waveorder.models._distill import distillation, prepare
from waveorder.models._distill import evaluate as evaluate_model
from waveorder.nn import Distilled
from waveorder.train import L1, MSE, SSIM, Distillation, Loss, Optimization, evaluate, export_onnx

__all__ = [
    "L1",
    "MSE",
    "SSIM",
    "Batch",
    "Distillation",
    "Distilled",
    "Field",
    "Flip",
    "FocusShift",
    "FovIndex",
    "GridPatches",
    "GroupSplit",
    "Holdout",
    "Loss",
    "Optimization",
    "PairSource",
    "PatchData",
    "PlaneCrop",
    "RandomPatches",
    "Recorded",
    "Transform",
    "distillation",
    "evaluate",
    "evaluate_model",
    "export_onnx",
    "index_from_plates",
    "prepare",
]
