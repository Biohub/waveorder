"""Training: tasks, their losses and metrics, the Lightning parts that run them, and what scores and exports the result.

:class:`Distillation` trains a student network (:mod:`waveorder.nn.networks`) on recorded reconstructions with the
loss terms here and the :class:`Optimization`; :class:`waveorder.models.DistillationModule` runs it with Lightning on a
:mod:`waveorder.train.lightning` data module, from ``wo distill fit`` or ``model.train``. :func:`evaluate` scores a
:class:`~waveorder.nn.networks.Bundle` on whole fields and :func:`export_onnx` writes it as a self-checked ONNX file.
Lightning is imported by :mod:`waveorder.train.lightning` only.
"""

from waveorder.train.evaluate import evaluate
from waveorder.train.export import export_onnx
from waveorder.train.losses import L1, MSE, SSIM, Loss
from waveorder.train.metrics import (
    dynamic_range_ratio,
    passband_error,
    plane_ssim,
    ssim,
    summarize,
    transfer_support,
)
from waveorder.train.optimization import Optimization
from waveorder.train.tasks import Distillation, StepResult, Task

__all__ = [
    "L1",
    "MSE",
    "SSIM",
    "Distillation",
    "Loss",
    "Optimization",
    "StepResult",
    "Task",
    "dynamic_range_ratio",
    "evaluate",
    "export_onnx",
    "passband_error",
    "plane_ssim",
    "ssim",
    "summarize",
    "transfer_support",
]
