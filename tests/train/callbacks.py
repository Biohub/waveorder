"""Parts the training tests name by class path: a preemption, a trace of the batches, a broken loss, an outside stem.

:class:`MeanThenConv` is a stem defined here, outside waveorder, registered nowhere and named by its class path.
"""

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import lightning.pytorch as pl  # noqa: TID253 (the callbacks a configuration names subclass Lightning's)
import torch
from torch import Tensor, nn

from waveorder.data import Batch
from waveorder.errors import ShapeError
from waveorder.nn.networks import Stem
from waveorder.nn.networks.layers import Norm, norm_layer
from waveorder.nn.networks.stems import StemModule
from waveorder.train import Loss


class Preempt(pl.Callback):
    """Kill the process, with no teardown, at the start of optimizer step ``at_step`` or of epoch ``at_epoch``."""

    def __init__(self, at_step: int | None = None, at_epoch: int | None = None) -> None:
        self.at_step, self.at_epoch = at_step, at_epoch

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: object, batch_idx: int
    ) -> None:
        del pl_module, batch, batch_idx
        if self.at_step is not None and trainer.global_step == self.at_step:
            os._exit(17)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if self.at_epoch is not None and trainer.current_epoch == self.at_epoch:
            os._exit(17)


class Trace(pl.Callback):
    """Append each batch's stage, epoch, step, field names and patch origins to ``<path>/rank<N>.jsonl``.

    The stage is ``train``, ``val`` (validation within a fit) or the trainer's function (``validate``, ``test``).
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def _write(self, trainer: pl.Trainer, stage: str, batch: Batch, index: int) -> None:
        line = {
            "stage": stage,
            "rank": trainer.global_rank,
            "world": trainer.world_size,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "batch": index,
            "positions": list(batch.position),
            "origins": batch.origin.tolist(),
        }
        Path(self.path).mkdir(parents=True, exist_ok=True)
        with (Path(self.path) / f"rank{trainer.global_rank}.jsonl").open("a") as file:
            file.write(json.dumps(line) + "\n")

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int
    ) -> None:
        del pl_module
        self._write(trainer, "train", batch, batch_idx)

    def on_validation_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        del pl_module, dataloader_idx
        stage = "val" if trainer.state.fn == "fit" else str(getattr(trainer.state.fn, "value", trainer.state.fn))
        self._write(trainer, stage, batch, batch_idx)


def traced(path: Path) -> dict[tuple, tuple]:
    """Return ``(stage, rank, epoch, batch) -> (positions, origins)`` of a trace, from each batch's last line."""
    steps = {}
    for file in sorted(path.glob("rank*.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            steps[row["stage"], row["rank"], row["epoch"], row["batch"]] = (row["positions"], row["origins"])
    return steps


class NotANumber(Loss):
    """A loss that is NaN from the first step."""

    def forward(self, estimate: Tensor, batch: Batch) -> Tensor:
        del batch
        return estimate.sum() * math.nan


class _MeanThenConv(StemModule):
    def __init__(self, planes: int, channels: int, condition_planes: int, norm: Norm) -> None:
        super().__init__()
        self.planes, self.fan_in, self.out_channels = planes, 1, channels
        self.conv = nn.Conv2d(1 + condition_planes, channels, kernel_size=3, padding=1)
        self.norm = norm_layer(channels, norm)

    def project(self, data: Tensor, condition: Mapping[str, Tensor] | None, planes: Tensor | None) -> Tensor:
        del condition
        if data.shape[2] != self.planes:
            raise ShapeError(f"MeanThenConv takes stacks of {self.planes} planes, got {data.shape[2]}")
        folded = data.mean(dim=2)
        if planes is not None:
            folded = torch.cat([folded, planes[:, :, None, None].expand(-1, -1, *folded.shape[-2:])], dim=1)
        return self.conv(folded)

    def finish(self, features: Tensor) -> Tensor:
        return self.norm(features)


class MeanThenConv(Stem):
    """A stem from outside waveorder, registered nowhere: the mean over z, then one 3x3 convolution.

    Its constructor arguments are attributes of the same names, which is what a bundle writes.
    """

    stride = 1

    def __init__(self, planes: int, channels: int = 8) -> None:
        self.planes, self.channels, self.data_channels = planes, channels, 1

    def planes_out(self, planes_in: int) -> int:
        if planes_in != self.planes:
            raise ShapeError(f"MeanThenConv takes stacks of {self.planes} planes, got {planes_in}")
        return 1

    def module(self, *, width: int | None, condition_planes: int, norm: Norm) -> StemModule:
        del width
        return _MeanThenConv(self.planes, self.channels, condition_planes, norm)
