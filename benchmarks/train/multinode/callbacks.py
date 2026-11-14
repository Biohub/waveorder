"""Callbacks of the two-node test, named by class path: a trace of every batch, and one forced requeue.

The sbatch file puts this folder on ``PYTHONPATH``, so a configuration names them ``callbacks.Trace`` and
``callbacks.RequeueOnce``.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import override

import lightning.pytorch as pl  # noqa: TID253 (the callbacks a configuration names subclass Lightning's)

from waveorder.data import Batch


class Trace(pl.Callback):
    """Append each batch's stage, rank, epoch, step, field names and patch origins to ``<path>/rank<N>.jsonl``."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _write(self, trainer: pl.Trainer, stage: str, batch: Batch, *, index: int) -> None:
        line = {
            "stage": stage,
            "rank": trainer.global_rank,
            "world": trainer.world_size,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "batch": index,
            "attempt": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "positions": list(batch.position),
            "origins": batch.origin.tolist(),
        }
        Path(self.path).mkdir(parents=True, exist_ok=True)
        with (Path(self.path) / f"rank{trainer.global_rank}.jsonl").open("a") as file:
            file.write(json.dumps(line) + "\n")

    @override
    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int
    ) -> None:
        self._write(trainer, "train", batch, index=batch_idx)

    @override
    def on_validation_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if trainer.state.fn == "fit":
            self._write(trainer, "val", batch, index=batch_idx)


class RequeueOnce(pl.Callback):
    """On the first attempt only, rank 0 runs ``scontrol requeue`` at the start of optimizer step ``at_step``.

    Slurm then sends SIGTERM to every task and puts the job back in the queue under the same id, as a preemption
    does: Lightning stops every rank at the next batch and ``ExceptionCheckpoint`` writes ``on_exception.ckpt``.
    """

    def __init__(self, at_step: int) -> None:
        self.at_step = at_step

    @override
    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int
    ) -> None:
        first = int(os.environ.get("SLURM_RESTART_COUNT", "0")) == 0
        if first and trainer.global_step == self.at_step and trainer.is_global_zero:
            where = f"step {trainer.global_step}, epoch {trainer.current_epoch}, batch {batch_idx}"
            print(f"REQUEUE at {where}", flush=True)
            subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=False)
