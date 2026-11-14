"""Model-agnostic Lightning parts of a training run: the data module, the run's callbacks, its warning filter.

Relies on Lightning 2.6.6 (pinned): ``use_distributed_sampler=False`` (each rank loads its own shard of fields);
the train loader's epoch is set through ``loader.sampler`` every epoch, restarts included, and its state is loaded
before the restart's first ``iter()``; steps per epoch come from the index, not ``estimated_stepping_batches``.
:class:`BestBundle` must be listed before the weight-averaging callback.
"""

import contextlib
import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import warnings
from collections.abc import Callable, Generator, Iterator, Mapping
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, override

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loops.progress import _ProcessedTracker
from lightning.pytorch.utilities.exceptions import SIGTERMException
from pydantic import NonNegativeInt, PositiveInt
from torch.utils.data import DataLoader

import waveorder
from waveorder.data import Batch, DeviceLoader, FovIndex, Holdout, PatchData, RandomPatches
from waveorder.data.index import MANIFEST, SPLIT
from waveorder.errors import ParameterError
from waveorder.train.optimization import Mode
from waveorder.train.tasks import Distillation

SCHEMA = "waveorder.distill_run/2"
"""The ``schema`` entry of a run folder's ``run.json``."""

_QUIET = (
    "does not have many workers",  # PatchData starts its own workers; the loader Lightning sees has none to count
    "The `srun` command is available",  # how the job was launched is the caller's choice
    "GPU available but not used",  # devices="cpu" is a choice
    "is smaller than the logging interval",
    r"Your `IterableDataset` has `__len__` defined",  # PatchData gives each worker its own share: the length is exact
    "Experiment logs directory .* exists and is not empty",  # a run folder resumed or written again
    "Checkpoint directory .* exists and is not empty",
    r"`isinstance\(treespec, LeafSpec\)` is deprecated",  # Lightning's own use of torch
    r"was deprecated in v4\.",  # LightningCLI's own use of jsonargparse (instantiate_classes, skip_none, ...)
    "By default only one JsonargparseDeprecationWarning",
)
"""Starts or parts of Lightning warnings that say nothing about a waveorder run; a run does not show them."""


@contextlib.contextmanager
def quiet() -> Generator[None]:
    """Hide the Lightning and jsonargparse warnings that say nothing about a waveorder run, for the ``with`` block."""
    with warnings.catch_warnings():
        for message in _QUIET:
            warnings.filterwarnings("ignore", message=f"(?s).*{message}")  # jsonargparse's start with a newline
        yield


class _EpochSampler:
    """What Lightning finds as ``loader.sampler``: its ``set_epoch`` is called every epoch, restarts included."""

    def __init__(self, data: PatchData) -> None:
        self.data = data

    def set_epoch(self, epoch: int) -> None:
        self.data.set_epoch(epoch)


class ResumableLoader:
    """The train loader of a run: a restart continues its epoch from the batch it stopped at.

    Trained batches are read from the loop's progress (Lightning fetches one batch ahead). Skipped batches are still
    made by the workers. ``iter()`` starts nothing, so persistent workers read the epoch set after it.
    """

    def __init__(self, loader: DataLoader[Batch] | DeviceLoader, data: PatchData, trainer: pl.Trainer) -> None:
        self.loader, self.data, self.trainer = loader, data, trainer
        self.sampler, self.skip = _EpochSampler(data), 0

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Batch]:
        skip, self.skip = self.skip, 0
        return self._batches(skip)

    def _batches(self, skip: int) -> Iterator[Batch]:
        batches = iter(self.loader)
        for _ in range(skip):
            next(batches)
        yield from batches

    def state_dict(self) -> dict[str, int]:
        """Return the epoch being read; what was trained of it is in the trainer's own progress."""
        return {"epoch": self.data.epoch}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        """Continue the epoch of ``state`` after the batches the restored loop trained, or start the next one."""
        progress = self.trainer.fit_loop.epoch_loop.batch_progress.current
        assert isinstance(progress, _ProcessedTracker)
        done = progress.processed
        epoch, skip = (state["epoch"] + 1, 0) if done >= len(self) else (state["epoch"], done)
        self.data.set_epoch(epoch)
        self.skip = skip


class PatchDataModule(pl.LightningDataModule):
    """Patches of the fields of an index (:class:`~waveorder.data.PatchData`); each rank loads its own shard.

    The loader uses the task's ``prepare``, conditions and normalizers, so training and inference agree.

    Parameters
    ----------
    index : str
        The index directory ``wo distill index`` writes.
    split : str
        The split column to read: ``split``, or a named split ``split_<name>``.
    holdout : Holdout, optional
        Plates and wells kept out of every split (their rows get the split ``holdout``).
    patch_size : tuple of two ints
        Patch height and width in pixels.
    per_field : int
        Patches drawn from a field each epoch.
    batch_size : int
        Patches per batch.
    buffer_fields : int
        Fields whose patches are mixed in a batch.
    workers : int
        Loader processes (spawned); 0 reads in the training process.
    cache_fields : int
        Fields kept decoded per worker.
    limit_fields : int, optional
        Use only this many fields of each split, for smoke runs.
    device : {"auto", "cpu"}
        Where fields are prepared and cut: ``"auto"`` on the rank's GPU when it trains on one, else ``"cpu"`` workers.
    seed : int
        Seed of the field order and patch origins; the CLI links it to ``seed_everything``.
    """

    def __init__(  # noqa: PLR0913 (one keyword per setting of the loader)
        self,
        index: str,
        *,
        split: str = SPLIT,
        holdout: Holdout | None = None,
        patch_size: tuple[PositiveInt, PositiveInt] = (256, 256),
        per_field: PositiveInt = 64,
        batch_size: PositiveInt = 16,
        buffer_fields: PositiveInt = 4,
        workers: NonNegativeInt = 8,
        cache_fields: NonNegativeInt = 0,
        limit_fields: PositiveInt | None = None,
        device: Literal["auto", "cpu"] = "auto",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.path, self.split, self.holdout, self.device_choice = index, split, holdout, device
        self.loading: dict[str, Any] = {
            "patches": RandomPatches(tuple(patch_size), per_field=per_field),
            "batch_size": batch_size,
            "buffer_fields": buffer_fields,
            "workers": workers,
            "cache_fields": cache_fields,
            "limit_fields": limit_fields,
            "seed": seed,
        }
        self.index: FovIndex | None = None
        self.data: PatchData | None = None
        self.shard: PatchData | None = None

    @override
    def setup(self, stage: str) -> None:
        if self.data is not None:
            return
        trainer = self.trainer
        assert trainer is not None
        task = trainer.lightning_module.task
        assert isinstance(task, Distillation), f"a PatchDataModule trains a Distillation, not {type(task).__name__}"
        self.index = FovIndex.load(self.path).assign(self.split, holdout=self.holdout)
        self.data = PatchData(
            self.index,
            conditions=task.conditions,
            prepare=task.prepare,
            input_normalizer=task.input_normalizer,
            target_normalizer=task.target_normalizer,
            stride=task.student.stride,
            **self.loading,
        )
        root = trainer.strategy.root_device
        device = root if self.device_choice == "auto" and root.type == "cuda" else "cpu"
        self.shard = self.data.shard(rank=trainer.global_rank, world_size=trainer.world_size, device=device)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch every loader of this rank reads next."""
        assert self.shard is not None
        self.shard.set_epoch(epoch)

    @override
    def train_dataloader(self) -> ResumableLoader:
        assert self.shard is not None
        assert self.trainer is not None
        return ResumableLoader(self.shard.loader("train"), self.shard, self.trainer)

    @override
    def val_dataloader(self) -> object:
        assert self.shard is not None
        return self.shard.loader("val")

    @override
    def test_dataloader(self) -> object:
        assert self.shard is not None
        return self.shard.loader("test", train=False)

    def batches_per_epoch(self) -> int:
        """Return the optimizer steps of a training epoch of this rank, from the index alone."""
        assert self.shard is not None
        return len(self.shard.loader("train"))

    def describe(self) -> dict[str, object]:
        """Return the data as JSON-able values for ``run.json``: the index, its hashes and the effective split."""
        assert self.index is not None
        assert self.data is not None
        manifest = Path(self.path) / MANIFEST
        holdout = None if self.holdout is None else dataclasses.asdict(self.holdout)
        return {
            "index": str(Path(self.path).absolute()),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "index_sha256": self.index.sha256,
            "split": {"column": self.split, "holdout": holdout, "counts": self.index.counts},
            "split_sha256": self.index.split_sha256,
            "loader": (self.data if self.shard is None else self.shard).describe(),  # this rank's, on its device
        }

    @override
    def teardown(self, stage: str) -> None:
        if self.data is not None:
            self.data.close()


@cache
def software() -> dict[str, object]:
    """Return the versions of waveorder and what it trains with, and the git commit it is imported from."""
    versions: dict[str, object] = {name: version(name) for name in ("waveorder", "iohub", "torch", "lightning")}
    return {"software": versions, "git_commit": _git_commit()}


def _git_commit() -> str | None:
    """Return the commit of the git checkout waveorder is imported from, or ``None`` when it is not in one."""
    source = Path(waveorder.__file__).resolve().parent
    try:
        done = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else None


def metrics_of(trainer: pl.Trainer) -> dict[str, float]:
    """Return the metrics the trainer holds as Python floats, by name."""
    return {name: float(value) for name, value in trainer.callback_metrics.items()}


class BestBundle(pl.Callback):
    """Write ``<run>/best`` whenever the monitored metric improves, and ``<run>/last`` at the end (rank 0).

    Each bundle records the metrics, step, epoch, index hashes, software versions and commit.

    Parameters
    ----------
    write : callable
        ``write(bundle, path, overwrite=True)``: :func:`waveorder.io.bundle.save_bundle`.
    monitor : str
        The validation metric: ``"val/ssim"``, ``"val/dynamic_range"``, ``"val/loss"`` or ``"val/loss/<name>"``.
    mode : {"max", "min"}
        Whether a larger or a smaller ``monitor`` is better.

    Raises
    ------
    ParameterError
        From the first validation epoch, if no metric named ``monitor`` was logged.
    """

    def __init__(self, write: Callable[..., object], *, monitor: str = "val/ssim", mode: Mode = "max") -> None:
        self.write, self.monitor, self.mode = write, monitor, mode
        self.best: float | None = None
        self.directory: Path | None = None

    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self.directory = Path(str(trainer.log_dir))  # it broadcasts: read on every rank, never on rank 0 alone

    def _save(self, trainer: pl.Trainer, name: str, validation: Mapping[str, float]) -> None:
        assert self.directory is not None
        task = trainer.lightning_module.task
        assert isinstance(task, Distillation), f"a BestBundle saves a Distillation, not {type(task).__name__}"
        data = getattr(trainer, "datamodule", None)  # an attribute Lightning sets when the run starts
        assert isinstance(data, PatchDataModule)
        record = {
            "metrics": {**metrics_of(trainer), **{f"val/{key}": value for key, value in validation.items()}},
            "steps": trainer.global_step,
            "epoch": trainer.current_epoch,
            "monitor": {"name": self.monitor, "mode": self.mode},
            "index_sha256": data.index.sha256,
            "splits": data.index.counts,
            "split_sha256": data.index.split_sha256,
            **software(),
        }
        self.write(task.bundle(record=record), self.directory / name, overwrite=True)

    @override
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        # The module's means, reduced over the ranks: its own logging of them reaches the trainer only after the
        # weight averaging has swapped the averaged weights out again.
        means = pl_module.validation_means()  # ty: ignore[call-non-callable] (a DistillationModule)
        if self.monitor.removeprefix("val/") not in means:
            logged = ", ".join(f"val/{name}" for name in sorted(means)) or "none"
            raise ParameterError(
                f"the run monitors {self.monitor!r}, but validation logged no such metric (logged: {logged}); "
                "'val/ssim' needs at least one validation field that is not blank",
                path="monitor",
            )
        value = float(means[self.monitor.removeprefix("val/")])
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if not math.isfinite(value) or not better:
            return
        self.best = value  # on every rank: the means are reduced over the ranks, so they agree
        if trainer.is_global_zero:
            self._save(trainer, "best", means)

    @override
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        task, data = pl_module.task, getattr(trainer, "datamodule", None)
        assert isinstance(task, Distillation)
        assert isinstance(data, PatchDataModule)
        assert data.data is not None
        if task.example is None:  # the run ended before its first validation: test tensors from a validation batch
            data = data.data.replace(workers=0, device=None)
            try:
                task.hold(next(data.batches("val", train=False)))
            finally:
                data.close()
        self._save(trainer, "last", {})

    @override
    def state_dict(self) -> dict[str, object]:
        return {"best": self.best}

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        best = state_dict.get("best")
        self.best = None if best is None else float(best)  # ty: ignore[invalid-argument-type]


class LastCheckpoint(ModelCheckpoint):
    """Write ``<run>/checkpoints/last.ckpt`` every ``every`` optimizer steps and at each epoch end; resume from it."""

    def __init__(self, every: int) -> None:
        super().__init__(
            filename="last", monitor=None, save_top_k=1, every_n_train_steps=every, enable_version_counter=False
        )

    @override
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking and self.dirpath is not None:
            trainer.save_checkpoint(Path(self.dirpath) / "last.ckpt", weights_only=False)


class ExceptionCheckpoint(pl.Callback):
    """Write ``<run>/checkpoints/on_exception.ckpt`` when Lightning stops every rank on SIGTERM (a preemption).

    Other exceptions save nothing (they may come from one rank alone). The file is removed when the run ends normally.
    """

    def __init__(self) -> None:
        self.path: Path | None = None

    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self.path = Path(str(trainer.log_dir)) / "checkpoints" / "on_exception.ckpt"  # it broadcasts: every rank

    @override
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        if isinstance(exception, SIGTERMException) and self.path is not None:
            trainer.save_checkpoint(self.path)  # collective: every rank stops on the same SIGTERM

    @override
    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.path is not None:
            trainer.strategy.remove_checkpoint(self.path)


class RunRecord(pl.Callback):
    """Write ``<run>/run.json`` and copies of the model file and index manifest into the run folder (rank 0)."""

    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        directory = Path(str(trainer.log_dir))  # it broadcasts: read on every rank
        if not trainer.is_global_zero:
            return
        directory.mkdir(parents=True, exist_ok=True)
        module = pl_module.describe()  # ty: ignore[call-non-callable] (a DistillationModule)
        datamodule = getattr(trainer, "datamodule", None)
        assert isinstance(datamodule, PatchDataModule)
        data = datamodule.describe()
        run = {"schema": SCHEMA, "stage": stage, "ckpt_path": trainer.ckpt_path, **module, "data": data, **software()}
        (directory / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True, default=str) + "\n")
        shutil.copy2(Path(str(data["index"])) / MANIFEST, directory / "index_manifest.json")
        source = Path(str(module["reconstruction"]))
        copy = directory / ("model" if source.is_dir() else f"model{source.suffix}")
        if copy.resolve() == source.resolve():  # model.train writes the model into the run folder itself
            return
        if source.is_dir():
            shutil.copytree(source, copy, dirs_exist_ok=True)
        else:
            shutil.copy2(source, copy)
