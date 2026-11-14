"""Training runs for the tests: the configuration of a tiny run on the phase pair, run here or as ``wo distill``.

A configuration names waveorder's parts by their bare class names and the test parts (:mod:`tests.train.callbacks`)
by their class paths, as a user's file does.
"""

import copy
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tests.data.plates import thin_phase_model
from waveorder.data import FovIndex
from waveorder.models import DistillCLI
from waveorder.train.lightning import quiet

ROOT = Path(__file__).resolve().parents[2]
"""The repository, on the path of every spawned run so that it imports ``tests.train.callbacks``."""

STUDENT = {
    "class_path": "UNet",
    "init_args": {
        "stem": {"class_path": "StridedVolume", "init_args": {"planes": 5}},
        "encoder": {"class_path": "ResNetEncoder", "init_args": {"widths": [8, 8, 8, 8], "depths": [1, 1, 1, 1]}},
        "head": "SubPixelHead",
        "decoder_channels": [8, 8, 8],
    },
}
"""The tiny strided U-Net of ``tests.nn.networks.tiny.tiny_unet`` for 5-plane stacks, as a configuration."""


@dataclass(frozen=True)
class Inputs:
    """What a run reads: the reconstruction model's folder and the index directory."""

    model: Path
    index: Path


def inputs(index: FovIndex, directory: Path) -> Inputs:
    """Save the phase model and ``index`` under ``directory``: the inputs of a run."""
    return Inputs(
        model=thin_phase_model().save(directory / "model"),
        index=index.save(directory / "index"),
    )


def configuration(
    given: Inputs,
    run: Path,
    *,
    epochs: int = 2,
    workers: int = 0,
    callbacks: Sequence[Mapping[str, object]] = (),
    model: Mapping[str, object] | None = None,
    data: Mapping[str, object] | None = None,
    trainer: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the configuration of a run on the CPU in the folder ``run``, traced to ``run/trace``.

    L1 and SSIM, a flip, the cosine schedule with warmup and weight averaging, ``last.ckpt`` after every step;
    ``model``, ``data`` and ``trainer`` replace entries of their blocks.
    """
    csv = "lightning.pytorch.loggers.CSVLogger"
    logger = {"class_path": csv, "init_args": {"save_dir": str(run), "name": "", "version": ""}}
    trace = {"class_path": "tests.train.callbacks.Trace", "init_args": {"path": str(run / "trace")}}
    return {
        "seed_everything": 0,
        "checkpoint_every": 1,
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": epochs,
            "logger": logger,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "callbacks": [trace, *callbacks],
            **(trainer or {}),
        },
        "model": {
            "reconstruction": str(given.model),
            "student": copy.deepcopy(STUDENT),
            "target_normalizer": {"class_path": "Affine", "init_args": {"percentiles": [1, 99]}},
            "losses": {"l1": "L1", "ssim": {"class_path": "SSIM", "init_args": {"window": 7}}},
            "weights": {"ssim": 0.5},
            "transforms": [{"class_path": "Flip"}],
            "optimization": {"lr": 1e-3, "warmup_epochs": 0.5, "ema": 0.9},
            **(model or {}),
        },
        "data": {
            "index": str(given.index),
            "patch_size": [32, 32],
            "per_field": 4,
            "batch_size": 4,
            "workers": workers,
            "device": "cpu",
            **(data or {}),
        },
    }


def fit_here(config: Mapping[str, object]) -> DistillCLI:
    """Build the run as ``wo distill fit`` does and fit it in this process; return the CLI, which holds the trainer."""
    with quiet():
        cli = DistillCLI(dict(config), run=False)
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    return cli


def started(command: str, config: Mapping[str, object], path: Path, *arguments: str) -> subprocess.Popen[str]:
    """Start ``wo distill <command> --config <path> <arguments>`` with ``config`` written to ``path``."""
    path.write_text(json.dumps(config, indent=2))
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    return subprocess.Popen(
        [sys.executable, "-m", "waveorder.cli", "distill", command, "--config", str(path), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        cwd=path.parent,
    )


def finished(process: subprocess.Popen[str], *, code: int = 0, timeout: float = 600) -> str:
    """Wait for a run; return its output after checking its exit status."""
    output, _ = process.communicate(timeout=timeout)
    assert process.returncode == code, f"exit {process.returncode}, expected {code}:\n{output[-4000:]}"
    return output


def checkpoint(run: Path) -> dict[str, Any]:
    """Return ``run/checkpoints/last.ckpt``."""
    return torch.load(run / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)


RESUMED = ("state_dict", "current_model_state", "averaging_state", "optimizer_states", "lr_schedulers")
"""What a resumed run must hold exactly as the uninterrupted one: weights, raw weights, EMA, AdamW, the schedule."""


def differences(first: object, second: object, path: str = "") -> Iterator[str]:
    """Yield where two nested states differ: tensors bit for bit, everything else by equality."""
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        if first.shape != second.shape or not torch.equal(first, second):
            yield path
    elif isinstance(first, Mapping) and isinstance(second, Mapping):
        for key in sorted(set(first) | set(second), key=str):
            if key not in first or key not in second:
                yield f"{path}.{key} (in one only)"
            else:
                yield from differences(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, list | tuple) and isinstance(second, list | tuple) and len(first) == len(second):
        for number, (a, b) in enumerate(zip(first, second, strict=True)):
            yield from differences(a, b, f"{path}[{number}]")
    elif first != second:
        yield f"{path}: {first!r} != {second!r}"
