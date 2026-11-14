"""How a training run optimizes: :class:`Optimization`, and the learning-rate schedules it describes."""

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
from pydantic import Field, NonNegativeFloat, PositiveFloat

from waveorder.errors import ParameterError

type Mode = Literal["max", "min"]
"""Whether a larger or a smaller value of the monitored metric is better."""

type Schedule = Literal["cosine", "plateau", "constant"]
"""How the learning rate changes after the warmup: a cosine to ``min_lr`` over the run (stepped every optimizer
step), halved when the monitored metric stops improving, or not at all."""

_PLATEAU_FACTOR = 0.5
_PLATEAU_PATIENCE = 5
"""The ``"plateau"`` schedule: the learning rate is halved after this many epochs without improvement."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Optimization:
    """How a run optimizes: AdamW, the learning-rate schedule, gradient clipping and weight averaging.

    Parameters
    ----------
    lr : float
        The learning rate after the warmup.
    weight_decay : float
        AdamW's decoupled weight decay.
    schedule : {"cosine", "plateau", "constant"}
        ``"cosine"``: ``lr`` to ``min_lr`` over the run, stepped every optimizer step. ``"plateau"``: halved after 5
        validation epochs without improvement, not below ``min_lr``. ``"constant"``: ``lr`` throughout.
    warmup_epochs : float
        Epochs of linear rise from near 0 to ``lr``; 0 for none.
    min_lr : float
        The smallest learning rate a schedule reaches.
    gradient_clip : float or None
        Largest gradient norm over all parameters; ``None`` does not clip.
    ema : float or None
        Decay of the per-step weight average used for validation, checkpoints and bundles; ``None`` keeps none.

    Raises
    ------
    ParameterError
        If a value is out of range; the message names it.
    """

    lr: PositiveFloat = 5e-4
    weight_decay: NonNegativeFloat = 1e-4
    schedule: Schedule = "cosine"
    warmup_epochs: NonNegativeFloat = 2.0
    min_lr: NonNegativeFloat = 1e-6
    gradient_clip: PositiveFloat | None = 2.0
    ema: Annotated[float, Field(gt=0, lt=1)] | None = 0.999

    def __post_init__(self) -> None:
        finite = math.isfinite
        rules = {
            "lr": (finite(self.lr) and self.lr > 0, "a positive number"),
            "weight_decay": (finite(self.weight_decay) and self.weight_decay >= 0, "a number of at least 0"),
            "schedule": (self.schedule in ("cosine", "plateau", "constant"), "one of 'cosine', 'plateau', 'constant'"),
            "warmup_epochs": (finite(self.warmup_epochs) and self.warmup_epochs >= 0, "a number of at least 0"),
            "min_lr": (finite(self.min_lr) and 0 <= self.min_lr <= self.lr, "a number from 0 to lr"),
            "gradient_clip": (
                self.gradient_clip is None or (finite(self.gradient_clip) and self.gradient_clip > 0),
                "a positive number or None",
            ),
            "ema": (self.ema is None or 0 < self.ema < 1, "a decay between 0 and 1 (such as 0.999) or None"),
        }
        for name, (valid, rule) in rules.items():
            if not valid:
                raise ParameterError(f"Optimization {name} must be {rule}, got {getattr(self, name)!r}", path=name)


def warmup_cosine(step: int, *, warmup: int, total: int, floor: float) -> float:
    """Return the learning-rate factor at optimizer step ``step`` (from 0) of ``total``.

    During the ``warmup`` steps the factor is ``(step + 1) / warmup``, so the first step is not 0; then a cosine
    falls to ``floor`` (``min_lr / lr``) at ``total`` and stays there.
    """
    if step < warmup:
        return (step + 1) / warmup
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def schedulers(
    optimizer: torch.optim.Optimizer, optimization: Optimization, *, steps: tuple[int, int], monitor: str, mode: Mode
) -> list[dict[str, object]]:
    """Return Lightning's scheduler configurations for ``optimization``: the warmup and the schedule after it.

    ``steps`` is the optimizer steps of the whole run and of one epoch; ``monitor`` and ``mode`` are what a
    ``"plateau"`` schedule watches. ``"cosine"`` is one scheduler stepped every optimizer step
    (:func:`warmup_cosine`); the others a linear warmup stepped every step, when there is one, and for
    ``"plateau"`` a ``ReduceLROnPlateau`` stepped every epoch.
    """
    total, per_epoch = steps
    warmup = min(round(optimization.warmup_epochs * per_epoch), max(0, total - 1))
    if optimization.schedule == "cosine":
        floor = min(1.0, optimization.min_lr / optimization.lr)
        cosine = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: warmup_cosine(step, warmup=warmup, total=total, floor=floor)
        )
        return [{"scheduler": cosine, "interval": "step"}]
    configured: list[dict[str, object]] = []
    if warmup > 1:
        ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / warmup, total_iters=warmup - 1)
        configured.append({"scheduler": ramp, "interval": "step"})
    if optimization.schedule == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=_PLATEAU_FACTOR, patience=_PLATEAU_PATIENCE, min_lr=optimization.min_lr
        )
        configured.append({"scheduler": plateau, "interval": "epoch", "monitor": monitor})
    return configured
