"""Training a distilled model with Lightning: :class:`DistillationModule` and :class:`DistillCLI`.

``wo distill fit|validate|test`` run :class:`DistillCLI`, Lightning's command line, on configuration files
(``--config``, repeatable and layered), flags and ``--ckpt_path``; ``model.train`` builds the same configuration in
Python and runs the same classes. A configuration names each part by its class: waveorder's own by their bare names
(``StridedVolume``, ``L1``), any other by its import path. The resolved configuration the run folder keeps
(``config.yaml``) and every bundle record full class paths, and absolute paths for the files read.
"""

import contextlib
import dataclasses
import difflib
import importlib
import inspect
import re
import sys
import traceback
from argparse import ArgumentError
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NoReturn, override

import lightning.pytorch as pl
import torch
from jsonargparse import Namespace
from lightning.pytorch.callbacks import EMAWeightAveraging
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pydantic import PositiveInt
from torch import Tensor

from waveorder.data import Batch, Transform
from waveorder.data.index import MANIFEST, read_manifest
from waveorder.data.samplers import draw_seed
from waveorder.errors import ConfigurationError, ParameterError, TrainingFailed, WaveorderError
from waveorder.io.bundle import load_bundle, save_bundle
from waveorder.models._base import ReconstructionModel
from waveorder.models._distill import distillation
from waveorder.nn.networks import Bundle, Condition, Normalizer, RestorationNetwork
from waveorder.train import L1, Distillation, Loss, Optimization
from waveorder.train.lightning import BestBundle, ExceptionCheckpoint, LastCheckpoint, PatchDataModule, RunRecord, quiet
from waveorder.train.optimization import Mode, schedulers

TRAINER_DEFAULTS: dict[str, object] = {
    "use_distributed_sampler": False,  # PatchData shards the fields itself, by rank
    "num_sanity_val_steps": 0,
    "enable_model_summary": False,
    "logger": {"class_path": "lightning.pytorch.loggers.CSVLogger", "init_args": {"save_dir": "runs"}},
}
"""What a distillation run's ``trainer`` block defaults to; any of it is overridden by a configuration or a flag."""


class DistillationModule(pl.LightningModule):
    """Train a student to give the estimates of a reconstruction model, from its recorded reconstructions.

    It holds the task (:class:`waveorder.train.Distillation`) and its optimization: AdamW with the schedule, gradient
    clipping and weight averaging of ``optimization``. Before training it checks the student's shape contract and
    fits the normalizers on the train split (or copies matching tensors from ``init_from``); a resumed run takes
    them from its checkpoint. Validation logs ``val/loss``, ``val/loss/<name>``, ``val/ssim`` and
    ``val/dynamic_range``, each a weighted mean over every rank from float sums and counts; testing logs the same
    under ``test/``.

    Parameters
    ----------
    reconstruction : str
        The waveorder configuration (YAML) the recorded targets were made with, or a saved model's folder.
    student : RestorationNetwork
        The network to train.
    index : str
        The index directory, for its voxel size; the CLI links it to ``data.index``.
    fields : list of str, optional
        The model's object fields the target channels hold, in their order; by default the first of the system.
    conditions : list of Condition, optional
        Conditions computed for every patch; the student's conditioning must read the same names and sizes.
    input_normalizer, target_normalizer : Normalizer, optional
        Normalizers of the input and the target; the target's must be invertible. ``Identity`` by default.
    losses : dict of str to Loss, optional
        Loss terms by name; ``{"l1": L1()}`` by default.
    weights : dict of str to float, optional
        The weight of each loss term by name, 1 by default.
    transforms : list of Transform, optional
        Augmentations of the training batches.
    optimization : Optimization, optional
        AdamW, the schedule, gradient clipping and weight averaging; ``Optimization()`` by default.
    monitor : str
        The validation metric that selects the best bundle.
    mode : {"max", "min"}
        Whether a larger or a smaller ``monitor`` is better.
    init_from : str, optional
        A bundle directory whose matching tensors start the student and the normalizers (a warm start).
    check : bool
        Check the student's shape contract before training.
    seed : int
        Seed of the transforms' random numbers; the CLI links it to ``seed_everything``.
    """

    task: Distillation

    def __init__(  # noqa: PLR0913 (one keyword per part of the run)
        self,
        reconstruction: str,
        student: RestorationNetwork,
        *,
        index: str = "",
        fields: list[str] | None = None,
        conditions: list[Condition] | None = None,
        input_normalizer: Normalizer | None = None,
        target_normalizer: Normalizer | None = None,
        losses: dict[str, Loss] | None = None,
        weights: dict[str, float] | None = None,
        transforms: list[Transform] | None = None,
        optimization: Optimization | None = None,  # a default instance breaks jsonargparse (slots, no vars())
        monitor: str = "val/ssim",
        mode: Mode = "max",
        init_from: str | None = None,
        check: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        losses = {"l1": L1()} if losses is None else dict(losses)
        weights = dict(weights or {})
        unknown = sorted(set(weights) - set(losses))
        if unknown:
            raise ParameterError(
                f"weights names {unknown}, which are no losses (losses: {sorted(losses)})", path="weights"
            )
        self.reconstruction = str(reconstruction)
        self.task = distillation(
            _reconstruction_model(self.reconstruction),
            student,
            pixel_size=_pixel_size(index),
            fields=fields,
            conditions=conditions or (),
            input_normalizer=input_normalizer,
            target_normalizer=target_normalizer,
            losses={name: (loss, weights.get(name, 1.0)) for name, loss in losses.items()},
            transforms=transforms or (),
        )
        self.optimization = Optimization() if optimization is None else optimization
        self.monitor, self.mode, self.init_from, self.check, self.seed = monitor, mode, init_from, check, seed
        self.generator = torch.Generator()
        self.initialized: dict[str, object] | None = None
        self._names = ["loss", *(f"loss/{name}" for name in self.task.losses), "ssim", "dynamic_range"]
        self._totals = torch.zeros(2, len(self._names), dtype=torch.float64)
        self._means: dict[str, float] | None = None

    # --- before training ---------------------------------------------------------------------------------------------

    @override
    def setup(self, stage: str) -> None:
        self.generator.manual_seed(draw_seed(self.seed, "transforms", self.global_rank))
        if stage != "fit":
            return
        data = self.datamodule().data  # set up before the module
        assert data is not None
        if self.check:
            planes = self.task.planes or data.index.planes
            self.task.student.check_shapes(planes=planes, sizes=[data.patches.size])
        if self.trainer.ckpt_path:  # resuming: the checkpoint holds the fitted normalizers
            return
        if self.init_from is not None:
            self.initialized = _initialize(self.task, load_bundle(self.init_from))
        else:
            self.task.fit_normalizers(data)

    @override
    def configure_callbacks(self) -> list[pl.Callback]:
        """Return the weight averaging of ``optimization.ema``, which runs after every callback of the trainer."""
        ema = self.optimization.ema
        return [] if ema is None else [EMAWeightAveraging(decay=ema)]

    @override
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        clip = self.optimization.gradient_clip if gradient_clip_val is None else gradient_clip_val
        if clip is not None:
            self.clip_gradients(optimizer, gradient_clip_val=clip, gradient_clip_algorithm=gradient_clip_algorithm)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = [parameter for parameter in self.task.parameters() if parameter.requires_grad]
        o = self.optimization
        optimizer = torch.optim.AdamW(parameters, lr=o.lr, weight_decay=o.weight_decay)
        # The batches of an epoch come from the index, not from trainer.estimated_stepping_batches: that iterates the
        # train loader now, before Lightning restores a restart's loops, so the loader's state would never load.
        per_epoch = max(1, self.datamodule().batches_per_epoch())
        trainer = self.trainer
        total = trainer.max_steps if trainer.max_steps > 0 else per_epoch * int(trainer.max_epochs or 1)
        configured = schedulers(optimizer, o, steps=(total, per_epoch), monitor=self.monitor, mode=self.mode)
        return [optimizer], configured  # ty: ignore[invalid-return-type] (Lightning takes scheduler dictionaries)

    @override
    def on_train_epoch_start(self) -> None:
        # Not called on the first epoch after a restart; the train loader's state and sampler set that epoch.
        self.datamodule().set_epoch(self.current_epoch)

    def datamodule(self) -> PatchDataModule:
        """Return the trainer's data module, which a run of this module always has."""
        module = getattr(self.trainer, "datamodule", None)  # a property ty does not see
        assert isinstance(module, PatchDataModule)
        return module

    # --- steps -------------------------------------------------------------------------------------------------------

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        del batch_idx
        result = self.task.step(batch, stage="train", generator=self.generator)
        scalars = result.scalars()
        if not bool(torch.isfinite(result.loss)):
            values = ", ".join(f"{name}={float(value):.4g}" for name, value in scalars.items())
            raise TrainingFailed(
                f"the loss became non-finite at step {self.global_step} (epoch {self.current_epoch}): {values}. "
                "Lower optimization.lr, keep optimization.gradient_clip set, and check that the normalizers were "
                "fitted on this data (blank or constant fields give statistics of zero)"
            )
        self.log_dict({f"train/{name}": value for name, value in scalars.items()}, batch_size=len(batch))
        return result.loss

    @override
    def on_train_epoch_end(self) -> None:
        if self.trainer.global_step == 0:
            raise TrainingFailed(
                "the first epoch ran no optimizer step: the train split gives no full batch; use more fields, more "
                "patches per field (data.per_field) or a smaller data.batch_size"
            )

    @override
    def on_validation_epoch_start(self) -> None:
        self._totals = torch.zeros(2, len(self._names), dtype=torch.float64, device=self.device)
        self._means = None

    on_test_epoch_start = on_validation_epoch_start

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        result = self.task.step(batch, stage="val")
        values = {**result.scalars(), **result.metrics}
        counts = dict.fromkeys(result.scalars(), len(batch)) | dict.fromkeys(result.metrics, result.scored)
        for number, name in enumerate(self._names):
            value, count = values.get(name), counts.get(name, 0)
            if value is not None and count and bool(torch.isfinite(value)):
                self._totals[0, number] += value.detach().double() * count
                self._totals[1, number] += count
        if batch_idx == 0 and self.trainer.is_global_zero:
            _log_panel(self.logger, self.task, batch, step=self.global_step)

    test_step = validation_step

    def validation_means(self) -> dict[str, float]:
        """Return this epoch's validation metrics by name, the weighted means over every rank.

        The float sums and counts of the ranks are summed (a collective call, made on every rank) once per epoch;
        :class:`~waveorder.train.lightning.BestBundle` calls it before the averaged weights are swapped out.
        """
        if self._means is None:
            totals = self.trainer.strategy.reduce(self._totals, reduce_op="sum")
            self._means = {
                name: float(totals[0, number] / totals[1, number])
                for number, name in enumerate(self._names)
                if totals[1, number] > 0
            }
        return self._means

    @override
    def on_validation_epoch_end(self) -> None:
        stage = "test" if self.trainer.testing else "val"
        self.log_dict({f"{stage}/{name}": value for name, value in self.validation_means().items()})

    on_test_epoch_end = on_validation_epoch_end

    # --- state -------------------------------------------------------------------------------------------------------

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["transforms_generator"] = self.generator.get_state()

    @override
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "transforms_generator" in checkpoint:
            self.generator.set_state(checkpoint["transforms_generator"])

    def describe(self) -> dict[str, object]:
        """Return the module as JSON-able values for ``run.json``."""
        return {
            "reconstruction": self.reconstruction,
            "model_config": self.task.model_config,
            "task": self.task.describe(),
            "optimization": dataclasses.asdict(self.optimization),
            "monitor": {"name": self.monitor, "mode": self.mode},
            "init_from": self.initialized,
        }


def _reconstruction_model(path: str) -> ReconstructionModel:
    """Return the model a configuration file, or a saved model's folder, describes."""
    return ReconstructionModel.load(path) if Path(path).is_dir() else ReconstructionModel.from_config(Path(path))


def _pixel_size(index: str) -> tuple[float, float, float]:
    """Return the voxel size an index's manifest records, reading nothing else."""
    if not index:
        raise ParameterError("DistillationModule needs the index it trains on (the CLI links data.index)", path="index")
    z, y, x = (float(size) for size in read_manifest(index)["pixel_size"])
    return (z, y, x)


def _initialize(task: Distillation, bundle: Bundle) -> dict[str, object]:
    """Copy the tensors of ``bundle`` whose names and shapes match into the task; return what was copied."""
    skipped: list[str] = []
    total = 0
    for key in ("student", "input_normalizer", "target_normalizer"):
        module = getattr(task, key)
        own, given = module.state_dict(), getattr(bundle, key).state_dict()
        matching = {name: value for name, value in given.items() if name in own and own[name].shape == value.shape}
        skipped.extend(f"{key}.{name}" for name in own if name not in matching)
        module.load_state_dict(matching, strict=False)
        total += len(own)
    return {"path": str(bundle.path), "sha256": bundle.sha256, "copied": total - len(skipped), "skipped": skipped}


def _log_panel(logger: object, task: Distillation, batch: Batch, *, step: int) -> None:
    """Log the task's panel of a validation batch when the logger takes images (W&B or TensorBoard)."""
    log_image = getattr(logger, "log_image", None)
    add_image = getattr(getattr(logger, "experiment", None), "add_image", None)
    if log_image is None and add_image is None:
        return
    panel = task.panel(batch)
    if panel is None:
        return
    if log_image is not None:
        from PIL import Image

        height, width = panel.shape  # an 8-bit PIL image, which a W&B logger takes without torchvision
        image = Image.frombytes("L", (width, height), bytes((panel * 255).round().to(torch.uint8).flatten().tolist()))
        log_image(key="val/examples", images=[image], step=step, caption=["input | target | student"])
    elif add_image is not None:
        add_image("val/examples", panel[None], global_step=step)


# --- the command line ---------------------------------------------------------------------------------------------


class _QuietParser(LightningArgumentParser):
    """jsonargparse prints the whole usage (about 70 lines) and its type machinery before an error; print one line."""

    @override
    def print_usage(self, file: object = None) -> None:
        del file

    @override
    def error(self, message: str, ex: Exception | None = None) -> NoReturn:
        if not self.exit_on_error:  # model.train, and jsonargparse's own parsers of nested types: raise
            super().error(message, ex)
        parser = getattr(ex, "subcommand_parser", None) or self
        sys.stderr.write(f"{parser.prog.replace(' [options]', '')}: error: {_condensed(message, parser)}\n")
        self.exit(2)


_NOISE = re.compile(
    r"For further information visit|Subtypes:|Errors:|Given value|Does not validate against any of the Union"
    r"|\d+ validation errors? for |Expected a <class 'NoneType'>|Problem with given class_path"
)
"""The lines of a jsonargparse error that describe its type machinery, not the mistake."""


def _condensed(message: str, parser: LightningArgumentParser) -> str:
    """Return a jsonargparse error as one line: the configuration key, then what is wrong with it."""
    found = re.match(r'Parser key "([^"]+)":\s*(.*)', message, flags=re.DOTALL)
    key, text = (found[1], found[2]) if found else ("", message)
    nested = re.search(r"NestedArg\(key='([^']+)'", text)
    key = f"{key}.{nested[1]}" if nested else key
    given = re.search(r"(?:Got value|Given value): (.*)", text)
    owner = re.search(r"Problem with given class_path '([^']+)'", text)
    keys = [action.dest for action in parser._actions]  # noqa: SLF001 (argparse has no public list)
    lines: list[str] = []
    for line in (line.strip().removeprefix("- ") for line in text.splitlines()):
        inner = re.match(r'Parser key "([^"]+)":\s*(.*)', line)
        if inner:  # a nested block's own key: optimization.lr
            key = key if key.endswith(f".{inner[1]}") else f"{key}.{inner[1]}"
            line = inner[2]
        unknown = re.match(r"(?:Subcommand|Group) '\w+' does not accept option '([^']+)'", line)
        option = re.match(r"Option '([^']+)' is not accepted", line)
        named = re.match(r"Expected a dot import path string: (\S+)", line)
        if unknown:
            key, line = unknown[1], "is not a configuration key" + _closest(unknown[1], keys)
        elif line.startswith("unrecognized arguments: "):
            flag = line.split()[2]
            key, line = flag, "is not a configuration key" + _closest(flag, [f"--{name}" for name in keys])
        elif option and owner:
            accepted = inspect.signature(_imported(owner[1])).parameters
            line = f"{owner[1].rpartition('.')[2]} takes no argument {option[1]!r}{_closest(option[1], accepted)}"
        elif named:
            line = f"no class is named {named[1]}: name a waveorder class (UNet, L1) or give an import path"
        elif not line or _NOISE.match(line):
            continue
        else:
            line = re.sub(r"\s*\[type=.*\]$|\.? Got value: .*$", "", line)  # pydantic's error code, the value
            line = re.sub(r"Expected a <class '(?:[\w.]+\.)?(\w+)'>", r"expected \1", line)
            line = line[:1].lower() + line[1:]
        if line not in lines:
            lines.append(line)
    text = "; ".join(lines)
    value = given[1].strip() if given else ""
    if value and not text.endswith(value) and "NestedArg" not in value:
        text += f", got {value}"
    return f"{key}: {text}" if key else text


def _closest(name: str, names: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(names), n=1)
    return f"; did you mean {close[0]}?" if close else ""


def _imported(path: str) -> type:
    """Return the class a configuration's ``class_path`` names."""
    module, _, name = path.rpartition(".")
    found = getattr(importlib.import_module(module), name)
    if not isinstance(found, type):
        raise ImportError(f"{path} is no class")
    return found


_FILES = {
    "model.reconstruction": "the waveorder configuration the targets were made with, or a saved model's folder",
    "data.index": "an index directory (wo distill index writes one)",
    "model.init_from": "a bundle directory",
}
"""The paths of a configuration, which a run records as absolute paths, and what each must be."""

_DESCRIPTIONS = {
    "fit": "Train a student on recorded reconstructions. Configuration files (--config, repeatable: a later one "
    "overrides an earlier one), then flags (--model.losses.ssim SSIM); a part is a waveorder class by its name "
    "(UNet, StridedVolume, L1) or any other by its import path. --ckpt_path RUN/checkpoints/last.ckpt resumes. "
    "Example: wo distill fit --config base.yml --config cytoland.yml --trainer.max_epochs 20",
    "validate": "Score a checkpoint on the val split. "
    "Example: wo distill validate --config RUN/config.yaml --ckpt_path RUN/checkpoints/last.ckpt",
    "test": "Score a checkpoint on patches of the test split. "
    "Example: wo distill test --config RUN/config.yaml --ckpt_path RUN/checkpoints/last.ckpt",
}


class DistillCLI(LightningCLI):
    """Lightning's command line for :class:`DistillationModule` and :class:`PatchDataModule`.

    Over Lightning's own it adds: ``data.index`` linked to ``model.index`` and ``seed_everything`` to both seeds; the
    files read and the loggers' folders made absolute, and the files checked, before the configuration is saved, so
    a run folder runs again from any directory; one located line instead of the usage or a traceback for a mistake
    in a configuration; the run's callbacks (the best and last bundles, the run record, ``last.ckpt`` every
    ``checkpoint_every`` steps and a checkpoint on SIGTERM); and ``checkpoint_every``.

    Parameters
    ----------
    args : list of str or dict, optional
        The command line after ``wo distill``, or a configuration as a mapping (with ``run=False``).
    run : bool
        Run the subcommand the arguments name; ``False`` only builds the module, the data and the trainer, and a
        mistake in the configuration raises :class:`~waveorder.errors.ConfigurationError` instead of exiting.
    """

    def __init__(self, args: ArgsType = None, *, run: bool = True) -> None:
        parser = {"prog": "wo distill"} | {name: {"description": text} for name, text in _DESCRIPTIONS.items()}
        super().__init__(
            DistillationModule,
            PatchDataModule,
            trainer_defaults=TRAINER_DEFAULTS,
            seed_everything_default=0,
            save_config_kwargs={"overwrite": True},
            parser_kwargs=parser if run else {"exit_on_error": False},
            parser_class=_QuietParser,
            auto_configure_optimizers=False,
            args=args,
            run=run,
        )

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint_every",
            type=PositiveInt,
            default=100,
            help="Optimizer steps between two writes of RUN/checkpoints/last.ckpt.",
        )
        parser.link_arguments("data.index", "model.index")
        parser.link_arguments("seed_everything", "data.seed")
        parser.link_arguments("seed_everything", "model.seed")

    @override
    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        # Lightning warns when args and sys.argv are both given, which is how wo distill fit and model.train call it.
        try:
            self.config = parser.parse_object(args) if isinstance(args, dict | Namespace) else parser.parse_args(args)
        except ArgumentError as error:  # only with run=False: a Python call
            key, _, message = _condensed(str(error), parser).partition(": ")
            raise ConfigurationError(key, message) if message else ConfigurationError("config", key) from None

    def _section(self, config: Namespace) -> Namespace:
        return config[self.subcommand] if self.subcommand else config

    def _refuse(self, key: str, message: str) -> NoReturn:
        """Exit with status 2 after one located line, or raise a ConfigurationError in a Python call."""
        if self.subcommand is None:
            raise ConfigurationError(key, message)
        sys.stderr.write(f"wo distill {self.subcommand}: error: {key}: {message}\n")
        raise SystemExit(2)

    @override
    def before_instantiate_classes(self) -> None:
        config = self._section(self.config)
        for key, what in _FILES.items():
            value = config.get(key)
            if not value:
                continue
            path = Path(str(value)).absolute()
            expected = {"data.index": MANIFEST, "model.init_from": "bundle.json"}.get(key)
            if not path.exists() or (expected is not None and not (path / expected).is_file()):
                self._refuse(key, f"{value} is not {what}")
            config[key] = str(path)
        config["model.index"] = config["data.index"]
        loggers = config.get("trainer.logger")
        for logger in loggers if isinstance(loggers, list) else [loggers]:  # outputs too: a rerun writes beside
            if isinstance(logger, Namespace) and logger.get("init_args.save_dir"):
                logger["init_args.save_dir"] = str(Path(str(logger["init_args.save_dir"])).absolute())
        if config.get("trainer.default_root_dir"):
            config["trainer.default_root_dir"] = str(Path(str(config["trainer.default_root_dir"])).absolute())

    @override
    def instantiate_classes(self) -> None:
        try:
            super().instantiate_classes()
        except WaveorderError as error:
            self._refuse(_located(error, self._section(self.config)), f"{type(error).__name__}: {error}")

    @override
    def instantiate_trainer(self, **kwargs: Any) -> pl.Trainer:
        config = self._section(self.config_init)
        given = config["trainer"].get("callbacks") or []
        module = self.model
        assert isinstance(module, DistillationModule)
        every = int(config["checkpoint_every"])
        run = [
            BestBundle(save_bundle, monitor=module.monitor, mode=module.mode),
            RunRecord(),
            LastCheckpoint(every),
            ExceptionCheckpoint(),
        ]
        return super().instantiate_trainer(callbacks=[*(given if isinstance(given, list) else [given]), *run], **kwargs)


def _located(error: BaseException, config: Namespace) -> str:
    """Return the configuration key of the part whose constructor raised ``error``: the innermost one configured.

    The error's ``path`` is added when it names an argument of that part (``model.student.head``).
    """
    keys: dict[type, str] = {DistillationModule: "model", PatchDataModule: "data"}
    for key, value in config.items():
        if key.endswith(".class_path") and isinstance(value, str):
            with contextlib.suppress(ImportError, AttributeError, ValueError):
                keys.setdefault(_imported(value), key.removesuffix(".class_path").replace(".init_args", ""))
    found, owner = "model", DistillationModule
    for frame, _ in traceback.walk_tb(error.__traceback__):
        cls = type(frame.f_locals.get("self"))
        if frame.f_code.co_name == "__init__" and cls in keys:
            found, owner = keys[cls], cls
    path = getattr(error, "path", None)
    if isinstance(path, str) and path.split(".")[0] in inspect.signature(owner).parameters:
        found = f"{found}.{path}"
    return found


def main(args: list[str]) -> None:
    """Run ``wo distill fit|validate|test`` with ``args``; a mistake in a part's arguments exits with status 2."""
    with quiet():
        try:
            DistillCLI(args)
        except WaveorderError as error:
            sys.stderr.write(f"wo distill {args[0]}: error: {type(error).__name__}: {error}\n")
            raise SystemExit(2) from None


def configured(value: object) -> object:
    """Return a value of a Python call as configuration data: parts as class specs, dataclasses as their fields."""
    from waveorder.registry import class_spec

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): configured(entry) for key, entry in value.items()}
    if isinstance(value, list | tuple):
        return [configured(entry) for entry in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: configured(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return class_spec(value)
