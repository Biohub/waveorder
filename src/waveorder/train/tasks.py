"""Training tasks: a :class:`Task` computes the loss of a batch; :class:`Distillation` trains a student network.

A task is plain torch; :class:`waveorder.models.DistillationModule` drives :meth:`Task.step` with Lightning.
Nothing here runs the imaging model: the targets are recorded reconstructions.
"""

import copy
import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import torch
from torch import Tensor, nn

from waveorder.data import Batch, PatchData, Transform
from waveorder.data.loader import Prepare
from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn import Distilled, ImagingModel
from waveorder.nn.networks import Bundle, Condition, Contract, Example, Identity, Normalizer, RestorationNetwork
from waveorder.registry import class_spec
from waveorder.train.losses import L1, Loss
from waveorder.train.metrics import dynamic_range_ratio, ssim

type Stage = Literal["train", "val"]
"""What a step is for: ``"train"`` applies the task's transforms, ``"val"`` also returns the metrics."""

_EXAMPLE_SAMPLES = 2
"""Samples of a batch kept as a bundle's test tensors."""

_THINNED = 256
"""Largest side, in pixels, of what one field contributes to a fitted normalizer and of a panel's images."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """What one step of a task returns.

    Attributes
    ----------
    loss : Tensor
        The scalar to minimize, with the graph to the trained parameters.
    terms : Mapping[str, Tensor]
        Each unweighted, detached loss term by name.
    metrics : Mapping[str, Tensor]
        Validation metrics by name, scalar means over the ``scored`` samples (``NaN`` when none); empty in training.
    scored : int
        Number of samples the metrics average over (their weight across batches).
    """

    loss: Tensor
    terms: Mapping[str, Tensor] = field(default_factory=dict)
    metrics: Mapping[str, Tensor] = field(default_factory=dict)
    scored: int = 0

    def scalars(self) -> dict[str, Tensor]:
        """Return the loss and its terms, detached, by the names they are logged under: ``loss``, ``loss/<term>``."""
        return {"loss": self.loss.detach(), **{f"loss/{name}": value for name, value in self.terms.items()}}


class Task(nn.Module, ABC):
    """What a fit trains: a module whose parameters are trained and whose :meth:`step` gives the loss of a batch."""

    @abstractmethod
    def step(self, batch: Batch, *, stage: Stage = "train", generator: torch.Generator | None = None) -> StepResult:
        """Return the loss of a batch the loader finished, on the task's device.

        ``stage="val"`` also returns the metrics; ``generator`` is the CPU generator random augmentations draw from.
        """

    def panel(self, batch: Batch) -> Tensor | None:
        """Return one image ``(Y, X)`` in ``[0, 1]`` that shows the task on a validation batch; ``None`` for none."""
        del batch


class Distillation(Task):
    """A student network trained to reproduce a physics solver's recorded reconstructions.

    The imaging model is held outside the module and never trained or run; it defines the contract and what the
    bundle records. The loader applies ``prepare`` and both normalizers, so :meth:`step` takes normalized batches.

    Parameters
    ----------
    imaging_model : ImagingModel
        The generative model the targets were made with.
    student : RestorationNetwork
        Must take the system's channel count, return one field per ``contract.fields`` and read ``conditions``.
    contract : Contract
        What the student takes and returns; recorded in the bundle and checked by the solver.
    prepare : callable, optional
        ``prepare(data) -> Tensor``: a recorded field ``(C, Z, Y, X)`` to what the system measures; must be picklable.
    conditions : sequence of Condition
        The named vectors the student reads besides the data.
    input_normalizer, target_normalizer : Normalizer, optional
        Default :class:`~waveorder.nn.networks.Identity`. The target normalizer needs an inverse.
    losses : mapping of str to Loss or (Loss, float), optional
        Loss terms by name with optional weight (default 1); the loss is their weighted sum. Default ``{"l1": L1()}``.
    transforms : sequence of Transform
        Augmentations applied in order to training batches, drawing from the generator given to :meth:`step`.
    model_config : mapping, optional
        The user-level model configuration as JSON-able values, stored in the bundle.

    Raises
    ------
    ParameterError
        If a loss weight is not finite, or a loss name is empty or holds a ``/`` or a ``.``.
    IncompatibleModelError
        If the contract, student, conditions and imaging model do not fit, or the target normalizer has no inverse.
    """

    student: RestorationNetwork
    input_normalizer: Normalizer
    target_normalizer: Normalizer
    imaging_model: ImagingModel

    def __init__(  # noqa: PLR0913 (the model and the student, and one keyword per part of the task)
        self,
        imaging_model: ImagingModel,
        student: RestorationNetwork,
        *,
        contract: Contract,
        prepare: Prepare | None = None,
        conditions: Sequence[Condition] = (),
        input_normalizer: Normalizer | None = None,
        target_normalizer: Normalizer | None = None,
        losses: Mapping[str, Loss | tuple[Loss, float]] | None = None,
        transforms: Sequence[Transform] = (),
        model_config: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        # A plain attribute, not a module of the task: the model's quantities are not trained.
        object.__setattr__(self, "imaging_model", imaging_model)
        self.student = student
        self.contract = contract
        self.fields, self.pixel_size, self.planes = contract.fields, contract.pixel_size, contract.planes
        self.prepare = prepare
        self.conditions = tuple(conditions)
        self.input_normalizer = Identity() if input_normalizer is None else input_normalizer
        self.target_normalizer = Identity() if target_normalizer is None else target_normalizer
        self.transforms = tuple(transforms)
        self.model_config = None if model_config is None else dict(model_config)
        self.example: Example | None = None
        self._expect_fit()
        terms, self.weights = _terms({"l1": L1()} if losses is None else losses)
        self.losses = nn.ModuleDict(terms)
        self._generator = torch.Generator().manual_seed(0)

    def _expect_fit(self) -> None:
        """Raise unless the contract fits the imaging model, and the student the contract and the conditions."""
        student, system = self.student, self.imaging_model.system
        name = type(student).__name__
        problems = self.contract.model_problems(self.imaging_model)
        if student.channels != system.channel_count:
            raise IncompatibleModelError(
                f"Distillation: the student {name} takes {student.channels} channel(s), but the model's "
                f"{type(system).__name__} measures {system.channel_count}; build the student with "
                f"channels={system.channel_count}"
            )
        if problems:
            raise IncompatibleModelError(f"Distillation: the contract does not fit the model: {'; '.join(problems)}")
        if student.fields != len(self.fields):
            raise IncompatibleModelError(
                f"Distillation: the student {name} returns {student.fields} field(s), but fields names "
                f"{len(self.fields)}: {self.fields}; name one field of the model's {type(system).__name__} "
                f"{system.fields} per target channel, in the order of the target channels, or build the student "
                f"with fields={len(self.fields)}"
            )
        given = {condition.name: condition.size for condition in self.conditions}
        read = dict(student.condition_sizes)
        if given != read:
            raise IncompatibleModelError(
                f"Distillation: the student {name} reads the conditions {read} (name: size), but the task gives "
                f"{given}; pass the same conditions to both: build the student for these, or change conditions=(...)"
            )
        if not self.target_normalizer.invertible:
            raise IncompatibleModelError(
                f"Distillation: the target normalizer {type(self.target_normalizer).__name__} has no inverse, so "
                "the student's output could not be returned in the model's units; use Affine.robust() or "
                "Identity() for the target"
            )

    def describe(self) -> dict[str, object]:
        """Return the task as JSON-able values for the run record."""
        return {
            "model": self.contract.model,
            "fields": list(self.fields),
            "pixel_size": list(self.pixel_size),
            "student": class_spec(self.student),
            "conditions": [class_spec(condition) for condition in self.conditions],
            "input_normalizer": class_spec(self.input_normalizer),
            "target_normalizer": class_spec(self.target_normalizer),
            "losses": {
                name: {"class": f"{type(loss).__module__}.{type(loss).__qualname__}", "weight": self.weights[name]}
                for name, loss in self.losses.items()
            },
            "transforms": [repr(transform) for transform in self.transforms],
        }

    @override
    def step(self, batch: Batch, *, stage: Stage = "train", generator: torch.Generator | None = None) -> StepResult:
        """Return the loss of a normalized batch and each of its terms.

        ``stage="train"`` first applies the transforms, drawing from ``generator`` (default: the task's own, seeded
        with 0). ``stage="val"`` adds the metrics ``"ssim"`` and ``"dynamic_range"`` and holds the first batch as the
        test tensors of :meth:`bundle`.

        Raises
        ------
        ParameterError
            If ``stage`` is neither ``"train"`` nor ``"val"``.
        ShapeError
            If the target does not have one channel per field, or the output is not shaped as the target.
        """
        if stage not in ("train", "val"):
            raise ParameterError(f"Distillation.step stage must be 'train' or 'val', got {stage!r}", path="stage")
        if stage == "train":
            for transform in self.transforms:
                batch = transform(batch, generator=self._generator if generator is None else generator)
        elif self.example is None:
            self.hold(batch)
        target = batch.target
        if target.ndim != 5 or target.shape[1] != len(self.fields):
            raise ShapeError(
                f"Distillation.step: batch.target has shape {tuple(target.shape)}, but the task distills "
                f"{len(self.fields)} field(s), {self.fields}, so targets are (B, {len(self.fields)}, Zout, Y, X); "
                "the index's target_channels must name one channel per field, in that order, or name the fields "
                "the target channels hold in the task's fields"
            )
        estimate = self.student(batch.data, batch.condition or None).float()
        terms: dict[str, Tensor] = {}
        total = estimate.new_zeros(())
        for name, loss in self.losses.items():
            value = loss(estimate, batch)
            terms[name] = value.detach()
            total = total + self.weights[name] * value
        if stage != "val":
            return StepResult(total, terms)
        with torch.no_grad():
            similarity = ssim(estimate, target)
            scored = torch.isfinite(similarity)
            metrics = {
                "ssim": similarity.nanmean(),
                "dynamic_range": dynamic_range_ratio(estimate, target)[scored].nanmean(),
            }
        return StepResult(total, terms, metrics, int(scored.sum()))

    @torch.no_grad()
    def predict(self, batch: Batch) -> Tensor:
        """Return the student's eval-mode estimate of a batch in the model's units, ``(B, F, Zout, Y, X)``."""
        training = self.training
        self.eval()
        try:
            return self.target_normalizer.inverse(self.student(batch.data, batch.condition or None).float())
        finally:
            self.train(training)

    @override
    def panel(self, batch: Batch) -> Tensor | None:
        """Return one ``(Y, 3 X)`` image in ``[0, 1]``: input, target and student mid planes of the first sample.

        Target and student share the target's range, in the model's units.
        """
        first = Batch(
            data=batch.data[:1],
            target=batch.target[:1],
            condition={name: value[:1] for name, value in batch.condition.items()},
            position=batch.position[:1],
            origin=batch.origin[:1],
        )
        student = self.predict(first)[0, 0]
        with torch.no_grad():
            target = self.target_normalizer.inverse(batch.target[:1].float())[0, 0]

        def plane(volume: Tensor) -> Tensor:
            return _thinned(volume[volume.shape[0] // 2])

        def unit(image: Tensor, low: Tensor, high: Tensor) -> Tensor:
            return ((image - low) / (high - low).clamp_min(1e-12)).clamp(0, 1)

        data = plane(first.data[0, 0].float())
        low, high = plane(target).amin(), plane(target).amax()
        images = [unit(data, data.amin(), data.amax()), unit(plane(target), low, high), unit(plane(student), low, high)]
        return torch.cat(images, dim=1).cpu()

    def fit_normalizers(self, data: PatchData, *, fields: int = 32) -> None:
        """Fit both normalizers on at most ``fields`` fields of the train split, thinned to 256 pixels a side.

        Call before the first loader is made: loader workers copy the normalizers when they start.

        Raises
        ------
        ParameterError
            If ``fields`` is below 1, or the train split has no field to read.
        """
        if fields < 1:
            raise ParameterError(
                f"Distillation.fit_normalizers fields must be a positive int, got {fields!r}", path="fields"
            )
        inputs: list[Tensor] = []
        targets: list[Tensor] = []
        for loaded in data.fields("train", limit=fields):
            inputs.append(_thinned(loaded.data))
            targets.append(_thinned(loaded.target))
        if not inputs:
            raise ParameterError(
                "Distillation.fit_normalizers found no field to read in the 'train' split of the index (blank fields "
                "are skipped); check the index's split column",
                path="data",
            )
        self.input_normalizer.fit(inputs)
        self.target_normalizer.fit(targets)

    def hold(self, batch: Batch) -> None:
        """Keep the first samples of a batch, un-normalized where possible, as the test tensors of :meth:`bundle`."""
        data = batch.data[:_EXAMPLE_SAMPLES].detach()
        if self.input_normalizer.invertible:
            with torch.no_grad():
                data = self.input_normalizer.inverse(data)
        condition = {name: value[:_EXAMPLE_SAMPLES].detach().cpu() for name, value in batch.condition.items()}
        self.example = Example(data=data.float().cpu(), condition=condition)

    def bundle(self, *, record: Mapping[str, object] | None = None, test: Batch | Example | None = None) -> Bundle:
        """Return the student with everything inference needs as a :class:`~waveorder.nn.networks.Bundle`.

        Holds eval-mode copies of the student and normalizers, the contract, the model record, ``record`` and test
        tensors with their CPU output (tolerance 1e-3 of its largest magnitude). ``test`` defaults to the batch held
        by the first validation step. :func:`waveorder.io.bundle.save_bundle` writes it.

        Raises
        ------
        ParameterError
            If no ``test`` was given and no validation step has run.
        """
        if isinstance(test, Batch):
            self.hold(test)
            test = self.example
        example = self.example if test is None else test
        if example is None:
            raise ParameterError(
                "Distillation.bundle needs test tensors, and the task holds none: pass test=<a batch of the loader>, "
                "or run task.step(batch, stage='val') once before",
                path="test",
            )
        parts = self._bundle(record=record, test=example)
        expected = parts.test_output("cpu").float()
        scale = float(expected.abs().max())
        tolerance = 1e-3 * scale if math.isfinite(scale) and scale > 0 else 1e-6
        test = Example(data=example.data, condition=example.condition, expected=expected, tolerance=tolerance)
        return dataclasses.replace(parts, test=test)

    def _bundle(self, *, record: Mapping[str, object] | None = None, test: Example | None = None) -> Bundle:
        """Return copies of the student and the normalizers in eval mode, as a bundle with the task's contract."""
        return Bundle(
            student=copy.deepcopy(self.student).eval(),
            contract=self.contract,
            conditions=self.conditions,
            input_normalizer=copy.deepcopy(self.input_normalizer).eval(),
            target_normalizer=copy.deepcopy(self.target_normalizer).eval(),
            model=self._model_record(),
            record=dict(record or {}),
            test=test,
        )

    def _model_record(self) -> dict[str, object]:
        """Return the model for the bundle: its parts' class names, and its configuration when the task has one."""
        model = self.imaging_model
        parts = {"system": model.system, "object_prior": model.object_prior, "noise": model.noise}
        described: dict[str, object] = {
            "model": self.contract.model,
            **{name: type(part).__name__ for name, part in parts.items()},
        }
        if self.model_config is not None:
            described["config"] = self.model_config
        return described

    def as_solver(self, **kwargs: object) -> Distilled:
        """Return the student as an eval-mode solver with copies of the current weights.

        ``kwargs`` are the other arguments of :class:`waveorder.nn.Distilled` (``attributes``, ``precision``).
        """
        return Distilled(self._bundle(), **kwargs)  # ty: ignore[invalid-argument-type]


def _terms(losses: Mapping[str, Loss | tuple[Loss, float]]) -> tuple[dict[str, Loss], dict[str, float]]:
    """Return the losses and their weights by name, after checking the names and the weights."""
    if not losses:
        raise ParameterError("Distillation losses must name at least one loss, e.g. {'l1': L1()}", path="losses")
    terms: dict[str, Loss] = {}
    weights: dict[str, float] = {}
    for name, entry in losses.items():
        loss, weight = entry if isinstance(entry, tuple) else (entry, 1.0)
        if not name or "." in name or "/" in name:
            raise ParameterError(
                f"Distillation losses cannot be named {name!r}: a name is a non-empty string without '.' or '/' "
                "(it is logged as 'loss/<name>'); rename the term",
                path=f"losses.{name}",
            )
        if not math.isfinite(weight):
            raise ParameterError(
                f"Distillation losses.{name} weight must be a finite number, got {weight!r}", path=f"losses.{name}"
            )
        terms[name] = loss
        weights[name] = float(weight)
    return terms, weights


def _thinned(image: Tensor) -> Tensor:
    """Return an image ``(..., Y, X)`` subsampled along y and x to at most 256 pixels a side."""
    stride = max(1, math.ceil(max(image.shape[-2:]) / _THINNED))
    return image[..., ::stride, ::stride]
