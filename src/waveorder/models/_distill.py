"""Distillation of a user-level model: its imaging model, ``prepare`` and contract for ``waveorder.train``.

:func:`distillation` builds the task :class:`waveorder.models.DistillationModule` trains; :func:`evaluate` and
:func:`export` take its bundle.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from waveorder import train
from waveorder.config import V4Config, dump
from waveorder.data import PatchData, Transform
from waveorder.errors import IncompatibleModelError, Rejected
from waveorder.io.bundle import load_bundle, record_file
from waveorder.models._arguments import PixelSize as VoxelSize
from waveorder.models._arguments import expect_instance
from waveorder.models._arguments import pixel_size as voxel_size
from waveorder.models._base import ReconstructionModel
from waveorder.models._stores import system_sha256
from waveorder.nn import Distilled
from waveorder.nn.networks import Bundle, Condition, Contract, Normalizer, RestorationNetwork
from waveorder.train.evaluate import as_solver


@dataclass(frozen=True, slots=True)
class Prepare:
    """The ``prepare`` of ``model`` as a picklable callable for loader workers; build it with :func:`prepare`."""

    model: ReconstructionModel

    def __call__(self, data: Tensor) -> Tensor:
        """Return a recorded field ``(C, Z, Y, X)`` as the model's system measures it, in float32."""
        return self.model.prepare(data).to(torch.float32)


def prepare(model: ReconstructionModel) -> Prepare:
    """Return the callable to give :class:`~waveorder.data.PatchData` as ``prepare``: the model's, picklable.

    It holds a copy of the model on the CPU, made now, so that a worker which unpickles it never touches a GPU.
    """
    return Prepare(copy.deepcopy(model).to("cpu"))


def distillation(  # noqa: PLR0913 (the model and the student, and one keyword per part of the task)
    model: ReconstructionModel,
    student: RestorationNetwork,
    *,
    pixel_size: VoxelSize,
    fields: Sequence[str] | None = None,
    conditions: Sequence[Condition] = (),
    input_normalizer: Normalizer | None = None,
    target_normalizer: Normalizer | None = None,
    losses: Mapping[str, train.Loss | tuple[train.Loss, float]] | None = None,
    transforms: Sequence[Transform] = (),
    planes: Sequence[int] | None = None,
) -> train.Distillation:
    """Return the task that trains ``student`` to give the estimates of the solver of ``model``.

    ``model`` is the model the recorded targets were made with. It supplies ``task.prepare``, the contract and the
    configuration the bundle records; its solver is not run. The trained student later runs in it as its solver
    (``model.solver = Distilled(load_bundle(...))``).

    Parameters
    ----------
    model : ReconstructionModel
        The model the targets were made with.
    student : RestorationNetwork
        The network to train.
    pixel_size : sequence of three floats or PixelSize
        Voxel size of the training data along z, y, x in um; recorded in the contract.
    fields : sequence of str, optional
        The model's object fields the target channels hold, in order (``("phase",)`` for a ``Phase2D`` target of a
        system with ``phase`` and ``absorption``); by default the first ``student.fields`` of the system's fields.
    conditions, input_normalizer, target_normalizer, losses, transforms
        As :class:`waveorder.train.Distillation` takes them.
    planes : sequence of int, optional
        The plane counts the contract accepts; by default whatever the student accepts.

    Raises
    ------
    ParameterError
        If an argument is not of its type or ``pixel_size`` is not three positive numbers.
    IncompatibleModelError
        If ``fields`` are not distinct fields of the model's system, or the student does not fit the model and the
        conditions (see :class:`waveorder.train.Distillation`).
    """
    where = "Distillation"
    expect_instance(model, ReconstructionModel, where=where, name="model")
    expect_instance(student, RestorationNetwork, where=where, name="student")
    size = voxel_size(pixel_size, where=where)
    system = model.system
    chosen = tuple(system.fields[: student.fields] if fields is None else fields)
    if not chosen or len(set(chosen)) != len(chosen) or any(name not in system.fields for name in chosen):
        raise IncompatibleModelError(
            f"Distillation fields must be distinct object fields of the model's {type(system).__name__}, "
            f"{system.fields}, got {chosen}"
        )
    contract = Contract.of(
        model.imaging_model,
        pixel_size=size,
        model=type(model).__name__,
        prepare_scope=type(model).prepare_scope,
        planes=planes,
        fields=chosen,
        sha256=system_sha256(model),
    )
    channels = [f"channel_{index}" for index in range(system.channel_count)]
    described = model.to_config(channels=channels, pixel_size=size)  # a model built in Python: its description
    config = dump(described) if isinstance(described, V4Config) else described
    return train.Distillation(
        model.imaging_model,
        student,
        contract=contract,
        prepare=prepare(model),
        conditions=conditions,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        losses=losses,
        transforms=transforms,
        model_config=config,
    )


def evaluate(
    solver_or_bundle: Distilled | Bundle | str | Path,
    data: PatchData,
    *,
    model: ReconstructionModel,
    split: str = "test",
    device: str | torch.device | None = None,
    limit_fields: int | None = None,
) -> dict[str, object]:
    """Score a student as the solver of ``model`` on whole fields: :func:`waveorder.train.evaluate` through a model.

    The student runs on the model's imaging model with the model's z padding, against each field's recorded target.
    A ``z_padding`` of ``"auto"`` is resolved for each field's stack (:meth:`ReconstructionModel.z_padding_for` at
    the student's voxel size). The model is moved to ``device`` when given.

    Raises
    ------
    IncompatibleModelError
        If the student cannot run on the model, or was trained for another imaging system (the contract's hash).
    ShapeError
        If the model's ``z_padding`` is ``"auto"`` and a field's stack has no more slices than it pads.
    """
    expect_instance(model, ReconstructionModel, where="evaluate", name="model")
    solver = as_solver(load_bundle(solver_or_bundle) if isinstance(solver_or_bundle, str | Path) else solver_or_bundle)
    solver.check(model.imaging_model)
    model._expect_trained_system(solver, where="evaluate")  # noqa: SLF001 (the model's own check of a student)
    if device is not None:
        model.to(device)
    z_padding = model.z_padding
    return train.evaluate(
        solver,
        data,
        split=split,
        imaging_model=model.imaging_model,
        z_padding=z_padding
        if isinstance(z_padding, int)
        else lambda sampling: model.z_padding_for(sampling.shape, pixel_size=sampling.pixel_size),
        device=device,
        limit_fields=limit_fields,
    )


def recorded_model(bundle: Bundle) -> ReconstructionModel:
    """Return the model a bundle records the configuration or description of, to prepare data as its training did.

    Raises
    ------
    Rejected
        Located at the bundle, if it records no configuration of its model or the recorded one does not build.
    """
    config = bundle.model.get("config")
    if not isinstance(config, Mapping):
        raise Rejected.one(
            f"records no configuration of its model ({bundle.model.get('model')}); pass the model's with --model",
            location=str(bundle.path),
        )
    try:
        if config.get("schema") == "waveorder.model/1":  # a model built in Python
            return ReconstructionModel.load(config)
        return ReconstructionModel.from_config(config)
    except Rejected as error:  # the recorded model is located at the bundle that records it
        here = f"{bundle.path} (its recorded model)"
        raise Rejected([replace(problem, location=here) for problem in error.problems]) from error


def export(bundle: Bundle | str | Path, path: str | Path | None = None) -> Path:
    """Write a bundle's student as an ONNX file (:func:`waveorder.train.export_onnx`) and return the file.

    ``bundle`` is a bundle or a bundle directory. A file written inside the bundle's directory (by default
    ``model.onnx``) is recorded in its ``bundle.json``, so a later load refuses an altered one.

    Raises
    ------
    BundleError
        As :func:`waveorder.io.bundle.load_bundle` and :func:`waveorder.train.export_onnx`.
    """
    loaded = load_bundle(bundle) if isinstance(bundle, str | Path) else bundle
    written = train.export_onnx(loaded, path)
    if loaded.path is not None and written.parent.resolve() == loaded.path.resolve():
        record_file(loaded.path, written.name)
    return written
