"""Score a distilled student on whole fields of an index split: :func:`evaluate`.

Estimates are compared with each field's recorded target, in the model's units.
"""

import time
from collections.abc import Callable

import torch
from torch import Tensor

from waveorder.data import Field, PatchData
from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn import Distilled, ImagingModel, Sampling, Solver, ZPadding, solve_context
from waveorder.nn.networks import Bundle, predict
from waveorder.nn.parameters import placement
from waveorder.nn.sampling import ZYX
from waveorder.train.metrics import dynamic_range_ratio, passband_error, ssim, summarize, transfer_support


def evaluate(  # noqa: PLR0913 (the student and the data, and one keyword per setting of the scoring)
    solver_or_bundle: Distilled | Bundle,
    data: PatchData,
    *,
    split: str = "test",
    imaging_model: ImagingModel | None = None,
    z_padding: int | Callable[[Sampling], int] = 0,
    device: str | torch.device | None = None,
    limit_fields: int | None = None,
) -> dict[str, object]:
    """Run a student on the whole fields of a split and score each estimate against the field's recorded target.

    Without ``imaging_model`` the student runs directly on each prepared field. With it, the student runs as the
    solver of that model, so the contract check, padding and field placement are part of what is scored.

    Parameters
    ----------
    solver_or_bundle : Distilled or Bundle
        The student (:func:`waveorder.io.bundle.load_bundle` reads a bundle).
    data : PatchData
        Built as for training; its normalizers are not used, the student's own are.
    split : str
        The split of the index to score.
    imaging_model : ImagingModel, optional
        The model the targets were made with. Needed for the passband error (shift-invariant systems only).
    z_padding : int or callable
        Mirrored slices added at each end of z for every solve through ``imaging_model``, or a function of a field's
        measured sampling that returns them, called per field.
    device : str or torch.device, optional
        Where to run; default where the imaging model is, else where the student is.
    limit_fields : int, optional
        Score at most this many fields, evenly spaced over the split.

    Returns
    -------
    dict
        ``"split"``, ``"fields"`` (scored field names), ``"per_field"`` (one ``(N,)`` tensor per metric: ``"ssim"``,
        ``"dynamic_range"``, ``"seconds"``, and ``"passband_inside"``/``"passband_outside"`` with a shift-invariant
        ``imaging_model``), ``"summary"`` (:func:`~waveorder.train.metrics.summarize` per metric) and
        ``"seconds_per_field"`` (mean, first field excluded as warm-up when there are more).

    Raises
    ------
    BundleError
        If a bundle directory is incomplete or altered.
    ParameterError
        If the split has no field to read.
    IncompatibleModelError
        If the student cannot run on ``imaging_model`` or on the data.
    ShapeError
        If a target does not have the shape of the student's estimate.
    """
    solver = as_solver(solver_or_bundle)

    target = device if device is not None or imaging_model is None else placement(imaging_model.system)[1]
    for module in (solver, imaging_model):
        if module is not None and target is not None:
            module.to(target)
    contract = solver.contract
    problems = [] if imaging_model is None else contract.model_problems(imaging_model)
    if problems:
        raise IncompatibleModelError(f"evaluate: the student does not fit the imaging model: {'; '.join(problems)}")
    fields = () if imaging_model is None else imaging_model.system.fields
    kept = [fields.index(name) for name in contract.fields if name in fields]
    names: list[str] = []
    values: dict[str, list[Tensor]] = {}
    supports: dict[tuple[int, int, int], Tensor | None] = {}
    for field in data.fields(split, limit=limit_fields):
        _, planes, height, width = field.data.shape
        name, shape = field.row.name, (planes, height, width)
        if imaging_model is None:
            estimate, seconds = _direct(solver, field)
        else:
            measured = Sampling(shape=shape, pixel_size=contract.pixel_size)
            slices = z_padding if isinstance(z_padding, int) else z_padding(measured)
            estimate, seconds = _through(imaging_model, solver, field, z_padding=slices)
            estimate = estimate[kept]
        if imaging_model is not None and shape not in supports:
            supports[shape] = _support(imaging_model, kept, Sampling(shape=shape, pixel_size=contract.pixel_size))
        scores = _scores(estimate, field.target, supports.get(shape), where=f"{name} with fields {contract.fields}")
        scores["seconds"] = torch.tensor([seconds])
        for key, value in scores.items():
            values.setdefault(key, []).append(value)
        names.append(name)
    if not names:
        raise ParameterError(
            f"evaluate found no field to read in the {split!r} split of the data, which has "
            f"{data.index.counts or 'no rows'} (blank fields are skipped)",
            path="split",
        )
    per_field = {key: torch.cat(listed) for key, listed in values.items()}
    seconds = per_field["seconds"]
    return {
        "split": split,
        "fields": names,
        "per_field": per_field,
        "summary": {key: summarize(listed) for key, listed in per_field.items()},
        "seconds_per_field": float((seconds[1:] if len(seconds) > 1 else seconds).mean()),
    }


def as_solver(source: Distilled | Bundle) -> Distilled:
    """Return a student as a solver: the solver itself, or the one of a bundle."""
    return source if isinstance(source, Distilled) else Distilled(source)


def _timed(device: torch.device) -> float:
    """Return the time now, after the device finished what it was given."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _direct(solver: Distilled, field: Field) -> tuple[Tensor, float]:
    """Return the student's estimate ``(F, Zout, Y, X)`` of one prepared field and the seconds it took."""
    _, device = placement(solver.student)
    data = field.data.to(device)[None]
    given = {name: value.to(device)[None] for name, value in field.condition.items()}
    start = _timed(device)
    with torch.no_grad():
        estimate = predict(
            solver.student,
            data,
            given,
            input_normalizer=solver.input_normalizer,
            target_normalizer=solver.target_normalizer,
            precision=solver.precision,
        )[0]
    return estimate, _timed(device) - start


def _through(imaging_model: ImagingModel, solver: Distilled, field: Field, *, z_padding: int) -> tuple[Tensor, float]:
    """Return the estimate ``(F, Z, Y, X)`` of one prepared field with ``solver`` on the model, and its seconds."""
    _, device = placement(imaging_model.system)
    height, width = field.data.shape[-2:]
    start = _timed(device)
    with solve_context(position=field.row.position, time=field.row.time, field_shape=(height, width)):
        estimate = _solved(imaging_model, solver, field.data, pixel_size=solver.contract.pixel_size, z=z_padding)
    return estimate, _timed(device) - start


def _solved(imaging_model: ImagingModel, solver: Solver, data: Tensor, *, pixel_size: ZYX[float], z: int) -> Tensor:
    """Return the solver's estimate ``(F, Z, Y, X)`` of one field ``(C, Z, Y, X)``, padded by ``z`` mirrored slices."""
    dtype, device = placement(imaging_model.system)
    padding = ZPadding(z)
    sampling = padding.sampling(Sampling(shape=(data.shape[1], data.shape[2], data.shape[3]), pixel_size=pixel_size))
    with torch.no_grad():
        imaging_model.check(solver, sampling)
        measured = padding.pad(data.to(device)).to(dtype)
        operator = imaging_model.system.bind(sampling) if solver.needs_operator else None
        return padding.crop(solver(measured, imaging_model, sampling, operator=operator)).estimate


def _scores(estimate: Tensor, target: Tensor, support: Tensor | None, *, where: str) -> dict[str, Tensor]:
    """Return each metric ``(1,)`` of one field's estimate against its target, both ``(F, Zout, Y, X)``."""
    estimate, target = estimate.float().cpu()[None], target.float().cpu()[None]
    if estimate.shape != target.shape:
        raise ShapeError(
            f"evaluate: the student's estimate of {where} is {tuple(estimate.shape[1:])}, but the target is "
            f"{tuple(target.shape[1:])} (F, Zout, Y, X); the index's target_channels must hold the fields the "
            "student returns, in that order"
        )
    scores = {"ssim": ssim(estimate, target), "dynamic_range": dynamic_range_ratio(estimate, target)}
    if support is not None:
        error = passband_error(estimate, target, support)
        scores["passband_inside"], scores["passband_outside"] = error["inside"], error["outside"]
    return scores


def _support(imaging_model: ImagingModel, kept: list[int], sampling: Sampling) -> Tensor | None:
    """Return the passband of the student's fields on ``sampling``, or ``None`` for a shift-variant system."""
    try:
        support = transfer_support(imaging_model.system, sampling)
    except IncompatibleModelError:
        return None
    return support[kept].cpu()
