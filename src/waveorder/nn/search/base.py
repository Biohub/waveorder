"""Calibration: estimate the imaging system's free quantities from data by minimizing a score plus their prior.

:func:`calibrate` runs search stages (:class:`SearchStage`) in order, each from the best point so far, on the
:class:`CalibrationObjective` defined here, and updates the model in place to the estimate. Stages chain on the device;
:func:`calibrate` reads back to the host once, after the last stage.
"""

import math
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import ClassVar, Literal, NamedTuple, Protocol, Self, runtime_checkable

import torch
from torch import Tensor
from torch.func import vmap

from waveorder.errors import CalibrationWarning, IncompatibleModelError, ParameterError
from waveorder.nn.model import ImagingModel, expect_on_system
from waveorder.nn.noise import Gaussian
from waveorder.nn.parameters import (
    Free,
    ParameterPrior,
    _assign_parameters,
    evaluate,
    free_specs,
    same_device,
)
from waveorder.nn.sampling import Data, Sampling
from waveorder.nn.scores import Score
from waveorder.observe import collect_steps, emit, observing, operation, recording


class Estimate(NamedTuple):
    """A point and the objective there, all on the model's device.

    Attributes
    ----------
    point : dict[str, Tensor]
        The free quantities by dotted name, as 0-d tensors in their own units.
    objective : Tensor
        0-d score plus parameter prior at :attr:`point`.
    """

    point: dict[str, Tensor]
    objective: Tensor


def score_candidates(
    data: Data,
    model: ImagingModel,
    sampling: Sampling,
    *,
    score: Score,
    candidates: Mapping[str, Tensor],
    chunk_size: int | None = None,
) -> Tensor:
    """Evaluate ``score`` at a batch of parameter values in one batched call.

    Parameters
    ----------
    data : Tensor
        Measured data on ``sampling``, in the dtype and on the device of the model's system.
    model : ImagingModel
        Model whose parameters are overridden; it is not modified.
    sampling : Sampling
        Grid of ``data``.
    score : Score
        The objective.
    candidates : Mapping[str, Tensor]
        Equal-length 1-D tensors keyed by dotted parameter (or buffer) name, each in the dtype and on the device of
        the quantity it replaces; parameters not listed keep their value.
    chunk_size : int, optional
        Evaluate this many candidates at a time, bounding memory; all at once by default.

    Returns
    -------
    Tensor
        Shape ``(B,)``: the score at each candidate.

    Raises
    ------
    ParameterError
        For an unknown name, a candidate in another dtype or on another device than its quantity, candidates that
        are not 1-D of one length, or a ``chunk_size`` below 1.
    ShapeError, PlacementError
        For ``data``, as :func:`~waveorder.nn.model.expect_on_system` raises them.
    """
    where = "score_candidates"
    expect_on_system(data, model.system, where=where, name="data", shape=model.system.data_shape(sampling))
    current = dict(model.named_parameters()) | dict(model.named_buffers())
    for name, value in candidates.items():
        target = current.get(name)
        if target is None:
            raise ParameterError(
                f"{where} candidates.{name} is not a parameter or buffer of the model; free parameters are "
                f"{sorted(n for n, _ in model.named_parameters())}, and fixed quantities are buffers",
                path=f"candidates.{name}",
            )
        if value.dtype != target.dtype or not same_device(value.device, target.device):
            raise ParameterError(
                f"{where} candidates.{name} must be {target.dtype} on {target.device}, as {name} is, got "
                f"{value.dtype} on {value.device}",
                path=f"candidates.{name}",
            )
    lengths = {name: tuple(value.shape) for name, value in candidates.items()}
    if not candidates or len(set(lengths.values())) != 1 or len(next(iter(lengths.values()))) != 1:
        raise ParameterError(
            f"{where} candidates must be non-empty 1-D tensors of one length, got shapes {lengths}", path="candidates"
        )
    if chunk_size is not None and chunk_size < 1:
        raise ParameterError(f"{where} chunk_size must be at least 1, got {chunk_size}", path="chunk_size")
    model.check(score, sampling)

    def one(values: dict[str, Tensor]) -> Tensor:
        return evaluate(lambda m: score(data, m, sampling), model, values)

    return vmap(one, chunk_size=chunk_size)(dict(candidates))


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationObjective:
    """``score + prior`` as a function of the free quantities: what every :class:`SearchStage` minimizes.

    Build it with :meth:`of`, which checks the model.

    Attributes
    ----------
    data : Tensor
        Measured data on :attr:`sampling`.
    model : ImagingModel
        The model; evaluating the objective never modifies it.
    sampling : Sampling
        Grid of :attr:`data`.
    score : Score
        The outer objective, for example :class:`~waveorder.nn.scores.Evidence`.
    prior : ParameterPrior
        Prior over the free quantities, with one row; a stage keeps its points within its bounds.
    """

    data: Data
    model: ImagingModel
    sampling: Sampling
    score: Score
    prior: ParameterPrior

    @classmethod
    def of(
        cls, data: Data, model: ImagingModel, sampling: Sampling, *, score: Score, prior: ParameterPrior | None = None
    ) -> Self:
        """Return the objective of calibrating ``model`` on ``data`` with ``score``.

        ``prior`` defaults to the one the :class:`~waveorder.nn.parameters.Free` specs of the model declare.

        Raises
        ------
        ParameterError
            If the model has no free quantity, ``prior`` has more than one row, or it covers a name that is no free
            quantity of the model.
        IncompatibleModelError
            If ``score`` cannot run on ``model`` or ``sampling``.
        """
        specs = free_specs(model)
        if not specs:
            raise ParameterError("the model has no free quantity to calibrate")
        model.check(score, sampling)
        prior = ParameterPrior(specs) if prior is None else prior
        if prior.tiles != 1:
            raise ParameterError(
                f"the prior holds {prior.tiles} tiles; pass one tile's prior, prior.select([t])", path="prior"
            )
        unknown = [name for name in prior.covered if name not in specs]
        if unknown:
            raise ParameterError(f"the prior covers {unknown}, which are no free quantities of the model", path="prior")
        return cls(data=data, model=model, sampling=sampling, score=score, prior=prior)

    @property
    def specs(self) -> dict[str, Free]:
        """The free quantities searched over: the :class:`~waveorder.nn.parameters.Free` specs of the model, by name."""
        return free_specs(self.model)

    def start(self) -> Estimate:
        """Return the model's current free quantities (cloned, detached) and the objective there, NaN as ``+inf``."""
        point = {name: self.model.get_parameter(name).detach().clone() for name in self.specs}
        with torch.no_grad():
            return Estimate(point, torch.nan_to_num(self(point), nan=math.inf))

    def __call__(self, values: Mapping[str, Tensor]) -> Tensor:
        """Return the objective at ``values`` (0-d tensors by dotted name), differentiable in them."""
        return evaluate(lambda m: self.score(self.data, m, self.sampling), self.model, values) + self.prior.nll(values)

    def batch(self, candidates: Mapping[str, Tensor], *, chunk_size: int | None = None) -> Tensor:
        """Return the objective at each of ``B`` candidates, shape ``(B,)``, without gradients; NaN reads as ``+inf``.

        ``candidates`` and ``chunk_size`` are those of :func:`score_candidates`, with a tensor for every name in
        :attr:`specs`.
        """
        with torch.no_grad():
            scores = score_candidates(
                self.data, self.model, self.sampling, score=self.score, candidates=candidates, chunk_size=chunk_size
            )
            return torch.nan_to_num(scores + self.prior.nll(candidates), nan=math.inf)


@runtime_checkable
class SearchStage(Protocol):
    """One stage of :func:`calibrate`: from the best estimate so far, return a better one or ``start`` itself.

    Its class name labels its entry in :attr:`CalibrationResult.trace`. It should keep its points within
    ``objective.prior``'s bounds and its tensors on the device, so stages chain without a host sync.
    """

    reads_back: ClassVar[bool]
    """Whether :meth:`run` reads values back to the host (``True`` for a line search that decides there);
    :func:`~waveorder.testing.check_search_stage` skips only its no-sync check for such a stage."""

    quantity_keyed: ClassVar[tuple[str, ...]]
    """Names of the stage's fields keyed by dotted quantity name (``()`` for none), such as
    :class:`~waveorder.nn.search.GridSearch`'s ``points``. Before the first stage runs, :func:`calibrate` checks that
    each exists and that the keys of each one that is a mapping are free quantities (:func:`expect_known_names`)."""

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return the best estimate found from ``start``, never worse than ``start``."""
        ...


def _expect_count(value: int, *, where: str, name: str, minimum: int) -> None:
    """Raise :class:`~waveorder.errors.ParameterError` if ``value`` is below ``minimum``."""
    if value < minimum:
        raise ParameterError(f"{where} {name} must be at least {minimum}, got {value}", path=name)


_BOUND_TOLERANCE = 1e-6

_records: ContextVar[list[object] | None] = ContextVar("waveorder.nn.search.records", default=None)


def keeping_records() -> bool:
    """Whether a ``calibrate(..., history=True)`` collects what the running stage saw (:func:`keep_record`)."""
    return _records.get() is not None


def keep_record(item: object) -> None:
    """Hand what a stage saw (a :class:`ScoreGrid` batch, a sweep result) to ``calibrate(..., history=True)``."""
    kept = _records.get()
    if kept is not None:
        kept.append(item)


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    """The points a grid stage scored and the objective at each.

    Attributes
    ----------
    names : tuple[str, ...]
        The gridded quantities by dotted name, in column order.
    values : Tensor
        ``(N, len(names) + 1)``: one row per point, its values and then the objective (``inf`` where not finite).
    """

    names: tuple[str, ...]
    values: Tensor


type SweepRule = Literal["otsu_cnr", "l_curve"]
"""How :class:`~waveorder.nn.search.WeightSweep` picks a weight from its reconstructions."""


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What a :class:`~waveorder.nn.search.WeightSweep` swept and picked.

    Attributes
    ----------
    weight : float
        The picked weight, in the object prior's units.
    index : int
        Index of the pick in :attr:`weights`.
    weights : tuple[float, ...]
        Every weight reconstructed, ascending.
    transfer_peak : float
        ``max |H|^2`` of the bound system, the anchor of a ``"relative"`` sweep.
    scores : tuple[float, ...]
        Otsu contrast-to-noise ratio per weight; empty for ``"l_curve"``.
    residual_norms, solution_norms : tuple[float, ...]
        ``||H x - g||`` and ``||x||`` per weight; empty for ``"otsu_cnr"``.
    scored_slice : int or None
        The z slice of the unpadded estimate that ``"otsu_cnr"`` scored, or that :attr:`slices` show; else ``None``.
    warnings : tuple[str, ...]
        A pick on either end of the sweep, which may mean the optimum lies outside it.
    rule : {"otsu_cnr", "l_curve"}
        The rule that picked.
    slices : Tensor, optional
        ``(N, Y, X)``: slice :attr:`scored_slice` of every estimate, on the CPU, with ``keep_slices``; else ``None``.
    data_slice : Tensor, optional
        ``(Y, X)``: that slice of the data, kept with :attr:`slices`.
    region : tuple[int, int, int], optional
        ``(y0, x0, side)`` of the crop of the field the data was, when the caller cropped it; else ``None``.
    """

    weight: float
    index: int
    weights: tuple[float, ...]
    transfer_peak: float
    scores: tuple[float, ...] = ()
    residual_norms: tuple[float, ...] = ()
    solution_norms: tuple[float, ...] = ()
    scored_slice: int | None = None
    warnings: tuple[str, ...] = field(default=())
    rule: SweepRule = "otsu_cnr"
    slices: Tensor | None = None
    data_slice: Tensor | None = None
    region: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The calibrated quantities and what is known about them.

    Attributes
    ----------
    parameters : dict[str, Tensor]
        Estimates by dotted name; the model holds them too.
    objective : Tensor
        Score plus parameter prior at :attr:`parameters`.
    at_bound : tuple[str, ...]
        Quantities whose estimate lies on a bound (within ``1e-6`` of the span): widen the bounds or hold it fixed.
    trace : tuple[tuple[str, float], ...]
        The best objective after each stage, by stage type.
    steps : tuple[float, ...]
        With ``history``: the objective at each evaluation of the gradient stages; empty otherwise.
    grid : ScoreGrid, optional
        With ``history``: the points and objectives of the last grid stage over every free quantity, float64 on the
        CPU; else ``None``.
    sweep : SweepResult, optional
        With ``history``: what the last :class:`~waveorder.nn.search.WeightSweep` stage swept and picked.
    """

    parameters: dict[str, Tensor]
    objective: Tensor
    at_bound: tuple[str, ...]
    trace: tuple[tuple[str, float], ...]
    steps: tuple[float, ...] = ()
    grid: ScoreGrid | None = None
    sweep: SweepResult | None = None


def calibrate(  # noqa: PLR0913 (data, model and grid; score, search and prior; whether to keep the history)
    data: Data,
    model: ImagingModel,
    sampling: Sampling,
    *,
    score: Score,
    search: Sequence[SearchStage] | None = None,
    prior: ParameterPrior | None = None,
    history: bool = False,
) -> CalibrationResult:
    """Estimate the free quantities of ``model`` by minimizing ``score + prior`` over them, stage by stage.

    A :class:`~waveorder.nn.noise.Gaussian` noise with ``sigma="estimate"`` first sets its start and bounds from
    ``data``. With an observer attached, the call reports ``calibration.stage.end`` per stage and ``calibration.end``.

    Parameters
    ----------
    data : Tensor
        Measured data on ``sampling``, real, in the dtype and on the device of the model's system.
    model : ImagingModel
        The model; its free quantities (those built with :class:`~waveorder.nn.parameters.Free`, whatever their
        ``requires_grad``) are set to the estimate. To hold a quantity fixed, build it with a plain number.
    sampling : Sampling
        Grid of ``data``.
    score : Score
        The outer objective, for example :class:`~waveorder.nn.scores.Evidence`.
    search : sequence of SearchStage, optional
        At least one, run in order. ``None`` runs ``GridSearch()`` then ``GradientSearch()``.
    prior : ParameterPrior, optional
        Prior over the free quantities, of one row; by default the one their ``Free`` specs declare.
    history : bool
        Also return what the stages saw (:attr:`CalibrationResult.steps`, :attr:`~CalibrationResult.grid`,
        :attr:`~CalibrationResult.sweep`).

    Returns
    -------
    CalibrationResult

    Raises
    ------
    ParameterError
        If ``search`` is empty, the model has no free quantity, a searched quantity has neither bounds nor a
        tolerance, or a stage's quantity-keyed field is missing or names no free quantity.
    IncompatibleModelError
        Under ``torch.inference_mode()`` (use ``torch.no_grad()``), or if ``score`` cannot run on ``model``.
    ShapeError
        If ``data`` is complex or does not have the system's data shape on ``sampling``.
    PlacementError
        If ``data`` is not in the dtype and on the device of ``model``'s system.

    Warns
    -----
    CalibrationWarning
        If ``score`` is a likelihood and a free quantity is one it cannot determine, such as the focus offset of a
        3D-object system (:meth:`~waveorder.nn.ImagingSystem.undetermined_by_likelihood`).
    """
    with operation():
        search = expect_searchable(data, model, sampling, search=search)
        for part in model.modules():
            if isinstance(part, Gaussian):
                part.start_from(data)
        objective = CalibrationObjective.of(data, model, sampling, score=score, prior=prior)
        specs = objective.specs
        for stage in search:
            _expect_stage_names(stage, specs)
        warn_undetermined(model, score)
        names = list(specs)
        watched = observing()
        record = watched and recording()
        clock = _StageClock(data.device) if watched else None
        best, reached, histories, kept = _run_stages(
            objective, search, record=record or history, keep=history, clock=clock
        )
        values, value = best
        _assign_parameters(model, values)  # the stages' own point: finite and in bounds, so nothing is read back
        grid, sweep = _seen(kept, names)
        # The one read-back of calibrate, in float64 so every value survives: stage objectives and bound flags;
        # then, for an observer, the values its events report; then the step histories and the grid kept.
        report = [*reached, on_bounds(specs, values)]
        observed = [v.detach() for v in values.values()] if watched else []
        steps = [h for h in histories if h is not None] if record or history else []
        tail = [] if grid is None else [grid.values]
        parts = [part.reshape(-1).to(torch.float64) for part in [*report, *observed, *steps, *tail]]
        packed_tensor = torch.cat(parts).cpu()
        cut = packed_tensor.shape[0] - (0 if grid is None else grid.values.numel())
        packed = packed_tensor[:cut].tolist()
        stages, quantities = len(search), len(names)
        levels, flags, seen = packed[:stages], packed[stages : stages + quantities], packed[stages + quantities :]
        trace = tuple((type(stage).__name__, level) for stage, level in zip(search, levels, strict=True))
        at_bound = tuple(name for name, flag in zip(names, flags, strict=True) if flag)
        if grid is not None:
            grid = ScoreGrid(grid.names, packed_tensor[cut:].reshape(grid.values.shape))
        kept_steps = tuple(seen[sum(v.numel() for v in observed) :]) if history else ()
        result = CalibrationResult(values, value.detach(), at_bound, trace, kept_steps, grid, sweep)
        if clock is not None:  # observed
            shown = histories if record else [None] * len(histories)
            _report(result, seen, histories=shown, elapsed=clock.seconds())
        return result


def _seen(kept: Sequence[Sequence[object]], names: Sequence[str]) -> tuple[ScoreGrid | None, SweepResult | None]:
    """Return the last grid over all of ``names`` and the last sweep result the stages kept, on the device."""
    grid, sweep = None, None
    for items in kept:
        batches = [item for item in items if isinstance(item, ScoreGrid)]
        if batches and set(batches[0].names) == set(names):
            grid = ScoreGrid(batches[0].names, torch.cat([batch.values for batch in batches]))
        sweep = next((item for item in reversed(items) if isinstance(item, SweepResult)), sweep)
    return grid, sweep


def expect_searchable(
    data: Data,
    model: ImagingModel,
    sampling: Sampling,
    *,
    search: Sequence[SearchStage] | None,
    where: str = "calibrate",
) -> list[SearchStage]:
    """Check the grad mode and the data's shape and placement before a calibration starts; return the stages.

    Returns ``search`` as a list, or ``GridSearch()`` then ``GradientSearch()`` for ``None``.
    """
    if torch.is_inference_mode_enabled():
        raise IncompatibleModelError(
            f"{where} cannot run under torch.inference_mode(): the search differentiates the score, and tensors made "
            f"in inference mode carry no gradient. Use torch.no_grad() instead, under which {where} runs"
        )
    expect_on_system(data, model.system, where=where, name="data", shape=model.system.data_shape(sampling))
    if search is None:
        # The stages build on this module (the objective and the stage protocol), so they are imported here.
        from waveorder.nn.search.descent import GradientSearch
        from waveorder.nn.search.grid import GridSearch

        return [GridSearch(), GradientSearch()]
    if not search:
        raise ParameterError(
            f"{where} search must name at least one search stage, as the YAML calibration.search does", path="search"
        )
    return list(search)


def warn_undetermined(model: ImagingModel, score: Score) -> None:
    """Warn (:class:`~waveorder.errors.CalibrationWarning`) if a likelihood score has undetermined free quantities."""
    if not score.is_likelihood:
        return
    specs = free_specs(model)
    free = [f"system.{name}" for name in model.system.undetermined_by_likelihood() if f"system.{name}" in specs]
    if free:
        warnings.warn(
            f"calibrate: {type(score).__name__} is a likelihood, and a likelihood does not determine "
            f"{', '.join(free)} on {type(model.system).__name__}. Such a quantity leaves the statistics of the data "
            "unchanged (the focus offset of a 3D object only translates the point-spread function), so its estimate "
            "means nothing. Hold it fixed: build the part with a number instead of Free(...)",
            CalibrationWarning,
            stacklevel=3,
        )


class _StageClock:
    """Seconds from the first stage's start to each stage's end, without a device sync (CUDA events on a GPU)."""

    def __init__(self, device: torch.device) -> None:
        self._stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        self._events: list[torch.cuda.Event] = []
        self._times: list[float] = []
        self.mark()

    def mark(self) -> None:
        """Record the end of a stage (the first call, from ``__init__``, is the start)."""
        if self._stream is None:
            self._times.append(time.perf_counter())
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record(self._stream)
        self._events.append(event)

    def seconds(self) -> list[float]:
        """Return each stage end's seconds since the start; on a GPU, call it after the read-back."""
        if self._stream is None:
            return [end - self._times[0] for end in self._times[1:]]
        return [self._events[0].elapsed_time(end) / 1000 for end in self._events[1:]]


def _run_stages(
    objective: CalibrationObjective,
    search: Sequence[SearchStage],
    *,
    record: bool,
    keep: bool,
    clock: _StageClock | None,
) -> tuple[Estimate, list[Tensor], list[Tensor | None], list[list[object]]]:
    """Run ``search`` in order from the model's point; return the best and each stage's objective, steps and records."""
    best = objective.start()
    reached: list[Tensor] = []
    histories: list[Tensor | None] = []
    kept: list[list[object]] = []
    for stage in search:
        items: list[object] = []
        token = _records.set(items if keep else None)
        try:
            with collect_steps(on=record) as steps:
                best = stage.run(objective, start=best)
        finally:
            _records.reset(token)
        reached.append(best.objective.detach().reshape(()))
        histories.append(torch.stack(steps) if steps else None)
        kept.append(items)
        if clock is not None:
            clock.mark()
    return best, reached, histories, kept


def _report(
    result: CalibrationResult, seen: Sequence[float], *, histories: Sequence[Tensor | None], elapsed: Sequence[float]
) -> None:
    """Emit ``calibration.stage.end`` per stage and ``calibration.end`` from the read-back values ``seen``."""
    names = list(result.parameters)
    offset = len(names)
    for index, ((stage, level), history) in enumerate(zip(result.trace, histories, strict=True)):
        payload: dict[str, object] = {"stage": stage, "index": index, "objective": level, "elapsed_s": elapsed[index]}
        if history is not None:
            payload["steps"] = list(seen[offset : offset + history.numel()])
            offset += history.numel()
        emit("calibration.stage.end", payload)
    emit(
        "calibration.end",
        {
            "parameters": dict(zip(names, seen[: len(names)], strict=True)),
            "at_bound": list(result.at_bound),
            "trace": [[stage, level] for stage, level in result.trace],
        },
    )


def expect_known_names(names: Iterable[str], specs: Mapping[str, Free], *, path: str) -> None:
    """Raise :class:`~waveorder.errors.ParameterError` for a name in a quantity-keyed setting that is no free quantity.

    Parameters
    ----------
    names : Iterable[str]
        The setting's keys (a mapping iterates its keys).
    specs : Mapping[str, Free]
        The free quantities by dotted name, :attr:`CalibrationObjective.specs`.
    path : str
        The setting's owner and field, ``"<Owner>.<field>"``, e.g. ``"GridSearch.points"``.

    Raises
    ------
    ParameterError
        If a name is not in ``specs``; its ``path`` is ``<field>.<first unknown name>``.
    """
    unknown = sorted(set(names) - set(specs))
    if unknown:
        field = path.rpartition(".")[2]
        raise ParameterError(
            f"{path} names {unknown}, which are not free quantities; free are {sorted(specs)}",
            path=f"{field}.{unknown[0]}",
        )


def _expect_stage_names(stage: SearchStage, specs: Mapping[str, Free]) -> None:
    """Check every quantity-keyed field a stage declares in ``quantity_keyed``: it exists, and a mapping's keys."""
    owner = type(stage).__name__
    for name in type(stage).quantity_keyed:
        if not hasattr(stage, name):
            raise ParameterError(f"{owner}.quantity_keyed names {name!r}, which {owner} has no field of", path=name)
        value = getattr(stage, name)
        if isinstance(value, Mapping):
            expect_known_names(value, specs, path=f"{owner}.{name}")


def _geometric(low: float, high: float) -> bool:
    """Whether the span ``(low, high)`` is searched on a log scale: positive and wider than a factor of 100."""
    return low > 0 and high / low > 100


def on_bounds(specs: Mapping[str, Free], values: Mapping[str, Tensor]) -> Tensor:
    """Return whether each free quantity is on a bound (within ``1e-6`` of its span), on the values' device.

    A quantity without bounds is never on one. Nothing reads back.

    Parameters
    ----------
    specs : Mapping[str, Free]
        The free quantities by dotted name, :attr:`CalibrationObjective.specs`.
    values : Mapping[str, Tensor]
        A tensor per name in ``specs``, all of one shape ``S`` (``()`` for one estimate, ``(T,)`` for a row per
        tile) and on one device.

    Returns
    -------
    Tensor
        ``(Q, *S)`` bool, in the order of ``specs``.

    Raises
    ------
    ParameterError
        If ``values`` lacks a name in ``specs``.
    """
    missing = [name for name in specs if name not in values]
    if missing:
        raise ParameterError(f"on_bounds needs a value for every free quantity, missing {missing}", path=missing[0])
    flags = []
    for name, spec in specs.items():
        value = values[name].detach()
        if spec.bounds is None:
            flags.append(torch.zeros_like(value, dtype=torch.bool))
        else:
            low, high = spec.bounds
            flags.append(torch.minimum(value - low, high - value) <= _BOUND_TOLERANCE * (high - low))
    return torch.stack(flags)
