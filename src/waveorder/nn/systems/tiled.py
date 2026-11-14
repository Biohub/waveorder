"""Field-dependent optics as independent tiles: per-tile calibration, parameter tables, operators and the bead fit.

Each tile of the field is calibrated and solved with its own values; :mod:`waveorder.tiling` blends the tiles'
reconstructions, not their parameters. A per-tile table (:data:`TileParameters`) maps dotted free-parameter names
(``"system.detection.numerical_aperture"``) to ``(tiles, *parameter.shape)`` tensors, row ``i`` for tile ``i``;
unlisted parameters keep the model's value. :class:`BeadFit`, :func:`zernike_terms` and :func:`zernike_search` are
the pieces of 3.x's per-tile Zernike recovery, run by :func:`waveorder.tiling.recover_zernikes`.
"""

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self, final, override

import torch
from torch import Tensor
from torch.func import vmap

from waveorder.errors import IncompatibleModelError, ParameterError, PlacementError, ShapeError
from waveorder.nn.model import ImagingModel
from waveorder.nn.operators import LinearOperator
from waveorder.nn.optics import ZernikeAberration
from waveorder.nn.parameters import ParameterPrior, evaluate, free_specs, same_device
from waveorder.nn.sampling import ZYX, Sampling
from waveorder.nn.scores import Score
from waveorder.nn.search.base import (
    CalibrationObjective,
    CalibrationResult,
    Estimate,
    SearchStage,
    _expect_count,
    expect_known_names,
    expect_searchable,
    on_bounds,
    score_candidates,
    warn_undetermined,
)
from waveorder.nn.search.descent import GradientSearch
from waveorder.nn.solvers import Solver, SolveResult
from waveorder.nn.verbs import InverseProblem
from waveorder.observe import operation

_WHERE = "calibrate_tiles"


@dataclass(frozen=True, slots=True)
class TileCalibration:
    """The per-tile estimates of :func:`calibrate_tiles`.

    Attributes
    ----------
    parameters : dict[str, Tensor]
        The estimates as a :data:`TileParameters` table: by dotted name, a ``(T,)`` tensor, row ``t`` for tile ``t``.
    objective : Tensor
        ``(T,)`` score plus parameter prior at each tile's estimate.
    tiles : tuple[CalibrationResult, ...]
        One :func:`~waveorder.nn.calibrate` result per tile.
    trace : tuple[tuple[str, Tensor], ...]
        The best ``(T,)`` objective after each stage, by stage type.
    """

    parameters: dict[str, Tensor]
    objective: Tensor
    tiles: tuple[CalibrationResult, ...]
    trace: tuple[tuple[str, Tensor], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _TileObjective(CalibrationObjective):
    """:class:`CalibrationObjective` over ``B`` tiles at once: ``data`` is ``(B, *data_shape)``, values ``(B,)``.

    The prior holds one shared row or ``B`` rows, tile ``b``'s at row ``b``.
    """

    origin: dict[str, Tensor]

    def start(self) -> Estimate:
        """Return the tiles' start values and their ``(B,)`` objective, NaN read as ``+inf``."""
        point = {name: value.detach().clone() for name, value in self.origin.items()}
        with torch.no_grad():
            return Estimate(point, torch.nan_to_num(self(point), nan=math.inf))

    def __call__(self, values: Mapping[str, Tensor]) -> Tensor:
        """Return the ``(B,)`` objective, tile ``b`` at ``values[name][b]``, differentiable in the values."""

        def one(data: Tensor, point: dict[str, Tensor]) -> Tensor:
            return evaluate(lambda m: self.score(data, m, self.sampling), self.model, point)

        return vmap(one)(self.data, dict(values)) + self.prior.nll(values)

    def batch(self, candidates: Mapping[str, Tensor], *, chunk_size: int | None = None) -> Tensor:
        """Return the objective of ``N`` candidates on every tile, ``(N, B)``, without gradients.

        A candidate tensor is ``(N,)`` (the same for every tile) or ``(N, B)`` (its own per tile).
        """
        dims = {name: 1 if value.dim() == 2 else None for name, value in candidates.items()}
        tiles = self.data.shape[0]

        def one(data: Tensor, column: dict[str, Tensor]) -> Tensor:
            return score_candidates(
                data, self.model, self.sampling, score=self.score, candidates=column, chunk_size=chunk_size
            )

        with torch.no_grad():
            scores = vmap(one, in_dims=(0, dims))(self.data, dict(candidates))
            table = {
                name: value if value.dim() == 2 else value[:, None].expand(-1, tiles)
                for name, value in candidates.items()
            }
            return torch.nan_to_num(scores.mT + self.prior.nll(table), nan=math.inf)

    def tile(self, index: int) -> CalibrationObjective:
        """Return the objective of tile ``index`` alone, for a stage that runs tile by tile."""
        return CalibrationObjective(
            data=self.data[index],
            model=self.model,
            sampling=self.sampling,
            score=self.score,
            prior=self.prior.select([index]),
        )


def calibrate_tiles(  # noqa: PLR0913 (tiles, model and grid; score and prior; search; start and chunks)
    tiles: Tensor,
    model: ImagingModel,
    sampling: Sampling,
    *,
    score: Score,
    search: Sequence[SearchStage] | None = None,
    prior: ParameterPrior | None = None,
    start: Mapping[str, Tensor] | None = None,
    chunk_size: int | None = None,
) -> TileCalibration:
    """Calibrate the free quantities of ``model`` on every tile independently, stage by stage.

    Each tile is calibrated as :func:`~waveorder.nn.calibrate` would calibrate it alone. A stage with
    ``reads_back = False`` (grid, random, gradient search) runs on a chunk of tiles at once; one that reads back
    (L-BFGS, Nelder-Mead) runs tile by tile. ``model`` is not modified. ``model``, ``sampling``, ``score`` and
    ``search`` are those of :func:`~waveorder.nn.calibrate`.

    Parameters
    ----------
    tiles : Tensor
        ``(T, *data_shape)``: ``T`` tiles of data, each on ``sampling``, in the model's dtype and on its device.
    search : sequence of SearchStage, optional
        A stage with ``reads_back = False`` must accept values with a leading tile axis, as the built-in ones do.
    prior : ParameterPrior, optional
        Prior over the free quantities; by default the one their specs declare. One row is shared by every tile; a
        prior of ``T`` rows gives each tile its own.
    start : Mapping[str, Tensor], optional
        A ``(T,)`` start per dotted name, such as a previous :attr:`TileCalibration.parameters`; names not listed
        start at the model's value.
    chunk_size : int, optional
        Tiles calibrated at once, bounding memory; all by default.

    Returns
    -------
    TileCalibration

    Raises
    ------
    ParameterError
        As :func:`~waveorder.nn.calibrate`, or for ``tiles`` without a tile axis, a ``start`` entry that names no
        free quantity or is not ``(T,)``, a ``chunk_size`` below 1, or a prior whose rows are neither 1 nor ``T``.
    ShapeError
        If a tile does not have the system's data shape on ``sampling`` or is complex.
    PlacementError
        If ``tiles``, or a ``start`` entry, is not in the model's dtype and on its device.
    IncompatibleModelError
        As :func:`~waveorder.nn.calibrate`, or if a stage with ``reads_back = False`` drops the tile axis.
    """
    with operation():
        if tiles.dim() < 1 or tiles.shape[0] == 0:
            raise ParameterError(
                f"{_WHERE} tiles must have a leading tile axis, got shape {tuple(tiles.shape)}", path="tiles"
            )
        search = expect_searchable(tiles[0], model, sampling, search=search, where=_WHERE)
        count = tiles.shape[0]
        size = count if chunk_size is None else chunk_size
        if chunk_size is not None:
            _expect_count(chunk_size, where=_WHERE, name="chunk_size", minimum=1)
        work = copy.deepcopy(model)
        specs = free_specs(work)
        if not specs:
            raise ParameterError("the model has no free quantity to calibrate")
        work.check(score, sampling)
        warn_undetermined(work, score)
        origin = _start_table(start, work, count=count)
        prior = ParameterPrior(specs) if prior is None else prior
        if prior.tiles not in (1, count):
            raise ParameterError(
                f"{_WHERE} prior holds {prior.tiles} tiles for {count} tiles of data; give one row or one per tile",
                path="prior",
            )
        names = list(specs)
        estimates: list[Estimate] = []
        levels: list[list[Tensor]] = []
        for first in range(0, count, size):
            rows = slice(first, min(first + size, count))
            objective = _TileObjective(
                data=tiles[rows],
                model=work,
                sampling=sampling,
                score=score,
                prior=prior.select(range(rows.start, rows.stop)),
                origin={name: value[rows] for name, value in origin.items()},
            )
            best, reached = _search(objective, search, length=rows.stop - rows.start)
            estimates.append(best)
            levels.append(reached)
        point = {name: torch.cat([estimate.point[name] for estimate in estimates]) for name in names}
        trace = tuple(
            (type(stage).__name__, torch.cat([chunk[k] for chunk in levels])) for k, stage in enumerate(search)
        )
        # One read-back: the trace and which quantities ended on a bound, per tile.
        steps = [level.tolist() for _, level in trace]
        flags = on_bounds(specs, point).tolist()
        objective = torch.cat([estimate.objective for estimate in estimates])
        results = tuple(
            CalibrationResult(
                {name: point[name][index] for name in names},
                objective[index],
                tuple(name for name, row in zip(names, flags, strict=True) if row[index]),
                tuple((name, step[index]) for (name, _), step in zip(trace, steps, strict=True)),
            )
            for index in range(count)
        )
        return TileCalibration(point, objective, results, trace)


def _start_table(start: Mapping[str, Tensor] | None, model: ImagingModel, *, count: int) -> dict[str, Tensor]:
    """Return the ``(T,)`` start of every free quantity: ``start``'s rows, else the model's value on every tile."""
    parameters = dict(model.named_parameters())
    table = {name: value.detach().expand(count).clone() for name, value in parameters.items()}
    if start is None:
        return table
    entries = dict(start)
    expect_known_names(entries, free_specs(model), path=f"{_WHERE}.start")
    for name, value in entries.items():
        path = f"start.{name}"
        target = parameters[name]
        if tuple(value.shape) != (count,):
            raise ParameterError(
                f"{_WHERE} {path} must have shape ({count},), one value per tile, got {tuple(value.shape)}", path=path
            )
        if value.dtype != target.dtype or not same_device(value.device, target.device):
            raise PlacementError(
                f"{_WHERE} {path} is {value.dtype} on {value.device} but {name} is {target.dtype} on {target.device}"
            )
        table[name] = value.detach().clone()
    return table


def _search(objective: _TileObjective, search: Sequence[SearchStage], *, length: int) -> tuple[Estimate, list[Tensor]]:
    """Run the stages on one chunk of ``length`` tiles; return the best and the ``(B,)`` objective after each stage."""
    best = objective.start()
    reached = []
    for stage in search:
        if stage.reads_back:
            rows = [
                stage.run(
                    objective.tile(b), start=Estimate({n: v[b] for n, v in best.point.items()}, best.objective[b])
                )
                for b in range(length)
            ]
            best = Estimate(
                {name: torch.stack([row.point[name].reshape(()) for row in rows]) for name in best.point},
                torch.stack([row.objective.reshape(()) for row in rows]),
            )
        else:
            best = stage.run(objective, start=best)
            _expect_tiled(stage, best, length=length)
        reached.append(best.objective.detach())
    return best, reached


def _expect_tiled(stage: SearchStage, estimate: object, *, length: int) -> None:
    """Check that a stage run on a chunk of tiles kept the tile axis, from metadata."""
    name = type(stage).__name__
    shapes = (
        [tuple(value.shape) for value in estimate.point.values()] + [tuple(estimate.objective.shape)]
        if isinstance(estimate, Estimate)
        else []
    )
    if not shapes or any(shape != (length,) for shape in shapes):
        raise IncompatibleModelError(
            f"{name}.run on {length} tiles returned shapes {shapes}, not ({length},) per quantity and objective; "
            "a stage with reads_back = False runs on all tiles of a chunk at once and must keep their leading axis"
        )


type TileParameters = Mapping[str, Tensor]
"""Per-tile values of free parameters: dotted path to a ``(tiles, *parameter.shape)`` tensor; see the module."""


def expect_tile_parameters(
    parameters: Mapping[str, Tensor], *, model: ImagingModel, tiles: int, where: str
) -> dict[str, Tensor]:
    """Return ``parameters`` as a dict after checking it against ``model`` and the number of ``tiles``.

    Raises
    ------
    ParameterError
        If a key is not a free parameter of ``model``, or a value does not have shape
        ``(tiles, *parameter.shape)``; ``path`` is ``parameters.<name>``.
    PlacementError
        If a value's dtype or device differs from its parameter's.
    """
    free = dict(model.named_parameters())
    table: dict[str, Tensor] = {}
    for name, value in parameters.items():
        path = f"parameters.{name}"
        if name not in free:
            raise ParameterError(
                f"{where}: {name!r} is not a free parameter of the model; it has {sorted(free)}", path=path
            )

        parameter = free[name]
        expected = (tiles, *parameter.shape)
        if tuple(value.shape) != expected:
            raise ParameterError(
                f"{where}: {path} has shape {tuple(value.shape)}, expected {expected}: one row per tile", path=path
            )
        if value.dtype != parameter.dtype or not same_device(value.device, parameter.device):
            raise PlacementError(
                f"{where}: {path} is {value.dtype} on {value.device} but the parameter is {parameter.dtype} on "
                f"{parameter.device}; move it with .to(dtype=..., device=...)"
            )
        table[name] = value
    return table


@final
class TileOperators:
    """Bind each tile's system once, with its own parameter values, and keep the operators for later volumes.

    The operators of the first tiles asked for are kept while their bytes fit in ``budget``; the rest bind afresh
    on every call, so memory stays within the budget.

    Parameters
    ----------
    model : ImagingModel
        The model whose system binds.
    table : TileParameters
        Checked per-tile values (:func:`expect_tile_parameters`).
    budget : int
        Device bytes the kept operators may hold.
    """

    def __init__(self, model: ImagingModel, table: TileParameters, *, budget: int) -> None:
        self.model = model
        self.table = table
        self.budget = budget
        self.held = 0
        self._operators: dict[tuple[int, Sampling], LinearOperator] = {}
        self._full = False

    def _row(self, index: int) -> dict[str, Tensor]:
        return {name: values[index] for name, values in self.table.items()}

    def __call__(self, index: int, sampling: Sampling) -> LinearOperator:
        """Return tile ``index``'s system bound to ``sampling`` with the tile's values."""
        key = (index, sampling)
        operator = self._operators.get(key)
        if operator is not None:
            return operator
        operator = evaluate(lambda model, s: model.system.bind(s), self.model, self._row(index), sampling)
        fourier = operator.as_fourier()
        size = fourier.transfer.nbytes if fourier is not None else self.budget + 1  # an unknown size is not kept
        if not self._full and self.held + size <= self.budget:
            self._operators[key] = operator
            self.held += size
        else:
            self._full = True  # keep the kept set a prefix of the asking order: later tiles never displace it
        return operator

    def solve(
        self, index: int, data: Tensor, sampling: Sampling, *, solver: Solver, operator: LinearOperator
    ) -> SolveResult:
        """Solve tile ``index`` with its values swapped into the model, given its operator from ``self(index, ...)``."""
        row = self._row(index)
        return evaluate(lambda model, d, s: solver(d, model, s, operator=operator), self.model, row, data, sampling)


LEARNING_RATES: dict[int, float] = {
    **dict.fromkeys((4, 5, 6), 0.05),
    **dict.fromkeys((7, 8, 9, 10), 0.03),
    **dict.fromkeys((11, 12, 13, 14, 15), 0.02),
}
"""3.x ``RecoverySettings.lr_per_mode``, in waves per step by Noll index; 3.x stepped a mode not listed by 0.01."""

BEAD_SIGMA: ZYX[float] = (0.15, 0.1, 0.1)
"""3.x ``RecoverySettings.bead_template_sigma_um``: the bead's Gaussian sigma along z, y, x, um."""


class BeadFit(Score):
    """Mean squared error between the data and the image of a known object, after a least-squares gain.

    ``mean((s H f - g)^2)`` with ``s = <H f, g> / <H f, H f>`` when ``scale_fit``, else ``s = 1``. Not a likelihood.

    Parameters
    ----------
    obj : Tensor
        The known object, in the system's object shape on the data's sampling, dtype and device.
    scale_fit : bool
        Fit the gain ``s``; ``True`` by default, as in 3.x.

    Raises
    ------
    ParameterError
        If ``obj`` is not a real floating tensor.
    """

    is_likelihood: ClassVar[bool] = False
    obj: Tensor

    def __init__(self, obj: Tensor, *, scale_fit: bool = True) -> None:
        super().__init__()

        if not obj.is_floating_point():
            raise ParameterError(f"BeadFit object must be a real floating tensor, got {obj.dtype}", path="obj")
        self.register_buffer("obj", obj)
        self.scale_fit = scale_fit

    @classmethod
    def gaussian(cls, sampling: Sampling, *, sigma: ZYX[float] = BEAD_SIGMA, like: Tensor) -> Self:
        """Return the fit to a Gaussian bead at the centre of a tile (3.x ``_build_gaussian_bead_template``).

        The bead peaks at 1 on voxel ``(Z // 2, Y // 2, X // 2)``; ``sigma`` is along z, y, x in um. The object is
        ``(1, Z, Y, X)`` in the dtype and on the device of ``like``.
        """
        axes = [
            (torch.arange(n, dtype=like.dtype, device=like.device) - n // 2) * size
            for n, size in zip(sampling.shape, sampling.pixel_size, strict=True)
        ]
        z, y, x = torch.meshgrid(*axes, indexing="ij")
        sz, sy, sx = sigma
        bead = torch.exp(-(z**2 / (2 * sz**2) + y**2 / (2 * sy**2) + x**2 / (2 * sx**2)))
        return cls(bead[None])

    @override
    def run(self, problem: InverseProblem) -> Tensor:
        operator = problem.operator
        if tuple(self.obj.shape) != operator.object_shape:
            raise ShapeError(
                f"BeadFit object has shape {tuple(self.obj.shape)}, the system's object on this sampling is "
                f"{operator.object_shape}"
            )
        predicted = operator.apply(self.obj)
        data = problem.data
        if self.scale_fit:
            gain = (predicted * data).sum() / (predicted * predicted).sum().clamp(min=1e-20)
            predicted = predicted * gain
        return (predicted - data).square().mean()


def zernike_terms(model: ImagingModel) -> dict[str, int]:
    """Return the model's free Zernike terms: dotted parameter name to Noll index, in parameter order."""
    free = dict(model.named_parameters())
    terms = {}
    for prefix, module in model.named_modules():
        if isinstance(module, ZernikeAberration):
            for key, noll in module.noll.items():
                name = f"{prefix}.{key}"
                if name in free:
                    terms[name] = noll
    return terms


def zernike_search(terms: Mapping[str, int]) -> GradientSearch:
    """Return 3.x's fit of ``terms``: Adam, 250 steps, the per-mode learning rates of :data:`LEARNING_RATES`."""
    rates = {name: LEARNING_RATES.get(noll, 0.01) for name, noll in terms.items()}
    return GradientSearch(iterations=250, learning_rates=rates, optimizer="adam")
