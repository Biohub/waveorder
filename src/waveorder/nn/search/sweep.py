"""Data-driven object weight: reconstruct at a sweep of Tikhonov weights and pick one by a rule (3.x ``autoreg``).

``"otsu_cnr"`` picks the highest contrast-to-noise ratio of an Otsu split of each reconstruction's local-variance
map, on the z slice whose local variance varies most across the sweep. ``"l_curve"`` picks the corner of
``log ||H x - g||`` against ``log ||x||``, by curvature on axes scaled to ``[0, 1]``. Neither rule is an oracle: read
the swept scores (:class:`SweepResult`) before trusting a pick.
"""

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F  # noqa: N812 (the torch convention)
from torch import Tensor

from waveorder.errors import CalibrationWarning
from waveorder.nn.model import ImagingModel
from waveorder.nn.parameters import evaluate
from waveorder.nn.preprocess import apodize, peak_prominence
from waveorder.nn.sampling import Data, Sampling
from waveorder.nn.search.base import CalibrationObjective, Estimate, SweepResult, SweepRule, keep_record
from waveorder.nn.solvers.wiener import Wiener
from waveorder.registry import register

_QUANTILE_LIMIT = 2**24  # torch.quantile refuses larger inputs; 3.x took a strided sample above it
_SLICE_CHUNK = 64  # slices per Otsu batch, bounding its histogram temporaries


@register("weight_sweep", kind="search")
@dataclass(frozen=True, slots=True, kw_only=True)
class WeightSweep:
    """Pick the object weight by reconstructing at ``samples`` weights and applying ``rule`` (3.x auto weight).

    The weights are ``10 ** linspace(search_min, search_max, samples)``, times ``max |H|^2`` when ``scale`` is
    ``"relative"``. Each reconstruction is the model's Wiener estimate, apodized; the Otsu rule leaves out the
    ``z_padding`` slices at each end. As a search stage it returns the pick, clamped to the weight's bounds, and holds
    every other quantity at ``start``.

    Parameters
    ----------
    rule : {"otsu_cnr", "l_curve"}
        The rule.
    search_min, search_max : float
        log10 of the smallest and largest weight, relative to ``max |H|^2`` or absolute.
    samples : int
        Number of weights, at least 3.
    scale : {"relative", "absolute"}
        Whether the bounds are relative to ``max |H|^2``.
    rolloff : float
        Apodization roll-off of each reconstruction (3.x ``apodization_rolloff``).
    z_padding : int
        Slices the data were padded with at each end of z, left out of the Otsu scores.
    name : str
        Dotted name of the weight in the model.
    keep_slices : bool
        Also keep the scored slice of every estimate (:attr:`SweepResult.slices`), at the cost of a second pass.
    """

    rule: SweepRule = "otsu_cnr"
    search_min: float = -6.0
    search_max: float = 2.0
    samples: int = 25
    scale: Literal["relative", "absolute"] = "relative"
    rolloff: float = 0.0
    z_padding: int = 0
    name: str = "object_prior.weight"
    keep_slices: bool = False
    reads_back: ClassVar[bool] = True
    """The rules decide on the host."""
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def sweep(
        self, data: Data, model: ImagingModel, sampling: Sampling, *, values: Mapping[str, Tensor] | None = None
    ) -> SweepResult:
        """Reconstruct ``data`` at every weight and pick one; the model is not modified.

        Parameters
        ----------
        data : Tensor
            Prepared data on ``sampling`` (padded and normalized as the solve sees it), one channel.
        model : ImagingModel
            A model with a Gaussian prior holding the weight under :attr:`name`, one field.
        sampling : Sampling
            Grid of ``data``.
        values : Mapping[str, Tensor], optional
            Other quantities to hold at these values, by dotted name.

        Returns
        -------
        SweepResult
        """
        held = dict(values or {})
        with torch.no_grad():
            operator = evaluate(lambda m: m.system.bind(sampling), model, held)
            peak = float(operator.norm() ** 2)
            powers = torch.linspace(self.search_min, self.search_max, self.samples, dtype=torch.float64)
            weights = (10.0**powers * (peak if self.scale == "relative" else 1.0)).tolist()
            wiener, statistics, spreads = Wiener(), [], []

            def estimate_at(weight: float) -> Tensor:
                swapped = held | {self.name: torch.tensor(weight, dtype=data.dtype, device=data.device)}
                estimate = evaluate(lambda m: wiener(data, m, sampling, operator=operator).estimate, model, swapped)
                return apodize(estimate, self.rolloff)

            for weight in weights:
                estimate = estimate_at(weight)
                volume = estimate[0, self.z_padding : estimate.shape[1] - self.z_padding]
                if self.rule == "otsu_cnr":
                    statistics.append(torch.stack(_slice_statistics(volume)))
                else:
                    residual = torch.linalg.vector_norm(operator.apply(estimate) - data)
                    statistics.append(torch.stack((residual, torch.linalg.vector_norm(estimate))))
                    if self.keep_slices:  # the slice to show, chosen as the Otsu rule chooses the one it scores
                        spreads.append(_local_variance(volume).var(dim=(-2, -1)))
            stacked = torch.stack(statistics)  # (N, 2, Z) for Otsu, (N, 2) for the L-curve
        scores: list[float] = []
        residuals: list[float] = []
        solutions: list[float] = []
        scored_slice = None
        if self.rule == "otsu_cnr":
            scored_slice = int(stacked[:, 1].mean(dim=0).argmax())
            scores = stacked[:, 0, scored_slice].tolist()
            index = scores.index(max(scores))
        else:
            residuals, solutions = stacked.T.tolist()
            index = l_curve_corner(residuals, solutions)
            if self.keep_slices:
                scored_slice = int(torch.stack(spreads).mean(dim=0).argmax())
        slices = data_slice = None
        if self.keep_slices:  # a second pass: the slice to keep is known only once every weight is scored
            assert scored_slice is not None
            z = self.z_padding + scored_slice
            with torch.no_grad():
                slices = torch.stack([estimate_at(weight)[0, z].cpu() for weight in weights])
            data_slice = data[0, z].detach().cpu()
        messages: tuple[str, ...] = ()
        if index in (0, self.samples - 1):
            end = "lower" if index == 0 else "upper"
            messages = (
                f"the {self.rule} pick landed on the {end} end of the sweep (10^{float(powers[index]):g}), so the best "
                "weight may lie outside the search bounds",
            )
            warnings.warn(messages[0], CalibrationWarning, stacklevel=2)
        return SweepResult(
            weights[index], index, tuple(weights), peak, tuple(scores), tuple(residuals), tuple(solutions),
            scored_slice, messages, rule=self.rule, slices=slices, data_slice=data_slice,
        )  # fmt: skip

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        """Return ``start`` with the weight replaced by the pick, clamped to its bounds, and the objective there.

        The sweep's result is kept for ``calibrate(..., history=True)`` (:attr:`CalibrationResult.sweep`).
        """
        held = {name: value for name, value in start.point.items() if name != self.name}
        result = self.sweep(objective.data, objective.model, objective.sampling, values=held)
        keep_record(result)
        point = dict(start.point)
        point[self.name] = start.point[self.name].new_tensor(result.weight)
        point = objective.prior.clamp(point)
        with torch.no_grad():
            return Estimate(point, objective(point))


def _local_variance(volume: Tensor, window: int = 3) -> Tensor:
    """Return the local variance of each slice of ``(Z, Y, X)`` over a ``window`` square, clipped at the 0.99 quantile.

    Edges are replicated; the quantile is taken over the whole volume (a strided sample above ``2**24`` values).
    """
    planes = volume[:, None]
    pad = [window // 2] * 4
    mean = F.avg_pool2d(F.pad(planes, pad, mode="replicate"), window, stride=1)
    square = F.avg_pool2d(F.pad(planes**2, pad, mode="replicate"), window, stride=1)
    variance = (square - mean**2).clamp(min=0)[:, 0]
    flat = variance.reshape(-1).float()
    if flat.shape[0] > _QUANTILE_LIMIT:
        flat = flat[:: flat.shape[0] // _QUANTILE_LIMIT + 1]
    return variance.clamp(max=float(torch.quantile(flat, 0.99)))


def otsu_cnr(flat: Tensor) -> Tensor:
    """Return the contrast-to-noise ratio across an Otsu threshold of each row of ``(N, M)``, shape ``(N,)``.

    Each row is scaled to ``[0, 1]``, binned into 256 bins and split at the bin centre of largest between-class
    variance (never the first or last bin); the ratio is ``|mean_above - mean_below| / std_below``.
    """
    low, high = flat.amin(dim=-1, keepdim=True), flat.amax(dim=-1, keepdim=True)
    normed = (flat - low) / (high - low).clamp(min=1e-12)
    edges = torch.linspace(0.0, 1.0, 257, device=flat.device, dtype=flat.dtype)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = torch.zeros(flat.shape[0], 256, device=flat.device, dtype=flat.dtype)
    counts.scatter_add_(1, torch.bucketize(normed, edges[1:-1]), torch.ones_like(normed))
    total = counts.sum(dim=-1, keepdim=True)
    below = counts.cumsum(dim=-1)
    moment = (counts * centers).cumsum(dim=-1)
    weight = below / total.clamp(min=1)
    between = (
        weight
        * (1 - weight)
        * (moment / below.clamp(min=1e-12) - (moment[:, -1:] - moment) / (total - below).clamp(min=1e-12)) ** 2
    )
    between[:, 0] = between[:, -1] = 0
    signal = (normed > centers[between.argmax(dim=-1)][:, None]).to(flat.dtype)
    background = 1 - signal
    signal_count, background_count = signal.sum(dim=-1).clamp(min=1), background.sum(dim=-1).clamp(min=1)
    signal_mean = (normed * signal).sum(dim=-1) / signal_count
    background_mean = (normed * background).sum(dim=-1) / background_count
    background_square = (normed**2 * background).sum(dim=-1) / background_count
    spread = torch.sqrt((background_square - background_mean**2).clamp(min=0)).clamp(min=1e-12)
    return (signal_mean - background_mean).abs() / spread


def _slice_statistics(volume: Tensor) -> tuple[Tensor, Tensor]:
    """Return the Otsu CNR of each slice's local variance and that map's variance (which ranks slices), ``(Z,)``."""
    variance = _local_variance(volume)
    spread = variance.var(dim=(-2, -1))
    cnr = torch.cat([otsu_cnr(chunk.reshape(chunk.shape[0], -1)) for chunk in variance.split(_SLICE_CHUNK)])
    return cnr, spread


def l_curve_curvature(residuals: list[float], solutions: list[float]) -> list[float]:
    """Return ``|curvature|`` along the L-curve of at least 2 samples (3.x ``l_curve_curvature``).

    Both log norms are scaled to ``[0, 1]``, smoothed by a 3-point mean with the ends repeated, and differentiated
    along the sample index.
    """

    def scaled(values: list[float]) -> list[float]:
        logs = [math.log(value) for value in values]
        low, span = min(logs), max(logs) - min(logs)
        return [(value - low) / span if span > 1e-12 else value - low for value in logs]

    def smoothed(values: list[float]) -> list[float]:
        padded = [values[0], *values, values[-1]]
        return [(padded[i] + padded[i + 1] + padded[i + 2]) / 3 for i in range(len(values))]

    def gradient(values: list[float]) -> list[float]:
        inner = [(values[i + 1] - values[i - 1]) / 2 for i in range(1, len(values) - 1)]
        return [values[1] - values[0], *inner, values[-1] - values[-2]]

    x, y = smoothed(scaled(residuals)), smoothed(scaled(solutions))
    dx, dy = gradient(x), gradient(y)
    ddx, ddy = gradient(dx), gradient(dy)
    return [
        abs((a * d - b * c) / max((a * a + b * b) ** 1.5, 1e-12)) for a, b, c, d in zip(dx, dy, ddx, ddy, strict=True)
    ]


def l_curve_corner(residuals: list[float], solutions: list[float]) -> int:
    """Return the index of the L-curve corner (3.x ``find_l_curve_corner``).

    With two or more curvature peaks of prominence at least a tenth of the largest curvature, the midpoint of the two
    most prominent; with one, the middle of the run around the largest curvature that stays above 0.9 of it. Fewer
    than 3 samples give 0, and a curvature everywhere below 1e-12 or without an interior peak its first maximum.
    """
    if len(residuals) < 3:
        return 0
    curvature = l_curve_curvature(residuals, solutions)
    top = max(curvature)
    first = curvature.index(top)
    peaks = _local_maxima(curvature)
    if top < 1e-12 or not peaks:
        return first
    prominences = [peak_prominence(curvature, peak)[0] for peak in peaks]
    significant = [(prominence, peak) for prominence, peak in zip(prominences, peaks, strict=True)
                   if prominence >= 0.10 * top]  # fmt: skip
    if len(significant) >= 2:
        left, right = sorted(peak for _, peak in sorted(significant, key=lambda pair: pair[0])[-2:])
        return (left + right) // 2
    start = end = first
    while start > 0 and curvature[start - 1] >= 0.90 * top:
        start -= 1
    while end < len(curvature) - 1 and curvature[end + 1] >= 0.90 * top:
        end += 1
    return (start + end) // 2


def _local_maxima(values: list[float]) -> list[int]:
    """Return the interior local maxima, the middle (rounded down) of a flat top, as scipy ``find_peaks``."""
    peaks, i = [], 1
    while i < len(values) - 1:
        if values[i - 1] < values[i]:
            ahead = i + 1
            while ahead < len(values) - 1 and values[ahead] == values[i]:
                ahead += 1
            if values[ahead] < values[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
        i += 1
    return peaks
