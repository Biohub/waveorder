"""The JSON report and PNG montage of a weight sweep (wo-4.0 ``auto_regularization`` ``report_path``, ``plot_path``).

:func:`sweep_report` is wo-4.0's ``AutoRegResult.to_dict`` for a :class:`~waveorder.nn.search.SweepResult`.
:func:`save_sweep_montage` draws with Pillow: the data slice, the rule's objective against
``log10(weight / max |H|^2)``, and the scored slice of every estimate (1st to 99th percentile stretch, pick framed).
"""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from waveorder.nn.search.base import SweepResult
from waveorder.nn.search.sweep import l_curve_curvature

if TYPE_CHECKING:  # Pillow is imported when a montage is drawn; the aliases below are evaluated only when inspected
    from PIL import Image, ImageDraw

type _Picture = Image.Image
type _Pen = ImageDraw.ImageDraw

_PICK, _GREY, _INK = (0, 170, 119), (150, 150, 150), (30, 30, 30)
_CELL, _GAP, _LINE = 160, 8, 14


def sweep_report(result: SweepResult) -> dict[str, object]:
    """Return the sweep as wo-4.0's report: the rule, the pick, the crop and every weight with its score or norms."""
    y0, x0, side = result.region if result.region is not None else (0, 0, None)
    return {
        "rule": result.rule,
        "regularization_strength": result.weight,
        "index": result.index,
        "lambda_over_h2max": result.weight / result.transfer_peak,
        "transfer_function_peak_squared": result.transfer_peak,
        "crop": {"y0": y0, "x0": x0, "size": side},
        "scored_z_index": result.scored_slice,
        "regularization_strengths": list(result.weights),
        "scores": list(result.scores),
        "residual_norms": list(result.residual_norms),
        "solution_norms": list(result.solution_norms),
        "warnings": list(result.warnings),
    }


def save_sweep_report(result: SweepResult, path: str | Path) -> Path:
    """Write :func:`sweep_report` of ``result`` as indented JSON at ``path``, creating its directory; return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sweep_report(result), indent=2))
    return path


def save_sweep_montage(result: SweepResult, path: str | Path, *, title: str | None = None) -> Path:
    """Write a PNG montage of the sweep at ``path``, creating its directory.

    Parameters
    ----------
    result : SweepResult
        From a :class:`~waveorder.nn.search.WeightSweep` with ``keep_slices=True``.
    path : str or Path
        The PNG to write.
    title : str, optional
        A first title line, for example the position and the time.

    Returns
    -------
    Path
        ``path``.

    Raises
    ------
    ValueError
        If ``result`` kept no slices.
    """
    from PIL import Image, ImageDraw

    if result.slices is None:
        raise ValueError("the sweep kept no slices to draw: sweep with WeightSweep(keep_slices=True)")
    count = len(result.weights)
    powers = [math.log10(weight / result.transfer_peak) for weight in result.weights]
    if result.rule == "otsu_cnr":
        objective, label = list(result.scores), "otsu_cnr"
    else:
        objective, label = l_curve_curvature(list(result.residual_norms), list(result.solution_norms)), "|kappa|"
    lines = [] if title is None else [title]
    pick = f"pick {result.weight:.4g} (weight/|H|^2max = 10^{powers[result.index]:+.2f})"
    lines.append(f"weight sweep: rule {result.rule}, {pick}, index {result.index} of {count}")
    if result.region is not None:
        y0, x0, side = result.region
        lines.append(f"crop {side}x{side} at (y={y0}, x={x0}), slice z={result.scored_slice}")
    if result.index in (0, count - 1):
        lines.append("the pick is at the edge of the sweep: widen the range")
    columns = max(2, math.ceil(math.sqrt(count)))
    slot, top = _CELL + _LINE + _GAP, 2 * _GAP + _LINE * len(lines)
    width = _GAP + columns * (_CELL + _GAP)
    # (255, 255, 255), not "white": PIL caches a named colour's tuple in ImageColor.colormap, which iohub reads as hex
    canvas = Image.new("RGB", (width, top + (math.ceil(count / columns) + 1) * slot + _GAP), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((_GAP, _GAP), "\n".join(lines), fill=_INK, spacing=2)

    def cell(image: Tensor, corner: tuple[int, int], caption: str, *, picked: bool = False) -> None:
        left, upper = corner
        draw.text((left, upper), caption, fill=_PICK if picked else _INK)
        canvas.paste(_grey(image).resize((_CELL, _CELL)), (left, upper + _LINE))
        frame = (left - 2, upper + _LINE - 2, left + _CELL + 1, upper + _LINE + _CELL + 1)
        draw.rectangle(frame, outline=_PICK if picked else _GREY, width=4 if picked else 1)

    if result.data_slice is not None:
        cell(result.data_slice, (_GAP, top), f"data, z={result.scored_slice}")
    box = (2 * _GAP + _CELL + 40, top + _LINE, width - _GAP, top + _CELL)
    _plot(draw, box, (powers, objective), pick=result.index, label=label)
    for index in range(count):
        corner = (_GAP + (index % columns) * (_CELL + _GAP), top + (1 + index // columns) * slot)
        caption = f"10^{powers[index]:+.2f}  {label} {objective[index]:.3g}"
        cell(result.slices[index], corner, caption, picked=index == result.index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


def _grey(image: Tensor) -> _Picture:
    """Return ``image`` ``(Y, X)`` as an 8-bit PIL image stretched to its 1st to 99th percentile."""
    from PIL import Image

    values = image.detach().to(torch.float32).cpu()
    low, high = torch.quantile(values.reshape(-1), torch.tensor([0.01, 0.99])).tolist()
    scaled = ((values - low) / max(high - low, 1e-12)).clamp(0, 1)
    height, width = values.shape
    return Image.frombytes("L", (width, height), bytes((scaled * 255).round().to(torch.uint8).flatten().tolist()))


def _plot(
    draw: _Pen,
    box: tuple[int, int, int, int],
    curve: tuple[list[float], list[float]],
    *,
    pick: int,
    label: str,
) -> None:
    """Draw ``curve`` (x and y) in ``box`` (left, upper, right, lower), the point ``pick`` marked."""
    x, y = curve
    left, upper, right, lower = box
    finite = [value for value in y if math.isfinite(value)] or [0.0]
    low, high = min(finite), max(finite)
    x_span, y_span = (x[-1] - x[0]) or 1.0, (high - low) or 1.0

    def at(u: float, v: float) -> tuple[float, float]:
        v = min(max(v, low), high) if math.isfinite(v) else low
        return left + (u - x[0]) / x_span * (right - left), lower - (v - low) / y_span * (lower - upper)

    draw.rectangle(box, outline=_GREY)
    points = [at(u, v) for u, v in zip(x, y, strict=True)]
    draw.line(points, fill=_INK, width=2)
    px, py = points[pick]
    draw.line([(px, upper), (px, lower)], fill=_PICK, width=2)
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=_PICK, width=2)
    draw.text((left, lower + 2), f"log10(weight/|H|^2max) {x[0]:+.1f} .. {x[-1]:+.1f}", fill=_INK)
    draw.text((left - 38, upper), f"{high:.3g}", fill=_INK)
    draw.text((left - 38, lower - _LINE), f"{low:.3g}", fill=_INK)
    draw.text((left + 4, upper + 2), label, fill=_INK)
