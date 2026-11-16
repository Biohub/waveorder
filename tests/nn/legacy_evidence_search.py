"""wo-4.0's ``random`` calibration stage draw, as a test-only reference for :class:`waveorder.nn.RandomSearch`.

Copied from WaveOrder 4.0 (commit 85282ce) ``waveorder/calibrate/evidence_search.py``: ``_stage_ranges`` and the
``SearchRandom`` branch and ``clamp`` of ``run_evidence_calibration`` (BSD-3-Clause, Chan Zuckerberg Biohub). Changes
from the original, none of which alters a draw: the draws are returned instead of scored, start from zeros instead
of the previous stage's point (every entry is overwritten), and ``tolerance == "free"`` is tested with
``isinstance(tolerance, str)`` for the type checker.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParameterSpec:
    """wo-4.0 ``ParameterSpec``: ``tolerance`` is a number or ``"free"``."""

    name: str
    nominal: float
    tolerance: float | str
    bounds: tuple[float, float] | None


def _stage_ranges(specs: list[ParameterSpec], centers: torch.Tensor) -> list[tuple[float, float]]:
    ranges = []
    for spec, center in zip(specs, centers.tolist(), strict=True):
        if spec.bounds is not None:
            ranges.append(spec.bounds)
        elif not isinstance(spec.tolerance, str):
            ranges.append((center - 2.0 * spec.tolerance, center + 2.0 * spec.tolerance))
        else:
            raise ValueError(f"grid/random stage needs bounds for 'free' parameter {spec.name!r}")
    return ranges


def random_draws(specs: list[ParameterSpec], *, samples: int, seed: int) -> torch.Tensor:
    """Return the ``(samples, len(specs))`` float64 points wo-4.0's random stage scored."""

    def clamp(values: torch.Tensor) -> torch.Tensor:
        clamped = values.clone()
        for i, spec in enumerate(specs):
            if spec.bounds is not None:
                clamped[i] = clamped[i].clamp(spec.bounds[0], spec.bounds[1])
        return clamped

    nominals = torch.tensor([spec.nominal for spec in specs], dtype=torch.float64)
    current = torch.zeros(len(specs), dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    ranges = _stage_ranges(specs, nominals)
    draws = []
    with torch.no_grad():
        for _ in range(samples):
            draw = current.clone()
            for i, spec in enumerate(specs):
                if isinstance(spec.tolerance, str):  # "free"
                    lo, hi = ranges[i]
                    draw[i] = lo + (hi - lo) * torch.rand((), generator=generator, dtype=torch.float64)
                else:
                    draw[i] = spec.nominal + spec.tolerance * torch.randn((), generator=generator, dtype=torch.float64)
            draws.append(clamp(draw))
    return torch.stack(draws)
