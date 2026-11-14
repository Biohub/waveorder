"""Tiny HCS plates with a simulated 3D phase volume, and the configurations the commands run on them."""

import itertools
from collections.abc import Sequence
from pathlib import Path

import iohub
import torch
import yaml

PIXEL_SIZE = (0.4, 0.15, 0.15)
SHAPE = (8, 32, 32)
TRUE_NA = 0.52
NA = "system.illumination.numerical_aperture"

PHASE = """\
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection: {numerical_aperture: 0.9}
  illumination:
    wavelength: 0.532
    numerical_aperture: {nominal: 0.45, tolerance: free, bounds: [0.3, 0.7]}
object: {prior: tikhonov, weight: 4e-4}
noise: {gaussian: {sigma: 1e-3}}
solver:
  calibration: {score: evidence, search: [{grid: {points: 17}}, {gradient: {max_iter: 30}}]}
"""
"""Thick phase with a free condenser aperture, calibrated by a grid then a gradient search."""

FIXED_PHASE = PHASE.replace(
    "numerical_aperture: {nominal: 0.45, tolerance: free, bounds: [0.3, 0.7]}", "numerical_aperture: 0.52"
).replace(
    "  calibration: {score: evidence, search: [{grid: {points: 17}}, {gradient: {max_iter: 30}}]}\n",
    "  reconstruction: closed_form\n",
)
"""Thick phase with nothing free: reconstructs without a store."""

FREE_PHASE = PHASE.replace("{grid: {points: 17}}, {gradient: {max_iter: 30}}", "{grid: {points: 5}}")
"""Thick phase with a free condenser aperture and a short search: ``wo rec`` calibrates it first."""

THIN_PHASE = FIXED_PHASE.replace("object: {prior: tikhonov, weight: 4e-4}", "object: {dimension: 2, weight: 1e-3}")
"""Thin phase: one output plane with ``Phase2D`` and ``Absorption2D``."""

FLUORESCENCE = """\
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection:
    wavelength: 0.52
    numerical_aperture: {nominal: 0.8, tolerance: free, bounds: [0.6, 1.0]}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
solver:
  calibration: {score: evidence, search: [{grid: {points: 3}}]}
"""
"""Thick fluorescence of channel ``BF`` with a free aperture: ``BF_Density3D``, with ``-t`` from ``calibrate-tf``."""

FIELD_TILED = FLUORESCENCE.replace(
    "  contrast: fluorescence\n",
    "  contrast: fluorescence\n"
    "  field_dependence: {tiled: {size_yx: [16, 16], overlap_yx: [0, 0], tile_blend: linear}}\n",
)
"""Fluorescence calibrated per 16 x 16 field tile."""

_copies = itertools.count()


def edited(path: Path, *settings: str) -> Path:
    """Write a copy of the YAML file at ``path`` with each ``key=value`` set, and return its path.

    ``key`` is a dotted path (an integer indexes a list) and ``value`` is read as YAML.
    """
    data = yaml.safe_load(path.read_text())
    for setting in settings:
        key, _, value = setting.partition("=")
        *parents, last = key.split(".")
        block = data
        for part in parents:
            block = block[int(part)] if isinstance(block, list) else block.setdefault(part, {})
        if isinstance(block, list):
            block[int(last)] = yaml.safe_load(value)
        else:
            block[last] = yaml.safe_load(value)
    copy = path.with_name(f"{path.stem}_{next(_copies)}{path.suffix}")
    copy.write_text(yaml.safe_dump(data, sort_keys=False))
    return copy


def simulate_phase(*, aperture: float = TRUE_NA, seed: int = 0) -> torch.Tensor:
    """Return ``(Z, Y, X)`` float32 brightfield contrast of a random phase object imaged at ``aperture``."""
    from waveorder.config import V4Config
    from waveorder.models import ReconstructionModel
    from waveorder.nn import Sampling, load_parameters

    model = ReconstructionModel.from_config(V4Config(**yaml.safe_load(PHASE)))
    load_parameters(model.imaging_model, {NA: aperture})
    generator = torch.Generator().manual_seed(seed)
    obj = 0.05 * torch.randn(1, *SHAPE, generator=generator)
    with torch.no_grad():
        data = model.imaging_model(obj, Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE))
    return (data + 1e-3 * torch.randn(data.shape, generator=generator))[0]


def intensities(contrast: torch.Tensor) -> torch.Tensor:
    """Return the brightfield intensities ``1000 * (1 + contrast)`` a camera records for simulated contrast."""
    return 1000 * (1 + contrast)


def series(contrast: torch.Tensor, times: int) -> torch.Tensor:
    """Return ``times`` brightfield intensity stacks ``1000 * (1 + contrast)``, slightly different (``T, Z, Y, X``)."""
    return torch.stack([intensities(contrast * (1 + 0.1 * t)) for t in range(times)])


def write_plate(
    path: Path,
    positions: Sequence[torch.Tensor],
    *,
    channel: str = "BF",
    pixel_size: tuple[float, float, float] = PIXEL_SIZE,
) -> list[Path]:
    """Write one position per ``(T, Z, Y, X)`` tensor (``A/1/0``, ``A/1/1``, ...); return their paths."""
    paths = []
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=[channel]) as plate:
        plate.zattrs["acquisition"] = {"microscope": "test"}
        for index, volume in enumerate(positions):
            position = plate.create_position("A", "1", str(index))
            position.create_image("0", volume[:, None].to(torch.float32).numpy())
            for axis, size in zip("zyx", pixel_size, strict=True):
                position.set_scale("0", axis, size)
            position.zattrs["biahub-deskew"] = {"angle": 30}
            paths.append(path / "A" / "1" / str(index))
    return paths


def read_output(path: Path, position: str = "A/1/0") -> tuple[list[str], torch.Tensor, dict]:
    """Return an output position's channel names, its image ``0`` and its attributes."""
    with iohub.open_ome_zarr(path / position, mode="r") as node:
        return list(node.channel_names), torch.as_tensor(node["0"][:]), dict(node.zattrs)
