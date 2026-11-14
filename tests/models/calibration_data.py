"""Tiny OME-Zarr plates with simulated volumes, and the configurations whose models calibrate them."""

from pathlib import Path

import iohub
import torch
import yaml
from iohub.ngff.models import ChannelAxisMeta, SpaceAxisMeta, TimeAxisMeta

from waveorder.config import V4Config
from waveorder.io import TfStore
from waveorder.models import ReconstructionModel
from waveorder.nn import Sampling, load_parameters

PIXEL_SIZE = (0.4, 0.15, 0.15)
SHAPE = (8, 32, 32)
TRUE_NA = 0.52
NA = "system.illumination.numerical_aperture"
WEIGHT = "object_prior.weight"

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

FIELD_TILED = """\
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  field_dependence: {tiled: {size_yx: [32, 32], overlap_yx: [0, 0], tile_blend: linear}}
  index_of_refraction_media: 1.33
  detection:
    wavelength: 0.52
    numerical_aperture: {nominal: 1.15, tolerance: free, bounds: [1.0, 1.3]}
object: {prior: tikhonov, weight: 1e-4}
noise: {gaussian: {sigma: 0.01}}
solver:
  calibration: {score: evidence, search: [{grid: {points: 3}}]}
"""
TILED_SHAPE = (8, 32, 64)
TILED_PIXEL_SIZE = (0.25, 0.1, 0.1)
TILED_NA = "system.detection.numerical_aperture"


def raw_config(text: str = PHASE) -> dict:
    """Return a configuration text as a mapping to edit before :func:`model_of`."""
    return yaml.safe_load(text)


def model_of(config: str | dict = PHASE) -> ReconstructionModel:
    """Return the model a configuration text (or its edited mapping) describes."""
    raw = yaml.safe_load(config) if isinstance(config, str) else config
    return ReconstructionModel.from_config(V4Config(**raw))


def simulate_phase(*, aperture: float = TRUE_NA, seed: int = 0) -> torch.Tensor:
    """Return ``(Z, Y, X)`` float32 brightfield contrast of a random phase object imaged at ``aperture``."""
    model = model_of(PHASE)
    load_parameters(model.imaging_model, {NA: aperture})
    generator = torch.Generator().manual_seed(seed)
    obj = 0.05 * torch.randn(1, *SHAPE, generator=generator)
    with torch.no_grad():
        data = model.imaging_model(obj, Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE))
    return (data + 1e-3 * torch.randn(data.shape, generator=generator))[0]


def simulate_tiled() -> torch.Tensor:
    """Return ``(Z, Y, X)`` fluorescence data of a random object on :data:`TILED_SHAPE`."""
    model = model_of(FIELD_TILED)
    obj = torch.randn(1, *TILED_SHAPE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return model.imaging_model(obj, Sampling(shape=TILED_SHAPE, pixel_size=TILED_PIXEL_SIZE))[0]


def intensities(contrast: torch.Tensor) -> torch.Tensor:
    """Return the brightfield intensities ``1000 * (1 + contrast)`` a camera records for simulated contrast."""
    return 1000 * (1 + contrast)


def write_position(
    path: Path, volume: torch.Tensor, *, channel: str = "BF", pixel_size: tuple[float, float, float] = PIXEL_SIZE
) -> Path:
    """Write a plate of one position ``A/1/0`` holding ``volume`` as ``(T=1, C=1, Z, Y, X)``; return the position.

    The axes are given, in um: iohub's default axes are objects shared by every node of the process, so a test that
    sets a unit on one changes the unit of every position written after it.
    """
    axes = [
        TimeAxisMeta(name="t", unit="second"),
        ChannelAxisMeta(name="c"),
        *(SpaceAxisMeta(name=name, unit="micrometer") for name in "zyx"),
    ]
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=[channel], axes=axes) as plate:
        position = plate.create_position("A", "1", "0")
        position.create_image("0", volume[None, None].numpy())
        for axis, size in zip("zyx", pixel_size, strict=True):
            position.set_scale("0", axis, size)
    return path / "A" / "1" / "0"


def store_of(names=(NA,), values=(0.5,), **fields) -> TfStore:
    """Return a one-row store of ``values``."""
    return TfStore(parameter_names=tuple(names), parameters=torch.tensor([list(values)], dtype=torch.float64), **fields)
