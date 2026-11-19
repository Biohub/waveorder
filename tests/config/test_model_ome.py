"""``ReconstructionModel.reconstruct_tiles`` on the volumes of OME-Zarr positions (:mod:`waveorder.tiling.merge`)."""

from typing import Any

import iohub
import pytest
import torch
import yaml
from tests.tiling.conftest import create_sink

from waveorder.config import V4Config
from waveorder.errors import ParameterError
from waveorder.io.positions import pixel_size
from waveorder.models import ReconstructionModel
from waveorder.tiling.merge import volumes

CONFIG = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.12}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection: {wavelength: 0.52, numerical_aperture: 1.2}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 2.0}}
"""
SHAPE = (8, 32, 32)
PIXEL_SIZE = (0.25, 0.1, 0.12)  # data.pixel_size of CONFIG


def _model() -> ReconstructionModel:
    return ReconstructionModel.from_config(V4Config(**yaml.safe_load(CONFIG)))


@pytest.fixture
def store(tmp_path):
    """A single-FOV store with 2 time points and channels BF, GFP at the configured voxel size."""
    data = torch.rand(2, 2, *SHAPE, generator=torch.Generator().manual_seed(0))
    path = tmp_path / "raw.zarr"
    with iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=["BF", "GFP"]) as position:
        position.create_image("0", data.numpy(), chunks=(1, 1, 8, 16, 16))
        for axis, size in zip("zyx", PIXEL_SIZE, strict=True):
            position.set_scale("0", axis, size)
    return path, data


def test_the_volumes_of_a_pair_of_positions_reconstruct_every_time_point_of_the_named_channel(store, tmp_path):
    path, data = store
    model = _model()
    plan = model.tile_plan(SHAPE, pixel_size=PIXEL_SIZE, tile=(8, 16, 16), overlap=(0, 4, 4))
    with (
        iohub.open_ome_zarr(path, mode="r") as source,
        create_sink(str(tmp_path / "recon.zarr"), like=source, fields=["fluorescence"]) as sink,
    ):
        pairs = volumes(source, sink, channels=["GFP"])
        stats = model.reconstruct_tiles(pairs, pixel_size=pixel_size(source), plan=plan)
    assert stats.tiles == 2 * len(plan.regions)
    with iohub.open_ome_zarr(tmp_path / "recon.zarr", mode="r") as written:
        result = torch.as_tensor(written["0"][:])
    for t in range(2):
        expected = torch.full((1, *SHAPE), float("nan"))
        model.reconstruct_tiles([(data[t, 1:2].clone(), expected)], pixel_size=PIXEL_SIZE, plan=plan)  # GFP: channel 1
        torch.testing.assert_close(result[t], expected, rtol=0, atol=0)


def test_a_pair_of_positions_is_refused_with_both_ways_to_reconstruct_it(store, tmp_path):
    path, _ = store
    model = _model()
    plan = model.tile_plan(SHAPE, pixel_size=PIXEL_SIZE, tile=(8, 16, 16))
    loose: Any = model  # ty rejects positions statically; the runtime check names the fix
    with (
        iohub.open_ome_zarr(path, mode="r") as source,
        create_sink(str(tmp_path / "recon.zarr"), like=source, fields=["fluorescence"]) as sink,
        pytest.raises(ParameterError, match=r"reconstruct_positions, or .*waveorder\.tiling\.merge\.volumes") as caught,
    ):
        loose.reconstruct_tiles([(source, sink)], pixel_size=PIXEL_SIZE, plan=plan)
    assert caught.value.path == "volumes.0"
