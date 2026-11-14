"""``imaging_system.field_dependence: tiled``: per-tile calibration, its tile map, and per-tile reconstruction."""

import pytest
import torch
import yaml

from waveorder.config import V4Config
from waveorder.errors import ConfigurationError, Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import Sampling, TileCalibration, load_parameters

CONFIG = """
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
SHAPE = (8, 32, 256)  # 8 tiles along x; each reads 96 voxels (halo 30), so the outer three see one aperture
PIXEL_SIZE = (0.25, 0.1, 0.1)  # data.pixel_size of CONFIG
NA = "system.detection.numerical_aperture"


def _raw() -> dict:
    return yaml.safe_load(CONFIG)


def _model(**solver: object) -> ReconstructionModel:
    raw = _raw()
    raw["solver"].update(solver)
    return ReconstructionModel.from_config(V4Config(**raw))


def _whole_field_model() -> ReconstructionModel:
    """The model of CONFIG without its field tiles: it calibrates the data it is given as one system."""
    raw = _raw()
    raw["imaging_system"]["field_dependence"] = "shift_invariant"
    return ReconstructionModel.from_config(V4Config(**raw))


def _two_aperture_volume(model: ReconstructionModel) -> torch.Tensor:
    """Data whose left half was imaged at NA 1.0 and right half at NA 1.3, each from a prior-like object."""
    sampling = Sampling(shape=(SHAPE[0], SHAPE[1], SHAPE[2] // 2), pixel_size=PIXEL_SIZE)
    generator = torch.Generator().manual_seed(0)
    halves = []
    for aperture in (1.0, 1.3):
        load_parameters(model.imaging_model, {NA: aperture})
        with torch.no_grad():
            image = model.imaging_model(torch.randn(1, *sampling.shape, generator=generator), sampling)
        halves.append(image + 0.01 * torch.randn(image.shape, generator=generator))
    load_parameters(model.imaging_model, {NA: 1.15})
    return torch.cat(halves, dim=-1)


def test_the_field_tile_plan_follows_the_tiled_settings():
    plan = _model().field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    assert (plan.tile, plan.overlap, plan.blend) == ((8, 32, 32), (0, 0, 0), "linear")
    assert len(plan.regions) == 8


@pytest.mark.parametrize(
    ("overlap", "field"),
    [([17, 24], "overlap_yx.y"), ([16, 25], "overlap_yx.x")],
    ids=["y", "x"],
)
def test_an_overlap_above_half_a_tile_is_refused_at_its_axis_of_the_tiled_block(overlap, field):
    raw = _raw()
    raw["imaging_system"]["field_dependence"] = {"tiled": {"size_yx": [32, 48], "overlap_yx": overlap}}
    with pytest.raises(Rejected) as error:
        ReconstructionModel.from_config(raw)
    (problem,) = error.value.problems
    assert problem.message.startswith(f"imaging_system.field_dependence.tiled.{field}: ")
    assert "exceeds half of size_yx" in problem.message
    raw["imaging_system"]["field_dependence"]["tiled"]["overlap_yx"] = [16, 24]
    tiles = ReconstructionModel.from_config(raw).field_tiles
    assert tiles is not None
    assert tiles.overlap_yx == (16, 24)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [([16, 16], (8, 8)), ([32, 48], (16, 24)), ([64, 256], (32, 32)), ([9, 100], (4, 32))],
    ids=["small", "mixed", "large", "odd"],
)
def test_the_default_overlap_is_32_or_half_a_small_tile(size, overlap):
    # The default of waveorder.models.FieldTiles: valid for every tile size, and 32 for tiles of 64 or more.
    raw = _raw()
    raw["imaging_system"]["field_dependence"] = {"tiled": {"size_yx": size}}
    tiles = ReconstructionModel.from_config(raw).field_tiles
    assert tiles is not None
    assert (tiles.size_yx, tiles.overlap_yx) == (tuple(size), overlap)


def test_each_tile_calibrates_as_its_read_region_alone():
    model = _model()
    plan = model.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    data = _two_aperture_volume(model)
    calibration = model.calibrate_tiles(data, pixel_size=PIXEL_SIZE, plan=plan)
    assert isinstance(calibration, TileCalibration)
    assert float(model.imaging_model.get_parameter(NA).detach()) == pytest.approx(1.15)  # the model keeps its value
    whole = _whole_field_model()  # ``calibrate`` takes one field, so a tile alone is calibrated without field tiles
    alone = []
    for region in plan.regions:
        (z0, z1), (y0, y1), (x0, x1) = region.read
        tile = data[:, z0:z1, y0:y1, x0:x1].contiguous()
        alone.append(float(whole.calibrate(tile, pixel_size=PIXEL_SIZE, prepared=True).parameters[NA]))
        load_parameters(whole.imaging_model, {NA: 1.15})
    torch.testing.assert_close(calibration.parameters[NA], torch.tensor(alone))
    assert alone[0] == pytest.approx(1.0)  # the outer tiles see one aperture each
    assert alone[-1] == pytest.approx(1.3)


def test_a_per_tile_table_solves_each_tile_with_its_own_values():
    model = _model()
    plan = model.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    data = _two_aperture_volume(model)

    def tiles(table: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        sink = torch.full((1, *SHAPE), float("nan"))
        model.reconstruct_tiles([(data, sink)], pixel_size=PIXEL_SIZE, plan=plan, parameters=table)
        return sink

    varying = tiles({NA: torch.tensor([1.0] * 4 + [1.3] * 4)})
    uniform = tiles({NA: torch.full((8,), 1.3)})
    load_parameters(model.imaging_model, {NA: 1.3})
    torch.testing.assert_close(uniform, tiles(), rtol=0, atol=0)  # a constant table is the model at that value
    torch.testing.assert_close(varying[..., 128:], uniform[..., 128:], rtol=0, atol=0)  # the cores of tiles 4 to 7
    assert (varying[..., :128] - uniform[..., :128]).abs().max() > 1e-3


def test_tiled_work_names_the_settings_it_cannot_honour():
    raw = _raw()
    raw["imaging_system"]["field_dependence"] = "shift_invariant"
    with pytest.raises(ConfigurationError) as error:
        ReconstructionModel.from_config(V4Config(**raw)).field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    assert error.value.field == "imaging_system.field_dependence"
    padded = _model(padding={"z": 2})
    plan = padded.field_tile_plan(SHAPE, pixel_size=PIXEL_SIZE)
    with pytest.raises(ConfigurationError) as error:
        padded.calibrate_tiles(torch.zeros(1, *SHAPE), pixel_size=PIXEL_SIZE, plan=plan)
    assert error.value.field == "solver.padding.z"
