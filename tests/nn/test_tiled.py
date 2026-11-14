"""Independent per-tile calibration: the batched stages give each tile exactly what calibrate gives it alone."""

from typing import ClassVar

import pytest
import torch

from tests.nn.conftest import FREE_NA, NA, make_model, simulate
from waveorder.errors import IncompatibleModelError, ParameterError, PlacementError, ShapeError
from waveorder.nn import (
    Evidence,
    GradientSearch,
    GridSearch,
    LBFGSSearch,
    NelderMeadSearch,
    RandomSearch,
    Sampling,
    calibrate,
    calibrate_tiles,
    load_parameters,
)
from waveorder.nn.search.base import CalibrationObjective, Estimate

SAMPLING = Sampling(shape=(8, 32, 32), pixel_size=(0.25, 0.1, 0.1))
APERTURES = (0.9, 1.0, 1.1)
WEIGHT = "object_prior.weight"


def _tiles(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.stack([simulate(make_model(na), SAMPLING, scale=3.0, seed=s)[1] for s, na in enumerate(APERTURES)]).to(
        dtype
    )


def _model(dtype: torch.dtype = torch.float32):
    return make_model(FREE_NA).to(dtype)


def _alone(tile: torch.Tensor, stages, *, dtype: torch.dtype = torch.float32):
    return calibrate(tile, _model(dtype), SAMPLING, score=Evidence(), search=stages)


@pytest.mark.parametrize("chunk_size", [None, 2], ids=["all_at_once", "chunks_of_two"])
def test_batched_stages_give_each_tile_its_own_calibration(chunk_size):
    stages = (GridSearch(points=7), RandomSearch(samples=6), GradientSearch(iterations=15))
    tiles = _tiles(torch.float64)
    model = _model(torch.float64)
    before = model.get_parameter(NA).item()
    result = calibrate_tiles(tiles, model, SAMPLING, score=Evidence(), search=stages, chunk_size=chunk_size)
    assert model.get_parameter(NA).item() == before  # the model is not modified
    for index, tile in enumerate(tiles):
        alone = _alone(tile, stages, dtype=torch.float64)
        assert result.parameters[NA][index].item() == pytest.approx(alone.parameters[NA].item(), rel=1e-12)
        assert result.objective[index].item() == pytest.approx(alone.objective.item(), rel=1e-12)
        assert [name for name, _ in result.tiles[index].trace] == [name for name, _ in alone.trace]
        assert [level for _, level in result.tiles[index].trace] == pytest.approx(
            [v for _, v in alone.trace], rel=1e-12
        )
    # Tiles simulated at apertures 0.9, 1.0 and 1.1 calibrate in that order.
    assert result.parameters[NA].tolist() == sorted(result.parameters[NA].tolist())
    assert result.parameters[NA].shape == (3,)
    assert result.parameters[NA].dtype == torch.float64


@pytest.mark.parametrize("stage", [LBFGSSearch(iterations=5), NelderMeadSearch(iterations=15)], ids=["lbfgs", "nelder"])
def test_a_stage_that_reads_back_runs_tile_by_tile(stage):
    stages = (GridSearch(points=5), stage)
    tiles = _tiles()
    result = calibrate_tiles(tiles, _model(), SAMPLING, score=Evidence(), search=stages, chunk_size=2)
    for index, tile in enumerate(tiles):
        alone = _alone(tile, stages)
        assert result.parameters[NA][index].item() == pytest.approx(alone.parameters[NA].item(), rel=1e-6)


def test_a_warm_start_starts_each_tile_at_its_row():
    tiles = _tiles()
    start = {NA: torch.tensor([0.95, 1.0, 1.05])}
    stages = (GradientSearch(iterations=5),)
    result = calibrate_tiles(tiles, _model(), SAMPLING, score=Evidence(), search=stages, start=start)
    for index, tile in enumerate(tiles):
        model = _model()
        load_parameters(model, {NA: start[NA][index]})  # the same prior; only the starting value moves
        alone = calibrate(tile, model, SAMPLING, score=Evidence(), search=stages)
        assert result.parameters[NA][index].item() == pytest.approx(alone.parameters[NA].item(), rel=1e-6)


class NoTileAxis:
    """A device-side plugin stage written for one tile: it collapses the tile axis."""

    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def run(self, objective: CalibrationObjective, *, start: Estimate) -> Estimate:
        del objective
        return Estimate({name: value.mean() for name, value in start.point.items()}, start.objective.mean())


def test_misuse_is_named():
    tiles, model = _tiles(), _model()
    cases = [
        (dict(start={"system.detection.numerical_apertur": torch.ones(3)}), "start.system.detection.numerical_apertur"),
        (dict(start={NA: torch.ones(2)}), f"start.{NA}"),
        (dict(chunk_size=0), "chunk_size"),
        (dict(search=()), "search"),
    ]
    for kwargs, path in cases:
        with pytest.raises(ParameterError) as error:
            calibrate_tiles(tiles, model, SAMPLING, score=Evidence(), **kwargs)  # ty: ignore[invalid-argument-type]
        assert error.value.path == path
    with pytest.raises(PlacementError, match=rf"calibrate_tiles start\.{NA}"):
        calibrate_tiles(tiles, model, SAMPLING, score=Evidence(), start={NA: torch.ones(3, dtype=torch.float64)})
    with pytest.raises(ShapeError, match="calibrate_tiles data has shape"):
        calibrate_tiles(tiles[..., :-1], model, SAMPLING, score=Evidence())
    with pytest.raises(ParameterError, match="leading tile axis") as error:
        calibrate_tiles(tiles[:0], model, SAMPLING, score=Evidence())
    assert error.value.path == "tiles"
    with pytest.raises(IncompatibleModelError, match=r"NoTileAxis\.run on 3 tiles returned shapes"):
        calibrate_tiles(tiles, model, SAMPLING, score=Evidence(), search=(NoTileAxis(),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batched_tiles_on_cuda_match_the_cpu():
    stages = (GridSearch(points=5), GradientSearch(iterations=5))
    tiles = _tiles(torch.float64)
    cpu = calibrate_tiles(tiles, _model(torch.float64), SAMPLING, score=Evidence(), search=stages)
    cuda = calibrate_tiles(tiles.cuda(), _model(torch.float64).cuda(), SAMPLING, score=Evidence(), search=stages)
    assert cuda.parameters[NA].device.type == "cuda"
    torch.testing.assert_close(cuda.parameters[NA].cpu(), cpu.parameters[NA], rtol=1e-9, atol=0)
    torch.testing.assert_close(cuda.objective.cpu(), cpu.objective, rtol=1e-9, atol=0)
