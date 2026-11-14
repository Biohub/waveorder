"""The solve context: where the data of a solve sits, set by the position methods and the tile driver."""

import pytest
import torch

from tests.models.plates import PIXEL_SIZE, SHAPE, fluorescence_model, write_plate
from waveorder.errors import ParameterError
from waveorder.nn import ImagingModel, InverseProblem, SolveContext, Solver, current_context, solve_context
from waveorder.nn.sampling import ZYX
from waveorder.nn.solvers import SolveResult

SEEN: list[SolveContext] = []
"""What the recording solver saw, one context per solve."""


class Recording(Solver):
    """A solver that records the solve context and returns zeros."""

    def check(self, model: ImagingModel) -> None:
        del model

    def tile_halo(self) -> ZYX[int]:
        return (0, 2, 2)

    def run(self, problem: InverseProblem) -> SolveResult:
        SEEN.append(current_context())
        estimate = problem.data.new_zeros(problem.model.system.object_shape(problem.sampling))
        return SolveResult(estimate, problem.data.new_zeros(()), 0, "direct")


@pytest.fixture(autouse=True)
def _forget():
    SEEN.clear()
    yield
    SEEN.clear()


def test_outside_any_block_nothing_is_known():
    assert current_context() == SolveContext()


def test_blocks_nest_and_each_keeps_what_it_does_not_set():
    with solve_context(position="A/1/0", time=2, field_shape=(64, 96)) as outer:
        assert outer == SolveContext(position="A/1/0", time=2, field_shape=(64, 96))
        with solve_context(region=(slice(0, 32), slice(16, 48))) as inner:
            assert inner == current_context()
            assert inner == SolveContext(
                position="A/1/0", time=2, field_shape=(64, 96), region=(slice(0, 32), slice(16, 48))
            )
        assert current_context() == outer
    assert current_context() == SolveContext()


def test_a_block_that_raises_restores_the_enclosing_context():
    with pytest.raises(RuntimeError), solve_context(position="A/1/0"):
        raise RuntimeError
    assert current_context() == SolveContext()


@pytest.mark.parametrize(
    ("argument", "match"),
    [
        ({"field_shape": (64,)}, "field_shape must be \\(Y, X\\), two positive ints"),
        ({"field_shape": (64, 0)}, "field_shape must be \\(Y, X\\), two positive ints"),
        ({"region": (slice(0, 8),)}, "region must be two slices"),
        ({"region": (slice(8, 4), slice(0, 8))}, "region must be two slices"),
        ({"region": (slice(0, 8, 2), slice(0, 8))}, "region must be two slices"),
    ],
)
def test_a_malformed_field_or_region_is_refused(argument, match):
    with pytest.raises(ParameterError, match=match), solve_context(**argument):
        pass


def test_the_tile_driver_sets_each_tiles_read_region_within_the_field():
    model = fluorescence_model(solver=Recording())
    plan = model.tile_plan((4, 48, 64), pixel_size=PIXEL_SIZE, tile=(4, 24, 32))
    data = 100 + torch.rand(1, 4, 48, 64)
    with solve_context(position="A/1/0", time=1, field_shape=(48, 64)):
        model.reconstruct_tiles([(data, torch.zeros(1, 4, 48, 64))], pixel_size=PIXEL_SIZE, plan=plan)
    regions = [(slice(y0, y1), slice(x0, x1)) for _, (y0, y1), (x0, x1) in (tile.read for tile in plan.regions)]
    assert len(regions) == 4
    assert [SolveContext(position="A/1/0", time=1, field_shape=(48, 64), region=region) for region in regions] == SEEN


def test_the_position_methods_set_position_time_and_field_around_each_unit(tmp_path):
    plate = tmp_path / "in.zarr"
    write_plate(plate, [100 + torch.rand(2, *SHAPE), 100 + torch.rand(2, *SHAPE)])
    model = fluorescence_model(solver=Recording())
    report = model.reconstruct_positions(plate, output=tmp_path / "whole.zarr", channels=["BF"])
    assert report.count("complete") == 4
    field = SHAPE[1:]
    assert sorted((seen.position, seen.time) for seen in SEEN) == [
        ("A/1/0", 0),
        ("A/1/0", 1),
        ("A/1/1", 0),
        ("A/1/1", 1),
    ]
    whole = (slice(0, field[0]), slice(0, field[1]))
    assert {seen.field_shape for seen in SEEN} == {field}
    assert all(seen.region in (None, whole) for seen in SEEN)  # the whole field, as one tile or none
    SEEN.clear()
    model.reconstruct_positions(plate, output=tmp_path / "tiled.zarr", channels=["BF"], tile=(8, 16, 16), times=[0])
    assert {seen.position for seen in SEEN} == {"A/1/0", "A/1/1"}
    assert {seen.field_shape for seen in SEEN} == {field}
    assert all(seen.region is not None and seen.region != whole and seen.time == 0 for seen in SEEN)
    assert len(SEEN) == 8  # four tiles per position
