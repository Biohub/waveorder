import re

import pytest
import torch

from waveorder.errors import ParameterError
from waveorder.tiling import TilePlan

PLANS = [
    # (shape, tile, overlap, chunks): ragged edges, a last core smaller than the overlap, a single-tile axis
    ((20, 50, 37), (8, 16, 12), (2, 4, 3), None),
    ((9, 64, 64), (9, 20, 20), (0, 6, 10), (3, 16, 16)),
    ((5, 41, 33), (5, 8, 8), (0, 4, 4), (1, 7, 5)),
]


def _coverage(plan: TilePlan) -> torch.Tensor:
    total = torch.zeros(plan.shape, dtype=torch.float64)
    for region in plan.regions:
        box = tuple(slice(a, b) for a, b in region.blend)
        wz, wy, wx = (plan.weights[axis][g] for axis, g in enumerate(region.grid))
        total[box] += wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return total


@pytest.mark.parametrize("blend", ["linear", "cosine"])
@pytest.mark.parametrize(("shape", "tile", "overlap", "chunks"), PLANS)
def test_blend_weights_sum_to_one(shape, tile, overlap, chunks, blend):
    plan = TilePlan(shape=shape, tile=tile, overlap=overlap, chunks=chunks, blend=blend)
    torch.testing.assert_close(_coverage(plan), torch.ones(shape, dtype=torch.float64), rtol=0, atol=1e-12)


@pytest.mark.parametrize("blend", ["linear", "cosine"])
def test_ramps_fade_out_at_neighbours_and_stay_flat_at_the_volume_edge(blend):
    plan = TilePlan(shape=(1, 1, 40), tile=(1, 1, 10), overlap=(0, 0, 4), blend=blend)
    first, second = plan.weights[2][0], plan.weights[2][1]
    assert torch.all(first[:6] == 1)  # no neighbour below x = 0: no ramp there
    assert torch.all(torch.diff(first[6:]) < 0)  # fades out across [6, 14)
    assert torch.all(torch.diff(second[:8]) > 0)  # and the neighbour fades in over the same voxels
    torch.testing.assert_close(first[6:] + second[:8], torch.ones(8, dtype=torch.float64), rtol=0, atol=1e-12)


@pytest.mark.parametrize(("shape", "tile", "overlap", "chunks"), PLANS)
def test_regions_nest_and_cores_partition_the_volume(shape, tile, overlap, chunks):
    plan = TilePlan(shape=shape, tile=tile, overlap=overlap, chunks=chunks)
    owners = torch.zeros(shape, dtype=torch.int64)
    for region in plan.regions:
        owners[tuple(slice(a, b) for a, b in region.core)] += 1
        for n, (c0, c1), (b0, b1), (r0, r1), o in zip(
            shape, region.core, region.blend, region.read, overlap, strict=True
        ):
            if (b0, b1) == (0, n):
                continue  # one tile on this axis: everything is read
            assert (b0, b1) == (max(c0 - o, 0), min(c1 + o, n))
            assert r0 <= b0  # the read region contains the blend region
            assert r1 >= b1
            assert r0 >= 0
            assert r1 <= n
    assert torch.all(owners == 1)


@pytest.mark.parametrize(("shape", "tile", "overlap", "chunks"), PLANS)
def test_every_tile_reads_one_shape(shape, tile, overlap, chunks):
    plan = TilePlan(shape=shape, tile=tile, overlap=overlap, chunks=chunks)
    assert len({region.read_shape for region in plan.regions}) == 1


def _smooth(n):
    for factor in (2, 3, 5, 7):
        while n % factor == 0:
            n //= factor
    return n == 1


@pytest.mark.parametrize(("shape", "tile", "overlap", "chunks"), PLANS)
def test_read_lengths_are_fft_friendly_where_the_volume_allows(shape, tile, overlap, chunks):
    plan = TilePlan(shape=shape, tile=tile, overlap=overlap, chunks=chunks)
    for n, g, length in zip(shape, plan.grid, plan.read_shape, strict=True):
        # One tile reads the whole axis; otherwise the length is smooth unless that would exceed the volume.
        assert length == n if g == 1 else (_smooth(length) or length == n)


def test_read_length_is_the_smallest_smooth_length_that_holds_the_tile():
    # 16 + 2 * 3 = 22 voxels are needed; 22 = 2 * 11, so the read grows to 24 = 2^3 * 3.
    plan = TilePlan(shape=(1, 200, 200), tile=(1, 16, 16), overlap=(0, 3, 3))
    assert plan.read_shape == (1, 24, 24)
    # A chunk of 17 has no smooth multiple; aligned reads keep their length.
    aligned = TilePlan(shape=(1, 200, 200), tile=(1, 17, 17), overlap=(0, 4, 4), chunks=(1, 17, 17))
    assert all(length % 17 == 0 for length in aligned.read_shape[1:])


def test_reads_are_aligned_to_chunks():
    shape, chunks = (8, 100, 90), (4, 16, 15)
    plan = TilePlan(shape=shape, tile=(8, 30, 30), overlap=(0, 5, 5), chunks=chunks)
    for region in plan.regions:
        for n, c, (r0, r1) in zip(shape, chunks, region.read, strict=True):
            # A tile shifted inward at the high edge ends at the volume boundary; every other bound is on a chunk.
            assert r0 % c == 0 or r1 == n
            assert r1 % c == 0 or r1 == n


def test_unaligned_reads_decode_chunks_in_part():
    # Without chunk information the reads start inside chunks; the aligned plan never does.
    shape, chunks = (1, 64, 64), (1, 16, 16)
    loose = TilePlan(shape=shape, tile=(1, 24, 24), overlap=(0, 2, 2))
    aligned = TilePlan(shape=shape, tile=(1, 24, 24), overlap=(0, 2, 2), chunks=chunks)

    def partial(plan):
        return sum(
            (a % c != 0 and a != 0) or (b % c != 0 and b != n)
            for region in plan.regions
            for n, c, (a, b) in zip(shape, chunks, region.read, strict=True)
        )

    assert partial(loose) > 0
    assert partial(aligned) == 0


def test_blocks_and_contributors_are_inverse():
    plan = TilePlan(shape=(20, 50, 37), tile=(8, 16, 12), overlap=(2, 4, 3))
    for region in plan.regions:
        for block in plan.blocks(region):
            assert region.index in plan.contributors(plan.regions[block])
    assert plan.peak_pending_blocks < len(plan.regions)


def test_no_overlap_needs_no_accumulator_beyond_one_core():
    plan = TilePlan(shape=(16, 64, 64), tile=(8, 16, 16))
    assert all(plan.contributors(region) == (region.index,) for region in plan.regions)
    assert plan.peak_pending_blocks == 1


def test_overlap_along_z_keeps_a_slab_of_rows_open():
    # Rows of the first z slab finish only once the second slab's tiles have contributed: all three stay open.
    stacked = TilePlan(shape=(24, 96, 80), tile=(12, 32, 32), overlap=(2, 4, 4))
    assert stacked.peak_open_rows > stacked.grid[1]
    # In one slab a row finishes during the next row's tiles; only the last two rows finish together.
    flat = TilePlan(shape=(24, 96, 80), tile=(24, 32, 32), overlap=(0, 4, 4))
    assert flat.peak_open_rows == 2
    assert TilePlan(shape=(24, 96, 80), tile=(24, 32, 32)).peak_open_rows == 1


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"overlap": (0, 9, 0)}, "overlap.y"),  # more than half the tile
        ({"tile": (8, 0, 16)}, "tile.y"),
        ({"chunks": (1, 1)}, "chunks"),
        ({"blend": "gaussian"}, "blend"),
    ],
)
def test_invalid_plans_name_the_field(kwargs, path):
    options = {"shape": (16, 64, 64), "tile": (8, 16, 16)} | kwargs
    with pytest.raises(ParameterError, match=rf"TilePlan {re.escape(path)}") as caught:
        TilePlan(**options)
    assert caught.value.path == path
