import pytest
import torch

from waveorder.data import GridPatches, RandomPatches
from waveorder.errors import ParameterError, ShapeError


def _covered(origins: torch.Tensor, size: tuple[int, int], field: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(field, dtype=torch.bool)
    for y, x in origins.tolist():
        mask[y : y + size[0], x : x + size[1]] = True
    return mask


def test_random_origins_are_inside_snapped_and_seeded():
    patches = RandomPatches((16, 24), per_field=200)
    origins = patches.origins((70, 61), generator=torch.Generator().manual_seed(3), stride=8)
    assert origins.shape == (200, 2)
    assert origins.dtype == torch.int64
    assert (origins % 8 == 0).all()
    assert (origins >= 0).all()
    assert (origins[:, 0] + 16 <= 70).all()
    assert (origins[:, 1] + 24 <= 61).all()
    assert origins[:, 0].max() == 48
    assert origins[:, 1].max() == 32
    again = patches.origins((70, 61), generator=torch.Generator().manual_seed(3), stride=8)
    assert torch.equal(origins, again)
    assert patches.count((70, 61), stride=8) == 200


def test_grid_covers_the_field_with_the_last_row_and_column_shifted_inward():
    patches = GridPatches((16, 16))
    origins = patches.origins((50, 40))
    assert _covered(origins, (16, 16), (50, 40)).all()
    ys, xs = sorted(set(origins[:, 0].tolist())), sorted(set(origins[:, 1].tolist()))
    assert ys == [0, 16, 32, 34]
    assert xs == [0, 16, 24]
    assert len(origins) == patches.count((50, 40)) == 12
    assert torch.equal(origins, patches.origins((50, 40)))


def test_grid_of_one_patch_and_stride_rounding():
    assert GridPatches((32, 32)).origins((32, 32)).tolist() == [[0, 0]]
    origins = GridPatches((16, 16)).origins((37, 37), stride=4)
    assert (origins % 4 == 0).all()
    assert _covered(origins, (16, 16), (37, 37))[:36, :36].all()  # up to stride - 1 edge pixels are left out


@pytest.mark.parametrize(
    "make",
    [
        lambda: RandomPatches((0, 16)),
        lambda: RandomPatches((16, 16), per_field=0),
        lambda: GridPatches((16, 0)),
    ],
)
def test_bad_arguments_are_refused(make):
    with pytest.raises(ParameterError):
        make()


def test_a_field_smaller_than_a_patch_is_refused():
    with pytest.raises(ShapeError, match="smaller than a patch"):
        GridPatches((64, 64)).origins((63, 128))
    with pytest.raises(ShapeError, match="smaller than a patch"):
        RandomPatches((64, 64)).origins((128, 32), generator=torch.Generator())
    with pytest.raises(ParameterError, match="stride"):
        RandomPatches((8, 8)).origins((32, 32), generator=torch.Generator(), stride=0)
