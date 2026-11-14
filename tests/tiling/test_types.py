from typing import Any

import pytest
import torch

from tests.tiling.conftest import Blur, blur_model
from waveorder.errors import ParameterError, ShapeError
from waveorder.nn import Sampling, Wiener
from waveorder.tiling import TilePlan, halo_for, reconstruct_series
from waveorder.tiling.run import expect_volumes

SHAPE = (8, 32, 32)
PIXEL = (1.0, 1.0, 1.0)
SAMPLING = Sampling(shape=SHAPE, pixel_size=PIXEL)
PLAN = TilePlan(shape=SHAPE, tile=(8, 16, 16))


def test_halo_for_checks_the_energy():
    for energy in (0.0, 1.0, 1.5):
        with pytest.raises(ParameterError, match=r"energy must be in \(0, 1\)") as caught:
            halo_for(Blur(1.0), SAMPLING, energy=energy)
        assert caught.value.path == "energy"


def _series(**overrides):
    arguments: dict[str, Any] = {
        "model": blur_model(0.0),
        "solver": Wiener(),
        "pixel_size": PIXEL,
        "plan": PLAN,
        "device": "cpu",
        "options": None,
    } | overrides
    reconstruct_series([(torch.zeros(1, *SHAPE), torch.zeros(1, *SHAPE))], **arguments)


def test_reconstruct_series_checks_pixel_size_values():
    with pytest.raises(ParameterError, match="pixel_size"):
        _series(pixel_size=(1.0, -1.0, 1.0))


def test_tile_plan_stores_lists_as_tuples():
    lists: dict[str, Any] = {"shape": list(SHAPE), "tile": [8, 16, 16], "overlap": [0, 2, 2]}  # outside the annotation
    from_lists = TilePlan(**lists)
    from_tuples = TilePlan(shape=SHAPE, tile=(8, 16, 16), overlap=(0, 2, 2))
    assert from_lists == from_tuples
    assert hash(from_lists) == hash(from_tuples)
    assert from_lists.shape == SHAPE


def _volume():
    return torch.zeros(1, *SHAPE), torch.zeros(1, *SHAPE)


def test_expect_volumes_accepts_lists_and_returns_tuples():
    first, second = _volume(), _volume()
    pairs: list[Any] = [list(first), second]  # a list pair: outside the annotation, accepted at run time
    volumes = expect_volumes(pairs, where="test")
    assert isinstance(volumes, tuple)
    assert all(isinstance(pair, tuple) for pair in volumes)
    assert volumes[0][0] is first[0]
    assert volumes[0][1] is first[1]
    # reconstruct_series takes the same list form.
    source, sink, expected = torch.randn(1, *SHAPE), torch.zeros(1, *SHAPE), torch.zeros(1, *SHAPE)
    arguments: dict[str, Any] = {
        "model": blur_model(1.0),
        "solver": Wiener(),
        "pixel_size": PIXEL,
        "plan": PLAN,
        "device": "cpu",
    }
    listed: list[Any] = [[source, sink]]
    reconstruct_series(listed, **arguments)
    reconstruct_series([(source, expected)], **arguments)
    torch.testing.assert_close(sink, expected, rtol=0, atol=0)


def test_expect_volumes_rejects_an_empty_series():
    with pytest.raises(ShapeError, match="volumes is empty"):
        expect_volumes([], where="test")


def test_expect_volumes_rejects_a_shared_sink():
    source, sink = _volume()
    with pytest.raises(ParameterError, match=r"volumes\.1\.sink is the sink of volumes\.0") as caught:
        expect_volumes([(source, sink), (source.clone(), sink)], where="test")
    assert caught.value.path == "volumes.1.sink"
