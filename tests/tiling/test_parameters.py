import pytest
import torch

from tests.tiling.conftest import Blur
from waveorder.errors import ParameterError, PlacementError
from waveorder.nn import Gaussian, ImagingModel, Tikhonov, Wiener
from waveorder.tiling import TileOptions, TilePlan, reconstruct_series
from waveorder.tiling.run import stack_tiles

SHAPE = (8, 32, 24)
PIXEL = (0.5, 0.25, 0.25)
# No overlap: each core is written by its own tile alone, so a tile's values decide its core exactly.
PLAN = TilePlan(shape=SHAPE, tile=(8, 16, 12), overlap=(0, 0, 0), halo=(0, 4, 4))
SIGMAS = torch.tensor([0.2, 0.5, 0.8, 1.1])


def _model(sigma):
    return ImagingModel(Blur(sigma, learnable=True), Tikhonov(1e-2), Gaussian(0.1))


def _data(seed=0):
    return torch.randn(1, *SHAPE, generator=torch.Generator().manual_seed(seed))


def _run(data, model, **kwargs):
    sink = torch.full((1, *SHAPE), float("nan"))
    stats = reconstruct_series(
        [(data, sink)], model=model, solver=Wiener(), pixel_size=PIXEL, plan=PLAN, device="cpu", **kwargs
    )
    return sink, stats


def test_each_tile_reconstructs_with_its_own_values():
    assert len(PLAN.regions) == len(SIGMAS)
    data = _data()
    sink, stats = _run(data, _model(0.3), parameters={"system.sigma": SIGMAS})
    assert stats.tiles == len(SIGMAS)
    for region, sigma in zip(PLAN.regions, SIGMAS.tolist(), strict=True):
        expected, _ = _run(data, _model(sigma))  # the whole volume at this tile's value
        box = (slice(None), *(slice(a, b) for a, b in region.core))
        torch.testing.assert_close(sink[box], expected[box], rtol=1e-6, atol=1e-6)
    # The tiles differ: a single value would not reproduce every core.
    plain, _ = _run(data, _model(0.3))
    assert not torch.allclose(sink, plain, atol=1e-3)


def test_a_uniform_table_equals_the_plain_run_and_leaves_the_model_unchanged():
    data, model = _data(1), _model(0.7)
    plain, _ = _run(data, model)
    table, _ = _run(data, model, parameters={"system.sigma": torch.full((len(PLAN.regions),), 0.7)})
    torch.testing.assert_close(table, plain, rtol=1e-6, atol=1e-6)
    assert model.system.get_parameter("sigma").item() == pytest.approx(0.7)


@pytest.mark.parametrize(("budget", "binds"), [(1 << 30, 4), (0, 8)])
def test_per_tile_operators_are_kept_across_volumes_within_the_budget(budget, binds):
    model = _model(0.3)
    volumes = [(_data(seed), torch.zeros(1, *SHAPE)) for seed in (2, 3)]
    reconstruct_series(
        volumes,
        model=model,
        solver=Wiener(),
        pixel_size=PIXEL,
        plan=PLAN,
        device="cpu",
        options=TileOptions(operator_cache_bytes=budget),
        parameters={"system.sigma": SIGMAS},
    )
    assert model.system.computed == binds  # one transfer per tile per bind


@pytest.mark.parametrize(
    ("table", "error", "path"),
    [
        ({"system.radius": SIGMAS}, ParameterError, "parameters.system.radius"),
        ({"system.sigma": SIGMAS[:3]}, ParameterError, "parameters.system.sigma"),
        ({"system.sigma": SIGMAS[:, None]}, ParameterError, "parameters.system.sigma"),
        ({"system.sigma": SIGMAS.double()}, PlacementError, None),
    ],
)
def test_bad_tables_name_the_entry(table, error, path):
    with pytest.raises(error) as caught:
        _run(_data(), _model(0.3), parameters=table)
    if path is not None:
        assert caught.value.path == path


def test_stack_tiles_rows_follow_the_region_order_and_share_one_sampling():
    data = _data(4)
    stack, sampling = stack_tiles(data, PLAN, pixel_size=PIXEL, device="cpu", dtype=torch.float32)
    assert stack.shape == (len(PLAN.regions), 1, *PLAN.read_shape)
    assert sampling.shape == PLAN.read_shape
    assert sampling.pixel_size == PIXEL
    for region in PLAN.regions:
        box = (slice(None), *(slice(a, b) for a, b in region.read))
        torch.testing.assert_close(stack[region.index], data[box], rtol=0, atol=0)
