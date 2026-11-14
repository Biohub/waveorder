import itertools

import pytest
import torch
import zarr

import waveorder.tiling
from tests.tiling.conftest import ArraySink, ArraySource, Blur, blur_model
from waveorder.errors import ConvergenceWarning, ParameterError, PlacementError, ShapeError
from waveorder.nn import FISTA, Gaussian, ImagingModel, Loop, NoPrior, Sampling, Wiener
from waveorder.tiling import TileOptions, TilePlan, estimate_memory, reconstruct_series
from waveorder.tiling.run import OperatorCache

SHAPE = (24, 96, 80)
PIXEL = (1.0, 1.0, 1.0)
MARGIN = 16  # the whole-volume reference wraps around the volume; compare away from its boundary


def _data(channels=1, *, seed=0):
    return torch.randn(channels, *SHAPE, generator=torch.Generator().manual_seed(seed))


def _whole(data, model):
    return Wiener()(data, model, Sampling(shape=SHAPE, pixel_size=PIXEL)).estimate


def _tiled(data, model, plan, *, device="cpu", options=None, sink=None):
    sink = torch.full((1, *SHAPE), float("nan")) if sink is None else sink
    stats = reconstruct_series(
        [(data, sink)], model=model, solver=Wiener(), pixel_size=PIXEL, plan=plan, device=device, options=options
    )
    return sink, stats


def _interior(volume):
    return volume[:, 8:-8, MARGIN:-MARGIN, MARGIN:-MARGIN]


@pytest.mark.parametrize("blend", ["linear", "cosine"])
@pytest.mark.parametrize("overlap", [(0, 0, 0), (2, 4, 6)])
def test_identity_reconstruction_stitches_exactly(blend, overlap):
    data, model = _data(), blur_model(0.0)
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=overlap, blend=blend)
    sink, stats = _tiled(data, model, plan)
    # Pointwise in Fourier space, so only float32 FFT round-off separates the stitched and whole volumes.
    torch.testing.assert_close(sink, _whole(data, model), rtol=1e-5, atol=1e-5)
    assert stats.tiles == len(plan.regions)


def test_blur_stitching_error_falls_with_the_halo_and_is_small_with_enough():
    # Zero data near the lateral boundary keeps the whole-volume reference free of its own wrap-around error there.
    data, model = _data(), blur_model(1.5)
    data[..., :MARGIN, :] = 0
    data[..., -MARGIN:, :] = 0
    data[..., :MARGIN] = 0
    data[..., -MARGIN:] = 0
    reference = _interior(_whole(data, model))
    errors = []
    for halo in (0, 4, 8, 24):
        plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=(2, 4, 4), halo=(halo, halo, halo))
        sink, _ = _tiled(data, model, plan)
        errors.append((_interior(sink) - reference).abs().max().item())
    scale = reference.abs().max().item()
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))
    assert errors[0] > 1e-1 * scale  # without a halo, seams are visible
    assert errors[-1] < 1e-5 * scale


def test_operator_is_bound_once_per_sampling_and_rebound_after_a_parameter_changes():
    data, model = _data(), blur_model(1.0)
    system = model.system
    assert isinstance(system, Blur)
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=(2, 4, 4), halo=(2, 4, 4))
    assert len(plan.regions) > 1
    _tiled(data, model, plan)
    assert system.computed == 1

    cache = OperatorCache(system)
    sampling = Sampling(shape=plan.regions[0].read_shape, pixel_size=PIXEL)
    first = cache(sampling)
    assert cache(sampling) is first
    with torch.no_grad():
        system.get_buffer("sigma").fill_(2.0)
    second = cache(sampling)
    assert second is not first
    assert cache.binds == 2
    impulse = torch.zeros(1, *sampling.shape)
    impulse[0, 0, 0, 0] = 1
    torch.testing.assert_close(second.apply(impulse), Blur(2.0).bind(sampling).apply(impulse))


def test_array_like_source_and_sink_with_chunks_and_parallel_writes():
    data, model = _data(), blur_model(1.0)
    chunks = (6, 16, 16)
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=(2, 4, 4), halo=(2, 8, 8), chunks=chunks)
    source = ArraySource(data[0], chunks=chunks)
    sink = ArraySink(SHAPE, chunks=(12, 32, 16))
    _, stats = _tiled(source, model, plan, sink=sink, options=TileOptions(write_workers=3, read_workers=3))
    reference, _ = _tiled(data, model, plan)
    torch.testing.assert_close(sink.values[None], reference, rtol=0, atol=0)
    # One write per row of cores (same z and y grid position, whole x extent), each of whole sink chunks, covering
    # every voxel exactly once.
    grid_z, grid_y, _ = plan.grid
    assert len(sink.writes) == grid_z * grid_y
    written = torch.zeros(SHAPE, dtype=torch.int64)
    for key in sink.writes:
        written[key] += 1
        for s, n, c in zip(key, SHAPE, sink.chunks, strict=True):
            assert isinstance(s, slice)
            assert s.start % c == 0
            assert s.stop % c == 0 or s.stop == n
    assert torch.all(written == 1)
    assert stats.bytes_written == 4 * data.numel()


def test_parallel_writes_into_partial_chunks_are_refused():
    plan = TilePlan(shape=SHAPE, tile=(12, 30, 32))
    sink = ArraySink(SHAPE, chunks=(12, 16, 16))
    with pytest.raises(ParameterError, match=r"plan\.tile\.y") as caught:
        _tiled(_data(), blur_model(0.0), plan, sink=sink, options=TileOptions(write_workers=2))
    assert caught.value.path == "options.write_workers"


def test_memory_budget_refuses_plans_that_do_not_fit():
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=(2, 4, 4), halo=(2, 4, 4))
    budget = 1 << 16
    options = TileOptions(memory_budget=budget)
    needed = estimate_memory(plan, channels=1, fields=1, source_itemsize=4, options=options).host
    assert needed > budget
    with pytest.raises(ParameterError, match=r"options\.memory_budget") as caught:
        _tiled(_data(), blur_model(0.0), plan, options=options)
    assert caught.value.path == "options.memory_budget"
    _tiled(_data(), blur_model(0.0), plan, options=TileOptions(memory_budget=needed))


def test_memory_estimate_counts_one_staging_slot_per_pinned_slot():
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), halo=(2, 4, 4))
    read = 2 * _numel(plan.regions[0].read_shape)  # uint16
    small = estimate_memory(plan, channels=1, fields=1, source_itemsize=2, options=TileOptions(pinned_slots=1))
    large = estimate_memory(plan, channels=1, fields=1, source_itemsize=2, options=TileOptions(pinned_slots=3))
    assert large.staging - small.staging == 2 * read
    # The estimate reports the shape every tile reads. Along z the halo is clipped at the volume edge: 12 + 2 = 14
    # = 2 * 7; laterally 32 + 2 * 4 = 40 = 2^3 * 5 (both already FFT-friendly).
    assert small.read_shape == plan.read_shape == (14, 40, 40)


def test_stats_count_nonfinite_tiles_on_the_device_and_unconverged_tiles_from_the_status():
    data = _data()
    data[0, 0, 0, 0] = float("nan")  # inside the read region of tile 0 only
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32))
    _, stats = _tiled(data, blur_model(1.0), plan)
    assert stats.nonfinite == 1
    assert stats.unconverged == 0  # Wiener reports "direct": no criterion requested, so never a failure

    solver = FISTA(loop=Loop(2, tolerance=1e-12))  # a criterion it cannot meet in two steps
    sink = torch.full((1, *SHAPE), float("nan"))
    model = ImagingModel(Blur(1.0), NoPrior(), Gaussian(0.1))
    with pytest.warns(ConvergenceWarning, match="without meeting tolerance"):
        stats = reconstruct_series(
            [(_data(), sink)], model=model, solver=solver, pixel_size=PIXEL, plan=plan, device="cpu"
        )
    assert stats.unconverged == stats.tiles == len(plan.regions)
    assert stats.nonfinite == 0


def _numel(shape):
    total = 1
    for n in shape:
        total *= n
    return total


class FailingSink(ArraySink):
    def __setitem__(self, key, value):
        raise OSError("sink is read-only")


def test_a_writer_error_is_raised():
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32))
    with pytest.raises(OSError, match="read-only"):
        _tiled(_data(), blur_model(0.0), plan, sink=FailingSink(SHAPE))


def test_mismatched_sink_and_device_are_rejected():
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32))
    with pytest.raises(ShapeError, match="sink"):
        _tiled(_data(), blur_model(0.0), plan, sink=torch.zeros(2, *SHAPE))
    with pytest.raises(ShapeError, match="source"):
        _tiled(_data(channels=2), blur_model(0.0), plan)
    if torch.cuda.is_available():
        with pytest.raises(PlacementError, match="device"):
            _tiled(_data(), blur_model(0.0), plan, device="cuda")
        # "cuda" and "cuda:<current>" are the same device.
        index = torch.cuda.current_device()
        sink, _ = _tiled(_data(), blur_model(0.0).to("cuda"), plan, device=f"cuda:{index}")
        assert not torch.isnan(sink).any()


def test_an_empty_series_is_rejected():
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32))
    with pytest.raises(ShapeError, match="volumes is empty"):
        reconstruct_series([], model=blur_model(0.0), solver=Wiener(), pixel_size=PIXEL, plan=plan, device="cpu")


def test_public_surface_is_the_plan_options_stats_and_entry_points():
    assert set(waveorder.tiling.__all__) == {
        "TilePlan",
        "TileOptions",
        "TileStats",
        "estimate_memory",
        "halo_for",
        "reconstruct_series",
        "recover_zernikes",
    }


def test_zarr_arrays_round_trip(tmp_path):
    data, model = _data(), blur_model(1.0)
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=(2, 4, 4), halo=(2, 8, 8), chunks=(6, 16, 16))
    source = zarr.open_array(tmp_path / "source.zarr", mode="w", shape=SHAPE, chunks=(6, 16, 16), dtype="uint16")
    source[...] = (data[0] * 100 + 1000).round().to(torch.int32).numpy()
    sink = zarr.open_array(tmp_path / "sink.zarr", mode="w", shape=SHAPE, chunks=(12, 32, 32), dtype="float32")
    reconstruct_series(
        [(source, sink)],
        model=model,
        solver=Wiener(),
        pixel_size=PIXEL,
        plan=plan,
        device="cpu",
        options=TileOptions(write_workers=2),
    )
    expected, _ = _tiled(torch.as_tensor(source[...]).to(torch.float32)[None], model, plan)
    torch.testing.assert_close(torch.as_tensor(sink[...])[None], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    data = _data()
    plan = TilePlan(shape=SHAPE, tile=(12, 32, 32), overlap=(2, 4, 4), halo=(2, 8, 8))
    source = (data * 100 + 1000).round().to(torch.uint16)
    model = blur_model(1.0).cuda()
    sink, stats = _tiled(source, model, plan, device="cuda", options=TileOptions())
    expected, _ = _tiled(source.to(torch.float32), blur_model(1.0), plan)
    torch.testing.assert_close(sink, expected, rtol=1e-5, atol=1e-3)
    assert stats.gpu_s > 0
    torch.cuda.empty_cache()
