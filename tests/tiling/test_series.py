import math
import threading
import warnings
from collections import Counter

import pytest
import torch

from tests.tiling.conftest import ArraySource, Blur, blur_model
from waveorder.errors import ParameterError
from waveorder.nn import ImagingModel, Sampling, Wiener
from waveorder.tiling import TileOptions, TilePlan, estimate_memory, halo_for, reconstruct_series
from waveorder.tiling.plan import chunk_keys
from waveorder.tiling.run import ChunkCache, OperatorCache, TileLoader

SHAPE = (16, 64, 48)
PIXEL = (1.0, 1.0, 1.0)
CHUNKS = (8, 16, 16)
PLAN = TilePlan(shape=SHAPE, tile=(16, 8, 12), overlap=(0, 4, 6), halo=(0, 2, 3))  # tiles smaller than chunks


def _data(seed=0, shape=SHAPE):
    return torch.randn(1, *shape, generator=torch.Generator().manual_seed(seed))


def _decoded_chunks(source: ArraySource) -> Counter:
    """How often each chunk was decoded: every source read decodes all chunks it touches in full."""
    counts = Counter()
    for key in source.reads:
        _, (z0, z1), (y0, y1), (x0, x1) = ((s.start, s.stop) for s in key if isinstance(s, slice))
        counts.update(chunk_keys(0, ((z0, z1), (y0, y1), (x0, x1)), CHUNKS))
    return counts


def _reconstruct(source, sink, model, plan, **options):
    return reconstruct_series(
        [(source, sink)],
        model=model,
        solver=Wiener(),
        pixel_size=PIXEL,
        plan=plan,
        device="cpu",
        options=TileOptions(**options),
    )


def _run(source, model, plan=PLAN, **options):
    sink = torch.full((1, *SHAPE), float("nan"))
    return sink, _reconstruct(source, sink, model, plan, **options)


@pytest.mark.parametrize(("workers", "prefetch"), [(1, 0), (4, 6)])
def test_chunk_cache_decodes_every_chunk_once_and_matches_direct_reads(workers, prefetch):
    data, model = _data(), blur_model(1.0)
    cached = ArraySource(data, chunks=(1, *CHUNKS))
    direct = ArraySource(data, chunks=(1, *CHUNKS))
    sink, stats = _run(cached, model, read_workers=workers, prefetch=prefetch)
    expected, direct_stats = _run(direct, model, read_workers=workers, prefetch=prefetch, chunk_cache=False)
    torch.testing.assert_close(sink, expected, rtol=0, atol=0)
    counts = _decoded_chunks(cached)
    touched = math.prod(math.ceil(n / c) for n, c in zip(SHAPE, CHUNKS, strict=True))
    assert set(counts.values()) == {1}
    assert len(counts) == touched
    assert stats.bytes_decoded == data.numel() * data.element_size()
    # Without the cache tiles smaller than a chunk decode each chunk many times.
    assert max(_decoded_chunks(direct).values()) > 4
    assert direct_stats.bytes_decoded == direct_stats.bytes_read > stats.bytes_decoded


def test_chunk_cache_peak_stays_within_the_estimate():
    data, model = _data(), blur_model(1.0)
    source = ArraySource(data, chunks=(1, *CHUNKS))
    options = TileOptions(read_workers=3, prefetch=2)
    _, stats = _run(source, model, read_workers=3, prefetch=2)
    estimate = estimate_memory(PLAN, channels=1, fields=1, source_itemsize=4, options=options, chunks=CHUNKS)
    assert 0 < stats.peak_cache_bytes <= estimate.decoded
    assert estimate.decoded < data.numel() * data.element_size()  # chunks are freed after their last tile


class SlowChunks(ArraySource):
    def __getitem__(self, key):
        threading.Event().wait(0.01)
        return super().__getitem__(key)


def test_chunk_cache_shared_by_concurrent_readers_decodes_once():
    data = _data()
    source = SlowChunks(data, chunks=(1, *CHUNKS))
    regions = [region.read for region in PLAN.regions]
    cache = ChunkCache([source], chunks=CHUNKS, reads=[(0, r) for r in regions * 3], channels=True)
    results = [None] * (3 * len(regions))

    def read(index):
        results[index] = cache.read(0, regions[index % len(regions)])

    threads = [threading.Thread(target=read, args=(i,)) for i in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for index, block in enumerate(results):
        box = tuple(slice(a, b) for a, b in regions[index % len(regions)])
        torch.testing.assert_close(block, data[(slice(None), *box)], rtol=0, atol=0)
    assert set(_decoded_chunks(source).values()) == {1}
    assert cache.live_bytes == 0  # every chunk was freed after its last use


def test_series_matches_volume_by_volume_and_reads_ahead_across_volumes():
    model = blur_model(1.0)
    sources = [ArraySource(_data(seed), chunks=(1, *CHUNKS)) for seed in range(3)]
    sinks = [torch.full((1, *SHAPE), float("nan")) for _ in sources]
    stats = reconstruct_series(
        list(zip(sources, sinks, strict=True)),
        model=model,
        solver=Wiener(),
        pixel_size=PIXEL,
        plan=PLAN,
        device="cpu",
        options=TileOptions(prefetch=3, read_workers=2, write_workers=2),
    )
    for seed, sink in enumerate(sinks):
        expected, _ = _run(_data(seed), model)
        torch.testing.assert_close(sink, expected, rtol=0, atol=0)
    assert stats.tiles == 3 * len(PLAN.regions)

    # When the last tile of volume 0 reaches the consumer, volume 1 is already being read.
    loader = TileLoader(
        [ArraySource(_data(seed), chunks=(1, *CHUNKS)) for seed in range(2)],
        PLAN,
        pixel_size=PIXEL,
        device=torch.device("cpu"),
        dtype=torch.float32,
        options=TileOptions(prefetch=3, read_workers=2),
    )
    last = len(PLAN.regions) - 1
    for volume, tile in loader:
        if volume == 0 and tile.region.index == last:
            threading.Event().wait(0.05)
            assert loader.sources[1].reads  # ty: ignore[missing-attribute]
            break


def test_series_rejects_volumes_of_different_shapes():
    with pytest.raises(ParameterError, match=r"plan\.shape"):
        TileLoader(
            [_data(), _data(shape=(16, 64, 40))],
            PLAN,
            pixel_size=PIXEL,
            device=torch.device("cpu"),
            dtype=torch.float32,
            options=TileOptions(),
        )


def test_float64_output_is_preserved_through_blending():
    data = _data().double()
    model = blur_model(0.0).double()
    plan = TilePlan(shape=SHAPE, tile=(8, 16, 16), overlap=(2, 4, 4), blend="cosine")
    sink64 = torch.full((1, *SHAPE), float("nan"), dtype=torch.float64)
    _reconstruct(data, sink64, model, plan, output_dtype=torch.float64)
    whole = Wiener()(data, model, Sampling(shape=SHAPE, pixel_size=PIXEL)).estimate
    torch.testing.assert_close(sink64, whole, rtol=1e-12, atol=1e-12)
    # A float32 accumulation of the same estimates is only good to float32 precision.
    sink32 = torch.full((1, *SHAPE), float("nan"), dtype=torch.float64)
    _reconstruct(data, sink32, model, plan)
    assert (sink32 - whole).abs().max() > 1e-9


def test_float16_output_halves_the_bytes_written():
    data, model = _data(), blur_model(1.0)
    sink16, stats16 = _run(data, model, output_dtype=torch.float16)
    sink32, stats32 = _run(data, model)
    torch.testing.assert_close(sink16, sink32.half().float(), rtol=0, atol=0)
    assert 2 * stats16.bytes_written == stats32.bytes_written


def test_invalid_output_dtype_names_the_field():
    with pytest.raises(ParameterError, match=r"TileOptions output_dtype") as caught:
        TileOptions(output_dtype=torch.int32)
    assert caught.value.path == "output_dtype"


def test_halo_for_the_psf_of_a_gaussian_blur_is_about_three_sigma():
    sigma = 1.5
    halo = halo_for(Blur(sigma), Sampling(shape=(32, 64, 64), pixel_size=PIXEL), energy=1 - 1e-3)
    # A Gaussian keeps 1 - 1e-3 of its mass within 3.29 sigma.
    assert all(math.floor(3.29 * sigma) - 1 <= h <= math.ceil(3.29 * sigma) + 1 for h in halo)
    assert halo_for(Blur(sigma), Sampling(shape=(32, 64, 64), pixel_size=PIXEL)) == halo  # energy 1 - 1e-3 default


@pytest.mark.parametrize("tail", [1e-3, 1e-5])
def test_half_the_halo_of_halo_for_stitches_with_a_larger_error(tail):
    shape, margin = (24, 96, 80), 16
    data = torch.randn(1, *shape, generator=torch.Generator().manual_seed(0))
    for axis in (2, 3):  # zero data near the lateral boundary: the whole-volume reference wraps there
        data.narrow(axis, 0, margin).zero_()
        data.narrow(axis, shape[axis - 1] - margin, margin).zero_()
    model, solver = blur_model(1.5), Wiener()
    sampling = Sampling(shape=shape, pixel_size=PIXEL)
    whole = solver(data, model, sampling).estimate[:, 8:-8, margin:-margin, margin:-margin]
    halo = halo_for(model, sampling, energy=1 - tail, solver=solver)

    def error(h):
        plan = TilePlan(shape=shape, tile=(12, 32, 32), overlap=(2, 4, 4), halo=h)
        sink = torch.full((1, *shape), float("nan"))
        _reconstruct(data, sink, model, plan)
        return (sink[:, 8:-8, margin:-margin, margin:-margin] - whole).abs().max().item()

    assert error(tuple(h // 2 for h in halo)) > error(halo)


def test_a_smaller_tail_of_halo_for_at_the_tiles_sampling_stitches_with_a_smaller_error():
    # halo_for measures the kernel of one tile's read region; that region grows with the halo, so iterate.
    shape, margin = (24, 96, 80), 16
    data = torch.randn(1, *shape, generator=torch.Generator().manual_seed(0))
    for axis in (2, 3):
        data.narrow(axis, 0, margin).zero_()
        data.narrow(axis, shape[axis - 1] - margin, margin).zero_()
    model, solver = blur_model(1.5), Wiener()
    whole = solver(data, model, Sampling(shape=shape, pixel_size=PIXEL)).estimate
    errors = []
    for tail in (1e-2, 1e-3, 1e-5):
        halo, plan = (0, 0, 0), None
        for _ in range(6):
            plan = TilePlan(shape=shape, tile=(12, 32, 32), overlap=(2, 4, 4), halo=halo)
            tile = Sampling(shape=plan.read_shape, pixel_size=PIXEL)
            if (halo := halo_for(model, tile, energy=1 - tail, solver=solver)) == plan.halo:
                break
        assert plan is not None
        assert plan.halo == halo
        sink = torch.full((1, *shape), float("nan"))
        _reconstruct(data, sink, model, plan)
        error = (sink - whole)[:, 8:-8, margin:-margin, margin:-margin].abs().max().item()
        errors.append(error)
    assert errors[0] > errors[1] > errors[2]


def test_halo_for_rejects_a_solver_without_a_model():
    with pytest.raises(ParameterError, match="ImagingModel"):
        halo_for(Blur(1.0), Sampling(shape=SHAPE, pixel_size=PIXEL), solver=Wiener())


def test_operator_cache_binds_afresh_when_the_operator_would_carry_gradients():
    system = Blur(1.0, learnable=True)
    cache = OperatorCache(system)
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL)
    first, second = cache(sampling), cache(sampling)
    assert first is not second
    assert first.transfer.requires_grad  # ty: ignore[missing-attribute]
    with torch.no_grad():
        assert cache(sampling) is cache(sampling)
    assert cache.binds == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_series_with_chunk_cache_matches_cpu():
    model = blur_model(1.0)
    sources = [
        ArraySource((_data(seed) * 100 + 1000).round().to(torch.int16), chunks=(1, *CHUNKS)) for seed in range(2)
    ]
    expected = [_run(source.values.float(), model)[0] for source in sources]
    sinks = [torch.full((1, *SHAPE), float("nan")) for _ in sources]
    stats = reconstruct_series(
        list(zip(sources, sinks, strict=True)),
        model=model.cuda(),
        solver=Wiener(),
        pixel_size=PIXEL,
        plan=PLAN,
        device="cuda",
        options=TileOptions(prefetch=3, read_workers=3),
    )
    for sink, want in zip(sinks, expected, strict=True):
        torch.testing.assert_close(sink, want, rtol=1e-5, atol=1e-3)
    assert stats.bytes_decoded == 2 * sources[0].values.numel() * 2
    torch.cuda.empty_cache()


def _synchronizations(plan: TilePlan, model: ImagingModel) -> int:
    """Count the host synchronizations torch reports while reconstructing one volume with ``plan`` on CUDA."""
    data = _data()
    sink = torch.full((1, *SHAPE), float("nan"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            reconstruct_series([(data, sink)], model=model, solver=Wiener(), pixel_size=PIXEL, plan=plan, device="cuda")
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert not sink.isnan().any()
    return sum("synchronizing CUDA operation" in str(warning.message) for warning in caught)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_series_does_not_synchronize_per_tile():
    model = blur_model(1.0).cuda()
    few = TilePlan(shape=SHAPE, tile=(16, 32, 24), overlap=(0, 4, 6))
    many = TilePlan(shape=SHAPE, tile=(16, 8, 12), overlap=(0, 4, 6))
    assert len(many.regions) >= 4 * len(few.regions)
    _synchronizations(few, model)  # warm up: transfer function, cuFFT plans, pinned memory
    # One read-back per run (the non-finite count, after the last tile), whatever the number of tiles.
    assert _synchronizations(many, model) == _synchronizations(few, model) == 1
    torch.cuda.empty_cache()
