import threading
import time

import pytest
import torch

from tests.tiling.conftest import ArraySource
from waveorder.errors import ParameterError, ShapeError
from waveorder.tiling import TileOptions, TilePlan
from waveorder.tiling.merge import StagingRing
from waveorder.tiling.run import TileLoader

PIXEL = (0.5, 0.2, 0.2)


def _source(shape=(2, 12, 40, 36), *, dtype=torch.int32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 4000, shape, generator=generator, dtype=torch.int32).to(dtype)


def _plan(shape=(12, 40, 36)):
    return TilePlan(shape=shape, tile=(6, 16, 16), overlap=(1, 2, 2), halo=(1, 3, 3))


def _loader(source, *, plan=None, device="cpu", dtype=torch.float32, **options):
    return TileLoader(
        [source],
        _plan() if plan is None else plan,
        pixel_size=PIXEL,
        device=torch.device(device),
        dtype=dtype,
        options=TileOptions(**options),
    )


def _numel(shape):
    total = 1
    for n in shape:
        total *= n
    return total


@pytest.mark.parametrize("dtype", [torch.uint16, torch.int32, torch.float32])
@pytest.mark.parametrize("prefetch", [0, 1, 3])
def test_tiles_are_the_source_read_regions_in_the_requested_dtype(dtype, prefetch):
    values = _source(dtype=dtype)
    plan = _plan()
    loader = _loader(ArraySource(values), plan=plan, dtype=torch.float64, prefetch=prefetch, read_workers=2)
    seen = []
    for volume, (data, sampling, region) in loader:
        box = tuple(slice(a, b) for a, b in region.read)
        assert volume == 0
        assert data.dtype == torch.float64
        torch.testing.assert_close(data, values[(slice(None), *box)].to(torch.float64), rtol=0, atol=0)
        assert sampling.shape == region.read_shape == plan.read_shape
        assert sampling.pixel_size == PIXEL
        seen.append(region.index)
    assert seen == list(range(len(plan.regions)))
    assert loader.stats.bytes_read == sum(2 * dtype.itemsize * _numel(r.read_shape) for r in plan.regions)


def test_tiles_do_not_alias_a_tensor_source():
    values = _source(dtype=torch.float32)
    original = values.clone()
    for _, (data, _, _) in _loader(values):
        data.zero_()
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_single_channel_sources_get_a_channel_axis():
    _, (data, _, region) = next(iter(_loader(_source(dtype=torch.float32)[0])))
    assert data.shape == (1, *region.read_shape)


class SlowSource(ArraySource):
    """Counts reads in flight and how far ahead of the consumer reading runs."""

    def __init__(self, values, *, delay: float) -> None:
        super().__init__(values)
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.peak_active = 0

    def __getitem__(self, key):
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(self.delay)
            return super().__getitem__(key)
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.parametrize(("prefetch", "workers", "slots"), [(1, 1, 1), (4, 3, 2), (6, 4, 1)])
def test_read_ahead_and_staging_ring_stay_bounded(prefetch, workers, slots):
    source = SlowSource(_source(), delay=0.005)
    plan = _plan()
    loader = _loader(source, plan=plan, prefetch=prefetch, read_workers=workers, pinned_slots=slots)
    ahead = []
    for consumed, _ in enumerate(loader):
        time.sleep(0.01)  # a slow consumer lets the readers run as far ahead as they are allowed
        ahead.append(len(source.reads) - (consumed + 1))
    assert max(ahead) <= prefetch
    assert max(ahead) == min(prefetch, len(plan.regions) - 1)
    assert source.peak_active <= min(prefetch, workers)
    assert loader.ring.allocated <= slots
    assert loader.ring.peak_in_use <= slots


def test_staging_ring_blocks_until_a_slot_is_released():
    ring = StagingRing(1, pin=False)
    buffer, _ = ring.acquire((4,), torch.float32)
    acquired = threading.Event()

    def take():
        second, _ = ring.acquire((4,), torch.float32)
        acquired.set()
        ring.release(second)

    thread = threading.Thread(target=take)
    thread.start()
    assert not acquired.wait(0.05)
    ring.release(buffer)
    assert acquired.wait(5)
    thread.join()
    assert ring.allocated == 1


class FailingSource(ArraySource):
    def __getitem__(self, key):
        if len(self.reads) == 3:
            raise OSError("disk went away")
        return super().__getitem__(key)


def test_a_reader_error_is_raised_from_the_iterator():
    loader = _loader(FailingSource(_source()), prefetch=2, read_workers=1)
    with pytest.raises(OSError, match="disk went away"):
        for _ in loader:
            pass


def test_closing_early_stops_the_readers():
    source = SlowSource(_source(), delay=0.01)
    tiles = iter(_loader(source, prefetch=3, read_workers=2))
    next(tiles)
    tiles.close()
    assert source.active == 0
    assert len(source.reads) <= 1 + 3


def test_mismatched_source_is_rejected():
    with pytest.raises(ParameterError, match=r"plan\.shape"):
        _loader(_source(shape=(2, 12, 40, 35)))


class ComplexSource(ArraySource):
    """Declares a real dtype but returns complex blocks, as a misdeclared backend would."""

    def __init__(self, values) -> None:
        super().__init__(values)
        self.dtype = torch.float32

    def __getitem__(self, key):
        return torch.complex(self.values[key].float(), self.values[key].float()).numpy()


def test_complex_sources_are_rejected():
    with pytest.raises(ShapeError, match="real"):
        _loader(_source(dtype=torch.float32).to(torch.complex64))
    with pytest.raises(ShapeError, match="real"):
        for _ in _loader(ComplexSource(_source(dtype=torch.float32))):
            pass


@pytest.mark.parametrize(("name", "value"), [("prefetch", -1), ("read_workers", 0), ("memory_budget", -5)])
def test_invalid_options_name_the_field(name, value):
    # The path is the constructor's argument; reconstruct_series prefixes "options." for errors it raises itself.
    with pytest.raises(ParameterError, match=f"TileOptions {name}") as caught:
        TileOptions(**{name: value})
    assert caught.value.path == name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tiles_match_the_source():
    values = _source(dtype=torch.uint16)
    loader = _loader(ArraySource(values), device="cuda", prefetch=2, read_workers=2)
    for _, (data, _, region) in loader:
        box = tuple(slice(a, b) for a, b in region.read)
        assert data.is_cuda
        torch.testing.assert_close(data.cpu(), values[(slice(None), *box)].float(), rtol=0, atol=0)
    assert loader.ring.pin
    assert loader.ring.allocated <= 2
