"""The prefetcher both loaders share: order, bounds, errors, shutdown, the CPU path and uploads that do not block."""

import random
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

import pytest
import torch

from waveorder.io.staging import Prefetcher

CPU = torch.device("cpu")


class Stages:
    """A stage that records how many keys are held at once and which were started, with an optional delay."""

    def __init__(self, *, delay: float = 0.0, jitter: float = 0.0) -> None:
        self.delay, self.jitter = delay, jitter
        self.lock = threading.Lock()
        self.started: list[int] = []
        self.held = 0
        self.peak_held = 0
        self.exited = 0

    @contextmanager
    def __call__(self, key: int) -> Generator[tuple[torch.Tensor, ...]]:
        with self.lock:
            self.started.append(key)
            self.held += 1
            self.peak_held = max(self.peak_held, self.held)
        try:
            time.sleep(self.delay + random.uniform(0, self.jitter))
            yield (torch.full((3, 4), key, dtype=torch.int16), torch.tensor([key, -key], dtype=torch.int64))
        finally:
            with self.lock:
                self.held -= 1
                self.exited += 1


@pytest.mark.parametrize(("workers", "ahead"), [(1, 0), (1, 2), (4, 3), (3, 8)])
def test_keys_come_back_in_order_with_their_tensors(workers, ahead):
    stages = Stages(jitter=0.004)
    keys = list(range(20))
    delivered = list(Prefetcher(CPU, workers=workers, ahead=ahead)(stages, keys))
    assert [key for key, _ in delivered] == keys
    for key, (block, pair) in delivered:
        assert block.dtype == torch.int16
        assert torch.equal(block, torch.full((3, 4), key, dtype=torch.int16))
        assert pair.tolist() == [key, -key]
    assert stages.exited == len(keys)


@pytest.mark.parametrize(("workers", "ahead"), [(1, 0), (1, 1), (2, 4), (4, 2)])
def test_staging_stays_within_ahead_keys_and_workers_threads(workers, ahead):
    stages = Stages(delay=0.003)
    ahead_of_consumer = []
    for consumed, _ in enumerate(Prefetcher(CPU, workers=workers, ahead=ahead)(stages, range(12))):
        time.sleep(0.01)  # a slow consumer lets staging run as far ahead as it may
        ahead_of_consumer.append(len(stages.started) - (consumed + 1))
    assert max(ahead_of_consumer) == min(ahead, 11)  # reaches the bound and never passes it
    assert stages.peak_held <= min(workers, max(ahead, 1))  # host memory held at once


def test_a_stage_error_is_raised_in_the_consumer_where_its_key_is_due():
    error = OSError("disk went away")

    @contextmanager
    def stage(key: int) -> Generator[tuple[torch.Tensor, ...]]:
        if key == 3:
            raise error
        yield (torch.tensor([key]),)

    delivered = Prefetcher(CPU, workers=2, ahead=2)(stage, range(8))
    assert [next(delivered)[0] for _ in range(3)] == [0, 1, 2]
    with pytest.raises(OSError, match="disk went away") as caught:
        next(delivered)
    assert caught.value is error


def test_closing_with_pending_work_cancels_it_and_waits_for_the_running_stages():
    stages = Stages(delay=0.02)
    delivered = Prefetcher(CPU, workers=2, ahead=3)(stages, range(50))
    next(delivered)
    delivered.close()
    assert stages.held == 0  # every running stage finished and gave its memory back
    assert stages.exited == len(stages.started) <= 1 + 3
    assert list(delivered) == []


def test_the_cpu_path_copies_out_of_the_staged_memory_without_cuda(monkeypatch):
    def no_cuda(*_: object, **__: object) -> None:
        raise AssertionError("the CPU path touched CUDA")

    monkeypatch.setattr(torch.cuda, "Stream", no_cuda)
    monkeypatch.setattr(torch.cuda, "Event", no_cuda)
    slot = torch.zeros(5, dtype=torch.uint16)

    @contextmanager
    def stage(key: int) -> Generator[tuple[torch.Tensor, ...]]:
        slot.fill_(key)
        yield (slot,)

    delivered = [(key, data) for key, (data,) in Prefetcher(CPU, ahead=2, dtype=torch.float32)(stage, range(4))]
    for key, data in delivered:  # the one slot was refilled after each copy; the copies keep their values
        assert data.dtype == torch.float32
        assert data.data_ptr() != slot.data_ptr()
        assert torch.equal(data, torch.full((5,), float(key)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_uploads_from_one_pinned_slot_never_make_the_consumer_wait_for_the_device():
    device = torch.device("cuda", torch.cuda.current_device())
    slot = torch.empty((64, 64), dtype=torch.uint16, pin_memory=True)
    lock = threading.Lock()

    @contextmanager
    def stage(key: int) -> Generator[tuple[torch.Tensor, ...]]:
        with lock:  # one pinned slot: it is filled again only once the previous upload has read it
            slot.fill_(key)
            yield (slot,)

    prefetch = Prefetcher(device, workers=2, ahead=2, dtype=torch.float32)
    list(prefetch(stage, range(6)))  # warm the allocator, so the run below allocates nothing new
    torch.cuda.synchronize(device)
    consumer = torch.cuda.current_stream(device)
    torch.cuda._sleep(1_000_000_000)  # keep the consumer's stream busy for a while  # noqa: SLF001
    delivered = [(key, data) for key, (data,) in prefetch(stage, range(6))]
    assert not consumer.query()  # every tile arrived while the consumer's stream still sleeps: no host wait on it
    for key, data in delivered:
        assert data.device == device
        assert data.dtype == torch.float32
        torch.testing.assert_close(data.cpu(), torch.full((64, 64), float(key)), rtol=0, atol=0)
