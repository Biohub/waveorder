"""Read blocks on background threads and hand them to the consumer on its device, in order.

:class:`Prefetcher` is shared by the device loader of :mod:`waveorder.data` and the tile loader of
:mod:`waveorder.tiling`. The caller reads each block into host memory it owns (pinned, for an asynchronous upload).
"""

from collections import deque
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from itertools import islice

import torch
from torch import Tensor

type Stage[K] = Callable[[K], AbstractContextManager[Sequence[Tensor]]]
"""``stage(key)``: a context whose value is the host tensors of ``key``; their memory is the caller's again on exit."""

type _Copied = tuple[tuple[Tensor, ...], torch.cuda.Event | None]


class Prefetcher:
    """Stage blocks on worker threads and hand them to the consumer on ``device``, in the order of their keys.

    A worker enters ``stage(key)``, copies the host tensors to new tensors (on CUDA, on a side stream) and exits the
    context once the copy has read them, so the caller may reuse that memory. The consumer's stream waits for the copy
    on the device; the consumer thread never blocks on the device.

    Parameters
    ----------
    device : torch.device
        Where the delivered tensors are.
    workers : int
        Worker threads.
    ahead : int
        Keys staged beyond the one the consumer holds; ``0`` stages each key when it is requested.
    dtype : torch.dtype, optional
        The dtype of the delivered tensors; ``None`` keeps the staged one.
    """

    def __init__(
        self, device: torch.device, *, workers: int = 1, ahead: int = 2, dtype: torch.dtype | None = None
    ) -> None:
        self.device = torch.device(device)
        self.workers, self.ahead, self.dtype = workers, ahead, dtype
        self._stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

    def __call__[K](self, stage: Stage[K], keys: Iterable[K]) -> Generator[tuple[K, tuple[Tensor, ...]]]:
        """Yield ``(key, tensors)`` for each of ``keys``, in order.

        An error ``stage`` raises is raised here, where its key is due. Closing the generator cancels the staging not
        yet started and waits for the rest.
        """
        pending: deque[tuple[K, Future[_Copied]]] = deque()
        remaining = iter(keys)
        with ThreadPoolExecutor(self.workers, thread_name_prefix="waveorder-staging") as pool:

            def submit() -> bool:  # start staging the next key, if any; whether a key is pending
                for key in islice(remaining, 1):
                    pending.append((key, pool.submit(self._copy, stage, key)))
                return bool(pending)

            for _ in range(self.ahead):
                submit()
            try:
                while pending or submit():
                    key, future = pending.popleft()
                    copied = future.result()
                    if self.ahead:
                        submit()
                    yield key, self._deliver(*copied)
            finally:
                for _, future in pending:
                    future.cancel()

    def _copy[K](self, stage: Stage[K], key: K) -> _Copied:
        """On a worker thread: stage ``key`` and copy what it holds; return the copies and the event they are ready."""
        with stage(key) as host:
            if self._stream is None:
                return tuple(tensor.to(self.dtype or tensor.dtype, copy=True) for tensor in host), None
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                copies = tuple(
                    torch.empty(tensor.shape, dtype=tensor.dtype, device=self.device).copy_(tensor, non_blocking=True)
                    for tensor in host
                )
                ready = torch.cuda.Event()
                ready.record(self._stream)
            ready.synchronize()  # the staged memory is the caller's again once the copy has read it
        return copies, ready

    def _deliver(self, copies: tuple[Tensor, ...], ready: torch.cuda.Event | None) -> tuple[Tensor, ...]:
        """On the consumer thread: order the copies before the work of the current stream, in ``dtype``."""
        if ready is None:
            return copies
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(ready)
        for tensor in copies:  # allocated on the side stream: keep its memory from the next copy while this one reads
            tensor.record_stream(stream)
        return copies if self.dtype is None else tuple(tensor.to(self.dtype) for tensor in copies)
