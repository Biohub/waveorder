"""What a spawned process runs in the data tests: a stand-in loader worker that measures the memory it holds."""

import tracemalloc
from multiprocessing.connection import Connection

import pyarrow as pa

from waveorder.data.loader import _Stream


def private_bytes() -> int:
    """Return this process's private resident memory (clean and dirty) in bytes, from ``/proc/self/smaps_rollup``."""
    with open("/proc/self/smaps_rollup", encoding="utf-8") as file:
        fields = dict(line.split(":", 1) for line in file.read().splitlines()[1:])
    return sum(int(fields[name].split()[0]) * 1024 for name in ("Private_Clean", "Private_Dirty"))


def _assignment(connection: Connection, worker: int, epoch: int) -> int:
    stream = connection.recv()
    assert isinstance(stream, _Stream)
    return sum(1 for _ in stream.assignment(worker, epoch=epoch))


def hold_assignment(connection: Connection, worker: int, epoch: int) -> None:
    """Receive streams as a loader worker does, make every row of their assignments, and send back what it cost.

    The first stream received only warms the process up: the modules unpickling and reading a stream imports are
    imported then, as a loader worker has them after its first epoch. Of the second it sends back ``(rows made,
    private bytes gained, Python peak bytes, bytes arrow allocated)``.
    """
    _assignment(connection, worker, epoch)
    before = private_bytes()
    tracemalloc.start()
    made = _assignment(connection, worker, epoch)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    connection.send((made, private_bytes() - before, peak, pa.total_allocated_bytes()))
