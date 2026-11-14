"""The reconstruction jobs the GUI starts: ``wo`` commands run one after another, each in its own process."""

import logging
import shlex
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from qtpy.QtCore import QObject, QProcess, Signal

logger = logging.getLogger(__name__)

WO = (sys.executable, "-m", "waveorder.cli")
"""How a job runs ``wo``: with the interpreter of the GUI, so the job sees the same installation."""


def shown(arguments: Sequence[str]) -> str:
    """Return the ``wo`` command line of a job's arguments, as a user would type it."""
    return shlex.join(["wo", *arguments])


@dataclass
class _Job:
    uid: str
    arguments: tuple[str, ...]
    process: QProcess | None = None
    cancelled: bool = False


class JobManager(QObject):
    """Run ``wo`` commands one at a time, in the order they were submitted.

    Every line a job prints (stdout and stderr merged) is emitted by :attr:`output`; :attr:`finished` gives its exit
    code, ``-1`` if it could not start or was cancelled.
    """

    output = Signal(str, str)
    """``(uid, line)``: a line the job printed."""
    finished = Signal(str, int)
    """``(uid, exit_code)``: the job ended."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.jobs: dict[str, _Job] = {}
        self._queue: deque[_Job] = deque()
        self._running: _Job | None = None

    def run_job(self, uid: str, arguments: Sequence[str]) -> None:
        """Queue ``wo <arguments>`` as job ``uid``.

        Raises
        ------
        ValueError
            If a job ``uid`` is queued or running.
        """
        if uid in self.jobs:
            raise ValueError(f"Job {uid} already running.")
        job = _Job(uid, tuple(arguments))
        self.jobs[uid] = job
        self._queue.append(job)
        logger.info("queued %s: %s", uid, shown(job.arguments))
        self._start_next()

    def cancel_job(self, uid: str) -> None:
        """Stop job ``uid``: remove it from the queue, or terminate its process."""
        job = self.jobs.get(uid)
        if job is None:
            return
        job.cancelled = True
        if job is self._running and job.process is not None:
            job.process.terminate()
        elif job in self._queue:
            self._queue.remove(job)
            self._end(job, -1)

    def is_running(self, uid: str) -> bool:
        """Whether job ``uid`` is queued or running."""
        return uid in self.jobs

    def _start_next(self) -> None:
        if self._running is not None or not self._queue:
            return
        job = self._running = self._queue.popleft()
        process = job.process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._read(job))
        process.finished.connect(lambda code, _status: self._done(job, code))
        process.errorOccurred.connect(lambda error: self._failed(job, error))
        logger.info("running %s: %s", job.uid, shown(job.arguments))
        process.start(WO[0], [*WO[1:], *job.arguments])

    def _read(self, job: _Job) -> None:
        assert job.process is not None
        text = bytes(job.process.readAllStandardOutput().data()).decode(errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.output.emit(job.uid, line.rstrip())

    def _failed(self, job: _Job, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.output.emit(job.uid, f"could not start {WO[0]}: {job.process.errorString() if job.process else ''}")
            self._done(job, -1)

    def _done(self, job: _Job, code: int) -> None:
        if job.uid not in self.jobs:
            return
        if job.process is not None:
            self._read(job)
        self._running = None
        self._end(job, -1 if job.cancelled else code)
        self._start_next()

    def _end(self, job: _Job, code: int) -> None:
        del self.jobs[job.uid]
        logger.info("%s ended with code %d", job.uid, code)
        self.finished.emit(job.uid, code)
