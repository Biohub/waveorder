"""What ``wo`` prints: progress while a command runs, tables, warnings, and the error boundary.

stdout carries results only; status lines (``wrote ...``), progress and warnings go to stderr. Plain text when
stderr is not a terminal, Rich when it is. The root group's :meth:`RootGroup.main` runs Typer with
``standalone_mode=False`` and handles errors itself, so the console script and ``CliRunner`` run the same code.
Exit codes: 0 success; 2 a request refused before anything ran (usage errors, :class:`~waveorder.errors.Rejected`);
1 a valid request that failed; 3 a ``--keep-going`` run with failed units; 130 an interrupt. An error that is not
waveorder's own (a bug) prints its stack.
"""

import inspect
import logging
import math
import os
import sys
import threading
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TextIO

import typer
from typer._click import Command, Context, Parameter
from typer._click.exceptions import NoArgsIsHelpError, UsageError
from typer._click.globals import get_current_context
from typer.core import TyperGroup, TyperOption

from waveorder.cli.options import PROGRESS_LOGGER
from waveorder.errors import PlacementError, Problem, ReconstructionFailed, Rejected, WaveorderError, WaveorderWarning

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

log = logging.getLogger(PROGRESS_LOGGER)

INTERRUPTED = 130
"""The exit code of a run stopped by Ctrl-C (SIGINT), as a shell reports it."""

RERUN_HINT = "fix the cause and rerun with --resume to skip the finished units; --keep-going runs past a failed unit"
"""What to do after a reconstruction stopped on a failed unit (exit 1)."""

SUMMARY_S = 60.0
"""Seconds between the summaries of a run shown as lines."""


def status(text: str) -> None:
    """Print a status line, such as ``wrote phase.yml``, on stderr."""
    typer.echo(text, err=True)


# --- progress -------------------------------------------------------------------------------------------------------


def _seconds(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else None


def _units(count: int) -> str:
    return "unit" if count == 1 else "units"


class RunDisplay:
    """Progress of one command on stderr, fed with the events of its calibration and its units.

    On a terminal, a Rich progress bar over the units; otherwise one timestamped line per unit and a summary every
    ``interval`` seconds. Lines go through the progress logger, so Rich prints them above the bar.

    Parameters
    ----------
    bar : bool
        A Rich bar over the units, else lines and summaries.
    interval : float
        Seconds between summaries without a bar.
    """

    def __init__(self, *, bar: bool, interval: float = SUMMARY_S) -> None:
        self.bar = bar
        self.interval = interval
        self.total = 0
        self.counts: Counter[str] = Counter()
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bar: Progress | None = None
        self._task: TaskID | None = None

    def event(self, name: str, payload: Mapping[str, object]) -> None:
        """Show one event: a calibration that starts, a calibration stage, the start of the units, a unit."""
        if name == "calibration.stage.end":
            elapsed = _seconds(payload, "elapsed_s")
            objective = payload.get("objective")
            log.info(
                "calibration stage %s %s: objective %s, %.1f s",
                payload.get("index"),
                payload.get("stage"),
                f"{objective:.6g}" if isinstance(objective, int | float) else objective,
                math.nan if elapsed is None else elapsed,
            )
        elif name == "tf.store":
            log.info("calibrating on %s at t=%s ...", payload.get("position"), payload.get("time"))
        elif name == "units.start":
            self._start(payload)
        elif name == "unit.end":
            self.unit(payload)

    def _start(self, payload: Mapping[str, object]) -> None:
        total = payload.get("total")
        self.total = int(total) if isinstance(total, int) else 0
        self._started = time.monotonic()
        if not self.total:
            return
        where = Path(str(payload.get("output"))).name
        if self.bar:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._bar = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("units, {task.fields[failed]} failed"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=Console(stderr=True),
            )
            self._task = self._bar.add_task(where, total=self.total, completed=0, failed=0)
            self._bar.start()
            return
        devices = payload.get("devices")
        on = f" on {', '.join(map(str, devices))}" if isinstance(devices, list) and devices else ""
        workers = payload.get("workers")
        shared = f" ({workers} workers)" if isinstance(workers, int) and workers > 1 else ""
        log.info("%s: %d %s%s%s", where, self.total, _units(self.total), on, shared)
        self._thread = threading.Thread(target=self._summaries, name="wo-progress", daemon=True)
        self._thread.start()

    def unit(self, result: Mapping[str, object]) -> None:
        """Count a unit that ended (the fields of its ``unit.end``) and show it; a failed unit is logged by the run."""
        status = str(result.get("status"))
        with self._lock:
            self.counts[status] += 1
            ended = sum(self.counts.values())
        if self._bar is not None and self._task is not None:
            self._bar.update(self._task, advance=1, failed=self.counts["failed"])
            return
        if status == "failed":
            return
        said = {"complete": "reconstructed", "empty": "empty (zeros written)"}.get(status, status)
        wall = _seconds(result, "wall_s")
        took = f" in {wall:.1f} s" if wall else ""
        on = f" on {result['device']}" if result.get("device") else ""
        log.info("unit %s %s%s%s; %d/%d %s", result.get("unit"), said, took, on, ended, self.total, _units(self.total))

    def summary(self) -> str:
        """Return the periodic summary: units ended, by outcome, and left; seconds elapsed."""
        with self._lock:
            counts = Counter(self.counts)
        ended = sum(counts.values())
        return (
            f"{ended}/{self.total} {_units(self.total)}: {counts['complete'] + counts['empty']} reconstructed, "
            f"{counts['failed']} failed, {self.total - ended} left; {time.monotonic() - self._started:.0f} s"
        )

    def _summaries(self) -> None:
        while not self._stop.wait(self.interval):
            log.info("%s", self.summary())

    def close(self) -> None:
        """Stop the bar and the summaries."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._bar is not None:
            self._bar.stop()
            self._bar = None


# --- the error boundary ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """Why an invocation failed: its exit code, the error lines, and what is printed around them.

    Parameters
    ----------
    code : int
        Exit code: 2 for a request refused before anything ran, 1 for a valid request that failed.
    problems : tuple[Problem, ...]
        The error lines.
    usage : str, optional
        The command's usage line and help hint, printed before a usage error.
    stack : str, optional
        The formatted stack of an error that is not waveorder's own.
    hint : str, optional
        What to do next, printed after the error lines.
    """

    code: int
    problems: tuple[Problem, ...]
    usage: str | None = None
    stack: str | None = None
    hint: str | None = None


class RootGroup(TyperGroup):
    """The ``wo`` group: hides the shell-completion options (the epilog names them) and owns the error boundary.

    It also joins the wrapped lines of every command's help (:func:`_unwrapped`), so the help fills the terminal.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 (click's arguments, passed through)
        super().__init__(*args, **kwargs)
        _unwrap_help(self)

    def get_params(self, ctx: Context) -> list[Parameter]:
        """Return the root's parameters, with ``--install-completion`` and ``--show-completion`` hidden."""
        params = super().get_params(ctx)
        for param in params:
            if isinstance(param, TyperOption) and param.name in {"install_completion", "show_completion"}:
                param.hidden = True
        return params

    def main(  # ty: ignore[incompatible-method-override]
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        **extra: Any,  # noqa: ANN401 (click's keyword arguments, passed through)
    ) -> NoReturn:
        """Run ``wo`` on ``args`` (default ``sys.argv[1:]``) and exit with its code; never returns."""
        argv = list(sys.argv[1:] if args is None else args)
        prog = prog_name or os.path.basename(sys.argv[0]) or "wo"
        extra.pop("standalone_mode", None)
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            words = _command_path(None, group=self, argv=argv, prog=prog)
            # ty types the attribute as the stdlib function; any callable of its signature is what warnings calls.
            warnings.showwarning = _WarningPrinter(fallback=words)  # ty: ignore[invalid-assignment]
            try:
                # Typer turns a KeyboardInterrupt into exit code 130 without a word; the boundary says so once.
                code = super().main(argv, prog, standalone_mode=False, **extra)
            except NoArgsIsHelpError:
                code = 2  # the help was printed when the error was made
            except Exception as error:  # noqa: BLE001 (the boundary: every failure becomes an exit code)
                failure = _failure(error)
                render_failure(failure, command=_command_path(error, group=self, argv=argv, prog=prog))
                code = failure.code
        if code == INTERRUPTED:
            typer.echo(f"{words}: interrupted", err=True)
        sys.exit(code if isinstance(code, int) else 0)


def _unwrap_help(command: Command) -> None:
    """Join the wrapped lines of the help of ``command`` and of every command below it; epilogs stay as written."""
    if command.help:
        command.help = _unwrapped(command.help)
    if isinstance(command, TyperGroup):
        for child in command.commands.values():
            _unwrap_help(child)


def _unwrapped(text: str) -> str:
    r"""Return a docstring with each paragraph's prose lines joined into one line.

    Rich help keeps a docstring's line breaks, so a paragraph wrapped at 120 columns breaks mid-sentence in a narrower
    terminal. Blank-line paragraph breaks stay; so do a ``\b`` paragraph (click's do-not-rewrap marker), an indented
    line and an ``Example:`` line, which are kept as written on lines of their own.
    """

    def prose(line: str) -> bool:
        return bool(line.strip()) and not line.startswith((" ", "\t", "Example:"))

    paragraphs = []
    for paragraph in inspect.cleandoc(text).split("\n\n"):
        if paragraph.startswith("\b"):
            paragraphs.append(paragraph)
            continue
        lines: list[str] = []
        for line in paragraph.split("\n"):
            if lines and prose(line) and prose(lines[-1]):
                lines[-1] = f"{lines[-1]} {line}"
            else:
                lines.append(line)
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


class _WarningPrinter:
    """A ``warnings.showwarning`` that prints each of waveorder's warnings once as ``<command>: warning: ...``.

    Other packages' warnings are not shown. A warning that carries a location (a configuration warning carries its
    dotted configuration path) is reported there; ``fallback`` names the command outside a context.
    """

    def __init__(self, *, fallback: str) -> None:
        self.fallback = fallback
        self.shown: set[Problem] = set()

    def __call__(  # noqa: PLR0917 (the warnings.showwarning signature)
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,  # noqa: ARG002
        lineno: int,  # noqa: ARG002
        file: TextIO | None = None,  # noqa: ARG002
        line: str | None = None,  # noqa: ARG002
    ) -> None:
        """Print ``message`` unless it was shown already or is not waveorder's."""
        if not issubclass(category, WaveorderWarning):
            return
        warning = Problem(str(message), getattr(message, "location", None))
        if warning in self.shown:
            return
        self.shown.add(warning)
        ctx = get_current_context(silent=True)
        render_warning(warning, command=ctx.command_path if ctx is not None else self.fallback)


def _command_path(error: Exception | None, *, group: TyperGroup, argv: list[str], prog: str) -> str:
    """``wo config check`` for an error in that command: from the error's context, else from the leading words."""
    ctx = getattr(error, "ctx", None)
    if ctx is not None:
        return ctx.command_path
    words = [prog]
    command: object = group
    for token in argv:
        if token.startswith("-"):
            continue
        commands = getattr(command, "commands", None)
        name = token.replace("_", "-")
        if not isinstance(commands, dict) or name not in commands:
            break
        command = commands[name]
        words.append(name)
    return " ".join(words)


def _failure(error: Exception) -> Failure:
    """Classify ``error`` into an exit code and error lines."""
    if isinstance(error, UsageError):
        usage = None
        if error.ctx is not None:
            hint = f"Try '{error.ctx.command_path} {error.ctx.help_option_names[0]}' for help."
            usage = f"{error.ctx.get_usage()}\n{hint}"
        return Failure(error.exit_code, (Problem(error.format_message()),), usage=usage)
    if isinstance(error, typer.Abort | typer.TyperException):
        message = error.format_message() if isinstance(error, typer.TyperException) else "aborted"
        return Failure(getattr(error, "exit_code", 1), (Problem(message),))
    if isinstance(error, Rejected):
        return Failure(2, error.problems)
    if isinstance(error, ReconstructionFailed):  # the message names the unit and its error
        return Failure(1, (Problem(str(error)),), hint=RERUN_HINT)
    line = (Problem(f"{type(error).__name__}: {error}"),)
    if isinstance(error, WaveorderError | OSError) and not isinstance(error, PlacementError):
        return Failure(1, line)
    stack = "".join(traceback.format_exception(error))
    return Failure(1, line, stack=stack, hint="this is a bug in waveorder; please report it with the stack")


def _message(problem: Problem) -> str:
    """Return the message without a leading repeat of a flag location (``--tile: --tile cannot ...``)."""
    location = problem.location
    if location and location.startswith("-"):
        for repeat in (f"{location}: ", f"{location} "):
            if problem.message.startswith(repeat):
                return problem.message.removeprefix(repeat)
    return problem.message


def _line(problem: Problem) -> str:
    return f"{problem.location}: {_message(problem)}" if problem.location else problem.message


def _rich_line(problem: Problem, *, command: str, kind: str) -> str:
    from rich.markup import escape

    style = "bold red" if kind == "error" else "bold yellow"
    location = f"[bold]{escape(problem.location)}[/bold]: " if problem.location else ""
    return f"[dim]{escape(command)}:[/dim] [{style}]{kind}:[/{style}] {location}{escape(_message(problem))}"


def render_warning(warning: Problem, *, command: str) -> None:
    """Print ``<command>: warning: <location>: <message>`` on stderr."""
    if sys.stderr.isatty():
        from rich.console import Console

        Console(stderr=True, highlight=False).print(
            _rich_line(warning, command=command, kind="warning"), soft_wrap=True
        )
        return
    typer.echo(f"{command}: warning: {_line(warning)}", err=True)


def render_failure(failure: Failure, *, command: str) -> None:
    """Print ``failure`` on stderr: the stack, the usage, ``<command>: error: <location>: <message>``, the hint."""
    if failure.stack is not None:
        sys.stderr.write(failure.stack)
    if sys.stderr.isatty():
        from rich.console import Console
        from rich.markup import escape

        console = Console(stderr=True, highlight=False)
        if failure.usage:
            console.print(escape(failure.usage), style="dim")
        for problem in failure.problems:
            console.print(_rich_line(problem, command=command, kind="error"), soft_wrap=True)
        if failure.hint:
            console.print(f"[dim]{escape(command)}:[/dim] {escape(failure.hint)}", soft_wrap=True)
        return
    if failure.usage:
        typer.echo(failure.usage, err=True)
    for problem in failure.problems:
        typer.echo(f"{command}: error: {_line(problem)}", err=True)
    if failure.hint:
        typer.echo(f"{command}: {failure.hint}", err=True)
