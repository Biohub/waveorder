"""Exceptions and warnings raised by waveorder.

Every exception derives from :class:`WaveorderError` and from the builtin type a caller would otherwise catch
(``ValueError``, ``NotImplementedError``, ...), so ``except ValueError`` keeps working. Messages name the offending
config field or dotted parameter path. Schema validation of a configuration raises pydantic's ``ValidationError``
(also a ``ValueError``) instead. This module imports no torch and nothing else of waveorder.
"""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Literal, Self

argument_names: ContextVar[Mapping[str, str]] = ContextVar(
    "waveorder.errors.argument_names", default=MappingProxyType({})
)
"""How the caller spells the arguments that messages name: Python argument name to shown name. Empty by default, so
messages use the Python names; the command line maps them to its flags (``{"positions": "-i", ...}``)."""


def named(argument: str) -> str:
    """Return how the caller spells ``argument`` (``--tile`` on the command line), else ``argument`` itself."""
    return argument_names.get().get(argument, argument)


@dataclass(frozen=True)
class Problem:
    """One problem with a request: what is wrong, and where.

    Parameters
    ----------
    message : str
        What is wrong and how to fix it.
    location : str, optional
        Where the offending value came from, e.g. ``"phase.yml"`` or ``"-i input.zarr"``.
    """

    message: str
    location: str | None = None

    def __str__(self) -> str:
        return f"{self.location}: {self.message}" if self.location else self.message


class WaveorderError(Exception):
    """Base class of every exception waveorder raises on purpose."""


class Rejected(WaveorderError, ValueError):  # noqa: N818 (the name says what happened to the request)
    """A request refused by the checks run before computing: nothing was computed or written.

    The checks collect every problem they find, so one ``Rejected`` carries them all.

    Parameters
    ----------
    problems : Sequence[Problem]
        What is wrong, each with where the offending value came from; at least one.

    Raises
    ------
    ValueError
        If ``problems`` is empty.
    """

    def __init__(self, problems: Sequence[Problem]) -> None:
        if not problems:
            raise ValueError("Rejected needs at least one problem")
        self.problems = tuple(problems)
        super().__init__("\n".join(str(problem) for problem in self.problems))

    @classmethod
    def one(cls, message: str, *, location: str | None = None) -> Self:
        """Refuse a request for one problem, e.g. ``Rejected.one("matches nothing", location="-i in.zarr")``."""
        return cls([Problem(message, location)])


class Problems:
    """Collects the problems a request's checks find, so that one :class:`Rejected` reports them all.

    Example::

        problems = Problems()
        with problems.collect():
            positions = expand_positions(patterns)  # may raise Rejected
        problems.raise_if_any()

    Attributes
    ----------
    items : list[Problem]
        The problems found so far, in the order found, each once.
    """

    def __init__(self) -> None:
        self.items: list[Problem] = []

    def add(self, message: str, location: str | None = None) -> None:
        """Add one problem: what is wrong and where the offending value came from."""
        self.extend([Problem(message, location)])

    def extend(self, problems: Sequence[Problem]) -> None:
        """Add ``problems``, leaving out those already collected."""
        for problem in problems:
            if problem not in self.items:
                self.items.append(problem)

    def take(self, error: Rejected, *, location: str | None = None) -> None:
        """Add the problems of ``error``; ``location`` locates those that have none."""
        self.extend([Problem(p.message, p.location or location) for p in error.problems])

    @contextmanager
    def collect(self, *, location: str | None = None) -> Generator[None]:
        """Run the block; a ``Rejected`` it raises is collected (at ``location`` when it has none) and not raised."""
        try:
            yield
        except Rejected as error:
            self.take(error, location=location)

    def raise_if_any(self) -> None:
        """Raise one ``Rejected`` with every problem collected, if there is any.

        Raises
        ------
        Rejected
            With the problems, in the order found.
        """
        if self.items:
            raise Rejected(self.items)


class ShapeError(WaveorderError, ValueError):
    """A tensor's shape does not match the shape its sampling and module declare."""


class PlacementError(ShapeError):
    """A tensor's dtype or device differs from the dtype and device its module computes in."""


class ParameterError(WaveorderError, ValueError):
    """A quantity or parameter is invalid: a non-finite value, bad bounds, or an unknown dotted path.

    Parameters
    ----------
    message : str
        What is wrong and how to fix it.
    path : str, optional
        Dotted path of the offending quantity within the module that raised, e.g. ``"detection.numerical_aperture"``;
        ``None`` when the error is not about one quantity.
    """

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class IncompatibleModelError(WaveorderError, ValueError):
    """A solver, score or sampling cannot be used with this combination of system, object prior and noise."""


class BackendUnavailableError(WaveorderError, RuntimeError):
    """Something explicitly requested cannot run on this machine, e.g. no CUDA device or an unsupported precision."""


class ReadError(WaveorderError, OSError):
    """Recorded data could not be read from a store: a missing or corrupt chunk, or a file system failure.

    The message names the position and the time point; the error the store raised is its ``__cause__``.
    """


class BundleError(WaveorderError, RuntimeError):
    """A student bundle cannot be used: a file is missing or altered, the schema is another, or a self-test failed.

    The message names the bundle directory and the file or value that is wrong.
    """


type UnitStatus = Literal["complete", "empty", "skipped", "failed"]
"""``complete``: reconstructed and written; ``empty``: the input is all zero or NaN, so nothing was written (the
output keeps its fill value, zero); ``skipped``: ``resume`` found it finished by an earlier run of the same
reconstruction; ``failed``: reading, solving or writing raised, or the worker process died."""

type ReportStatus = Literal["finished", "planned", "failed"]
"""``finished``: every unit is complete, empty or skipped; ``planned``: ``dry_run=True``, nothing was written;
``failed``: a unit failed (the run stopped there unless ``keep_going=True``)."""


@dataclass(frozen=True, kw_only=True)
class UnitResult:
    """How one (position, time) unit of a reconstruction of positions ended.

    Attributes
    ----------
    unit : str
        ``<position>/t<T>``, e.g. ``A/1/0/t3``.
    position : str
        Plate-relative name of the position, e.g. ``A/1/0``.
    time : int
        Absolute time index.
    status : {"complete", "empty", "skipped", "failed"}
        How it ended.
    error, error_type : str, optional
        ``Type: message`` of a failed unit, and the class name alone (``"ReadError"``, ``"OutOfMemoryError"``, ...).
    wall_s : float
        Seconds the unit took.
    read_s, solve_s, write_s : float, optional
        Of those, seconds spent reading, solving and writing.
    host, device : str
        Where it ran.
    """

    unit: str
    position: str
    time: int
    status: UnitStatus
    error: str | None = None
    error_type: str | None = None
    wall_s: float = 0.0
    read_s: float | None = None
    solve_s: float | None = None
    write_s: float | None = None
    host: str = ""
    device: str = ""

    def to_json(self) -> dict[str, object]:
        """Return the result as JSON-ready values."""
        return asdict(self)


@dataclass(kw_only=True)
class ReconstructionReport:
    """What :meth:`~waveorder.models.ReconstructionModel.reconstruct_positions` did.

    Attributes
    ----------
    output : str
        The output plate.
    status : {"finished", "planned", "failed"}
        How the run ended.
    total : int
        The (position, time) units selected: ``len(results) + len(pending)``.
    results : list[UnitResult]
        One per unit that ended, in finishing order.
    pending : tuple[str, ...]
        Names (``A/1/0/t3``) of the selected units not run, in run order: all of a ``dry_run``, or those after a
        failure.
    """

    output: str
    status: ReportStatus
    total: int = 0
    results: list[UnitResult] = field(default_factory=list)
    pending: tuple[str, ...] = ()

    def count(self, status: UnitStatus) -> int:
        """Return how many of :attr:`results` ended with ``status``."""
        return sum(1 for result in self.results if result.status == status)


class ReconstructionFailed(WaveorderError, RuntimeError):  # noqa: N818 (named for what happened to the run)
    """A reconstruction of positions stopped by a unit's failure, or by a worker that exited or could not start.

    Parameters
    ----------
    message : str
        Which unit failed and why.
    report : ReconstructionReport
        The report so far, with status ``"failed"``.
    """

    def __init__(self, message: str, *, report: ReconstructionReport) -> None:
        super().__init__(message)
        self.report = report


class TrainingFailed(WaveorderError, RuntimeError):  # noqa: N818 (named for what happened to the run)
    """A distillation fit stopped before it finished: the loss became non-finite, or no step could be run.

    The message says at which step and in which term, and what to change. What the run wrote stays in its directory.
    """


class ConfigurationError(WaveorderError, ValueError):
    """A configuration is valid against the schema but cannot be built into modules.

    Parameters
    ----------
    field : str
        Dotted config path of the offending value, e.g. ``"imaging_system.illumination.tilt.zenith"``.
    message : str
        What is wrong and how to fix it.
    """

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field


class RegistrationError(ConfigurationError):
    """Registering a config option or builder conflicts with an existing one, e.g. a duplicate name."""


class UnsupportedOptionError(WaveorderError, NotImplementedError):
    """A configuration option the schema accepts cannot be built for this value; ``reason`` says why.

    Parameters
    ----------
    field : str
        Dotted config path of the option.
    option : str
        The option's value, e.g. ``"phase"``.
    reason : str
        Why this value of the option cannot be built.
    """

    def __init__(self, field: str, option: str, *, reason: str) -> None:
        super().__init__(f"{field}: cannot represent {option}: {reason}")
        self.field = field
        self.option = option
        self.reason = reason


class WaveorderWarning(UserWarning):
    """Base class of every warning waveorder emits."""


class ConfigurationWarning(WaveorderWarning):
    """A configuration is valid but likely to behave badly, e.g. a grid search with millions of points.

    Parameters
    ----------
    message : str
        What is likely to go wrong and what to do instead.
    field : str, optional
        Dotted config path the warning is about, e.g. ``"solver.calibration.search.0"``.

    Attributes
    ----------
    location : str or None
        ``field``, where the warning is reported.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field
        self.location: str | None = field


class ConvergenceWarning(WaveorderWarning):
    """An iterative solver stopped at its iteration limit without meeting its tolerance."""


class CalibrationWarning(WaveorderWarning):
    """A calibration ran on assumptions that likely bias it, e.g. a likelihood score with defaulted noise and prior."""


class PositionWarning(WaveorderWarning):
    """A method on OME-Zarr positions goes ahead on something the caller should know about.

    E.g. a stated voxel size differs from the recorded one (the stated one is used, as in 3.x), or ``resume`` cannot
    track an output's progress (every time point is recomputed).

    Parameters
    ----------
    message : str
        What happens, and what to do instead.
    location : str, optional
        The argument or the path it is about.
    """

    def __init__(self, message: str, *, location: str | None = None) -> None:
        super().__init__(message)
        self.location: str | None = location


class StoreWarning(WaveorderWarning):
    """A transfer-function store is used or returned, but part of it is ignored or suspect.

    E.g. a quantity ended on a bound, a stored quantity the model fixes, or a prior that constrains nothing.

    Parameters
    ----------
    message : str
        What is ignored or suspect.
    location : str, optional
        The argument with the store's path (``prior p.zarr``), or the position calibrated on.
    """

    def __init__(self, message: str, *, location: str | None = None) -> None:
        super().__init__(message)
        self.location: str | None = location
