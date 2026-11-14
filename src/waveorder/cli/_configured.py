"""The step from a configuration file to a model, which a command takes once it runs.

A command reads its configuration with :func:`resolve_config`, builds the model with :func:`build_model` and calls
the model's methods inside ``with located(...)``, so a configuration error is reported at the file and the dotted
path of its setting. This module imports the configuration, the models and torch; the commands import it in their
bodies, so ``wo --help`` does not.
"""

import dataclasses
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from waveorder.cli.options import ARGUMENT_FLAGS, split_devices
from waveorder.config import CONFIG_ERRORS, V4Config, load, problems, read
from waveorder.errors import ConfigurationError, ConfigurationWarning, Problems, Rejected
from waveorder.io.tf_store import TfStore
from waveorder.models import ReconstructionModel
from waveorder.nn.parameters import resolve_device

_CONFIG_PATHS = frozenset(ARGUMENT_FLAGS[name] for name in ("channels", "pixel_size", "times"))


@dataclass(frozen=True, kw_only=True)
class ResolvedConfig:
    """A configuration file, with ``--device`` applied, validated.

    Attributes
    ----------
    path : Path
        The file.
    config : V4Config
        The validated configuration.
    """

    path: Path
    config: V4Config

    @property
    def data(self) -> dict[str, Any]:
        """The ``data`` block as the ``channels``, ``pixel_size`` and ``times`` arguments of the position methods."""
        return self.config.data.arguments()

    @property
    def channels(self) -> list[str]:
        """``data.input_channel_names``."""
        return self.data["channels"]

    @property
    def pixel_size(self) -> tuple[float, float, float] | None:
        """``data.pixel_size`` as ``(z, y, x)`` in um; ``None`` for ``auto`` (each position's recorded size)."""
        return self.data["pixel_size"]

    @property
    def times(self) -> list[int] | None:
        """``data.time_indices``: ``None`` for ``all``."""
        return self.data["times"]

    @property
    def wants_transfer(self) -> bool:
        """Whether ``outputs`` asks for the transfer function, which a store then holds per shape."""
        return self.config.outputs == "all" or "transfer_function" in self.config.outputs


def resolve_config(path: Path, *, device: str | None = None) -> ResolvedConfig:
    """Read ``path``, set ``solver.device`` to the first device of ``device`` and validate it.

    Raises
    ------
    Rejected
        With every configuration error, each naming the dotted path of its setting, located at the file.
    """
    if device is None:
        return ResolvedConfig(path=path, config=load(path))
    try:
        data = read(path)
    except (*CONFIG_ERRORS, OSError) as error:
        raise Rejected(problems(error, source=str(path))) from error
    solver = data.setdefault("solver", {})
    if isinstance(solver, dict):  # another value is left for validation to refuse
        solver["device"] = split_devices(device)[0]
    try:
        config = load(data)
    except Rejected as error:  # located at the mapping, which is the file's
        raise Rejected([dataclasses.replace(problem, location=str(path)) for problem in error.problems]) from error
    return ResolvedConfig(path=path, config=config)


@contextmanager
def located(resolved: ResolvedConfig) -> Generator[None]:
    """Report what the block raises about the configuration at the file.

    A configuration error (:data:`waveorder.config.CONFIG_ERRORS`) becomes a ``Rejected`` at the file, and so does
    a problem of the model's checks placed at ``data.input_channel_names``, ``data.pixel_size`` or
    ``data.time_indices``. Any other error passes through.
    """
    try:
        yield
    except CONFIG_ERRORS as error:
        raise Rejected(problems(error, source=str(resolved.path))) from error
    except Rejected as error:
        if not any(problem.location in _CONFIG_PATHS for problem in error.problems):
            raise
        moved = [
            dataclasses.replace(problem, location=str(resolved.path)) if problem.location in _CONFIG_PATHS else problem
            for problem in error.problems
        ]
        raise Rejected(moved) from error


@contextmanager
def collecting(problems: Problems, resolved: ResolvedConfig) -> Generator[None]:
    """Run the block; what it refuses is added to ``problems``, located as :func:`located` does, and not raised.

    A command runs each independent check in a block of its own and raises once (``problems.raise_if_any()``), so
    a request with several mistakes is refused with all of them.
    """
    with problems.collect(), located(resolved):
        yield


def load_store(path: Path | None, *, flag: str, problems: Problems) -> TfStore | None:
    """Read the store ``flag`` gives (``None``: not given); a store that cannot be read is one of ``problems``."""
    if path is None:
        return None
    try:
        return TfStore.load(path)
    except Rejected as error:
        problems.extend([dataclasses.replace(problem, location=f"{flag} {path}") for problem in error.problems])
        return None


def build_model(resolved: ResolvedConfig, *, on_cpu: bool = False) -> ReconstructionModel:
    """Build the model of a resolved configuration (:meth:`waveorder.models.ReconstructionModel.from_config`).

    Parameters
    ----------
    resolved : ResolvedConfig
        The configuration.
    on_cpu : bool
        Build on the CPU whatever ``solver.device`` says, as ``wo config check`` does: a login node checks a GPU
        configuration. A ``solver.device`` this machine does not have is then a ``ConfigurationWarning``, since the
        run checks it where it executes.

    Raises
    ------
    Rejected
        With what the build refuses, located at the file.
    """
    with located(resolved):
        config = _cpu_config(resolved.config) if on_cpu else resolved.config
        try:
            return ReconstructionModel.from_config(config)
        except Rejected as error:  # given a validated configuration, the build locates its problems at no file
            here = str(resolved.path)
            raise Rejected([dataclasses.replace(problem, location=here) for problem in error.problems]) from error


def _cpu_config(config: V4Config) -> V4Config:
    """Return ``config`` placed on the CPU, after checking that its device is one."""
    name = config.solver.device
    if name is not None and name != "auto":
        try:
            torch.device(name)
        except RuntimeError:
            message = f"{name!r} is not a device; use cpu, cuda, cuda:1, mps or auto"
            raise ConfigurationError("solver.device", message) from None
        try:
            resolve_device(name)
        except ConfigurationError as error:
            reason = str(error).removeprefix("solver.device: ").splitlines()[0]
            warning = ConfigurationWarning(
                f"{reason}; the run checks it again where it executes", field="solver.device"
            )
            warnings.warn(warning, stacklevel=2)
    solver = config.solver.model_copy(update={"device": None})
    return config.model_copy(update={"solver": solver})
