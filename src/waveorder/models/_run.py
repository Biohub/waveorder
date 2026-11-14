"""What every method of a model runs through: its events, the bind, and the calibration score."""

import importlib.metadata
import time
import warnings
from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import nn

from waveorder.errors import CalibrationWarning, ConfigurationError, ParameterError
from waveorder.models._base import ReconstructionModel
from waveorder.models._stores import seed_stored_transfer
from waveorder.nn import Evidence, LinearOperator, Sampling, Score, free_specs
from waveorder.observe import emit, flush, observing


def placed[M: nn.Module](module: M, *, device: torch.device, dtype: torch.dtype) -> M:
    """Move ``module`` to ``device`` and its floating tensors to ``dtype``, in place; return it.

    A complex tensor goes to the complex dtype of ``dtype`` (``nn.Module.to(dtype=...)`` would drop its imaginary
    part), and a fixed quantity is filled again from the number it was registered with, so ``1.2`` in float64 is
    ``1.2`` and not float32's ``1.2000000476837158``.
    """

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_complex():
            return tensor.to(device=device, dtype=dtype.to_complex())
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    return module._apply(convert)  # noqa: SLF001 (the hook nn.Module.to, .double and .cuda run through)


@contextmanager
def running(model: ReconstructionModel, verb: str, *, reported: bool = True) -> Generator[None]:
    """Run a method's body between ``run.start`` and ``run.end`` (method ``model.<verb>``) if reported.

    ``run.end`` first flushes the events waiting for device values (:func:`waveorder.observe.flush`), then reports
    ``"status"`` ``"ok"`` or ``"error"`` (with the exception's message) and the method's wall time.
    """
    method = f"model.{verb}"
    if not (reported and observing()):
        yield
        return
    emit("run.start", _run_start(model, method))
    started = time.perf_counter()
    end: dict[str, object] = {"method": method, "status": "ok"}
    try:
        yield
    except BaseException as error:
        end |= {"status": "error", "error": f"{type(error).__name__}: {error}"}
        raise
    finally:
        flush()
        emit("run.end", end | {"wall_s": time.perf_counter() - started})


def _run_start(model: ReconstructionModel, method: str) -> dict[str, object]:
    """Return the payload of ``run.start``: host values only."""
    versions: dict[str, str | None] = {}
    for package in ("waveorder", "torch", "iohub", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["cuda"] = torch.version.cuda
    config = model.config
    return {
        "method": method,
        "config": None if config is None else config.model_dump(mode="json"),
        "device": str(model.device),
        "versions": versions,
    }


def bind(model: ReconstructionModel, sampling: Sampling, *, stored: bool = False) -> LinearOperator:
    """Bind the model's system to ``sampling``; with ``stored``, a loaded store's fitting transfer is used."""
    with running(model, "bind", reported=False):
        if stored:
            seed_stored_transfer(model, sampling)
        return model.system.bind(sampling)


def calibration_score(model: ReconstructionModel, *, where: str, stacklevel: int = 3, warn: bool = True) -> Score:
    """Return the score a calibration minimizes: the model's, or the evidence when it has none.

    With ``warn``, a likelihood score warns once (``CalibrationWarning``) if the noise or the object prior is the
    default of a model built without one, and once if the model pads z: both bias the estimate.

    Parameters
    ----------
    model : ReconstructionModel
        The model about to be calibrated.
    where : str
        The calibrating method, for the messages.
    stacklevel : int
        How many frames above this function the method's caller is, for the warnings.
    warn : bool
        Emit the warnings; a check that only asks whether a calibration could run passes ``False``.

    Raises
    ------
    ConfigurationError
        At ``imaging_system``, for a model built from a configuration in which no quantity is free.
    ParameterError
        For a model built in Python in which no quantity is free.
    """
    if not free_specs(model.imaging_model):
        elsewhere = _free_elsewhere(model)
        if model.config is not None:
            raise ConfigurationError(
                "imaging_system",
                "has no free quantity, so there is nothing to calibrate; write the quantity to estimate as "
                "{nominal: ..., tolerance: ..., bounds: [low, high]}, e.g. under detection.numerical_aperture"
                + elsewhere,
            )
        raise ParameterError(
            f"{where}: the model has no free quantity to calibrate; give the quantity to estimate as "
            "Free(nominal, bounds=(low, high)), e.g. Detection(numerical_aperture=Free(1.2, bounds=(1.0, 1.3)))"
            + elsewhere
        )
    score = model.score
    if score is None:
        score = Evidence().to(model.device)
    if warn:
        _warn_of_bias(model, score, where=where, stacklevel=stacklevel + 1)
    return score


def _free_elsewhere(model: ReconstructionModel) -> str:
    """Return a sentence naming the free quantities of the solver and the score, which no calibration estimates."""
    held = {"model.solver": model.solver, "model.score": model.score}
    found = {name: sorted(free_specs(part)) for name, part in held.items() if part is not None}
    named = [f"{names} of {name}" for name, names in found.items() if names]
    if not named:
        return ""
    return (
        f". The free quantities {' and '.join(named)} are not calibrated: a calibration estimates the free "
        "quantities of model.imaging_model (the system, the object prior and the noise)"
    )


def _warn_of_bias(model: ReconstructionModel, score: Score, *, where: str, stacklevel: int) -> None:
    """Warn of what biases a calibration with a likelihood score: default noise or prior, and z padding."""
    defaulted = model.defaulted
    if score.is_likelihood and defaulted:
        parts = " and ".join(f"{name} ({getattr(model, name)!r})" for name in defaulted)
        verb = "is the default" if len(defaulted) == 1 else "are the defaults"
        warnings.warn(
            f"{where}: {type(score).__name__} is a likelihood of the data under the model's noise and object prior, "
            f"but {parts} {verb} (none was given), not values chosen for this data, so the estimate "
            "may follow them instead of the data. Pass noise=Gaussian(sigma=...) with sigma in "
            "the units of the data as the system measures it (after prepare), and object_prior=Tikhonov(Free(...)) "
            "to estimate the weight with the other free quantities (or a number: the noise variance over the "
            "object's spectral variance); in a configuration, noise.gaussian.sigma and object.weight (auto).",
            CalibrationWarning,
            stacklevel=stacklevel,
        )
    if score.is_likelihood and model.z_padding:
        setting = "solver.padding.z" if model.config is not None else "z_padding"
        warnings.warn(
            f"{where}: {setting} is {model.z_padding}, so {type(score).__name__} scores the stack with its mirrored "
            "slices as if they were measured. They repeat measured slices and are no data the imaging model "
            "produces, so the estimate is biased. Calibrate with "
            f"{setting} 0 and pad only the reconstruction: in Python, set model.z_padding = 0 before the "
            f"calibration and model.z_padding = {model.z_padding!r} after it.",
            CalibrationWarning,
            stacklevel=stacklevel,
        )
