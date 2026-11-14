"""Bead calibration: fit the system's free quantities to a volume of beads of known diameter (3.x ``beads``).

The beads' positions, amplitudes and the background are nuisance variables, not free quantities of the model:
:func:`calibrate_beads` steps them with the system's free quantities in one Adam optimizer, each group at its own rate.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.model import ImagingModel, expect_on_system
from waveorder.nn.noise import Gaussian, estimate_sigma
from waveorder.nn.parameters import ParameterPrior, free_specs
from waveorder.nn.priors import Beads
from waveorder.nn.sampling import Data, Sampling


@dataclass(frozen=True, slots=True)
class BeadCalibration:
    """What :func:`calibrate_beads` fitted, on the model's device.

    Attributes
    ----------
    parameters : dict[str, Tensor]
        The system's free quantities by dotted name; the model holds them too.
    positions : Tensor
        ``(N, 3)`` bead positions ``(z, y, x)`` in voxels, float64.
    amplitudes : Tensor
        ``(N,)`` bead amplitudes in data units, float64.
    background : Tensor
        0-d constant background, data units.
    noise_sigma : Tensor
        0-d noise standard deviation: fitted when the noise was built with ``sigma="estimate"``, else the model's.
        The model's noise is left as it was.
    trace : Tensor
        ``(iterations,)`` objective before each step.
    """

    parameters: dict[str, Tensor]
    positions: Tensor
    amplitudes: Tensor
    background: Tensor
    noise_sigma: Tensor
    trace: Tensor


def calibrate_beads(
    data: Data, model: ImagingModel, sampling: Sampling, *, iterations: int = 200, learning_rate: float = 0.01
) -> BeadCalibration:
    """Fit the system's free quantities, the beads and a background to ``data`` by maximum a posteriori.

    The data are modelled as ``H (sum_i a_i s(r - r_i)) + b`` plus white Gaussian noise of deviation ``sigma``, with
    ``s`` the sphere of :class:`~waveorder.nn.priors.Beads`. The objective, in float64, is
    ``sum (predicted - g)^2 / (2 sigma^2) + N log(2 pi sigma^2) / 2`` plus the free quantities' prior. Beads start
    at :meth:`~waveorder.nn.priors.Beads.detect`'s peaks, the background at the median and an estimated ``sigma`` at
    ``1.4826 * MAD``. Adam runs the system's quantities and ``log sigma`` at ``learning_rate``, positions, log
    amplitudes and background at 10 times it, and clamps the system's quantities to their bounds after each step. The
    last iterate is kept, as in 3.x.

    Parameters
    ----------
    data : Tensor
        One-channel data ``(1, Z, Y, X)`` on ``sampling``, in the dtype and on the device of the model's system.
    model : ImagingModel
        System with at least one free quantity, :class:`~waveorder.nn.priors.Beads` prior and Gaussian noise. Its
        system's free quantities are set to the estimate.
    sampling : Sampling
        Grid of ``data``.
    iterations : int
        Adam steps; the gradient stage's ``max_iter`` in the config.
    learning_rate : float
        Step of the system's quantities, in their units.

    Returns
    -------
    BeadCalibration

    Raises
    ------
    IncompatibleModelError
        If the prior is not :class:`~waveorder.nn.priors.Beads`, the noise is not Gaussian, or the system's object
        is not a volume on the data's planes (a thin 2D system).
    ParameterError
        If the system has no free quantity, or no bead is found.
    """
    beads, noise = model.object_prior, model.noise
    if not isinstance(beads, Beads) or not isinstance(noise, Gaussian):
        raise IncompatibleModelError(
            f"calibrate_beads needs a Beads prior and Gaussian noise, got {type(beads).__name__} and "
            f"{type(noise).__name__}"
        )
    object_shape = model.system.object_shape(sampling)
    if tuple(object_shape[1:]) != sampling.shape:
        raise IncompatibleModelError(
            f"calibrate_beads fits a {sampling.shape} bead volume, but {type(model.system).__name__} takes an object "
            f"of shape {tuple(object_shape)}; bead calibration supports 3D fluorescence systems "
            "(WidefieldFluorescence3D, ConfocalFluorescence3D), not a single-plane one"
        )
    expect_on_system(data, model.system, where="calibrate_beads", name="data", shape=model.system.data_shape(sampling))
    specs = {name: spec for name, spec in free_specs(model).items() if name.startswith("system.")}
    if not specs:
        raise ParameterError("calibrate_beads needs at least one free quantity of the system")
    prior, parameters = ParameterPrior(specs), dict(model.named_parameters())
    volume = data[0]
    wide = torch.float64
    target = volume.to(wide)
    peaks = beads.detect(volume)
    median = volume.median()
    positions = peaks.to(wide).requires_grad_()
    with torch.no_grad():  # a bead of amplitude a peaks at a * peak(H s) in the data, not at a
        unit = beads.render(positions.new_zeros(1, 3), positions.new_ones(1), sampling).to(volume.dtype)
        unit_peak = model.system(unit[None], sampling)[0].max().to(wide)
    log_amplitudes = ((volume[peaks.unbind(1)] - median).clamp(min=1e-6).to(wide) / unit_peak).log().requires_grad_()
    background = median.to(wide).clone().requires_grad_()
    sigma = estimate_sigma(volume, method="mad") if noise.estimated else noise.sigma.detach().clamp(min=1e-6)
    log_sigma = sigma.to(wide).log().clone().requires_grad_(noise.estimated)
    groups = [
        {"params": [parameters[name] for name in specs], "lr": learning_rate},
        {"params": [positions], "lr": 10 * learning_rate},
        {"params": [log_amplitudes, background], "lr": 10 * learning_rate},
    ]
    if noise.estimated:
        groups.append({"params": [log_sigma], "lr": learning_rate})
    optimizer = torch.optim.Adam(groups)

    def objective() -> Tensor:
        obj = beads.render(positions, log_amplitudes.exp(), sampling).to(volume.dtype)
        predicted = model.system(obj[None], sampling)[0].to(wide) + background
        variance = (2 * log_sigma).exp()
        misfit = (predicted - target).square().sum() / (2 * variance)
        normalization = 0.5 * volume.numel() * torch.log(2 * math.pi * variance)
        return misfit + normalization + prior.nll({name: parameters[name] for name in specs})

    trace = []
    for _ in range(iterations):
        optimizer.zero_grad()
        value = objective()
        value.backward()
        optimizer.step()
        prior.clamp_(model)
        trace.append(value.detach())
    return BeadCalibration(
        parameters={name: parameters[name].detach().clone() for name in specs},
        positions=positions.detach(),
        amplitudes=log_amplitudes.detach().exp(),
        background=background.detach(),
        noise_sigma=log_sigma.detach().exp(),
        trace=torch.stack(trace),
    )
