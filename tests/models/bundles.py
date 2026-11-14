"""Tiny distilled students of the thin phase model, saved as bundles, for the tests of the distilled solver."""

from pathlib import Path

import torch

from tests.models.plates import PIXEL_SIZE, thin_phase_model
from tests.nn.networks import tiny
from waveorder.io.bundle import save_bundle
from waveorder.models import ReconstructionModel
from waveorder.models._stores import system_sha256
from waveorder.nn.networks import Bundle, Contract, Example


def phase_bundle(path: Path, model: ReconstructionModel | None = None, *, planes: int = 8, hashed: bool = True) -> Path:
    """Save a one-field phase student for stacks of ``planes`` planes, trained for ``model``'s imaging system.

    The contract records the sha256 of the system of ``model`` (by default :func:`thin_phase_model`), as a fit of a
    student for it does; with ``hashed=False`` it records none, as a contract made without a configuration form.
    """
    model = thin_phase_model() if model is None else model
    contract = Contract.of(
        model.imaging_model,
        pixel_size=PIXEL_SIZE,
        model=type(model).__name__,
        prepare_scope=model.prepare_scope,
        planes=(planes,),
        fields=("phase",),
        sha256=system_sha256(model) if hashed else None,
    )
    data = torch.randn(1, 1, planes, 32, 32, generator=torch.Generator().manual_seed(0))
    bundle = Bundle(
        student=tiny.strided(planes, halo=16),
        contract=contract,
        model={"class": type(model).__name__},
        test=Example(data=data),
    )
    return save_bundle(bundle, path)
