"""``admm``, the Richardson-Lucy variants and ``poisson_gaussian`` noise."""

import pytest
import torch
import yaml

from waveorder.errors import Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import ADMM, PoissonGaussian, RichardsonLucy, Sampling

CONFIG = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection: {wavelength: 0.52, numerical_aperture: 1.2}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
"""
SHAPE = (8, 16, 16)
PIXEL_SIZE = (0.25, 0.1, 0.1)  # data.pixel_size of CONFIG


def _raw(*, object_block: dict | None = None, noise: dict | None = None, reconstruction: object = None) -> dict:
    raw = yaml.safe_load(CONFIG)
    if object_block is not None:
        raw["object"] = object_block
    if noise is not None:
        raw["noise"] = noise
    if reconstruction is not None:
        raw["solver"] = {"reconstruction": reconstruction}
    return raw


def _refused_at(raw: dict) -> str:
    """Return the config path of the one problem ``from_config`` refuses ``raw`` with."""
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(raw)
    (problem,) = rejected.value.problems
    return problem.message.split(": ", 1)[0]


def test_admm_takes_its_iterations_and_penalty_and_reconstructs():
    raw = _raw(object_block={"prior": "tv", "weight": 1e-3}, reconstruction={"admm": {"iterations": 7, "rho": 0.01}})
    model = ReconstructionModel.from_config(raw)
    solver = model.solver
    assert isinstance(solver, ADMM)
    assert (solver.loop.iterations, solver.rho) == (7, 0.01)
    sampling = Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE)
    with torch.no_grad():
        data = model.imaging_model(torch.rand(1, *SHAPE, generator=torch.Generator().manual_seed(0)), sampling)
    result = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True)
    assert result.estimate.shape == (1, *SHAPE)
    assert result.iterations == 7


def test_admm_with_a_prior_it_cannot_split_is_refused_at_the_reconstruction():
    assert _refused_at(_raw(reconstruction="admm")) == "solver.reconstruction"  # the default tikhonov prior


def _richardson_lucy(**options: object) -> dict:
    return _raw(
        object_block={"prior": "none"},
        noise={"poisson": {}},
        reconstruction={"richardson_lucy": {"iterations": 3, **options}},
    )


@pytest.mark.parametrize(
    "options",
    [
        {"back_projector": "wiener", "bp_alpha": 0.01},
        {"back_projector": "wiener_butterworth", "bp_alpha": 0.01, "bp_beta": 0.01},
        {"back_projector": "gaussian"},
        {"gradient_consensus": True},
    ],
    ids=["wiener", "wiener_butterworth", "gaussian", "rlgc"],
)
def test_richardson_lucy_variants_take_their_settings(options):
    solver = ReconstructionModel.from_config(_richardson_lucy(**options)).solver
    assert isinstance(solver, RichardsonLucy)
    expected = {"back_projector": "matched", "bp_alpha": None, "bp_beta": None, "gradient_consensus": False} | options
    assert {name: getattr(solver, name) for name in expected} == expected


def test_richardson_lucy_refuses_consensus_with_an_unmatched_projector_at_its_key():
    raw = _richardson_lucy(back_projector="wiener", gradient_consensus=True)
    assert _refused_at(raw) == "solver.reconstruction.richardson_lucy.gradient_consensus"


def test_poisson_gaussian_noise_takes_gain_offset_and_sigma():
    noise = {"poisson_gaussian": {"gain": 2.0, "offset": 100.0, "sigma": 3.0}}
    built = ReconstructionModel.from_config(_raw(object_block={"prior": "none"}, noise=noise)).noise
    assert isinstance(built, PoissonGaussian)
    assert (float(built.gain), float(built.offset), float(built.sigma)) == (2.0, 100.0, 3.0)


def test_richardson_lucy_solves_poisson_gaussian_noise_as_a_model_built_in_python_does():
    # The solver a model chooses for shot noise with read noise, so the configuration that writes it is valid.
    noise = {"poisson_gaussian": {"sigma": 3.0}}
    raw = _raw(object_block={"prior": "none"}, noise=noise, reconstruction="richardson_lucy")
    assert isinstance(ReconstructionModel.from_config(raw).noise, PoissonGaussian)
    # The Gaussian solvers still need a white variance, and Richardson-Lucy a noise that counts photons.
    assert _refused_at(raw | {"solver": {"reconstruction": "fista"}}) == "solver.reconstruction"
    assert _refused_at(raw | {"noise": {"gaussian": {}}}) == "solver.reconstruction"


def test_a_poisson_gaussian_sigma_to_estimate_is_refused_at_its_key():
    noise = {"poisson_gaussian": {"sigma": "estimate"}}
    raw = _raw(object_block={"prior": "none"}, noise=noise, reconstruction="richardson_lucy")
    assert _refused_at(raw) == "noise.poisson_gaussian.sigma"
