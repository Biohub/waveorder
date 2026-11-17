"""wo-4.0's version 4 files build as wo-4.0 ran them: in the form wo-4.0 writes, and where wo-4.0 read no noise."""

import warnings

import pytest
import torch
import yaml

from waveorder.config import load
from waveorder.errors import ConfigurationError, ConfigurationWarning, Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import ADMM, Gaussian, Tikhonov, TotalVariation, Wiener

# wo-4.0's tests wrote a bead calibration short (BEAD_SHORT); its CLI tests wrote
# yaml.safe_dump(V4Config(**short).model_dump()): every default and null spelled out, closed_form included.
BEAD_SHORT = """\
version: 4
data: {input_channel_names: [GFP], pixel_size: {yx: 0.2, z: 0.4}}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.532
    numerical_aperture: 1.2
    pupil_aberrations: {astigmatism_0: {nominal: 0.0, tolerance: free}}
noise: {gaussian: {sigma: 1.0}}
object: {prior: {beads: {diameter: 0.5}}}
solver: {calibration: {score: evidence, search: [{gradient: {max_iter: 150}}]}}
"""
BEAD_DUMP = """\
version: 4
data: {input_channel_names: [GFP], pixel_size: {yx: 0.2, z: 0.4}, time_indices: all}
imaging_system:
  contrast: fluorescence
  field_dependence: shift_invariant
  index_of_refraction_media: 1.3
  detection:
    numerical_aperture: 1.2
    pinhole: none
    pupil_aberrations:
      astigmatism_0: {bounds: null, nominal: 0.0, tolerance: free}
      zernike_up_to_order: null
    wavelength: 0.532
    z_focus_offset: 0.0
  illumination:
    invert_phase_contrast: false
    numerical_aperture: null
    tilt: {azimuth: 0.0, zenith: 0.0}
    wavelength: null
noise: {gaussian: {sigma: 1.0, snr_db: null}, poisson: null, poisson_gaussian: null}
object:
  constraints: []
  dimension: 3
  prior: {beads: {diameter: 0.5, max_beads: 50, min_separation: 5, snr_threshold: 6.0}}
  source: null
  weight: 0.001
outputs: [reconstruction, transfer_function]
simulate: null
solver:
  apodization_rolloff: 0.0
  calibration:
    advanced: null
    final_estimate: best_H
    score: evidence
    search: [{gradient: {max_iter: 150, method: split}}]
  device: null
  padding: {yx: 0, z: 0}
  reconstruction: closed_form
"""
SIM_DUMP = """\
version: 4
data: {input_channel_names: [GFP], pixel_size: {yx: 0.2, z: 0.4}, time_indices: all}
imaging_system:
  contrast: fluorescence
  field_dependence: shift_invariant
  index_of_refraction_media: 1.3
  detection:
    numerical_aperture: 1.2
    pinhole: none
    pupil_aberrations: {spherical: 0.08, zernike_up_to_order: null}
    wavelength: 0.532
    z_focus_offset: 0.0
  illumination:
    invert_phase_contrast: false
    numerical_aperture: null
    tilt: {azimuth: 0.0, zenith: 0.0}
    wavelength: null
noise: {gaussian: null, poisson: {gain: 1.0, offset: 0.0, photons: 50.0}, poisson_gaussian: null}
object: {constraints: [], dimension: 3, prior: tikhonov, source: {gaussian_spectrum: {exponent: 2.0}}, weight: 0.001}
outputs: [reconstruction, transfer_function]
simulate: {seed: 0, shape: [16, 32, 32]}
solver: {apodization_rolloff: 0.0, calibration: null, device: null, padding: {yx: 0, z: 0}, reconstruction: closed_form}
"""

PHASE = """\
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection: {numerical_aperture: 0.9}
  illumination: {wavelength: 0.532, numerical_aperture: 0.5}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 1.0}}
"""
PIXEL_SIZE = (0.4, 0.15, 0.15)


def _quiet(source: dict) -> ReconstructionModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfigurationWarning)
        return ReconstructionModel.from_config(source)


def test_a_wo40_bead_calibration_dump_builds_as_its_short_form():
    # A bead calibration never solves with the closed_form the dump writes, as when the key is left out.
    model, expected = _quiet(yaml.safe_load(BEAD_DUMP)), _quiet(yaml.safe_load(BEAD_SHORT))
    parts = ("system", "object_prior", "noise", "solver")
    assert [type(getattr(model, name)) for name in parts] == [type(getattr(expected, name)) for name in parts]


def test_a_wo40_poisson_simulation_dump_simulates_photon_counts():
    data, _ = _quiet(yaml.safe_load(SIM_DUMP)).simulate(seed=0)
    assert data.shape == (1, 16, 32, 32)
    assert float(data.min()) >= 0  # Poisson counts of photons: 50, as wo-4.0 drew them
    assert torch.equal(data, data.round())


def _phase(**blocks: object) -> dict:
    raw = yaml.safe_load(PHASE)
    for name, value in blocks.items():
        block, _, key = name.partition("__")
        if key:
            raw[block][key] = value
        else:
            raw[block] = value
    return raw


@pytest.mark.parametrize(
    ("blocks", "solver", "prior"),
    [
        ({"noise": {"poisson": {}}}, Wiener, Tikhonov),
        ({"noise": {"poisson": {"gain": 2.0}}, "solver": {"reconstruction": "closed_form"}}, Wiener, Tikhonov),
        ({"noise": {"poisson_gaussian": {"sigma": "estimate"}}}, Wiener, Tikhonov),
        (
            {"noise": {"poisson": {}}, "solver": {"calibration": None}, "object": {"prior": "none"}},
            None,
            None,
        ),
        (
            {
                "noise": {"poisson": {}},
                "object": {"prior": "tv", "weight": 1e-3},
                "solver": {"reconstruction": {"admm": {}}},
            },
            ADMM,
            TotalVariation,
        ),
    ],
    ids=["poisson", "poisson_written_closed_form", "poisson_gaussian", "prior_none_still_refused", "tv_admm"],
)
def test_wo40_rec_read_no_photon_noise_with_closed_form_tikhonov_or_admm_tv_and_solves_the_same(blocks, solver, prior):
    # wo-4.0's wo rec ran Tikhonov (closed_form) and TV (admm) without reading noise; prior none needed RL there.
    raw = _phase(**blocks)
    if solver is None:
        with pytest.raises(Rejected, match=r"solver\.reconstruction"):
            _quiet(raw)
        return
    kind = next(iter(raw["noise"]))
    with pytest.warns(ConfigurationWarning, match="nor was it by wo-4.0's wo rec") as caught:
        model = ReconstructionModel.from_config(raw)
    assert f"noise.{kind}" in {getattr(warning.message, "location", None) for warning in caught}
    assert (type(model.solver), type(model.object_prior), type(model.noise)) == (solver, prior, Gaussian)
    gaussian = _quiet(raw | {"noise": {"gaussian": {"sigma": 1.0}}})
    data = 1 + 0.01 * torch.rand(1, 8, 32, 32, generator=torch.Generator().manual_seed(0))
    estimate = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate
    assert torch.equal(estimate, gaussian.reconstruct(data, pixel_size=PIXEL_SIZE).estimate)


def test_a_photon_noise_block_under_an_evidence_calibration_is_still_refused():
    # wo-4.0's evidence path refused non-Gaussian noise (NotImplementedError), so nothing is solved without it here.
    raw = _phase(noise={"poisson": {}}, solver={"calibration": {"score": "evidence"}})
    raw["imaging_system"]["detection"]["numerical_aperture"] = {"nominal": 0.9, "tolerance": 0.05, "bounds": [0.8, 1.0]}
    with pytest.raises(Rejected, match=r"solver\.reconstruction"):
        _quiet(raw)


def test_the_denoiser_prior_loads_and_a_reconstruction_refuses_it():
    # wo-4.0 loaded object.prior: {denoiser: ...}; its wo rec refused it ("not runnable ... object.prior: denoiser").
    raw = _phase(object={"prior": {"denoiser": {"checkpoint": "x.pt"}}})
    with pytest.warns(ConfigurationWarning, match="a reconstruction refuses it"):
        model = ReconstructionModel.from_config(raw)
    data = torch.ones(1, 8, 32, 32)
    with pytest.raises(ConfigurationError) as refused:
        model.reconstruct(data, pixel_size=PIXEL_SIZE)
    assert refused.value.field == "object.prior.denoiser"
    assert "not in this version" in str(refused.value)
    with pytest.raises(ConfigurationError, match=r"object\.prior\.denoiser"):
        model.check((8, 32, 32), pixel_size=PIXEL_SIZE)


@pytest.mark.parametrize(
    "blocks",
    [
        {"object": {"prior": "none"}, "solver": {"reconstruction": "closed_form"}},
        {"noise": {"poisson": {}}, "solver": {"reconstruction": {"fista": {}}}},
        {"noise": {"poisson": {}}, "object": {"prior": "none"}, "solver": {"reconstruction": {"richardson_lucy": {}}}},
    ],
    ids=["closed_form_without_prior", "fista_poisson", "rl_on_phase"],
)
def test_a_refused_build_names_configuration_keys_not_python_arguments(blocks):
    with pytest.raises(Rejected) as rejected:
        _quiet(_phase(**blocks))
    (problem,) = rejected.value.problems
    for python in ("model.solver", "noise=", "object_prior=", "solver=", "NoPrior()"):
        assert python not in problem.message
    assert "leave solver.reconstruction out" in problem.message
    assert "object.prior: " in problem.message
    assert "noise: " in problem.message


def test_load_locates_the_warnings_of_a_mapping_at_the_file_it_names(tmp_path):
    raw = _phase()
    raw["data"]["pixel_size"] = {"z": 0.4, "yx": 0.15}  # as wo-4.0 writes it
    path = tmp_path / "wo40.yml"
    with pytest.warns(ConfigurationWarning) as caught:
        load(raw, path=path)
    found = [w.message for w in caught if isinstance(w.message, ConfigurationWarning)]
    (warning,) = [message for message in found if "{z, yx}" in str(message)]
    assert warning.location == str(path)
    assert f"wo config migrate {path}" in str(warning)
