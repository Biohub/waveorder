"""The nn.Module-per-distribution layer wraps the existing numerics exactly.

These are the equivalence/behavior tests of the WaveOrder 4.0 distribution
layer (``waveorder.distributions``). Each one pins the OO layer to the tested
function API it wraps, so the regression suite and the new layer cannot drift.
"""

import numpy as np
import torch
import yaml

from waveorder import distributions as D
from waveorder.calibrate.scores import calibration_penalty, whittle_evidence
from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.config.schema import V4Config
from waveorder.models import phase_thick_3d
from waveorder.reconstruct import tikhonov_regularized_inverse_filter

ZYX = (16, 32, 32)
OPTICS = dict(
    yx_pixel_size=0.2,
    z_pixel_size=0.4,
    wavelength_emission=0.532,
    index_of_refraction_media=1.3,
    numerical_aperture_detection=1.2,
)


def _fluorescence_system(coefficients=None, normalize="peak"):
    return D.Fluorescence3D(
        wavelength_emission=OPTICS["wavelength_emission"],
        index_of_refraction_media=OPTICS["index_of_refraction_media"],
        numerical_aperture_detection=OPTICS["numerical_aperture_detection"],
        zernike_coefficients=coefficients or {},
        normalize=normalize,
    )


def _sampling(shape=ZYX, z_padding=0):
    return D.Sampling(
        zyx_shape=shape,
        yx_pixel_size=OPTICS["yx_pixel_size"],
        z_pixel_size=OPTICS["z_pixel_size"],
        z_padding=z_padding,
    )


def _simulate(otf_truth, noise_sigma, regularization_strength, seed=0):
    """Draw g = H f + n with f from the Tikhonov prior S_k = sigma^2 / lambda."""
    generator = torch.Generator().manual_seed(seed)
    prior_std = noise_sigma / regularization_strength**0.5
    f = torch.randn(otf_truth.shape, generator=generator) * prior_std
    g = torch.real(torch.fft.ifftn(torch.fft.fftn(f) * otf_truth))
    g = g + torch.randn(otf_truth.shape, generator=generator) * noise_sigma
    return g.double()


# --- 1. evidence == whittle_evidence ---------------------------------------- #


def test_evidence_matches_whittle_evidence():
    s = _sampling()
    coefficients = {"astigmatism_0": 0.12, "spherical": -0.05}
    system = _fluorescence_system(coefficients)
    sigma, weight = 0.1, 1e-2
    g = torch.randn(ZYX, dtype=torch.float64)

    value = D.evidence(g, system, D.Tikhonov(weight), D.Gaussian(sigma), s)
    reference = whittle_evidence(
        data=g,
        otf=fluorescence_otf(zyx_shape=ZYX, zernike_coefficients=coefficients, **OPTICS),
        noise_sigma=sigma,
        regularization_strength=weight,
    )
    assert abs(float(value) - float(reference)) < 1e-10


def test_evidence_matches_whittle_evidence_exclude_dc():
    s = _sampling()
    system = _fluorescence_system({"coma_x": 0.07})
    sigma, weight = 0.2, 5e-3
    g = torch.randn(ZYX, dtype=torch.float64)

    value = D.evidence(g, system, D.Tikhonov(weight), D.Gaussian(sigma), s, exclude_dc=True)
    reference = whittle_evidence(
        data=g,
        otf=fluorescence_otf(zyx_shape=ZYX, zernike_coefficients={"coma_x": 0.07}, **OPTICS),
        noise_sigma=sigma,
        regularization_strength=weight,
        exclude_dc=True,
    )
    assert abs(float(value) - float(reference)) < 1e-10


# --- 2. Fluorescence3D.transfer == fluorescence_otf ------------------------- #


def test_fluorescence_transfer_matches_otf():
    s = _sampling()
    coefficients = {"astigmatism_0": 0.12, "spherical": -0.05, "defocus": 0.03}
    system = _fluorescence_system(coefficients)
    transfer = system.transfer(s)
    reference = fluorescence_otf(zyx_shape=ZYX, zernike_coefficients=coefficients, **OPTICS)
    assert torch.equal(transfer, reference)


def test_fluorescence_transfer_with_z_padding():
    s = _sampling(z_padding=2)
    system = _fluorescence_system({"spherical": 0.1})
    transfer = system.transfer(s)
    reference = fluorescence_otf(
        zyx_shape=(ZYX[0] + 4, ZYX[1], ZYX[2]),
        zernike_coefficients={"spherical": 0.1},
        **OPTICS,
    )
    assert torch.equal(transfer, reference)


# --- 3. ParameterPrior.nll == calibration_penalty -------------------------- #


def test_parameter_prior_matches_calibration_penalty():
    values = {"na": 1.25, "spherical": 0.1, "defocus": 0.5}
    nominals = {"na": 1.2, "spherical": 0.0, "defocus": 0.0}
    tolerances = {"na": 0.05, "spherical": 0.1, "defocus": "free"}
    prior = D.ParameterPrior(nominals, tolerances)
    assert torch.equal(prior.nll(values), calibration_penalty(values, nominals, tolerances))


def test_parameter_prior_from_system():
    system = _fluorescence_system({"spherical": D.free(0.0, 0.1), "astigmatism_0": 0.2})
    prior = D.ParameterPrior.from_system(system)
    assert prior.nominals == {"pupil_aberrations.spherical": 0.0}
    assert prior.tolerances == {"pupil_aberrations.spherical": 0.1}
    values = {"pupil_aberrations.spherical": 0.05}
    assert torch.equal(prior.nll(values), calibration_penalty(values, prior.nominals, prior.tolerances))


# --- 4. reconstruct (Tikhonov) == closed-form Wiener ----------------------- #


def test_reconstruct_matches_tikhonov_closed_form_phase():
    s = _sampling(shape=(12, 24, 24), z_padding=0)
    system = D.Phase3D(
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
    )
    sigma, weight = 0.1, 1e-2
    noise = D.Gaussian(sigma)
    obj = D.Tikhonov(weight)
    g = torch.randn(12, 24, 24, dtype=torch.float64)

    estimate = D.reconstruct(g, system, obj, noise, s)
    otf = system.transfer(s)
    regularization = float(noise.variance() / obj.spectrum(s, noise))
    inverse_filter = tikhonov_regularized_inverse_filter(otf, regularization)
    reference = torch.real(torch.fft.ifftn(torch.fft.fftn(g.to(torch.complex64)) * inverse_filter))
    assert torch.allclose(estimate, reference, atol=1e-6)


def test_reconstruct_tikhonov_regularization_is_sigma2_over_s_k():
    # S_k = sigma^2 / lambda, so sigma^2 / S_k = lambda exactly.
    s = _sampling()
    system = _fluorescence_system({"spherical": 0.1})
    sigma, weight = 0.3, 7e-3
    noise = D.Gaussian(sigma)
    obj = D.Tikhonov(weight)
    g = torch.randn(ZYX, dtype=torch.float64)

    estimate = D.reconstruct(g, system, obj, noise, s)
    inverse_filter = tikhonov_regularized_inverse_filter(system.transfer(s), weight)
    reference = torch.real(torch.fft.ifftn(torch.fft.fftn(g.to(torch.complex64)) * inverse_filter))
    assert torch.allclose(estimate, reference, atol=1e-6)


# --- 5. Phase3D.transfer == phase_thick_3d.calculate_transfer_function ----- #


def test_phase_transfer_matches_model():
    s = _sampling(shape=(8, 16, 16), z_padding=1)
    system = D.Phase3D(
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
        tilt_angle_zenith=0.05,
    )
    real_tf, imag_tf = system.transfer_real_imag(s)
    reference_real, reference_imag = phase_thick_3d.calculate_transfer_function(
        (8, 16, 16),
        0.2,
        0.4,
        0.532,
        1,
        1.3,
        0.9,
        1.2,
        tilt_angle_zenith=0.05,
    )
    assert torch.equal(real_tf, reference_real)
    assert torch.equal(imag_tf, reference_imag)
    # transfer() returns the phase (real-potential) WOTF as the single H_k.
    assert torch.equal(system.transfer(s), reference_real)


# --- 6. going blind flips an argument and calibrate recovers --------------- #


def test_going_blind_flips_one_argument_to_parameter():
    fixed = _fluorescence_system({"spherical": 0.0})
    assert fixed.free_parameter_names() == []
    blind = _fluorescence_system({"spherical": D.free(0.0, 0.5)})
    assert blind.free_parameter_names() == ["pupil_aberrations.spherical"]
    # The free value is an nn.Parameter; a plain number is a buffer.
    names = dict(blind.named_parameters())
    assert isinstance(names["pupil_aberrations.spherical"], torch.nn.Parameter)
    assert "pupil_aberrations.spherical" not in dict(fixed.named_parameters())


def test_calibrate_recovers_injected_aberration():
    # Only the magnitude of a lone astigmatism term is identifiable (the
    # PSF-flip sign ambiguity), matching test_synthesis_scores.py.
    s = _sampling()
    sigma, weight = 0.05, 1e-2
    truth = 0.2
    otf_truth = fluorescence_otf(zyx_shape=ZYX, zernike_coefficients={"astigmatism_0": truth}, **OPTICS)
    g = _simulate(otf_truth, sigma, weight, seed=3)

    system = _fluorescence_system({"astigmatism_0": D.free(0.0, 0.5)})
    prior = D.ParameterPrior.from_system(system)
    result = D.calibrate(
        g, system, prior, D.Tikhonov(weight), D.Gaussian(sigma), s, max_iter=200, lr=0.02, exclude_dc=True
    )
    recovered = result.parameters["pupil_aberrations.astigmatism_0"]
    assert abs(abs(recovered) - truth) < 0.11, result.parameters
    assert result.covariance.shape == (1, 1)
    assert np.isfinite(result.covariance).all()


# --- 7. state_dict round-trip, .to, dotted tf.zarr names ------------------- #


def test_state_dict_round_trip():
    s = _sampling()
    system = _fluorescence_system({"spherical": D.free(0.17, 0.1), "astigmatism_0": 0.05})
    restored = _fluorescence_system({"spherical": D.free(0.0, 0.1), "astigmatism_0": 0.05})
    restored.load_state_dict(system.state_dict())
    assert torch.equal(system.transfer(s), restored.transfer(s))


def test_to_device_cpu():
    system = _fluorescence_system({"spherical": D.free(0.1, 0.1)})
    moved = system.to("cpu")
    assert moved is system
    s = _sampling()
    assert moved.transfer(s).shape == ZYX


def test_dotted_names_usable_as_tf_zarr_parameter_names():
    import tempfile

    from waveorder.calibrate.tfzarr import read_tf_zarr, write_tf_zarr

    system = _fluorescence_system({"astigmatism_0": D.free(0.1, 0.05), "spherical": D.free(-0.02, 0.05)})
    names = system.free_parameter_names()
    assert names == ["pupil_aberrations.astigmatism_0", "pupil_aberrations.spherical"]
    values = np.array([[float(p.detach()) for _, p in system.named_parameters()]])
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/tf.zarr"
        write_tf_zarr(path, values, names, tile_grid={"size_yx": [32, 32]})
        store = read_tf_zarr(path)
    assert store.parameter_names == names
    assert np.allclose(store.parameters, values)


# --- config adapter -------------------------------------------------------- #


def test_build_system_from_config_fluorescence():
    cfg = V4Config(
        **yaml.safe_load(
            """
version: 4
data:
  pixel_size: {yx: 0.2, z: 0.4}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.532
    numerical_aperture: 1.2
    pupil_aberrations:
      spherical: {nominal: 0.0, tolerance: 0.1}
      astigmatism_0: 0.05
object: {prior: tikhonov, weight: 0.01}
noise: {gaussian: {sigma: 0.1}}
"""
        )
    )
    system, obj, noise, prior, sampling = D.build_system_from_config(cfg)
    assert isinstance(system, D.Fluorescence3D)
    assert system.free_parameter_names() == ["pupil_aberrations.spherical"]
    assert isinstance(obj, D.Tikhonov)
    assert isinstance(noise, D.Gaussian)
    assert prior.tolerances == {"pupil_aberrations.spherical": 0.1}
    assert float(obj.weight) == 0.01
    assert float(noise.sigma) == 0.1


def test_build_system_from_config_phase_free_na():
    cfg = V4Config(
        **yaml.safe_load(
            """
version: 4
data:
  pixel_size: {yx: 0.2, z: 0.4}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.3
  detection: {numerical_aperture: 1.2}
  illumination:
    wavelength: 0.532
    numerical_aperture: {nominal: 0.9, tolerance: 0.05}
object: {prior: tikhonov, weight: 0.001}
noise: {gaussian: {sigma: 1.0}}
"""
        )
    )
    system, _, _, prior, _ = D.build_system_from_config(cfg)
    assert isinstance(system, D.Phase3D)
    assert system.free_parameter_names() == ["numerical_aperture_illumination"]
    assert prior.tolerances == {"numerical_aperture_illumination": 0.05}


# --- stubs / Poisson / FittedSpectrum -------------------------------------- #


def test_fitted_spectrum_requires_data_and_otf():
    import pytest

    s = _sampling()
    with pytest.raises(ValueError):
        D.FittedSpectrum().spectrum(s, D.Gaussian(0.1))


def test_fitted_spectrum_matches_reference_shell_fit():
    """The |H|^2-weighted shell fit matches a direct scatter-add reference."""
    s = _sampling()
    system = _fluorescence_system({"astigmatism_0": 0.1})
    otf = system.transfer(s)
    g = torch.randn(ZYX, dtype=torch.float64)
    noise = D.Gaussian(0.1)
    prior = D.FittedSpectrum(n_shells=24)

    data_fft = torch.fft.fftn(g, norm="ortho")
    s_k = prior.spectrum(s, noise, data_fft=data_fft, otf=otf)

    # direct reference: S_shell = sum h2 (P - sigma^2) / sum h2^2, clamped
    shell, _ = s.shell_index(24)
    shell = shell.reshape(-1)
    power = (data_fft.abs() ** 2).reshape(-1).to(torch.float64)
    h2 = (otf.abs() ** 2).reshape(-1).to(torch.float64)
    sigma2 = noise.variance().to(torch.float64)
    num = torch.zeros(24, dtype=torch.float64)
    den = torch.zeros(24, dtype=torch.float64)
    num.scatter_add_(0, shell, h2 * torch.clamp(power - sigma2, min=0.0))
    den.scatter_add_(0, shell, h2 * h2)
    ref_shell = (num / den.clamp_min(1e-30)).clamp_min(1e-10)
    assert torch.allclose(s_k.reshape(-1), ref_shell[shell], atol=0, rtol=0)
    # S_k is positive everywhere (keeps the Gaussian evidence / Wiener finite)
    assert (s_k > 0).all()


def test_evidence_runs_with_fitted_spectrum():
    """evidence() auto-profiles the data for a data-profiled prior."""
    s = _sampling()
    system = _fluorescence_system({"astigmatism_0": 0.1})
    g = torch.randn(ZYX, dtype=torch.float64)
    value = D.evidence(g, system, D.FittedSpectrum(n_shells=24), D.Gaussian(0.1), s)
    assert torch.isfinite(value)


def test_fitted_spectrum_beats_flat_tikhonov_on_structured_object():
    """On an object with a non-flat (power-law) spectrum, the data-profiled
    FittedSpectrum gives a higher evidence (lower NLL) at the true system than
    a mis-specified flat Tikhonov prior, i.e. it fits the object spectrum the
    flat prior gets wrong. This is the winning-strategy property."""
    s = _sampling()
    system = _fluorescence_system({"astigmatism_0": 0.15, "coma_x": 0.1})
    otf = system.transfer(s)

    # simulate g = H f + n with a STRUCTURED (power-law) object spectrum
    gen = torch.Generator().manual_seed(3)
    fz = np.fft.fftfreq(ZYX[0])
    fy = np.fft.fftfreq(ZYX[1])
    fx = np.fft.fftfreq(ZYX[2])
    kr = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    s_true = torch.as_tensor(1.0 / (kr**2 + 1e-2), dtype=torch.float64)  # red spectrum
    f = torch.randn(ZYX, dtype=torch.float64, generator=gen) * s_true.sqrt()
    g = torch.real(torch.fft.ifftn(torch.fft.fftn(f) * otf))
    g = g + torch.randn(ZYX, dtype=torch.float64, generator=gen) * 0.05

    nll_flat = D.evidence(g, system, D.Tikhonov(1e-2), D.Gaussian(0.05), s)
    nll_fit = D.evidence(g, system, D.FittedSpectrum(n_shells=32), D.Gaussian(0.05), s)
    assert nll_fit < nll_flat  # data-profiled spectrum explains the data better


def test_gaussian_noise_sample_and_variance():
    noise = D.Gaussian(0.5)
    assert abs(float(noise.variance()) - 0.25) < 1e-12
    generator = torch.Generator().manual_seed(0)
    draw = noise.sample(torch.zeros(5000, dtype=torch.float64), generator=generator)
    assert abs(float(draw.std()) - 0.5) < 0.05


def test_poisson_sample_mean():
    noise = D.Poisson(gain=1.0)
    generator = torch.Generator().manual_seed(0)
    draw = noise.sample(torch.full((5000,), 10.0, dtype=torch.float64), generator=generator)
    assert abs(float(draw.mean()) - 10.0) < 0.3
