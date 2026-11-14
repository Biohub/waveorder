"""Noise models sample what they claim, and the parameter prior penalizes and clamps."""

import pytest
import torch
from torch import nn

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn import (
    CalibratableModule,
    Detection,
    Free,
    Gaussian,
    Noise,
    NoPrior,
    ParameterPrior,
    Poisson,
    Tikhonov,
)
from waveorder.testing import assert_noise_gradient


@pytest.mark.parametrize("noise", [Gaussian(0.3), Poisson(gain=2.0, offset=10.0)], ids=["gaussian", "poisson"])
def test_samples_match_the_declared_variance(noise: Noise):
    mean = torch.full((200_000,), 50.0)
    data = noise.sample(mean, generator=torch.Generator().manual_seed(0))
    offset = noise.offset if isinstance(noise, Poisson) else 0.0
    assert (data - offset).mean() == pytest.approx(50.0, rel=0.01)
    assert data.var() == pytest.approx(noise.variance(mean)[0].item(), rel=0.02)


def test_poisson_likelihood_is_finite_below_the_offset():
    # Read noise puts camera values below the offset; they count as zero photons, not as an undefined likelihood.
    noise = Poisson(gain=2.0, offset=100.0)
    generator = torch.Generator().manual_seed(0)
    data = 100.0 + 20.0 * torch.randn(256, generator=generator, dtype=torch.float64)
    assert bool((data < 100.0).any())
    mean = 1.0 + 10.0 * torch.rand(256, generator=generator, dtype=torch.float64)
    assert torch.isfinite(noise.nll(data, mean))
    below = data < 100.0
    torch.testing.assert_close(noise.gradient(data, mean)[below], torch.full_like(mean[below], 1 / 2.0))
    assert_noise_gradient(noise, data, mean)


class _Read(Gaussian):
    """A subclass of Gaussian noise (say, with a different sampler) inherits its capability."""


def test_noise_reports_a_white_gaussian_variance_only_when_it_has_one():
    variance = _Read(0.3).white_gaussian_variance()
    assert variance is not None
    assert variance.item() == pytest.approx(0.09)
    assert Poisson(gain=2.0).white_gaussian_variance() is None


def test_only_noise_declaring_photon_counts_reports_them():
    data = torch.tensor([90.0, 100.0, 107.0], dtype=torch.float64)
    counts = Poisson(gain=2.0, offset=100.0).photon_counts(data)
    torch.testing.assert_close(counts, torch.tensor([0.0, 0.0, 7.0], dtype=torch.float64))  # signal above offset
    with pytest.raises(IncompatibleModelError, match="has_photon_counts"):
        Gaussian(0.3).photon_counts(data)
    with pytest.raises(IncompatibleModelError, match="has_curvature_bound"):
        Poisson(gain=2.0).curvature_bound()


def test_fixed_quantities_are_exact_after_double_and_loaded_values_survive():
    detection = Detection(wavelength=0.5, numerical_aperture=1.2, z_focus_offset=0.3).double()
    assert detection.numerical_aperture.dtype == torch.float64
    assert detection.numerical_aperture.item() == 1.2  # not 1.2000000476837158, the float32 rounding
    assert detection.float().double().z_focus_offset.item() == 0.3
    loaded = Detection(wavelength=0.5, numerical_aperture=1.2)
    loaded.load_state_dict({**loaded.state_dict(), "numerical_aperture": torch.tensor(0.9)})
    assert loaded.double().numerical_aperture.item() == pytest.approx(0.9)


def test_only_the_flat_prior_reports_flat():
    assert NoPrior().is_flat
    assert NoPrior(nonnegative=True).is_flat
    assert not Tikhonov(1.0).is_flat


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nominal": 1.0, "tolerance": 0.0}, "tolerance"),
        ({"nominal": 1.0, "bounds": (2.0, 1.0)}, "low < high"),
        ({"nominal": 3.0, "bounds": (0.0, 1.0)}, "outside bounds"),
        ({"nominal": float("nan")}, "finite"),
        ({"nominal": 1.0, "bounds": (0.0, float("inf"))}, "finite"),
    ],
)
def test_invalid_free_values_rejected(kwargs, match):
    with pytest.raises(ParameterError, match=match):
        Free(**kwargs)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), Free(0.5), Free(0.5, bounds=(0.0, 1.0))],
    ids=["zero", "negative", "nan", "free_unbounded", "free_zero_bound"],
)
def test_positive_quantities_cannot_reach_zero(value):
    # Calibration clamps to bounds, so a zero lower bound would let sigma or a weight become 0 and divide by it.
    with pytest.raises(ParameterError, match=r"Gaussian\.sigma must"):
        Gaussian(value)


def test_flat_parameter_prior_follows_the_parameters_dtype():
    prior = ParameterPrior.from_module(Tikhonov(Free(1.0, bounds=(0.1, 2.0))))
    assert prior.nll({"weight": torch.tensor(1.5, dtype=torch.float64)}).dtype == torch.float64


class _Two(CalibratableModule):
    def __init__(self) -> None:
        super().__init__()
        self.register_quantity("a", Free(1.0, tolerance=0.5, bounds=(0.0, 2.0)))
        self.register_quantity("b", Free(0.0))  # flat prior, no bounds
        self.register_quantity("c", 3.0)  # fixed


def test_prior_penalizes_only_toleranced_quantities():
    prior = ParameterPrior.from_module(_Two())
    assert list(prior.specs) == ["a", "b"]
    params = {"a": torch.tensor(2.0), "b": torch.tensor(100.0)}
    assert prior.nll(params).item() == pytest.approx(0.5 * (1.0 / 0.5) ** 2)


def test_clamp_in_place_respects_bounds():
    module = _Two()
    prior = ParameterPrior.from_module(module)
    with torch.no_grad():
        module.get_parameter("a").fill_(5.0)
    prior.clamp_(module)
    assert module.get_parameter("a").item() == 2.0
    assert [name for name, _ in module.named_parameters()] == ["a", "b"]
    assert isinstance(module.get_buffer("c"), torch.Tensor)
    assert not isinstance(module.get_buffer("c"), nn.Parameter)


def test_the_sigma_estimators_are_the_wo40_starts():
    from tests.nn.wo40_fixtures import sweep_volume
    from waveorder.nn.noise import estimate_sigma

    data = sweep_volume()
    # wo-4.0 evidence path: data.std(); bead path: 1.4826 * median|g - median g| (float32 data).
    assert float(estimate_sigma(data)) == pytest.approx(17.058837890625, rel=1e-7)
    assert float(estimate_sigma(data, method="mad")) == pytest.approx(6.2940144538879395, rel=1e-7)
    assert float(estimate_sigma(torch.ones(10), method="mad")) == pytest.approx(1e-6)


def test_an_estimated_sigma_starts_from_the_data():
    noise = Gaussian("estimate")
    assert noise.estimated
    assert "sigma" in noise.free_quantities()
    data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    noise.double().start_from(data)
    scale = float(data.std())
    assert float(noise.sigma.detach()) == pytest.approx(scale)
    assert noise.free_quantities()["sigma"] == Free(scale, bounds=(1e-3 * scale, scale))
    fixed = Gaussian(0.5)
    fixed.start_from(data)
    assert float(fixed.sigma) == 0.5
    assert not fixed.free_quantities()
