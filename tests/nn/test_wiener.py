"""Wiener, evidence and the posterior mean agree with dense-matrix references on small grids, for odd and even X."""

import math

import pytest
import torch

from tests.nn.conftest import MatrixSystem, make_system, random_transfer
from waveorder.errors import IncompatibleModelError, ShapeError
from waveorder.nn import Evidence, Gaussian, ImagingModel, Sampling, Tikhonov, Wiener, ZernikeAberration
from waveorder.nn.operators import LinearOperator
from waveorder.nn.solvers.wiener import LinearGaussian

WEIGHT, SIGMA = 0.25, 0.125  # exact in float32, so the float32-registered quantities equal the dense reference's

CASES = {
    "fluorescence_even": (
        lambda: make_system(1.2, z_focus_offset=0.7, pupil_aberrations=ZernikeAberration({"coma_x": 0.1})),
        Sampling(shape=(4, 6, 8), pixel_size=(0.4, 0.15, 0.15)),
    ),
    "fluorescence_odd": (
        lambda: make_system(1.1, z_focus_offset=-0.5, pupil_aberrations=ZernikeAberration({"coma_y": 0.08})),
        Sampling(shape=(5, 5, 7), pixel_size=(0.4, 0.12, 0.15)),
    ),
    "two_fields_three_channels": (
        lambda: MatrixSystem(random_transfer(3, 2, (4, 4, 6), dtype=torch.float64), fields=("a", "b")),
        Sampling(shape=(4, 4, 6), pixel_size=(0.3, 0.1, 0.1)),
    ),
    "two_fields_two_channels_odd": (
        lambda: MatrixSystem(random_transfer(2, 2, (3, 4, 5), seed=3, dtype=torch.float64), fields=("a", "b")),
        Sampling(shape=(3, 4, 5), pixel_size=(0.3, 0.1, 0.1)),
    ),
}


def _dense(system, sampling) -> torch.Tensor:
    """The real matrix of H, columns = H applied to unit objects."""
    n = math.prod(system.object_shape(sampling))
    basis = torch.eye(n, dtype=torch.float64).reshape(n, *system.object_shape(sampling))
    with torch.no_grad():
        return torch.stack([system(b, sampling).flatten() for b in basis], dim=1)


def _case(name):
    build, sampling = CASES[name]
    model = ImagingModel(build(), Tikhonov(WEIGHT), Gaussian(SIGMA)).double()
    data = torch.randn(
        model.system.data_shape(sampling), generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    return model, sampling, data, _dense(model.system, sampling)


@pytest.mark.parametrize("name", CASES)
def test_wiener_equals_the_dense_map_estimate(name):
    model, sampling, data, A = _case(name)
    expected = torch.linalg.solve(A.T @ A + WEIGHT * torch.eye(A.shape[1], dtype=A.dtype), A.T @ data.flatten())
    result = Wiener()(data, model, sampling)
    torch.testing.assert_close(result.estimate.flatten(), expected, rtol=1e-8, atol=1e-10)
    assert result.status == "direct"


def _linear_gaussian(model, sampling) -> LinearGaussian:
    return LinearGaussian.of(model, model.system.bind(sampling))


@pytest.mark.parametrize("name", CASES)
def test_evidence_equals_the_dense_gaussian_likelihood_without_the_zero_frequency(name):
    model, sampling, data, A = _case(name)
    prior_variance = SIGMA**2 / WEIGHT
    covariance = prior_variance * A @ A.T + SIGMA**2 * torch.eye(A.shape[0], dtype=A.dtype)
    dense = -torch.distributions.MultivariateNormal(torch.zeros(A.shape[0], dtype=A.dtype), covariance).log_prob(
        data.flatten()
    )
    # The zero frequency's term, with its share C log(2 pi) / 2 of the normalization.
    channels = model.system.data_shape(sampling)[0]
    zero = _linear_gaussian(model, sampling).evidence_terms(data)[0, 0, 0] + 0.5 * channels * math.log(2 * math.pi)
    torch.testing.assert_close(Evidence()(data, model, sampling), dense - zero, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", CASES)
def test_evidence_ignores_the_zero_frequency_of_the_data(name):
    # A constant added to each channel moves only the zero-frequency coefficient, which Evidence leaves out.
    model, sampling, data, _ = _case(name)
    offsets = torch.arange(1.0, data.shape[0] + 1, dtype=data.dtype).reshape(-1, 1, 1, 1)
    shifted = data + 10.0 * offsets
    terms = _linear_gaussian(model, sampling)
    assert not torch.allclose(terms.evidence_terms(shifted)[0, 0, 0], terms.evidence_terms(data)[0, 0, 0])
    torch.testing.assert_close(
        Evidence()(shifted, model, sampling), Evidence()(data, model, sampling), rtol=1e-12, atol=1e-9
    )


@pytest.mark.parametrize("name", CASES)
def test_posterior_mean_equals_the_dense_posterior_mean(name):
    model, sampling, data, A = _case(name)
    precision = A.T @ A / SIGMA**2 + torch.eye(A.shape[1], dtype=A.dtype) * WEIGHT / SIGMA**2
    expected = torch.linalg.solve(precision, A.T @ data.flatten() / SIGMA**2)
    mean = _linear_gaussian(model, sampling).posterior(data).mean()
    torch.testing.assert_close(mean.flatten(), expected, rtol=1e-8, atol=1e-10)


class _Generic(LinearOperator):
    """Another operator's actions without its Fourier structure, as a shift-variant system binds."""

    def __init__(self, inner: LinearOperator) -> None:
        super().__init__(inner.sampling, object_shape=inner.object_shape, data_shape=inner.data_shape)
        self.inner = inner

    @property
    def placement(self) -> tuple[torch.dtype, torch.device]:
        return self.inner.placement

    def apply(self, obj: torch.Tensor) -> torch.Tensor:
        return self.inner.apply(obj)

    def adjoint(self, data: torch.Tensor) -> torch.Tensor:
        return self.inner.adjoint(data)

    def norm(self) -> torch.Tensor:
        return self.inner.norm()


def test_linear_gaussian_needs_a_fourier_diagonal_operator():
    model, sampling, data, _ = _case("two_fields_three_channels")
    generic = _Generic(model.system.bind(sampling))
    with pytest.raises(IncompatibleModelError, match="needs a Fourier-diagonal operator, got _Generic"):
        LinearGaussian.of(model, generic)
    # Wiener refuses it sooner: the system declared a Fourier-diagonal operator.
    with pytest.raises(IncompatibleModelError, match="Wiener operator is a _Generic, but MatrixSystem declares"):
        Wiener()(data, model, sampling, operator=generic)


def test_data_in_another_dtype_than_the_model_is_rejected_before_solving():
    model, sampling, data, _ = _case("fluorescence_even")
    with pytest.raises(ShapeError, match="float32"):
        Wiener()(data.float(), model, sampling)


def test_objective_rejects_an_operator_bound_to_another_sampling():
    model, sampling, data, _ = _case("fluorescence_even")
    other = Sampling(shape=sampling.shape, pixel_size=(0.4, 0.2, 0.2))
    estimate = torch.zeros(model.system.object_shape(sampling), dtype=torch.float64)
    with pytest.raises(ShapeError, match="bound to"):
        model.objective(data, estimate, sampling, operator=model.system.bind(other))


def test_in_focus_otf_matches_the_analytic_disk_autocorrelation():
    # Independent of the code under test: the incoherent in-focus OTF of a uniform circular pupil of radius NA/lambda
    # is (2/pi)(acos(v) - v sqrt(1 - v^2)) with v = |k| / (2 NA / lambda).
    sampling = Sampling(shape=(16, 256, 256), pixel_size=(0.25, 0.05, 0.05))
    system = make_system(1.0).double()
    assert system.upsampling(sampling) == (1, 1, 1)  # so the transfer is not cropped and z = 0 is a sampled plane
    in_focus = system.transfer(sampling)[0, 0].mean(dim=0)  # IFFT along z evaluated at z = 0
    along_x = (in_focus[0] / in_focus[0, 0]).real
    frequencies = torch.fft.rfftfreq(256, d=0.05, dtype=torch.float64)  # the half spectrum's x frequencies
    v = frequencies / (2 * 1.0 / 0.52)
    inside = (v >= 0) & (v < 0.9)
    analytic = (2 / math.pi) * (torch.acos(v[inside]) - v[inside] * torch.sqrt(1 - v[inside] ** 2))
    torch.testing.assert_close(along_x[inside], analytic, atol=0.02, rtol=0)
