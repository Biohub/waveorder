"""Zernike evaluation follows the orthonormal Noll convention; half-spectrum helpers describe real volumes.

The per-frequency algebra runs on real storage; every check compares it with the same algebra on complex tensors.
"""

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from waveorder.errors import IncompatibleModelError, ShapeError
from waveorder.nn import Sampling, functional
from waveorder.nn.functional import (
    apply_adjoint,
    apply_transfer,
    channel_covariance,
    cholesky_hermitian,
    cholesky_logdet,
    cholesky_solve,
    complex_multiply,
    crop_half_spectrum,
    from_half_spectrum,
    gram,
    half_spectrum_frequencies,
    half_spectrum_weights,
    hermitian_half,
    hermitian_inverse,
    irfftn,
    largest_singular_value,
    matrix_vector,
    widefield_otf,
    zernike,
)
from waveorder.nn.parameters import placement
from waveorder.nn.systems._checks import expect_feasible_grid, expect_resolved_aperture
from waveorder.zernike_names import name_to_noll, noll_indices_up_to_order


@pytest.fixture(scope="module")
def disk():
    n = 512
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, dtype=torch.float64), torch.linspace(-1, 1, n, dtype=torch.float64), indexing="ij"
    )
    rho, theta = torch.hypot(x, y), torch.atan2(y, x)
    return rho, theta, rho <= 1


def test_order_4_modes_are_orthonormal(disk):
    rho, theta, inside = disk
    modes = torch.stack([zernike(j, rho, theta)[inside] for j in noll_indices_up_to_order(4)])
    gram_matrix = modes @ modes.T / inside.sum()
    assert torch.allclose(gram_matrix, torch.eye(len(modes), dtype=gram_matrix.dtype), atol=0.02)


def test_orientation_conventions():
    """x-named terms are cosine terms (peak at theta = 0), y-named are sine."""
    rho = torch.tensor([1.0])

    def at(name, theta):
        return float(zernike(name_to_noll(name), rho, torch.tensor([theta])))

    assert at("coma_x", 0.0) > 1.0
    assert abs(at("coma_x", math.pi / 2)) < 1e-5
    assert at("coma_y", math.pi / 2) > 1.0
    assert abs(at("coma_y", 0.0)) < 1e-5
    assert at("astigmatism_0", 0.0) > 1.0
    assert at("astigmatism_0", math.pi / 2) < -1.0
    assert abs(at("astigmatism_45", 0.0)) < 1e-5
    assert at("z15", math.pi / 8) > 1.0


def _random_volume(shape, *, seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _half(volume):
    return torch.view_as_real(torch.fft.rfftn(volume))


def _low(full, size):
    return [*range((size + 1) // 2), *range(full - size // 2, full)]


def test_crop_half_spectrum_keeps_the_low_frequencies_of_the_full_spectrum():
    fine, shape = (7, 10, 9), (4, 6, 6)
    volume = _random_volume(fine)
    full = torch.fft.fftn(volume)
    expected = full[_low(fine[0], shape[0])][:, _low(fine[1], shape[1])][..., : shape[2] // 2 + 1]
    torch.testing.assert_close(crop_half_spectrum(_half(volume), shape), torch.view_as_real(expected))


def test_crop_half_spectrum_does_not_copy_an_unchanged_grid_and_rejects_finer_grids():
    spectrum = _half(_random_volume((4, 6, 8)))
    unchanged = crop_half_spectrum(spectrum, (4, 6, 8))
    assert unchanged.data_ptr() == spectrum.data_ptr()
    assert torch.equal(unchanged, spectrum)
    with pytest.raises(ShapeError, match="cannot crop 6 frequencies to 7"):
        crop_half_spectrum(spectrum, (4, 7, 8))
    with pytest.raises(ShapeError, match="half spectrum of 5 columns"):
        crop_half_spectrum(spectrum, (4, 6, 10))


@pytest.mark.parametrize("nx", [7, 8])
def test_half_spectrum_weights_recover_full_spectrum_sums(nx):
    volume = _random_volume((3, 4, nx))
    full = torch.fft.fftn(volume).abs().square().sum()
    half = torch.fft.rfftn(volume).abs().square()
    weights = half_spectrum_weights(nx, device=half.device, dtype=half.dtype)
    assert weights.shape == (nx // 2 + 1,)
    torch.testing.assert_close((half * weights).sum(), full)


def _irfftn(spectrum, shape):
    return torch.fft.irfftn(torch.view_as_complex(spectrum.contiguous()), s=shape)


def test_hermitian_half_is_the_projection_irfftn_uses_on_a_cropped_even_grid():
    # Cropping 9 -> 6 along y keeps the Nyquist row y = -3 without its partner y = +3, so the self-conjugate columns
    # x = 0 and x = nx / 2 are no longer Hermitian.
    shape = (4, 6, 8)
    spectrum = crop_half_spectrum(_half(_random_volume((5, 9, 8))), shape)
    projected = hermitian_half(spectrum, shape[-1])
    assert not torch.allclose(projected, spectrum), "the crop must leave non-Hermitian columns"
    real = _irfftn(spectrum, shape)
    torch.testing.assert_close(_irfftn(projected, shape), real)
    torch.testing.assert_close(_half(real), projected)  # it is exactly the spectrum of a real volume
    torch.testing.assert_close(hermitian_half(projected, shape[-1]), projected)  # and a projection


def test_hermitian_half_of_a_real_spectrum_makes_its_self_conjugate_columns_even():
    shape = (4, 6, 8)
    spectrum = crop_half_spectrum(_half(_random_volume((5, 9, 8))), shape)[..., 0]
    projected = hermitian_half(spectrum, shape[-1], real=True)
    as_complex = torch.stack((spectrum, torch.zeros_like(spectrum)), dim=-1)
    torch.testing.assert_close(projected, hermitian_half(as_complex, shape[-1])[..., 0])
    flipped = torch.roll(torch.flip(projected[..., 0], dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
    torch.testing.assert_close(flipped, projected[..., 0])


def test_hermitian_half_leaves_odd_columns_but_x_zero_alone():
    shape = (4, 6, 7)
    spectrum = crop_half_spectrum(_half(_random_volume((5, 9, 7))), shape)
    projected = hermitian_half(spectrum, shape[-1])
    torch.testing.assert_close(projected[..., 1:, :], spectrum[..., 1:, :])
    torch.testing.assert_close(_irfftn(projected, shape), _irfftn(spectrum, shape))


def test_hermitian_half_batches_under_vmap_and_passes_gradients():
    # The projection is real-linear and self-adjoint, so the gradient of <w, P s> is P w: w itself on the interior
    # columns. A broken out-of-place rewrite would drop or detach columns.
    shape = (4, 6, 8)
    batch = torch.stack([crop_half_spectrum(_half(_random_volume((5, 9, 8), seed=s)), shape) for s in (1, 2)])
    batched = torch.vmap(lambda s: hermitian_half(s, shape[-1]))(batch)
    torch.testing.assert_close(batched, torch.stack([hermitian_half(s, shape[-1]) for s in batch]))
    spectrum = batch[0].clone().requires_grad_()
    weights = crop_half_spectrum(_half(_random_volume((5, 9, 8), seed=3)), shape)
    (hermitian_half(spectrum, shape[-1]) * weights).sum().backward()
    assert spectrum.grad is not None
    torch.testing.assert_close(spectrum.grad[..., 1:4, :], weights[..., 1:4, :])
    torch.testing.assert_close(spectrum.grad[..., [0, 4], :], hermitian_half(weights, shape[-1])[..., [0, 4], :])


GRID = (2, 3, 4)  # (Z, Y, X') of the per-frequency toys


def _complex_matrices(n, m, *, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(*(torch.randn(n, m, *GRID, generator=generator, dtype=torch.float64) for _ in range(2)))


def _dense(matrices):
    return matrices.movedim((0, 1), (-2, -1))


def _hermitian(n, *, seed=0):
    """Random Hermitian positive-definite matrices, complex per-frequency ``(n, n, *GRID)``."""
    raw = _dense(_complex_matrices(n, n, seed=seed))
    return (raw @ raw.mH + torch.eye(n, dtype=raw.dtype)).movedim((-2, -1), (0, 1))


def _stored(matrices, *, real):
    return matrices.real if real else torch.view_as_real(matrices)


def _restored(stored):
    """Per-frequency matrices back to complex: six dimensions carry an imaginary part, five are real."""
    return torch.view_as_complex(stored.contiguous()) if stored.dim() == 6 else stored.to(torch.complex128)


def _vector(stored):
    return torch.view_as_complex(stored.contiguous())


def _algebra(matrices, vector, data, spectrum, variance):
    return (
        matrix_vector(matrices, vector),
        matrix_vector(matrices, data, adjoint=True),
        gram(matrices),
        channel_covariance(matrices, spectrum, variance),
        complex_multiply(vector, data[:1]),
    )


@pytest.mark.parametrize("real", [False, True], ids=["complex", "real"])
@pytest.mark.parametrize(("n", "m"), [(1, 1), (3, 2), (2, 3)])
def test_real_storage_algebra_matches_complex_arithmetic(n, m, real):
    reference = _complex_matrices(n, m)
    if real:
        reference = reference.real.to(reference.dtype)
    matrices = _stored(reference, real=real)
    vector = torch.view_as_real(_complex_matrices(m, 1, seed=1)[:, 0])
    data = torch.view_as_real(_complex_matrices(n, 1, seed=2)[:, 0])
    spectrum = torch.rand(m, *GRID, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    variance = torch.tensor(0.3, dtype=torch.float64)
    applied, adjoint, normal, covariance, elementwise = _algebra(matrices, vector, data, spectrum, variance)
    torch.testing.assert_close(_vector(applied), torch.einsum("nm...,m...->n...", reference, _vector(vector)))
    torch.testing.assert_close(_vector(adjoint), torch.einsum("nm...,n...->m...", reference.conj(), _vector(data)))
    torch.testing.assert_close(_restored(normal), torch.einsum("cf...,cg...->fg...", reference.conj(), reference))
    expected = torch.einsum("cf...,df...->cd...", reference * spectrum, reference.conj())
    expected = expected + variance * torch.eye(n, dtype=expected.dtype).reshape(n, n, 1, 1, 1)
    torch.testing.assert_close(_restored(covariance), expected)
    torch.testing.assert_close(_vector(elementwise), _vector(vector) * _vector(data[:1]))
    singular = torch.linalg.matrix_norm(_dense(reference), ord=2).amax()
    torch.testing.assert_close(largest_singular_value(matrices), singular)


@pytest.mark.parametrize("real", [False, True], ids=["complex", "real"])
@pytest.mark.parametrize("n", [1, 3])
def test_one_cholesky_factor_gives_solve_logdet_and_inverse(n, real):
    reference = _hermitian(n)
    if real or n == 1:  # a 1 x 1 Hermitian matrix is real
        reference = reference.real.to(reference.dtype)
    matrices = _stored(reference, real=real or n == 1)
    factor = cholesky_hermitian(matrices)
    assert factor.embedded == (not real and n > 1)
    dense = _dense(reference)
    rhs = torch.view_as_real(_complex_matrices(n, 1, seed=1)[:, 0])
    expected = torch.linalg.solve(dense, _vector(rhs).movedim(0, -1).unsqueeze(-1)).squeeze(-1).movedim(-1, 0)
    torch.testing.assert_close(_vector(cholesky_solve(factor, rhs)), expected)
    torch.testing.assert_close(cholesky_logdet(factor), torch.linalg.slogdet(dense).logabsdet)
    inverse = _restored(hermitian_inverse(matrices)) if matrices.dim() == 6 else hermitian_inverse(matrices)
    torch.testing.assert_close(_dense(inverse).to(dense.dtype), torch.linalg.inv(dense))


@pytest.mark.parametrize("real", [False, True], ids=["complex", "real"])
def test_apply_and_adjoint_match_complex_ffts(real):
    shape = (4, 6, 8)
    generator = torch.Generator().manual_seed(6)
    raw = torch.randn(2, 3, 4, 6, 5, 2, generator=generator, dtype=torch.float64)
    if real:
        raw[..., 1] = 0
    stored = hermitian_half(raw, shape[-1])
    transfer = torch.view_as_complex(stored)
    if real:
        stored = stored[..., 0]
    obj = _random_volume((3, *shape), seed=8)
    data = _random_volume((2, *shape), seed=9)
    imaged = torch.einsum("cf...,f...->c...", transfer, torch.fft.rfftn(obj, dim=(-3, -2, -1)))
    torch.testing.assert_close(apply_transfer(obj, stored), torch.fft.irfftn(imaged, s=shape))
    back = torch.einsum("cf...,c...->f...", transfer.conj(), torch.fft.rfftn(data, dim=(-3, -2, -1)))
    torch.testing.assert_close(apply_adjoint(data, stored), torch.fft.irfftn(back, s=shape))


@pytest.mark.parametrize("shape", [(4, 5, 6), (3, 6, 7)], ids=["even-x", "odd-x"])
def test_half_spectrum_frequencies_broadcast_to_the_rfftn_grid(shape):
    sampling = Sampling(shape=shape, pixel_size=(0.5, 0.2, 0.1))
    nz, ny, nx = shape
    fz, fy, fx = half_spectrum_frequencies(sampling, dtype=torch.float64)
    assert (fz.shape, fy.shape, fx.shape) == ((nz, 1, 1), (1, ny, 1), (1, 1, nx // 2 + 1))
    assert (fz**2 + fy**2 + fx**2).shape == torch.fft.rfftn(torch.zeros(shape)).shape
    torch.testing.assert_close(fz.flatten(), torch.fft.fftfreq(nz, d=0.5, dtype=torch.float64))
    torch.testing.assert_close(fy.flatten(), torch.fft.fftfreq(ny, d=0.2, dtype=torch.float64))
    torch.testing.assert_close(fx.flatten(), torch.fft.rfftfreq(nx, d=0.1, dtype=torch.float64))
    # A plane wave at one of these frequencies has its whole spectrum on that half-spectrum sample.
    z, y, x = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) * d for n, d in zip(shape, sampling.pixel_size, strict=True)),
        indexing="ij",
    )
    index = (1, 2, 1)
    frequency = (fz[index[0], 0, 0], fy[0, index[1], 0], fx[0, 0, index[2]])
    wave = torch.cos(2 * math.pi * (frequency[0] * z + frequency[1] * y + frequency[2] * x))
    spectrum = torch.fft.rfftn(wave).abs()
    assert spectrum[index] == pytest.approx(spectrum.max().item())
    assert spectrum[index] > 100 * spectrum.sum().sub(spectrum[index]).item()


def test_half_spectrum_frequencies_default_to_the_cpu_and_the_default_dtype():
    fz, fy, fx = half_spectrum_frequencies(Sampling(shape=(2, 3, 4), pixel_size=(1.0, 1.0, 1.0)))
    assert {f.dtype for f in (fz, fy, fx)} == {torch.get_default_dtype()}
    assert {f.device.type for f in (fz, fy, fx)} == {"cpu"}


def test_the_unit_guards_refuse_units_that_cannot_be_meant():
    sampling = Sampling(shape=(4, 16, 16), pixel_size=(0.5, 0.1, 0.1))
    expect_resolved_aperture(sampling, numerical_aperture=0.9, wavelength=0.52, where="Mine")
    expect_feasible_grid(sampling, (1, 2, 2), where="Mine")
    with pytest.raises(IncompatibleModelError, match=r"^Mine: .*wavelength is in um, got 520 "):
        expect_resolved_aperture(sampling, numerical_aperture=0.9, wavelength=520, where="Mine")
    in_nm = Sampling(shape=(4, 16, 16), pixel_size=(500, 100, 100))
    with pytest.raises(IncompatibleModelError, match=r"^Mine: .*pixel_size is in um, got 500 along z "):
        expect_feasible_grid(in_nm, (900, 700, 700), where="Mine")
    assert placement(torch.nn.Linear(2, 2).double()) == (torch.float64, torch.device("cpu"))


class _FftOps(TorchDispatchMode):
    """Record the transformed dims of every aten FFT call."""

    def __init__(self) -> None:
        super().__init__()
        self.dims: list[tuple[str, tuple[int, ...]]] = []

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        name = func.overloadpacket.__name__
        if name in ("_fft_c2c", "_fft_c2r", "_fft_r2c"):
            self.dims.append((name, tuple(args[1])))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("norm", [None, "ortho", "forward"])
@pytest.mark.parametrize("shape", [(13, 20, 17), (4, 6, 8), (1, 5, 9)])
def test_irfftn_on_the_cpu_is_torch_irfftn_one_axis_at_a_time(shape, norm):
    # MKL's multi-dimensional complex transform over the leading axes of a half spectrum corrupts the heap on Intel
    # CPUs (Slurm job 38077315, wo rec on (17, 2048, 1025)); a CPU irfftn must hand MKL one axis per call.
    spectrum = torch.view_as_complex(_half(_random_volume((2, *shape))).float().contiguous()) * 1.5
    with _FftOps() as ops:
        values = irfftn(spectrum, shape, norm=norm)
        stored = from_half_spectrum(torch.view_as_real(spectrum), shape, norm=norm)
    assert ops.dims
    assert all(len(dims) == 1 for _, dims in ops.dims), ops.dims
    expected = torch.fft.irfftn(spectrum.to(torch.complex128), s=shape, dim=(-3, -2, -1), norm=norm)
    torch.testing.assert_close(values.double(), expected, rtol=1e-5, atol=1e-5 * float(expected.abs().max()))
    torch.testing.assert_close(stored, values)


def test_widefield_otf_in_slabs_of_planes_equals_one_batch_and_differentiates_kz(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    nz, ny, nx = 37, 48, 40
    pupil = torch.randn(ny, nx, 2, generator=generator, dtype=torch.float64)
    kz = torch.rand(ny, nx, generator=generator, dtype=torch.float64).requires_grad_()
    z = torch.fft.fftfreq(nz, 1 / nz, dtype=torch.float64) * 0.25
    shape = (18, 23, 20)
    whole = widefield_otf(pupil, kz, z, shape=shape)
    assert whole.shape == (18, 23, 11, 2)
    monkeypatch.setattr(functional, "_SLAB_ELEMENTS", 5 * ny * nx)  # 8 slabs of at most 5 planes
    slabs = widefield_otf(pupil, kz, z, shape=shape)
    torch.testing.assert_close(slabs, whole, rtol=0, atol=1e-15)
    slabs.sum().backward()
    assert kz.grad is not None
    assert torch.isfinite(kz.grad).all()
    assert kz.grad.abs().max() > 0
