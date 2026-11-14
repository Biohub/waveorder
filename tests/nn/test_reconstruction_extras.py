"""The 3.x reconstruction extras: ADMM, RL back projectors and RLGC, Poisson-Gaussian noise."""

import pytest
import torch

from tests.nn import legacy_admm, legacy_rl
from tests.nn.conftest import MatrixSystem, make_system, random_transfer
from tests.nn.test_contracts import slice_weighted_blur
from waveorder.errors import ConfigurationWarning, IncompatibleModelError, ParameterError
from waveorder.nn import (
    FISTA,
    Gaussian,
    ImagingModel,
    NoPrior,
    Poisson,
    RichardsonLucy,
    Sampling,
    Tikhonov,
    TotalVariation,
)
from waveorder.nn.functional import from_half_spectrum
from waveorder.nn.noise import PoissonGaussian
from waveorder.nn.solvers import ADMM, BACK_PROJECTORS, Loop
from waveorder.nn.solvers.richardson_lucy import back_projector
from waveorder.testing import check_noise, check_solver

SAMPLING = Sampling(shape=(16, 64, 64), pixel_size=(0.25, 0.1, 0.1))
SMALL = Sampling(shape=(8, 16, 32), pixel_size=(0.25, 0.1, 0.1))


def _transfer(model: ImagingModel, sampling: Sampling) -> torch.Tensor:
    fourier = model.system.bind(sampling).as_fourier()
    assert fourier is not None
    return fourier.transfer


def _full_otf(model: ImagingModel, sampling: Sampling) -> torch.Tensor:
    """The 3.x full-spectrum complex OTF of a one-channel system, rebuilt from its half-spectrum storage."""
    transfer = _transfer(model, sampling)[0, 0]
    storage = transfer if transfer.dim() == 4 else torch.stack((transfer, torch.zeros_like(transfer)), dim=-1)
    return torch.fft.fftn(from_half_spectrum(storage, sampling.shape), dim=(-3, -2, -1))


def _poisson_data(model: ImagingModel, sampling: Sampling) -> torch.Tensor:
    obj = 50 * torch.rand(1, *sampling.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with torch.no_grad():
        return model.noise.sample(model(obj, sampling), generator=torch.Generator().manual_seed(1))


# --- Richardson-Lucy back projectors and RLGC -------------------------------------------------------------------


@pytest.mark.parametrize("kind", BACK_PROJECTORS)
def test_back_projectors_match_3x(kind):
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson()).double()
    otf = _full_otf(model, SAMPLING)
    reference = torch.view_as_real(legacy_rl.calculate_back_projector(otf, kind)[..., : SAMPLING.shape[-1] // 2 + 1])
    mine = back_projector(_transfer(model, SAMPLING), SAMPLING.shape, kind=kind, alpha=None, beta=None)
    torch.testing.assert_close(mine, reference, rtol=0, atol=1e-12 * reference.abs().max().item())


@pytest.mark.parametrize(
    ("kind", "iterations", "consensus"),
    [("matched", 10, False), ("wiener_butterworth", 1, False), ("gaussian", 3, False), ("matched", 10, True)],
    ids=["rl", "wiener_butterworth", "gaussian", "rlgc"],
)
def test_richardson_lucy_variants_match_3x_at_equal_iterations(kind, iterations, consensus):
    # 3.x starts from ones; it is given 4.x's start (the mean signal) so the iterates can be compared one to one.
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson()).double()
    data = _poisson_data(model, SAMPLING)
    otf = _full_otf(model, SAMPLING)
    kernel = legacy_rl.calculate_back_projector(otf, kind)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x, dim=(-3, -2, -1)) * otf, dim=(-3, -2, -1)))

    def transpose(y: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.fft.ifftn(torch.fft.fftn(y, dim=(-3, -2, -1)) * kernel, dim=(-3, -2, -1)))

    measured = data[0].clamp_min(0)
    reference = legacy_rl.richardson_lucy(
        measured,
        forward,
        transpose,
        num_iterations=iterations,
        method="RLGC" if consensus else "RL",
        guess=torch.full_like(measured, measured.mean().item()),
        generator=torch.Generator().manual_seed(0) if consensus else None,
    )
    solver = RichardsonLucy(loop=Loop(iterations), back_projector=kind, gradient_consensus=consensus)
    with torch.no_grad():
        estimate = solver(data, model, SAMPLING).estimate[0]
    torch.testing.assert_close(estimate, reference, rtol=0, atol=1e-12 * reference.abs().max().item())


@pytest.mark.parametrize(
    ("kind", "iterations"), [("matched", 50), ("wiener_butterworth", 2), ("gaussian", 2)], ids=["rl", "wb", "gaussian"]
)
def test_richardson_lucy_started_from_ones_reproduces_3x_with_a_background(kind, iterations):
    # 3.x starts from ones; 4.x by default from the mean signal, a gap of 6.8e-3 for RL at 50 iterations with b = 10
    # and 34-38% for unmatched back projectors at 2 (measured by the parity harness). start="ones" closes it.
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson()).double()
    data = _poisson_data(model, SAMPLING) + 10.0
    otf = _full_otf(model, SAMPLING)
    kernel = legacy_rl.calculate_back_projector(otf, kind)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.fft.ifftn(torch.fft.fftn(x, dim=(-3, -2, -1)) * otf, dim=(-3, -2, -1)))

    def transpose(y: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.fft.ifftn(torch.fft.fftn(y, dim=(-3, -2, -1)) * kernel, dim=(-3, -2, -1)))

    reference = legacy_rl.richardson_lucy(
        data[0].clamp_min(0), forward, transpose, num_iterations=iterations, background=10.0
    )
    solver = RichardsonLucy(loop=Loop(iterations), background=10.0, back_projector=kind, start="ones")
    with torch.no_grad():
        estimate = solver(data, model, SAMPLING).estimate[0]
    torch.testing.assert_close(estimate, reference, rtol=0, atol=1e-12 * reference.abs().max().item())
    with pytest.raises(ParameterError, match="start must be 'mean' or 'ones'") as error:
        RichardsonLucy(start="zeros")  # ty: ignore[invalid-argument-type]
    assert error.value.path == "start"


def test_one_wiener_butterworth_iteration_does_the_work_of_many_matched_ones():
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson()).double()
    obj = torch.zeros(1, *SAMPLING.shape, dtype=torch.float64)
    obj[0, 8, 20:44:6, 20:44:6] = 1000.0  # beads
    with torch.no_grad():
        data = model(obj, SAMPLING) + 1.0
        one = RichardsonLucy(loop=Loop(1), back_projector="wiener_butterworth")(data, model, SAMPLING)
        five = RichardsonLucy(loop=Loop(5))(data, model, SAMPLING)
    assert one.objective < five.objective


def test_rlgc_and_unmatched_back_projectors_are_refused_where_they_cannot_run():
    with pytest.raises(ParameterError, match="gradient_consensus needs back_projector='matched'") as error:
        RichardsonLucy(back_projector="wiener", gradient_consensus=True)
    assert error.value.path == "gradient_consensus"
    with pytest.raises(ParameterError, match="back_projector must be one of") as error:
        RichardsonLucy(back_projector="tikhonov")  # ty: ignore[invalid-argument-type]
    assert error.value.path == "back_projector"
    with pytest.raises(ParameterError, match="bp_alpha must be positive"):
        RichardsonLucy(back_projector="wiener", bp_alpha=0.0)
    with pytest.warns(ConfigurationWarning, match="reach their result in 1 to 5 iterations"):
        RichardsonLucy(loop=Loop(6), back_projector="wiener_butterworth")
    # They measure the PSF on the host or draw random numbers, which vmap cannot.
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson()).double()
    batch = torch.stack([_poisson_data(model, SMALL)] * 2)
    for solver in (RichardsonLucy(loop=Loop(1), back_projector="gaussian"), RichardsonLucy(gradient_consensus=True)):
        with pytest.raises(IncompatibleModelError, match="run it eagerly"):
            torch.func.vmap(lambda values, s=solver: s(values, model, SMALL).estimate)(batch)
    # Back projectors need a single-field Fourier-diagonal operator.
    blur = ImagingModel(slice_weighted_blur(SMALL), NoPrior(), Poisson()).double()
    blur.system.preserves_nonnegativity = True
    with pytest.raises(IncompatibleModelError, match="needs a Fourier-diagonal operator"):
        RichardsonLucy(loop=Loop(1), back_projector="gaussian")(_poisson_data(blur, SMALL).abs(), blur, SMALL)


def test_rlgc_resists_the_noise_amplification_of_over_iterated_rl():
    model = ImagingModel(make_system(1.0), NoPrior(), Poisson()).double()
    obj = torch.zeros(1, *SAMPLING.shape, dtype=torch.float64)
    obj[0, 8, 24:40, 24:40] = 20.0
    with torch.no_grad():
        data = model.noise.sample(model(obj, SAMPLING) + 2.0, generator=torch.Generator().manual_seed(3))
        plain = RichardsonLucy(loop=Loop(100))(data, model, SAMPLING).estimate
        consensus = RichardsonLucy(loop=Loop(100), gradient_consensus=True)(data, model, SAMPLING).estimate
    outside = obj == 0
    assert consensus[outside].max() < plain[outside].max()


# --- Poisson-Gaussian noise -------------------------------------------------------------------------------------


def test_poisson_gaussian_noise_passes_the_kit_and_samples_its_declared_moments():
    noise = PoissonGaussian(gain=2.0, offset=10.0, sigma=3.0).double()
    generator = torch.Generator().manual_seed(0)
    mean = 20.0 + 5.0 * torch.rand(50, generator=generator, dtype=torch.float64)
    check_noise(noise, noise.sample(mean, generator=generator), mean)
    flat = torch.full((200_000,), 40.0, dtype=torch.float64)
    drawn = noise.sample(flat, generator=torch.Generator().manual_seed(1))
    assert (drawn - 10.0).mean().item() == pytest.approx(40.0, rel=0.01)
    assert drawn.var().item() == pytest.approx(noise.variance(flat)[0].item(), rel=0.02)  # 2 * 40 + 9


def test_richardson_lucy_with_read_noise_is_shot_noise_rl_with_the_shifted_background():
    # The shifted-Poisson model: counts and rate both shift by sigma^2 / gain^2, so it is Poisson RL whose offset
    # drops by sigma^2 / gain and whose background rises by as much.
    gain, offset, sigma = 2.0, 10.0, 3.0
    shift = sigma**2 / gain
    system = make_system(1.0)
    read = ImagingModel(system, NoPrior(), PoissonGaussian(gain=gain, offset=offset, sigma=sigma)).double()
    shot = ImagingModel(system, NoPrior(), Poisson(gain=gain, offset=offset - shift)).double()
    data = _poisson_data(read, SMALL)
    with torch.no_grad():
        mine = RichardsonLucy(loop=Loop(10))(data, read, SMALL)
        reference = RichardsonLucy(loop=Loop(10), background=shift)(data, shot, SMALL)
    torch.testing.assert_close(mine.estimate, reference.estimate, rtol=1e-12, atol=0)
    with pytest.raises(IncompatibleModelError, match="Lipschitz gradient"):
        FISTA()(data, ImagingModel(system, Tikhonov(), PoissonGaussian()).double(), SMALL)


# --- ADMM -------------------------------------------------------------------------------------------------------


def test_admm_matches_3x_where_isotropic_and_anisotropic_tv_agree():
    # 3.x thresholds each axis separately; 4.x follows its isotropic TV prior. They coincide for an object that varies
    # along one axis, and with a vanishing TV weight; anything else is the documented difference.
    model = ImagingModel(make_system(1.0), TotalVariation(0.05), Gaussian(0.3)).double()
    otf = _full_otf(model, SMALL)
    profile = torch.zeros(SMALL.shape[-1], dtype=torch.float64)
    profile[8:20] = 1.0
    with torch.no_grad():
        clean = model(profile.expand(1, *SMALL.shape).clone(), SMALL)
    reference = legacy_admm.admm_tv_deconvolution_3d(clean[0], otf, rho=0.1, reg_re=1e-3, lambda_re=0.05, itr=10)
    estimate = ADMM(loop=Loop(10), rho=0.1, damping=1e-3)(clean, model, SMALL).estimate[0]
    torch.testing.assert_close(estimate, reference, rtol=0, atol=1e-12 * reference.abs().max().item())
    vanishing = ImagingModel(make_system(1.0), TotalVariation(1e-14), Gaussian(0.3)).double()
    data = torch.randn(1, *SMALL.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    reference = legacy_admm.admm_tv_deconvolution_3d(data[0], otf, rho=0.1, reg_re=1e-2, lambda_re=1e-14, itr=5)
    estimate = ADMM(loop=Loop(5), rho=0.1, damping=1e-2)(data, vanishing, SMALL).estimate[0]
    torch.testing.assert_close(estimate, reference, rtol=0, atol=1e-10 * reference.abs().max().item())


def test_admm_reaches_the_minimizer_fista_approaches():
    model = ImagingModel(make_system(1.0), TotalVariation(0.05, inner_iterations=50), Gaussian(0.1)).double()
    obj = torch.zeros(1, *SMALL.shape, dtype=torch.float64)
    obj[:, 2:6, 4:12, 8:24] = 1.0
    with torch.no_grad():
        data = model.noise.sample(model(obj, SMALL), generator=torch.Generator().manual_seed(0))
    fista = FISTA(loop=Loop(400))(data, model, SMALL)
    admm = ADMM(loop=Loop(300), rho=0.1, damping=0.0)(data, model, SMALL)
    assert admm.objective <= fista.objective + 1e-6 * fista.objective.abs()
    torch.testing.assert_close(admm.estimate, fista.estimate, rtol=0, atol=2e-2 * obj.max().item())


def test_admm_passes_the_solver_kit():
    model = ImagingModel(make_system(1.0), TotalVariation(0.05), Gaussian(0.1)).double()
    data = torch.randn(1, *SMALL.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    check_solver(ADMM(loop=Loop(3), rho=0.1), data=data, model=model, sampling=SMALL)


@pytest.mark.parametrize(
    ("channels", "fields", "dtype", "rtol"),
    [(3, 2, torch.float64, 1e-10), (1, 2, torch.float64, 1e-10), (2, 1, torch.float32, 1e-4)],
)
def test_admm_first_update_solves_its_normal_equations_for_any_channels_and_fields(channels, fields, dtype, rtol):
    # 3.x has one field and one channel: here the per-frequency solve is held to the operator's own actions.
    system = MatrixSystem(0.3 * random_transfer(channels, fields, SMALL.shape, dtype=dtype), fields=("a", "b")[:fields])
    model = ImagingModel(system, TotalVariation(0.05), Gaussian(0.3))
    model = model.double() if dtype == torch.float64 else model
    data = torch.randn(channels, *SMALL.shape, generator=torch.Generator().manual_seed(0), dtype=dtype)
    rho, damping = 0.1, 1e-2
    x = ADMM(loop=Loop(1), rho=rho, damping=damping)(data, model, SMALL).estimate  # from f = z = u = 0
    operator, dims = system.bind(SMALL), (-3, -2, -1)
    differences = [torch.roll(x, -1, dims=dim) - x for dim in dims]
    laplacian = sum(torch.roll(d, 1, dims=dim) - d for d, dim in zip(differences, dims, strict=True))
    expected = operator.adjoint(data)
    torch.testing.assert_close(
        operator.normal_action()(x) + rho * laplacian + damping * x,
        expected,
        rtol=0,
        atol=rtol * expected.abs().max().item(),
    )


def test_admm_refuses_models_it_cannot_split():
    data = torch.zeros(1, *SMALL.shape)
    for model, match in (
        (ImagingModel(make_system(1.0), Tikhonov(), Gaussian()), "total-variation prior"),
        (ImagingModel(make_system(1.0), TotalVariation(nonnegative=True), Gaussian()), r"\(nonnegative\)"),
        (ImagingModel(make_system(1.0), TotalVariation(), Poisson()), "white Gaussian noise"),
    ):
        with pytest.raises(IncompatibleModelError, match=match):
            ADMM()(data, model, SMALL)
    blur = ImagingModel(slice_weighted_blur(SMALL), TotalVariation(), Gaussian()).double()
    with pytest.raises(IncompatibleModelError, match="binds to a FourierDiagonal"):
        ADMM()(data.double(), blur, SMALL)
    with pytest.raises(ParameterError, match="ADMM rho must be positive and finite") as error:
        ADMM(rho=0.0)
    assert error.value.path == "rho"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
def test_admm_and_read_noise_richardson_lucy_pass_the_solver_kit_on_cuda():
    # The kit vmaps over candidates and, on CUDA, asserts that a solve without a tolerance never waits on the host.
    data = torch.randn(1, *SMALL.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64).cuda()
    tv = ImagingModel(make_system(1.0), TotalVariation(0.05), Gaussian(0.1)).double().cuda()
    check_solver(ADMM(loop=Loop(3), rho=0.1), data=data, model=tv, sampling=SMALL)
    read = ImagingModel(make_system(1.0), NoPrior(), PoissonGaussian(gain=2.0, sigma=3.0)).double().cuda()
    check_solver(RichardsonLucy(loop=Loop(3)), data=10 + data.abs(), model=read, sampling=SMALL)
