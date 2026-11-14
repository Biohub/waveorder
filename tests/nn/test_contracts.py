"""Every adapter passes the public contract kit in waveorder.testing."""

import math
from typing import ClassVar

import pytest
import torch
from torch import nn
from torch.func import vmap

from tests.nn.conftest import FREE_NA, MatrixSystem, make_model, make_system, random_transfer
from waveorder.nn import (
    FISTA,
    Evidence,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    ImagingModel,
    ImagingSystem,
    LBFGSSearch,
    NoPrior,
    Poisson,
    RandomSearch,
    RichardsonLucy,
    Sampling,
    Tikhonov,
    TotalVariation,
    Wiener,
    ZernikeAberration,
    evaluate,
)
from waveorder.nn.operators import LinearOperator
from waveorder.nn.search import CalibrationObjective, Estimate
from waveorder.nn.solvers import Loop
from waveorder.testing import (
    assert_adjoint,
    assert_gradients_match_finite_differences,
    assert_no_host_sync,
    assert_noise_gradient,
    assert_operator_norm,
    assert_operator_normal,
    assert_preserves_nonnegativity,
    assert_solver_descends,
    assert_state_roundtrip,
    assert_vmap_matches_loop,
    check_noise,
    check_operator,
    check_prior,
    check_score,
    check_search_stage,
    check_solver,
    check_system,
)

SMALL = Sampling(shape=(6, 8, 10), pixel_size=(0.4, 0.15, 0.15))
ODD = Sampling(shape=(5, 9, 7), pixel_size=(0.4, 0.12, 0.15))


class _Callables(LinearOperator):
    """A generic operator: ``forward`` and ``adjoint`` callables, a ``norm`` callable and the placement of ``like``."""

    def __init__(self, sampling, *, object_shape, data_shape, forward, adjoint, norm, like):
        super().__init__(sampling, object_shape=object_shape, data_shape=data_shape)
        self.forward, self.backward, self.bound, self.like = forward, adjoint, norm, like

    @property
    def placement(self):
        return self.like.dtype, self.like.device

    def apply(self, obj):
        return self.forward(obj)

    def adjoint(self, data):
        return self.backward(data)

    def norm(self):
        return self.bound()


class SliceWeightedBlur(ImagingSystem):
    """A shift-variant toy: z slice ``k`` scaled by ``weights[k]``, then a shift-invariant blur; bound matrix-free.

    Its operator carries the norm bound ``||blur|| max|weights|``.
    """

    def __init__(self, transfer, weights):
        super().__init__(fields=("a",), channel_count=transfer.shape[0])
        self.blur = MatrixSystem(transfer, fields=("a",))
        self.weights = nn.Parameter(weights)

    def compute_operator(self, sampling) -> LinearOperator:
        blur = self.blur.bind(sampling)
        weights = self.weights[:, None, None]
        return _Callables(
            sampling,
            object_shape=self.object_shape(sampling),
            data_shape=self.data_shape(sampling),
            forward=lambda f: blur.apply(weights * f),
            adjoint=lambda g: weights * blur.adjoint(g),
            norm=lambda: blur.norm() * self.weights.abs().max(),
            like=self.weights,
        )


def slice_weighted_blur(sampling):
    transfer = random_transfer(2, 1, sampling.shape, dtype=torch.float64)
    weights = torch.linspace(0.5, 1.5, sampling.shape[0], dtype=torch.float64)
    return SliceWeightedBlur(transfer, weights)


SYSTEMS = {
    "fluorescence": lambda: make_system(1.2, z_focus_offset=0.6, pupil_aberrations=ZernikeAberration({"coma_x": 0.1})),
    "fluorescence_free": lambda: make_system(
        Free(1.05, tolerance=0.1, bounds=(0.8, 1.2)), pupil_aberrations=ZernikeAberration({"spherical": Free(0.03)})
    ),
    "matrix_1x2": lambda: MatrixSystem(random_transfer(1, 2, SMALL.shape), fields=("a", "b")),
    "matrix_3x2": lambda: MatrixSystem(random_transfer(3, 2, SMALL.shape), fields=("a", "b")),
    "slice_weighted_blur": lambda: slice_weighted_blur(SMALL),
}


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="host syncs exist only on CUDA")


@pytest.mark.parametrize("name", SYSTEMS)
def test_systems_pass_the_conformance_suite(name):
    check_system(SYSTEMS[name], SMALL)


@CUDA
@pytest.mark.parametrize("name", ["fluorescence", "matrix_3x2", "slice_weighted_blur"])
def test_systems_bind_and_apply_without_host_syncs_on_cuda(name):
    check_system(lambda: SYSTEMS[name]().cuda(), SMALL)


@pytest.mark.parametrize("sampling", [SMALL, ODD], ids=["even", "odd"])
def test_fluorescence_adjoint_on_even_and_odd_anisotropic_grids(sampling):
    assert_adjoint(make_system(1.1, z_focus_offset=-0.4), sampling)


class _Gained(MatrixSystem):
    """Scales ``H`` by a plain Python attribute, which a state dict does not carry."""

    def __init__(self, transfer, *, gain):
        super().__init__(transfer, fields=("a",))
        self.gain = gain

    def compute_transfer(self, sampling):
        return self.gain * super().compute_transfer(sampling)


def test_system_suite_catches_state_a_state_dict_does_not_carry():
    transfer = random_transfer(1, 1, SMALL.shape, dtype=torch.float64)
    gains = iter([2.0, 1.0])  # the configured instance, then the fresh one the suite loads its state into
    with pytest.raises(AssertionError, match="load_state_dict did not restore"):
        check_system(lambda: _Gained(transfer, gain=next(gains)), SMALL)


def test_operator_norm_kit_accepts_a_loose_bound_and_reports_its_slack():
    # ||blur|| max|weights| bounds ||blur weights|| but is not tight; the check passes and returns the ratio.
    slack = assert_operator_norm(slice_weighted_blur(SMALL), SMALL)
    assert slack > 1.1


class _UnderclaimedNorm(SliceWeightedBlur):
    """Binds its operator with a norm bound half the true norm, as a wrong hand-derived bound would."""

    def compute_operator(self, sampling) -> LinearOperator:
        exact = super().compute_operator(sampling)  # with unit weights its bound is ||H|| itself
        return _Callables(
            sampling,
            object_shape=exact.object_shape,
            data_shape=exact.data_shape,
            forward=exact.apply,
            adjoint=exact.adjoint,
            norm=lambda: 0.5 * exact.norm(),
            like=self.weights,
        )


def test_operator_norm_kit_catches_a_bound_below_the_norm():
    system = _UnderclaimedNorm(
        random_transfer(2, 1, SMALL.shape, dtype=torch.float64), torch.ones(6, dtype=torch.float64)
    )
    with pytest.raises(AssertionError, match=r"norm\(\) = .* < \|\|H\|\|"):
        assert_operator_norm(system, SMALL)


class _WrongNormal(_Callables):
    """Claims a cheaper normal operator that forgets a factor of two."""

    def normal_action(self):
        return lambda obj: 0.5 * self.adjoint(self.apply(obj))


def test_operator_normal_kit_catches_a_wrong_shortcut():
    blur = make_system(1.0).double().bind(SMALL)
    callables = {
        "object_shape": blur.object_shape,
        "data_shape": blur.data_shape,
        "forward": blur.apply,
        "adjoint": blur.adjoint,
        "norm": blur.norm,
        "like": blur.norm(),
    }
    generic = _Callables(SMALL, **callables)
    wrong = _WrongNormal(SMALL, **callables)
    assert_operator_normal(blur)
    check_operator(generic)
    with pytest.raises(AssertionError, match=r"_WrongNormal\.normal_action differs from adjoint\(apply\)"):
        assert_operator_normal(wrong)
    with pytest.raises(AssertionError, match=r"_WrongNormal\.normal_action differs from adjoint\(apply\)"):
        check_operator(wrong)


PRIORS = {
    "none": NoPrior(),
    "nonneg": NoPrior(nonnegative=True),
    "tikhonov": Tikhonov(0.5),
    "tikhonov_nonneg": Tikhonov(0.5, nonnegative=True),
    "tv": TotalVariation(0.3, inner_iterations=300),
    "tv_nonneg": TotalVariation(0.3, nonnegative=True, inner_iterations=300),
}


@pytest.mark.parametrize("name", PRIORS)
def test_priors_pass_the_conformance_suite(name):
    obj = torch.randn(1, 4, 6, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    check_prior(PRIORS[name].double(), obj, step=0.7, scale=1.3)


class _MisstatedTikhonov(Tikhonov):
    """Claims a prior variance twice the one its penalty implies, so Wiener and FISTA would disagree."""

    def spectrum(self, sampling, *, scale):
        return 2 * super().spectrum(sampling, scale=scale)


def test_prior_suite_catches_a_spectrum_that_contradicts_the_penalty():
    obj = torch.randn(1, 4, 6, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with pytest.raises(AssertionError, match="penalty disagrees with its spectrum"):
        check_prior(_MisstatedTikhonov(0.5).double(), obj)


class _PhysicalTikhonov(Tikhonov):
    """A spectrum scaled by the voxel volume in um^3: what a prior stated in physical units would give."""

    def spectrum(self, sampling, *, scale):
        return super().spectrum(sampling, scale=scale) * math.prod(sampling.pixel_size)


def test_prior_suite_catches_a_spectrum_in_physical_units():
    obj = torch.randn(1, 4, 6, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with pytest.raises(AssertionError, match="penalty disagrees with its spectrum"):
        check_prior(_PhysicalTikhonov(0.5).double(), obj)


class _HalfPenalty(Tikhonov):
    """States half the penalty its prox minimizes: the penalty is the bug, not the prox."""

    def penalty(self, obj, *, scale):
        return 0.5 * super().penalty(obj, scale=scale)


def test_prior_suite_names_both_the_prox_and_the_penalty_when_they_disagree():
    obj = torch.randn(1, 4, 6, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with pytest.raises(AssertionError, match=r"_HalfPenalty\.prox and _HalfPenalty\.penalty disagree"):
        check_prior(_HalfPenalty(0.5).double(), obj)


class _SyncingProx(Tikhonov):
    def prox(self, obj, step, *, scale):
        return super().prox(obj, step, scale=scale) * (step.item() >= 0)


@CUDA
def test_prior_suite_catches_a_prox_that_reads_back_on_cuda():
    obj = torch.randn(1, 4, 6, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64).cuda()
    check_prior(Tikhonov(0.5).double().cuda(), obj)
    with pytest.raises(AssertionError, match=r"_SyncingProx\.prox reads values back to the host"):
        check_prior(_SyncingProx(0.5).double().cuda(), obj)


@pytest.mark.parametrize("noise", [Gaussian(0.3), Poisson(gain=2.0, offset=10.0)], ids=["gaussian", "poisson"])
def test_noises_pass_the_conformance_suite(noise):
    generator = torch.Generator().manual_seed(0)
    mean = 20.0 + 5.0 * torch.rand(50, generator=generator, dtype=torch.float64)
    data = noise.double().sample(mean, generator=generator)
    check_noise(noise.double(), data, mean)


SOLVERS = {
    "wiener": (Wiener, Tikhonov(1e-2), Gaussian(0.05)),
    "fista": (lambda: FISTA(loop=Loop(20)), Tikhonov(1e-2, nonnegative=True), Gaussian(0.05)),
    "fista_tv": (lambda: FISTA(loop=Loop(10)), TotalVariation(1e-2, inner_iterations=20), Gaussian(0.05)),
    "richardson_lucy": (lambda: RichardsonLucy(loop=Loop(10)), NoPrior(nonnegative=True), Poisson(gain=1.0)),
}


@pytest.mark.parametrize("name", SOLVERS)
def test_solvers_pass_the_conformance_suite(name):
    build, prior, noise = SOLVERS[name]
    model = ImagingModel(make_system(1.0), prior, noise).double()
    generator = torch.Generator().manual_seed(0)
    obj = 1.0 + torch.rand(model.system.object_shape(TINY), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        data = model.noise.sample(model(obj, TINY), generator=generator)
    check_solver(build(), data=data, model=model, sampling=TINY)


@CUDA
@pytest.mark.parametrize("name", ["wiener", "fista"])
def test_solvers_run_without_host_syncs_on_cuda(name):
    build, prior, noise = SOLVERS[name]
    model = ImagingModel(make_system(1.0), prior, noise).double().cuda()
    generator = torch.Generator().manual_seed(0)
    obj = 1.0 + torch.rand(model.system.object_shape(TINY), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        data = model.noise.sample(model(obj.cuda(), TINY))
    check_solver(build(), data=data, model=model, sampling=TINY)


def _problem(*, device, nominal=1.15):
    model = make_model(Free(nominal, tolerance=0.1, bounds=(0.8, 1.2))).double().to(device)
    generator = torch.Generator().manual_seed(0)
    obj = torch.rand(model.system.object_shape(TINY), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        data = model.noise.sample(model(obj.to(device), TINY), generator=None if device.type == "cuda" else generator)
    objective = CalibrationObjective.of(data, model, TINY, score=Evidence())
    return objective, objective.start()


STAGES = {
    "grid": GridSearch(points=5),
    "random": RandomSearch(samples=6),
    "gradient": GradientSearch(iterations=5),
    "lbfgs": LBFGSSearch(iterations=3),
}
DEVICES = [
    torch.device("cpu"),
    pytest.param(
        torch.device("cuda"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no host sync on CUDA")
    ),
]


@pytest.mark.parametrize("device", DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("name", STAGES)
def test_search_stages_pass_the_conformance_suite(name, device):
    objective, start = _problem(device=device)
    check_search_stage(STAGES[name], objective, start)


class _Overreaching:
    """A search stage that steps past the upper bound."""

    reads_back: ClassVar[bool] = False
    quantity_keyed: ClassVar[tuple[str, ...]] = ()

    def run(self, objective, *, start):
        point = {name: value + 1.0 for name, value in start.point.items()}
        return Estimate(point, objective(point))


def test_search_stage_suite_catches_a_stage_that_leaves_the_bounds():
    objective, start = _problem(device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="left the bounds"):
        check_search_stage(_Overreaching(), objective, start)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("device", DEVICES, ids=["cpu", "cuda"])
def test_score_passes_the_conformance_suite(device, dtype):
    model = make_model(Free(1.05, tolerance=0.1, bounds=(0.8, 1.2))).to(dtype=dtype, device=device)
    data = torch.randn(1, *TINY.shape, generator=torch.Generator().manual_seed(0), dtype=dtype).to(device)
    check_score(Evidence(), data=data, model=model, sampling=TINY)


class _ReadingScore(Evidence):
    """Evidence that inspects its value on the host, as a plugin that logs or branches would."""

    def run(self, problem):
        value = super().run(problem)
        if value.item() > 1e300:
            raise RuntimeError("unreachable")
        return value


@CUDA
def test_score_suite_names_the_line_of_a_score_that_reads_back_on_cuda():
    model = make_model(1.0).double().cuda()
    data = torch.randn(1, *TINY.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64).cuda()
    with pytest.raises(AssertionError, match=r"test_contracts\.py:\d+ in run \(if value\.item\(\) > 1e300:\)"):
        check_score(_ReadingScore(), data=data, model=model, sampling=TINY)


@CUDA
def test_no_host_sync_kit_passes_device_work_and_returns_its_result():
    values = torch.arange(4.0, device="cuda")
    assert torch.equal(assert_no_host_sync(torch.cumsum, values, 0), torch.tensor([0.0, 1.0, 3.0, 6.0], device="cuda"))
    with pytest.raises(AssertionError, match="reads values back to the host"):
        assert_no_host_sync(lambda: values.sum().item())


class _VectorScore(Evidence):
    """Returns one value per voxel instead of a scalar."""

    def run(self, problem):
        return super().run(problem) * torch.ones_like(problem.data)


def test_score_suite_catches_a_score_that_is_not_a_scalar():
    model = make_model(1.0).double()
    data = torch.randn(1, *TINY.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with pytest.raises(AssertionError, match="not a 0-dim tensor"):
        check_score(_VectorScore(), data=data, model=model, sampling=TINY)


class _ClaimsRealTransfer(MatrixSystem):
    """Declares a real H while its transfer is complex: the real storage would drop its imaginary part."""

    real_transfer = True


def test_system_suite_catches_a_false_real_transfer_claim():
    with pytest.raises(AssertionError, match="declares real_transfer, but its transfer has an imaginary part"):
        check_system(lambda: _ClaimsRealTransfer(random_transfer(1, 1, SMALL.shape), fields=("a",)), SMALL)


def test_transfer_vmaps_like_a_loop():
    assert_vmap_matches_loop(
        lambda m: m.transfer(SMALL),
        make_system(FREE_NA),
        name="detection.numerical_aperture",
        values=torch.tensor([0.9, 1.0, 1.1]),
    )


def test_evidence_gradients_match_finite_differences():
    model = make_model(
        Free(1.05, tolerance=0.1, bounds=(0.8, 1.2)),
        z_focus_offset=Free(0.3, bounds=(-2.0, 2.0)),
        pupil_aberrations=ZernikeAberration({"spherical": Free(0.03, bounds=(-0.2, 0.2))}),
    ).double()
    data = torch.randn(1, *SMALL.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert_gradients_match_finite_differences(lambda m: Evidence()(data, m, SMALL), model)


def test_models_round_trip_through_state_dict_and_deepcopy():
    def build() -> ImagingModel:
        return make_model(FREE_NA)

    model = build()
    model.load_state_dict({**model.state_dict(), "system.detection.numerical_aperture": torch.tensor(0.9)})
    data = torch.randn(1, *SMALL.shape, generator=torch.Generator().manual_seed(0))
    assert_state_roundtrip(model, build, probe=lambda m: Evidence()(data, m, SMALL))


def test_poisson_gradient_is_consistent_at_zero_and_tiny_means():
    noise = Poisson(gain=2.0).double()
    mean = torch.tensor([0.0, 1e-300, 1e-6, 3.0], dtype=torch.float64)
    assert_noise_gradient(noise, torch.tensor([0.0, 0.0, 2.0, 4.0], dtype=torch.float64), mean)


def test_zero_operator_has_zero_norm_and_fista_stays_finite():
    system = MatrixSystem(torch.zeros(1, 1, *SMALL.shape, dtype=torch.complex64), fields=("a",))
    model = ImagingModel(system, Tikhonov(0.5), Gaussian(1.0))
    operator = system.bind(SMALL)
    assert operator.norm() == 0
    result = FISTA(loop=Loop(5))(torch.randn(1, *SMALL.shape), model, SMALL)
    assert torch.isfinite(result.estimate).all()
    assert result.estimate.abs().max() == 0  # the prior's minimizer, since there is no data term


TINY = Sampling(shape=(4, 8, 6), pixel_size=(0.4, 0.15, 0.15))


def _shift_variant_problem():
    model = ImagingModel(slice_weighted_blur(TINY), Tikhonov(1e-2), Gaussian(0.1)).double()
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(3, *model.system.data_shape(TINY), generator=generator, dtype=torch.float64)
    return model, data


def test_matrix_free_fista_vmaps_over_data_like_a_loop():
    model, data = _shift_variant_problem()

    def solve(g):
        return FISTA(loop=Loop(5))(g, model, TINY).estimate

    torch.testing.assert_close(vmap(solve)(data), torch.stack([solve(g) for g in data]))


def test_matrix_free_fista_vmaps_over_system_parameters():
    # The operator then closes over batched weights, so its norm cannot be read back; it must still match eager.
    model, data = _shift_variant_problem()
    weights = torch.stack([model.system.weights.detach() * scale for scale in (0.5, 1.0, 2.0)])

    def solve(w):
        return evaluate(lambda m: FISTA(loop=Loop(5))(data[0], m, TINY).estimate, model, {"system.weights": w})

    looped = torch.stack([solve(w) for w in weights])
    torch.testing.assert_close(vmap(solve)(weights), looped)


def test_a_shift_variant_operator_with_a_norm_bound_passes_the_kit_and_solves():
    model, data = _shift_variant_problem()
    assert_adjoint(model.system, TINY, rtol=1e-10)
    assert_solver_descends(FISTA(loop=Loop(10)), data=data[0], model=model, sampling=TINY)


class _ClaimsNonnegative(MatrixSystem):
    preserves_nonnegativity = True


@pytest.mark.parametrize(
    "sampling",
    [Sampling(shape=(8, 16, 16), pixel_size=(0.25, 0.1, 0.1)), Sampling(shape=(4, 16, 16), pixel_size=(1.0, 0.5, 0.5))],
    ids=["nyquist", "coarse"],
)
def test_fluorescence_preserves_nonnegativity(sampling):
    # The PSF is clamped after a coarse grid crops its transfer, so neither grid rings beyond rounding.
    assert_preserves_nonnegativity(make_system(1.0, z_focus_offset=0.6), sampling, ringing=1e-5)


class _CopiedGain(MatrixSystem):
    """Copies its free gain into a buffer in place: fine eagerly, impossible for a vmapped candidate batch."""

    def __init__(self, transfer) -> None:
        super().__init__(transfer, fields=("a",))
        self.register_quantity("gain", Free(1.0, bounds=(0.5, 2.0)))
        self.register_buffer("scratch", torch.zeros((), dtype=transfer.real.dtype))

    def compute_transfer(self, sampling):
        scratch = self.get_buffer("scratch")
        scratch.copy_(self.quantity("gain"))
        return scratch * super().compute_transfer(sampling)


def test_score_suite_vmaps_over_parameter_candidates():
    model = ImagingModel(
        _CopiedGain(random_transfer(1, 1, TINY.shape, dtype=torch.float64)), Tikhonov(0.5), Gaussian(1.0)
    )
    model = model.double()
    data = torch.randn(1, *TINY.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with torch.no_grad():
        assert torch.isfinite(Evidence()(data, model, TINY)), "the score itself runs"
    with pytest.raises(AssertionError, match=r"does not vmap over candidate values of \['system.gain'\]"):
        check_score(Evidence(), data=data, model=model, sampling=TINY)


@CUDA
def test_no_host_sync_kit_names_the_caller_the_library_frame_and_the_tolerance_remedy():
    model = make_model(1.0).cuda()
    data = torch.ones(1, *TINY.shape, device="cuda")
    with pytest.raises(AssertionError, match=r"test_contracts\.py:\d+ in <lambda> .* via .*parameters\.py:\d+ in "
                       r"read_flag .*tolerance=1e-06 to _checked.*use tolerance=None"):  # fmt: skip
        assert_no_host_sync(lambda: FISTA(loop=Loop(20, tolerance=1e-6))(data, model, TINY))


def test_nonnegativity_check_catches_a_psf_with_a_negative_lobe():
    psf = torch.zeros(SMALL.shape, dtype=torch.float64)
    psf[0, 0, 0] = 1.0
    psf[0, 0, 1] = -0.5  # a lobe at half the peak, far beyond band-limiting ringing
    transfer = torch.fft.fftn(psf)[None, None]
    with pytest.raises(AssertionError, match="point-spread function reaches"):
        assert_preserves_nonnegativity(_ClaimsNonnegative(transfer, fields=("a",)).double(), SMALL)
