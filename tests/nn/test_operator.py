"""Bound operators, model checks, and solvers and scores refusing what they cannot do."""

import copy
import pickle
from collections.abc import Callable
from typing import override

import pytest
import torch
from torch import Tensor

from tests.nn.conftest import MatrixSystem, make_model, make_system, random_transfer
from waveorder.errors import IncompatibleModelError, PlacementError, ShapeError, WaveorderError
from waveorder.nn import (
    FISTA,
    Evidence,
    FourierDiagonal,
    Free,
    Gaussian,
    ImagingModel,
    ImagingSystem,
    InverseProblem,
    LinearOperator,
    NoPrior,
    Poisson,
    RichardsonLucy,
    Sampling,
    Tikhonov,
    TotalVariation,
    Wiener,
)
from waveorder.nn.operators import ReshapedFourierDiagonal
from waveorder.nn.solvers import Loop
from waveorder.testing import check_operator

SAMPLING = Sampling(shape=(4, 8, 6), pixel_size=(0.3, 0.1, 0.1))
CPU = torch.device("cpu")


class _Generic(LinearOperator):
    """An operator seen only through its two actions and a given norm: no structure the solvers can use."""

    def __init__(
        self,
        sampling: Sampling,
        *,
        object_shape: tuple[int, ...],
        data_shape: tuple[int, ...],
        forward: Callable[[Tensor], Tensor],
        adjoint: Callable[[Tensor], Tensor],
        norm: Tensor | float,
        dtype: torch.dtype = torch.float64,
        device: torch.device = CPU,
    ) -> None:
        super().__init__(sampling, object_shape=object_shape, data_shape=data_shape)
        self._forward, self._adjoint = forward, adjoint
        self._norm = torch.as_tensor(norm, dtype=dtype, device=device)
        self._placement = (dtype, device)

    @override
    def apply(self, obj):
        return self._forward(obj)

    @override
    def adjoint(self, data):
        return self._adjoint(data)

    @override
    def norm(self):
        return self._norm

    @property
    @override
    def placement(self):
        return self._placement


def _generic(operator: LinearOperator, *, forward: Callable[[Tensor], Tensor] | None = None) -> _Generic:
    """``operator`` seen only through its two actions (``forward`` replacing ``apply``) and its norm."""
    dtype, device = operator.placement
    return _Generic(
        operator.sampling,
        object_shape=operator.object_shape,
        data_shape=operator.data_shape,
        forward=forward or operator.apply,
        adjoint=operator.adjoint,
        norm=operator.norm(),
        dtype=dtype,
        device=device,
    )


def _random(shape, *, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _power_norm(operator: LinearOperator, *, iterations: int = 500) -> Tensor:
    """A power-iteration estimate of ``||H||_2``: it approaches the norm from below."""
    dtype, device = operator.placement
    vector = _random(operator.object_shape, seed=0).to(dtype=dtype, device=device)
    estimate = torch.zeros((), dtype=dtype, device=device)
    for _ in range(iterations):
        vector = operator.adjoint(operator.apply(vector / vector.norm()))
        estimate = vector.norm()
    return estimate.sqrt()


def _fourier(channels, fields, *, seed=0, sampling=SAMPLING):
    transfer = random_transfer(channels, fields, sampling.shape, seed=seed, dtype=torch.float64)
    return FourierDiagonal(MatrixSystem(transfer, fields=tuple("abcd"[:fields])).transfer(sampling), sampling)


def _on_cuda(operator: FourierDiagonal) -> FourierDiagonal:
    return FourierDiagonal(operator.transfer.cuda(), operator.sampling, hermitian=True)


def _real_fourier(channels, fields, *, seed=0):
    """A Fourier-diagonal operator with a real ``H`` (an even PSF), stored without an imaginary part."""
    half = random_transfer(channels, fields, SAMPLING.shape, seed=seed, dtype=torch.float64).real
    return FourierDiagonal(half[..., : SAMPLING.shape[-1] // 2 + 1], SAMPLING)


def _slice_weights(fields):
    """A shift-variant diagonal: z slice ``k`` scaled by a weight in [-1.5, 0.5], so ``||W|| = 1.5`` exactly."""
    weights = torch.linspace(-1.5, 0.5, SAMPLING.shape[0], dtype=torch.float64)[:, None, None]
    shape = (fields, *SAMPLING.shape)
    return _Generic(
        SAMPLING,
        object_shape=shape,
        data_shape=shape,
        forward=lambda f: weights * f,
        adjoint=lambda g: weights * g,
        norm=1.5,
    )


def _crop(channels):
    """Drop the outer x columns: ``(C, Z, Y, X)`` to ``(C, Z, Y, X - 2)``; its adjoint zero-pads them back."""
    shape = (channels, *SAMPLING.shape)
    return _Generic(
        SAMPLING,
        object_shape=shape,
        data_shape=(*shape[:-1], shape[-1] - 2),
        forward=lambda g: g[..., 1:-1],
        adjoint=lambda h: torch.nn.functional.pad(h, (1, 1)),
        norm=1.0,
    )


def test_duplicate_field_names_rejected():
    with pytest.raises(ValueError, match="distinct names"):
        MatrixSystem(random_transfer(1, 2, SAMPLING.shape), fields=("phase", "phase"))


def test_model_check_rejects_a_focus_offset_that_wraps_the_stack(sampling):
    model = ImagingModel(make_system(1.0, z_focus_offset=Free(0.0, bounds=(-12.0, 12.0))), Tikhonov(), Gaussian())
    with pytest.raises(IncompatibleModelError, match="z_focus_offset can reach"):
        model.check(Wiener(), sampling)  # 16 slices: +/-8 at most


@pytest.mark.parametrize(
    ("verb", "prior", "noise", "match"),
    [
        (Wiener(), Tikhonov(), Poisson(), "Gaussian noise"),
        (Evidence(), Tikhonov(), Poisson(), "Gaussian noise"),
        (Wiener(), TotalVariation(), Gaussian(), "Gaussian object prior"),
        (Wiener(), Tikhonov(nonnegative=True), Gaussian(), "nonnegativity"),
        (FISTA(), Tikhonov(), Poisson(), "Lipschitz"),
        (RichardsonLucy(), NoPrior(), Gaussian(), "Poisson noise"),
        (RichardsonLucy(), Tikhonov(), Poisson(), "likelihood alone"),
    ],
)
def test_verbs_reject_models_that_break_their_assumptions(verb, prior, noise, match, sampling):
    model = ImagingModel(make_system(1.0), prior, noise)
    with pytest.raises(IncompatibleModelError, match=match):
        model.check(verb, sampling)


@pytest.mark.parametrize("verb", [Wiener(), Evidence()])
def test_verbs_reject_data_that_would_broadcast(verb, sampling):
    model = make_model(1.0)
    squashed = torch.zeros(1, 1, *sampling.shape[1:])  # a single z plane would broadcast over the stack
    with pytest.raises(ShapeError, match="data has shape"):
        verb(squashed, model, sampling)


def test_problem_rejects_an_operator_bound_to_another_sampling(sampling):
    model = make_model(1.0)
    other = model.system.bind(Sampling(shape=(8, 32, 32), pixel_size=sampling.pixel_size))
    with pytest.raises(ShapeError, match="operator was bound to"):
        InverseProblem.of(torch.zeros(1, *sampling.shape), model, sampling, operator=other)
    with pytest.raises(ShapeError, match="operator was bound to"):
        FISTA(loop=Loop(1))(torch.zeros(1, *sampling.shape), model, sampling, operator=other)


def test_problem_rejects_an_operator_with_other_shapes():
    model = ImagingModel(MatrixSystem(random_transfer(1, 2, SAMPLING.shape), fields=("a", "b")), Tikhonov(), Gaussian())
    three_channels = FourierDiagonal(
        MatrixSystem(random_transfer(3, 2, SAMPLING.shape), fields=("a", "b")).transfer(SAMPLING), SAMPLING
    )
    with pytest.raises(ShapeError, match=r"operator maps data of shape \(3,"):
        InverseProblem.of(torch.zeros(1, *SAMPLING.shape), model, SAMPLING, operator=three_channels)
    one_field = FourierDiagonal(
        MatrixSystem(random_transfer(1, 1, SAMPLING.shape), fields=("a",)).transfer(SAMPLING), SAMPLING
    )
    with pytest.raises(ShapeError, match=r"operator maps objects of shape \(1,"):
        InverseProblem.of(torch.zeros(1, *SAMPLING.shape), model, SAMPLING, operator=one_field)


def test_problem_rejects_an_operator_of_another_structure():
    model = ImagingModel(MatrixSystem(random_transfer(1, 1, SAMPLING.shape), fields=("a",)), Tikhonov(), Gaussian())
    generic = _generic(model.system.bind(SAMPLING))
    with pytest.raises(IncompatibleModelError, match="declares operator_type FourierDiagonal"):
        InverseProblem.of(torch.zeros(1, *SAMPLING.shape), model, SAMPLING, operator=generic)


def _inner(a, b):
    return (a * b).sum()


@pytest.mark.parametrize("shape", [(4, 8, 6), (5, 7, 9)], ids=["even", "odd"])
def test_fourier_diagonal_adjoint_and_exact_norm(shape):
    sampling = Sampling(shape=shape, pixel_size=(0.3, 0.1, 0.1))
    transfer = MatrixSystem(random_transfer(3, 2, shape, dtype=torch.float64), fields=("a", "b")).transfer(sampling)
    operator = FourierDiagonal(transfer, sampling)
    generator = torch.Generator().manual_seed(1)
    obj = torch.randn(operator.object_shape, generator=generator, dtype=torch.float64)
    data = torch.randn(operator.data_shape, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        _inner(operator.apply(obj), data), _inner(obj, operator.adjoint(data)), rtol=1e-12, atol=0
    )
    norm = operator.norm()
    reference = _power_norm(operator)
    assert norm >= reference * (1 - 1e-9), "an exact norm is never below what power iteration reaches"
    torch.testing.assert_close(norm, reference, rtol=1e-3, atol=0)


def test_fourier_diagonal_rejects_inputs_of_the_wrong_shape():
    operator = make_system(1.0).bind(SAMPLING)
    with pytest.raises(ShapeError, match="object has shape"):
        operator.apply(torch.zeros(1, *SAMPLING.shape[1:]))
    with pytest.raises(ShapeError, match="data must be real"):
        operator.adjoint(torch.zeros(1, *SAMPLING.shape, dtype=torch.complex64))


def test_richardson_lucy_rejects_systems_that_do_not_declare_a_nonnegative_psf():
    signed = MatrixSystem(random_transfer(1, 1, SAMPLING.shape), fields=("a",))
    model = ImagingModel(signed, NoPrior(), Poisson())
    with pytest.raises(IncompatibleModelError, match="nonnegative point-spread function"):
        model.check(RichardsonLucy(), SAMPLING)


def test_errors_share_one_catchable_base():
    with pytest.raises(WaveorderError):
        Sampling(shape=(0, 1, 1), pixel_size=(1.0, 1.0, 1.0))
    with pytest.raises(WaveorderError):
        Loop(0)
    with pytest.raises(WaveorderError):
        make_system(1.0)(torch.zeros(3), SAMPLING)


OPERATORS = {
    "fourier": lambda: _fourier(3, 2),
    "more_fields_than_channels": lambda: _fourier(1, 2),
    "real_fourier": lambda: _real_fourier(3, 2),
    "reshaped": lambda: ReshapedFourierDiagonal(_fourier(3, 2), SAMPLING, data_shape=(1, 12, 8, 6)),
    "generic": lambda: _generic(_fourier(3, 2)),
    "slice_weights": lambda: _slice_weights(2),
    "crop": lambda: _crop(3),
}


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_are_adjoint_consistent_and_bound_their_norm(name):
    operator = OPERATORS[name]()
    check_operator(operator, rtol=1e-10)
    reference = _power_norm(operator)
    assert operator.norm() >= reference * (1 - 1e-9), "a norm bound must not be below ||H||"
    if isinstance(operator, FourierDiagonal | ReshapedFourierDiagonal):
        torch.testing.assert_close(operator.norm(), reference, rtol=1e-3, atol=0)


class _Halved(LinearOperator):
    """Half a blur that still claims the blur's Fourier structure."""

    def __init__(self, fourier: FourierDiagonal) -> None:
        super().__init__(fourier.sampling, object_shape=fourier.object_shape, data_shape=fourier.data_shape)
        self.fourier = fourier

    @override
    def apply(self, obj):
        return 0.5 * self.fourier.apply(obj)

    @override
    def adjoint(self, data):
        return 0.5 * self.fourier.adjoint(data)

    @override
    def norm(self):
        return 0.5 * self.fourier.norm()

    @override
    def as_fourier(self):
        return self.fourier

    @property
    @override
    def placement(self):
        return self.fourier.placement


class _ForgetsTheSquare(_Halved):
    """Its normal action is ``H f``, not ``H* H f``."""

    @override
    def as_fourier(self):
        return None

    @override
    def normal_action(self):
        return self.apply


def test_the_operator_suite_catches_a_wrong_fourier_structure_and_a_wrong_normal_action():
    with pytest.raises(AssertionError, match=r"_Halved\.as_fourier disagrees with apply"):
        check_operator(_Halved(_fourier(3, 2)), rtol=1e-10)
    with pytest.raises(AssertionError, match=r"_ForgetsTheSquare\.normal_action differs from adjoint\(apply\)"):
        check_operator(_ForgetsTheSquare(_fourier(2, 2)), rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host syncs exist only on CUDA")
def test_the_operator_suite_catches_an_action_that_reads_back_on_cuda():
    fourier = _on_cuda(_fourier(3, 2))
    check_operator(fourier, rtol=1e-10)

    def forward(obj: Tensor) -> Tensor:
        peak = obj.abs().max().cpu()  # a copy to the host: a sync on CUDA, also under vmap
        return fourier.apply(obj) * (peak.to(obj.device) >= 0)

    syncing = _generic(fourier, forward=forward)
    with pytest.raises(AssertionError, match=r"_Generic\.apply reads values back to the host"):
        check_operator(syncing, rtol=1e-10)


def test_an_operator_stays_copyable_after_a_transform_over_its_normal():
    # A functorch-wrapped Gram once stayed in a cross-call cache after a transform: deepcopy and pickle then failed.
    operator, rhs = _fourier(3, 2), _random((2, *SAMPLING.shape), seed=1)
    normal = operator.normal_action()
    gradient = torch.func.grad(lambda x: normal(x).square().sum())(rhs)
    torch.testing.assert_close(gradient, 2 * normal(normal(rhs)))
    batched = torch.func.vmap(normal)(torch.stack([rhs, 2 * rhs]))
    torch.testing.assert_close(batched[1], 2 * normal(rhs))
    for copied in (copy.deepcopy(operator), pickle.loads(pickle.dumps(operator))):
        torch.testing.assert_close(copied.normal_action()(rhs), normal(rhs))


def test_an_in_place_write_to_the_transfer_reaches_the_normal_action():
    operator, rhs = _fourier(3, 2), _random((2, *SAMPLING.shape), seed=1)
    before = operator.normal_action()(rhs)
    with torch.no_grad():
        operator.transfer.mul_(2.0)  # H -> 2 H, so H* H -> 4 H* H
    torch.testing.assert_close(operator.normal_action()(rhs), 4 * before)


@pytest.mark.parametrize("shape", [(4, 8, 6), (5, 7, 9)], ids=["even", "odd"])
def test_fourier_diagonal_normal_matches_adjoint_after_apply(shape):
    sampling = Sampling(shape=shape, pixel_size=(0.3, 0.1, 0.1))
    operator = _fourier(3, 2, sampling=sampling)
    obj = _random(operator.object_shape, seed=1)
    torch.testing.assert_close(operator.normal_action()(obj), operator.adjoint(operator.apply(obj)))


def test_fourier_diagonal_normal_differentiates_the_transfer_after_a_no_grad_call():
    # A solver may iterate without a graph, then evaluate the fixed point with one on the same operator.
    transfer = _fourier(2, 2).transfer.clone().requires_grad_()
    operator = FourierDiagonal(transfer, SAMPLING, hermitian=True)  # already projected: no shared projection graph
    obj, weights = _random(operator.object_shape, seed=1), _random(operator.object_shape, seed=2)
    with torch.no_grad():
        operator.normal_action()(obj)
    (gradient,) = torch.autograd.grad(_inner(operator.normal_action()(obj), weights), transfer)
    (expected,) = torch.autograd.grad(_inner(operator.adjoint(operator.apply(obj)), weights), transfer)
    torch.testing.assert_close(gradient, expected)


def test_gram_follows_in_place_writes_to_the_transfer():
    operator = _fourier(2, 2)
    obj = _random(operator.object_shape, seed=1)
    operator.normal_action()(obj)
    with torch.no_grad():
        operator.transfer.mul_(2.0)
    torch.testing.assert_close(operator.normal_action()(obj), operator.adjoint(operator.apply(obj)))


def test_fourier_diagonal_projects_a_transfer_that_is_not_hermitian_so_its_norm_is_exact():
    # A purely imaginary DC coefficient: irfftn drops it, so the operator is 0 and so must its norm be.
    imaginary_dc = torch.zeros(1, 1, 4, 8, 4, 2, dtype=torch.float64)
    imaginary_dc[0, 0, 0, 0, 0, 1] = 1.0
    operator = FourierDiagonal(imaginary_dc, SAMPLING)
    assert operator.norm().item() == 0.0
    assert operator.apply(torch.ones(operator.object_shape, dtype=torch.float64)).abs().max() == 0
    # Any raw half spectrum (here with its crop's unpaired Nyquist rows) describes the operator apply implements.
    raw = torch.view_as_real(random_transfer(2, 2, SAMPLING.shape, seed=3, dtype=torch.float64)[..., :4])
    projected = FourierDiagonal(raw, SAMPLING)
    torch.testing.assert_close(projected.norm(), _power_norm(projected), rtol=1e-3, atol=0)


def test_fourier_diagonal_rejects_a_complex_transfer():
    with pytest.raises(ShapeError, match="view_as_real"):
        FourierDiagonal(random_transfer(1, 1, SAMPLING.shape)[..., :4], SAMPLING)


def test_as_fourier_exposes_the_closed_form_structure():
    fourier = _fourier(3, 2)
    assert fourier.as_fourier() is fourier
    assert ReshapedFourierDiagonal(fourier, SAMPLING, data_shape=(1, 12, 8, 6)).as_fourier() is fourier
    assert _generic(fourier).as_fourier() is None


class _SwitchableSystem(ImagingSystem):
    """One class that binds to a FourierDiagonal or, configured shift-variant, to a generic operator."""

    def __init__(self, *, shift_variant: bool) -> None:
        super().__init__(fields=("a",), channel_count=1)
        self.shift_variant = shift_variant
        self.operator_type = _Generic if shift_variant else FourierDiagonal

    @override
    def compute_operator(self, sampling: Sampling) -> LinearOperator:
        nz, ny, nx = sampling.shape
        blur = FourierDiagonal(torch.ones(1, 1, nz, ny, nx // 2 + 1), sampling)
        if not self.shift_variant:
            return blur
        return _generic(blur)


def test_operator_type_may_depend_on_the_instance():
    data = torch.zeros(1, *SAMPLING.shape)
    structured = ImagingModel(_SwitchableSystem(shift_variant=False), Tikhonov(), Gaussian())
    assert Wiener()(data, structured, SAMPLING).estimate.shape == (1, *SAMPLING.shape)
    shift_variant = ImagingModel(_SwitchableSystem(shift_variant=True), Tikhonov(), Gaussian())
    with pytest.raises(IncompatibleModelError, match="binds to _Generic"):
        Wiener()(data, shift_variant, SAMPLING)
    assert FISTA(loop=Loop(2))(data, shift_variant, SAMPLING).estimate.shape == (1, *SAMPLING.shape)


def test_bind_checks_the_declared_operator_type():
    system = _SwitchableSystem(shift_variant=True)
    system.operator_type = FourierDiagonal
    with pytest.raises(IncompatibleModelError, match="declares operator_type FourierDiagonal, but compute_operator"):
        system.bind(SAMPLING)


def test_operators_refuse_inputs_of_another_dtype_or_device():
    single = make_system(1.0).bind(SAMPLING)  # float32
    with pytest.raises(PlacementError, match=r"object is torch\.float64 on cpu but FourierDiagonal computes in"):
        single.apply(torch.zeros(single.object_shape, dtype=torch.float64))
    with pytest.raises(PlacementError, match=r"data is torch\.float64"):
        single.adjoint(torch.zeros(single.data_shape, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a second device")
def test_operators_refuse_an_input_on_another_device():
    cuda = _on_cuda(_fourier(3, 2))
    with pytest.raises(PlacementError, match=r"object is torch\.float64 on cpu"):
        cuda.apply(torch.ones(cuda.object_shape, dtype=torch.float64))
