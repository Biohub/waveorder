"""Contract checks any waveorder adapter can run in its own tests.

Each check raises ``AssertionError`` naming the broken contract. They are deterministic and cheap on small samplings;
run them in float64 when a tolerance matters. A ``check_*`` suite runs every check that applies to one kind of
adapter (system, operator, prior, noise, score, solver, search stage), including, on CUDA, that nothing reads back to
the host (:func:`assert_no_host_sync`)::

    check_system(lambda: MySystem(...), Sampling(shape=(4, 8, 6), pixel_size=(0.3, 0.1, 0.1)))
"""

import copy
import linecache
import warnings
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.func import vmap

from waveorder.errors import ParameterError
from waveorder.nn.functional import abs2, half_spectrum_weights, is_complex_matrix, to_half_spectrum
from waveorder.nn.model import ImagingModel
from waveorder.nn.noise import Noise
from waveorder.nn.operators import FourierDiagonal, LinearOperator, ReshapedFourierDiagonal
from waveorder.nn.parameters import Free, evaluate, free_specs, placement
from waveorder.nn.priors import GaussianObjectPrior, ObjectPrior
from waveorder.nn.sampling import Sampling
from waveorder.nn.scores import Score
from waveorder.nn.search.base import CalibrationObjective, Estimate, SearchStage, expect_known_names, score_candidates
from waveorder.nn.solvers import IterativeSolver, Solver
from waveorder.nn.systems.base import ImagingSystem, ShiftInvariantSystem
from waveorder.nn.verbs import InverseProblem


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _random(shape: tuple[int, ...], module: nn.Module, *, seed: int) -> Tensor:
    """Seeded standard normal tensor in the module's dtype and on its device, drawn on the CPU for determinism."""
    dtype, device = placement(module)
    return torch.randn(shape, generator=_generator(seed), dtype=dtype).to(device)


def _random_for(operator: LinearOperator, shape: tuple[int, ...], *, seed: int) -> Tensor:
    dtype, device = operator.placement
    return torch.randn(shape, generator=_generator(seed), dtype=dtype).to(device)


def assert_operator_adjoint(operator: LinearOperator, *, seed: int = 0, rtol: float = 1e-5) -> None:
    """Check ``<H f, g> = <f, H* g>`` for seeded random ``f`` and ``g`` in the operator's placement, in float64."""
    obj = _random_for(operator, operator.object_shape, seed=seed)
    data = _random_for(operator, operator.data_shape, seed=seed + 1)
    with torch.no_grad():
        lhs = (operator.apply(obj).double() * data.double()).sum()
        rhs = (obj.double() * operator.adjoint(data).double()).sum()
    scale = max(lhs.abs().item(), rhs.abs().item(), 1e-30)
    assert abs(lhs.item() - rhs.item()) <= rtol * scale, (
        f"{type(operator).__name__} adjoint mismatch: <Hf, g> = {lhs.item()}, <f, H*g> = {rhs.item()}"
    )


def assert_adjoint(system: ImagingSystem, sampling: Sampling, *, seed: int = 0, rtol: float = 1e-5) -> None:
    """Check ``<H f, g> = <f, H* g>`` on ``system.bind(sampling)`` (see :func:`assert_operator_adjoint`)."""
    with torch.no_grad():
        operator = system.bind(sampling)
    assert_operator_adjoint(operator, seed=seed, rtol=rtol)


def assert_operator_normal(operator: LinearOperator, *, seed: int = 0, rtol: float = 1e-5) -> None:
    """Check that ``normal_action()(f)``, the action the iterative solvers run, equals ``adjoint(apply(f))``."""
    obj = _random_for(operator, operator.object_shape, seed=seed)
    with torch.no_grad():
        expected = operator.adjoint(operator.apply(obj))
        torch.testing.assert_close(
            operator.normal_action()(obj),
            expected,
            rtol=rtol,
            atol=rtol * expected.abs().max().item(),
            msg=lambda message: f"{type(operator).__name__}.normal_action differs from adjoint(apply): {message}",
        )


def _estimated_norm(operator: LinearOperator, *, iterations: int, seed: int) -> Tensor:
    """Return a power-iteration estimate of ``||H||_2`` from a seeded start: it approaches the norm from below."""
    vector = _random_for(operator, operator.object_shape, seed=seed)
    vector = vector / torch.linalg.vector_norm(vector)
    estimate = torch.zeros((), dtype=vector.dtype, device=vector.device)
    for _ in range(iterations):
        normal = operator.adjoint(operator.apply(vector))
        estimate = torch.linalg.vector_norm(normal)
        vector = normal / estimate.clamp_min(torch.finfo(vector.dtype).tiny)
    return estimate.sqrt()


def assert_operator_norm(system: ImagingSystem, sampling: Sampling, *, rtol: float = 1e-3, slack: float = 1.1) -> float:
    """Check that the bound operator's ``norm()`` is a safe upper bound on ``||H||_2``, tight where it can be.

    The reference is a 200-step power iteration, which approaches ``||H||_2`` from below; a smaller ``norm()`` would
    make first-order solvers unstable. A Fourier-diagonal operator's norm must also be within ``slack`` of it.

    Returns
    -------
    float
        The ratio ``norm() / ||H||``.
    """
    with torch.no_grad():
        operator = system.bind(sampling)
        claimed = operator.norm()
        estimated = _estimated_norm(operator, iterations=200, seed=0).to(claimed.dtype)
    assert claimed >= estimated * (1 - rtol), f"norm() = {claimed.item()} < ||H|| ~ {estimated.item()}"
    ratio = (claimed / estimated.clamp_min(torch.finfo(estimated.dtype).tiny)).item()
    if isinstance(operator, FourierDiagonal | ReshapedFourierDiagonal):
        assert ratio <= slack * (1 + rtol), (
            f"norm() = {claimed.item()} is looser than {slack} x ||H|| ~ {estimated.item()}"
        )
    return ratio


def assert_preserves_nonnegativity(
    system: ImagingSystem, sampling: Sampling, *, seed: int = 0, ringing: float = 1e-2
) -> None:
    """Check a system's ``preserves_nonnegativity`` claim: nonnegative objects give nonnegative data.

    The PSF and the image of a random nonnegative object must not go below ``-ringing`` times their maximum; the
    default admits the ringing of band-limiting a nonnegative PSF to the grid.
    """
    assert system.preserves_nonnegativity, f"{type(system).__name__} does not claim preserves_nonnegativity"
    point = torch.zeros(system.object_shape(sampling), dtype=placement(system)[0], device=placement(system)[1])
    point[..., 0, 0, 0] = 1.0
    random = _random(system.object_shape(sampling), system, seed=seed).abs()
    with torch.no_grad():
        operator = system.bind(sampling)
        for what, obj in (("point-spread function", point), ("image of a nonnegative object", random)):
            image = operator.apply(obj)
            lowest, highest = image.min().item(), image.max().item()
            assert lowest >= -ringing * highest, f"{what} reaches {lowest}, below -{ringing} x its maximum {highest}"


def assert_solver_descends(
    solver: Solver, *, data: Tensor, model: ImagingModel, sampling: Sampling, rtol: float = 1e-6
) -> None:
    """Check that a solve returns a finite estimate whose objective is no worse than the solver's start.

    The start is an :class:`~waveorder.nn.solvers.IterativeSolver`'s first state, else the zero object.
    """
    with torch.no_grad():
        result = solver(data, model, sampling)
        operator = model.system.bind(sampling)
        if isinstance(solver, IterativeSolver):
            problem = InverseProblem.of(data, model, sampling, operator=operator)
            reference = solver.objective(solver.start(problem), problem)
        else:
            reference = model.objective(data, torch.zeros_like(result.estimate), sampling, operator=operator)
    assert bool(torch.isfinite(result.estimate).all()), f"{type(solver).__name__} returned a non-finite estimate"
    assert result.objective <= reference + rtol * reference.abs(), (
        f"{type(solver).__name__} ended at objective {result.objective.item()}, above its start {reference.item()}"
    )


def assert_prox_optimal(
    prior: ObjectPrior, obj: Tensor, *, step: float, scale: float = 1.0, trials: int = 16, seed: int = 0
) -> None:
    """Check ``prior.prox(v, step)`` against nearby points of the support on ``step*P(x) + |x-v|^2/2``.

    A zero step must return the projection of ``v``. Approximate proximal maps pass when their error is below the
    perturbation size, so give such priors enough inner iterations in the test.
    """
    name, dtype = type(prior).__name__, obj.dtype
    step_t = obj.new_tensor(step)
    scale_t = obj.new_tensor(scale)

    def objective(x: Tensor) -> Tensor:
        return step_t * prior.penalty(x, scale=scale_t) + 0.5 * ((x - obj) ** 2).sum()

    with torch.no_grad():
        at_zero = prior.prox(obj, obj.new_zeros(()), scale=scale_t)
        torch.testing.assert_close(
            at_zero,
            prior.project(obj),
            msg=lambda m: f"{name}.prox with step 0 must be {name}.project in {dtype}: {m}",
        )
        proximal = prior.prox(obj, step_t, scale=scale_t)
        assert torch.equal(prior.project(proximal), proximal), f"{name}.prox left the prior's support in {dtype}"
        best = objective(proximal)
        for trial in range(trials):
            direction = torch.randn(obj.shape, generator=_generator(seed + trial), dtype=obj.dtype)
            direction = direction.to(obj.device)
            candidate = prior.project(proximal + 1e-2 * direction / direction.norm() * proximal.norm().clamp_min(1))
            assert objective(candidate) >= best - 1e-6 * best.abs().clamp_min(1), (
                f"{name}.prox and {name}.penalty disagree in {dtype}: at prox(v, step) the objective step * "
                f"penalty(x) + |x - v|^2 / 2 is {best.item()}, but a nearby point scores "
                f"{objective(candidate).item()}; either prox does not minimize this penalty or penalty is not the "
                "function prox minimizes"
            )


def assert_noise_gradient(noise: Noise, data: Tensor, mean: Tensor, *, rtol: float = 1e-5) -> None:
    """Check ``noise.gradient`` against autograd of ``noise.nll`` with respect to the mean."""
    mean = mean.detach().requires_grad_()
    (expected,) = torch.autograd.grad(noise.nll(data, mean), mean)
    with torch.no_grad():
        actual = noise.gradient(data, mean)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=rtol * expected.abs().max().item())


def assert_vmap_matches_loop[M: nn.Module](
    fn: Callable[[M], Tensor], module: M, *, name: str, values: Tensor, atol: float = 1e-5
) -> None:
    """Check that ``vmap`` over parameter ``name`` agrees with a Python loop of :func:`~waveorder.nn.evaluate`."""
    with torch.no_grad():
        batched = vmap(lambda value: evaluate(fn, module, {name: value}))(values)
        looped = torch.stack([evaluate(fn, module, {name: value}) for value in values])
    torch.testing.assert_close(batched, looped, atol=atol, rtol=0)


def assert_gradients_match_finite_differences[M: nn.Module](
    fn: Callable[[M], Tensor], module: M, *, eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3
) -> None:
    """Run ``torch.autograd.gradcheck`` of a scalar ``fn(module)`` over every free parameter of a float64 ``module``."""
    names = list(free_specs(module))
    point = tuple(module.get_parameter(name).detach().double().requires_grad_() for name in names)

    def scalar(*values: Tensor) -> Tensor:
        return evaluate(fn, module, dict(zip(names, values, strict=True)))

    assert torch.autograd.gradcheck(scalar, point, eps=eps, atol=atol, rtol=rtol)


def assert_state_roundtrip[M: nn.Module](module: M, build: Callable[[], M], *, probe: Callable[[M], Tensor]) -> None:
    """Check that ``deepcopy`` and ``load_state_dict`` into a fresh ``build()`` keep specs and the output of ``probe``.

    Comparing ``probe(module)``, not only the state tensors, catches Python state a state dict does not carry.
    """
    copied = copy.deepcopy(module)
    assert free_specs(copied) == free_specs(module), "deepcopy lost Free specs"
    fresh = build()
    fresh.load_state_dict(module.state_dict())
    with torch.no_grad():
        expected = probe(module)
        torch.testing.assert_close(probe(copied), expected, msg="deepcopy changed the module's output")
        torch.testing.assert_close(probe(fresh), expected, msg="load_state_dict did not restore the module's output")


def check_system[S: ImagingSystem](
    build: Callable[[], S], sampling: Sampling, *, seed: int = 0, rtol: float = 1e-5, ringing: float = 1e-2
) -> None:
    """Run every check an imaging system must pass, on ``build()`` and ``sampling``.

    The bound operator must be an :attr:`~ImagingSystem.operator_type` and pass :func:`check_operator` with a safe
    ``norm()``; the system must survive ``deepcopy`` and a state-dict round trip; ``preserves_nonnegativity`` must
    hold up to ``ringing``; gradients of free quantities must match finite differences in float64; and on CUDA,
    ``bind`` and ``apply`` must not sync with the host. Any exception the system raises is an ``AssertionError``.
    """
    assert not isinstance(build, ImagingSystem), (
        f"check_system takes a builder, got the system {type(build).__name__}; pass `lambda: system`"
    )
    with _assertions_only("check_system"):
        _check_system(build, sampling, seed=seed, rtol=rtol, ringing=ringing)


def _check_system[S: ImagingSystem](
    build: Callable[[], S], sampling: Sampling, *, seed: int, rtol: float, ringing: float
) -> None:
    system = build()
    assert_adjoint(system, sampling, seed=seed, rtol=rtol)
    assert_operator_norm(system, sampling)
    with torch.no_grad():
        operator = system.bind(sampling)
    assert isinstance(operator, system.operator_type), (
        f"{type(system).__name__} declares operator_type {system.operator_type.__name__}, "
        f"binds {type(operator).__name__}"
    )
    check_operator(operator, seed=seed, rtol=rtol)
    if isinstance(system, ShiftInvariantSystem) and system.real_transfer:
        _expect_real_transfer(system, sampling)
    if system.preserves_nonnegativity:
        assert_preserves_nonnegativity(system, sampling, seed=seed, ringing=ringing)
    obj = _random(system.object_shape(sampling), system, seed=seed)
    if obj.is_cuda:
        with torch.no_grad():
            assert_no_host_sync(lambda: system.bind(sampling).apply(obj), what=f"{type(system).__name__}.bind")
    assert_state_roundtrip(system, build, probe=lambda module: module(obj, sampling))
    if free_specs(system):
        double = build().double()
        probe = obj.double()
        weights = _random(double.data_shape(sampling), double, seed=seed + 1)
        assert_gradients_match_finite_differences(lambda module: (module(probe, sampling) * weights).sum(), double)


def _expect_real_transfer(system: ShiftInvariantSystem, sampling: Sampling, *, rtol: float = 1e-5) -> None:
    """Check that a declared ``real_transfer`` drops only an imaginary part near zero from :meth:`compute_transfer`."""
    with torch.no_grad():
        raw = system.compute_transfer(sampling)
        if raw.is_complex():
            real, imaginary = raw.real, raw.imag
        elif is_complex_matrix(raw):
            real, imaginary = raw[..., 0], raw[..., 1]
        else:
            return
        size, dropped = real.abs().max().item(), imaginary.abs().max().item()
    assert dropped <= rtol * max(size, 1e-30), (
        f"{type(system).__name__} declares real_transfer, but its transfer has an imaginary part of {dropped:.3g} "
        f"(largest real part {size:.3g}), which the real storage would drop"
    )


def check_operator(operator: LinearOperator, *, seed: int = 0, rtol: float = 1e-6) -> None:
    """Run every check a linear operator must pass; every failure, and any exception, is an ``AssertionError``.

    Checked: the adjoint identity, ``normal_action`` equal to ``adjoint(apply)``, an ``as_fourier`` that applies as
    the operator does, and on CUDA, no host sync in ``apply``, ``adjoint`` and ``normal_action``.
    """
    assert not isinstance(operator, ImagingSystem), (
        f"check_operator takes a bound operator, got the system {type(operator).__name__}; pass `system.bind(sampling)`"
    )
    with _assertions_only(f"check_operator({type(operator).__name__})"):
        _check_operator(operator, seed=seed, rtol=rtol)


@contextmanager
def _assertions_only(what: str) -> Generator[None]:
    """Re-raise any exception but an :class:`AssertionError` as one, naming the check and the exception."""
    try:
        yield
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f"{what} raised {type(error).__name__}: {error}") from error


def _check_operator(operator: LinearOperator, *, seed: int, rtol: float) -> None:
    name = type(operator).__name__
    assert_operator_adjoint(operator, seed=seed, rtol=rtol)
    assert_operator_normal(operator, seed=seed, rtol=rtol)
    obj = _random_for(operator, operator.object_shape, seed=seed + 2)
    _expect_fourier(operator, obj, rtol=rtol)
    if not obj.is_cuda:
        return
    data = _random_for(operator, operator.data_shape, seed=seed + 3)
    with torch.no_grad():
        for action, value in ((operator.apply, obj), (operator.adjoint, data)):
            assert_no_host_sync(action, value, what=f"{name}.{action.__name__}")
        assert_no_host_sync(lambda: operator.normal_action()(obj), what=f"{name}.normal_action")


def _expect_fourier(operator: LinearOperator, obj: Tensor, *, rtol: float) -> None:
    """Check that a declared :meth:`~LinearOperator.as_fourier` applies as the operator does, its data reshaped."""
    fourier = operator.as_fourier()
    if fourier is None:
        return
    name = type(operator).__name__
    assert isinstance(fourier, FourierDiagonal), f"{name}.as_fourier returned {type(fourier).__name__}"
    with torch.no_grad():
        expected = operator.apply(obj)
        torch.testing.assert_close(
            fourier.apply(obj).reshape(expected.shape),
            expected,
            rtol=rtol,
            atol=rtol * expected.abs().max().item(),
            msg=lambda m: f"{name}.as_fourier disagrees with apply: {m}",
        )


def check_score(score: Score, *, data: Tensor, model: ImagingModel, sampling: Sampling, rtol: float = 1e-4) -> None:
    """Run every check a calibration score must pass on one problem.

    The value must be a finite real 0-dim tensor; with free quantities, ``vmap`` over candidate values
    (:func:`~waveorder.nn.score_candidates`) must match a loop, and in a float64 model gradients must match finite
    differences; on CUDA it must not sync with the host.
    """
    name = type(score).__name__
    if data.is_cuda:
        with torch.no_grad():
            assert_no_host_sync(score, data, model, sampling, what=name)
    with torch.no_grad():
        value = score(data, model, sampling)
    assert isinstance(value, Tensor), f"{name} returned {type(value).__name__}, not a tensor"
    assert value.dim() == 0, f"{name} returned shape {tuple(value.shape)}, not a 0-dim tensor"
    assert value.is_floating_point(), f"{name} returned {value.dtype}, not a real float"
    assert bool(torch.isfinite(value)), f"{name} returned {value.item()}"
    if free_specs(model):
        _expect_candidates_vmap(score, data=data, model=model, sampling=sampling, rtol=rtol)
    if free_specs(model) and placement(model)[0] == torch.float64:
        assert_gradients_match_finite_differences(lambda module: score(data, module, sampling), model)


def _close(actual: Tensor, expected: Tensor, *, rtol: float, what: str) -> None:
    scale = expected.detach().abs().max().item() if expected.numel() else 0.0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=rtol * scale, msg=lambda m: f"{what}: {m}")


def _candidate_values(value: Tensor, spec: Free) -> Tensor:
    """Three values around ``value``, half a tolerance (or a tenth of the bounds, or 1%) apart, within the bounds."""
    if spec.tolerance is not None:
        step = 0.5 * spec.tolerance
    elif spec.bounds is not None:
        step = 0.1 * (spec.bounds[1] - spec.bounds[0])
    else:
        step = 0.01 * max(abs(spec.nominal), 1.0)
    offsets = torch.tensor([-step, 0.0, step], dtype=value.dtype).to(value.device)
    values = value.detach() + offsets
    return values if spec.bounds is None else values.clamp(*spec.bounds)


def _expect_candidates_vmap(
    score: Score, *, data: Tensor, model: ImagingModel, sampling: Sampling, rtol: float
) -> None:
    name = type(score).__name__
    specs = free_specs(model)
    candidates = {key: _candidate_values(model.get_parameter(key), spec) for key, spec in specs.items()}
    with torch.no_grad():
        try:
            batched = score_candidates(data, model, sampling, score=score, candidates=candidates)
        except RuntimeError as error:
            raise AssertionError(
                f"{name} does not vmap over candidate values of {sorted(specs)}, as every candidate search stage "
                f"does (an in-place write of a parameter into a buffer?): {error}"
            ) from error
        for index in range(3):
            point = {key: values[index] for key, values in candidates.items()}
            looped = evaluate(lambda module: score(data, module, sampling), model, point)
            _close(batched[index], looped, rtol=rtol, what=f"{name} vmapped over candidates [{index}]")


def _candidates(model: ImagingModel) -> dict[str, Tensor]:
    """Three values of every free quantity of ``model``, else of every 0-d fixed one (1% apart)."""
    specs = free_specs(model)
    if specs:
        return {key: _candidate_values(model.get_parameter(key), spec) for key, spec in specs.items()}
    return {
        name: buffer.detach() * buffer.new_tensor([0.99, 1.0, 1.01])
        for name, buffer in model.named_buffers()
        if buffer.dim() == 0 and buffer.is_floating_point()
    }


def assert_solver_vmaps_over_candidates(
    solver: Solver, *, data: Tensor, model: ImagingModel, sampling: Sampling, rtol: float = 1e-4
) -> None:
    """Check that ``solver`` runs under ``vmap`` over candidate values of the model's quantities, as in a grid search.

    The candidates are three values of every free quantity (of every 0-d fixed one when none is free); the estimates
    must match a loop.
    """
    name = type(solver).__name__
    candidates = _candidates(model)
    assert candidates, f"the model has no scalar quantity to map {name} over"

    def solve(point: Mapping[str, Tensor]) -> Tensor:
        return evaluate(lambda module: solver(data, module, sampling).estimate, model, point)

    with torch.no_grad():
        try:
            batched = vmap(solve)(candidates)
        except RuntimeError as error:
            raise AssertionError(
                f"{name} does not run under vmap over candidate values of {sorted(candidates)}, as a grid search's "
                f"score runs it: {error}"
            ) from error
        for index in range(3):
            looped = solve({key: values[index] for key, values in candidates.items()})
            _close(batched[index], looped, rtol=rtol, what=f"{name} vmapped over candidates [{index}]")


_ODD_PIXEL_SIZE = (0.7, 1.3, 0.9)
"""A pixel size in um that no voxel-unit spectrum can depend on and every physical-unit one does."""


def check_prior(prior: ObjectPrior, obj: Tensor, *, step: float = 0.7, scale: float = 1.3) -> None:
    """Run every check an object prior must pass, at ``obj`` ``(F, Z, Y, X)``.

    The proximal map must be optimal for the penalty and the penalty a finite real 0-dim tensor. A
    :class:`~waveorder.nn.GaussianObjectPrior`'s penalty must equal ``sum_k |F_k|^2 / (2 S_k)`` over the orthonormal
    spectrum, asked for on an anisotropic pixel size so that a spectrum in physical units fails. On CUDA, ``prox`` and
    ``penalty`` must not sync with the host.
    """
    name = type(prior).__name__
    if obj.is_cuda:
        step_t, scale_t = (torch.full((), value, dtype=obj.dtype, device=obj.device) for value in (step, scale))
        with torch.no_grad():
            assert_no_host_sync(lambda: prior.prox(obj, step_t, scale=scale_t), what=f"{name}.prox")
            assert_no_host_sync(lambda: prior.penalty(obj, scale=scale_t), what=f"{name}.penalty")
    assert_prox_optimal(prior, obj, step=step, scale=scale)
    scale_t = obj.new_tensor(scale)
    with torch.no_grad():
        penalty = prior.penalty(obj, scale=scale_t)
        assert penalty.dim() == 0, f"{name}.penalty has shape {tuple(penalty.shape)}, not a 0-dim tensor"
        assert not penalty.is_complex(), f"{name}.penalty is {penalty.dtype}, not real"
        assert bool(torch.isfinite(penalty)), f"{name}.penalty is {penalty.item()}"
        if isinstance(prior, GaussianObjectPrior) and not prior.nonnegative:
            _, nz, ny, nx = obj.shape
            sampling = Sampling(shape=(nz, ny, nx), pixel_size=_ODD_PIXEL_SIZE)
            spectrum = prior.spectrum(sampling, scale=scale_t)
            power = abs2(to_half_spectrum(obj, norm="ortho"))
            weights = half_spectrum_weights(obj.shape[-1], device=obj.device, dtype=obj.dtype)
            expected = 0.5 * (power * weights / spectrum).sum()
            torch.testing.assert_close(
                penalty, expected, msg=lambda m: f"{name}.penalty disagrees with its spectrum: {m}"
            )


def check_noise(noise: Noise, data: Tensor, mean: Tensor, *, seed: int = 0) -> None:
    """Run every check a noise model must pass, at ``data`` and ``mean`` of one shape.

    ``gradient`` must match autograd of ``nll``; ``nll`` must be a finite real 0-dim tensor; ``sample`` must keep the
    mean's shape, dtype and device; a reported white Gaussian variance must be a positive 0-dim tensor.
    """
    assert_noise_gradient(noise, data, mean)
    with torch.no_grad():
        nll = noise.nll(data, mean)
        assert nll.dim() == 0, f"nll has shape {tuple(nll.shape)}, not a 0-dim tensor"
        assert bool(torch.isfinite(nll)), f"nll is {nll.item()}"
        drawn = noise.sample(mean, generator=_generator(seed) if mean.device.type == "cpu" else None)
        assert (drawn.shape, drawn.dtype, drawn.device) == (mean.shape, mean.dtype, mean.device), (
            f"sample returned {tuple(drawn.shape)} {drawn.dtype} on {drawn.device}"
        )
        assert bool(torch.isfinite(drawn).all()), "sample returned non-finite values"
        variance = noise.white_gaussian_variance()
        if variance is not None:
            assert variance.dim() == 0, f"white_gaussian_variance has shape {tuple(variance.shape)}"
            assert bool(variance > 0), f"white_gaussian_variance is {variance.item()}"


def check_solver(solver: Solver, *, data: Tensor, model: ImagingModel, sampling: Sampling, rtol: float = 1e-4) -> None:
    """Run every check a solver must pass on one problem.

    The solve must descend from its start. A solver without ``Loop(tolerance=...)`` must also match a loop under
    ``vmap`` over candidate values and, on CUDA, run without a host sync.
    """
    assert_solver_descends(solver, data=data, model=model, sampling=sampling)
    if isinstance(solver, IterativeSolver) and solver.loop.tolerance is not None:
        return
    assert_solver_vmaps_over_candidates(solver, data=data, model=model, sampling=sampling, rtol=rtol)
    if data.is_cuda:
        with torch.no_grad():
            assert_no_host_sync(lambda values: solver(values, model, sampling), data, what=type(solver).__name__)


_TORCH_ROOT = str(Path(torch.__file__).parent)
_WAVEORDER_ROOT = str(Path(__file__).parent)


def _culprit(error: BaseException) -> str:
    """Return where ``error``'s read-back was asked for: the innermost caller frame and the waveorder frame under it."""
    frames: list[tuple[str, int, str, dict[str, object]]] = []
    trace = error.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        frames.append((code.co_filename, trace.tb_lineno, code.co_name, trace.tb_frame.f_locals))
        trace = trace.tb_next
    outside = [frame for frame in frames if not frame[0].startswith(_TORCH_ROOT)]
    if not outside:
        return "inside torch"

    def where(frame: tuple[str, int, str, dict[str, object]]) -> str:
        filename, line, function, _ = frame
        return f"{filename}:{line} in {function} ({linecache.getline(filename, line).strip()})"

    library = outside[-1]
    callers = [frame for frame in outside if not frame[0].startswith(_WAVEORDER_ROOT)]
    report = where(library) if not callers or callers[-1] is library else f"{where(callers[-1])} via {where(library)}"
    if library[2] == "read_flag":
        tolerant = [frame for frame in outside if frame[3].get("tolerance") is not None]
        if tolerant:
            _, _, function, local = tolerant[0]
            report += (
                f"; a tolerance was passed (tolerance={local['tolerance']} to {function}), which reads back to check "
                "it: use tolerance=None and read the result's residual and finite on the device"
            )
    return report


def assert_no_host_sync[R](fn: Callable[..., R], *args: object, what: str | None = None) -> R:
    """Run ``fn(*args)`` under ``torch.cuda.set_sync_debug_mode("error")`` and return its result.

    A host sync (``.item()``, a copy to pageable memory, a data-dependent shape) fails with the frame that issued it.
    Without CUDA ``fn`` simply runs. In this mode ``bool()`` of a CUDA tensor aborts the process instead of raising,
    so code meant to pass reads back with ``.item()``.
    """
    if not torch.cuda.is_available():
        return fn(*args)
    name = what or getattr(fn, "__qualname__", "the call")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        with warnings.catch_warnings():
            # torch warns that the sync debug mode is a prototype each time it is switched on.
            warnings.filterwarnings("ignore", message="Synchronization debug mode is a prototype")
            torch.cuda.set_sync_debug_mode("error")
            return fn(*args)
    except RuntimeError as error:
        if "synchroniz" in str(error):
            raise AssertionError(f"{name} reads values back to the host at {_culprit(error)}: {error}") from error
        raise
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def check_search_stage(
    stage: SearchStage, objective: CalibrationObjective, start: Estimate, *, rtol: float = 1e-6
) -> None:
    """Run every check a calibration search stage must pass from ``start``.

    The fields in ``quantity_keyed`` must be real numbers or mappings from free quantities to real numbers. The result
    must hold the same quantities on the start's device, within their bounds, with an objective no worse than the
    start's that matches the objective at the returned point. On CUDA the stage must not sync with the host unless
    it declares ``reads_back = True``.
    """
    device = start.objective.device
    name = type(stage).__name__
    keyed = getattr(type(stage), "quantity_keyed", None)
    assert isinstance(keyed, tuple), (
        f"{name} must declare quantity_keyed: ClassVar[tuple[str, ...]] (() for none), got {keyed!r}"
    )
    for field in keyed:
        assert hasattr(stage, field), f"{name}.quantity_keyed names {field!r}, which {name} has no field of"
        value = getattr(stage, field)
        assert isinstance(value, Mapping | int | float), (
            f"{name}.{field} is declared quantity-keyed but is a {type(value).__name__}, not a mapping or a number"
        )
        entries = dict(value) if isinstance(value, Mapping) else {"": value}
        for key, entry in entries.items():
            real = isinstance(entry, int | float) and not isinstance(entry, bool)
            assert real, f"{name}.{field}{'.' if key else ''}{key} must be a real number, not a {type(entry).__name__}"
        if isinstance(value, Mapping):
            try:
                expect_known_names(value, objective.specs, path=f"{name}.{field}")
            except ParameterError as error:
                raise AssertionError(str(error)) from error
    if device.type == "cuda" and not stage.reads_back:
        result = assert_no_host_sync(lambda: stage.run(objective, start=start), what=name)
    else:
        result = stage.run(objective, start=start)
    assert set(result.point) == set(start.point), f"{name} returned quantities {sorted(result.point)}"
    for key, value in (*result.point.items(), ("objective", result.objective)):
        assert value.device == device, f"{name} returned {key} on {value.device}, the start is on {device}"
    for key, value in result.point.items():
        bounds = objective.specs[key].bounds
        if bounds is not None:
            # The bounds rounded to the value's dtype: a float32 grid's 1.2 is 1.2000000476837158.
            low, high = (torch.tensor(bound, dtype=value.dtype).item() for bound in bounds)
            assert low <= value.item() <= high, f"{name} left the bounds {bounds} of {key}: {value.item()}"
    best, reference = result.objective.item(), start.objective.item()
    assert best <= reference + rtol * abs(reference), f"{name} ended at objective {best}, above its start {reference}"
    with torch.no_grad():
        recomputed = objective(result.point).item()
    assert abs(recomputed - best) <= 1e-4 * max(abs(best), 1.0), (
        f"{name} reported objective {best}, but the objective at its point is {recomputed}"
    )
