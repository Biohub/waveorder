"""Fixed and free scalar quantities on ``nn.Module`` trees, the priors over the free ones, and their evaluation.

:func:`evaluate` calls any function of a module with parameter values swapped in (``torch.func.functional_call``);
the module's own parameters are left unchanged.
"""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Concatenate, Self, cast, override

import torch
from torch import Tensor, nn
from torch.func import functional_call

from waveorder.errors import ConfigurationError, ParameterError
from waveorder.registry import VALUE, register


@register("free", kind=VALUE)
@dataclass(frozen=True, slots=True)
class Free:
    """A quantity estimated from data, with its prior, in the quantity's own units.

    Only quantities built with ``Free(...)`` are calibrated; ``requires_grad_(False)`` does not hold one fixed. To fix
    a quantity, build its part with a plain number.

    Parameters
    ----------
    nominal : float
        Centre of the prior and the starting value of the estimate.
    tolerance : float, optional
        Standard deviation of a Gaussian prior around ``nominal``. ``None`` means a flat prior. Keyword-only.
    bounds : tuple[float, float], optional
        Hard limits; calibration clamps the estimate to them after each step. Keyword-only.

    Raises
    ------
    ParameterError
        If ``nominal`` is not finite, ``tolerance`` is not positive and finite, ``bounds`` are not finite with
        ``low < high``, or ``nominal`` lies outside them; ``path`` is the field.
    """

    nominal: float
    _: KW_ONLY
    tolerance: float | None = None
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.nominal):
            raise ParameterError(f"nominal must be finite, got {self.nominal}", path="nominal")
        if self.tolerance is not None and not (math.isfinite(self.tolerance) and self.tolerance > 0):
            raise ParameterError(f"tolerance must be positive and finite, got {self.tolerance}", path="tolerance")
        if self.bounds is not None:
            low, high = self.bounds
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ParameterError(f"bounds must be finite with low < high, got {self.bounds}", path="bounds")
            if not low <= self.nominal <= high:
                raise ParameterError(f"nominal = {self.nominal} lies outside bounds = {self.bounds}", path="nominal")


type Quantity = float | Free
"""A plain number (fixed, stored as a buffer) or a :class:`Free` value (estimated, stored as an ``nn.Parameter``)."""


def value_range(value: Quantity) -> tuple[float, float] | None:
    """Return the values a quantity can take: ``(v, v)`` if fixed, its bounds if free, ``None`` if unbounded."""
    match value:
        case Free(bounds=(low, high)):
            return (low, high)
        case Free():
            return None
        case _:
            return (float(value), float(value))


def expect_positive(value: Quantity, *, what: str, path: str | None = None) -> None:
    """Raise :class:`ParameterError` unless ``value`` stays strictly positive.

    A fixed value must be positive; a free one needs ``bounds`` with a positive lower limit. ``what`` names the
    quantity in the message; ``path`` is its dotted parameter path, carried on the error.
    """
    match value:
        case Free(bounds=bounds) if bounds is None or bounds[0] <= 0:
            raise ParameterError(
                f"{what} must stay positive; give Free(...) bounds with a positive lower limit, got {value}", path=path
            )
        case Free():
            pass
        case _ if not float(value) > 0:
            raise ParameterError(f"{what} must be positive, got {value}", path=path)


def expect_fixed(value: Quantity, *, what: str, path: str) -> None:
    """Raise :class:`ParameterError` at ``path`` unless ``value`` is a positive finite number, not a :class:`Free`.

    For a wavelength or a refractive index: they set the transfer function's grid, which is fixed before calibration.
    """
    match value:
        case Free():
            raise ParameterError(
                f"{what} cannot be free: it sets the grid the transfer function is computed on, which is fixed "
                f"before a calibration; pass a number, got {value}",
                path=path,
            )
        case _ if not (math.isfinite(value) and value > 0):
            raise ParameterError(f"{what} must be positive and finite, got {value}", path=path)


class CalibratableModule(nn.Module):
    """An ``nn.Module`` whose scalar quantities are either fixed buffers or free parameters with a prior.

    :meth:`register_quantity` stores a plain number as a 0-d buffer and a :class:`Free` value as a 0-d
    ``nn.Parameter``, so ``named_parameters()`` lists exactly the free quantities under their dotted names. ``repr``
    lists the quantities (:meth:`extra_repr`).
    """

    def __init__(self) -> None:
        super().__init__()
        self._free: dict[str, Free] = {}
        self._fixed: dict[str, float] = {}
        self._quantity_order: list[str] = []

    def register_quantity(self, name: str, value: Quantity, *, positive: bool = False) -> None:
        """Register a fixed or free scalar quantity under ``name``.

        Parameters
        ----------
        name : str
            Attribute name; it becomes the last component of the dotted parameter name.
        value : float or Free
            A plain number (any ``numbers.Real``, stored as ``float(value)``) is fixed; a :class:`Free` value is
            estimated.
        positive : bool
            Whether the quantity must stay strictly positive (a variance, an aperture, a weight); see
            :func:`expect_positive`.

        Raises
        ------
        ParameterError
            For a non-finite value, or a positive quantity that could leave the positive reals; ``path`` is ``name``.
        """
        owner = f"{type(self).__name__}.{name}"
        if positive:
            expect_positive(value, what=owner, path=name)
        match value:
            case Free(nominal=nominal):
                self.register_parameter(name, nn.Parameter(torch.tensor(nominal)))
                self._free[name] = value
            case _:
                number = float(value)
                if not math.isfinite(number):
                    raise ParameterError(f"{owner} must be finite, got {value}", path=name)
                self.register_buffer(name, torch.tensor(number))
                self._fixed[name] = number
        self._quantity_order.append(name)

    def quantity(self, name: str) -> Tensor:
        """Return the current tensor of quantity ``name``, including values swapped in by ``functional_call``."""
        value = getattr(self, name)
        if not isinstance(value, Tensor):
            raise ParameterError(f"{type(self).__name__}.{name} is not a quantity", path=name)
        return value

    @override
    def extra_repr(self) -> str:
        """Return the quantities as ``name=value``, in registration order.

        A fixed quantity shows its value, ``numerical_aperture=1.2``; a free one its spec, followed by ``at 1.213`` once
        its value has left the nominal one. Subclasses with other settings join them with :func:`repr_parts`.
        """
        return ", ".join(f"{name}={self._described(name)}" for name in self._quantity_order)

    def _described(self, name: str) -> str:
        current = _shown(getattr(self, name, None))
        spec = self._free.get(name)
        if spec is None:
            return f"{self._fixed[name]:g}" if current is None else current
        text = _spec_text(spec)
        return text if current is None or current == f"{spec.nominal:.6g}" else f"{text} at {current}"

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """Convert like ``nn.Module``, then re-fill fixed quantities from their exact Python floats.

        A fixed buffer is built in the default dtype, so ``.double()`` alone would turn ``1.2`` into
        ``1.2000000476837158``. A buffer loaded with another value from a ``state_dict`` is left as converted.
        """
        nominal = [name for name, value in self._fixed.items() if _holds(getattr(self, name), value)]
        super()._apply(fn, recurse=recurse)
        with torch.no_grad():
            for name in nominal:
                buffer = getattr(self, name)
                if isinstance(buffer, Tensor) and buffer.is_floating_point():
                    buffer.fill_(self._fixed[name])
        return self

    def free_quantities(self) -> dict[str, Free]:
        """Return the :class:`Free` specs registered directly on this module, keyed by attribute name."""
        return dict(self._free)

    def fixed_quantities(self) -> dict[str, float]:
        """Return the number each fixed quantity of this module was registered with, keyed by attribute name.

        These are the exact Python floats; the buffers hold them rounded to their dtype, or a value loaded since.
        """
        return dict(self._fixed)

    def current_argument(self, name: str, given: object) -> object:
        """Return constructor argument ``name`` as it would be given now: a quantity at its current value.

        ``given`` is what the constructor received. A free quantity becomes its :class:`Free` spec with the current
        value as nominal (clamped to its bounds), a fixed one the number :func:`quantity_number` writes. A mapping of
        quantities (Zernike terms) is followed term by term; any other argument is ``given``.
        """
        if isinstance(given, Mapping) and given and all(key in self._quantity_order for key in given):
            return {key: self.current_argument(key, value) for key, value in given.items()}
        tensor = getattr(self, name, None) if name in self._quantity_order else None
        if not isinstance(tensor, Tensor) or tensor.numel() != 1:
            return given
        spec = self._free.get(name)
        if spec is None:
            return quantity_number(tensor, self._fixed[name])
        value = quantity_number(tensor, spec.nominal)
        if spec.bounds is not None:
            value = min(max(value, spec.bounds[0]), spec.bounds[1])
        return dataclasses.replace(spec, nominal=value)


def _holds(buffer: object, value: float) -> bool:
    """Whether ``buffer`` is a tensor equal to ``value`` rounded to its own dtype."""
    if not isinstance(buffer, Tensor) or not buffer.is_floating_point() or buffer.is_meta:
        return False
    return bool(buffer == torch.tensor(value, dtype=buffer.dtype, device=buffer.device))


def _shown(tensor: object) -> str | None:
    """Return a one-element tensor's value for a repr, or ``None`` where it cannot be read on the host."""
    if not isinstance(tensor, Tensor) or tensor.is_meta or tensor.numel() != 1 or not can_read_back(tensor):
        return None
    return f"{tensor.detach().item():.6g}"


def _spec_text(spec: Free) -> str:
    """``Free(1.2, tolerance=0.1, bounds=(0.8, 1.3))``: a spec as it is written, without its unset fields."""
    parts = [f"{spec.nominal:g}"]
    if spec.tolerance is not None:
        parts.append(f"tolerance={spec.tolerance:g}")
    if spec.bounds is not None:
        parts.append(f"bounds=({spec.bounds[0]:g}, {spec.bounds[1]:g})")
    return f"Free({', '.join(parts)})"


def repr_parts(*parts: str) -> str:
    """Join the non-empty ``parts`` of an ``extra_repr`` with commas.

    E.g. ``repr_parts(f"wavelength={self.wavelength:g}", super().extra_repr())``.
    """
    return ", ".join(part for part in parts if part)


def free_specs(module: nn.Module) -> dict[str, Free]:
    """Collect the :class:`Free` spec of every free quantity in a module tree, keyed by dotted parameter name.

    Names and order match ``module.named_parameters()``.
    """
    found: dict[str, Free] = {}
    for prefix, child in module.named_modules():
        if isinstance(child, CalibratableModule):
            for name, spec in child.free_quantities().items():
                found[f"{prefix}.{name}" if prefix else name] = spec
    return found


def placement(module: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Return the real floating dtype and the device a module computes in, from its first tensor.

    A complex buffer maps to its real counterpart (``complex64`` to ``float32``); a module with no tensors uses the
    default dtype on the CPU.
    """
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor.real.dtype, tensor.device
    return torch.get_default_dtype(), torch.device("cpu")


def _missing(specs: Mapping[str, Free], params: Mapping[str, Tensor]) -> None:
    missing = [name for name in specs if name not in params]
    if missing:
        raise ParameterError(
            f"parameter values are missing for {missing}", path=missing[0] if len(missing) == 1 else None
        )


def _clamped(specs: Mapping[str, Free], params: Mapping[str, Tensor]) -> dict[str, Tensor]:
    clamped = dict(params)
    for name, spec in specs.items():
        if spec.bounds is not None and name in clamped:
            clamped[name] = clamped[name].clamp(*spec.bounds)
    return clamped


@dataclass(frozen=True, slots=True)
class ParameterPrior:
    """The calibration term ``C(H) = -log p(H)`` over free quantities, independent per quantity.

    A quantity with a ``tolerance`` contributes a Gaussian term; one without contributes nothing (a flat prior).
    Bounds are hard limits applied by :meth:`clamp`, not a term in :meth:`nll`.

    Parameters
    ----------
    specs : Mapping[str, Free]
        The free quantities by dotted name; a quantity's term is centred on its ``nominal``.
    centers : Mapping[str, Tensor], optional
        Centres that replace the ``nominal`` of the quantities named, as CPU tensors of shape ``(1,)`` (shared by
        every tile) or ``(T,)`` (one per tile), in the quantity's units, e.g. from an earlier calibration.
    """

    specs: Mapping[str, Free]
    centers: Mapping[str, Tensor] = field(default_factory=dict)
    _placed: dict[tuple[str, torch.dtype, torch.device], Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def from_module(cls, module: nn.Module) -> Self:
        """Build the prior from the :class:`Free` specs of a module tree."""
        return cls(free_specs(module))

    @property
    def covered(self) -> tuple[str, ...]:
        """The quantities with a ``tolerance``."""
        return tuple(name for name, spec in self.specs.items() if spec.tolerance is not None)

    @property
    def tiles(self) -> int:
        """Rows of the prior: the most rows of :attr:`centers`, 1 without them."""
        return max((int(center.shape[0]) for center in self.centers.values()), default=1)

    def _center(self, name: str, like: Tensor) -> float | Tensor:
        """Return the centre of ``name``: a number, or its ``(T,)`` rows in ``like``'s dtype and device, cast once."""
        center = self.centers.get(name)
        if center is None:
            return self.specs[name].nominal
        if center.shape[0] == 1:
            return float(center[0])
        key = (name, like.dtype, like.device)
        if key not in self._placed:
            self._placed[key] = center.to(dtype=like.dtype, device=like.device)
        return self._placed[key]

    def nll(self, params: Mapping[str, Tensor]) -> Tensor:
        """Return ``C(H)`` for parameter values keyed by the same dotted names as :attr:`specs`.

        The result has the dtype, device and batch shape of ``params``; with no toleranced quantity it is zero.
        """
        _missing(self.specs, params)
        reference = next(iter(params.values()), None)
        total = torch.zeros(()) if reference is None else reference.new_zeros(())
        for name, spec in self.specs.items():
            if spec.tolerance is not None:
                total = total + 0.5 * ((params[name] - self._center(name, params[name])) / spec.tolerance) ** 2
        return total

    def clamp(self, params: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return ``params`` with each bounded quantity clamped to its bounds."""
        return _clamped(self.specs, params)

    def clamp_(self, module: nn.Module) -> None:
        """Clamp the bounded parameters of ``module`` in place, for use after an optimizer step."""
        parameters = dict(module.named_parameters())
        with torch.no_grad():
            for name, spec in self.specs.items():
                if spec.bounds is not None:
                    parameters[name].clamp_(*spec.bounds)

    def select(self, tiles: Sequence[int]) -> Self:
        """Return the prior of the tiles ``tiles``, in that order; a one-row prior returns itself."""
        if self.tiles == 1:
            return self
        rows = list(tiles)
        return type(self)(
            self.specs,
            {name: center if center.shape[0] == 1 else center[rows] for name, center in self.centers.items()},
        )


# --- Where a module computes, and what may be read back from there --------------------------------------------------


def can_read_back(tensor: Tensor) -> bool:
    """Whether ``tensor`` can be read on the host: it is not wrapped by a ``torch.func`` transform such as ``grad``."""
    return not torch._C._functorch.is_functorch_wrapped_tensor(tensor)  # noqa: SLF001


def read_flag(flag: Tensor) -> bool:
    """Read a 0-d bool tensor on the host: one device sync, made with ``.item()``.

    Every read-back goes through here so a sync check (``waveorder.testing.assert_no_host_sync``) can report it;
    ``.item()`` raises a catchable error under ``torch.cuda.set_sync_debug_mode("error")`` where ``bool()`` aborts.
    """
    return bool(flag.item())


def same_device(first: torch.device | str, second: torch.device | str) -> bool:
    """Return whether two devices name the same hardware.

    Unlike ``torch.device`` equality, ``"cuda"`` matches ``"cuda:0"`` when device 0 is current, and a CPU index is
    ignored. Reads no tensor.

    Parameters
    ----------
    first, second : torch.device or str
        The devices, as ``torch.device`` objects or strings such as ``"cpu"``, ``"cuda"`` or ``"cuda:1"``.

    Returns
    -------
    bool
        ``True`` if both resolve to the same type and index.

    Raises
    ------
    RuntimeError
        From ``torch.device`` if a string is not a device name (``"gpu"``, say).
    """
    return _resolved(torch.device(first)) == _resolved(torch.device(second))


def _resolved(device: torch.device) -> tuple[str, int | None]:
    if device.type in ("cpu", "meta"):
        return device.type, None
    if device.index is not None:
        return device.type, device.index
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is not None and accelerator.type == device.type:
        return device.type, torch.accelerator.current_device_index()
    return device.type, 0


def resolve_device(name: str | None, *, field: str = "solver.device") -> torch.device:
    """Map a device name to a device: ``None`` is the CPU, ``"auto"`` the first of CUDA, MPS, CPU available.

    Parameters
    ----------
    name : str, optional
        The value of ``solver.device``, or of a model's ``device`` argument.
    field : str
        What an error names: the config path, or the argument.

    Raises
    ------
    ConfigurationError
        At ``field``, if the named device does not exist here.
    """
    if name is None:
        return torch.device("cpu")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        device = torch.device(name)
    except RuntimeError as error:
        raise ConfigurationError(field, f"{name!r} is not a device: {error}") from error
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ConfigurationError(field, f"{name!r} is not available here: no Apple GPU (MPS) on this machine")
    try:
        torch.empty((), device=device)
    except (RuntimeError, AssertionError) as error:
        raise ConfigurationError(field, f"{name!r} is not available here: {error}") from error
    return device


# --- The numbers a configuration or a store writes for the values tensors hold ---------------------------------


def shortest_decimal(value: float, dtype: torch.dtype) -> float:
    """Return the shortest decimal that ``dtype`` rounds to ``value``, so a float32 ``0.52`` is written ``0.52``.

    ``value`` itself when it is not finite or no tensor of ``dtype`` holds it.
    """
    return _shortest(value, lambda candidate: _rounded(candidate, dtype) == value)


def quantity_number(tensor: Tensor, registered: float | None = None) -> float:
    """Return the number to write for the scalar quantity ``tensor`` holds.

    ``registered`` (the number the quantity was built with) while the tensor still holds it, so an untouched ``1.2``
    stays ``1.2``; else the shortest decimal the tensor's dtype rounds to its current value.
    """
    current = float(tensor.detach())
    dtype = tensor.dtype

    def held(candidate: float) -> bool:
        return _rounded(candidate, dtype) == current

    if registered is not None and held(registered):
        return float(registered)
    return _shortest(current, held)


def _rounded(value: float, dtype: torch.dtype) -> float:
    return float(torch.tensor(value, dtype=dtype))


def _shortest(value: float, accept: Callable[[float], bool]) -> float:
    if not math.isfinite(value):
        return value
    for digits in range(1, 18):
        candidate = float(f"{value:.{digits}g}")
        if accept(candidate):
            return candidate
    return value


class _Caller(nn.Module):
    """Call a function on a module, so ``functional_call`` can swap that module's tensors, named ``target.<name>``."""

    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        self.target = target

    @override
    def forward(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return fn(self.target, *args, **kwargs)


def _known_names(module: nn.Module) -> set[str]:
    return {name for name, _ in module.named_parameters()} | {name for name, _ in module.named_buffers()}


def check_names(module: nn.Module, names: Mapping[str, object] | list[str]) -> None:
    """Raise :class:`~waveorder.errors.ParameterError` for any name that is not a parameter or buffer of ``module``."""
    known = _known_names(module)
    unknown = sorted(set(names) - known)
    if unknown:
        free = sorted(name for name, _ in module.named_parameters())
        path = unknown[0] if len(unknown) == 1 else None
        raise ParameterError(f"unknown parameter path(s) {unknown}; free parameters are {free}", path=path)


def evaluate[M: nn.Module, **P, R](
    fn: Callable[Concatenate[M, P], R], module: M, values: Mapping[str, Tensor], *args: P.args, **kwargs: P.kwargs
) -> R:
    """Return ``fn(module, *args, **kwargs)`` with ``values`` swapped in for the named parameters or buffers.

    Parameters
    ----------
    fn : Callable
        Takes the module first, e.g. ``lambda m, s: m.system.transfer(s)``.
    module : nn.Module
        The module; it is not modified.
    values : Mapping[str, Tensor]
        Replacement tensors keyed by dotted name, each of the shape, dtype and device of the tensor it replaces;
        unlisted parameters keep their values.

    Raises
    ------
    ParameterError
        If a name is not a parameter or buffer of ``module``.
    """
    check_names(module, values)
    swapped = {f"target.{name}": value for name, value in values.items()}
    return cast(R, functional_call(_Caller(module), swapped, (fn, *args), kwargs, tie_weights=False))


def load_parameters(module: nn.Module, values: Mapping[str, Tensor | float]) -> None:
    """Copy ``values`` into the named parameters of ``module`` in place, e.g. to apply the best grid candidate.

    Every value is checked before any is copied, so a refused call leaves ``module`` unchanged. Reads each tensor
    value back once (a device sync on CUDA).

    Parameters
    ----------
    module : nn.Module
        The module whose parameters are set; nothing else changes.
    values : Mapping[str, Tensor or float]
        By dotted parameter name: a finite one-element tensor or Python number, in the quantity's units and within
        the bounds of its :class:`Free`; unlisted parameters keep their value.

    Raises
    ------
    ParameterError
        If a name is not a parameter of ``module`` (a fixed quantity is named as fixed), or a value is not one
        element, not finite, or outside the bounds of its ``Free(...)`` by more than 1e-6 of their span or size;
        ``path`` is the name.
    """
    where = "load_parameters"
    _expect_parameter_names(module, values)
    specs = free_specs(module)
    for name, value in values.items():
        if isinstance(value, Tensor) and value.numel() != 1:
            raise ParameterError(f"{name} takes a scalar, got shape {tuple(value.shape)}", path=name)
        number = value.detach().item() if isinstance(value, Tensor) else value
        if not math.isfinite(number):
            raise ParameterError(f"{where} {name} must be finite, got {value}", path=name)
        _expect_within_bounds(name, number, specs.get(name))
    _assign_parameters(module, values)


_BOUND_SLACK = 1e-6
"""How far past a bound a loaded value may lie, as a fraction of the span or of the bound: a float32-rounded bound."""


def _expect_parameter_names(module: nn.Module, values: Mapping[str, object]) -> None:
    """Raise :class:`ParameterError` for a name that is no parameter of ``module``, saying so of a fixed quantity."""
    parameters = sorted(name for name, _ in module.named_parameters())
    unknown = sorted(set(values) - set(parameters))
    if not unknown:
        return
    path = unknown[0] if len(unknown) == 1 else None
    fixed_names = {
        f"{prefix}.{name}" if prefix else name
        for prefix, child in module.named_modules()
        if isinstance(child, CalibratableModule)
        for name in child.fixed_quantities()
    }
    fixed = [name for name in unknown if name in fixed_names]
    if len(fixed) == 1:
        raise ParameterError(
            f"{fixed[0]} is fixed: rebuild with the value or give it as Free(...); free parameters are {parameters}",
            path=fixed[0],
        )
    if fixed:
        raise ParameterError(
            f"{fixed} are fixed: rebuild with the values or give them as Free(...); free parameters are {parameters}",
            path=path,
        )
    raise ParameterError(f"unknown parameter path(s) {unknown}; free parameters are {parameters}", path=path)


def _expect_within_bounds(name: str, value: float, spec: Free | None) -> None:
    """Raise :class:`ParameterError` if ``value`` lies outside the bounds of the free quantity ``name``."""
    if spec is None or spec.bounds is None:
        return
    low, high = spec.bounds
    slack = _BOUND_SLACK * max(high - low, abs(low), abs(high))
    if not low - slack <= value <= high + slack:
        raise ParameterError(
            f"load_parameters {name} = {value:g} lies outside the bounds ({low:g}, {high:g}) of its Free(...): "
            "calibration keeps the quantity inside them, and a saved configuration could not describe the model. "
            "Pass a value within the bounds, or rebuild the part with wider ones",
            path=name,
        )


def _assign_parameters(module: nn.Module, values: Mapping[str, Tensor | float]) -> None:
    """Copy checked ``values`` into ``module``'s parameters with nothing read back: :func:`load_parameters`' copy."""
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            if isinstance(value, Tensor):
                parameters[name].copy_(value.reshape(()))
            else:
                parameters[name].fill_(value)
