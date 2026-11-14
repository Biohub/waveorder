"""Imaging systems: modules whose parameters determine a linear map ``H`` from object fields to data channels."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self, final, get_args, override

import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn.functional import hermitian_half
from waveorder.nn.operators import FourierDiagonal, LinearOperator
from waveorder.nn.parameters import CalibratableModule, can_read_back
from waveorder.nn.sampling import Data, HalfTransfer, Object, Sampling

type CacheMode = Literal["off", "auto"]
"""``"off"`` recomputes H on every call; ``"auto"`` reuses it while nothing that determines it has changed."""


type _Signature = tuple[int, torch.dtype, torch.device, int]
"""A state tensor's write version, dtype, device and storage address."""


@dataclass(frozen=True, slots=True, eq=False)
class StateKey:
    """The identity and write state of a system's tensors; see :meth:`ImagingSystem.state_key`.

    Equal when it holds the same tensor objects with the same signatures. The strong references keep ids from being
    reused while a key lives.
    """

    tensors: tuple[Tensor, ...]
    signature: tuple[_Signature, ...]

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StateKey)
            and len(self.tensors) == len(other.tensors)
            and all(a is b for a, b in zip(self.tensors, other.tensors, strict=True))
            and self.signature == other.signature
        )

    @override
    def __hash__(self) -> int:
        return hash((tuple(map(id, self.tensors)), self.signature))


class ImagingSystem(CalibratableModule, ABC):
    """The imaging system ``H``: a linear map from named real object fields to measured channels.

    The object is real, shape ``(F, Z, Y, X)``, one volume per name in :attr:`fields`; data are real, shape
    ``(C, Z, Y, X)`` with ``C = channel_count``. Subclasses implement :meth:`compute_operator`; callers use
    :meth:`bind`.

    Parameters
    ----------
    fields : tuple[str, ...]
        Names of the object fields, in tensor order.
    channel_count : int
        Number of measured channels.

    Attributes
    ----------
    operator_type : type[LinearOperator]
        The operator class :meth:`bind` returns; the closed-form solvers need a
        :class:`~waveorder.nn.operators.FourierDiagonal`. Set per class, or per instance in ``__init__``.
    preserves_nonnegativity : bool
        Whether ``H`` maps nonnegative objects to nonnegative data (an intensity point-spread function).
        Richardson-Lucy requires it. Default ``False``.
    """

    operator_type: type[LinearOperator] = LinearOperator
    preserves_nonnegativity: bool = False

    def __init__(self, *, fields: tuple[str, ...], channel_count: int) -> None:
        super().__init__()
        if not fields or len(set(fields)) != len(fields):
            raise ParameterError(f"fields must be distinct names, at least one, got {fields}")
        if channel_count < 1:
            raise ParameterError(f"channel_count must be at least 1, got {channel_count}")
        self.fields = fields
        self.channel_count = channel_count

    def object_shape(self, sampling: Sampling) -> tuple[int, ...]:
        """Return the object shape on ``sampling``: ``(F, Z, Y, X)``."""
        return (len(self.fields), *sampling.shape)

    def data_shape(self, sampling: Sampling) -> tuple[int, ...]:
        """Return the data shape on ``sampling``: ``(C, Z, Y, X)``."""
        return (self.channel_count, *sampling.shape)

    def check_sampling(self, sampling: Sampling) -> None:
        """Raise :class:`IncompatibleModelError` if this system cannot image on ``sampling``.

        The base class accepts every sampling. :meth:`bind` calls it on every bind, so it reads Python numbers only.
        """

    def undetermined_by_likelihood(self) -> tuple[str, ...]:
        """Return the quantities of this system that a likelihood score cannot determine.

        A likelihood sees ``H`` only through the statistics of ``H f``, which do not change when the point-spread
        function is translated, so a quantity that only translates it (for a 3D object, the focus offset) is
        undetermined. :func:`~waveorder.nn.calibrate` warns with :class:`~waveorder.errors.CalibrationWarning` when
        one of these is free and the score is a likelihood.

        Returns
        -------
        tuple[str, ...]
            Dotted names relative to the system, e.g. ``("detection.z_focus_offset",)``; empty in the base class.
        """
        return ()

    @abstractmethod
    def compute_operator(self, sampling: Sampling) -> LinearOperator:
        """Return ``H`` on ``sampling`` for the current parameters, including any swapped in by ``functional_call``."""

    @final
    def bind(self, sampling: Sampling) -> LinearOperator:
        """Return ``H`` on ``sampling``, after :meth:`check_sampling`.

        Raises
        ------
        IncompatibleModelError
            If :meth:`check_sampling` refuses ``sampling``, or :meth:`compute_operator` returned another class than
            :attr:`operator_type`.
        """
        self.check_sampling(sampling)
        operator = self.compute_operator(sampling)
        if not isinstance(operator, self.operator_type):
            raise IncompatibleModelError(
                f"{type(self).__name__} declares operator_type {self.operator_type.__name__}, but compute_operator "
                f"returned a {type(operator).__name__}"
            )
        return operator

    @final
    @override
    def forward(self, obj: Object, sampling: Sampling) -> Data:
        """Return the noiseless data ``H f`` of a real object ``(F, Z, Y, X)``: ``bind(sampling).apply(f)``."""
        return self.bind(sampling).apply(obj)

    def state_key(self) -> StateKey | None:
        """Return the key a result derived from the parameters may be cached under.

        Equal while no parameter or buffer was replaced or modified. ``None`` when the result must not outlive the
        call: in inference mode, or when it would carry gradients.
        """
        state = (*self.parameters(), *self.buffers())
        return StateKey(state, _signature(state)) if _cacheable(state) else None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    sampling: Sampling
    key: StateKey
    transfer: Tensor


def _signature(state: tuple[Tensor, ...]) -> tuple[_Signature, ...]:
    # torch has no public counter of in-place writes; optimizer steps bump ``_version``. ``param.data = ...`` and
    # module conversions keep the parameter object and its version but change its dtype, device or storage.
    return tuple((t._version, t.dtype, t.device, t.data_ptr()) for t in state)  # noqa: SLF001


def _cacheable(state: tuple[Tensor, ...]) -> bool:
    # Inference tensors cannot be saved for backward, so neither a result computed in inference mode nor one
    # computed from inference-mode parameters may outlive the call.
    if torch.is_inference_mode_enabled() or any(t.is_inference() for t in state):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in state):
        return False
    return all(can_read_back(t) for t in state)


class ShiftInvariantSystem(ImagingSystem):
    """A shift-invariant system: ``H`` is a ``(C, F)`` matrix per spatial frequency, the transfer function.

    Subclasses implement :meth:`compute_transfer` on the ``rfftn`` half spectrum; :meth:`bind` wraps it in a
    :class:`~waveorder.nn.operators.FourierDiagonal`.

    Parameters
    ----------
    fields : tuple[str, ...]
        Names of the object fields, in tensor order.
    channel_count : int
        Number of measured channels.
    cache : {"off", "auto"}
        ``"auto"`` reuses the transfer while the sampling and every parameter and buffer are unchanged. A write
        through ``.data`` (``p.data.clamp_(...)``) is not seen: write under ``torch.no_grad()`` or with
        :func:`~waveorder.nn.load_parameters`. Do not modify a returned transfer in place.

    Raises
    ------
    ParameterError
        If ``cache`` is not ``"off"`` or ``"auto"``.

    Attributes
    ----------
    real_transfer : bool
        Whether ``H`` is real at every frequency; then the transfer is stored as ``(C, F, Z, Y, X // 2 + 1)`` with
        no imaginary part. Set from the configuration, never from parameter values; nothing checks it at run time.
    """

    operator_type: type[LinearOperator] = FourierDiagonal
    real_transfer: bool = False

    def __init__(self, *, fields: tuple[str, ...], channel_count: int, cache: CacheMode = "off") -> None:
        super().__init__(fields=fields, channel_count=channel_count)
        modes = get_args(CacheMode.__value__)
        if cache not in modes:  # a typo must not turn caching on: transfer() tests only for "off"
            raise ParameterError(f"{type(self).__name__} cache must be one of {modes}, got {cache!r}", path="cache")
        self.cache: CacheMode = cache
        self._cache_entry: _CacheEntry | None = None

    @abstractmethod
    def compute_transfer(self, sampling: Sampling) -> Tensor:
        """Compute the transfer on ``sampling``, in FFT order along z and y.

        Return a complex ``(C, F, Z, Y, X // 2 + 1)`` half spectrum, or its real storage
        ``(C, F, Z, Y, X // 2 + 1, 2)``; with :attr:`real_transfer`, a real ``(C, F, Z, Y, X // 2 + 1)`` tensor.
        :func:`~waveorder.nn.functional.half_spectrum_frequencies` gives the frequencies ``(fz, fy, fx)`` of that grid.
        """

    @final
    def transfer(self, sampling: Sampling) -> HalfTransfer:
        """Return the transfer on ``sampling``: the cache's (``cache="auto"``, :meth:`seed_transfer`), else computed."""
        entry = self._cache_entry
        if self.cache == "off" and entry is None:
            return self._computed(sampling)
        key = self.state_key()
        if key is None:
            return self._computed(sampling)
        if entry is not None and entry.sampling == sampling and entry.key == key:
            return entry.transfer
        transfer = self._computed(sampling)
        if self.cache == "auto":
            self._cache_entry = _CacheEntry(sampling, key, transfer)
        return transfer

    @final
    def seed_transfer(self, sampling: Sampling, transfer: HalfTransfer) -> bool:
        """Hand the system its transfer on ``sampling`` for its current state; :meth:`transfer` returns it after.

        Returns ``False``, keeping nothing, when :meth:`state_key` is ``None``.
        """
        key = self.state_key()
        if key is not None:
            self._cache_entry = _CacheEntry(sampling, key, transfer)
        return key is not None

    @final
    @override
    def compute_operator(self, sampling: Sampling) -> FourierDiagonal:
        return FourierDiagonal(self.transfer(sampling), sampling, hermitian=True)

    def _computed(self, sampling: Sampling) -> HalfTransfer:
        transfer = self.compute_transfer(sampling)
        if transfer.is_complex():
            transfer = transfer.real if self.real_transfer else torch.view_as_real(transfer)
        nz, ny, nx = sampling.shape
        layout = () if self.real_transfer else (2,)
        expected = (self.channel_count, len(self.fields), nz, ny, nx // 2 + 1, *layout)
        if tuple(transfer.shape) != expected:
            raise ShapeError(
                f"{type(self).__name__}.compute_transfer returned a transfer of shape {tuple(transfer.shape)}, "
                f"expected {expected}: {_transfer_rule(real=self.real_transfer)}"
            )
        return hermitian_half(transfer, nx, real=self.real_transfer)

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        # ``.to``, ``.double()`` and ``.cuda()`` swap tensor data under the same parameter objects.
        converted = super()._apply(fn, recurse)
        self._cache_entry = None
        return converted


def _transfer_rule(*, real: bool) -> str:
    """Return what :meth:`ShiftInvariantSystem.compute_transfer` must give, for the error on any other shape."""
    helper = "waveorder.nn.functional.half_spectrum_frequencies(sampling) gives (fz, fy, fx) on that grid"
    if real:
        return f"real_transfer is True, so return a real (C, F, Z, Y, X // 2 + 1) half spectrum; {helper}"
    return (
        "return a complex (C, F, Z, Y, X // 2 + 1) half spectrum (it is stored real, with a trailing axis of 2), or "
        f"set self.real_transfer = True in __init__ to return a real one; {helper}"
    )
