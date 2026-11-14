"""Linear operators: an imaging system bound to one sampling.

:meth:`ImagingSystem.bind <waveorder.nn.systems.base.ImagingSystem.bind>` returns one of these, so whatever a system
computes for a sampling (a transfer function, a norm) is computed once per solve. :class:`FourierDiagonal` is
diagonal in Fourier space, which the closed-form verbs require.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self, override

import torch
from torch import Tensor

from waveorder.errors import PlacementError, ShapeError
from waveorder.nn.functional import (
    apply_adjoint,
    apply_transfer,
    gram,
    hermitian_half,
    is_complex_matrix,
    largest_singular_value,
)
from waveorder.nn.parameters import same_device
from waveorder.nn.sampling import Data, HalfTransfer, Object, Sampling, expect_real, expect_shape

type _Fourier = FourierDiagonal | None
"""What :meth:`LinearOperator.as_fourier` returns (a lazily evaluated alias: the class is defined below)."""


class LinearOperator(ABC):
    """``H`` on one sampling: a linear map between real tensors of fixed shapes.

    The operator a system binds maps ``(F, Z, Y, X)`` objects to ``(C, Z, Y, X)`` data on ``sampling``.

    Parameters
    ----------
    sampling : Sampling
        The grid the operator acts on.
    object_shape : tuple[int, ...]
        Shape of the input, ``(F, Z, Y, X)`` for a system's operator.
    data_shape : tuple[int, ...]
        Shape of the output, ``(C, Z, Y, X)`` for a system's operator.
    """

    def __init__(self, sampling: Sampling, *, object_shape: tuple[int, ...], data_shape: tuple[int, ...]) -> None:
        self.sampling = sampling
        self.object_shape = tuple(object_shape)
        self.data_shape = tuple(data_shape)

    @property
    @abstractmethod
    def placement(self) -> tuple[torch.dtype, torch.device]:
        """The real dtype and the device the operator computes in, from its tensors' metadata."""

    def _expect_placement(self, tensor: Tensor, *, what: str) -> None:
        dtype, device = self.placement
        if tensor.dtype != dtype or not same_device(tensor.device, device):
            raise PlacementError(
                f"{what} is {tensor.dtype} on {tensor.device} but {type(self).__name__} computes in {dtype} on "
                f"{device}; move it with .to(dtype=..., device=...)"
            )

    def expect_object(self, obj: Tensor) -> None:
        """Raise unless ``obj`` is real with shape ``object_shape`` on the operator's dtype and device.

        Raises :class:`~waveorder.errors.ShapeError` for the shape or a complex dtype, and
        :class:`~waveorder.errors.PlacementError` for another real dtype or device.
        """
        expect_shape(obj, self.object_shape, what="object")
        expect_real(obj, what="object")
        self._expect_placement(obj, what="object")

    def expect_data(self, data: Tensor) -> None:
        """Raise unless ``data`` is real with shape ``data_shape`` on the operator's dtype and device (see above)."""
        expect_shape(data, self.data_shape, what="data")
        expect_real(data, what="data")
        self._expect_placement(data, what="data")

    @abstractmethod
    def apply(self, obj: Object) -> Data:
        """Return ``H f``."""

    @abstractmethod
    def adjoint(self, data: Data) -> Object:
        """Return ``H* g``: ``<H f, g> = <f, H* g>`` under the real inner product."""

    @abstractmethod
    def norm(self) -> Tensor:
        """Return ``||H||_2``."""

    def normal_action(self) -> Callable[[Object], Object]:
        """Return ``f -> H* H f`` with the work that does not depend on ``f`` done once, now.

        Take it once per solver call: it holds the operator's current values and their autograd state.
        """
        return lambda obj: self.adjoint(self.apply(obj))

    def as_fourier(self) -> _Fourier:
        """Return the :class:`FourierDiagonal` the closed-form algebra runs on, or ``None`` without that structure."""
        return None


class FourierDiagonal(LinearOperator):
    """A shift-invariant ``H``: one ``(C, F)`` matrix per frequency of the ``rfftn`` half spectrum.

    Circular convolution over the sampled volume. The constructor makes the self-conjugate columns Hermitian
    (:func:`~waveorder.nn.functional.hermitian_half`), as ``irfftn`` does anyway, so every method describes the same
    real operator.

    Parameters
    ----------
    transfer : Tensor
        Half-spectrum transfer stored real: ``(C, F, Z, Y, X // 2 + 1, 2)`` for a complex ``H``, or
        ``(C, F, Z, Y, X // 2 + 1)`` for a real ``H`` (an even point-spread function).
    sampling : Sampling
        The grid of the transfer.
    hermitian : bool
        Whether the self-conjugate columns are Hermitian already, which skips the projection.

    Raises
    ------
    ShapeError
        If the transfer is complex or does not match ``sampling``.
    """

    def __init__(self, transfer: HalfTransfer, sampling: Sampling, *, hermitian: bool = False) -> None:
        if transfer.is_complex():
            raise ShapeError(f"transfer must be stored real, got {transfer.dtype}; pass torch.view_as_real(transfer)")
        (nz, ny, nx), complex_valued = sampling.shape, is_complex_matrix(transfer)
        # Any other number of dimensions is reported here, against the real layout.
        expected = (*transfer.shape[:2], nz, ny, nx // 2 + 1, *((2,) if complex_valued else ()))
        expect_shape(transfer, expected, what="transfer")
        channels, fields = transfer.shape[:2]
        super().__init__(sampling, object_shape=(fields, *sampling.shape), data_shape=(channels, *sampling.shape))
        self.transfer = transfer if hermitian else hermitian_half(transfer, nx, real=not complex_valued)

    @property
    def real_transfer(self) -> bool:
        """Whether the transfer is stored as a real ``H``, without an imaginary part."""
        return not is_complex_matrix(self.transfer)

    @override
    def apply(self, obj: Object) -> Data:
        self.expect_object(obj)
        return apply_transfer(obj, self.transfer)

    @override
    def adjoint(self, data: Data) -> Object:
        self.expect_data(data)
        return apply_adjoint(data, self.transfer)

    @override
    def normal_action(self) -> Callable[[Object], Object]:
        """Return ``f -> H* H f`` over a Gram computed once, now, with the transfer's current values and autograd."""
        transfer_gram = gram(self.transfer)

        def normal(obj: Object) -> Object:
            self.expect_object(obj)
            return apply_transfer(obj, transfer_gram)

        return normal

    @override
    def norm(self) -> Tensor:
        """Return the exact ``||H||_2``: the largest singular value of ``H_k`` over all frequencies."""
        return largest_singular_value(self.transfer)

    @override
    def as_fourier(self) -> Self:
        return self

    @property
    @override
    def placement(self) -> tuple[torch.dtype, torch.device]:
        return self.transfer.dtype, self.transfer.device


class ReshapedFourierDiagonal(LinearOperator):
    """A :class:`FourierDiagonal` on another grid whose data are viewed in another shape.

    For example a thin object ``(F, 1, Y, X)`` imaged into a defocus stack: ``inner`` acts on the plane sampling
    ``(1, Y, X)`` with ``C Z`` channels, and its data ``(C Z, 1, Y, X)`` are viewed as ``(C, Z, Y, X)``.

    Parameters
    ----------
    inner : FourierDiagonal
        The operator on its own grid; the object shape is its object shape.
    sampling : Sampling
        The data's grid.
    data_shape : tuple[int, ...]
        Shape of the data, with as many elements as ``inner.data_shape``.

    Raises
    ------
    ShapeError
        If the data shapes hold different numbers of elements.
    """

    def __init__(self, inner: FourierDiagonal, sampling: Sampling, *, data_shape: tuple[int, ...]) -> None:
        if math.prod(data_shape) != math.prod(inner.data_shape):
            raise ShapeError(f"cannot view data of shape {inner.data_shape} as {tuple(data_shape)}")
        super().__init__(sampling, object_shape=inner.object_shape, data_shape=data_shape)
        self.inner = inner

    @override
    def apply(self, obj: Object) -> Data:
        return self.inner.apply(obj).reshape(self.data_shape)

    @override
    def adjoint(self, data: Data) -> Object:
        self.expect_data(data)
        return self.inner.adjoint(data.reshape(self.inner.data_shape))

    @override
    def normal_action(self) -> Callable[[Object], Object]:
        return self.inner.normal_action()

    @override
    def norm(self) -> Tensor:
        """Return the exact norm of ``inner``: a reshape keeps every singular value."""
        return self.inner.norm()

    @override
    def as_fourier(self) -> FourierDiagonal:
        return self.inner

    @property
    @override
    def placement(self) -> tuple[torch.dtype, torch.device]:
        return self.inner.placement
