"""The generative model ``g = H f + n`` as one module tree."""

from typing import Protocol, override

import torch
from torch import Tensor, nn

from waveorder.errors import IncompatibleModelError, PlacementError, ShapeError
from waveorder.nn.noise import Noise
from waveorder.nn.operators import LinearOperator
from waveorder.nn.parameters import placement, same_device
from waveorder.nn.priors import ObjectPrior
from waveorder.nn.sampling import Data, Object, Sampling, expect_real, expect_shape
from waveorder.nn.systems.base import ImagingSystem

type _Model = ImagingModel
"""A lazily evaluated alias: :class:`ImagingModel` is defined below and named by :class:`Checked`."""


class Checked(Protocol):
    """What :meth:`ImagingModel.check` asks of a verb: a check of the model (every :class:`~waveorder.nn.Verb`)."""

    def check(self, model: _Model) -> None:
        """Raise :class:`~waveorder.errors.IncompatibleModelError` if the verb cannot run on ``model``."""
        ...


class ImagingModel(nn.Module):
    """The three distributions of ``g = H f + n`` under one root.

    Parameter names are prefixed by the child: ``system.``, ``object_prior.`` and ``noise.``, from the config blocks
    ``imaging_system``, ``object`` and ``noise``. The children must share one dtype and device when a verb runs
    (:meth:`check`); move them together with ``ImagingModel.to(...)``.

    Parameters
    ----------
    system : ImagingSystem
        ``H`` and its free quantities.
    object_prior : ObjectPrior
        ``p(f)``, the prior on the object.
    noise : Noise
        ``p(n)``.
    """

    system: ImagingSystem
    object_prior: ObjectPrior
    noise: Noise

    def __init__(self, system: ImagingSystem, object_prior: ObjectPrior, noise: Noise) -> None:
        super().__init__()
        self.system = system
        self.object_prior = object_prior
        self.noise = noise

    def expect_one_placement(self) -> None:
        """Raise :class:`~waveorder.errors.PlacementError` unless every child with tensors shares one placement.

        The system's placement is the reference; a child without floating tensors matches any placement.
        """
        reference: tuple[str, torch.dtype, torch.device] | None = None
        for name, child in (("system", self.system), ("object_prior", self.object_prior), ("noise", self.noise)):
            tensors = (*child.parameters(), *child.buffers())
            if not any(tensor.is_floating_point() or tensor.is_complex() for tensor in tensors):
                continue
            placed = placement(child)
            if reference is None:
                reference = (name, *placed)
                continue
            first, dtype, device = reference
            if placed[0] != dtype or not same_device(placed[1], device):
                raise PlacementError(
                    f"ImagingModel {name} computes in {placed[0]} on {placed[1]} but {first} computes in {dtype} "
                    f"on {device}; move all three together with ImagingModel.to(dtype=..., device=...) (on a "
                    "waveorder.models model: model.to(device, dtype=...), which also moves its solver and score)"
                )

    def check(self, verb: Checked, sampling: Sampling) -> None:
        """Raise :class:`~waveorder.errors.IncompatibleModelError` if ``verb`` cannot run here on ``sampling``.

        Solvers and scores call this on every call; call it directly to fail before reading any data. Also raises
        :class:`~waveorder.errors.PlacementError` if the children were moved apart, and
        :class:`~waveorder.errors.IncompatibleModelError` if the model computes in a dtype other than float32 or
        float64.
        """
        self.expect_one_placement()
        dtype, _ = placement(self.system)
        if dtype not in (torch.float32, torch.float64):
            raise IncompatibleModelError(
                f"{type(verb).__name__}: the model computes in {dtype}, but the Fourier transforms run in "
                "torch.float32 and torch.float64 only; move it with ImagingModel.to(dtype=torch.float32) (on a "
                "waveorder.models model: model.to(dtype=torch.float32))"
            )
        self.system.check_sampling(sampling)
        verb.check(self)

    def objective(
        self, data: Data, estimate: Object, sampling: Sampling, *, operator: LinearOperator | None = None
    ) -> Tensor:
        """Return the MAP objective ``D(g; H f) + P(f)`` of an object estimate, excluding constants of ``f``.

        Pass ``operator`` to reuse ``H`` already bound to ``sampling``; see :func:`expect_on_system` and
        :func:`bound_operator` for what this raises.
        """
        where = "ImagingModel.objective"
        system = self.system
        expect_on_system(data, system, where=where, name="data", shape=system.data_shape(sampling))
        expect_on_system(estimate, system, where=where, name="estimate", shape=system.object_shape(sampling))
        bound = bound_operator(system, sampling, operator=operator, where=where)
        penalty = self.object_prior.penalty(estimate, scale=self.noise.scale())
        return self.noise.nll(data, bound.apply(estimate)) + penalty

    @override
    def forward(self, obj: Object, sampling: Sampling) -> Data:
        """Return the noiseless data ``H f``."""
        return self.system(obj, sampling)


def expect_on_system(
    tensor: Tensor, system: ImagingSystem, *, where: str, name: str, shape: tuple[int, ...] | None = None
) -> None:
    """Raise unless ``tensor`` is real, in the dtype and on the device ``system`` computes in, and of ``shape``.

    The one check for data and objects handed to a model; it reads only metadata.

    Parameters
    ----------
    tensor : Tensor
        The data or object.
    system : ImagingSystem
        The system whose dtype and device it must match.
    where, name : str
        The caller and the argument, which every message starts with: ``"calibrate data has shape ..."``.
    shape : tuple of int, optional
        The exact shape required, e.g. ``system.data_shape(sampling)`` or ``system.object_shape(sampling)``.

    Raises
    ------
    ShapeError
        If ``tensor`` is complex (data and objects are real), or does not have ``shape``.
    PlacementError
        If it has another dtype or lives on another device than the system's parameters.
    """
    expect_real(tensor, what=f"{where} {name}")
    dtype, device = placement(system)
    if tensor.dtype != dtype or not same_device(tensor.device, device):
        raise PlacementError(
            f"{where} {name} is {tensor.dtype} on {tensor.device} but {type(system).__name__} computes in {dtype} on "
            f'{device}; convert the tensor with .to(dtype={dtype}, device="{device}"), or move the model with '
            "ImagingModel.to(dtype=..., device=...) (on a waveorder.models model: model.to(device, dtype=...))"
        )
    if shape is not None:
        expect_shape(tensor, tuple(shape), what=f"{where} {name}")


def bound_operator(
    system: ImagingSystem, sampling: Sampling, *, operator: LinearOperator | None, where: str
) -> LinearOperator:
    """Return ``operator``, or ``system`` bound to ``sampling``, after checking it stands for ``H`` on ``sampling``.

    Reads only metadata.

    Raises
    ------
    IncompatibleModelError
        If the operator is not of the system's declared
        :attr:`~waveorder.nn.systems.base.ImagingSystem.operator_type`.
    ShapeError
        If the operator was bound to another sampling, or maps objects or data of other shapes than the system on
        ``sampling``.
    PlacementError
        If the operator computes in another real dtype or on another device than the system's parameters.
    """
    bound = system.bind(sampling) if operator is None else operator
    name = f"{type(system).__name__}.bind" if operator is None else f"{where} operator"
    if not isinstance(bound, system.operator_type):
        raise IncompatibleModelError(
            f"{name} is a {type(bound).__name__}, but {type(system).__name__} declares operator_type "
            f"{system.operator_type.__name__}"
        )
    if bound.sampling != sampling:
        raise ShapeError(f"{name} was bound to {bound.sampling}, not to the data's {sampling}")
    for what, actual, expected in (
        ("objects", bound.object_shape, system.object_shape(sampling)),
        ("data", bound.data_shape, system.data_shape(sampling)),
    ):
        if actual != expected:
            raise ShapeError(
                f"{name} maps {what} of shape {actual}, but {type(system).__name__} on this sampling maps {what} of "
                f"shape {expected}"
            )
    placed = bound.placement
    dtype, device = placement(system)
    if placed[0] != dtype or not same_device(placed[1], device):
        raise PlacementError(
            f"{name} computes in {placed[0]} on {placed[1]} but {type(system).__name__} computes in {dtype} on "
            f"{device}; the operator was bound before the model moved: bind it again with model.system.bind(sampling), "
            "or pass operator=None"
        )
    return bound
