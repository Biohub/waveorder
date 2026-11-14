"""The normalizers around a student, and the bundle that carries a trained student to inference.

A normalizer acts on the data before the student and, inverted, on its estimate after, so a distilled solver returns
values in the units of the model's estimate; fitted statistics are buffers, saved with the weights. :func:`predict` is
what a bundle computes: normalize, pad to the student's stride, run the student, crop, invert the target normalizer.
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Self, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from waveorder.errors import BackendUnavailableError, BundleError, IncompatibleModelError, ParameterError, ShapeError
from waveorder.nn.networks.base import PRECISIONS, Contract, Precision, RestorationNetwork
from waveorder.nn.networks.conditioning import Condition
from waveorder.nn.parameters import placement
from waveorder.registry import register

type Scope = Literal["voxel", "field"]
"""What a normalizer reads to normalize one voxel: that voxel, or the whole field of view."""


class Normalizer(nn.Module, ABC):
    """A map applied to values ``(..., Z, Y, X)``, with its inverse where it has one.

    Attributes
    ----------
    scope : {"voxel", "field"}
        ``"voxel"``: each voxel by itself, so any tile or patch is normalized as the whole field would be.
        ``"field"``: by statistics of the whole field of view; such data cannot be reconstructed tile by tile, and
        training normalizes each field before it cuts patches.
    invertible : bool
        Whether :meth:`inverse` exists. A target normalizer must be invertible.
    """

    scope: ClassVar[Scope] = "voxel"
    invertible: ClassVar[bool] = True

    @override
    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """Return ``values`` ``(..., Z, Y, X)`` normalized, with the same shape."""

    @abstractmethod
    def inverse(self, values: Tensor) -> Tensor:
        """Return normalized ``values`` in their original units.

        Raises
        ------
        IncompatibleModelError
            If the normalizer is not :attr:`invertible`.
        """

    def fit(self, samples: Iterable[Tensor]) -> None:
        """Fit statistics to ``samples`` (tensors of any shape, one per field); a no-op unless the class has any."""
        del samples


@register("identity", kind="normalizer")
class Identity(Normalizer):
    """Values as they are."""

    @override
    def forward(self, values: Tensor) -> Tensor:
        return values

    @override
    def inverse(self, values: Tensor) -> Tensor:
        return values


@register("field_standardize", kind="normalizer")
class FieldStandardize(Normalizer):
    """``(x - mean) / (std + eps)`` with the mean and standard deviation of each ``(Z, Y, X)`` volume.

    The statistics are those of the tensor given, per channel and field, so the scope is ``"field"`` and there is no
    inverse.

    Parameters
    ----------
    eps : float
        Added to the standard deviation, so a constant field maps to zeros; positive.
    """

    scope = "field"
    invertible = False

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if not eps > 0:
            raise ParameterError(f"FieldStandardize eps must be a positive number, got {eps!r}", path="eps")
        self.eps = float(eps)

    @override
    def forward(self, values: Tensor) -> Tensor:
        axes = (-3, -2, -1)
        mean = values.mean(axes, keepdim=True)
        return (values - mean) / (values.std(axes, keepdim=True, correction=0) + self.eps)

    @override
    def inverse(self, values: Tensor) -> Tensor:
        raise IncompatibleModelError(
            "FieldStandardize has no inverse: the statistics of a field are not kept. Use it for the input; "
            "normalize targets with Affine"
        )

    @override
    def extra_repr(self) -> str:
        return f"eps={self.eps:g}"


_SAMPLE = 1 << 16
"""Values kept per sample when fitting percentiles."""

_QUANTILE_LIMIT = 1 << 24
"""The most values ``torch.quantile`` takes."""


@register("affine", kind="normalizer")
class Affine(Normalizer):
    """``(x - offset) / scale`` with one offset and one scale, in the units of the values; the inverse is exact.

    The offset and the scale are buffers. With ``percentiles`` (see :meth:`robust`), :meth:`fit` sets the offset to
    the low percentile and the scale to the distance between the two, so most values land in [0, 1].

    Parameters
    ----------
    offset : float
        Subtracted from the values.
    scale : float
        Divides the values; not 0.
    percentiles : tuple[float, float], optional
        The low and the high percentile, in [0, 100], that :meth:`fit` reads; ``None`` makes :meth:`fit` a no-op.

    Raises
    ------
    ParameterError
        If ``offset`` or ``scale`` is not finite, ``scale`` is 0, or ``percentiles`` is not
        ``0 <= low < high <= 100``.
    """

    offset: Tensor
    scale: Tensor

    def __init__(self, offset: float = 0.0, scale: float = 1.0, *, percentiles: Sequence[float] | None = None) -> None:
        super().__init__()
        for name, value in (("offset", offset), ("scale", scale)):
            if not torch.isfinite(torch.tensor(value)):
                raise ParameterError(f"Affine {name} must be a finite number, got {value!r}", path=name)
        if scale == 0:
            raise ParameterError("Affine scale must not be 0", path="scale")
        if percentiles is not None:
            pair = tuple(percentiles)
            if len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 100:
                raise ParameterError(
                    f"Affine percentiles must be (low, high) with 0 <= low < high <= 100, got {percentiles!r}",
                    path="percentiles",
                )
            self.percentiles: tuple[float, float] | None = (float(pair[0]), float(pair[1]))
        else:
            self.percentiles = None
        self.register_buffer("offset", torch.tensor(float(offset)))
        self.register_buffer("scale", torch.tensor(float(scale)))

    @classmethod
    def robust(cls, low: float = 1.0, high: float = 99.0) -> Self:
        """Return an affine normalizer that :meth:`fit` sets from the ``low`` and ``high`` percentiles of samples."""
        return cls(percentiles=(low, high))

    @override
    def forward(self, values: Tensor) -> Tensor:
        return (values - self.offset) / self.scale

    @override
    def inverse(self, values: Tensor) -> Tensor:
        return values * self.scale + self.offset

    @override
    def fit(self, samples: Iterable[Tensor]) -> None:
        """Set the offset and the scale from the percentiles of ``samples``; a no-op without ``percentiles``.

        Each sample contributes at most 65536 finite values at even steps, so fields of any size weigh alike.

        Raises
        ------
        ParameterError
            If there is no finite value, or the two percentiles coincide (constant data).
        """
        if self.percentiles is None:
            return
        kept = []
        for sample in samples:
            flat = sample.detach().reshape(-1)
            flat = flat[torch.isfinite(flat)]
            kept.append(flat[:: max(1, -(-flat.numel() // _SAMPLE))].to(device="cpu", dtype=torch.float64))
        values = torch.cat(kept) if kept else torch.empty(0, dtype=torch.float64)
        if not values.numel():
            raise ParameterError("Affine.fit was given no finite value to fit percentiles to", path="samples")
        values = values[:: max(1, -(-values.numel() // _QUANTILE_LIMIT))]
        low, high = torch.quantile(values, torch.tensor(self.percentiles, dtype=torch.float64) / 100).tolist()
        if not high > low:
            raise ParameterError(
                f"Affine.fit: the {self.percentiles[0]:g} and {self.percentiles[1]:g} percentiles are both {low:g}, "
                "so the samples give no scale",
                path="samples",
            )
        self.offset.fill_(low)
        self.scale.fill_(high - low)

    @override
    def extra_repr(self) -> str:
        return "" if self.percentiles is None else f"percentiles={self.percentiles}"


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Example:
    """One input of a bundle and the output it must give: the bundle's self-test.

    Parameters
    ----------
    data : Tensor
        Data as a solver receives it, batched: ``(B, C, Z, Y, X)`` float32, prepared and not yet normalized.
    condition : Mapping[str, Tensor]
        The conditions of that data by name, each ``(B, K)``.
    expected : Tensor, optional
        The estimate ``(B, F, Zout, Y, X)`` float32 in the units of the model's estimate. ``None`` lets
        :func:`waveorder.io.bundle.save_bundle` compute it from the parts it saves, in evaluation mode on the CPU.
    tolerance : float
        The largest absolute difference from ``expected`` that still passes, in the units of the estimate.
    """

    data: Tensor
    condition: Mapping[str, Tensor] = field(default_factory=dict)
    expected: Tensor | None = None
    tolerance: float = 1e-4


def autocast(precision: Precision, device: torch.device) -> AbstractContextManager[object]:
    """Return the context a student runs in for ``precision`` on ``device``: nothing for float32, else autocast.

    Raises
    ------
    ParameterError
        If ``precision`` is none of ``"float32"``, ``"bf16"`` and ``"fp16"``.
    BackendUnavailableError
        If the device cannot compute in that precision: float16 runs on CUDA only, bfloat16 on the CPU and on CUDA
        devices that support it.
    """
    if precision not in PRECISIONS:
        raise ParameterError(f"precision must be one of {', '.join(PRECISIONS)}, got {precision!r}", path="precision")
    if precision == "float32":
        return nullcontext()
    if device.type == "cuda":
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise BackendUnavailableError(f"precision 'bf16' is not supported by {device}; use 'fp16' or 'float32'")
    elif device.type != "cpu":
        raise BackendUnavailableError(
            f"precision {precision!r} runs on CUDA and the CPU, not on {device}; use 'float32'"
        )
    elif precision == "fp16":
        raise BackendUnavailableError(
            "precision 'fp16' runs on CUDA only; on the CPU use 'bf16' or 'float32', or move the model to a GPU"
        )
    return torch.autocast(device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def predict(
    student: RestorationNetwork,
    data: Tensor,
    condition: Mapping[str, Tensor] | None = None,
    *,
    input_normalizer: Normalizer | None = None,
    target_normalizer: Normalizer | None = None,
    precision: Precision = "float32",
) -> Tensor:
    """Return the estimate ``(B, F, Zout, Y, X)`` of prepared data ``(B, C, Z, Y, X)``, in the model's units.

    The data is cast to the student's dtype and normalized, ``Y`` and ``X`` are reflection-padded at their far ends to
    a multiple of the student's stride, the student runs (under autocast for ``"bf16"`` and ``"fp16"``), and the
    estimate is cropped, cast back and passed through the inverse target normalizer. ``data`` is on the student's
    device; ``condition`` holds ``(B, K)`` tensors by name. The caller sets the gradient and training modes.

    Raises
    ------
    ShapeError
        If ``data`` is not five-dimensional, is too small to pad by reflection, or the student refuses its shape.
    BackendUnavailableError
        If the device cannot compute in ``precision``.
    """
    if data.dim() != 5:
        raise ShapeError(f"predict data has shape {tuple(data.shape)}, expected (B, C, Z, Y, X)")
    dtype, _ = placement(student)
    values = data.to(dtype)
    if input_normalizer is not None:
        values = input_normalizer(values)
    height, width = values.shape[-2:]
    stride = student.stride
    pad_y, pad_x = -height % stride, -width % stride
    if pad_y or pad_x:
        if pad_y >= height or pad_x >= width:
            raise ShapeError(
                f"data of Y, X = {(height, width)} is too small to pad by reflection to multiples of the student's "
                f"stride {stride}; give at least {stride // 2 + 1} pixels along each"
            )
        values = functional.pad(values, (0, pad_x, 0, pad_y, 0, 0), mode="reflect")
    given = {name: value.to(device=values.device, dtype=dtype) for name, value in (condition or {}).items()}
    with autocast(precision, values.device):
        estimate = student(values, given)
    estimate = estimate[..., :height, :width].to(dtype)
    return estimate if target_normalizer is None else target_normalizer.inverse(estimate)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Bundle:
    """A student with everything inference needs: what a training run gives and :class:`~waveorder.nn.Distilled` runs.

    :func:`waveorder.io.bundle.save_bundle` writes it as a directory; :func:`waveorder.io.bundle.load_bundle` reads it
    back, in evaluation mode on the CPU.

    Attributes
    ----------
    student : RestorationNetwork
        The network.
    contract : Contract
        What data the student takes and what it returns.
    conditions : tuple of Condition
        One per condition the student reads, with its name and size.
    input_normalizer, target_normalizer : Normalizer
        Applied to the data, and inverted on the student's output; by default :class:`Identity`.
    model : Mapping[str, object]
        The model the training targets were made with: its configuration, or its class names.
    record : Mapping[str, object]
        The training record: index hash, split sizes, metrics, steps, software versions.
    test : Example, optional
        One input and the output it must give; a bundle is written only with one.
    path : Path, optional
        The directory the bundle was read from; ``None`` for one built in memory.
    sha256 : str, optional
        For a bundle read from a directory, the hash of its ``bundle.json`` (which holds every other file's hash).
    """

    student: RestorationNetwork
    contract: Contract
    conditions: tuple[Condition, ...] = ()
    input_normalizer: Normalizer = field(default_factory=Identity)
    target_normalizer: Normalizer = field(default_factory=Identity)
    model: Mapping[str, object] = field(default_factory=dict)
    record: Mapping[str, object] = field(default_factory=dict)
    test: Example | None = None
    path: Path | None = None
    sha256: str | None = None

    def predict(
        self, data: Tensor, condition: Mapping[str, Tensor] | None = None, *, precision: Precision = "float32"
    ) -> Tensor:
        """Return :func:`predict` of ``data`` ``(B, C, Z, Y, X)`` with the bundle's student and normalizers."""
        return predict(
            self.student,
            data,
            condition,
            input_normalizer=self.input_normalizer,
            target_normalizer=self.target_normalizer,
            precision=precision,
        )

    def test_output(self, device: torch.device | str = "cpu") -> Tensor:
        """Return what copies of the parts give for the test input, in evaluation mode on ``device``.

        Raises
        ------
        BundleError
            If the bundle has no test.
        """
        if self.test is None:
            raise BundleError(f"{self.path or 'the bundle'}: the bundle has no test input")
        parts = copy.deepcopy((self.student, self.input_normalizer, self.target_normalizer))
        for module in parts:
            module.to(device).eval()
        student, input_normalizer, target_normalizer = parts
        with torch.no_grad():
            return predict(
                student,
                self.test.data.to(device),
                self.test.condition,
                input_normalizer=input_normalizer,
                target_normalizer=target_normalizer,
            )

    def self_test(self, device: torch.device | str | None = None) -> float:
        """Run the bundle's test input and return the largest absolute difference from its expected output.

        The parts are copied and run on ``device`` (by default where the student is); the bundle is left as it is.

        Raises
        ------
        BundleError
            If the bundle's test has no expected output, the difference exceeds the test's tolerance, or the output
            is not finite: this machine, device or software does not reproduce the bundle.
        """
        where = self.path or "the bundle"
        if self.test is None or self.test.expected is None:
            raise BundleError(f"{where}: the bundle's test has no expected output")
        target = torch.device(device) if device is not None else placement(self.student)[1]
        error = float((self.test_output(target).cpu() - self.test.expected.cpu()).abs().max())
        if not error <= self.test.tolerance:
            raise BundleError(
                f"{where}: the self-test differs from the expected output by {error:g} on {target}, above the "
                f"tolerance {self.test.tolerance:g}: this device or software does not reproduce the bundle"
            )
        return error
