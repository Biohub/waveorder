import math

import iohub
import torch
from torch import Tensor

from waveorder.nn import Gaussian, ImagingModel, Sampling, ShiftInvariantSystem, Tikhonov
from waveorder.tiling.merge import Index


class Blur(ShiftInvariantSystem):
    """A Gaussian blur of standard deviation ``sigma`` um, centred at ``shift`` um (z, y, x); ``sigma = 0`` with no
    shift is the identity. A shift makes the kernel asymmetric about the origin."""

    def __init__(
        self, sigma: float, *, channels: int = 1, learnable: bool = False, shift: tuple[float, float, float] = (0, 0, 0)
    ) -> None:
        super().__init__(fields=("f",), channel_count=channels)
        self.learnable = learnable
        self.shift = shift
        if learnable:
            self.register_parameter("sigma", torch.nn.Parameter(torch.tensor(sigma)))
        else:
            self.register_buffer("sigma", torch.tensor(sigma))
        self.computed = 0

    def compute_transfer(self, sampling: Sampling) -> Tensor:
        self.computed += 1
        (nz, ny, nx), (dz, dy, dx) = sampling.shape, sampling.pixel_size
        sigma = self.quantity("sigma")  # includes a value swapped in by functional_call (per-tile tables)
        kwargs = {"dtype": sigma.dtype, "device": sigma.device}
        fz = torch.fft.fftfreq(nz, d=dz, **kwargs)[:, None, None]
        fy = torch.fft.fftfreq(ny, d=dy, **kwargs)[None, :, None]
        fx = torch.fft.rfftfreq(nx, d=dx, **kwargs)[None, None, :]
        magnitude = torch.exp(-2 * math.pi**2 * sigma**2 * (fz**2 + fy**2 + fx**2))
        sz, sy, sx = self.shift
        phase = -2 * math.pi * (fz * sz + fy * sy + fx * sx)
        h = torch.polar(magnitude, phase.expand_as(magnitude))
        return h.expand(self.channel_count, 1, *h.shape)


def blur_model(sigma: float, *, channels: int = 1) -> ImagingModel:
    return ImagingModel(Blur(sigma, channels=channels), Tikhonov(1e-2), Gaussian(0.1))


class ArraySource:
    """A non-tensor array-like: slicing returns a numpy array, as zarr does; records reads in flight."""

    def __init__(self, values: Tensor, *, chunks: tuple[int, ...] | None = None) -> None:
        self.values = values
        self.shape = tuple(values.shape)
        self.dtype = values.dtype
        self.chunks = chunks
        self.reads: list[Index] = []

    def __getitem__(self, key: Index):
        self.reads.append(key)
        return self.values[key].numpy()


class ArraySink:
    """A non-tensor writable array-like: assignment receives numpy arrays; records every written box."""

    def __init__(self, shape: tuple[int, ...], *, chunks: tuple[int, ...] = (1, 1, 1)) -> None:
        self.values = torch.full(shape, float("nan"))
        self.shape = shape
        self.chunks = chunks
        self.writes: list[Index] = []

    def __getitem__(self, key: Index):
        return self.values[key].numpy()

    def __setitem__(self, key: Index, value) -> None:
        self.writes.append(key)
        self.values[key] = torch.as_tensor(value)


def create_sink(path: str, *, like, fields: list[str], times: int | None = None):
    """Create a single-FOV OME-Zarr store for the estimates of ``like``'s data: one channel per field, ``like``'s
    extents, axes and scale, and chunks of one time point and one field, so several writer threads may write it."""
    t, _, z, y, x = (int(n) for n in like["0"].shape)
    chunks = (1, 1, *(int(n) for n in like["0"].chunks[2:]))
    sink = iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=fields, axes=like.axes)
    sink.create_zeros("0", shape=(t if times is None else times, len(fields), z, y, x), dtype="float32", chunks=chunks)
    for axis, value in zip(sink.axis_names, like.get_effective_scale("0"), strict=True):
        if axis != "c":
            sink.set_scale("0", axis, float(value))
    return sink
