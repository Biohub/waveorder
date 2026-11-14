import pytest
import torch
from torch import Tensor

from waveorder.nn import (
    Detection,
    Free,
    Gaussian,
    ImagingModel,
    Sampling,
    ShiftInvariantSystem,
    Tikhonov,
    WidefieldFluorescence3D,
)
from waveorder.nn.parameters import Quantity, placement

WAVELENGTH = 0.52
INDEX = 1.33
SIGMA = 0.05
WEIGHT = 1e-2
NA = "system.detection.numerical_aperture"


@pytest.fixture
def sampling() -> Sampling:
    return Sampling(shape=(16, 64, 64), pixel_size=(0.25, 0.1, 0.1))


def make_system(numerical_aperture: Quantity = 1.0, **detection_kwargs) -> WidefieldFluorescence3D:
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=numerical_aperture, **detection_kwargs)
    return WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX)


def make_model(numerical_aperture: Quantity = 1.0, **detection_kwargs) -> ImagingModel:
    return ImagingModel(make_system(numerical_aperture, **detection_kwargs), Tikhonov(WEIGHT), Gaussian(SIGMA))


FREE_NA = Free(1.1, tolerance=0.2, bounds=(0.8, 1.2))


def transfer_of(model: ImagingModel, sampling: Sampling) -> Tensor:
    system = model.system
    assert isinstance(system, ShiftInvariantSystem)
    return system.transfer(sampling)


class MatrixSystem(ShiftInvariantSystem):
    """A system with a given transfer matrix: exercises any field and channel count.

    The matrix may be a full spectrum ``(C, F, Z, Y, X)`` (its half is used) or a half spectrum
    ``(C, F, Z, Y, X // 2 + 1)``.
    """

    def __init__(self, transfer: Tensor, *, fields: tuple[str, ...]) -> None:
        super().__init__(fields=fields, channel_count=transfer.shape[0])
        self.register_buffer("matrix", transfer)

    def compute_transfer(self, sampling: Sampling) -> Tensor:
        return self.get_buffer("matrix")[..., : sampling.shape[-1] // 2 + 1]


def random_transfer(channels: int, fields: int, shape, *, seed: int = 0, dtype=torch.float32) -> Tensor:
    """A random full-spectrum transfer matrix; :meth:`transfer` keeps its half and makes it exactly Hermitian."""
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(channels, fields, *shape, generator=generator, dtype=dtype)
    return torch.complex(real, torch.randn(channels, fields, *shape, generator=generator, dtype=dtype))


def simulate(model: ImagingModel, sampling: Sampling, *, scale: float, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    dtype, device = placement(model.system)
    obj = torch.randn(model.system.object_shape(sampling), generator=generator, dtype=dtype).to(device) * scale
    with torch.no_grad():
        data = model.noise.sample(model(obj, sampling), generator=generator)
    return obj, data
