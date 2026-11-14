"""The detection pupil's aperture edge: the antialiased ramp by default, the exact 3.x sigmoid on request."""

import pytest
import torch

from tests.nn.legacy_phase import frequencies, generate_pupil
from waveorder.errors import ParameterError
from waveorder.nn import Detection, Free, Sampling
from waveorder.nn.optics import PupilGrid

SAMPLING = Sampling(shape=(4, 48, 64), pixel_size=(0.25, 0.1, 0.12))
WAVELENGTH = 0.52


def _grid() -> PupilGrid:
    return PupilGrid.from_sampling(SAMPLING, device=torch.device("cpu"), dtype=torch.float64)


def test_the_sigmoid_edge_is_the_3x_pupil():
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid").double()
    pupil = detection.pupil(_grid())
    fyy, fxx = frequencies(SAMPLING.shape[1:], *SAMPLING.pixel_size[1:], dtype=torch.float64)
    legacy = generate_pupil(torch.sqrt(fxx**2 + fyy**2), 1.2, WAVELENGTH)
    torch.testing.assert_close(pupil[..., 0], legacy, atol=1e-12, rtol=0)
    assert bool((pupil[..., 1] == 0).all())


def test_the_ramp_edge_keeps_the_disk_area_and_a_numerical_aperture_gradient():
    # Same optics as the hard 3.x disk (equal area to a fraction of the edge ring), but differentiable in the NA.
    grid = _grid()
    ramp = Detection(wavelength=WAVELENGTH, numerical_aperture=Free(1.2, bounds=(1.0, 1.3))).double()
    sigmoid = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid").double()
    area = ramp.pupil(grid)[..., 0].sum()
    hard_area = sigmoid.pupil(grid)[..., 0].sum().item()
    ring = 2 * torch.pi * (1.2 / WAVELENGTH) / grid.spacing  # samples on the edge circle
    assert abs(area.item() - hard_area) < 0.1 * ring
    area.backward()
    assert ramp.numerical_aperture.grad is not None
    assert ramp.numerical_aperture.grad.item() > 0


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"numerical_aperture": 1.2, "edge": "hard"}, "edge"),
        ({"numerical_aperture": 0.0}, "numerical_aperture"),
        ({"numerical_aperture": Free(1.0)}, "numerical_aperture"),  # free but unbounded: could reach zero
        ({"numerical_aperture": 1.2, "z_focus_offset": float("nan")}, "z_focus_offset"),
    ],
    ids=["edge", "zero_aperture", "unbounded_free_aperture", "nan_focus"],
)
def test_errors_carry_the_parameter_path(kwargs, path):
    with pytest.raises(ParameterError) as error:
        Detection(wavelength=WAVELENGTH, **kwargs)
    assert error.value.path == path
