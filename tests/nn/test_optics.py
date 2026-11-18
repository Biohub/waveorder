"""The detection pupil's aperture edge: the antialiased ramp by default, the exact 3.x sigmoid on request."""

import pytest
import torch

from waveorder import registry
from waveorder.errors import ParameterError
from waveorder.nn import Detection, Free, Illumination, Sampling, ThickPhase3D, ThinPhase2D
from waveorder.nn.optics import PupilGrid

SAMPLING = Sampling(shape=(4, 48, 64), pixel_size=(0.25, 0.1, 0.12))
WAVELENGTH = 0.52


def _grid() -> PupilGrid:
    return PupilGrid.from_sampling(SAMPLING, device=torch.device("cpu"), dtype=torch.float64)


def test_the_sigmoid_edge_is_the_3x_pupil():
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid").double()
    pupil = detection.pupil(_grid())
    fy = torch.fft.fftfreq(SAMPLING.shape[1], SAMPLING.pixel_size[1], dtype=torch.float64)
    fx = torch.fft.fftfreq(SAMPLING.shape[2], SAMPLING.pixel_size[2], dtype=torch.float64)
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    expected = torch.sigmoid(1e4 * (1.2 / WAVELENGTH - radius))  # the 3.x pupil
    torch.testing.assert_close(pupil[..., 0], expected, atol=1e-12, rtol=0)
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


def test_the_sigmoid_steepness_sets_the_edge_and_its_default_is_the_3x_pupil():
    grid = _grid()
    default = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid").double()
    explicit = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid", edge_steepness=1e4).double()
    soft = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid", edge_steepness=100.0).double()
    assert torch.equal(explicit.pupil(grid), default.pupil(grid))
    expected = torch.sigmoid(100.0 * (1.2 / WAVELENGTH - grid.radius))  # 3.x generate_pupil(steepness=100)
    torch.testing.assert_close(soft.pupil(grid)[..., 0], expected, atol=1e-12, rtol=0)
    assert (soft.pupil(grid) - default.pupil(grid)).abs().max() > 0.1


# wo-4.0 (85282ce) isotropic_thin_3d.calculate_transfer_function((16, 16), 1.3, z=(25, 0, -25), 0.45, 1.0,
# numerical_aperture_illumination=0.1, numerical_aperture_detection=0.15, pupil_steepness=100.0), run in its own
# environment (float32): (absorption_tf, phase_tf) at [plane, y, x]. At the default 1e4 these are 1.6918, 0.9944,
# -0.07464 and 0.07271.
OLD_SOFT_EDGE = {
    (0, 1, 15, 2): 1.7273892164230347,
    (0, 0, 1, 2): 1.0284197330474854,
    (1, 2, 0, 8): -0.06735335290431976,
    (1, 0, 8, 1): 0.0662728101015091,
}


def test_a_thin_transfer_at_steepness_100_is_wo40s_at_pupil_steepness_100():
    sampling = Sampling(shape=(3, 16, 16), pixel_size=(25.0, 1.3, 1.3))
    illumination = Illumination(wavelength=0.45, numerical_aperture=0.1)
    detection = Detection(numerical_aperture=0.15, edge="sigmoid", edge_steepness=100.0)
    system = ThinPhase2D(illumination, detection, index_of_refraction_media=1.0).double()
    transfer = torch.view_as_complex(system.plane_transfers(sampling)[:, :, 0].contiguous())  # (Z, F, Y, X // 2 + 1)
    for (field, plane, y, x), expected in OLD_SOFT_EDGE.items():
        index = 1 - field  # 3.x lists absorption first, the system phase first
        assert float(transfer[plane, index, y, x].real) == pytest.approx(expected, rel=1e-5)


def test_a_steepness_for_a_calibration_holds_inside_the_block_and_no_cached_transfer_outlives_it():
    detection = Detection(numerical_aperture=1.2, edge="sigmoid")
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=0.5)
    system = ThickPhase3D(illumination, detection, index_of_refraction_media=1.3, cache="auto").double()
    sharp = system.transfer(SAMPLING)
    with detection.using_edge_steepness(100.0):
        assert detection.edge_steepness == 100.0
        assert registry.describe(detection)["detection"]["edge_steepness"] == 1e4  # the detection as built
        soft = system.transfer(SAMPLING)
    assert detection.edge_steepness == 1e4
    assert not torch.equal(soft, sharp)
    assert torch.equal(system.transfer(SAMPLING), sharp)


def test_the_steepness_round_trips_through_describe_and_resolve():
    soft = Detection(wavelength=WAVELENGTH, numerical_aperture=1.2, edge="sigmoid", edge_steepness=100.0)
    rebuilt = registry.resolve(registry.describe(soft), "detection", Detection)
    assert rebuilt.edge_steepness == 100.0
    assert torch.equal(rebuilt.pupil(_grid()), soft.pupil(_grid()))


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"numerical_aperture": 1.2, "edge": "hard"}, "edge"),
        ({"numerical_aperture": 0.0}, "numerical_aperture"),
        ({"numerical_aperture": Free(1.0)}, "numerical_aperture"),  # free but unbounded: could reach zero
        ({"numerical_aperture": 1.2, "z_focus_offset": float("nan")}, "z_focus_offset"),
        ({"numerical_aperture": 1.2, "edge_steepness": 100.0}, "edge_steepness"),  # the ramp edge has none
        ({"numerical_aperture": 1.2, "edge": "sigmoid", "edge_steepness": 0.0}, "edge_steepness"),
    ],
    ids=["edge", "zero_aperture", "unbounded_free_aperture", "nan_focus", "ramp_steepness", "zero_steepness"],
)
def test_errors_carry_the_parameter_path(kwargs, path):
    with pytest.raises(ParameterError) as error:
        Detection(wavelength=WAVELENGTH, **kwargs)
    assert error.value.path == path
