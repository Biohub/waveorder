"""Thin (2D) fluorescence: the system kit, refusals; the cropped widefield and confocal PSF stays nonnegative."""

import math

import pytest
import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn import (
    ConfocalFluorescence3D,
    Detection,
    Free,
    Gaussian,
    ImagingModel,
    Sampling,
    ThinFluorescence2D,
    Tikhonov,
    WidefieldFluorescence3D,
    Wiener,
    ZernikeAberration,
)
from waveorder.nn.optics import PupilEdge
from waveorder.nn.systems._checks import (
    MAX_TRANSFER_VOXELS,
    MAX_UPSAMPLING,
    expect_feasible_grid,
    expect_resolved_aperture,
)
from waveorder.testing import check_system

WAVELENGTH = 0.532
INDEX = 1.33
NA = 0.9


def _relative(actual: Tensor, expected: Tensor) -> float:
    return ((actual - expected).norm() / expected.norm()).item()


def _system(*, offset: float = 0.0, edge: PupilEdge = "sigmoid", positions=None) -> ThinFluorescence2D:
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=NA, z_focus_offset=offset, edge=edge)
    return ThinFluorescence2D(detection, index_of_refraction_media=INDEX, z_positions=positions).double()


@pytest.mark.parametrize(
    "detection",
    [
        pytest.param(
            dict(numerical_aperture=Free(0.9, bounds=(0.8, 1.0)), z_focus_offset=Free(0.2, bounds=(-1, 1))), id="free"
        ),
        pytest.param(
            dict(numerical_aperture=0.9, pupil_aberrations=ZernikeAberration({"coma_x": 0.2})),
            id="complex-transfer",
        ),
    ],
)
def test_thin_fluorescence_passes_the_system_suite(detection):
    sampling = Sampling(shape=(4, 10, 9), pixel_size=(0.5, 0.15, 0.15))

    def build() -> ThinFluorescence2D:
        arm = Detection(wavelength=WAVELENGTH, **detection)
        return ThinFluorescence2D(arm, index_of_refraction_media=INDEX).double()

    assert build().real_transfer == ("pupil_aberrations" not in detection)
    check_system(build, sampling)


REFUSED = {
    "index nan": (
        lambda: ThinFluorescence2D(
            Detection(wavelength=WAVELENGTH, numerical_aperture=NA), index_of_refraction_media=math.nan
        ),
        "index_of_refraction_media",
        "positive and finite",
    ),
    "aperture above index": (
        lambda: ThinFluorescence2D(
            Detection(wavelength=WAVELENGTH, numerical_aperture=1.4), index_of_refraction_media=INDEX
        ),
        "detection.numerical_aperture",
        "above index_of_refraction_media",
    ),
    "z_positions empty": (
        lambda: ThinFluorescence2D(
            Detection(wavelength=WAVELENGTH, numerical_aperture=NA), index_of_refraction_media=INDEX, z_positions=()
        ),
        "z_positions",
        "at least one",
    ),
}


@pytest.mark.parametrize("case", REFUSED)
def test_thin_fluorescence_refuses_wrong_arguments_at_their_path(case):
    build, path, fragment = REFUSED[case]
    with pytest.raises(ParameterError, match=fragment) as caught:
        build()
    assert caught.value.path == path


def test_thin_fluorescence_refuses_a_stack_with_another_plane_count():
    system = _system(positions=(0.0, 0.5))
    with pytest.raises(IncompatibleModelError, match="z_positions lists 2 planes, the data have 3"):
        system.bind(Sampling(shape=(3, 8, 8), pixel_size=(0.5, 0.1, 0.1)))


@pytest.mark.parametrize("pinhole", [None, 0.3], ids=["widefield", "confocal"])
def test_the_cropped_point_spread_function_is_clamped_at_zero(pinhole):
    """On a grid that undersamples the OTF the crop rings the PSF below zero (0.9% of its mass widefield, 12%
    confocal); the systems clamp it there and renormalize."""
    sampling = Sampling(shape=(8, 16, 16), pixel_size=(0.8, 0.3, 0.3))
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=NA, edge="sigmoid")
    if pinhole is None:
        system = WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX).double()
        assert system.upsampling(sampling) == (2, 3, 3)
    else:
        system = ConfocalFluorescence3D(detection, pinhole_diameter=pinhole, index_of_refraction_media=INDEX).double()
    transfer = system.transfer(sampling)[0, 0]
    psf = torch.fft.irfftn(transfer.to(torch.complex128), s=sampling.shape)
    assert psf.min() >= -1e-15 * psf.max()


@pytest.mark.parametrize("kind", [WidefieldFluorescence3D, ThinFluorescence2D])
def test_fluorescence_requires_the_emission_wavelength(kind):
    # Only the phase systems have an illumination to take the wavelength from.
    with pytest.raises(ParameterError, match=rf"{kind.__name__} needs detection\.wavelength, the emission") as error:
        kind(Detection(numerical_aperture=NA), index_of_refraction_media=INDEX)
    assert error.value.path == "detection.wavelength"


@pytest.mark.parametrize("kind", [WidefieldFluorescence3D, ThinFluorescence2D])
def test_an_emission_wavelength_in_nm_is_refused_before_anything_is_computed(kind):
    sampling = Sampling(shape=(4, 16, 16), pixel_size=(0.5, 0.1, 0.1))
    system = kind(Detection(wavelength=520, numerical_aperture=NA), index_of_refraction_media=INDEX)
    with pytest.raises(IncompatibleModelError, match=rf"^{kind.__name__}: .*wavelength is in um, got 520 "):
        system.check_sampling(sampling)
    model = ImagingModel(system, Tikhonov(1e-3), Gaussian(1.0))
    with pytest.raises(IncompatibleModelError, match="the pupil holds no frequency but zero"):
        Wiener()(torch.zeros(system.data_shape(sampling)), model, sampling)
    kind(Detection(wavelength=0.52, numerical_aperture=NA), index_of_refraction_media=INDEX).check_sampling(sampling)


def test_the_aperture_check_refuses_only_a_cut_off_below_one_frequency_step():
    # 16 x 20 pixels of 0.1 um: frequency steps 0.625 and 0.5 cycles/um; the finer one, of the wider axis, decides.
    sampling = Sampling(shape=(4, 16, 20), pixel_size=(0.5, 0.1, 0.1))
    expect_resolved_aperture(sampling, numerical_aperture=0.5, wavelength=1.0, where="System")  # cut-off = one step
    expect_resolved_aperture(sampling, numerical_aperture=0.3, wavelength=0.5, where="System")
    with pytest.raises(IncompatibleModelError, match=r"NA / wavelength = 0\.49 / 1 = 0\.49 cycles/um is below one"):
        expect_resolved_aperture(sampling, numerical_aperture=0.49, wavelength=1.0, where="System")
    with pytest.raises(IncompatibleModelError, match=r"0\.5 cycles/um, of a 16 x 20 grid of 0\.1 x 0\.1 um pixels"):
        expect_resolved_aperture(sampling, numerical_aperture=0.9, wavelength=532, where="System")


@pytest.mark.parametrize("kind", [WidefieldFluorescence3D, ThinFluorescence2D])
def test_a_pixel_size_in_nm_is_refused_before_anything_is_allocated(kind):
    """150 nm written as 150 um asks for a transfer function grid a thousand times finer along y and x: terabytes,
    which used to end in the allocator's error after half a minute."""
    in_nm = Sampling(shape=(8, 32, 32), pixel_size=(400, 150, 150))
    system = kind(Detection(wavelength=0.52, numerical_aperture=NA), index_of_refraction_media=INDEX)
    message = (
        rf"^{kind.__name__}: a pixel size of 400 x 150 x 150 um \(z, y, x\) is \d+ x 1039 x 1039 times the steps that "
        r"sample these optics, .*pixel_size is in um, got 150 along y \(a value in nm is 1000 times too large\); "
        r"a camera pixel is divided by the magnification$"
    )
    with pytest.raises(IncompatibleModelError, match=message):
        system.check_sampling(in_nm)
    model = ImagingModel(system, Tikhonov(1e-3), Gaussian(1.0))
    with pytest.raises(IncompatibleModelError, match="pixel_size is in um, got 150 along y"):
        Wiener()(torch.zeros(system.data_shape(in_nm)), model, in_nm)
    with pytest.raises(IncompatibleModelError, match="pixel_size is in um, got 150 along y"):
        system(torch.zeros(system.object_shape(in_nm)), in_nm)  # bind checks the sampling first
    # A coarse pixel that runs today still passes: 6.4 um is 45 times the Nyquist step of these optics.
    coarse = Sampling(shape=(4, 16, 16), pixel_size=(0.5, 6.4, 6.4))
    system.check_sampling(coarse)
    assert 40 < max(system.upsampling(coarse)) <= MAX_UPSAMPLING


def test_a_z_step_in_nm_is_named_along_z():
    system = WidefieldFluorescence3D(Detection(wavelength=0.52, numerical_aperture=NA), index_of_refraction_media=INDEX)
    with pytest.raises(IncompatibleModelError, match=r"is 540 x 1 x 1 times .*pixel_size is in um, got 400 along z "):
        system.check_sampling(Sampling(shape=(8, 32, 32), pixel_size=(400, 0.1, 0.1)))


def test_the_focus_offset_of_a_3d_object_is_what_a_likelihood_cannot_determine():
    detection = Detection(wavelength=0.52, numerical_aperture=NA)
    widefield = WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX)
    confocal = ConfocalFluorescence3D(detection, pinhole_diameter=0.3, index_of_refraction_media=INDEX)
    thin = ThinFluorescence2D(detection, index_of_refraction_media=INDEX)
    assert widefield.undetermined_by_likelihood() == ("detection.z_focus_offset",)
    assert confocal.undetermined_by_likelihood() == ("detection.z_focus_offset",)
    assert thin.undetermined_by_likelihood() == ()  # the offset selects the plane in focus: the data change with it


def test_the_grid_check_refuses_only_an_absurd_factor_or_voxel_count():
    sampling = Sampling(shape=(4, 16, 20), pixel_size=(0.5, 0.1, 0.1))
    expect_feasible_grid(sampling, (1, 1, 1), where="System")
    expect_feasible_grid(sampling, (MAX_UPSAMPLING, MAX_UPSAMPLING, MAX_UPSAMPLING), where="System")
    with pytest.raises(IncompatibleModelError, match=r"^System: .*is 1 x 1 x 101 times .*got 0\.1 along x "):
        expect_feasible_grid(sampling, (1, 1, MAX_UPSAMPLING + 1), where="System")
    # A grid that no pixel-size mistake explains is refused for its size, with the way out.
    huge = Sampling(shape=(4096, 4096, 4096), pixel_size=(0.5, 0.1, 0.1))
    assert MAX_TRANSFER_VOXELS == 4096**3
    expect_feasible_grid(huge, (1, 1, 1), where="System")
    with pytest.raises(IncompatibleModelError, match=r"on a 8192 x 4096 x 4096 grid, .* tile by tile$"):
        expect_feasible_grid(huge, (2, 1, 1), where="System")
