"""Thin (2D) fluorescence: parity with unmodified 3.x ``isotropic_fluorescent_thin_3d``, the system kit, refusals.

3.x parity uses the 3.x pupil edge (``Detection(edge="sigmoid")``) on grids where the 3.x central crop keeps the
lowest frequencies (even axes, or odd axes oversampled by an odd factor); the crop fix and the default ramp edge are
pinned separately.
"""

import math

import pytest
import torch
from torch import Tensor

from tests.nn import legacy_fluorescence as legacy
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
from waveorder.nn.operators import ReshapedFourierDiagonal
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


def _reference(system: ThinFluorescence2D, sampling: Sampling, *, crop: str = "central") -> Tensor:
    """Unmodified 3.x transfers; 3.x took explicit positions, with the focus offset added by its caller."""
    nz, ny, nx = sampling.shape
    offset = float(system.detection.z_focus_offset)
    if system.z_positions is None:
        z = legacy.thin_z_positions(nz, sampling.pixel_size[0], offset=offset)
    else:
        z = torch.tensor(system.z_positions, dtype=torch.float64) + offset * sampling.pixel_size[0]
    return legacy.thin_fluorescence_transfer(
        (ny, nx), sampling.pixel_size[1], z, wavelength=WAVELENGTH, index=INDEX, na_detection=NA, crop=crop
    )


def _planes(system: ThinFluorescence2D, sampling: Sampling) -> Tensor:
    """The stored plane transfers, ``(Z, Y, X // 2 + 1)``."""
    operator = system.compute_operator(sampling)
    return operator.inner.transfer[:, 0, 0]


THIN_CASES = [
    pytest.param(10, 0.2, 0.0, None, id="even-oversampled-by-2"),
    pytest.param(9, 0.1, 0.3, None, id="odd-not-oversampled-offset"),
    pytest.param(9, 0.35, 1.0, None, id="odd-oversampled-by-3-offset"),
    pytest.param(12, 0.35, 0.0, (1.3, 0.2, -0.4, -1.5, 0.9), id="uneven-planes"),
]


@pytest.mark.parametrize(("size", "lateral", "offset", "positions"), THIN_CASES)
def test_thin_fluorescence_transfer_with_the_sigmoid_edge_matches_unmodified_3x(size, lateral, offset, positions):
    sampling = Sampling(shape=(5, size, size), pixel_size=(0.5, lateral, lateral))
    system = _system(offset=offset, positions=positions)
    reference = _reference(system, sampling)
    assert reference.imag.abs().max() <= 1e-12, "an even PSF: 3.x's complex transfer is real"
    operator = system.bind(sampling)
    assert isinstance(operator, ReshapedFourierDiagonal)
    assert (operator.object_shape, operator.data_shape) == ((1, 1, size, size), (1, 5, size, size))
    assert _relative(_planes(system, sampling), reference.real[..., : size // 2 + 1]) <= 1e-12
    single = _planes(system.float(), sampling)
    assert _relative(single.double(), reference.real[..., : size // 2 + 1]) <= 1e-5


def test_thin_fluorescence_keeps_the_lowest_frequencies_where_the_3x_crop_moves_them():
    # An odd axis oversampled by an even factor: 3.x's central crop keeps -5..3, the 4.x crop -4..4.
    sampling = Sampling(shape=(3, 9, 9), pixel_size=(0.5, 0.2, 0.2))
    system = _system()
    assert system.upsampling(sampling) == (1, 2, 2)
    planes = _planes(system, sampling)
    lowest = _reference(system, sampling, crop="lowest").real[..., :5]
    assert _relative(planes, lowest) <= 1e-12
    assert _relative(planes, _reference(system, sampling).real[..., :5]) > 0.05


def test_thin_fluorescence_ramp_edge_gap_to_unmodified_3x_is_pinned():
    """Measured at 64 x 64 (0.1 um), 5 planes 0.5 um apart: 3.4%."""
    sampling = Sampling(shape=(5, 64, 64), pixel_size=(0.5, 0.1, 0.1))
    system = _system(edge="ramp")
    gap = _relative(_planes(system, sampling), _reference(system, sampling).real[..., :33])
    assert gap == pytest.approx(0.0339, abs=2e-3)


def _scene() -> tuple[ThinFluorescence2D, Sampling, Tensor, Tensor]:
    sampling = Sampling(shape=(7, 32, 32), pixel_size=(0.6, 0.1, 0.1))
    system = _system(offset=0.5)
    y, x = torch.meshgrid(*(torch.arange(32, dtype=torch.float64) - 16,) * 2, indexing="ij")
    density = torch.sigmoid(4 * (1.5 - torch.sqrt(y**2 + x**2) * 0.1 / 0.3)) + 0.3 * (x > 4)
    return system, sampling, density, _reference(system, sampling)


def test_thin_fluorescence_forward_model_matches_unmodified_3x():
    system, sampling, density, transfer = _scene()
    data = system.forward(density[None, None], sampling)
    assert data.shape == (1, 7, 32, 32)
    assert _relative(data[0], legacy.thin_fluorescence_forward(density, transfer)) <= 1e-12


def test_thin_fluorescence_tikhonov_reconstruction_matches_unmodified_3x():
    system, sampling, density, transfer = _scene()
    data = legacy.thin_fluorescence_forward(density, transfer) + 10  # 3.x simulates a background of 10
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    estimate = Wiener()(data[None], model, sampling).estimate
    assert estimate.shape == (1, 1, 32, 32)
    expected = legacy.thin_fluorescence_inverse(data, transfer, regularization_strength=weight)
    assert _relative(estimate[0, 0], expected) <= 1e-9


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


def test_widefield_tikhonov_reconstruction_on_raw_counts_matches_unmodified_3x():
    # 16 x 32 x 32 at 0.5 / 0.25 / 0.25 um and NA 0.75: oversampled 2x in y and x, not in z.
    sampling = Sampling(shape=(16, 32, 32), pixel_size=(0.5, 0.25, 0.25))
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=0.75, edge="sigmoid")
    system = WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX).double()
    assert system.upsampling(sampling) == (1, 2, 2)
    otf = legacy.thick_transfer((16, 32, 32), 0.25, 0.5, wavelength=WAVELENGTH, index=INDEX, na_detection=0.75)
    assert _relative(system.transfer(sampling)[0, 0], otf.real[..., :17]) <= 1e-12
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) - n // 2 for n in sampling.shape), indexing="ij")
    density = torch.sigmoid(4 * (1.5 - torch.sqrt((0.5 * z) ** 2 + (0.25 * y) ** 2 + (0.25 * x) ** 2)))
    counts = legacy.thick_forward(density, otf) + 10  # 3.x simulates a background of 10
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    estimate = Wiener()(counts[None], model, sampling).estimate
    assert _relative(estimate[0], legacy.thick_inverse(counts, otf, regularization_strength=weight)) <= 1e-12


@pytest.mark.parametrize("pinhole", [None, 0.3], ids=["widefield", "confocal"])
def test_the_cropped_point_spread_function_is_clamped_at_zero_as_in_3x(pinhole):
    """On a grid that undersamples the OTF the crop rings the PSF below zero (0.9% of its mass widefield, 12%
    confocal). 3.x clamps it there and renormalizes; without that the transfer differs by 3% and 12%. The reference
    matches 3.x itself (wo-4.0 at 85282ce, float32) to 4e-7 on this grid."""
    sampling = Sampling(shape=(8, 16, 16), pixel_size=(0.8, 0.3, 0.3))
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=NA, edge="sigmoid")
    if pinhole is None:
        system = WidefieldFluorescence3D(detection, index_of_refraction_media=INDEX).double()
        assert system.upsampling(sampling) == (2, 3, 3)
    else:
        system = ConfocalFluorescence3D(detection, pinhole_diameter=pinhole, index_of_refraction_media=INDEX).double()

    def reference(*, clamp: bool) -> Tensor:
        return legacy.thick_transfer(
            sampling.shape,
            0.3,
            0.8,
            wavelength=WAVELENGTH,
            index=INDEX,
            na_detection=NA,
            pinhole_diameter=pinhole,
            zero_frequency=math.pi / 4,
            clamp=clamp,
        )

    clamped, unclamped = reference(clamp=True), reference(clamp=False)
    transfer = system.transfer(sampling)[0, 0]
    assert _relative(transfer, clamped.real[..., :9]) <= 1e-12
    assert _relative(transfer, unclamped.real[..., :9]) > 0.02
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
