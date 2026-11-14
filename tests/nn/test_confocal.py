"""Confocal fluorescence: parity with the 3.x confocal path of ``isotropic_fluorescent_thick_3d``, the kit, refusals.

3.x parity uses the 3.x pupil edge (``Detection(edge="sigmoid")``) on grids where the 3.x central crop keeps the lowest
frequencies, and the pinhole OTF's limit at zero frequency, which 3.x set to another value; that difference is pinned.
"""

import math

import pytest
import torch
from torch import Tensor

from tests.nn import legacy_fluorescence as legacy
from waveorder.errors import ParameterError
from waveorder.nn import (
    ConfocalFluorescence3D,
    Detection,
    Free,
    Gaussian,
    ImagingModel,
    Sampling,
    Tikhonov,
    WidefieldFluorescence3D,
    Wiener,
)
from waveorder.nn.functional import crop_half_spectrum
from waveorder.nn.optics import PupilEdge
from waveorder.testing import check_system

WAVELENGTH = 0.532
INDEX = 1.33
NA = 0.9


def _relative(actual: Tensor, expected: Tensor) -> float:
    return ((actual - expected).norm() / expected.norm()).item()


def _system(pinhole: float, *, edge: PupilEdge = "sigmoid") -> ConfocalFluorescence3D:
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=NA, edge=edge)
    return ConfocalFluorescence3D(detection, pinhole_diameter=pinhole, index_of_refraction_media=INDEX).double()


def _reference(sampling: Sampling, pinhole: float, *, zero_frequency: float) -> Tensor:
    dz, dy, _ = sampling.pixel_size
    return legacy.thick_transfer(
        sampling.shape,
        dy,
        dz,
        wavelength=WAVELENGTH,
        index=INDEX,
        na_detection=NA,
        pinhole_diameter=pinhole,
        zero_frequency=zero_frequency,
    )


CONFOCAL_CASES = [
    pytest.param((8, 10, 10), (0.4, 0.2, 0.2), 0.3, id="even-oversampled-by-2-and-3"),
    pytest.param((6, 12, 12), (0.3, 0.1, 0.1), 1.0, id="wide-pinhole"),
    pytest.param((8, 10, 10), (0.4, 0.2, 0.2), 0.0, id="point-pinhole"),
]


@pytest.mark.parametrize(("shape", "pixel", "pinhole"), CONFOCAL_CASES)
def test_confocal_transfer_matches_3x_with_the_pinhole_limit_at_zero_frequency(shape, pixel, pinhole):
    sampling = Sampling(shape=shape, pixel_size=pixel)
    system = _system(pinhole)
    reference = _reference(sampling, pinhole, zero_frequency=math.pi / 4)
    assert reference.imag.abs().max() <= 1e-12, "an even PSF: 3.x's complex transfer is real"
    assert system.real_transfer
    transfer = system.transfer(sampling)[0, 0]
    assert _relative(transfer, reference.real[..., : shape[2] // 2 + 1]) <= 1e-12


def test_confocal_gap_to_unmodified_3x_is_its_zero_frequency_pinhole_value():
    """3.x's ``0.5`` at zero frequency (the limit is ``pi / 4``): 4.3% at 16 x 32 x 32, a 0.4 um pinhole."""
    sampling = Sampling(shape=(16, 32, 32), pixel_size=(0.25, 0.1, 0.1))
    transfer = _system(0.4).transfer(sampling)[0, 0]
    unmodified = _reference(sampling, 0.4, zero_frequency=0.5).real[..., :17]
    assert _relative(transfer, unmodified) == pytest.approx(0.043, abs=3e-3)
    # A point pinhole is constant either way: there the 3.x value changes nothing.
    point = _system(0.0).transfer(sampling)[0, 0]
    assert _relative(point, _reference(sampling, 0.0, zero_frequency=0.5).real[..., :17]) <= 1e-12


def test_a_point_pinhole_squares_the_widefield_psf_and_a_wide_one_tends_to_widefield():
    # Independent of 3.x: d = 0 gives PSF^2; a pinhole much wider than the PSF passes the whole widefield PSF.
    sampling = Sampling(shape=(8, 16, 16), pixel_size=(0.4, 0.1, 0.1))
    widefield = WidefieldFluorescence3D(
        Detection(wavelength=WAVELENGTH, numerical_aperture=NA, edge="sigmoid"), index_of_refraction_media=INDEX
    ).double()
    point, fine = _system(0.0), sampling.upsampled(_system(0.0).upsampling(sampling))
    assert widefield.upsampling(fine) == (1, 1, 1)  # so its cropped OTF is the one of the fine grid
    psf = torch.fft.irfftn(torch.view_as_complex(widefield._cropped_otf(fine)), s=fine.shape)  # noqa: SLF001
    squared = crop_half_spectrum(torch.view_as_real(torch.fft.rfftn(psf * psf / (psf * psf).sum())), sampling.shape)
    assert _relative(point._cropped_otf(sampling), squared) <= 1e-12  # noqa: SLF001
    wide = _system(50.0).transfer(sampling)[0, 0]
    assert _relative(wide, point.transfer(sampling)[0, 0]) > 0.1
    assert _relative(wide, _system(0.0).transfer(sampling)[0, 0]) > 0.1


def test_confocal_tikhonov_reconstruction_matches_3x():
    sampling = Sampling(shape=(8, 16, 16), pixel_size=(0.4, 0.1, 0.1))
    system = _system(0.3)
    otf = _reference(sampling, 0.3, zero_frequency=math.pi / 4)
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) - n // 2 for n in sampling.shape), indexing="ij")
    density = torch.sigmoid(4 * (1.0 - torch.sqrt((0.4 * z) ** 2 + (0.1 * y) ** 2 + (0.1 * x) ** 2)))
    data = legacy.thick_forward(density, otf)
    assert _relative(system.forward(density[None], sampling)[0], data) <= 1e-12
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    estimate = Wiener()(data[None], model, sampling).estimate
    assert _relative(estimate[0], legacy.thick_inverse(data, otf, regularization_strength=weight)) <= 1e-9


def test_confocal_passes_the_system_suite_with_a_free_pinhole():
    # Finer than the confocal Nyquist steps for the reachable NA 1 (0.067 um laterally, 0.29 um axially): coarser
    # data crop the doubled frequency support, and the cropped PSF rings below zero.
    sampling = Sampling(shape=(6, 10, 9), pixel_size=(0.25, 0.06, 0.06))

    def build() -> ConfocalFluorescence3D:
        detection = Detection(wavelength=WAVELENGTH, numerical_aperture=Free(0.9, bounds=(0.8, 1.0)))
        pinhole = Free(0.3, bounds=(0.0, 1.0))
        return ConfocalFluorescence3D(detection, pinhole_diameter=pinhole, index_of_refraction_media=INDEX).double()

    assert "pinhole_diameter" in dict(build().named_parameters())
    check_system(build, sampling)


@pytest.mark.parametrize(
    ("pinhole", "fragment"),
    [
        (-0.1, "must stay at least 0 um"),
        (Free(0.3), "must stay at least 0 um"),
        (Free(0.3, bounds=(-0.5, 1.0)), "must stay at least 0 um"),
        (math.inf, "must be finite"),
    ],
)
def test_confocal_refuses_a_pinhole_diameter_that_can_go_below_zero_or_is_not_finite(pinhole, fragment):
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=NA)
    with pytest.raises(ParameterError, match=fragment) as caught:
        ConfocalFluorescence3D(detection, pinhole_diameter=pinhole, index_of_refraction_media=INDEX)
    assert caught.value.path == "pinhole_diameter"
