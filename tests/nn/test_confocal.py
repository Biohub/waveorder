"""Confocal fluorescence: a point pinhole squares the widefield PSF, the kit, refusals."""

import math

import pytest
import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.nn import (
    ConfocalFluorescence3D,
    Detection,
    Free,
    Sampling,
    WidefieldFluorescence3D,
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


def test_a_point_pinhole_squares_the_widefield_psf_and_a_wide_one_tends_to_widefield():
    # d = 0 gives PSF^2; a pinhole much wider than the PSF passes the whole widefield PSF.
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
