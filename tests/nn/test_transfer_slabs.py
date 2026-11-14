"""The transfer-function builds in slabs of planes give what the whole-grid builds gave, and keep their gradients.

Each ``_whole_*`` function is a private copy of a build without the slabs: every plane of the oversampled grid at
once, cropped to the data's frequencies at the end, of which the half of the Hermitian part is kept (the transfer
function of ``real(ifftn(crop))``). The slabbed builds must match them to rounding, for any slab size.
"""

import math

import pytest
import torch
from torch import Tensor

from waveorder.nn import (
    ConfocalFluorescence3D,
    Detection,
    Free,
    Sampling,
    ThickPhase3D,
    ThinFluorescence2D,
    ThinPhase2D,
    WidefieldFluorescence3D,
    functional,
)
from waveorder.nn.functional import (
    abs2,
    axial_nyquist,
    complex_multiply,
    irfftn,
    lateral_frequencies,
    transverse_nyquist,
)
from waveorder.nn.optics import (
    Illumination,
    IlluminationTilt,
    PupilGrid,
    ZernikeAberration,
    pinhole_otf,
)
from waveorder.nn.parameters import evaluate
from waveorder.nn.systems.phase import _greens_denominator, _stored_pupil

WAVELENGTH = 0.532
INDEX = 1.33
NA_ILL = 0.5
NA_DET = 0.9


# The whole-grid builds, as they were.


def _defocus(oblique: Tensor, z: Tensor) -> Tensor:
    phase = 2 * math.pi * z[:, None, None] * oblique
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def _conjugate(values: Tensor) -> Tensor:
    real, imaginary = values.unbind(-1)
    return torch.stack((real, -imaginary), dim=-1)


def _times_i(values: Tensor) -> Tensor:
    real, imaginary = values.unbind(-1)
    return torch.stack((-imaginary, real), dim=-1)


def _fft(values: Tensor, *, dim: tuple[int, ...], inverse: bool = False) -> Tensor:
    spectrum = torch.view_as_complex(values.contiguous())
    if not inverse:
        return torch.view_as_real(torch.fft.fftn(spectrum, dim=dim))
    points = math.prod(spectrum.shape[axis] for axis in dim)
    return torch.view_as_real(torch.fft.ifftn(spectrum, dim=dim, norm="forward")) / points


def _axial_fft(values: Tensor) -> Tensor:
    return _fft(values.movedim(-4, -2), dim=(-1,)).movedim(-2, -4)


def _real_operator_half(full: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Crop complex full spectra over their last ``len(shape)`` axes to the lowest frequencies of ``shape``, then
    keep the half of the crop's Hermitian part ``0.5 (H(k) + conj H(-k))``, stored real."""
    dims = tuple(range(-len(shape), 0))
    for dim, size in zip(dims, shape, strict=True):
        n = full.shape[dim]
        keep = torch.cat([torch.arange((size + 1) // 2), torch.arange(n - size // 2, n)])
        full = full.index_select(dim, keep)
    hermitian = 0.5 * (full + torch.roll(torch.flip(full, dims), (1,) * len(dims), dims).conj())
    return torch.view_as_real(hermitian[..., : shape[-1] // 2 + 1].contiguous())


def _whole_thick_phase(system: ThickPhase3D, sampling: Sampling) -> Tensor:
    reference = system.detection.numerical_aperture
    dtype, device = reference.dtype, reference.device
    fine = sampling.upsampled(system.upsampling(sampling))
    grid = PupilGrid.from_sampling(fine, device=device, dtype=dtype)
    fy, fx = lateral_frequencies(fine, device=device, dtype=dtype)
    pupil, support = _stored_pupil(system.detection, grid)
    source = system.illumination.source(
        fy[:, None], fx[None, :], index_of_refraction_media=system.index_of_refraction_media, spacing=grid.spacing
    )
    medium_wavelength = system.illumination.wavelength / system.index_of_refraction_media
    oblique = torch.sqrt(torch.clamp(1 - medium_wavelength**2 * grid.radius**2, min=0.0)) / medium_wavelength
    nz, dz = fine.shape[0], fine.pixel_size[0]
    planes = torch.fft.fftfreq(nz, d=1 / nz, device=device, dtype=dtype)
    z = planes * dz + system.detection.z_focus_offset * sampling.pixel_size[0]
    if system.invert_phase_contrast:
        z = -z
    propagation = support[..., None] * _defocus(oblique, z)
    denominator = _greens_denominator(oblique, support, system.detection.edge)
    greens = _times_i(propagation / (-4 * math.pi * denominator)[..., None])
    lateral = (-2, -1)
    illuminated = _fft(complex_multiply(source[..., None] * pupil, propagation), dim=lateral)
    scattered = _fft(complex_multiply(pupil, greens), dim=lateral)
    window = torch.fft.ifftshift(torch.hann_window(nz, periodic=False, dtype=dtype, device=device))
    window = window[:, None, None, None]
    first = _axial_fft(_fft(complex_multiply(_conjugate(illuminated), scattered), dim=lateral, inverse=True) * window)
    second = _axial_fft(_fft(complex_multiply(illuminated, _conjugate(scattered)), dim=lateral, inverse=True) * window)
    scale = dz / (source * abs2(pupil)).sum()
    stacked = torch.stack(((first + second) * scale, _times_i(first - second) * scale))
    return _real_operator_half(torch.view_as_complex(stacked.contiguous()), sampling.shape)


def _whole_widefield_psf(system: WidefieldFluorescence3D, sampling: Sampling) -> Tensor:
    reference = system.detection.numerical_aperture
    fine = sampling.upsampled(system.upsampling(sampling))
    grid = PupilGrid.from_sampling(fine, device=reference.device, dtype=reference.dtype)
    pupil = system.detection.pupil(grid)
    n_over_lambda = system.index_of_refraction_media / system.detection.wavelength
    kz = torch.sqrt(torch.clamp(n_over_lambda**2 - grid.radius**2, min=0.0))
    nz, dz = fine.shape[0], fine.pixel_size[0]
    planes = torch.fft.fftfreq(nz, d=1 / nz, device=reference.device, dtype=reference.dtype)
    z = planes * dz + system.detection.z_focus_offset * sampling.pixel_size[0]
    field = complex_multiply(pupil, _defocus(kz, z))
    intensity = abs2(torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(field))))
    return intensity / intensity.sum(dim=(-3, -2, -1), keepdim=True)


def _whole_widefield(system: WidefieldFluorescence3D, sampling: Sampling) -> Tensor:
    return _real_operator_half(torch.fft.fftn(_whole_widefield_psf(system, sampling)), sampling.shape)


def _whole_confocal(system: ConfocalFluorescence3D, sampling: Sampling) -> Tensor:
    widefield = torch.fft.rfftn(_whole_widefield_psf(system, sampling))
    fine = sampling.upsampled(system.upsampling(sampling))
    psf = irfftn(widefield, fine.shape)
    detected = irfftn(widefield * pinhole_otf(system.pinhole_diameter, fine), fine.shape)
    confocal = psf * detected
    return _real_operator_half(torch.fft.fftn(confocal / confocal.sum()), sampling.shape)


def _whole_thin_phase(system: ThinPhase2D, sampling: Sampling) -> Tensor:
    reference = system.detection.numerical_aperture
    dtype, device = reference.dtype, reference.device
    _, ny, nx = sampling.shape
    fine = Sampling(shape=(1, ny, nx), pixel_size=sampling.pixel_size).upsampled(system.upsampling(sampling))
    grid = PupilGrid.from_sampling(fine, device=device, dtype=dtype)
    fy, fx = lateral_frequencies(fine, device=device, dtype=dtype)
    pupil, support = _stored_pupil(system.detection, grid)

    def source_of(illumination: Illumination) -> Tensor:
        return illumination.source(
            fy[:, None], fx[None, :], index_of_refraction_media=system.index_of_refraction_media, spacing=grid.spacing
        )

    if system.clamps_source:
        na_ill, na_det = system.illumination.numerical_aperture, system.detection.numerical_aperture
        aperture = torch.where(na_ill >= na_det, 0.9 * na_det, na_ill)
        source = evaluate(source_of, system.illumination, {"numerical_aperture": aperture})
    else:
        source = source_of(system.illumination)
    medium_wavelength = system.illumination.wavelength / system.index_of_refraction_media
    oblique = torch.sqrt(torch.clamp(1 - medium_wavelength**2 * grid.radius**2, min=0.0)) / medium_wavelength
    defocus = _defocus(oblique, system.plane_positions(sampling))
    detected = complex_multiply(support[..., None] * pupil, defocus)
    lateral = (-2, -1)
    illuminated = _fft(source[..., None] * detected, dim=lateral)
    transmitted = _fft(detected, dim=lateral)
    first = _fft(complex_multiply(_conjugate(illuminated), transmitted), dim=lateral, inverse=True)
    second = _fft(complex_multiply(illuminated, _conjugate(transmitted)), dim=lateral, inverse=True)
    direct = (source * abs2(detected)).sum(dim=(-2, -1), keepdim=True)[..., None]
    stacked = torch.stack((_times_i(first - second) / direct, (first + second) / direct), dim=1)
    return _real_operator_half(torch.view_as_complex(stacked.contiguous()), (ny, nx)).unsqueeze(2)


def _whole_thin_fluorescence(system: ThinFluorescence2D, sampling: Sampling) -> Tensor:
    reference = system.detection.numerical_aperture
    _, ny, nx = sampling.shape
    fine = Sampling(shape=(1, ny, nx), pixel_size=sampling.pixel_size).upsampled(system.upsampling(sampling))
    grid = PupilGrid.from_sampling(fine, device=reference.device, dtype=reference.dtype)
    n_over_lambda = system.index_of_refraction_media / system.detection.wavelength
    kz = torch.sqrt(torch.clamp(n_over_lambda**2 - grid.radius**2, min=0.0))
    field = complex_multiply(system.detection.pupil(grid), _defocus(kz, system.plane_positions(sampling)))
    intensity = abs2(torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(field))))
    spectrum = torch.fft.fft2(intensity / intensity.sum(dim=(-2, -1), keepdim=True))
    planes = _real_operator_half(spectrum, (ny, nx))[:, None, None]
    return planes[..., 0] if system.real_transfer else planes


# The systems, each on grids that oversample by given factors.


def _pixels(factors: tuple[int, int, int], *, lateral: float, axial: float) -> tuple[float, float, float]:
    """Pixel sizes that :meth:`upsampling` turns into ``factors``: just over ``factor - 1`` steps."""
    kz, ky, kx = factors
    return ((kz - 0.5) * axial, (ky - 0.5) * lateral, (kx - 0.5) * lateral)


def _thick_phase(*, free: bool = False) -> tuple[ThickPhase3D, float, float]:
    na_ill = Free(NA_ILL, bounds=(0.4, 0.6)) if free else NA_ILL
    na_det = Free(NA_DET, bounds=(0.8, 1.0)) if free else NA_DET
    tilt = IlluminationTilt(zenith=Free(0.1, bounds=(-0.2, 0.2)) if free else 0.1, azimuth=0.7)
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=na_ill, tilt=tilt)
    offset = Free(0.3, bounds=(-1.0, 1.0)) if free else 0.3
    detection = Detection(numerical_aperture=na_det, z_focus_offset=offset)
    system = ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX, absorption=0.2)
    reach_ill, reach_det = (0.6, 1.0) if free else (NA_ILL, NA_DET)
    return system, transverse_nyquist(WAVELENGTH, reach_ill + reach_det), axial_nyquist(WAVELENGTH, reach_det, INDEX)


def _detection(*, free: bool) -> Detection:
    if not free:
        return Detection(wavelength=WAVELENGTH, numerical_aperture=NA_DET, z_focus_offset=0.3)
    return Detection(
        wavelength=WAVELENGTH,
        numerical_aperture=Free(NA_DET, bounds=(0.8, 1.0)),
        z_focus_offset=Free(0.3, bounds=(-1.0, 1.0)),
        pupil_aberrations=ZernikeAberration({"coma_x": Free(0.05, bounds=(-0.2, 0.2))}),
    )


def _widefield(*, free: bool = False) -> tuple[WidefieldFluorescence3D, float, float]:
    system = WidefieldFluorescence3D(_detection(free=free), index_of_refraction_media=INDEX)
    reach = system.na_limit
    return system, transverse_nyquist(WAVELENGTH, 2 * reach), axial_nyquist(WAVELENGTH, reach, INDEX)


def _confocal(*, free: bool = False) -> tuple[ConfocalFluorescence3D, float, float]:
    pinhole = Free(0.3, bounds=(0.0, 1.0)) if free else 0.3
    system = ConfocalFluorescence3D(_detection(free=free), pinhole_diameter=pinhole, index_of_refraction_media=INDEX)
    reach = system.na_limit
    return system, transverse_nyquist(WAVELENGTH, 2 * reach) / 2, axial_nyquist(WAVELENGTH, reach, INDEX) / 2


BUILDS = {
    "thick_phase": (_thick_phase, lambda system, sampling: system.potential_transfers(sampling), _whole_thick_phase),
    "widefield": (_widefield, lambda system, sampling: system._cropped_otf(sampling), _whole_widefield),  # noqa: SLF001
    "confocal": (_confocal, lambda system, sampling: system._cropped_otf(sampling), _whole_confocal),  # noqa: SLF001
}
GRIDS = [((6, 9, 8), (1, 4, 1)), ((5, 8, 7), (2, 3, 3)), ((4, 7, 10), (1, 1, 2))]


def _case(name: str, shape: tuple[int, int, int], factors: tuple[int, int, int], dtype: torch.dtype, *, free: bool):
    make, build, whole = BUILDS[name]
    system, lateral, axial = make(free=free)
    system = system.to(dtype)
    sampling = Sampling(shape=shape, pixel_size=_pixels(factors, lateral=lateral, axial=axial))
    assert system.upsampling(sampling) == factors
    return system, sampling, build, whole


def _error(actual: Tensor, expected: Tensor) -> float:
    return float((actual - expected).detach().abs().max() / expected.detach().abs().max())


@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize(("shape", "factors"), GRIDS)
@pytest.mark.parametrize("name", list(BUILDS))
def test_a_build_in_slabs_is_the_whole_grid_build(name, shape, factors, dtype, tolerance):
    system, sampling, build, whole = _case(name, shape, factors, dtype, free=False)
    with torch.no_grad():
        actual, expected = build(system, sampling), whole(system, sampling)
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    assert _error(actual, expected) <= tolerance


@pytest.mark.parametrize("planes", [1, 3])
@pytest.mark.parametrize("name", list(BUILDS))
def test_a_build_does_not_depend_on_the_slab_size(name, planes, monkeypatch):
    shape, factors = GRIDS[1]
    system, sampling, build, _ = _case(name, shape, factors, torch.float64, free=False)
    fine = sampling.upsampled(factors)
    with torch.no_grad():
        one_slab = build(system, sampling)
        monkeypatch.setattr(functional, "_SLAB_ELEMENTS", planes * fine.shape[1] * fine.shape[2])
        slabs = build(system, sampling)
    assert _error(slabs, one_slab) <= 1e-13


@pytest.mark.parametrize("name", list(BUILDS))
def test_a_build_in_slabs_has_the_whole_grid_gradients(name, monkeypatch):
    shape, factors = GRIDS[1]
    system, sampling, build, whole = _case(name, shape, factors, torch.float64, free=True)
    monkeypatch.setattr(functional, "_SLAB_ELEMENTS", 2 * math.prod(sampling.upsampled(factors).shape[1:]))
    names, parameters = zip(*system.named_parameters(), strict=True)
    generator = torch.Generator().manual_seed(3)
    weights = torch.randn(build(system, sampling).shape, generator=generator, dtype=torch.float64)
    actual = torch.autograd.grad((build(system, sampling) * weights).sum(), parameters)
    expected = torch.autograd.grad((whole(system, sampling) * weights).sum(), parameters)
    for parameter, slab, full in zip(names, actual, expected, strict=True):
        assert float(full.abs()) > 0, parameter
        assert float((slab - full).abs()) <= 1e-10 * float(full.abs()), parameter


def _thin_phase(dtype: torch.dtype) -> tuple[ThinPhase2D, Sampling, Tensor]:
    """Return the system on a grid upsampled ``(1, 4, 3)``, its sampling and its whole-grid plane transfers."""
    tilt = IlluminationTilt(zenith=0.1, azimuth=0.7)
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=NA_ILL, tilt=tilt)
    detection = Detection(numerical_aperture=Free(NA_DET, bounds=(0.8, 1.0)), z_focus_offset=0.3)
    system = ThinPhase2D(illumination, detection, index_of_refraction_media=INDEX).to(dtype)
    lateral = transverse_nyquist(WAVELENGTH, NA_ILL + 1.0)
    sampling = Sampling(shape=(5, 9, 8), pixel_size=_pixels((1, 4, 3), lateral=lateral, axial=0.4))
    return system, sampling, _whole_thin_phase(system, sampling)


def _thin_fluorescence(dtype: torch.dtype) -> tuple[ThinFluorescence2D, Sampling, Tensor]:
    """Return the system on a grid upsampled ``(1, 4, 3)``, its sampling and its whole-grid plane transfers."""
    system = ThinFluorescence2D(_detection(free=True), index_of_refraction_media=INDEX).to(dtype)
    lateral = transverse_nyquist(WAVELENGTH, 2 * system.na_limit)
    sampling = Sampling(shape=(5, 8, 7), pixel_size=_pixels((1, 4, 3), lateral=lateral, axial=0.4))
    return system, sampling, _whole_thin_fluorescence(system, sampling)


THIN = {"thin_phase": _thin_phase, "thin_fluorescence": _thin_fluorescence}


@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize("name", list(THIN))
def test_a_thin_build_in_slabs_is_the_whole_grid_build_with_its_gradients(name, dtype, tolerance, monkeypatch):
    system, sampling, expected = THIN[name](dtype)
    assert system.upsampling(sampling) == (1, 4, 3)
    monkeypatch.setattr(functional, "_SLAB_ELEMENTS", 2 * math.prod(sampling.upsampled((1, 4, 3)).shape[1:]))
    actual = system.plane_transfers(sampling)
    assert actual.shape == expected.shape
    assert _error(actual, expected) <= tolerance
    names, parameters = zip(*system.named_parameters(), strict=True)
    weights = torch.randn(actual.shape, generator=torch.Generator().manual_seed(3), dtype=dtype)
    slab_gradients = torch.autograd.grad((actual * weights).sum(), parameters)
    whole_gradients = torch.autograd.grad((expected * weights).sum(), parameters)
    for parameter, slab, full in zip(names, slab_gradients, whole_gradients, strict=True):
        assert float(full.abs()) > 0, parameter
        assert float((slab - full).abs()) <= 1e3 * tolerance * float(full.abs()), parameter
