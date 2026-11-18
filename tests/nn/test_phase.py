"""Brightfield phase systems: units, first-order oracles, inverses and the module contracts."""

import math
from typing import Any, Literal

import pytest
import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn import (
    FISTA,
    Detection,
    Evidence,
    Free,
    Gaussian,
    GradientSearch,
    GridSearch,
    ImagingModel,
    Sampling,
    Tikhonov,
    Wiener,
    calibrate,
)
from waveorder.nn.functional import lateral_frequencies
from waveorder.nn.optics import Illumination, IlluminationTilt, PupilEdge, PupilGrid
from waveorder.nn.preprocess import brightfield_contrast
from waveorder.nn.systems import phase as phase_module
from waveorder.nn.systems.phase import ThickPhase3D, ThinPhase2D
from waveorder.testing import (
    assert_adjoint,
    assert_gradients_match_finite_differences,
    assert_operator_adjoint,
    assert_operator_norm,
    assert_operator_normal,
    assert_vmap_matches_loop,
)

WAVELENGTH = 0.532
INDEX = 1.33
NA_ILL = 0.5
NA_DET = 0.9


def _optics(
    *,
    na_ill: float | Free = NA_ILL,
    na_det: float | Free = NA_DET,
    zenith: float | Free = 0.0,
    azimuth: float | Free = 0.0,
    offset: float = 0.0,
    edge: PupilEdge = "ramp",
) -> tuple[Illumination, Detection]:
    tilt = IlluminationTilt(zenith=zenith, azimuth=azimuth)
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=na_ill, tilt=tilt)
    detection = Detection(wavelength=WAVELENGTH, numerical_aperture=na_det, z_focus_offset=offset, edge=edge)
    return illumination, detection


def _relative(actual: Tensor, expected: Tensor) -> float:
    wide = torch.complex128 if actual.is_complex() or expected.is_complex() else torch.float64
    return float((actual.to(wide) - expected.to(wide)).norm() / expected.to(wide).norm())


def _correlation(first: Tensor, second: Tensor) -> float:
    a, b = first - first.mean(), second - second.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _ball(shape: tuple[int, ...], pixel_size: tuple[float, ...], *, radius: float) -> Tensor:
    """A unit ball of ``radius`` um at the grid centre blurred by a Gaussian of two lateral pixels, peak 1, float64
    (the 3.x ``generate_sphere_target``)."""
    axes = [(torch.arange(n, dtype=torch.float64) - n // 2) * d for n, d in zip(shape, pixel_size, strict=True)]
    squared = torch.stack(torch.meshgrid(*axes, indexing="ij")).square().sum(dim=0)
    blur = 2 * min(pixel_size[-2:])
    kernel = torch.fft.fftn(torch.fft.ifftshift(torch.exp(-squared / (2 * blur**2))))
    blurred = torch.fft.ifftn(torch.fft.fftn((squared < radius**2).double()) * kernel).real.clamp(min=0)
    return blurred / blurred.max()


def _thin_scene(zenith: float, *, edge: PupilEdge = "ramp") -> tuple[ThinPhase2D, Sampling, Tensor, Tensor]:
    sampling = Sampling(shape=(7, 32, 32), pixel_size=(0.6, 0.1, 0.1))
    optics = _optics(na_ill=0.4, zenith=zenith, azimuth=0.6, edge=edge)
    system = ThinPhase2D(*optics, index_of_refraction_media=INDEX).double()
    ball = _ball((32, 32), (0.1, 0.1), radius=0.8)
    obj = torch.stack((0.3 * ball, 0.05 * torch.roll(ball, 3, -1)))[:, None]
    intensity = 10 * (1 + system.forward(obj, sampling))
    return system, sampling, obj, intensity


def test_thin_tilted_inverse_solves_the_normal_equations_per_frequency() -> None:
    """Tilt makes the thin transfer complex: Wiener solves the Tikhonov normal equations per frequency (3.x built
    its inverse from ``U S_reg Vh`` without conjugates, which is right only for a real, on-axis transfer)."""
    system, sampling, obj, intensity = _thin_scene(0.3, edge="sigmoid")
    operator = system.bind(sampling)
    impulse = torch.zeros(2, 2, 1, 32, 32, dtype=torch.float64)
    impulse[0, 0, 0, 0, 0] = impulse[1, 1, 0, 0, 0] = 1
    phase_tf, absorption_tf = (torch.fft.fft2(operator.apply(delta)[0]) for delta in impulse)  # (Z, Y, X) each
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    contrast = brightfield_contrast(intensity, per_plane=True)
    estimate = Wiener()(contrast, model, sampling).estimate[:, 0]
    # Oracle: (A^H A + weight I)^-1 A^H d per frequency on the full spectrum, A[z, f].
    matrix = torch.stack((phase_tf, absorption_tf), dim=-1).permute(1, 2, 0, 3)
    spectrum = torch.fft.fft2(contrast[0]).permute(1, 2, 0)[..., None]
    adjoint = matrix.conj().transpose(-1, -2)
    solved = torch.linalg.solve(adjoint @ matrix + weight * torch.eye(2, dtype=matrix.dtype), adjoint @ spectrum)
    oracle = torch.fft.ifft2(solved[..., 0].permute(2, 0, 1)).real
    assert _relative(estimate, oracle) <= 1e-9
    assert _relative(estimate[0], obj[0, 0]) < 0.6


def _thin_transfers(system: ThinPhase2D, sampling: Sampling) -> Tensor:
    """Return the full-spectrum transfer of each field, ``(2, Z, Y, X)``: ``[H_phase, H_absorption]``."""
    impulse = torch.zeros(2, 2, 1, *sampling.shape[1:], dtype=torch.float64)
    impulse[0, 0, 0, 0, 0] = impulse[1, 1, 0, 0, 0] = 1
    operator = system.bind(sampling)
    return torch.stack([torch.fft.fft2(operator.apply(delta)[0]) for delta in impulse])


@pytest.mark.parametrize("zenith", [0.0, 0.3], ids=["on-axis", "tilted"])
def test_per_field_wiener_is_each_fields_tikhonov_inverse(zenith) -> None:
    """``coupling="per_field"`` is 3.x's ``decomposition="per_channel"``: ``conj(H_f) . d / (||H_f||^2 + weight)`` over
    the planes for each field; with the fields' transfers orthogonal over z (on axis) it is the joint solve."""
    system, sampling, _, intensity = _thin_scene(zenith, edge="sigmoid")
    transfers = _thin_transfers(system, sampling)
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    contrast = brightfield_contrast(intensity, per_plane=True)
    estimate = Wiener(coupling="per_field")(contrast, model, sampling).estimate[:, 0]
    spectrum = torch.fft.fft2(contrast[0])
    filtered = (transfers.conj() * spectrum).sum(dim=1) / (transfers.abs().square().sum(dim=1) + weight)
    assert _relative(estimate, torch.fft.ifft2(filtered).real) <= 1e-9
    joint = Wiener()(contrast, model, sampling).estimate[:, 0]
    cross = (transfers[0].conj() * transfers[1]).sum(dim=0).abs().max()
    if zenith == 0.0:
        assert cross < 1e-12
        assert _relative(estimate, joint) <= 1e-9
    else:
        assert cross > 1e-2  # the tilt couples the fields, and the estimators differ
        assert _relative(estimate, joint) > 1e-2


def _tilted_phase_correlation(coupling: Literal["joint", "per_field"], azimuth: float) -> float:
    """#600's ``test_tilted_round_trip`` scene: a phase-only stack imaged at azimuth 1.45, reconstructed at
    ``azimuth``; the correlation with the phase in the 12.5-50% midband of the ``2 NA_det / wavelength`` cutoff."""
    n, pixel, wavelength, na_det = 128, 1.3, 0.45, 0.15
    sampling = Sampling(shape=(9, n, n), pixel_size=(25.0, pixel, pixel))

    def system(at: float) -> ThinPhase2D:
        tilt = IlluminationTilt(zenith=0.15, azimuth=at)
        illumination = Illumination(wavelength=wavelength, numerical_aperture=0.1, tilt=tilt)
        detection = Detection(numerical_aperture=na_det, edge="sigmoid")
        return ThinPhase2D(illumination, detection, index_of_refraction_media=1.0).double()

    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij")
    phase = torch.zeros(n, n, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    for _ in range(30):
        cy, cx, angle = (
            torch.rand(3, generator=generator, dtype=torch.float64) * torch.tensor([n, n, math.pi])
        ).tolist()
        u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
        v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        phase += 0.5 * torch.exp(-((u / 8.0) ** 2 + (v / 5.0) ** 2))
    data = system(1.45).forward(torch.stack((phase, torch.zeros_like(phase)))[:, None], sampling)
    model = ImagingModel(system(azimuth), Tikhonov(1e-3), Gaussian(1.0)).double()
    estimate = Wiener(coupling=coupling)(data, model, sampling).estimate[0, 0]
    frequencies = torch.fft.fftfreq(n, d=pixel, dtype=torch.float64)
    radius = torch.sqrt(frequencies[None] ** 2 + frequencies[:, None] ** 2) / (2 * na_det / wavelength)
    band = (radius > 0.125) & (radius < 0.5)
    return _correlation(*(torch.fft.ifft2(torch.fft.fft2(image) * band).real for image in (estimate, phase)))


@pytest.mark.parametrize("coupling", ["joint", "per_field"])
def test_a_tilted_phase_object_is_recovered_at_its_own_azimuth(coupling) -> None:
    """3.x's unconjugated inverse closed only at azimuth + pi (#598); both estimators must close at the true one."""
    at_truth = _tilted_phase_correlation(coupling, 1.45)
    assert at_truth > 0.95
    assert at_truth > _tilted_phase_correlation(coupling, 1.45 + math.pi) + 0.5


def _per_field_estimate(model: ImagingModel, data: Tensor) -> Tensor:
    return Wiener(coupling="per_field")(data, model, SMALL).estimate


def test_per_field_wiener_is_differentiable_and_vmaps_in_the_tilt() -> None:
    model = ImagingModel(ThinPhase2D(*_free_optics(), index_of_refraction_media=INDEX), Tikhonov(1e-2), Gaussian(1.0))
    model = model.double()
    generator = torch.Generator().manual_seed(9)
    data = torch.randn(model.system.data_shape(SMALL), generator=generator, dtype=torch.float64)
    weights = torch.randn(model.system.object_shape(SMALL), generator=generator, dtype=torch.float64)
    assert_gradients_match_finite_differences(
        lambda module: (_per_field_estimate(module, data) * weights).sum(), model, atol=1e-6
    )
    candidates = torch.tensor([-0.3, 0.0, 0.35], dtype=torch.float64)
    assert_vmap_matches_loop(
        lambda module: _per_field_estimate(module, data),
        model,
        name="system.illumination.tilt.zenith",
        values=candidates,
        atol=1e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_per_field_wiener_runs_on_cuda_as_on_the_cpu() -> None:
    system, sampling, _, intensity = _thin_scene(0.3)
    model = ImagingModel(system, Tikhonov(1e-3), Gaussian(1.0)).double()
    contrast = brightfield_contrast(intensity, per_plane=True)
    on_cpu = Wiener(coupling="per_field")(contrast, model, sampling).estimate
    on_cuda = Wiener(coupling="per_field")(contrast.cuda(), model.cuda(), sampling).estimate
    assert _relative(on_cuda.cpu(), on_cpu) <= 1e-9


# --- units: what the object fields are --------------------------------------------------------------------------------


def _slab_optics(
    *, zenith: float, azimuth: float, offset: float, na_ill: float = 0.4, na_det: float = 0.8
) -> tuple[Illumination, Detection]:
    """A binary pupil (the 3.x edge), so the thick and thin normalizations agree exactly."""
    tilt = IlluminationTilt(zenith=zenith, azimuth=azimuth)
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=na_ill, tilt=tilt)
    return illumination, Detection(numerical_aperture=na_det, z_focus_offset=offset, edge="sigmoid")


def _slab_images(thick: ThickPhase3D, thin: ThinPhase2D, sampling: Sampling) -> tuple[Tensor, Tensor]:
    """Image one random pattern per field as a one-voxel slab at the centre slice, and as the thin object.

    Returns ``(2, Z, Y, X)`` each: the stack of ``phase`` alone, then of ``absorption`` alone.
    """
    nz, ny, nx = sampling.shape
    patterns = torch.randn(2, ny, nx, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    thick_images, thin_images = [], []
    for field in range(2):
        slab = torch.zeros(2, nz, ny, nx, dtype=torch.float64)
        slab[field, nz // 2] = patterns[field]
        plane = torch.zeros(2, 1, ny, nx, dtype=torch.float64)
        plane[field, 0] = patterns[field]
        thick_images.append(thick(slab, sampling)[0])
        thin_images.append(thin(plane, sampling)[0])
    return torch.stack(thick_images), torch.stack(thin_images)


def _slab_in_thin_units(sampling: Sampling, *, index: float) -> Tensor:
    """A one-voxel slab of potential 1 um^-2 in the thin object's units, per slice, shape ``(Z, 1, 1)``.

    The on-axis conversion ``dz * wavelength / (4 pi n)`` times the 3.x Hann window over the stack, which is 1 (up to
    half a slice on an even stack) at the slab and tapers the thick response away from it.
    """
    nz, dz = sampling.shape[0], sampling.pixel_size[0]
    window = torch.hann_window(nz, periodic=False, dtype=torch.float64)
    return (window * dz * WAVELENGTH / (4 * math.pi * index))[:, None, None]


SLAB_CASES = [
    pytest.param(1.0, 0.15, (15, 20, 22), 0.0, 0.0, id="air-on-axis"),
    pytest.param(1.33, 0.3, (16, 20, 22), 0.0, 0.0, id="water-even-stack"),
    pytest.param(1.52, 0.3, (15, 21, 22), 0.2, 0.7, id="oil-tilted-with-a-focus-offset"),
]


@pytest.mark.parametrize(("index", "dz", "shape", "zenith", "offset"), SLAB_CASES)
def test_thick_slab_is_a_thin_object_of_dz_wavelength_over_4_pi_n(
    monkeypatch, index, dz, shape, zenith, offset
) -> None:
    """The unit of the thick fields: a scattering potential in um^-2, not a phase per voxel.

    A one-voxel slab of potential ``f`` images like a thin object of ``f * dz * wavelength / (4 pi n)``: radians of
    phase, log amplitude of absorption. With the Green's function divided by its on-axis ``kappa = n / wavelength``
    the two transfer functions are the same algebra, so the stacks agree to rounding for any index, z step, tilt and
    focus offset. The thin model takes the opposite z direction (3.x), so the same stack needs its tilt azimuth
    turned by pi and its focus offset negated.
    """
    sampling = Sampling(shape=shape, pixel_size=(dz, 0.12, 0.12))
    on_axis = index / WAVELENGTH
    monkeypatch.setattr(phase_module, "_greens_denominator", lambda oblique, *_: torch.full_like(oblique, on_axis))
    thick = ThickPhase3D(
        *_slab_optics(zenith=zenith, azimuth=0.4, offset=offset),
        index_of_refraction_media=index,
        absorption="independent",
    ).double()
    thin = ThinPhase2D(
        *_slab_optics(zenith=zenith, azimuth=0.4 + math.pi, offset=-offset), index_of_refraction_media=index
    ).double()
    assert thick.upsampling(sampling)[0] == 1  # the slab is one voxel of the transfer function's own grid
    thick_images, thin_images = _slab_images(thick, thin, sampling)
    expected = thin_images * _slab_in_thin_units(sampling, index=index)
    assert _relative(thick_images[0], expected[0]) < 1e-12  # phase
    assert _relative(thick_images[1], expected[1]) < 1e-12  # absorption


@pytest.mark.parametrize(("index", "dz"), [(1.0, 4.0), (1.33, 4.0), (1.33, 8.0), (1.52, 8.0)])
def test_thick_slab_matches_the_thin_model_at_low_aperture(index, dz) -> None:
    """The same relation with the transfer functions as they are, where it holds on axis.

    The thick model counts each detected direction's path ``dz / cos(theta)`` through the slab and the thin model
    does not; at a detection aperture of 0.1 that is under 1% in phase and 0.2% in absorption (measured: 0.6% to
    0.95% and 0.05% to 0.18%), and it grows to about 20% and 10% at 0.9. A wrong factor of ``n``, ``2 pi`` or
    ``dz`` would miss by far more for every case here.
    """
    sampling = Sampling(shape=(16, 32, 32), pixel_size=(dz, 1.0, 1.0))
    optics = {"zenith": 0.0, "azimuth": 0.0, "offset": 0.0, "na_ill": 0.05, "na_det": 0.1}
    thick = ThickPhase3D(*_slab_optics(**optics), index_of_refraction_media=index, absorption="independent").double()
    thin = ThinPhase2D(*_slab_optics(**optics), index_of_refraction_media=index).double()
    assert thick.upsampling(sampling) == (1, 1, 1)
    thick_images, thin_images = _slab_images(thick, thin, sampling)
    expected = thin_images * _slab_in_thin_units(sampling, index=index)
    assert _relative(thick_images[0], expected[0]) < 1.5e-2  # phase
    assert _relative(thick_images[1], expected[1]) < 5e-3  # absorption


@pytest.mark.parametrize(("edge", "zenith"), [("sigmoid", 0.0), ("ramp", 0.2)])
def test_thin_model_is_the_first_order_image_of_a_transmittance(edge, zenith) -> None:
    """The unit of the thin fields: the amplitude transmittance is ``exp(absorption + i phase)``, phase in radians.

    The reference images that transmittance source point by source point (Abbe): a plane wave at each source
    frequency shifts the object's spectrum, the defocused pupil filters it, and the intensities add with the source
    weights. The model is its first order, so the two differ by the second order: 1e-5 relative for an object of
    1e-5.
    """
    sampling = Sampling(shape=(5, 21, 23), pixel_size=(0.4, 0.15, 0.15))
    illumination, detection = _optics(na_ill=0.4, na_det=0.8, zenith=zenith, azimuth=0.4, offset=0.3, edge=edge)
    system = ThinPhase2D(illumination, detection, index_of_refraction_media=INDEX).double()
    assert system.upsampling(sampling) == (1, 1, 1)  # so the reference runs on the data's own frequency grid
    _, ny, nx = sampling.shape
    obj = 1e-5 * torch.randn(2, 1, ny, nx, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    obj = obj - obj.mean(dim=(-2, -1), keepdim=True)  # the per-plane mean normalization removes the object's mean
    phase, absorption = obj[0, 0], obj[1, 0]

    cpu = torch.device("cpu")
    grid = PupilGrid.from_sampling(sampling, device=cpu, dtype=torch.float64)
    fy, fx = lateral_frequencies(sampling, device=cpu, dtype=torch.float64)
    source = illumination.source(fy[:, None], fx[None, :], index_of_refraction_media=INDEX, spacing=grid.spacing)
    pupil = torch.view_as_complex(detection.pupil(grid))
    kappa = torch.sqrt(torch.clamp((INDEX / WAVELENGTH) ** 2 - grid.radius**2, min=0.0))
    defocus = system.plane_positions(sampling)[:, None, None]
    defocused = pupil * pupil.abs() * torch.exp(2j * math.pi * defocus * kappa)  # (Z, Y, X)
    spectrum = torch.fft.fft2(torch.exp(absorption + 1j * phase))
    intensity = torch.zeros(sampling.shape, dtype=torch.float64)
    for iy, ix in torch.nonzero(source > 1e-14).tolist():
        shifted = torch.roll(spectrum, shifts=(iy, ix), dims=(0, 1))  # the object lit by the plane wave (iy, ix)
        intensity += source[iy, ix] * torch.fft.ifft2(defocused * shifted).abs() ** 2
    contrast = intensity / intensity.mean(dim=(-2, -1), keepdim=True) - 1
    assert _relative(system(obj, sampling)[0], contrast) < 1e-4


# --- the detection takes the illumination's wavelength ----------------------------------------------------------------


@pytest.mark.parametrize("kind", [ThickPhase3D, ThinPhase2D])
def test_a_detection_without_a_wavelength_takes_the_illuminations(kind) -> None:
    illumination = Illumination(wavelength=WAVELENGTH, numerical_aperture=NA_ILL)
    detection = Detection(numerical_aperture=NA_DET)
    assert not detection.has_wavelength
    assert "wavelength=None" in repr(detection)
    with pytest.raises(ParameterError, match="Detection has no wavelength: pass Detection") as error:
        _ = detection.wavelength
    assert error.value.path == "wavelength"
    system = kind(illumination, detection, index_of_refraction_media=INDEX).double()
    assert detection.has_wavelength
    assert detection.wavelength == WAVELENGTH
    explicit = kind(*_optics(), index_of_refraction_media=INDEX).double()
    obj = torch.randn(system.object_shape(SMALL), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert torch.equal(system(obj, SMALL), explicit(obj, SMALL))
    # The detection now holds that wavelength: a system with another illumination wavelength refuses it.
    other = Illumination(wavelength=0.6, numerical_aperture=NA_ILL)
    with pytest.raises(ParameterError, match=r"detection\.wavelength = 0\.532 um must equal illumination") as error:
        kind(other, detection, index_of_refraction_media=INDEX)
    assert error.value.path == "detection.wavelength"


@pytest.mark.parametrize("kind", [ThickPhase3D, ThinPhase2D])
def test_a_wavelength_in_nm_is_refused_before_anything_is_computed(kind) -> None:
    """In nm the cut-off ``NA / wavelength`` is a thousandth of a frequency step: the pupil would hold only the zero
    frequency and a reconstruction would return zeros with status ``direct``."""
    in_nm = Illumination(wavelength=532, numerical_aperture=NA_ILL)
    system = kind(in_nm, Detection(numerical_aperture=NA_DET), index_of_refraction_media=INDEX)
    with pytest.raises(IncompatibleModelError, match=rf"^{kind.__name__}: .*wavelength is in um, got 532 "):
        system.check_sampling(SMALL)
    model = ImagingModel(system, Tikhonov(1e-3), Gaussian(1.0))
    with pytest.raises(IncompatibleModelError, match="below one frequency step"):
        Wiener()(torch.zeros(system.data_shape(SMALL)), model, SMALL)
    kind(*_optics(), index_of_refraction_media=INDEX).check_sampling(SMALL)  # the same optics in um


@pytest.mark.parametrize("kind", [ThickPhase3D, ThinPhase2D])
def test_a_pixel_size_in_nm_is_refused_before_anything_is_allocated(kind) -> None:
    """(400, 150, 150) nm written as um: the transfer function grid would be 790 times finer along y and x, which
    used to try to allocate 10 TB."""
    in_nm = Sampling(shape=(8, 32, 32), pixel_size=(400, 150, 150))
    system = kind(*_optics(), index_of_refraction_media=INDEX)
    message = (
        rf"^{kind.__name__}: a pixel size of 400 x 150 x 150 um \(z, y, x\) is \d+ x 790 x 790 times the steps that "
        r"sample these optics, .*pixel_size is in um, got 150 along y \(a value in nm is 1000 times too large\)"
    )
    with pytest.raises(IncompatibleModelError, match=message):
        system.check_sampling(in_nm)
    model = ImagingModel(system, Tikhonov(1e-3), Gaussian(1.0))
    with pytest.raises(IncompatibleModelError, match="pixel_size is in um, got 150 along y"):
        Wiener()(torch.zeros(system.data_shape(in_nm)), model, in_nm)
    with pytest.raises(IncompatibleModelError, match="pixel_size is in um, got 150 along y"):
        system(torch.zeros(system.object_shape(in_nm)), in_nm)  # bind checks the sampling first
    kind(*_optics(), index_of_refraction_media=INDEX).check_sampling(SMALL)  # the same optics on um pixels


def test_the_focus_offset_of_a_thick_object_is_what_a_likelihood_cannot_determine() -> None:
    thick = ThickPhase3D(*_optics(), index_of_refraction_media=INDEX)
    thin = ThinPhase2D(*_optics(), index_of_refraction_media=INDEX)
    assert thick.undetermined_by_likelihood() == ("detection.z_focus_offset",)
    assert thin.undetermined_by_likelihood() == ()  # the offset selects the plane in focus: the data change with it


# --- contracts --------------------------------------------------------------------------------------------------------

SMALL = Sampling(shape=(6, 12, 11), pixel_size=(0.4, 0.15, 0.15))


def _systems() -> list[object]:
    def tilted() -> tuple[Illumination, Detection]:
        return _optics(zenith=0.2, azimuth=0.6)

    return [
        pytest.param(ThickPhase3D(*tilted(), index_of_refraction_media=INDEX), id="thick-phase"),
        pytest.param(
            ThickPhase3D(*tilted(), index_of_refraction_media=INDEX, absorption="independent"),
            id="thick-two-fields",
        ),
        pytest.param(ThinPhase2D(*tilted(), index_of_refraction_media=INDEX), id="thin"),
    ]


@pytest.mark.parametrize("system", _systems())
def test_adjoint_and_norm(system) -> None:
    system = system.double()
    assert_adjoint(system, SMALL, rtol=1e-10)
    assert_operator_norm(system, SMALL)


@pytest.mark.parametrize("system", _systems())
def test_bound_operator_is_adjoint_consistent(system) -> None:
    operator = system.double().bind(SMALL)
    assert_operator_adjoint(operator, rtol=1e-10)
    assert_operator_normal(operator, rtol=1e-10)


def _free_optics() -> tuple[Illumination, Detection]:
    return _optics(
        na_ill=Free(0.47, bounds=(0.3, 0.7)),
        na_det=Free(0.93, bounds=(0.8, 1.0)),
        zenith=Free(0.13, bounds=(-0.4, 0.4)),
        azimuth=Free(0.7, bounds=(-3.2, 3.2)),
    )


@pytest.mark.parametrize("kind", ["thick", "thin"])
def test_gradients_in_aperture_and_tilt_match_finite_differences(kind) -> None:
    illumination, detection = _free_optics()
    system = (
        ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX)
        if kind == "thick"
        else ThinPhase2D(illumination, detection, index_of_refraction_media=INDEX)
    ).double()
    generator = torch.Generator().manual_seed(3)
    obj = torch.randn(system.object_shape(SMALL), generator=generator, dtype=torch.float64)
    weights = torch.randn(system.data_shape(SMALL), generator=generator, dtype=torch.float64)
    assert_gradients_match_finite_differences(
        lambda module: (module.forward(obj, SMALL) * weights).sum(), system, atol=1e-6
    )


def test_vmap_over_tilt_candidates_matches_a_loop() -> None:
    illumination, detection = _free_optics()
    model = ImagingModel(
        ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX), Tikhonov(1e-2), Gaussian(0.05)
    )
    model = model.double()
    data = torch.randn(1, *SMALL.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    candidates = torch.tensor([-0.3, -0.1, 0.0, 0.2, 0.35], dtype=torch.float64)
    assert_vmap_matches_loop(
        lambda m: Evidence()(data, m, SMALL),
        model,
        name="system.illumination.tilt.zenith",
        values=candidates,
        atol=1e-8,
    )
    thin = ThinPhase2D(*_free_optics(), index_of_refraction_media=INDEX).double()
    obj = torch.randn(thin.object_shape(SMALL), generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    assert_vmap_matches_loop(
        lambda s: s.forward(obj, SMALL), thin, name="illumination.tilt.azimuth", values=candidates, atol=1e-10
    )


def test_wiener_and_fista_recover_a_thick_phantom() -> None:
    # The thick transfer is small (|H|^2 <= ~5e-4 for a potential in um^-2), so useful weights are tiny.
    sampling = Sampling(shape=(16, 32, 32), pixel_size=(0.25, 0.1, 0.1))
    system = ThickPhase3D(*_optics(zenith=0.1, azimuth=0.4), index_of_refraction_media=INDEX).double()
    phantom = 0.05 * _ball(sampling.shape, sampling.pixel_size, radius=0.6)[None]
    noise = torch.randn(1, *sampling.shape, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    data = system.forward(phantom, sampling) + 1e-6 * noise
    sharp = ImagingModel(system, Tikhonov(1e-7), Gaussian(1.0)).double()
    assert _correlation(Wiener()(data, sharp, sampling).estimate, phantom) > 0.9
    smooth = ImagingModel(system, Tikhonov(1e-5), Gaussian(1.0)).double()
    wiener = Wiener()(data, smooth, sampling).estimate
    fista = FISTA(iterations=300)(data, smooth, sampling).estimate
    assert _relative(fista, wiener) < 1e-3


def test_wiener_and_fista_recover_a_thin_phantom() -> None:
    system, sampling, obj, intensity = _thin_scene(0.2)
    noise = torch.randn(intensity.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    data = brightfield_contrast(intensity + 1e-2 * noise, per_plane=True)
    model = ImagingModel(system, Tikhonov(1e-3), Gaussian(1.0)).double()
    wiener = Wiener()(data, model, sampling).estimate
    assert _correlation(wiener[0], obj[0]) > 0.95
    assert _correlation(wiener[1], obj[1]) > 0.95
    smooth = ImagingModel(system, Tikhonov(5e-2), Gaussian(1.0)).double()  # |H|^2 <= 4: well conditioned
    fista = FISTA(iterations=300)(data, smooth, sampling).estimate
    assert _relative(fista, Wiener()(data, smooth, sampling).estimate) < 1e-3


def test_evidence_calibrates_tilt_and_aperture_within_bounds() -> None:
    """An object drawn from the Tikhonov prior, so the evidence is the right likelihood of the optics."""
    sampling = Sampling(shape=(8, 20, 20), pixel_size=(0.4, 0.15, 0.15))
    truth = ThickPhase3D(*_optics(na_ill=0.6, zenith=0.25, azimuth=0.5), index_of_refraction_media=INDEX).double()
    generator = torch.Generator().manual_seed(7)
    spread, sigma = 0.02, 2e-5  # object and noise standard deviations; the signal's is ~9e-5
    obj = spread * torch.randn(1, *sampling.shape, generator=generator, dtype=torch.float64)
    data = truth.forward(obj, sampling) + sigma * torch.randn(
        1, *sampling.shape, generator=generator, dtype=torch.float64
    )
    illumination, detection = _optics(
        na_ill=Free(0.5, bounds=(0.35, 0.7)), zenith=Free(0.1, bounds=(0.0, 0.4)), azimuth=0.5
    )
    system = ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX)
    model = ImagingModel(system, Tikhonov(sigma**2 / spread**2), Gaussian(sigma)).double()
    result = calibrate(
        data, model, sampling, score=Evidence(), search=(GridSearch(points=5), GradientSearch(iterations=60))
    )
    zenith = float(result.parameters["system.illumination.tilt.zenith"])
    aperture = float(result.parameters["system.illumination.numerical_aperture"])
    assert abs(zenith - 0.25) < 0.02
    assert abs(aperture - 0.6) < 0.02
    assert float(system.illumination.tilt.zenith.detach()) == zenith  # calibrate leaves the estimate in the model


# --- validation -------------------------------------------------------------------------------------------------------


def test_tilt_that_moves_the_illumination_axis_out_of_the_aperture_is_rejected() -> None:
    # n sin(0.8) = 0.95 > NA_det = 0.9: the unscattered light misses the objective (darkfield, not brightfield).
    with pytest.raises(ParameterError, match=r"illumination\.tilt\.zenith"):
        ThickPhase3D(*_optics(zenith=0.8), index_of_refraction_media=INDEX)
    with pytest.raises(ParameterError, match=r"illumination\.tilt\.zenith"):
        ThickPhase3D(*_optics(zenith=Free(0.1, bounds=(-0.9, 0.2))), index_of_refraction_media=INDEX)
    with pytest.raises(ParameterError, match=r"illumination\.tilt\.zenith must have bounds"):
        IlluminationTilt(zenith=Free(0.1, tolerance=0.1))
    with pytest.raises(ParameterError, match=r"illumination\.tilt\.zenith must stay inside"):
        IlluminationTilt(zenith=2.0)


def test_apertures_beyond_the_medium_or_a_mismatched_wavelength_are_rejected() -> None:
    with pytest.raises(ParameterError, match=r"illumination\.numerical_aperture"):
        ThickPhase3D(*_optics(na_ill=Free(1.0, bounds=(0.5, 1.4))), index_of_refraction_media=INDEX)
    illumination, _ = _optics()
    with pytest.raises(ParameterError, match=r"detection\.wavelength"):
        ThickPhase3D(
            illumination, Detection(wavelength=0.6, numerical_aperture=NA_DET), index_of_refraction_media=INDEX
        )


def test_thin_rejects_an_illumination_aperture_exceeding_the_detection_one() -> None:
    with pytest.raises(ParameterError, match=r"illumination\.numerical_aperture"):
        ThinPhase2D(*_optics(na_ill=Free(0.8, bounds=(0.5, 0.95))), index_of_refraction_media=INDEX)


# wo-4.0 (85282ce) isotropic_thin_3d.calculate_transfer_function((16, 16), 1.3, z=(25, 0, -25), 0.45, 1.0,
# numerical_aperture_illumination=0.15, numerical_aperture_detection=0.15), run in its own environment (float32):
# (absorption_tf, phase_tf) at [plane, y, x]. 3.x warns and computes the source at 0.9 * 0.15.
OLD_EQUAL_APERTURE = {
    (0, 0, 1, 2): 0.951616644859314,
    (0, 2, 3, 1): 0.5379888415336609,
    (0, 1, 0, 1): 1.869504451751709,
    (1, 0, 1, 2): 0.1006699725985527,
    (1, 2, 3, 1): -0.11136256903409958,
}


def test_thin_with_equal_apertures_gives_the_wo40_transfer() -> None:
    """ops_track_phase_2d sets both apertures to 0.15: the source then takes 0.9 of the detection's, as in wo-4.0."""
    sampling = Sampling(shape=(3, 16, 16), pixel_size=(25.0, 1.3, 1.3))
    illumination = Illumination(wavelength=0.45, numerical_aperture=0.15)
    system = ThinPhase2D(
        illumination, Detection(numerical_aperture=0.15, edge="sigmoid"), index_of_refraction_media=1.0
    ).double()
    transfer = torch.view_as_complex(system.plane_transfers(sampling)[:, :, 0].contiguous())  # (Z, F, Y, X // 2 + 1)
    for (field, plane, y, x), expected in OLD_EQUAL_APERTURE.items():
        index = 1 - field  # 3.x lists absorption first, the system phase first
        assert float(transfer[plane, index, y, x].real) == pytest.approx(expected, rel=1e-5)


def test_thin_plane_count_must_match_its_positions() -> None:
    system = ThinPhase2D(*_optics(), index_of_refraction_media=INDEX, z_positions=(0.5, 0.0, -0.5))
    with pytest.raises(IncompatibleModelError, match="z_positions lists 3 planes"):
        system.check_sampling(SMALL)


def test_brightfield_contrast_normalizes_per_stack_or_per_plane() -> None:
    intensity = torch.tensor([[[[1.0, 3.0]], [[10.0, 30.0]]]])  # (1, Z=2, Y=1, X=2)
    assert torch.allclose(brightfield_contrast(intensity), intensity / 11.0 - 1)
    assert torch.allclose(brightfield_contrast(intensity, per_plane=True), torch.tensor([[[[-0.5, 0.5]]] * 2]))


def test_thick_rejects_a_focus_offset_that_wraps() -> None:
    illumination, detection = _optics(offset=4.0)
    system = ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX)
    with pytest.raises(IncompatibleModelError, match="z_focus_offset"):
        system.check_sampling(SMALL)


def test_parameter_names_follow_the_config_paths() -> None:
    system = ThinPhase2D(*_free_optics(), index_of_refraction_media=INDEX)
    names = {name for name, _ in system.named_parameters()}
    assert names == {
        "illumination.numerical_aperture",
        "illumination.tilt.zenith",
        "illumination.tilt.azimuth",
        "detection.numerical_aperture",
    }
    assert math.isclose(system.illumination.tilt.largest_zenith, 0.4)


def _thick_with(**changes: object) -> ThickPhase3D:
    """Build ``ThickPhase3D`` from the default optics with some arguments replaced (wrong types allowed)."""
    illumination, detection = _optics()
    arguments: dict[str, Any] = {
        "illumination": illumination,
        "detection": detection,
        "index_of_refraction_media": INDEX,
    } | changes
    return ThickPhase3D(**arguments)


def _thin_with(**changes: object) -> ThinPhase2D:
    """Build ``ThinPhase2D`` from the default optics with some arguments replaced (wrong types allowed)."""
    illumination, detection = _optics()
    arguments: dict[str, Any] = {
        "illumination": illumination,
        "detection": detection,
        "index_of_refraction_media": INDEX,
    } | changes
    return ThinPhase2D(**arguments)


REFUSED_ARGUMENTS = {
    # (the construction, the error's path, a fragment of its message)
    "Illumination wavelength inf": (
        lambda: Illumination(wavelength=math.inf, numerical_aperture=NA_ILL),
        "illumination.wavelength",
        "positive and finite",
    ),
    "Illumination wavelength nan": (
        lambda: Illumination(wavelength=math.nan, numerical_aperture=NA_ILL),
        "illumination.wavelength",
        "positive and finite",
    ),
    "ThickPhase3D index nan": (
        lambda: _thick_with(index_of_refraction_media=math.nan),
        "index_of_refraction_media",
        "positive and finite",
    ),
    "ThinPhase2D index zero": (
        lambda: _thin_with(index_of_refraction_media=0.0),
        "index_of_refraction_media",
        "positive and finite",
    ),
    "ThinPhase2D z_positions nan": (lambda: _thin_with(z_positions=(0.0, math.nan)), "z_positions", "finite"),
}


@pytest.mark.parametrize("case", REFUSED_ARGUMENTS)
def test_the_phase_constructors_refuse_wrong_arguments_at_their_path(case):
    build, path, fragment = REFUSED_ARGUMENTS[case]
    with pytest.raises(ParameterError, match=fragment) as caught:
        build()
    assert caught.value.path == path


def test_the_inverse_transform_of_a_2048_grid_is_scaled_once_in_single_precision() -> None:
    # torch's MKL applies 1 / N twice to a scaled complex64 2048 x 2048 transform on more than two CPU threads
    # (seen on AVX2 nodes): a thin-phase transfer function came out 2^22 too small. The helper scales by hand.
    values = torch.randn(2048, 2048, 2, generator=torch.Generator().manual_seed(0))
    single = phase_module._fft(values, dim=(-2, -1), inverse=True)  # noqa: SLF001 (the helper under test)
    double = phase_module._fft(values.double(), dim=(-2, -1), inverse=True)  # noqa: SLF001
    torch.testing.assert_close(single.double(), double, rtol=0, atol=1e-5 * double.abs().max().item())
