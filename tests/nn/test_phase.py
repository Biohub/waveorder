"""Brightfield phase systems: parity with the 3.x transfer functions and inverses, and the module contracts.

The 3.x reference lives in :mod:`tests.nn.legacy_phase` and is unmodified 3.x. Parity runs the 4.x systems with the
3.x pupil edge (``Detection(edge="sigmoid")``) on grids where the 3.x crop is right, and with ``z_focus_offset = 1``
where 3.x inverted the stack; separate tests pin the gap of the default ramp edge and each deliberate difference.
"""

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from tests.nn import legacy_phase as legacy
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
    Loop,
    Sampling,
    Tikhonov,
    Wiener,
    calibrate,
)
from waveorder.nn.functional import hermitian_half, lateral_frequencies
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


def _stored_half(full: Tensor, nx: int) -> Tensor:
    """A 3.x full transfer as the 4.x operator sees it: half columns, stored real, Hermitian-projected."""
    return hermitian_half(torch.view_as_real(full[..., : nx // 2 + 1].contiguous()), nx)


def _relative(actual: Tensor, expected: Tensor) -> float:
    wide = torch.complex128 if actual.is_complex() or expected.is_complex() else torch.float64
    return float((actual.to(wide) - expected.to(wide)).norm() / expected.to(wide).norm())


def _correlation(first: Tensor, second: Tensor) -> float:
    a, b = first - first.mean(), second - second.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


# Odd and even grids, on axis and tilted, oversampled by 1, 2 or 3 along y/x and by 2 along z; an odd inverted stack.
# (Odd axes oversampled by an even factor hit the 3.x crop defect; see the crop test.)
THICK_CASES = [
    pytest.param((7, 10, 10), (0.4, 0.2, 0.2), 0.0, 0.0, False, id="even-oversampled-on-axis"),
    pytest.param((5, 9, 9), (0.6, 0.45, 0.45), -0.3, 0.8, False, id="odd-oversampled-by-3-tilted"),
    pytest.param((5, 8, 7), (0.3, 0.15, 0.15), 0.2, 0.1, False, id="nyquist-sampled-tilted"),
    pytest.param((7, 10, 10), (0.4, 0.2, 0.2), -0.3, 0.8, False, id="even-tilted"),
    pytest.param((4, 8, 8), (1.0, 0.15, 0.15), 0.1, 0.3, False, id="z-oversampled"),
    pytest.param((7, 12, 11), (0.4, 0.15, 0.15), 0.1, 0.3, True, id="odd-inverted"),
]


def _thick_reference(system: ThickPhase3D, sampling: Sampling, **kwargs: object) -> tuple[Tensor, Tensor]:
    """Unmodified 3.x transfers for the optics of ``system`` (3.x has no focus offset)."""
    return legacy.thick_transfer(
        sampling.shape,
        sampling.pixel_size,
        wavelength=WAVELENGTH,
        index=INDEX,
        na_illumination=float(system.illumination.numerical_aperture),
        na_detection=float(system.detection.numerical_aperture),
        zenith=float(system.illumination.tilt.zenith),
        azimuth=float(system.illumination.tilt.azimuth),
        invert_phase_contrast=system.invert_phase_contrast,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


@pytest.mark.parametrize(("shape", "pixel_size", "zenith", "azimuth", "invert"), THICK_CASES)
def test_thick_transfer_with_the_sigmoid_edge_matches_unmodified_3x(shape, pixel_size, zenith, azimuth, invert) -> None:
    sampling = Sampling(shape=shape, pixel_size=pixel_size)
    # 3.x flipped the FFT-ordered planes to invert: exactly 4.x inversion with the focus one slice off (odd stacks).
    optics = _optics(zenith=zenith, azimuth=azimuth, offset=1.0 if invert else 0.0, edge="sigmoid")
    system = ThickPhase3D(
        *optics, index_of_refraction_media=INDEX, invert_phase_contrast=invert, absorption="independent"
    ).double()
    real_tf, imag_tf = _thick_reference(system, sampling)
    transfer = system.transfer(sampling)
    nx = shape[2]
    assert transfer.shape == (1, 2, *shape[:2], nx // 2 + 1, 2)
    assert _relative(transfer[0, 0], _stored_half(real_tf, nx)) <= 1e-10
    assert _relative(transfer[0, 1], _stored_half(imag_tf, nx)) <= 1e-10
    single = system.float().transfer(sampling)
    assert _relative(single[0, 0], _stored_half(real_tf, nx)) <= 1e-5
    assert _relative(single[0, 1], _stored_half(imag_tf, nx)) <= 1e-5


def test_thick_absorption_ratio_is_the_3x_effective_transfer() -> None:
    sampling = Sampling(shape=(6, 10, 10), pixel_size=(0.6, 0.3, 0.25))
    illumination, detection = _optics(zenith=0.2, azimuth=0.5, edge="sigmoid")
    system = ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX, absorption=0.3).double()
    real_tf, imag_tf = _thick_reference(system, sampling)
    assert system.fields == ("phase",)
    assert _relative(system.transfer(sampling)[0, 0], _stored_half(real_tf + 0.3 * imag_tf, 10)) <= 1e-10


def _thick_scene(edge: PupilEdge, sampling: Sampling) -> tuple[ThickPhase3D, Tensor, Tensor, Tensor]:
    """Raw 3.x intensities of a phase sphere, and the 3.x transfers that imaged them."""
    illumination, detection = _optics(zenith=0.15, azimuth=1.0, edge=edge)
    system = ThickPhase3D(illumination, detection, index_of_refraction_media=INDEX, absorption=0.2).double()
    real_tf, imag_tf = _thick_reference(system, sampling)
    phantom = 0.05 * legacy.sphere(sampling.shape, sampling.pixel_size, radius=0.6)
    noise = torch.randn(sampling.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    contrast = torch.fft.ifftn(torch.fft.fftn(phantom) * (real_tf + 0.2 * imag_tf)).real
    return system, 1e3 * (1 + contrast) + 2.0 * noise, real_tf, imag_tf


def _reconstruct_like_3x(system: ThickPhase3D, intensity: Tensor, sampling: Sampling, *, weight: float) -> Tensor:
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    return Wiener()(brightfield_contrast(intensity[None]), model, sampling).estimate[0]


def test_thick_tikhonov_reconstruction_matches_unmodified_3x() -> None:
    """Wiener with a Tikhonov prior on brightfield_contrast is the 3.x Tikhonov inverse of the raw intensities."""
    sampling = Sampling(shape=(8, 16, 15), pixel_size=(0.4, 0.15, 0.15))
    system, intensity, real_tf, imag_tf = _thick_scene("sigmoid", sampling)
    weight = 1e-6
    expected = legacy.thick_inverse(intensity, real_tf, imag_tf, absorption_ratio=0.2, regularization_strength=weight)
    assert _relative(_reconstruct_like_3x(system, intensity, sampling, weight=weight), expected) <= 1e-10


def test_default_ramp_edge_gap_to_unmodified_3x_is_pinned() -> None:
    """The default antialiased edge weights the pupil's edge ring, where 3.x's sigmoid is a hard edge. Measured at
    16 x 64 x 64 (0.4, 0.1, 0.1 um): 8.3% (phase) and 4.4% (absorption) in the transfer, 1.7% in a Tikhonov
    reconstruction of a 1 um sphere (weight 1e-6). A change to either number is a change of optics."""
    sampling = Sampling(shape=(16, 64, 64), pixel_size=(0.4, 0.1, 0.1))
    ramp = ThickPhase3D(*_optics(), index_of_refraction_media=INDEX, absorption="independent").double()
    real_tf, imag_tf = _thick_reference(ramp, sampling)
    transfer = ramp.transfer(sampling)
    assert _relative(transfer[0, 0], _stored_half(real_tf, 64)) == pytest.approx(0.0830, abs=2e-3)
    assert _relative(transfer[0, 1], _stored_half(imag_tf, 64)) == pytest.approx(0.0440, abs=2e-3)
    phantom = 0.05 * legacy.sphere(sampling.shape, sampling.pixel_size, radius=1.0)
    intensity = 1e3 * (1 + torch.fft.ifftn(torch.fft.fftn(phantom) * real_tf).real)
    expected = legacy.thick_inverse(intensity, real_tf, imag_tf, regularization_strength=1e-6)
    phase_only = ThickPhase3D(ramp.illumination, ramp.detection, index_of_refraction_media=INDEX).double()
    gap = _relative(_reconstruct_like_3x(phase_only, intensity, sampling, weight=1e-6), expected)
    assert gap == pytest.approx(0.0170, abs=1e-3)


def test_greens_function_divides_by_kz_on_the_whole_pupil() -> None:
    """With the ramp edge the Green's function is ``P exp(i 2 pi z k_z) / k_z`` on the edge ring too. 3.x's guard
    ``k_z + (1 - P)`` is exact only for a binary pupil; on the ramp it biases the phase transfer, by 5% on this coarse
    16 x 16 grid (the edge ring is a large share of the pupil) and by about 1% at 64 x 64."""
    sampling = Sampling(shape=(8, 16, 16), pixel_size=(0.4, 0.15, 0.15))
    system = ThickPhase3D(*_optics(), index_of_refraction_media=INDEX, absorption="independent").double()
    assert system.upsampling(sampling) == (1, 1, 1)
    grid = PupilGrid.from_sampling(sampling, device=torch.device("cpu"), dtype=torch.float64)
    stored = system.detection.pupil(grid)
    pupil = torch.complex(stored[..., 0], stored[..., 1])
    assert ((pupil.abs() > 0) & (pupil.abs() < 1)).any()  # the grid has edge-ring samples
    fyy, fxx = legacy.frequencies(sampling.shape[1:], 0.15, 0.15, dtype=torch.float64)
    frr = torch.sqrt(fyy**2 + fxx**2)
    medium = WAVELENGTH / INDEX
    oblique = torch.sqrt(torch.clamp(1 - medium**2 * frr**2, min=0.0)) / medium
    z = torch.fft.ifftshift((torch.arange(8, dtype=torch.float64) - 4) * 0.4)
    propagation = legacy.generate_propagation_kernel(frr, pupil.abs(), medium, z)
    defocus = torch.exp(1j * 2 * math.pi * z[:, None, None] * oblique)
    zero = torch.tensor(0.0, dtype=torch.float64)
    aperture = torch.tensor(NA_ILL, dtype=torch.float64)
    illumination = legacy.generate_tilted_pupil(fxx, fyy, aperture, WAVELENGTH, INDEX, zero, zero)

    def transfer(denominator: Tensor) -> Tensor:
        greens = -0.25j / math.pi * pupil.abs() * defocus / denominator
        return legacy.wotf_3d(illumination, pupil, propagation, greens, 0.4)[0]

    exact = transfer(torch.where(oblique > 0, oblique, torch.ones_like(oblique)))
    biased = transfer(oblique + (1 - pupil.abs()) + 1e-15)
    computed = system.transfer(sampling)[0, 0]
    assert _relative(computed, _stored_half(exact, 16)) <= 1e-12
    assert _relative(computed, _stored_half(biased, 16)) == pytest.approx(0.050, abs=5e-3)


def test_sigmoid_parity_holds_on_samples_within_the_cut_off_transition(monkeypatch) -> None:
    """On this grid's transfer function grid (oversampled 3x laterally) four samples sit within the sigmoid's
    transition at the cut-off, where 3.x's ``1 - pupil`` guard in the Green's function adds 6e-4 to ``k_z``. The
    sigmoid edge keeps that guard, so the transfer matches 3.x there too (the port matches 3.x itself to its float32
    rounding, 3.5e-7, here); dividing by ``k_z`` differs by 3e-5."""
    sampling = Sampling(shape=(5, 7, 9), pixel_size=(0.6, 0.5, 0.4))
    optics = _optics(zenith=-0.3, azimuth=0.8, edge="sigmoid")
    system = ThickPhase3D(*optics, index_of_refraction_media=INDEX).double()
    fine = sampling.upsampled(system.upsampling(sampling))
    grid = PupilGrid.from_sampling(fine, device=torch.device("cpu"), dtype=torch.float64)
    support = torch.linalg.vector_norm(system.detection.pupil(grid), dim=-1)
    assert int(((support > 1e-6) & (support < 1 - 1e-6)).sum()) == 4
    real_tf, _ = _thick_reference(system, sampling)
    expected = _stored_half(real_tf, 9)
    assert _relative(system.transfer(sampling)[0, 0], expected) <= 1e-10
    greens_denominator = phase_module._greens_denominator  # noqa: SLF001
    monkeypatch.setattr(
        phase_module, "_greens_denominator", lambda oblique, support, _: greens_denominator(oblique, support, "ramp")
    )
    assert 1e-6 < _relative(system.transfer(sampling)[0, 0], expected) < 1e-4


def test_3x_central_crop_moves_the_zero_frequency_on_odd_grids() -> None:
    """Characterizes a 3.x defect the port fixes: odd Y oversampled by 2 is cropped one frequency off."""
    sampling = Sampling(shape=(7, 9, 8), pixel_size=(0.4, 0.2, 0.2))
    system = ThickPhase3D(*_optics(edge="sigmoid"), index_of_refraction_media=INDEX).double()
    assert system.upsampling(sampling) == (1, 2, 2)
    shifted, _ = _thick_reference(system, sampling, crop="central")
    lowest, _ = _thick_reference(system, sampling, crop="lowest")
    assert _relative(shifted, lowest) > 0.1
    # The 3.x crop holds the y frequency -1 / (Y dy) at index 0, where the zero frequency belongs; 4.x is right.
    assert torch.allclose(shifted[:, 0], lowest[:, -1], atol=1e-12)
    assert _relative(system.transfer(sampling)[0, 0], _stored_half(lowest, 8)) <= 1e-10


def test_3x_invert_flip_is_a_negation_plus_one_slice_of_focus() -> None:
    """3.x flipped the FFT-ordered planes; 4.x negates them. Without the focus offset the two differ by a slice."""
    sampling = Sampling(shape=(7, 10, 10), pixel_size=(0.4, 0.15, 0.15))
    optics = _optics(edge="sigmoid")
    system = ThickPhase3D(*optics, index_of_refraction_media=INDEX, invert_phase_contrast=True).double()
    flipped, _ = _thick_reference(system, sampling)
    assert _relative(system.transfer(sampling)[0, 0], _stored_half(flipped, 10)) > 0.5


def test_on_axis_transfer_is_hermitian_and_tilted_is_not() -> None:
    """Physics on axis. Tilted and oversampled, the cropped 3.x transfer is not Hermitian: the tilted cap reaches
    past the oversampled band and aliases. The port keeps that (only its Hermitian projection acts on real data)."""
    sampling = Sampling(shape=(7, 12, 12), pixel_size=(0.4, 0.25, 0.25))

    def asymmetry(zenith: float) -> float:
        system = ThickPhase3D(*_optics(zenith=zenith, azimuth=0.8), index_of_refraction_media=INDEX).double()
        assert system.upsampling(sampling) == (1, 2, 2)
        raw = system.potential_transfers(sampling)[0]
        return _relative(raw, hermitian_half(raw, 12))

    assert asymmetry(0.0) <= 1e-12
    assert asymmetry(-0.65) >= 0.1


# --- thin -------------------------------------------------------------------------------------------------------------

THIN_CASES = [
    pytest.param(9, 9, (0.15, 0.15), 0.0, False, None, id="odd-on-axis"),
    pytest.param(10, 10, (0.2, 0.2), 0.25, True, None, id="even-oversampled-tilted-inverted"),
    pytest.param(9, 9, (0.15, 0.15), 0.25, False, (1.3, 0.2, -0.4, -1.5, 0.9), id="uneven-planes-tilted"),
]


def _thin_reference(system: ThinPhase2D, sampling: Sampling) -> tuple[Tensor, Tensor]:
    """Unmodified 3.x transfers; 3.x took explicit positions, with the focus offset added by its caller."""
    _, ny, nx = sampling.shape
    offset = float(system.detection.z_focus_offset)
    if system.z_positions is None:
        z = legacy.thin_z_positions(sampling.shape[0], sampling.pixel_size[0], offset=offset)
    else:
        z = torch.tensor(system.z_positions, dtype=torch.float64) + offset * sampling.pixel_size[0]
    return legacy.thin_transfer(
        (ny, nx),
        sampling.pixel_size[1:],
        z,
        wavelength=WAVELENGTH,
        index=INDEX,
        na_illumination=float(system.illumination.numerical_aperture),
        na_detection=float(system.detection.numerical_aperture),
        zenith=float(system.illumination.tilt.zenith),
        azimuth=float(system.illumination.tilt.azimuth),
        invert_phase_contrast=system.invert_phase_contrast,
    )


@pytest.mark.parametrize(("ny", "nx", "lateral", "zenith", "invert", "positions"), THIN_CASES)
def test_thin_transfer_with_the_sigmoid_edge_matches_unmodified_3x(ny, nx, lateral, zenith, invert, positions) -> None:
    sampling = Sampling(shape=(5, ny, nx), pixel_size=(0.5, *lateral))
    illumination, detection = _optics(zenith=zenith, azimuth=0.7, offset=0.3, edge="sigmoid")
    system = ThinPhase2D(
        illumination, detection, index_of_refraction_media=INDEX, invert_phase_contrast=invert, z_positions=positions
    ).double()
    absorption_tf, phase_tf = _thin_reference(system, sampling)
    operator = system.bind(sampling)
    assert operator.object_shape == (2, 1, ny, nx)
    assert operator.data_shape == (1, 5, ny, nx)
    transfer = system.compute_operator(sampling).inner.transfer[:, :, 0]  # (Z, F, Y, X // 2 + 1, 2)
    assert _relative(transfer[:, 0], _plane_half(phase_tf, nx)) <= 1e-10
    assert _relative(transfer[:, 1], _plane_half(absorption_tf, nx)) <= 1e-10
    single = system.float().compute_operator(sampling).inner.transfer[:, :, 0]
    assert _relative(single[:, 0], _plane_half(phase_tf, nx)) <= 1e-5


def test_thin_ramp_edge_gap_to_unmodified_3x_is_pinned() -> None:
    """Measured at 64 x 64 (0.1 um), 5 planes 0.5 um apart: 6.6% (phase) and 3.8% (absorption)."""
    sampling = Sampling(shape=(5, 64, 64), pixel_size=(0.5, 0.1, 0.1))
    system = ThinPhase2D(*_optics(), index_of_refraction_media=INDEX).double()
    absorption_tf, phase_tf = _thin_reference(system, sampling)
    transfer = system.compute_operator(sampling).inner.transfer[:, :, 0]
    assert _relative(transfer[:, 0], _plane_half(phase_tf, 64)) == pytest.approx(0.0663, abs=2e-3)
    assert _relative(transfer[:, 1], _plane_half(absorption_tf, 64)) == pytest.approx(0.0376, abs=2e-3)


def _plane_half(full: Tensor, nx: int) -> Tensor:
    """``_stored_half`` per plane: the plane index is not a frequency axis, so project over y alone."""
    return _stored_half(full[:, None], nx)[:, 0]


def _thin_scene(zenith: float, *, edge: PupilEdge = "ramp") -> tuple[ThinPhase2D, Sampling, Tensor, Tensor]:
    sampling = Sampling(shape=(7, 32, 32), pixel_size=(0.6, 0.1, 0.1))
    optics = _optics(na_ill=0.4, zenith=zenith, azimuth=0.6, edge=edge)
    system = ThinPhase2D(*optics, index_of_refraction_media=INDEX).double()
    ball = legacy.sphere((3, 32, 32), (1.0, 0.1, 0.1), radius=0.8)[1]
    obj = torch.stack((0.3 * ball, 0.05 * torch.roll(ball, 3, -1)))[:, None]
    intensity = 10 * (1 + system.forward(obj, sampling))
    return system, sampling, obj, intensity


def test_thin_tikhonov_reconstruction_matches_unmodified_3x_on_axis() -> None:
    system, sampling, _, intensity = _thin_scene(0.0, edge="sigmoid")
    absorption_tf, phase_tf = _thin_reference(system, sampling)
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    estimate = Wiener()(brightfield_contrast(intensity, per_plane=True), model, sampling).estimate
    absorption, phase = legacy.thin_inverse(
        intensity[0], legacy.thin_singular_system(absorption_tf, phase_tf), regularization_strength=weight
    )
    assert estimate.shape == (2, 1, 32, 32)
    assert _relative(estimate[0, 0], phase) <= 1e-9
    assert _relative(estimate[1, 0], absorption) <= 1e-9


def test_thin_tilted_inverse_fixes_the_3x_conjugated_filter() -> None:
    """3.x built the thin inverse from ``U S_reg Vh`` without conjugates: right for a real (on-axis) transfer, the
    conjugate filter once tilt makes it complex. Wiener solves the Tikhonov normal equations per frequency."""
    system, sampling, obj, intensity = _thin_scene(0.3, edge="sigmoid")
    absorption_tf, phase_tf = _thin_reference(system, sampling)
    weight = 1e-3
    model = ImagingModel(system, Tikhonov(weight), Gaussian(1.0)).double()
    contrast = brightfield_contrast(intensity, per_plane=True)
    estimate = Wiener()(contrast, model, sampling).estimate[:, 0]
    # Independent oracle: (A^H A + weight I)^-1 A^H d per frequency on the full spectrum, A[z, f].
    matrix = torch.stack((phase_tf, absorption_tf), dim=-1).permute(1, 2, 0, 3)
    spectrum = torch.fft.fft2(contrast[0]).permute(1, 2, 0)[..., None]
    adjoint = matrix.conj().transpose(-1, -2)
    solved = torch.linalg.solve(adjoint @ matrix + weight * torch.eye(2, dtype=matrix.dtype), adjoint @ spectrum)
    oracle = torch.fft.ifft2(solved[..., 0].permute(2, 0, 1)).real
    assert _relative(estimate, oracle) <= 1e-9
    _, phase_3x = legacy.thin_inverse(
        intensity[0], legacy.thin_singular_system(absorption_tf, phase_tf), regularization_strength=weight
    )
    assert _relative(phase_3x, estimate[0]) > 0.5
    assert _relative(estimate[0], obj[0, 0]) < 0.6 < _relative(phase_3x, obj[0, 0])


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


@pytest.mark.parametrize(("index", "dz"), [(1.0, 0.15), (1.33, 0.25), (1.52, 0.3)])
def test_radians_per_voxel_is_the_conversion_the_slab_tests_measure(index, dz) -> None:
    sampling = Sampling(shape=(15, 20, 22), pixel_size=(dz, 0.12, 0.12))
    thick = ThickPhase3D(*_slab_optics(zenith=0.0, azimuth=0.0, offset=0.0), index_of_refraction_media=index)
    factor = thick.radians_per_voxel(sampling)
    assert isinstance(factor, float)
    assert factor == pytest.approx(dz * WAVELENGTH / (4 * math.pi * index), rel=1e-15)
    # The slab's own slice, where the window is 1: the factor the two slab tests pin against the thin model.
    assert _slab_in_thin_units(sampling, index=index)[15 // 2].item() == pytest.approx(factor, rel=1e-12)
    # A potential of 8 pi^2 n delta_n / wavelength^2 is the delay 2 pi delta_n dz / wavelength of one voxel.
    delta_n = 0.01
    potential = 8 * math.pi**2 * index * delta_n / WAVELENGTH**2
    assert potential * factor == pytest.approx(2 * math.pi * delta_n * dz / WAVELENGTH, rel=1e-12)


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
    phantom = 0.05 * legacy.sphere(sampling.shape, sampling.pixel_size, radius=0.6)[None]
    noise = torch.randn(1, *sampling.shape, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    data = system.forward(phantom, sampling) + 1e-6 * noise
    sharp = ImagingModel(system, Tikhonov(1e-7), Gaussian(1.0)).double()
    assert _correlation(Wiener()(data, sharp, sampling).estimate, phantom) > 0.9
    smooth = ImagingModel(system, Tikhonov(1e-5), Gaussian(1.0)).double()
    wiener = Wiener()(data, smooth, sampling).estimate
    fista = FISTA(loop=Loop(300))(data, smooth, sampling).estimate
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
    fista = FISTA(loop=Loop(300))(data, smooth, sampling).estimate
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
    z = legacy.thin_z_positions(3, 25.0)
    absorption_tf, phase_tf = legacy.thin_transfer(
        (16, 16), (1.3, 1.3), z, wavelength=0.45, index=1.0, na_illumination=0.15, na_detection=0.15
    )
    stored = system.compute_operator(sampling).inner.transfer[:, :, 0]
    assert _relative(stored[:, 0], _plane_half(phase_tf, 16)) <= 1e-10
    assert _relative(stored[:, 1], _plane_half(absorption_tf, 16)) <= 1e-10


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
