import numpy as np
import pytest
import torch

from waveorder._pixel_size import YXPixelSize
from waveorder.models import isotropic_thin_3d


@pytest.mark.parametrize("invert_phase_contrast", (True, False))
def test_calculate_transfer_function(invert_phase_contrast):
    Hu, Hp = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(100, 101),
        yx_pixel_size=6.5 / 40,
        z_position_list=[-1, 0, 1],
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
        invert_phase_contrast=invert_phase_contrast,
    )

    assert Hu.shape == (3, 100, 101)
    assert Hp.shape == (3, 100, 101)


def test_calculate_transfer_function_isotropic_yx_pixel_size_equivalence():
    """Anisotropic YXPixelSize with y == x reproduces the legacy scalar output."""
    common = dict(
        yx_shape=(64, 64),
        z_position_list=[-1.0, 0.0, 1.0],
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    Hu_scalar, Hp_scalar = isotropic_thin_3d.calculate_transfer_function(yx_pixel_size=0.2, **common)
    Hu_model, Hp_model = isotropic_thin_3d.calculate_transfer_function(
        yx_pixel_size=YXPixelSize.isotropic(0.2), **common
    )
    assert torch.allclose(Hu_scalar, Hu_model)
    assert torch.allclose(Hp_scalar, Hp_model)


def test_calculate_transfer_function_anisotropic_runs():
    """y != x produces finite transfer functions of the expected shape."""
    Hu, Hp = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(64, 64),
        yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
        z_position_list=[-1.0, 0.0, 1.0],
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    assert Hu.shape == (3, 64, 64)
    assert Hp.shape == (3, 64, 64)
    assert torch.isfinite(Hu).all()
    assert torch.isfinite(Hp).all()


def test_reconstruct_anisotropic_smoke():
    """Anisotropic reconstruct returns finite absorption and phase of the right shape."""
    yx_shape = (32, 32)
    z_position_list = [-1.0, 0.0, 1.0]
    zyx_data = torch.rand((len(z_position_list),) + yx_shape)
    absorption, phase = isotropic_thin_3d.reconstruct(
        zyx_data,
        yx_pixel_size=YXPixelSize(y=0.3, x=0.2),
        z_position_list=z_position_list,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )
    assert absorption.shape == yx_shape
    assert phase.shape == yx_shape
    assert np.all(np.isfinite(absorption.numpy()))
    assert np.all(np.isfinite(phase.numpy()))


def test_reconstruct():
    yx_shape = (32, 32)
    z_position_list = [-1.0, 0.0, 1.0]
    zyx_data = torch.rand((len(z_position_list),) + yx_shape)

    absorption, phase = isotropic_thin_3d.reconstruct(
        zyx_data,
        yx_pixel_size=6.5 / 40,
        z_position_list=z_position_list,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
    )

    assert absorption.shape == yx_shape
    assert phase.shape == yx_shape
    assert np.all(np.isfinite(absorption.numpy()))
    assert np.all(np.isfinite(phase.numpy()))


def test_reconstruct_apodization_rolloff_smoke():
    """Inverse-filter apodization runs and zeroes the output Nyquist content."""
    yx_shape = (32, 32)
    z_position_list = [-1.0, 0.0, 1.0]
    zyx_data = torch.rand((len(z_position_list),) + yx_shape)

    kwargs = dict(
        yx_pixel_size=0.325,
        z_position_list=z_position_list,
        wavelength_illumination=0.45,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
    )
    _, phase_hard = isotropic_thin_3d.reconstruct(zyx_data, **kwargs)
    _, phase_apod = isotropic_thin_3d.reconstruct(zyx_data, apodization_rolloff=0.25, **kwargs)

    assert phase_apod.shape == yx_shape
    assert np.all(np.isfinite(phase_apod.numpy()))

    # apodized output has (numerically) no content at the Nyquist frequency
    nyquist_row_apod = np.abs(np.fft.fft2(phase_apod.numpy())[16, :])
    nyquist_row_hard = np.abs(np.fft.fft2(phase_hard.numpy())[16, :])
    assert nyquist_row_apod.max() < 1e-3 * nyquist_row_hard.max()


def test_gram_singular_system_matches_svd():
    """The gradient-mode decomposition reconstructs identically to the SVD."""
    torch.manual_seed(0)
    zyx_data = torch.rand(4, 32, 32) + 10.0
    absorption_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(32, 32),
        yx_pixel_size=6.5 / 20,
        z_position_list=[-2, -1, 0, 1],
        wavelength_illumination=0.5,
        index_of_refraction_media=1.0,
        numerical_aperture_illumination=0.4,
        numerical_aperture_detection=0.55,
    )

    results = []
    for use_svd in (True, False):
        singular_system = isotropic_thin_3d.calculate_singular_system(absorption_tf, phase_tf, use_svd=use_svd)
        results.append(
            isotropic_thin_3d.apply_inverse_transfer_function(zyx_data, singular_system, regularization_strength=1e-3)
        )
    for svd_result, gram_result in zip(*results):
        assert torch.allclose(svd_result, gram_result, atol=1e-4)


def test_gram_singular_system_finite_gradients():
    """Gradients stay finite where singular values are degenerate or zero."""
    z_focus_offset = torch.tensor(0.0, requires_grad=True)
    z_positions = (-torch.arange(4) + 2 + z_focus_offset) * 0.25
    _, phase_yx = isotropic_thin_3d.reconstruct(
        torch.rand(4, 32, 32) + 10.0,
        yx_pixel_size=0.1,
        z_position_list=z_positions,
        wavelength_illumination=0.532,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.9,
        numerical_aperture_detection=1.2,
    )
    loss = (phase_yx**2).mean()
    loss.backward()
    assert torch.isfinite(z_focus_offset.grad)
