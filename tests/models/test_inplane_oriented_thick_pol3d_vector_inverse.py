import pytest
import torch

from waveorder import util
from waveorder.models import inplane_oriented_thick_pol3d_vector


@pytest.fixture(scope="module")
def transfer_function():
    return inplane_oriented_thick_pol3d_vector.calculate_transfer_function(
        swing=0.1,
        scheme="5-State",
        zyx_shape=(10, 96, 96),
        yx_pixel_size=0.2,
        z_pixel_size=0.5,
        wavelength_illumination=0.532,
        z_padding=0,
        index_of_refraction_media=1.3,
        numerical_aperture_illumination=0.5,
        numerical_aperture_detection=1.2,
        fourier_oversample_factor=1,
    )


def test_inverse_filter_is_tikhonov_pseudoinverse(transfer_function):
    """The applied inverse must match the Tikhonov-regularized pseudoinverse.

    The transfer function is complex and not Hermitian-symmetric, so the
    inverse filter needs conj(U) and conj(Vh) (issue #598); without them the
    operator is not the least-squares inverse and the phase channel flips
    sign.
    """
    sfzyx_transfer_function, _, singular_system = transfer_function
    U, S, Vh = singular_system
    regularization_strength = 1e-6

    S_reg = S / (S**2 + regularization_strength)
    applied_filter = torch.einsum("sjzyx,jzyx,jfzyx->sfzyx", torch.conj(U), S_reg, torch.conj(Vh))

    generator = torch.Generator().manual_seed(0)
    Z, Y, X = sfzyx_transfer_function.shape[-3:]
    for _ in range(20):
        z, y, x = (int(torch.randint(0, dim, (1,), generator=generator)) for dim in (Z, Y, X))
        A = sfzyx_transfer_function[:, :, z, y, x]  # (s, f) forward matrix
        if torch.linalg.norm(A) < 1e-4:
            continue
        # Tikhonov pseudoinverse built independently with matrix ops:
        # (A^H A + reg I)^-1 A^H = V diag(S / (S^2 + reg)) U^H
        U_ref, S_ref, Vh_ref = torch.linalg.svd(A, full_matrices=False)
        expected = Vh_ref.mH @ torch.diag((S_ref / (S_ref**2 + regularization_strength)).cfloat()) @ U_ref.mH  # (f, s)
        applied = applied_filter[:, :, z, y, x].T  # filter is (s, f) -> transpose
        assert torch.allclose(applied, expected, atol=1e-4 * torch.linalg.norm(expected))


def test_round_trip_recovers_phantom(transfer_function):
    """Forward then inverse recovers the star phantom with positive correlation.

    The unconjugated filter returns the f00 (phase) channel with the sign
    flipped (Pearson r about -0.7); the correct filter gives r about +0.7.
    """
    sfzyx_transfer_function, intensity_to_stokes_matrix, singular_system = transfer_function

    zyx_shape = sfzyx_transfer_function.shape[-3:]
    yx_star, yx_theta, _ = util.generate_star_target(yx_shape=zyx_shape[1:], blur_px=1, margin=20)
    fzyx_object = torch.zeros((3,) + tuple(zyx_shape))
    fzyx_object[0, zyx_shape[0] // 2] = yx_star
    fzyx_object[1, zyx_shape[0] // 2] = -torch.sin(2 * yx_theta) * yx_star
    fzyx_object[2, zyx_shape[0] // 2] = -torch.cos(2 * yx_theta) * yx_star

    szyx_data = torch.real(
        inplane_oriented_thick_pol3d_vector.apply_transfer_function(
            fzyx_object, sfzyx_transfer_function, intensity_to_stokes_matrix
        )
    )

    fzyx_recon = inplane_oriented_thick_pol3d_vector.apply_inverse_transfer_function(
        szyx_data,
        singular_system,
        intensity_to_stokes_matrix,
        regularization_strength=1e-3,
    )

    for channel in range(3):
        estimate = fzyx_recon[channel].flatten().double()
        truth = fzyx_object[channel].flatten().double()
        estimate = estimate - estimate.mean()
        truth = truth - truth.mean()
        pearson_r = float(estimate @ truth / (estimate.norm() * truth.norm()))
        assert pearson_r > 0.5, f"channel {channel}: r = {pearson_r:.3f}"
