from __future__ import annotations

import warnings
from typing import Literal, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from waveorder import optics, sampling, util
from waveorder._pixel_size import YXPixelSize
from waveorder.filter import apply_filter_bank
from waveorder.reconstruct import tikhonov_regularized_inverse_filter


def generate_test_phantom(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    wavelength_illumination: float,
    index_of_refraction_media: float,
    index_of_refraction_sample: float,
    sphere_radius: float,
) -> Tuple[Tensor, Tensor]:
    yx_pixel_size = YXPixelSize.from_value(yx_pixel_size)
    sphere, _, _ = util.generate_sphere_target(
        (3,) + yx_shape,
        yx_pixel_size,
        z_pixel_size=1.0,
        radius=sphere_radius,
        blur_size=2 * min(yx_pixel_size.y, yx_pixel_size.x),
    )
    yx_phase = (
        sphere[1] * (index_of_refraction_sample - index_of_refraction_media) * 0.1 / wavelength_illumination
    )  # phase in radians

    yx_absorption = torch.clone(yx_phase)

    return yx_absorption, yx_phase


def calculate_transfer_function(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    z_position_list: Union[list, Tensor],
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    """Calculate the transfer function for 2D phase imaging.

    Parameters
    ----------
    yx_shape : tuple[int, int]
        Shape of YX dimensions
    yx_pixel_size : float
        Pixel size in YX plane
    z_position_list : list or Tensor
        Defocus distances in micrometers
    wavelength_illumination : float
        Wavelength of illumination light
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_illumination : float or Tensor
        Illumination numerical aperture
    numerical_aperture_detection : float or Tensor
        Detection numerical aperture
    invert_phase_contrast : bool, optional
        Invert phase contrast, by default False
    tilt_angle_zenith : float or Tensor, optional
        Illumination tilt zenith angle in radians, by default 0.0
        Scalar for shared tilt; ``(B,)`` tensor for per-tile tilt
        (produces batched output).
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
        Scalar for shared tilt; ``(B,)`` tensor for per-tile tilt.
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 10000.0

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(absorption_tf, phase_tf)`` with shape ``(Z, Y, X)`` or
        ``(B, Z, Y, X)`` when batched tilt angles are provided.
    """
    # Extract float values for Nyquist computation (not in gradient chain)
    na_ill_val = float(torch.as_tensor(numerical_aperture_illumination).detach())
    na_det_val = float(torch.as_tensor(numerical_aperture_detection).detach())

    yx_pixel_size = YXPixelSize.from_value(yx_pixel_size)

    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination,
        na_ill_val,
        na_det_val,
    )
    y_factor = int(np.ceil(yx_pixel_size.y / transverse_nyquist))
    x_factor = int(np.ceil(yx_pixel_size.x / transverse_nyquist))

    (
        absorption_2d_to_3d_transfer_function,
        phase_2d_to_3d_transfer_function,
    ) = _calculate_wrap_unsafe_transfer_function(
        (
            yx_shape[0] * y_factor,
            yx_shape[1] * x_factor,
        ),
        YXPixelSize(y=yx_pixel_size.y / y_factor, x=yx_pixel_size.x / x_factor),
        z_position_list,
        wavelength_illumination,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )

    # nd_fourier_central_cuboid preserves leading batch dims
    return (
        sampling.nd_fourier_central_cuboid(absorption_2d_to_3d_transfer_function, yx_shape),
        sampling.nd_fourier_central_cuboid(phase_2d_to_3d_transfer_function, yx_shape),
    )


def _calculate_wrap_unsafe_transfer_function(
    yx_shape: Tuple[int, int],
    yx_pixel_size: float,
    z_position_list: Union[list, Tensor],
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor],
    numerical_aperture_detection: Union[float, Tensor],
    invert_phase_contrast: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    na_ill = torch.as_tensor(numerical_aperture_illumination, dtype=torch.float32)
    na_det = torch.as_tensor(numerical_aperture_detection, dtype=torch.float32)

    # Clamp illumination NA if >= detection NA (differentiable)
    clamped_ill = torch.where(
        na_ill >= na_det,
        0.9 * na_det,
        na_ill,
    )
    if (na_ill >= na_det).any():
        warnings.warn(
            "numerical_aperture_illumination is >= "
            "numerical_aperture_detection. Setting "
            "numerical_aperture_illumination to 0.9 * "
            "numerical_aperture_detection to avoid singularities."
        )

    z_positions = torch.as_tensor(z_position_list, dtype=torch.float32)
    if invert_phase_contrast:
        z_positions = -z_positions

    with torch.no_grad():
        fyy, fxx = util.generate_frequencies(yx_shape, yx_pixel_size, device=z_positions.device)
        radial_frequencies = torch.sqrt(fyy**2 + fxx**2)

    # Detect batched tilt angles
    tilt_zenith_t = torch.as_tensor(tilt_angle_zenith, dtype=torch.float32)
    tilt_azimuth_t = torch.as_tensor(tilt_angle_azimuth, dtype=torch.float32)
    batched = tilt_zenith_t.ndim >= 1 and tilt_zenith_t.shape[0] > 1

    detection_pupil = optics.generate_pupil(
        radial_frequencies,
        na_det,
        wavelength_illumination,
        steepness=pupil_steepness,
    )
    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        detection_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_positions,
    )

    # det_prop: (Z, Yos, Xos)
    det_prop = detection_pupil.unsqueeze(0) * propagation_kernel

    # Generate tilted illumination pupil
    # For batched (B,) tilt angles, reshape to (B, 1, 1) so
    # generate_tilted_pupil broadcasts against (Yos, Xos) grids
    if batched:
        tilt_angle_zenith = tilt_zenith_t[:, None, None]
        tilt_angle_azimuth = tilt_azimuth_t[:, None, None]

    illumination_pupil = optics.generate_tilted_pupil(
        fxx,
        fyy,
        clamped_ill,
        wavelength_illumination,
        index_of_refraction_media,
        tilt_angle_zenith,
        tilt_angle_azimuth,
    )  # (Yos, Xos) or (B, Yos, Xos)

    if not batched:
        # Unbatched WOTF: ill (Yos, Xos) broadcasts against det_prop (Z, Yos, Xos)
        return optics.compute_weak_object_transfer_function_2d(illumination_pupil, det_prop)

    # Batched WOTF: ill (B, 1, Yos, Xos) broadcasts against
    # det_prop (1, Z, Yos, Xos) -> (B, Z, Yos, Xos)
    return optics.compute_weak_object_transfer_function_2d(illumination_pupil[:, None], det_prop[None])


def calculate_singular_system(
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
    use_svd: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates the singular system of the absorption and phase transfer
    functions.

    Parameters
    ----------
    absorption_2d_to_3d_transfer_function : Tensor
        Transfer function for absorption, shape ``(Z, Vy, Vx)`` or
        ``(B, Z, Vy, Vx)``
    phase_2d_to_3d_transfer_function : Tensor
        Transfer function for phase, same shape as absorption TF
    use_svd : bool
        If True (default), use torch.linalg.svd. Set to False for an
        equivalent closed-form decomposition of the 2x2 Gram matrix with
        epsilon-smoothed branch points, which supports gradient
        backpropagation on all devices

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        - U : ``(2, 2, Vy, Vx)`` or ``(B, 2, 2, Vy, Vx)``
        - S : ``(2, Vy, Vx)`` or ``(B, 2, Vy, Vx)``
        - Vh : ``(2, Z, Vy, Vx)`` or ``(B, 2, Z, Vy, Vx)``
    """
    batched = absorption_2d_to_3d_transfer_function.ndim == 4
    if not batched:
        absorption_2d_to_3d_transfer_function = absorption_2d_to_3d_transfer_function.unsqueeze(0)
        phase_2d_to_3d_transfer_function = phase_2d_to_3d_transfer_function.unsqueeze(0)

    # sfYX shape: (B, s=2, Z, Vy, Vx)
    sfYX = torch.stack(
        (
            absorption_2d_to_3d_transfer_function,
            phase_2d_to_3d_transfer_function,
        ),
        dim=1,
    )
    B, s, Z, Vy, Vx = sfYX.shape

    if use_svd:
        # Full SVD over the (s=2, Z) matrix at each spatial frequency.
        # Note: torch.linalg.svd backward produces non-finite gradients
        # where singular values are degenerate (e.g. both zero beyond the
        # band limit) and fails outright with complex tensors on some GPU
        # types (svd_backward: e^{i phi} error). Use use_svd=False when
        # gradients are needed.
        BVyVxsZ = sfYX.permute(0, 3, 4, 1, 2)
        Up, Sp, Vhp = torch.linalg.svd(BVyVxsZ, full_matrices=False)
        U = Up.permute(0, 3, 4, 1, 2)  # (B, s, s, Vy, Vx)
        S = Sp.permute(0, 3, 1, 2)  # (B, s, Vy, Vx)
        Vh = Vhp.permute(0, 3, 4, 1, 2)  # (B, s, Z, Vy, Vx)
    else:
        # Closed-form singular system from the 2x2 Gram matrix G = H H^†.
        # This is the same decomposition the SVD branch computes (up to
        # the phase gauge, which the reconstruction is invariant to), but
        # with epsilon-smoothed branch points so gradients stay finite
        # where the singular values are degenerate or zero, where
        # torch.linalg.svd backward does not. Works on all devices.
        absorption_row = sfYX[:, 0]  # (B, Z, Vy, Vx)
        phase_row = sfYX[:, 1]
        # G = [[a, b], [conj(b), c]] at each spatial frequency. The Gram
        # entries square the transfer function, so accumulate and solve the
        # (Z-free) eigenproblem in double precision to keep the
        # decomposition tight in float32 overall.
        a = (absorption_row.abs() ** 2).sum(dim=1, dtype=torch.float64)  # (B, Vy, Vx)
        c = (phase_row.abs() ** 2).sum(dim=1, dtype=torch.float64)
        b = (absorption_row * phase_row.conj()).sum(dim=1, dtype=torch.complex128)

        mean = (a + c) / 2
        # Smooth the eigenvalue gap with a relative epsilon (so healthy
        # frequencies are perturbed by <1e-9 relative) plus a tiny absolute
        # floor that keeps sqrt backward finite where H is exactly zero.
        tiny = 1e-30
        half_gap = torch.sqrt(((a - c) / 2) ** 2 + b.abs() ** 2 + (1e-9 * mean) ** 2 + tiny)
        lam1 = mean + half_gap
        lam2 = torch.clamp(mean - half_gap, min=0.0)
        S = torch.stack((torch.sqrt(lam1 + tiny), torch.sqrt(lam2 + tiny)), dim=1).to(
            absorption_row.real.dtype
        )  # (B, 2, Vy, Vx)

        # Leading eigenvector of G, taking whichever closed form is better
        # conditioned at each frequency: [b, lam1 - a] or [lam1 - c, conj(b)].
        # Where H is ~zero both forms vanish and the normalized vector decays
        # to zero along with S, so no explicit degenerate branch is needed.
        u_x, u_y = b, (lam1 - a).to(b.dtype)
        w_x, w_y = (lam1 - c).to(b.dtype), b.conj()
        use_u = (u_x.abs() ** 2 + u_y.abs() ** 2) >= (w_x.abs() ** 2 + w_y.abs() ** 2)
        v1_x = torch.where(use_u, u_x, w_x)
        v1_y = torch.where(use_u, u_y, w_y)
        norm1 = torch.sqrt(v1_x.abs() ** 2 + v1_y.abs() ** 2 + tiny)
        v1_x = v1_x / norm1
        v1_y = v1_y / norm1

        v1_x = v1_x.to(sfYX.dtype)
        v1_y = v1_y.to(sfYX.dtype)

        # Columns of U: v1 and its orthonormal complement [-conj(v1_y), conj(v1_x)]
        U = torch.stack(
            (
                torch.stack((v1_x, -v1_y.conj()), dim=1),
                torch.stack((v1_y, v1_x.conj()), dim=1),
            ),
            dim=1,
        )  # (B, s, s, Vy, Vx)

        # Vh = S^-1 U^† H; rows have unit norm where S > 0 and vanish with S.
        Vh = torch.stack(
            (
                (v1_x.conj()[:, None] * absorption_row + v1_y.conj()[:, None] * phase_row) / S[:, 0, None],
                (-v1_y[:, None] * absorption_row + v1_x[:, None] * phase_row) / S[:, 1, None],
            ),
            dim=1,
        )  # (B, s, Z, Vy, Vx)

    if not batched:
        U = U.squeeze(0)
        S = S.squeeze(0)
        Vh = Vh.squeeze(0)

    return U, S, Vh


def visualize_transfer_function(
    viewer,
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> None:
    """Note: unlike other `visualize_transfer_function` calls, this transfer
    function is a mixed 3D-to-2D transfer function, so it cannot reuse
    util.add_transfer_function_to_viewer. If more 3D-to-2D transfer functions
    are added, consider refactoring.
    """
    arrays = [
        (torch.imag(absorption_2d_to_3d_transfer_function), "Im(absorb TF)"),
        (torch.real(absorption_2d_to_3d_transfer_function), "Re(absorb TF)"),
        (torch.imag(phase_2d_to_3d_transfer_function), "Im(phase TF)"),
        (torch.real(phase_2d_to_3d_transfer_function), "Re(phase TF)"),
    ]

    for array in arrays:
        lim = (0.5 * torch.max(torch.abs(array[0]))).item()
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1, 1, 1),
        )
    viewer.dims.order = (2, 0, 1)


def visualize_point_spread_function(
    viewer,
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> None:
    arrays = [
        (torch.fft.ifftn(absorption_2d_to_3d_transfer_function), "absorb PSF"),
        (torch.fft.ifftn(phase_2d_to_3d_transfer_function), "phase PSF"),
    ]

    for array in arrays:
        lim = (0.5 * torch.max(torch.abs(array[0]))).item()
        viewer.add_image(
            torch.fft.ifftshift(array[0], dim=(1, 2)).cpu().numpy(),
            name=array[1],
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=(1, 1, 1),
        )
    viewer.dims.order = (0, 1, 2)


def apply_transfer_function(
    yx_absorption: Tensor,
    yx_phase: Tensor,
    absorption_2d_to_3d_transfer_function: Tensor,
    phase_2d_to_3d_transfer_function: Tensor,
) -> Tensor:
    # Very simple simulation, consider adding noise and bkg knobs

    # simulate absorbing object
    yx_absorption_hat = torch.fft.fftn(yx_absorption)
    zyx_absorption_data_hat = yx_absorption_hat[None, ...] * absorption_2d_to_3d_transfer_function
    zyx_absorption_data = torch.real(torch.fft.ifftn(zyx_absorption_data_hat, dim=(1, 2)))

    # simulate phase object
    yx_phase_hat = torch.fft.fftn(yx_phase)
    zyx_phase_data_hat = yx_phase_hat[None, ...] * phase_2d_to_3d_transfer_function
    zyx_phase_data = torch.real(torch.fft.ifftn(zyx_phase_data_hat, dim=(1, 2)))

    # sum and add background
    data = zyx_absorption_data + zyx_phase_data
    data = data + 10  # Add a direct background
    return data


def apply_inverse_transfer_function(
    zyx_data: Tensor,
    singular_system: Tuple[Tensor, Tensor, Tensor],
    reconstruction_algorithm: Literal["Tikhonov", "TV", "RL", "RLGC"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    reg_p: float = 1e-6,  # TODO: use this parameter
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    bg_filter: bool = False,
    apodization_rolloff: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Reconstructs absorption and phase from zyx_data.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    singular_system : Tuple[Tensor, Tensor, Tensor]
        Singular system ``(U, S, Vh)``. Unbatched shapes:
        ``(2, 2, Vy, Vx)``, ``(2, Vy, Vx)``, ``(2, Z, Vy, Vx)``.
        Batched shapes have a leading ``B`` dimension.
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov". "TV" is not implemented.
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    reg_p : float, optional
        TV-specific phase regularization parameter, by default 1e-6
    TV_rho_strength : float, optional
        TV-specific rho strength, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    bg_filter : bool, optional
        Slow-varying 2D background normalization, by default False
    apodization_rolloff : float, optional
        Raised-cosine roll-off fraction applied to the inverse filter at
        the transverse Nyquist edge. Suppresses Nyquist-rate checkerboard
        artifacts in the reconstruction when the optical band limit
        exceeds the sampling Nyquist frequency. Note that apodizing the
        transfer function instead is counterproductive here: the
        Tikhonov inverse divides the window back out and amplifies noise
        where the windowed singular values cross ``sqrt(reg)``. By
        default 0.0 (no apodization, previous behavior). Must be
        between 0 and 1; if you see checkerboarding artifacts, start
        with 0.25.

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(yx_absorption, yx_phase)`` with shape ``(Y, X)`` or
        ``(B, Y, X)``.
    """
    batched = zyx_data.ndim == 4
    if not batched:
        zyx_data = zyx_data.unsqueeze(0)

    # Normalize: (B, Z, Y, X)
    zyx = util.inten_normalization(zyx_data, bg_filter=bg_filter)

    # TODO Consider refactoring with vectorial transfer function SVD
    if reconstruction_algorithm == "Tikhonov":
        U, S, Vh = singular_system
        batched_ss = S.ndim == 4  # (B, 2, Vy, Vx)

        S_reg = tikhonov_regularized_inverse_filter(S, regularization_strength, apodization_rolloff=apodization_rolloff)

        if not batched_ss:
            # Shared singular system: compute inverse filter once
            sfyx_inverse_filter = torch.einsum("sj...,j...,jf...->fs...", U, S_reg, Vh)
            results = []
            for b in range(zyx.shape[0]):
                results.append(apply_filter_bank(sfyx_inverse_filter, zyx[b]))
            output = torch.stack(results, dim=0)  # (B, 2, Y, X)
        else:
            # Per-tile singular system: compute inverse filter per tile
            results = []
            for b in range(zyx.shape[0]):
                filt_b = torch.einsum("sj...,j...,jf...->fs...", U[b], S_reg[b], Vh[b])
                results.append(apply_filter_bank(filt_b, zyx[b]))
            output = torch.stack(results, dim=0)  # (B, 2, Y, X)

    # ADMM deconvolution with anisotropic TV regularization
    elif reconstruction_algorithm == "TV":
        raise NotImplementedError

    elif reconstruction_algorithm in ("RL", "RLGC"):
        raise NotImplementedError("RL/RLGC reconstruction is only implemented for 3D fluorescence")

    absorption_yx = output[:, 0]  # (B, Y, X)
    phase_yx = output[:, 1]  # (B, Y, X)

    if not batched:
        absorption_yx = absorption_yx.squeeze(0)
        phase_yx = phase_yx.squeeze(0)

    return absorption_yx, phase_yx


def reconstruct(
    zyx_data: Tensor,
    yx_pixel_size: float,
    z_position_list: Union[list, Tensor],
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: Union[float, Tensor] = 0.9,
    numerical_aperture_detection: Union[float, Tensor] = 1.2,
    invert_phase_contrast: bool = False,
    reconstruction_algorithm: Literal["Tikhonov", "TV", "RL", "RLGC"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    reg_p: float = 1e-6,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    bg_filter: bool = False,
    tilt_angle_zenith: Union[float, Tensor] = 0.0,
    tilt_angle_azimuth: Union[float, Tensor] = 0.0,
    pupil_steepness: float = 10000.0,
    apodization_rolloff: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct 2D absorption and phase from a brightfield defocus stack.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    yx_pixel_size : float
        Pixel size in the transverse (Y, X) dimensions
    z_position_list : list or Tensor
        Defocus distances in micrometers
    wavelength_illumination : float
        Wavelength of illumination light
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_illumination : float or Tensor
        Illumination numerical aperture
    numerical_aperture_detection : float or Tensor
        Detection numerical aperture
    invert_phase_contrast : bool, optional
        Invert phase contrast, by default False
    reconstruction_algorithm : {"Tikhonov", "TV"}, optional
        By default "Tikhonov".
    regularization_strength : float, optional
        Regularization parameter, by default 1e-3
    reg_p : float, optional
        TV-specific phase regularization parameter, by default 1e-6
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    bg_filter : bool, optional
        Slow-varying 2D background normalization, by default False
    tilt_angle_zenith : float or Tensor, optional
        Illumination tilt zenith angle in radians, by default 0.0
        Scalar for shared tilt, ``(B,)`` tensor for per-tile tilt.
    tilt_angle_azimuth : float or Tensor, optional
        Illumination tilt azimuth angle in radians, by default 0.0
        Scalar for shared tilt, ``(B,)`` tensor for per-tile tilt.
    pupil_steepness : float, optional
        Sigmoid steepness for smooth pupil cutoff, by default 10000.0
    apodization_rolloff : float, optional
        Raised-cosine roll-off fraction applied to the inverse filter at
        the transverse Nyquist edge, by default 0.0 (no apodization).
        Suppresses Nyquist-rate checkerboard artifacts. Must be
        between 0 and 1; if you see checkerboarding artifacts, start
        with 0.25. See
        ``apply_inverse_transfer_function``.

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(yx_absorption, yx_phase)`` with shape ``(Y, X)`` or ``(B, Y, X)``.
    """
    absorption_tf, phase_tf = calculate_transfer_function(
        zyx_data.shape[-2:],
        yx_pixel_size,
        z_position_list,
        wavelength_illumination,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
        tilt_angle_zenith=tilt_angle_zenith,
        tilt_angle_azimuth=tilt_angle_azimuth,
        pupil_steepness=pupil_steepness,
    )
    # Both decompositions produce the same reconstruction, but
    # torch.linalg.svd backward is non-finite at degenerate singular values
    # (ubiquitous beyond the band limit) and unsupported for complex inputs
    # on some GPUs, so gradient mode uses the closed-form Gram decomposition.
    needs_grad = absorption_tf.requires_grad or phase_tf.requires_grad
    singular_system = calculate_singular_system(absorption_tf, phase_tf, use_svd=not needs_grad)
    return apply_inverse_transfer_function(
        zyx_data,
        singular_system,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        reg_p=reg_p,
        TV_rho_strength=TV_rho_strength,
        TV_iterations=TV_iterations,
        bg_filter=bg_filter,
        apodization_rolloff=apodization_rolloff,
    )
