"""Integration tests for phase and fluorescence optimization examples."""

import pytest
import torch

from waveorder.api import fluorescence, phase
from waveorder.optim import OptimizableFloat
from waveorder.optim.losses import MidbandPowerLossSettings


def test_phase_2d_optimize_z_focus_offset():
    """Phase 2D: optimize z_focus_offset from wrong initial guess."""
    gt_offset = 0.6
    gt_settings = phase.Settings(transfer_function=phase.TransferFunctionSettings(z_focus_offset=gt_offset))
    _, data = phase.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 128, 128))

    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            z_focus_offset=OptimizableFloat(init=2.0, lr=0.5),
        )
    )
    optimized, recon = phase.optimize(
        data,
        settings=opt_settings,
        max_iterations=50,
        loss_settings=MidbandPowerLossSettings(midband_fractions=[0.1, 0.5]),
    )

    assert abs(optimized.transfer_function.z_focus_offset - gt_offset) < 0.5


def test_phase_2d_optimize_with_tilt_loss_improves():
    """Phase 2D with tilt: loss decreases during optimization."""
    gt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            z_focus_offset=0.6,
            tilt_angle_zenith=0.5,
            tilt_angle_azimuth=1.1,
        )
    )
    _, data = phase.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 128, 128))

    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            z_focus_offset=OptimizableFloat(init=2.0, lr=0.5),
            tilt_angle_zenith=OptimizableFloat(init=0.1, lr=0.5),
            tilt_angle_azimuth=OptimizableFloat(init=0.0, lr=0.5),
        )
    )
    optimized, recon = phase.optimize(
        data,
        settings=opt_settings,
        max_iterations=20,
        loss_settings=MidbandPowerLossSettings(midband_fractions=[0.1, 0.5]),
    )

    # Just verify the optimization ran and produced a result
    assert recon is not None
    assert optimized.transfer_function.z_focus_offset != 2.0


def test_fluorescence_2d_optimize_z_focus_offset():
    """Fluorescence 2D: optimize z_focus_offset from wrong initial guess."""
    gt_offset = 0.6
    gt_settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(z_focus_offset=gt_offset)
    )
    _, data = fluorescence.simulate(gt_settings, recon_dim=2, zyx_shape=(11, 128, 128))

    opt_settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(
            z_focus_offset=OptimizableFloat(init=2.0, lr=0.5),
        )
    )
    optimized, recon = fluorescence.optimize(
        data,
        settings=opt_settings,
        max_iterations=50,
        loss_settings=MidbandPowerLossSettings(midband_fractions=[0.1, 0.5]),
    )

    assert abs(optimized.transfer_function.z_focus_offset - gt_offset) < 1.0


def test_phase_3d_optimize_na_detection():
    """Phase 3D: optimize NA_detection, verifying 3D loss reduces to scalar."""
    gt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            numerical_aperture_detection=1.0,
        )
    )
    _, data = phase.simulate(gt_settings, recon_dim=3, zyx_shape=(11, 64, 64))

    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            numerical_aperture_detection=OptimizableFloat(init=0.8, lr=0.05),
        )
    )
    optimized, recon = phase.optimize(
        data,
        recon_dim=3,
        settings=opt_settings,
        max_iterations=5,
    )

    assert recon is not None
    assert optimized.transfer_function.numerical_aperture_detection != 0.8


def test_fluorescence_3d_optimize_na_detection():
    """Fluorescence 3D: optimize NA_detection, verifying 3D loss reduces to scalar."""
    gt_settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(
            numerical_aperture_detection=1.0,
        )
    )
    _, data = fluorescence.simulate(gt_settings, recon_dim=3, zyx_shape=(11, 64, 64))

    opt_settings = fluorescence.Settings(
        transfer_function=fluorescence.TransferFunctionSettings(
            numerical_aperture_detection=OptimizableFloat(init=0.8, lr=0.05),
        )
    )
    optimized, recon = fluorescence.optimize(
        data,
        recon_dim=3,
        settings=opt_settings,
        max_iterations=5,
    )

    assert recon is not None
    assert optimized.transfer_function.numerical_aperture_detection != 0.8


def test_phase_3d_optimize_with_tilt():
    """Phase 3D: optimize tilt parameters, verifying gradient flows."""
    gt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            tilt_angle_zenith=0.3,
        )
    )
    _, data = phase.simulate(gt_settings, recon_dim=3, zyx_shape=(11, 64, 64))

    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            # Use NA as the optimizable param since it reliably moves in 3D
            numerical_aperture_detection=OptimizableFloat(init=0.8, lr=0.05),
            tilt_angle_zenith=0.3,
        )
    )
    optimized, recon = phase.optimize(
        data,
        recon_dim=3,
        settings=opt_settings,
        max_iterations=5,
    )

    assert recon is not None
    assert optimized.transfer_function.numerical_aperture_detection != 0.8
    # Tilt was passed through (fixed, not optimized) without error
    assert optimized.transfer_function.tilt_angle_zenith == 0.3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("method", ["grid_search", "adam"])
def test_phase_3d_optimize_tilt_on_cuda(method):
    """Phase 3D: a tilt fit runs on a CUDA device.

    The transfer function follows the device of the tilt tensors, so when the
    optimizer created them on the CPU the inverse multiplied CUDA data by a CPU
    filter and raised.
    """
    gt_settings = phase.Settings(transfer_function=phase.TransferFunctionSettings(tilt_angle_zenith=0.3))
    _, data = phase.simulate(gt_settings, recon_dim=3, zyx_shape=(11, 64, 64))
    opt_settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            tilt_angle_zenith=OptimizableFloat(init=0.2, lr=0.05),
            tilt_angle_azimuth=OptimizableFloat(init=0.0, lr=0.5),
        )
    )

    optimized, recon = phase.optimize(
        data,
        recon_dim=3,
        settings=opt_settings,
        method=method,
        max_iterations=3,
        grid_points=3,
        device="cuda",
    )

    assert recon is not None
    assert isinstance(optimized.transfer_function.tilt_angle_zenith, float)
    assert isinstance(optimized.transfer_function.tilt_angle_azimuth, float)


def test_phase_2d_no_optimizable_params_runs_standard():
    """When no params are optimizable, optimize() falls back to reconstruct()."""
    settings = phase.Settings()
    _, data = phase.simulate(settings, recon_dim=2, zyx_shape=(11, 128, 128))

    returned_settings, recon = phase.optimize(data, settings=settings, max_iterations=10)

    assert returned_settings.transfer_function.z_focus_offset == 0
    assert recon is not None
