"""Tests for z_focus_offset: 'auto' resolution."""

import pytest

from waveorder.api import phase
from waveorder.optim._types import OptimizableFloat

SIM_SETTINGS = phase.Settings(
    transfer_function=dict(
        yx_pixel_size=0.325,
        z_pixel_size=2.0,
        wavelength_illumination=0.45,
        index_of_refraction_media=1.0,
        numerical_aperture_detection=0.55,
        numerical_aperture_illumination=0.4,
        z_focus_offset=2.0,
    ),
    apply_inverse=dict(regularization_strength=1e-3),
)


def _settings_with_offset(offset):
    dumped = SIM_SETTINGS.model_dump()
    dumped["transfer_function"]["z_focus_offset"] = offset
    return phase.Settings.model_validate(dumped)


@pytest.fixture(scope="module")
def defocused_data():
    """Simulated 7-slice stack whose focus sits 2 slices from center."""
    _, data = phase.simulate(settings=SIM_SETTINGS, recon_dim=2, zyx_shape=(7, 128, 128))
    return data


def test_auto_accepted_in_settings():
    settings = _settings_with_offset("auto")
    assert settings.transfer_function.z_focus_offset == "auto"

    settings = _settings_with_offset({"init": "auto", "lr": 0.05})
    offset = settings.transfer_function.z_focus_offset
    assert isinstance(offset, OptimizableFloat)
    assert offset.initial_value == "auto"


def test_unresolved_auto_value_raises():
    with pytest.raises(ValueError, match="resolved from the data"):
        OptimizableFloat(init="auto", lr=0.05).value


def test_simulate_rejects_auto():
    with pytest.raises(ValueError, match="measured data"):
        phase.simulate(settings=_settings_with_offset("auto"), recon_dim=2, zyx_shape=(5, 32, 32))


def test_resolve_recovers_integer_offset(defocused_data):
    resolved = phase.resolve_z_focus_offset(defocused_data, _settings_with_offset("auto"))
    offset = resolved.transfer_function.z_focus_offset
    assert isinstance(offset, float)
    assert offset == 2.0


def test_resolve_keeps_learning_rate(defocused_data):
    settings = _settings_with_offset(OptimizableFloat(init="auto", lr=0.05))
    resolved = phase.resolve_z_focus_offset(defocused_data, settings)
    offset = resolved.transfer_function.z_focus_offset
    assert isinstance(offset, OptimizableFloat)
    assert offset.initial_value == 2.0
    assert offset.learning_rate == 0.05


def test_resolve_passes_through_numeric(defocused_data):
    settings = _settings_with_offset(1.5)
    resolved = phase.resolve_z_focus_offset(defocused_data, settings)
    assert resolved is settings


def test_compute_transfer_function_drops_auto_in_3d(defocused_data):
    with pytest.warns(UserWarning, match="3D reconstructions do not use a focus offset"):
        phase.compute_transfer_function(defocused_data, recon_dim=3, settings=_settings_with_offset("auto"))


def test_reconstruct_resolves_auto(defocused_data):
    resolved = phase.resolve_z_focus_offset(defocused_data, _settings_with_offset("auto"))
    fixed = phase.reconstruct(defocused_data, recon_dim=2, settings=resolved)
    auto = phase.reconstruct(defocused_data, recon_dim=2, settings=_settings_with_offset("auto"))
    assert abs(auto - fixed).max() == 0.0
