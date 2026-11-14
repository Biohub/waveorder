"""benchmarks.phantoms and benchmarks.simulate, against numbers wo-4.0 (85282ce) computed on the same inputs."""

import pytest
import torch
from benchmarks.phantoms import grid_beads_gaussian, single_bead
from benchmarks.simulate import (
    field_coefficients,
    normalize_pupil_coefficients,
    simulate_fluorescence_3d,
    simulate_phase_3d,
    simulate_shift_variant_fluorescence_3d,
)

_VOXELS = [(8, 16, 16), (6, 16, 20), (10, 12, 16)]


def _small_bead():
    return single_bead(
        shape=(16, 32, 32),
        pixel_sizes=(0.25, 0.1, 0.1),
        bead_radius_um=1.0,
        refractive_index_diff=0.05,
        fluorescence_intensity=1.0,
        fluorescence_background=10.0,
    )


def _small_grid():
    return grid_beads_gaussian(shape=(8, 64, 64), grid_shape=(3, 3), sigma_um=(0.15, 0.1, 0.1))


class TestPhantomsEqualWo40:
    def test_single_bead(self):
        phantom = _small_bead()
        assert phantom.phase.double().sum().item() == 82.3500059848542
        assert phantom.phase[8, 16, 16].item() == pytest.approx(0.04999999329447746, rel=1e-6)
        assert phantom.phase[8, 16, 25].item() == pytest.approx(0.03497292846441269, rel=1e-6)
        assert phantom.metadata["bead_radius_um"] == 1.0

    def test_grid_beads_gaussian(self):
        phantom = _small_grid()
        assert phantom.fluorescence.double().sum().item() == pytest.approx(85.1873723082984, rel=1e-9)
        assert phantom.metadata["centers_pix"][4] == [4, 32, 32]
        assert phantom.metadata["centers_pix"][0] == [4, 11, 11]


class TestSimulationsMatchWo40:
    """wo-4.0's ``phase_thick_3d`` / ``isotropic_fluorescent_thick_3d`` / shift-variant forward on the same grid."""

    def test_phase(self):
        data = simulate_phase_3d(_small_bead())  # 0.532 um, n 1.3, NA 0.9 / 1.2
        assert (data.double() - 1000).abs().sum().item() == pytest.approx(765.02294921875, rel=1e-5)
        expected = [1000.0299072265625, 1000.0973510742188, 999.9537353515625]
        assert [data[v].item() for v in _VOXELS] == pytest.approx(expected, rel=1e-6)

    def test_fluorescence(self):
        data = simulate_fluorescence_3d(_small_bead(), background=0)  # 0.532 um, n 1.3, NA 1.2
        assert data.double().sum().item() == pytest.approx(165487.0, rel=1e-5)
        expected = [10.350459098815918, 10.305017471313477, 10.30501651763916]
        assert [data[v].item() for v in _VOXELS] == pytest.approx(expected, rel=1e-5)

    def test_shift_variant(self):
        coefficients = {"4_2_0": 0.2, "5_1_0": 0.1, "6_0_1": 0.1}
        data = simulate_shift_variant_fluorescence_3d(_small_grid(), coefficients, n_tiles_yx=(3, 3))
        assert data.double().sum().item() == pytest.approx(85.18736840780184, rel=1e-5)
        expected = [0.08005251735448837, 0.03873225301504135, 0.0354430265724659]
        assert [data[v].item() for v in [(4, 32, 32), (4, 10, 10), (3, 53, 11)]] == pytest.approx(expected, rel=1e-5)


class TestSimulate:
    def test_phase_shape_and_contrast(self):
        phantom = _small_bead()
        data = simulate_phase_3d(phantom)
        assert data.shape == phantom.phase.shape
        assert torch.all(torch.isfinite(data))
        assert data.std() > 0

    def test_fluorescence_background_and_positivity(self):
        phantom = _small_bead()
        data = simulate_fluorescence_3d(phantom, background=5)
        assert data.shape == phantom.fluorescence.shape
        assert data.min() >= 5
        assert data.std() > 0


class TestPupilCoefficients:
    def test_string_and_tuple_keys(self):
        assert normalize_pupil_coefficients({"4_2_0": 0.2, (5, 1, 0): 0.1}) == {(4, 2, 0): 0.2, (5, 1, 0): 0.1}

    def test_bad_key(self):
        with pytest.raises(ValueError, match="j_m_n"):
            normalize_pupil_coefficients({"4_2": 0.2})

    def test_field_polynomial(self):
        coefficients = {(4, 2, 0): 0.2, (4, 0, 2): 0.2, (5, 1, 0): 0.1, (6, 0, 1): 0.0}
        assert field_coefficients(coefficients, y_norm=0.5, x_norm=-1.0) == pytest.approx({4: 0.25, 5: -0.1})
