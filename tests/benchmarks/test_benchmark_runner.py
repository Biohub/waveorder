"""benchmarks.runner and benchmarks.config: experiment files, case inputs and outputs, and the metrics."""

import numpy as np
import pytest
import torch
import yaml
from benchmarks.config import (
    CropConfig,
    ExperimentConfig,
    PhantomConfig,
    RecoveryConfig,
    load_experiment,
    resolve_recon_config,
)
from benchmarks.metrics import comparison_metrics, compute_metrics
from benchmarks.runner import (
    MIGRATED_CONFIG_FILENAME,
    SIZE_LIMIT_BYTES,
    _cleanup_large_outputs,
    _read_zarr,
    _wo_rec_argv,
    crop_input_position,
    migrate_config,
    recovery_score,
    run_synthetic_case,
)
from benchmarks.simulate import field_coefficients, normalize_pupil_coefficients
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from pydantic import ValidationError

from waveorder.config.migrate import migrate

_FLUORESCENCE_3X = {
    "reconstruction_dimension": 3,
    "input_channel_names": ["GFP"],
    "fluorescence": {
        "transfer_function": {
            "yx_pixel_size": 0.1,
            "z_pixel_size": 0.25,
            "wavelength_emission": 0.532,
            "index_of_refraction_media": 1.3,
            "numerical_aperture_detection": 1.2,
        },
    },
}


class TestExperimentConfig:
    def test_base_phantom_inherited_unless_the_case_has_one(self):
        exp = ExperimentConfig(
            name="test",
            base_phantom={"function": "single_bead", "shape": [16, 32, 32]},
            cases={
                "plain": {"type": "synthetic"},
                "custom": {"type": "synthetic", "phantom": {"function": "grid_beads_gaussian", "shape": [8, 16, 16]}},
            },
        )
        assert exp.cases["plain"].phantom == exp.base_phantom
        assert exp.cases["custom"].phantom == PhantomConfig(function="grid_beads_gaussian", shape=(8, 16, 16))

    def test_base_config_inherited(self):
        exp = ExperimentConfig(
            name="test",
            base_config="configs/phase.yml",
            cases={"case_a": {"type": "synthetic", "phantom": {"function": "single_bead"}}},
        )
        assert exp.cases["case_a"].config == "configs/phase.yml"

    def test_overrides_reach_the_resolved_config(self, tmp_path):
        config_path = tmp_path / "base.yml"
        config_path.write_text(yaml.dump({"phase": {"apply_inverse": {"regularization_strength": 0.01}}}))
        exp_dict = {
            "name": "test",
            "base_config": str(config_path),
            "cases": {
                "case_a": {
                    "type": "synthetic",
                    "phantom": {"function": "single_bead"},
                    "overrides": {"phase.apply_inverse.regularization_strength": 0.001},
                }
            },
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp_dict))
        resolved = resolve_recon_config(load_experiment(path).cases["case_a"], tmp_path)
        assert resolved["phase"]["apply_inverse"]["regularization_strength"] == 0.001

    @pytest.mark.parametrize(
        ("model", "data", "field"),
        [
            (PhantomConfig, {"function": "random_beads"}, "function"),
            (RecoveryConfig, {"lr_schedule": "cosine"}, "lr_schedule"),
            (CropConfig, {"y": [0, 10], "bogus": 1}, "bogus"),
        ],
        ids=["unknown_phantom", "recovery_option_without_counterpart", "crop_extra_field"],
    )
    def test_invalid_entries_rejected(self, model, data, field):
        with pytest.raises(ValidationError, match=field):
            model.model_validate(data)

    def test_every_regression_case_config_migrates(self, pytestconfig):
        experiment = pytestconfig.rootpath / "benchmarks" / "experiments"
        exp = load_experiment(experiment / "regression.yml")
        for name, case in exp.cases.items():
            result = migrate(resolve_recon_config(case, experiment))
            assert not result.rejections, (name, result.rejections)


class TestMigrateConfig:
    def test_3x_is_migrated_and_a_current_config_used_as_is(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(_FLUORESCENCE_3X))
        migrated = migrate_config(config_path)
        assert migrated == tmp_path / MIGRATED_CONFIG_FILENAME
        assert yaml.safe_load(migrated.read_text())["version"] == 4
        assert migrate_config(migrated) == migrated

    def test_refusal_is_reported(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump({**_FLUORESCENCE_3X, "birefringence": {}}))
        with pytest.raises(RuntimeError, match="birefringence"):
            migrate_config(config_path)


def test_tiled_case_passes_its_tiles_to_wo_rec(tmp_path):
    argv = _wo_rec_argv(tmp_path / "i", tmp_path / "c", tmp_path / "o", tile=(63, 320, 320), tile_overlap=(0, 32, 32))
    assert argv[:2] == ["wo", "rec"]
    assert argv[-4:] == ["--tile", "63,320,320", "--tile-overlap", "0,32,32"]


def _write_source_zarr(path, shape):
    ds = open_ome_zarr(path, layout="hcs", mode="w", channel_names=["GFP", "BF"])
    pos = ds.create_position("A", "1", "029029")
    pos.create_zeros(
        "0",
        (1, *shape),
        dtype=np.float32,
        transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 2.0, 0.65, 0.65])],
    )
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    pos["0"][0] = data
    ds.close()
    return data


def _cropped(path):
    dst = open_ome_zarr(path, layout="hcs", mode="r")
    data = np.asarray(next(iter(dst.positions()))[1]["0"][0])
    dst.close()
    return data


class TestCropInputPosition:
    def test_writes_the_selected_region(self, tmp_path):
        src = tmp_path / "source.zarr"
        data = _write_source_zarr(src, shape=(2, 7, 128, 128))
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        crop = CropConfig(z=(1, 4), y=(32, 96), x=(16, 48))
        out = crop_input_position(str(src), "A/1/029029", crop=crop, case_dir=case_dir)
        assert out == case_dir / "cropped_input.zarr"
        np.testing.assert_array_equal(_cropped(out), data[:, 1:4, 32:96, 16:48])

    def test_overwrites_an_earlier_crop(self, tmp_path):
        src = tmp_path / "source.zarr"
        _write_source_zarr(src, shape=(2, 7, 128, 128))
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        crop_input_position(str(src), "A/1/029029", crop=CropConfig(y=(0, 32)), case_dir=case_dir)
        out = crop_input_position(str(src), "A/1/029029", crop=CropConfig(y=(0, 64)), case_dir=case_dir)
        assert _cropped(out).shape == (2, 7, 64, 128)


_TF_STORE = "reconstruction_transfer_function_config_v4.zarr"


@pytest.mark.parametrize(
    ("name", "size", "save_all", "kept"),
    [
        (_TF_STORE, 100, False, False),
        ("cropped_input.zarr", 100, False, False),
        ("reconstruction.zarr", 100, False, True),
        ("reconstruction.zarr", SIZE_LIMIT_BYTES + 1, False, False),
        ("simulated.zarr", 100, False, True),
        ("simulated.zarr", SIZE_LIMIT_BYTES + 1, False, False),
        (_TF_STORE, 100, True, True),
        ("reconstruction.zarr", SIZE_LIMIT_BYTES + 1, True, True),
    ],
)
def test_output_storage_cap(tmp_path, name, size, save_all, kept):
    (tmp_path / name).mkdir()
    (tmp_path / name / "0").write_bytes(b"x" * size)
    _cleanup_large_outputs(tmp_path, save_all=save_all)
    assert (tmp_path / name).exists() is kept


class TestMetrics:
    def test_identical_volumes_compare_perfectly(self):
        v = torch.randn((16, 32, 32), generator=torch.Generator().manual_seed(0))
        m = comparison_metrics(v, v)
        assert m["mse"] == 0.0
        assert m["ssim"] == pytest.approx(1.0, abs=1e-5)
        assert m["correlation"] == pytest.approx(1.0, abs=1e-5)

    def test_comparison_sections_only_when_given(self):
        v = torch.randn((16, 32, 32), generator=torch.Generator().manual_seed(0))

        def metrics(**comparisons):
            return compute_metrics(v, NA_det=1.2, wavelength=0.532, pixel_size=0.1, **comparisons)

        alone = metrics()
        assert alone["image_quality"]["midband_power"] > 0
        assert "with_reference" not in alone
        assert "with_phantom" not in alone
        assert "ssim" in metrics(reference=v)["with_reference"]
        assert "mse" in metrics(phantom=v)["with_phantom"]

    def test_recovery_score_perfect_and_zero(self):
        truth = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        assert recovery_score(truth.clone(), truth, [4, 5])["recovery_score"] == pytest.approx(1.0)
        result = recovery_score(torch.zeros_like(truth), truth, [4, 5])
        assert result["recovery_score"] == pytest.approx(0.0)
        assert result["per_mode_correlation"]["Z5"] is None
        assert set(result["per_mode_rmse"]) == {"Z4", "Z5"}


class TestPupilCoefficients:
    def test_string_and_tuple_keys(self):
        assert normalize_pupil_coefficients({"4_2_0": 0.2, (5, 1, 0): 0.1}) == {(4, 2, 0): 0.2, (5, 1, 0): 0.1}

    def test_bad_key(self):
        with pytest.raises(ValueError, match="j_m_n"):
            normalize_pupil_coefficients({"4_2": 0.2})

    def test_field_polynomial(self):
        coefficients = {(4, 2, 0): 0.2, (4, 0, 2): 0.2, (5, 1, 0): 0.1, (6, 0, 1): 0.0}
        assert field_coefficients(coefficients, y_norm=0.5, x_norm=-1.0) == pytest.approx({4: 0.25, 5: -0.1})


class TestRunSyntheticCase:
    """End to end through ``wo config migrate`` and ``wo rec``."""

    def test_phase(self, tmp_path):
        phantom_config = PhantomConfig(
            function="single_bead",
            shape=(32, 64, 64),
            pixel_sizes=(0.25, 0.1, 0.1),
            bead_radius_um=2.0,
            refractive_index_diff=0.05,
        )
        recon_config = {
            "reconstruction_dimension": 3,
            "input_channel_names": ["Brightfield"],
            "phase": {
                "transfer_function": {
                    "yx_pixel_size": 0.1,
                    "z_pixel_size": 0.25,
                    "wavelength_illumination": 0.532,
                    "index_of_refraction_media": 1.3,
                    "numerical_aperture_illumination": 0.9,
                    "numerical_aperture_detection": 1.2,
                },
            },
        }
        case_dir = tmp_path / "phase_case"
        metrics = run_synthetic_case(phantom_config, recon_config, case_dir, modality="phase")

        assert metrics["config_migrated"] is True
        assert "image_quality" in metrics
        assert metrics["with_phantom"]["correlation"] > 0.5
        for name in (
            "timing.json",
            "metrics.json",
            "phantom_config.json",
            "cli_command.sh",
            "simulated.zarr",
            "reconstruction.zarr",
            MIGRATED_CONFIG_FILENAME,
        ):
            assert (case_dir / name).exists(), name
        assert "wo rec" in (case_dir / "cli_command.sh").read_text()

    def test_fluorescence(self, tmp_path):
        phantom_config = PhantomConfig(
            function="single_bead",
            shape=(16, 32, 32),
            pixel_sizes=(0.25, 0.1, 0.1),
            bead_radius_um=1.0,
            fluorescence_intensity=1.0,
        )
        case_dir = tmp_path / "fluor_case"
        metrics = run_synthetic_case(phantom_config, _FLUORESCENCE_3X, case_dir, modality="fluorescence")
        assert metrics["with_phantom"]["correlation"] > 0.5
        reconstruction = _read_zarr(case_dir / "reconstruction.zarr")
        assert reconstruction.shape == (1, 16, 32, 32)
        assert np.isfinite(reconstruction.numpy()).all()
