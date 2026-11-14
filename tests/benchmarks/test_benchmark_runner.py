"""benchmarks.runner and benchmarks.config."""

import numpy as np
import pytest
import torch
import yaml
from benchmarks.config import ExperimentConfig, PhantomConfig, RecoveryConfig, load_experiment, resolve_recon_config
from benchmarks.runner import (
    MIGRATED_CONFIG_FILENAME,
    _build_phantom,
    _read_zarr,
    _wo_rec_argv,
    migrate_config,
    recovery_score,
    run_synthetic_case,
)
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


class TestPhantomConfig:
    def test_valid(self):
        pc = PhantomConfig(function="single_bead", shape=(16, 32, 32))
        assert pc.function == "single_bead"
        assert pc.shape == (16, 32, 32)

    def test_extra_kwargs_allowed(self):
        pc = PhantomConfig(function="single_bead", shape=(16, 32, 32), bead_radius_um=2.0, refractive_index_diff=0.05)
        assert pc.model_dump()["bead_radius_um"] == 2.0

    def test_invalid_function_rejected(self):
        with pytest.raises(ValidationError, match="function"):
            PhantomConfig.model_validate({"function": "random_beads"})


class TestExperimentConfig:
    def test_valid(self):
        exp = ExperimentConfig(
            name="test",
            cases={"case_a": {"type": "synthetic", "phantom": {"function": "single_bead", "shape": [16, 32, 32]}}},
        )
        assert exp.name == "test"
        assert exp.cases["case_a"].phantom == PhantomConfig(function="single_bead", shape=(16, 32, 32))

    def test_base_phantom_inherited(self):
        exp = ExperimentConfig(
            name="test",
            base_phantom={"function": "single_bead", "shape": [16, 32, 32]},
            cases={"case_a": {"type": "synthetic"}, "case_b": {"type": "synthetic"}},
        )
        assert exp.cases["case_a"].phantom == exp.cases["case_b"].phantom == exp.base_phantom

    def test_case_phantom_not_overwritten_by_base(self):
        exp = ExperimentConfig(
            name="test",
            base_phantom={"function": "single_bead", "shape": [16, 32, 32]},
            cases={
                "custom": {"type": "synthetic", "phantom": {"function": "grid_beads_gaussian", "shape": [8, 16, 16]}}
            },
        )
        assert exp.cases["custom"].phantom == PhantomConfig(function="grid_beads_gaussian", shape=(8, 16, 16))

    def test_base_config_inherited(self):
        exp = ExperimentConfig(
            name="test",
            base_config="configs/phase.yml",
            cases={"case_a": {"type": "synthetic", "phantom": {"function": "single_bead"}}},
        )
        assert exp.cases["case_a"].config == "configs/phase.yml"

    def test_tile_fields(self):
        exp = ExperimentConfig(
            name="test", cases={"tiled": {"type": "hpc", "tile": [63, 320, 320], "tile_overlap": [0, 32, 32]}}
        )
        assert exp.cases["tiled"].tile == (63, 320, 320)
        assert exp.cases["tiled"].tile_overlap == (0, 32, 32)

    def test_recovery_drops_options_without_a_counterpart(self):
        with pytest.raises(ValidationError, match="lr_schedule"):
            RecoveryConfig.model_validate({"lr_schedule": "cosine"})


class TestLoadExperiment:
    def test_from_yaml(self, tmp_path):
        exp_dict = {
            "name": "test",
            "cases": {"case_a": {"type": "synthetic", "phantom": {"function": "single_bead", "shape": [16, 32, 32]}}},
        }
        path = tmp_path / "exp.yml"
        path.write_text(yaml.dump(exp_dict))
        exp = load_experiment(path)
        assert exp.cases["case_a"].phantom == PhantomConfig(function="single_bead", shape=(16, 32, 32))

    def test_overrides(self, tmp_path):
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


class TestBuildPhantom:
    def test_from_config(self):
        phantom = _build_phantom(PhantomConfig(function="single_bead", shape=(16, 32, 32), bead_radius_um=1.0))
        assert phantom.phase.shape == (16, 32, 32)

    def test_from_dict(self):
        phantom = _build_phantom({"function": "grid_beads_gaussian", "shape": [8, 32, 32], "grid_shape": [2, 2]})
        assert phantom.fluorescence.shape == (8, 32, 32)


class TestRegressionExperiment:
    def test_cases(self, pytestconfig):
        exp = load_experiment(pytestconfig.rootpath / "benchmarks" / "experiments" / "regression.yml")
        assert exp.name == "regression"
        assert len(exp.cases) == 19
        assert "kazansky_birefringence_3d" not in exp.cases
        assert exp.cases["neuromast_tile_phase_3d"].tile == (63, 320, 320)

    def test_every_case_config_migrates(self, pytestconfig):
        experiment = pytestconfig.rootpath / "benchmarks" / "experiments"
        exp = load_experiment(experiment / "regression.yml")
        for name, case in exp.cases.items():
            result = migrate(resolve_recon_config(case, experiment))
            assert not result.rejections, (name, result.rejections)


class TestWoRecArgv:
    def test_plain(self, tmp_path):
        argv = _wo_rec_argv(tmp_path / "in.zarr/0/0/0", tmp_path / "c.yml", tmp_path / "out.zarr")
        assert argv[:2] == ["wo", "rec"]
        assert "--tile" not in argv

    def test_tiled(self, tmp_path):
        argv = _wo_rec_argv(
            tmp_path / "i", tmp_path / "c", tmp_path / "o", tile=(63, 320, 320), tile_overlap=(0, 32, 32)
        )
        assert argv[-4:] == ["--tile", "63,320,320", "--tile-overlap", "0,32,32"]


class TestMigrateConfig:
    def test_3x_is_migrated_with_wo_config_migrate(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(_FLUORESCENCE_3X))
        migrated = migrate_config(config_path)
        assert migrated == tmp_path / MIGRATED_CONFIG_FILENAME
        assert yaml.safe_load(migrated.read_text())["version"] == 4

    def test_current_config_is_used_as_is(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(_FLUORESCENCE_3X))
        current = migrate_config(config_path)
        assert migrate_config(current) == current

    def test_refusal_is_reported(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump({**_FLUORESCENCE_3X, "birefringence": {}}))
        with pytest.raises(RuntimeError, match="birefringence"):
            migrate_config(config_path)


class TestRecoveryScore:
    def test_perfect_and_zero(self):
        truth = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        assert recovery_score(truth.clone(), truth, [4, 5])["recovery_score"] == pytest.approx(1.0)
        result = recovery_score(torch.zeros_like(truth), truth, [4, 5])
        assert result["recovery_score"] == pytest.approx(0.0)
        assert result["per_mode_correlation"]["Z5"] is None
        assert set(result["per_mode_rmse"]) == {"Z4", "Z5"}


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
