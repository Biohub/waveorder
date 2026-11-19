"""The ``wo bm`` (``wo benchmark``) Typer commands."""

import json
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from waveorder.cli import app
from waveorder.cli.bench import app as benchmark


@pytest.mark.parametrize("name", ["bm", "benchmark"])
def test_help(name):
    result = CliRunner().invoke(app, [name, "run", "--help"])
    assert result.exit_code == 0, result.output


def test_wo40s_migrate_configs_flag_is_accepted():
    # wo-4.0's bm run migrated the cases only with --migrate-configs; every case is migrated here.
    result = CliRunner().invoke(app, ["bm", "run", "--migrate-configs", "--help"])
    assert result.exit_code == 0, result.output


def _make_run_dir(base, name, metadata, summary):
    d = base / name
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(metadata))
    (d / "summary.json").write_text(json.dumps(summary))
    return d


class TestHistory:
    def test_empty(self, tmp_path):
        result = CliRunner().invoke(benchmark, ["history", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "No benchmark runs found" in result.output

    def test_lists_runs(self, tmp_path):
        _make_run_dir(tmp_path, "2026-04-20T10-00-00_regression", {"git_hash": "abc123"}, {"case_a": {}})
        _make_run_dir(tmp_path, "2026-04-20T11-00-00_regression", {"git_hash": "def456"}, {"case_a": {}, "case_b": {}})
        result = CliRunner().invoke(benchmark, ["history", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "2026-04-20T10-00-00_regression" in result.output
        assert "2026-04-20T11-00-00_regression" in result.output

    def test_env_var_used_when_no_output_option(self, tmp_path, monkeypatch):
        _make_run_dir(tmp_path, "2026-04-20T10-00-00_regression", {"git_hash": "abc123"}, {"case_a": {}})
        monkeypatch.setenv("WAVEORDER_BENCH_OUTPUT", str(tmp_path))
        result = CliRunner().invoke(benchmark, ["history"])
        assert result.exit_code == 0
        assert "2026-04-20T10-00-00_regression" in result.output

    def test_output_option_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WAVEORDER_BENCH_OUTPUT", "/nonexistent/path")
        result = CliRunner().invoke(benchmark, ["history", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "No benchmark runs found" in result.output


def test_latest_shows_the_latest_run(tmp_path):
    summary = {
        "case_a": {
            "image_quality": {
                "midband_power": 0.01,
                "histogram": {"bin_edges": [0.0, 0.5, 1.0], "counts": [5, 10]},
            },
            "elapsed_s": 2.5,
        },
    }
    _make_run_dir(tmp_path, "2026-04-20T10-00-00_regression", {"git_hash": "abc", "git_branch": "main"}, summary)
    result = CliRunner().invoke(benchmark, ["latest", "-o", str(tmp_path)])
    assert result.exit_code == 0
    assert "2026-04-20T10-00-00_regression" in result.output
    assert "case_a" in result.output


class TestCompare:
    def test_needs_two_runs(self, tmp_path):
        result = CliRunner().invoke(benchmark, ["compare", "-o", str(tmp_path)])
        assert "Need at least 2 runs" in result.output

    def test_compares_two_latest(self, tmp_path):
        summary_a = {"case_a": {"image_quality": {"midband_power": 0.01}, "with_phantom": {"mse": 0.1, "ssim": 0.9}}}
        summary_b = {"case_a": {"image_quality": {"midband_power": 0.02}, "with_phantom": {"mse": 0.05, "ssim": 0.95}}}
        _make_run_dir(tmp_path, "2026-04-20T10-00-00_regression", {"git_hash": "a"}, summary_a)
        _make_run_dir(tmp_path, "2026-04-20T11-00-00_regression", {"git_hash": "b"}, summary_b)
        result = CliRunner().invoke(benchmark, ["compare", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "case_a" in result.output
        assert "ssim" in result.output


_SYNTHETIC = {
    "type": "synthetic",
    "phantom": {"function": "single_bead", "shape": [16, 32, 32]},
    "config": "recon.yml",
}
_HPC = {"type": "hpc", "input": "/fake/input.zarr", "position": "A/1/000000", "config": "recon.yml"}


def _write_experiment(tmp_path, cases):
    exp_yml = tmp_path / "exp.yml"
    exp_yml.write_text(yaml.dump({"name": "test_exp", "cases": cases}))
    (tmp_path / "recon.yml").write_text(yaml.dump({"phase": {"transfer_function": {}}}))
    return exp_yml


def _case(**kwargs):
    case_dir = kwargs["case_dir"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "timing.json").write_text(json.dumps({"elapsed_s": 0.1}))
    return {"image_quality": {"midband_power": 0.01}}


def _run(exp_yml, scope, out):
    return CliRunner().invoke(benchmark, ["run", "-e", str(exp_yml), "--scope", scope, "-o", str(out)])


class TestRun:
    """Mock-driven tests of the run command's orchestration."""

    def test_synthetic_scope_skips_hpc(self, tmp_path):
        exp_yml = _write_experiment(tmp_path, {"syn_case": _SYNTHETIC, "hpc_case": _HPC})
        with (
            patch("benchmarks.runner.run_synthetic_case", side_effect=_case) as mock_syn,
            patch("benchmarks.runner.run_hpc_case", side_effect=_case) as mock_hpc,
        ):
            result = _run(exp_yml, "synthetic", tmp_path)
        assert result.exit_code == 0, result.output
        assert mock_syn.call_count == 1
        assert mock_hpc.call_count == 0
        assert "skipping 1 hpc" in result.output

    def test_all_scope_runs_hpc(self, tmp_path):
        exp_yml = _write_experiment(tmp_path, {"hpc_case": _HPC})
        with patch("benchmarks.runner.run_hpc_case", side_effect=_case) as mock_hpc:
            result = _run(exp_yml, "all", tmp_path)
        assert result.exit_code == 0, result.output
        assert mock_hpc.call_count == 1

    def test_records_error_on_failure(self, tmp_path):
        exp_yml = _write_experiment(tmp_path, {"bad_case": _SYNTHETIC})
        with patch("benchmarks.runner.run_synthetic_case", side_effect=RuntimeError("boom")):
            result = _run(exp_yml, "synthetic", tmp_path)
        assert result.exit_code == 0  # run continues past case failures
        assert "FAILED" in result.output
        runs = [p for p in tmp_path.iterdir() if p.is_dir() and "test_exp" in p.name]
        assert len(runs) == 1
        summary = json.loads((runs[0] / "summary.json").read_text())
        assert "error" in summary["bad_case"]
