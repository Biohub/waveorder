"""``import waveorder`` imports nothing heavy, and its submodules are attributes imported on first use."""

import subprocess
import sys
import textwrap

import pytest

import waveorder


def _run(code: str) -> None:
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_import_does_not_import_torch():
    _run(
        """
        import sys
        import waveorder
        import waveorder.cli
        assert "torch" not in sys.modules, "importing waveorder imported torch"
        """
    )


def test_errors_tiling_and_testing_are_attributes_in_a_fresh_interpreter():
    _run(
        """
        import sys
        import waveorder as wo
        assert issubclass(wo.errors.Rejected, wo.errors.WaveorderError)
        assert "torch" not in sys.modules, "waveorder.errors imported torch"
        assert callable(wo.tiling.reconstruct_series)
        assert callable(wo.testing.check_system)
        assert {"errors", "tiling", "testing"} <= set(wo.__all__) <= set(dir(wo))
        """
    )


def test_public_names():
    assert set(waveorder.__all__) == {"config", "errors", "io", "models", "nn", "testing", "tiling"}
    assert set(waveorder.__all__) <= set(dir(waveorder))
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = waveorder.missing
