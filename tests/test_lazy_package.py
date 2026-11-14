"""``import waveorder`` imports nothing heavy, and its public names keep their meaning in every import order."""

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
        assert callable(wo.tiling.halo_for)
        assert callable(wo.testing.check_system)
        assert {"errors", "tiling", "testing"} <= set(wo.__all__) <= set(dir(wo))
        """
    )


@pytest.mark.parametrize(
    "first",
    [
        "import waveorder",
        "import waveorder.observe",
        "from waveorder.observe import emit",
        "import waveorder.config.build",
    ],
)
def test_observe_stays_the_function(first):
    _run(
        f"""
        import inspect
        {first}
        import waveorder
        before = waveorder.observe
        import waveorder.observe
        from waveorder import observe, Observer
        from waveorder.observe import emit, observe as function
        assert callable(observe) and not inspect.ismodule(observe), observe
        assert observe is function is waveorder.observe is before
        assert inspect.isclass(Observer) and waveorder.Observer is Observer
        assert callable(emit)
        """
    )


def test_public_names():
    assert set(waveorder.__all__) <= set(dir(waveorder))
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = waveorder.missing


@pytest.mark.parametrize("replacement", [None, print])
def test_observe_can_be_patched_and_restored(monkeypatch, replacement):
    original = waveorder.observe
    monkeypatch.setattr(waveorder, "observe", replacement)
    assert waveorder.observe is replacement
    monkeypatch.undo()
    assert waveorder.observe is original


def test_import_as_binds_the_function():
    # ``import a.b as m`` reads the attribute ``b`` of ``a``: the function, as before the package went lazy (the
    # eager ``from waveorder.observe import observe`` in 3.x-era ``__init__`` rebound the name the same way).
    _run(
        """
        import inspect
        import waveorder.observe as m
        import sys
        assert callable(m) and not inspect.ismodule(m), m
        assert m is sys.modules["waveorder.observe"].observe
        """
    )


def test_reload_keeps_the_names():
    _run(
        """
        import importlib, inspect, sys
        import waveorder, waveorder.observe
        reloaded = importlib.reload(waveorder)
        assert reloaded is sys.modules["waveorder"]
        assert callable(reloaded.observe) and not inspect.ismodule(reloaded.observe)
        assert reloaded.observe is sys.modules["waveorder.observe"].observe
        assert inspect.ismodule(reloaded.errors)
        """
    )


def test_observe_pickles():
    _run(
        """
        import pickle
        import waveorder
        assert pickle.loads(pickle.dumps(waveorder.observe)) is waveorder.observe
        """
    )
