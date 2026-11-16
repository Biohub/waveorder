"""``import waveorder`` and the command line need no Qt; ``--help`` of the napari commands needs no display."""

import os
import subprocess
import sys
import textwrap


def test_importing_waveorder_imports_no_gui():
    code = """
        import sys
        import waveorder
        import waveorder.cli
        import waveorder.config
        import waveorder.models
        loaded = {name.split(".")[0] for name in sys.modules}
        assert not loaded & {"napari", "qtpy", "PyQt6", "PyQt5", "PySide6", "magicgui", "pyqtgraph"}, loaded
        """
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_interactive_help_needs_no_display():
    environment = {key: value for key, value in os.environ.items() if key not in ("DISPLAY", "QT_QPA_PLATFORM")}
    for command in ("interactive", "gui"):
        result = subprocess.run(
            [sys.executable, "-m", "waveorder.cli", command, "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        assert "reconstruction GUI" in result.stdout


def test_view_help_needs_no_display():
    environment = {key: value for key, value in os.environ.items() if key not in ("DISPLAY", "QT_QPA_PLATFORM")}
    for command in (["view"], ["v"], ["bm", "view"]):
        result = subprocess.run(
            [sys.executable, "-m", "waveorder.cli", *command, "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        assert "in napari" in result.stdout
