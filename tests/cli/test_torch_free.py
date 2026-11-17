"""Help, the version and ``config new`` never import torch."""

import subprocess
import sys

import pytest

_PROBE = """
import atexit, sys
atexit.register(lambda: sys.stderr.write("torch imported\\n" if "torch" in sys.modules else "torch-free\\n"))
from waveorder.cli import main
sys.argv = ["wo", *sys.argv[1:]]
main()
"""


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        (["--help"], 0),
        (["--version"], 0),
        (["config", "show", "-h"], 0),
        (["config", "new", "--contrast", "phase", "--dimension", "3"], 0),
        (["rec", "-h"], 0),
        (["simulate", "-h"], 0),
        (["calibrate-tf", "-h"], 0),
    ],
)
def test_torch_free_commands(arguments, code):
    result = subprocess.run([sys.executable, "-c", _PROBE, *arguments], capture_output=True, text=True, check=False)
    assert result.returncode == code, result.stderr
    assert result.stderr.strip().endswith("torch-free"), result.stderr
