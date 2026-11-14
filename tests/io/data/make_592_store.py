"""Generate the #592 ``tf.zarr`` fixtures read by ``tests/io/test_tf_store_592.py``.

The bytes come from #592's own writer: ``write_tf_zarr`` in ``waveorder/calibrate/tfzarr.py`` at commit
``761d726a55198234449a309adc22a9d2587a24c2`` (``origin/feat/wo4-config-calibrate``). Its source is read with
``git show`` and executed as a module; it needs only numpy and zarr, so nothing is stubbed.

Two stores, shaped like the two ways #592's ``calibrate-tf`` called the writer:

* ``tf_592.zarr``: ``score: evidence`` on one tile, so a Laplace covariance, the shift-invariant transfer function
  ``[Z, Y, X]`` complex64, a ``score_trace`` array and the evidence provenance entries.
* ``tf_592_tiled.zarr``: ``score: sharpness`` on two tiles, so no covariance (#592 writes zeros) and no transfer
  function (tiled stores regenerate it per tile).

Run from the repository root: ``.venv/bin/python tests/io/data/make_592_store.py``.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import torch

COMMIT = "761d726a55198234449a309adc22a9d2587a24c2"
SOURCE = "waveorder/calibrate/tfzarr.py"
HERE = Path(__file__).parent

EVIDENCE = HERE / "tf_592.zarr"
EVIDENCE_NAMES = ["defocus", "astigmatism_0", "spherical", "z12"]
EVIDENCE_PARAMETERS = [[0.125, -0.0625, 0.03125, 0.015625]]
EVIDENCE_COVARIANCE = [
    [
        [4.0e-4, 1.0e-5, 0.0, 2.0e-6],
        [1.0e-5, 9.0e-4, 3.0e-6, 0.0],
        [0.0, 3.0e-6, 2.5e-4, 1.0e-6],
        [2.0e-6, 0.0, 1.0e-6, 1.6e-3],
    ]
]
EVIDENCE_TRANSFER_SHAPE = (4, 8, 8)
EVIDENCE_TILE_GRID = {"size_yx": [8, 8], "overlap_yx": [0, 0], "centers_yx": [[0.0, 0.0]]}
EVIDENCE_PROVENANCE = {
    "kind": "refined",
    "sources": ["beads.zarr/0/0/0"],
    "config_hash": "c" * 64,
    "score": "evidence",
    "time_index": 0,
    "best_score": -1234.5,
    "noise_sigma": 0.01,
    "weight": 0.001,
}

TILED = HERE / "tf_592_tiled.zarr"
TILED_NAMES = ["defocus", "coma_x", "z13"]
TILED_PARAMETERS = [[0.25, -0.125, 0.0625], [0.5, 0.0, -0.03125]]
TILED_TILE_GRID = {"size_yx": [8, 8], "overlap_yx": [2, 2], "centers_yx": [[1.5, 1.5], [1.5, 4.5]]}
TILED_PROVENANCE = {
    "kind": "refined",
    "sources": ["beads.zarr/0/0/0"],
    "config_hash": "d" * 64,
    "score": "sharpness",
    "time_index": 0,
}


def transfer_function() -> torch.Tensor:
    """Return the evidence store's complex64 ``[Z, Y, X]`` transfer function (its values are arbitrary)."""
    z, y, x = EVIDENCE_TRANSFER_SHAPE
    real = torch.arange(z * y * x, dtype=torch.float32).reshape(z, y, x) / (z * y * x)
    return torch.complex(real, -real)


def load_592_writer() -> ModuleType:
    """Execute #592's ``tfzarr.py`` from git and return the module."""
    text = subprocess.run(
        ["git", "show", f"{COMMIT}:{SOURCE}"], check=True, capture_output=True, text=True, cwd=HERE
    ).stdout
    spec = importlib.util.spec_from_loader("tfzarr_592", loader=None)
    if spec is None:
        raise RuntimeError("could not make a module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module's string annotations through sys.modules
    exec(compile(text, f"{COMMIT[:8]}:{SOURCE}", "exec"), module.__dict__)
    return module


def main() -> None:
    """Write both stores next to this script with #592's ``write_tf_zarr``."""
    tfzarr = load_592_writer()
    tfzarr.write_tf_zarr(
        EVIDENCE,
        parameters=EVIDENCE_PARAMETERS,
        parameter_names=EVIDENCE_NAMES,
        tile_grid=EVIDENCE_TILE_GRID,
        provenance=EVIDENCE_PROVENANCE,
        parameters_covariance=EVIDENCE_COVARIANCE,
        transfer_function=transfer_function(),
        extra_arrays={"score_trace": [-1000.0, -1200.0, -1234.5]},
    )
    tfzarr.write_tf_zarr(
        TILED,
        parameters=TILED_PARAMETERS,
        parameter_names=TILED_NAMES,
        tile_grid=TILED_TILE_GRID,
        provenance=TILED_PROVENANCE,
    )


if __name__ == "__main__":
    main()
