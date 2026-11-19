"""``tf.zarr`` stores in #592's layout, written from arrays for the tests of the reader and of ``load_calibration``."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import zarr
from zarr.core.common import JSON

TILE_GRID_592 = {"size_yx": [8, 8], "overlap_yx": [0, 0], "centers_yx": [[0.0, 0.0]]}
PROVENANCE_592 = {"kind": "refined", "sources": ["in.zarr/0/0/0"], "config_hash": "c" * 64}


def write_592(
    path: Path,
    *,
    names: Sequence[str],
    parameters: torch.Tensor,
    covariance: torch.Tensor | None = None,
    transfer: torch.Tensor | None = None,
    arrays: Mapping[str, torch.Tensor] | None = None,
    tile_grid: Mapping[str, JSON] = TILE_GRID_592,
    provenance: Mapping[str, JSON] = PROVENANCE_592,
) -> Path:
    """Write a store as #592's ``write_tf_zarr`` did (``waveorder/calibrate/tfzarr.py`` at 761d726); return ``path``.

    ``parameters`` is ``(tiles, len(names))`` float64, and the covariance zeros unless given, as #592 wrote it without
    a Laplace step. ``transfer`` is the complex ``(Z, Y, X)`` transfer function of a shift-invariant store, and
    ``arrays`` holds extra arrays such as ``score_trace``.
    """
    tiles, size = parameters.shape
    if covariance is None:
        covariance = torch.zeros(tiles, size, size, dtype=torch.float64)
    entries = {"parameters": parameters, "parameters_covariance": covariance, **(arrays or {})}
    if transfer is not None:
        entries["transfer_function"] = transfer
    group = zarr.open_group(str(path), mode="w")
    for name, value in entries.items():
        data = value.numpy()
        group.create_array(name, shape=data.shape, dtype=str(data.dtype), chunks=data.shape)[:] = data
    group.attrs["parameter_names"] = list(names)
    group.attrs["tile_grid"] = dict(tile_grid)
    group.attrs["provenance"] = dict(provenance)
    return path
