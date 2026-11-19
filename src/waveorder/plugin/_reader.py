"""The napari reader of ``*.zarr`` and ``*.tif`` data: one layer per position of an HCS plate or a TIFF dataset.

An OME-Zarr store that is not a plate goes to napari-ome-zarr's reader.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from iohub import open_ome_zarr, read_images
from iohub.fov import BaseFOVMapping
from iohub.ngff import Plate

type LayerData = tuple[Any, dict[str, Any]]
type Reader = Callable[[str | list[str]], list[LayerData]]


def napari_get_reader(path: str | list[str]) -> Reader | None:
    """Return the reader of ``path``, or ``None`` when ``path`` is not one path."""
    if not isinstance(path, str):
        return None
    if ".zarr" in path:
        if _is_plate(path):
            return hcs_zarr_reader
        from napari_ome_zarr._reader import napari_get_reader as fallback_reader

        return fallback_reader(path)
    return ome_tif_reader


def _is_plate(path: str) -> bool:
    try:
        node = open_ome_zarr(path, mode="r")
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    try:
        return isinstance(node, Plate)
    finally:
        node.close()


def hcs_zarr_reader(path: str | list[str]) -> list[LayerData]:
    """Return one layer per position of the plate at ``path``, named by its path (``A/1/0``)."""
    plate = open_ome_zarr(Path(path if isinstance(path, str) else path[0]), mode="r")
    if not isinstance(plate, Plate):
        raise ValueError(f"{path} is not an HCS plate")
    return [(position["0"], {"name": name}) for name, position in plate.positions()]


def ome_tif_reader(path: str | list[str]) -> list[LayerData]:
    """Return one layer per position of the OME-TIFF or NDTiff dataset at ``path``."""
    dataset = read_images(Path(path if isinstance(path, str) else path[0]))
    if not isinstance(dataset, BaseFOVMapping):
        raise ValueError(f"{path} is not a TIFF dataset")
    fovs = list(dataset.items())
    return [(fov, {"name": "Pos000_000" if len(fovs) == 1 else str(name)}) for name, fov in fovs]
