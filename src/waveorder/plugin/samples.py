"""napari's sample data for waveorder: a polarization target and a zebrafish embryo, downloaded from Zenodo once."""

import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Literal

from iohub import open_ome_zarr
from iohub.ngff import Plate
from platformdirs import user_data_dir

logger = logging.getLogger(__name__)

_RECORD = "https://zenodo.org/record/8386856/files"
_SAMPLES = {
    "target": ("sample_contribution", "10 MB"),
    "embryo": ("sample_contribution_embryo", "92 MB"),
}

type LayerData = tuple[Any, dict[str, Any]]


def _layers(dataset: Plate, layer_names: list[str]) -> list[LayerData]:
    position = dataset["0/0/0"]
    image = position["0"]
    layers = []
    for channel_name in layer_names:
        channel_index = dataset.channel_names.index(channel_name)
        layers.append(((image[:, channel_index],), {"name": channel_name, "scale": position.scale[3:]}))
    return layers


def download_and_unzip(data_type: Literal["target", "embryo"]) -> tuple[Path, Path]:
    """Download a sample ``.zip`` from Zenodo, unzip it, and return the paths of its raw data and reconstruction.

    The download is skipped when the files are already in :func:`platformdirs.user_data_dir`.

    Raises
    ------
    urllib.error.URLError
        If Zenodo cannot be reached.
    """
    old_data_path = Path(user_data_dir("waveorder-sample-v1.4"))
    if old_data_path.exists():
        shutil.rmtree(old_data_path)

    directory = Path(user_data_dir("waveorder-sample-v1.5"))
    directory.mkdir(exist_ok=True, parents=True)
    name, size = _SAMPLES[data_type]
    data_dirpath = directory / name
    archive = data_dirpath.with_suffix(".zip")
    if not archive.exists():
        logger.warning("Downloading %s sample contribution. This might take a moment...", size)
        partial = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(f"{_RECORD}/{name}.zip?download=1", partial)
        partial.rename(archive)
    if not data_dirpath.exists():
        shutil.unpack_archive(archive, extract_dir=directory)
    return data_dirpath / "raw_data.zarr", data_dirpath / "reconstruction.zarr"


def read_polarization_target_data() -> list[LayerData]:
    """Return the polarization data sample contribution."""
    data_path, _ = download_and_unzip("target")
    dataset = open_ome_zarr(data_path)
    assert isinstance(dataset, Plate)
    return _layers(dataset, list(dataset.channel_names))


def read_polarization_target_reconstruction() -> list[LayerData]:
    """Return the polarization target reconstruction sample contribution."""
    _, recon_path = download_and_unzip("target")
    dataset = open_ome_zarr(recon_path)
    assert isinstance(dataset, Plate)
    return _layers(dataset, ["Phase3D", "Retardance", "Orientation"])


def read_zebrafish_embryo_reconstruction() -> list[LayerData]:
    """Return the embryo reconstruction sample contribution."""
    _, recon_path = download_and_unzip("embryo")
    dataset = open_ome_zarr(recon_path)
    assert isinstance(dataset, Plate)
    return _layers(dataset, ["Retardance", "Orientation"])
