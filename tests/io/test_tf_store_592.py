"""Stores written by #592's own ``write_tf_zarr`` (``tests/io/data/make_592_store.py``) read by the 4.x reader."""

import torch
import zarr

from tests.io.data import make_592_store as made
from waveorder.io.tf_store import TfStore, dotted_592_name

PREFIX = "system.detection.pupil_aberrations."


def test_the_evidence_store_reads_with_its_transfer_and_ignores_its_covariance():
    store = TfStore.load(made.EVIDENCE)
    assert store.origin == "#592"
    assert store.path == made.EVIDENCE
    assert store.parameter_names == (
        f"{PREFIX}defocus",
        f"{PREFIX}astigmatism_0",
        f"{PREFIX}spherical",
        f"{PREFIX}z12",
    )
    assert store.parameter_names == tuple(dotted_592_name(name) for name in made.EVIDENCE_NAMES)
    assert store.parameters.dtype == torch.float64
    assert store.parameters.tolist() == made.EVIDENCE_PARAMETERS
    assert [(item.shape, item.attrs, item.transfer) for item in store.transfers] == [
        (made.EVIDENCE_TRANSFER_SHAPE, {"layout": "#592 transfer_function array"}, None)
    ]
    assert store.tile_grid == made.EVIDENCE_TILE_GRID | {"shape_zyx": None, "blend": None, "units": "um"}
    assert store.provenance == made.EVIDENCE_PROVENANCE
    assert store.scores["score_trace"].tolist() == [-1000.0, -1200.0, -1234.5]  # wo-4.0's name, read as it is


def test_the_evidence_store_holds_what_the_592_writer_was_given():
    group = zarr.open_group(str(made.EVIDENCE), mode="r")
    assert sorted(group.array_keys()) == ["parameters", "parameters_covariance", "score_trace", "transfer_function"]
    assert "format" not in group.attrs
    transfer = group["transfer_function"]
    assert isinstance(transfer, zarr.Array)
    assert str(transfer.dtype) == "complex64"
    torch.testing.assert_close(torch.as_tensor(transfer[...]), made.transfer_function())


def test_the_tiled_store_reads_its_rows():
    store = TfStore.load(made.TILED)
    assert store.origin == "#592"
    assert store.parameter_names == (f"{PREFIX}defocus", f"{PREFIX}coma_x", f"{PREFIX}z13")
    assert store.tiles == 2
    assert store.parameters.tolist() == made.TILED_PARAMETERS
    assert store.row(1) == dict(zip(store.parameter_names, made.TILED_PARAMETERS[1], strict=True))
    assert store.transfers == ()
    assert store.tile_grid["centers_yx"] == made.TILED_TILE_GRID["centers_yx"]
    assert store.tile_grid["overlap_yx"] == [2, 2]
    assert store.provenance == made.TILED_PROVENANCE


def test_a_592_store_saves_as_the_current_format(tmp_path):
    store = TfStore.load(made.EVIDENCE)
    saved = TfStore.load(store.save(tmp_path / "tf.zarr"))
    assert saved.origin == "waveorder.tf/2"
    assert saved.parameter_names == store.parameter_names
    assert saved.parameters_sha256 == store.parameters_sha256
