"""The ``tf.zarr`` store: v2 round trip through tensordict, #592 stores, atomic publication, validation."""

import json
import math
from pathlib import Path

import pytest
import torch
import zarr
from tensordict import NonTensorData, PersistentTensorDict

from waveorder.errors import Rejected
from waveorder.io import TfStore
from waveorder.io.tf_store import (
    FORMAT,
    NAMES_592,
    TransferArray,
    dotted_592_name,
    read_transfer,
    store_layout,
)
from waveorder.nn import Free, ParameterPrior

NAMES = ("system.illumination.numerical_aperture", "system.detection.pupil_aberrations.defocus", "object_prior.weight")


def _store(**changes: object) -> TfStore:
    fields: dict[str, object] = {
        "parameter_names": NAMES,
        "parameters": torch.tensor([[0.52, 0.01, 1e-3], [0.55, -0.02, 2e-3]], dtype=torch.float64),
        "tile_grid": {
            "size_yx": [32, 32],
            "overlap_yx": [0, 0],
            "centers_yx": [[2.3, 2.3], [2.3, 7.1]],
            "shape_zyx": [8, 32, 64],
            "blend": "linear",
            "units": "um",
        },
        "provenance": {"kind": "calibrated", "calibration_key": "k" * 64, "at_bound": [[], []]},
    }
    return TfStore(**fields | changes)  # ty: ignore[invalid-argument-type]


def _filled(value: float) -> TfStore:
    return _store(parameters=torch.full((2, 3), value, dtype=torch.float64))


def _beside(target: Path) -> list[str]:
    return sorted(path.name for path in target.parent.iterdir())


def _edit(path: Path, **entries: object) -> None:
    """Replace entries of a v2 store through tensordict: a tensor, or any other value as a JSON entry."""
    data = PersistentTensorDict.from_zarr(path, mode="r+")
    for name, value in entries.items():
        key = tuple(name.split("__"))
        if key in data.keys(include_nested=True):
            del data[key]
        data[key] = value if isinstance(value, torch.Tensor) else NonTensorData(value)
    data.close()


def test_a_store_round_trips_through_tensordict_as_zarr_3(tmp_path):
    transfer = torch.randn(1, 1, 4, 8, 5, dtype=torch.complex64)
    attrs: dict[str, object] = {
        "real": False,
        "pixel_size_zyx_um": [0.4, 0.1, 0.1],
        "dtype": "float32",
        "parameters_sha256": "p" * 64,
    }
    scores = {"score_trace": torch.tensor([3.0, 2.5]), "score_grid": torch.arange(6.0).reshape(3, 2).double()}
    store = _store(transfers=(TransferArray(shape=(4, 8, 8), attrs=attrs, transfer=transfer),), scores=scores)
    path = store.save(tmp_path / "tf.zarr")
    assert path == tmp_path / "tf.zarr"
    assert json.loads((path / "zarr.json").read_text())["zarr_format"] == 3
    read = TfStore.load(path)
    assert read.origin == FORMAT
    assert read.path == path
    assert read.parameter_names == NAMES
    assert torch.equal(read.parameters, store.parameters)
    assert read.tile_grid == store.tile_grid
    assert read.provenance == store.provenance
    assert read.parameters_sha256 == store.parameters_sha256
    assert [(item.shape, item.attrs, item.transfer) for item in read.transfers] == [((4, 8, 8), attrs, None)]
    assert read.scores.keys() == scores.keys()
    assert all(torch.equal(read.scores[name], values.double()) for name, values in scores.items())
    stored = read_transfer(path, (4, 8, 8))
    assert stored is not None
    assert stored.attrs == attrs
    assert stored.transfer is not None
    assert stored.transfer.dtype == torch.complex64
    assert torch.equal(stored.transfer, transfer)  # exact, in its own dtype
    assert read_transfer(path, (4, 8, 16)) is None
    data = PersistentTensorDict.from_zarr(path, mode="r")  # the entries, as any tensordict reader sees them
    assert data.get_non_tensor("format") == FORMAT
    assert data.get_non_tensor("parameter_names") == list(NAMES)
    assert torch.equal(data["transfer_function", "4x8x8", "transfer"], transfer)
    data.close()


def test_the_repr_is_one_line_without_tensors(tmp_path):
    tiled = repr(_store())
    assert tiled == f"TfStore(kind='calibrated', tiles=2, parameters={list(NAMES)}, transfers=0)"
    one = _store(parameters=torch.tensor([[0.52, 0.01, 1e-3]], dtype=torch.float64))
    assert f"parameters={{'{NAMES[0]}': 0.52, '{NAMES[1]}': 0.01, '{NAMES[2]}': 0.001}}" in repr(one)
    read = TfStore.load(_store().save(tmp_path / "tf.zarr"))
    assert repr(read).endswith(f"path={str(tmp_path / 'tf.zarr')!r})")
    assert "tensor(" not in repr(read)
    assert "\n" not in repr(read)


def test_a_store_gives_the_prior_of_the_free_quantities_it_names():
    other = Free(1.0, tolerance=0.1)
    specs = {NAMES[0]: Free(0.5, tolerance=0.05, bounds=(0.3, 0.7)), "system.other": other}
    prior = _store().to_prior(specs)
    assert isinstance(prior, ParameterPrior)
    assert prior.tiles == 2  # one centre per tile for the quantity the store names; the other keeps its nominal
    values = {NAMES[0]: torch.tensor([0.52, 0.55], dtype=torch.float64), "system.other": torch.ones(2).double()}
    torch.testing.assert_close(prior.nll(values), torch.zeros(2, dtype=torch.float64))
    one = {NAMES[0]: torch.tensor(0.6, dtype=torch.float64), "system.other": torch.tensor(1.0, dtype=torch.float64)}
    torch.testing.assert_close(prior.select([1]).nll(one), torch.tensor(0.5, dtype=torch.float64))
    assert _store().to_prior({"system.other": other}).tiles == 1  # it names none of them


def _write_592(path, *, names, parameters, covariance=None, transfer=True):
    """Write a store as #592's ``write_tf_zarr`` does (``waveorder/calibrate/tfzarr.py`` on that branch)."""
    tiles, size = parameters.shape
    if covariance is None:
        covariance = torch.zeros(tiles, size, size, dtype=torch.float64)
    group = zarr.open_group(str(path), mode="w")
    group.create_array("parameters", shape=parameters.shape, dtype="float64")[:] = parameters.numpy()
    group.create_array("parameters_covariance", shape=covariance.shape, dtype="float64")[:] = covariance.numpy()
    if transfer:
        otf = torch.ones(4, 8, 8, dtype=torch.complex64).numpy()
        group.create_array("transfer_function", shape=otf.shape, dtype=str(otf.dtype), chunks=otf.shape)[:] = otf
    group.attrs["parameter_names"] = list(names)
    group.attrs["tile_grid"] = {"size_yx": [8, 8], "overlap_yx": [0, 0], "centers_yx": [[0.0, 0.0]]}
    group.attrs["provenance"] = {"kind": "refined", "sources": ["in.zarr/0/0/0"], "config_hash": "c" * 64}


def test_a_592_store_reads_through_the_name_map(tmp_path):
    path = tmp_path / "tf_592.zarr"
    parameters = torch.tensor([[0.1, -0.05, 0.02]], dtype=torch.float64)
    _write_592(path, names=["defocus", "coma_x", "z12"], parameters=parameters)
    store = TfStore.load(path)
    assert store.origin == "#592"
    prefix = "system.detection.pupil_aberrations."
    assert store.parameter_names == (f"{prefix}defocus", f"{prefix}coma_x", f"{prefix}z12")
    assert store.row() == dict(zip(store.parameter_names, [0.1, -0.05, 0.02], strict=True))
    assert [item.shape for item in store.transfers] == [(4, 8, 8)]
    assert store.tile_grid["centers_yx"] == [[0.0, 0.0]]
    assert store.provenance["kind"] == "refined"


def test_a_store_written_with_a_covariance_array_still_loads(tmp_path):
    # #592 wrote parameters_covariance; the array is ignored.
    covariance = torch.tensor([[[1e-4, 1e-5], [1e-5, 2e-4]]], dtype=torch.float64)
    older = tmp_path / "tf_592.zarr"
    _write_592(older, names=["defocus", "spherical"], parameters=torch.zeros(1, 2).double(), covariance=covariance)
    assert TfStore.load(older).parameters.tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize(
    ("name", "dotted"),
    [
        ("defocus", "defocus"),
        ("astigmatism_45", "astigmatism_45"),
        ("astigmatism_0", "astigmatism_0"),
        ("coma_y", "coma_y"),
        ("coma_x", "coma_x"),
        ("trefoil_y", "trefoil_y"),
        ("trefoil_x", "trefoil_x"),
        ("spherical", "spherical"),
        ("z4", "defocus"),
        ("z11", "spherical"),
        ("z12", "z12"),
        ("z22", "z22"),
    ],
)
def test_the_592_name_table(name, dotted):
    assert dotted_592_name(name) == f"system.detection.pupil_aberrations.{dotted}"


def test_the_592_table_covers_its_named_terms_and_refuses_others():
    assert set(NAMES_592) == {
        "defocus",
        "astigmatism_45",
        "astigmatism_0",
        "coma_y",
        "coma_x",
        "trefoil_y",
        "trefoil_x",
        "spherical",
    }
    with pytest.raises(Rejected, match="not a Zernike term name"):
        dotted_592_name("numerical_aperture")


def _write_3x(path) -> None:
    group = zarr.open_group(str(path), mode="w")
    group.create_array("real_potential_transfer_function", shape=(4, 8, 8), dtype="complex64")


def test_a_3x_store_is_refused_with_the_recompute_command(tmp_path):
    path = tmp_path / "tf_3x.zarr"
    _write_3x(path)
    with pytest.raises(Rejected, match=r"made by waveorder 3\.x.*recompute it with wo compute-tf"):
        TfStore.load(path)


def test_another_format_and_a_missing_store_are_refused(tmp_path):
    path = _store().save(tmp_path / "tf.zarr")
    _edit(path, format="waveorder.tf/3")
    with pytest.raises(Rejected, match=r"has format 'waveorder.tf/3'; .*newer waveorder"):
        TfStore.load(path)
    with pytest.raises(Rejected, match="does not exist"):
        TfStore.load(tmp_path / "absent.zarr")
    with pytest.raises(Rejected, match="does not exist"):
        read_transfer(tmp_path / "absent.zarr", (4, 8, 8))


def test_the_layout_names_each_kind_of_store(tmp_path):
    current = _store().save(tmp_path / "tf.zarr")
    _write_592(tmp_path / "tf_592.zarr", names=["defocus"], parameters=torch.zeros(1, 1).double())
    _write_3x(tmp_path / "tf_3x.zarr")
    zarr.open_group(str(tmp_path / "other.zarr"), mode="w")
    (tmp_path / "plain").mkdir()
    assert store_layout(current) == "waveorder.tf"
    assert store_layout(tmp_path / "tf_592.zarr") == "#592"
    assert store_layout(tmp_path / "tf_3x.zarr") == "3.x"
    assert store_layout(tmp_path / "other.zarr") is None
    assert store_layout(tmp_path / "plain") is None
    assert store_layout(tmp_path / "absent.zarr") is None


_REPEATED = [*NAMES[:2], NAMES[0]]
_INFINITE = torch.tensor([[0.5, math.inf, 1.0], [0.5, 0.0, 1.0]]).double()


@pytest.mark.parametrize(
    ("change", "match"),
    [({"parameter_names": _REPEATED}, "repeats a name"), ({"parameters": _INFINITE}, "parameters must be finite")],
    ids=["names", "finite"],
)
def test_validation_on_read(tmp_path, change, match):
    path = _store().save(tmp_path / "tf.zarr")
    _edit(path, **change)
    with pytest.raises(Rejected, match=match):
        TfStore.load(path)


def test_an_invalid_store_is_not_saved(tmp_path):
    with pytest.raises(Rejected, match="parameters must be finite"):
        _filled(math.inf).save(tmp_path / "tf.zarr")
    assert _beside(tmp_path / "tf.zarr") == []


def test_publication_refuses_then_replaces_and_cleans_up(tmp_path):
    target = tmp_path / "deep" / "tf.zarr"
    _store().save(target)
    with pytest.raises(Rejected, match="pass overwrite to replace it") as error:
        _filled(0.6).save(target)
    assert error.value.problems[0].location == str(target)
    assert TfStore.load(target).parameters[0, 0].item() == 0.52
    _filled(0.6).save(target, overwrite=True)
    assert TfStore.load(target).parameters[0, 0].item() == 0.6
    assert _beside(target) == ["tf.zarr"]


def test_overwrite_replaces_only_a_transfer_function_store(tmp_path):
    other = tmp_path / "plate.zarr"
    zarr.open_group(str(other), mode="w").attrs["kept"] = True
    with pytest.raises(Rejected, match="is not a transfer-function store"):
        _store().save(other, overwrite=True)
    assert zarr.open_group(str(other), mode="r").attrs["kept"] is True
    file = tmp_path / "notes.txt"
    file.write_text("kept")
    with pytest.raises(Rejected, match="is a file"):
        _store().save(file, overwrite=True)
    assert file.read_text() == "kept"
    assert _beside(other) == ["notes.txt", "plate.zarr"]


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        (lambda p: _edit(p, transfer_function__big__transfer=torch.zeros(2)), "transfer_function/big is not"),
        (lambda p: _edit(p, tile_grid=[1, 2]), "attribute tile_grid must be a mapping, got list"),
        (lambda p: _edit(p, provenance="x"), "attribute provenance must be a mapping, got str"),
        (lambda p: _edit(p, parameter_names="abc"), "parameter_names must be a list of names"),
        (lambda p: _edit(p, tile_grid={"shape_zyx": [1, "a", 2]}), "tile_grid.shape_zyx must be"),
        (lambda p: _edit(p, tile_grid={"centers_yx": [[1]]}), "tile_grid.centers_yx must be"),
        (lambda p: _edit(p, parameters="abc"), "parameters must be a numeric array"),
        (lambda p: _edit(p, transfer_function="abc"), "transfer_function must be a group"),
    ],
    ids=["group_name", "grid_type", "provenance_type", "names_type", "shape_type", "centres_type", "array_type", "tf"],
)
def test_a_corrupt_store_is_rejected_at_its_path(tmp_path, damage, match):
    path = _store().save(tmp_path / "tf.zarr")
    damage(path)
    with pytest.raises(Rejected, match=match) as error:
        TfStore.load(path)
    assert error.value.problems[0].location == str(path)


def test_a_save_writes_a_temporary_store_then_renames_it(tmp_path, monkeypatch):
    target = tmp_path / "tf.zarr"
    rename = Path.rename
    renames: list[tuple[str, str, bool]] = []

    def recording(self, destination):
        renames.append((self.name, Path(destination).name, (self / "zarr.json").is_file()))
        return rename(self, destination)

    monkeypatch.setattr(Path, "rename", recording)
    _store().save(target)
    _filled(0.6).save(target, overwrite=True)
    monkeypatch.undo()
    (first, published, whole), (replaced, aside, _), (second, again, complete) = renames
    assert first.startswith("tf.zarr.tmp-")
    assert (published, whole) == ("tf.zarr", True)
    suffix = aside.removeprefix("tf.zarr.old-")
    assert replaced == "tf.zarr"
    assert len(suffix) == 12
    assert (second, again, complete) == (f"tf.zarr.tmp-{suffix}", "tf.zarr", True)
    assert TfStore.load(target).parameters[0, 0].item() == 0.6
    assert _beside(target) == ["tf.zarr"]


def test_a_save_that_fails_while_writing_leaves_no_temporary(tmp_path, monkeypatch):
    from waveorder.io import tf_store

    def failing(work: Path, _: TfStore) -> None:
        work.mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(tf_store, "_write", failing)
    with pytest.raises(OSError, match="disk full"):
        _store().save(tmp_path / "tf.zarr")
    assert _beside(tmp_path / "tf.zarr") == []
