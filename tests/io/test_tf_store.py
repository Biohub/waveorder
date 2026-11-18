"""The ``tf.zarr`` store: round trip as plain zarr, #592 and wo-4.0 stores, atomic publication, validation."""

import json
import math
from pathlib import Path

import pytest
import torch
import zarr
from zarr.core.common import JSON

from tests.io.stores import PROVENANCE_592, TILE_GRID_592, write_592
from waveorder.errors import CalibrationWarning, Rejected
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


def _edit(path: Path, **entries: torch.Tensor | JSON) -> None:
    """Replace attributes of a store, or its arrays: a tensor, at a path whose parts ``__`` separates."""
    group = zarr.open_group(str(path), mode="r+")
    for name, value in entries.items():
        if isinstance(value, torch.Tensor):
            key = name.replace("__", "/")
            if key in group:
                del group[key]
            group.create_array(key, data=value.numpy())
        else:
            group.attrs[name] = value


def test_a_store_round_trips_as_plain_zarr_3(tmp_path):
    transfer = torch.randn(1, 1, 4, 8, 5, dtype=torch.complex64)
    attrs: dict[str, object] = {
        "real": False,
        "pixel_size_zyx_um": [0.4, 0.1, 0.1],
        "dtype": "float32",
        "parameters_sha256": "p" * 64,
    }
    scores = {"score_trace": torch.tensor([3.0, 2.5]), "score_grid": torch.arange(6.0).reshape(3, 2).double()}
    covariance = torch.zeros(2, 3, 3, dtype=torch.float64)
    covariance[0, :2, :2] = torch.tensor([[1e-4, -2e-5], [-2e-5, 3e-4]])  # the system's quantities of tile 0
    covariance[1, :2, :2] = math.nan  # tile 1: a quantity on a bound
    transfers = (TransferArray(shape=(4, 8, 8), attrs=attrs, transfer=transfer),)
    store = _store(parameters_covariance=covariance, transfers=transfers, scores=scores)
    path = store.save(tmp_path / "tf.zarr")
    assert path == tmp_path / "tf.zarr"
    read = TfStore.load(path)
    assert read.origin == FORMAT
    assert read.path == path
    assert read.parameter_names == NAMES
    assert torch.equal(read.parameters, store.parameters)
    assert read.parameters_covariance is not None
    torch.testing.assert_close(read.parameters_covariance, covariance, rtol=0, atol=0, equal_nan=True)
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
    group = zarr.open_group(str(path), mode="r")  # what any zarr reader sees
    assert json.loads((path / "zarr.json").read_text())["zarr_format"] == 3
    assert dict(group.attrs) == {
        "format": FORMAT,
        "parameter_names": list(NAMES),
        "tile_grid": store.tile_grid,
        "provenance": store.provenance,
    }
    written = group["parameters_covariance"]
    assert isinstance(written, zarr.Array)
    assert written.shape == (2, 3, 3)
    assert dict(group["transfer_function/4x8x8"].attrs) == attrs


def test_a_store_without_a_covariance_writes_zeros_and_its_metadata_as_strict_json(tmp_path):
    path = _store(provenance={"kind": "calibrated", "objective": [math.nan]}).save(tmp_path / "tf.zarr")
    read = TfStore.load(path)
    assert read.parameters_covariance is not None
    assert torch.equal(read.parameters_covariance, torch.zeros(2, 3, 3, dtype=torch.float64))
    assert read.provenance["objective"] == ["NaN"]  # a NaN as its string: zarr.json stays strict JSON
    json.loads((path / "zarr.json").read_text(), parse_constant=lambda name: pytest.fail(f"zarr.json holds {name}"))


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


_COVARIANCE = torch.tensor([[4e-4, 3e-4], [3e-4, 9e-4]], dtype=torch.float64)  # correlation 0.5
_SPECS = {
    NAMES[0]: Free(0.5, tolerance=0.05, bounds=(0.3, 0.7)),
    NAMES[1]: Free(0.0, bounds=(-0.5, 0.5)),  # flat in the model: the stored covariance gives it its width
    NAMES[2]: Free(1e-3, tolerance=1e-3, bounds=(1e-6, 1.0)),
}


def _one_row(covariance: torch.Tensor) -> TfStore:
    """One row; the system's block ``covariance``, the weight's row and column zero (no covariance)."""
    full = torch.zeros(1, 3, 3, dtype=torch.float64)
    full[0, :2, :2] = covariance
    return _store(parameters=torch.tensor([[0.52, 0.01, 1e-3]], dtype=torch.float64), parameters_covariance=full)


def test_a_stored_covariance_is_the_prior_over_the_quantities_it_covers_as_a_quadratic_form():
    prior = _one_row(_COVARIANCE).to_prior(_SPECS)
    assert prior.precision_names == NAMES[:2]  # the weight's zero row and column are left out
    assert prior.covered == NAMES
    offset = torch.tensor([0.02, -0.03], dtype=torch.float64)
    values = {NAMES[0]: 0.52 + offset[0], NAMES[1]: 0.01 + offset[1], NAMES[2]: torch.tensor(3e-3).double()}
    form = 0.5 * offset @ torch.linalg.inv(_COVARIANCE) @ offset  # the correlation counts, not the diagonal alone
    weight = 0.5 * ((3e-3 - 1e-3) / 1e-3) ** 2  # outside the block: its own tolerance
    torch.testing.assert_close(prior.nll(values), form + weight)
    diagonal = 0.5 * float((offset.square() / _COVARIANCE.diagonal()).sum())
    assert abs(float(form) - diagonal) > 0.1 * diagonal
    batched = {name: value.expand(4) for name, value in values.items()}  # batch shape and dtype as the values'
    torch.testing.assert_close(prior.nll(batched), (form + weight).expand(4))
    single = {name: value.float() for name, value in values.items()}
    assert prior.nll(single).dtype == torch.float32


def test_an_all_zero_or_absent_covariance_leaves_the_tolerances():
    for store in (_one_row(torch.zeros(2, 2, dtype=torch.float64)), _store(parameters_covariance=None)):
        prior = store.to_prior(_SPECS)
        assert prior.precision is None
        assert prior.covered == (NAMES[0], NAMES[2])


def test_a_nan_covariance_falls_back_to_the_tolerances_with_a_warning():
    with pytest.warns(CalibrationWarning, match=rf"over {NAMES[0]}, {NAMES[1]} holds NaN") as record:
        prior = _one_row(torch.full((2, 2), math.nan, dtype=torch.float64)).to_prior(_SPECS)
    assert len(record) == 1
    assert prior.precision is None
    assert prior.covered == (NAMES[0], NAMES[2])


def test_a_singular_covariance_is_refused():
    singular = torch.tensor([[1e-4, 1e-4], [1e-4, 1e-4]], dtype=torch.float64)
    with pytest.raises(Rejected, match="block over 2 covered parameters is singular and cannot serve as a Gaussian"):
        _one_row(singular).to_prior(_SPECS)


def test_a_tiled_store_gives_each_tile_its_own_precision():
    covariance = torch.zeros(2, 3, 3, dtype=torch.float64)
    covariance[0, :2, :2] = _COVARIANCE
    covariance[1, :2, :2] = 4 * _COVARIANCE
    prior = _store(parameters_covariance=covariance).to_prior(_SPECS)
    assert prior.tiles == 2
    offset = torch.tensor([0.02, -0.03], dtype=torch.float64)
    centres = _store().parameters[:, :2]
    values = {NAMES[0]: centres[:, 0] + offset[0], NAMES[1]: centres[:, 1] + offset[1]}
    values[NAMES[2]] = torch.tensor([1e-3, 2e-3], dtype=torch.float64)
    form = 0.5 * offset @ torch.linalg.inv(_COVARIANCE) @ offset
    torch.testing.assert_close(prior.nll(values), torch.stack([form, form / 4]))
    one = {name: value[1] for name, value in values.items()}
    torch.testing.assert_close(prior.select([1]).nll(one), form / 4)


PREFIX_592 = "system.detection.pupil_aberrations."


def test_a_592_store_reads_through_the_name_map(tmp_path):
    path = tmp_path / "tf_592.zarr"
    parameters = torch.tensor([[0.1, -0.05, 0.02]], dtype=torch.float64)
    transfer = torch.ones(4, 8, 8, dtype=torch.complex64)
    write_592(path, names=["defocus", "coma_x", "z12"], parameters=parameters, transfer=transfer)
    store = TfStore.load(path)
    assert store.origin == "#592"
    assert store.path == path
    assert store.parameter_names == (f"{PREFIX_592}defocus", f"{PREFIX_592}coma_x", f"{PREFIX_592}z12")
    assert store.parameters.dtype == torch.float64
    assert store.row() == dict(zip(store.parameter_names, [0.1, -0.05, 0.02], strict=True))
    assert [(item.shape, item.attrs, item.transfer) for item in store.transfers] == [
        ((4, 8, 8), {"layout": "#592 transfer_function array"}, None)
    ]
    assert store.tile_grid == TILE_GRID_592 | {"shape_zyx": None, "blend": None, "units": "um"}
    assert store.provenance == PROVENANCE_592


def test_a_wo_4_0_evidence_store_keeps_its_scores_provenance_and_covariance(tmp_path):
    covariance = torch.tensor([[[1e-4, 1e-5], [1e-5, 2e-4]]], dtype=torch.float64)
    provenance = PROVENANCE_592 | {"score": "evidence", "time_index": 0, "best_score": -1234.5, "noise_sigma": 0.01}
    path = write_592(
        tmp_path / "tf_592.zarr",
        names=["astigmatism_0", "spherical"],
        parameters=torch.tensor([[-0.0625, 0.03125]], dtype=torch.float64),
        covariance=covariance,
        arrays={"score_trace": torch.tensor([-1000.0, -1200.0, -1234.5], dtype=torch.float64)},
        provenance=provenance,
    )
    store = TfStore.load(path)
    assert store.parameters.tolist() == [[-0.0625, 0.03125]]
    assert store.scores["score_trace"].tolist() == [-1000.0, -1200.0, -1234.5]  # wo-4.0's name, read as it is
    assert store.provenance == provenance
    assert store.transfers == ()
    assert store.parameters_covariance is not None
    assert torch.equal(store.parameters_covariance, covariance)


def test_a_tiled_592_store_reads_its_rows(tmp_path):
    parameters = [[0.25, -0.125, 0.0625], [0.5, 0.0, -0.03125]]
    grid = {"size_yx": [8, 8], "overlap_yx": [2, 2], "centers_yx": [[1.5, 1.5], [1.5, 4.5]]}
    path = write_592(
        tmp_path / "tf_592_tiled.zarr",
        names=["defocus", "coma_x", "z13"],
        parameters=torch.tensor(parameters, dtype=torch.float64),
        tile_grid=grid,
    )
    store = TfStore.load(path)
    assert store.parameter_names == (f"{PREFIX_592}defocus", f"{PREFIX_592}coma_x", f"{PREFIX_592}z13")
    assert store.tiles == 2
    assert store.parameters.tolist() == parameters
    assert store.row(1) == dict(zip(store.parameter_names, parameters[1], strict=True))
    assert store.tile_grid["centers_yx"] == grid["centers_yx"]
    assert store.tile_grid["overlap_yx"] == [2, 2]


def test_a_592_store_saves_as_the_current_format(tmp_path):
    path = write_592(tmp_path / "tf_592.zarr", names=["defocus", "z12"], parameters=torch.zeros(1, 2).double())
    store = TfStore.load(path)
    saved = TfStore.load(store.save(tmp_path / "tf.zarr"))
    assert saved.origin == FORMAT
    assert saved.parameter_names == store.parameter_names
    assert saved.parameters_sha256 == store.parameters_sha256
    assert saved.parameters_covariance is not None
    assert torch.equal(saved.parameters_covariance, torch.zeros(1, 2, 2, dtype=torch.float64))


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


def _write_v2(path: Path) -> None:
    """Write a store as earlier 4.x wrote ``waveorder.tf/2``: its root marker, and JSON entries as byte arrays."""
    group = zarr.open_group(str(path), mode="w", attributes={"__tensordict__": {"batch_size": [], "version": 1}})
    for name, text in (("format", '"waveorder.tf/2"'), ("parameter_names", json.dumps(list(NAMES)))):
        group.create_array(name, data=torch.frombuffer(bytearray(text.encode()), dtype=torch.uint8).numpy())
    group.create_array("parameters", data=torch.zeros(1, 3, dtype=torch.float64).numpy())


def test_a_v2_store_is_refused_with_the_recompute_message_and_may_be_overwritten(tmp_path):
    path = tmp_path / "tf.zarr"
    _write_v2(path)
    with pytest.raises(Rejected, match=r"earlier waveorder 4\.x \(format waveorder.tf/2\).*recompute it") as error:
        TfStore.load(path)
    assert error.value.problems[0].location == str(path)
    assert store_layout(path) == "waveorder.tf"
    _store().save(path, overwrite=True)
    assert TfStore.load(path).origin == FORMAT


def test_another_format_and_a_missing_store_are_refused(tmp_path):
    path = _store().save(tmp_path / "tf.zarr")
    _edit(path, format="waveorder.tf/4")
    with pytest.raises(Rejected, match=r"has format 'waveorder.tf/4'; .*newer waveorder"):
        TfStore.load(path)
    with pytest.raises(Rejected, match="does not exist"):
        TfStore.load(tmp_path / "absent.zarr")
    with pytest.raises(Rejected, match="does not exist"):
        read_transfer(tmp_path / "absent.zarr", (4, 8, 8))


def test_the_layout_names_each_kind_of_store(tmp_path):
    current = _store().save(tmp_path / "tf.zarr")
    write_592(tmp_path / "tf_592.zarr", names=["defocus"], parameters=torch.zeros(1, 1).double())
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
_INFINITE_COVARIANCE = torch.full((2, 3, 3), math.inf, dtype=torch.float64)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"parameter_names": _REPEATED}, "repeats a name"),
        ({"parameters": _INFINITE}, "parameters must be finite"),
        ({"parameters_covariance": _INFINITE_COVARIANCE}, "parameters_covariance must hold no infinity"),
        ({"parameters_covariance": torch.zeros(2, 3)}, r"parameters_covariance must be \[tiles, 3, 3\]"),
    ],
    ids=["names", "finite", "covariance_finite", "covariance_shape"],
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
        (lambda p: _edit(p, parameters=torch.ones(2, 3, dtype=torch.bool)), "parameters must be a numeric array"),
        (lambda p: _edit(p, transfer_function=torch.zeros(2)), "transfer_function must be a group"),
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
