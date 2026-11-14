"""The bundle on disk: written whole or not at all, rebuilt by name, refused when altered, and its own test."""

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.models.plates import thin_phase_model
from tests.nn.networks import tiny
from waveorder import registry
from waveorder.errors import BundleError, ParameterError, Rejected, ShapeError
from waveorder.io.bundle import SCHEMA, load_bundle, record_file, save_bundle
from waveorder.nn import Distilled
from waveorder.nn.networks import (
    Affine,
    Bundle,
    Contract,
    Example,
    FieldPosition,
    FieldStandardize,
    PlaneHead,
    PlaneOffsets,
    SlidingPlanes,
    UNet,
    predict,
)
from waveorder.nn.networks.conditioning import Bias
from waveorder.registry import arguments

FIRST = Path(__file__).parent / "data" / "student_v1"
"""A bundle written by the ``waveorder.student/1`` writer: a sliding-planes student over a conditioned U-Net."""

SECOND = Path(__file__).parent / "data" / "student_v2"
"""A bundle written by the ``waveorder.student/2`` writer: a strided U-Net with a field-position bias."""


def _contract() -> Contract:
    model = thin_phase_model().imaging_model
    return Contract.of(model, pixel_size=(0.4, 0.15, 0.15), model="Phase2D", prepare_scope="block", fields=("phase",))


def _student() -> UNet:
    return tiny.strided(5, conditioning=Bias(sizes={"field_position": 2}))


def _example(**changes: object) -> Example:
    generator = torch.Generator().manual_seed(0)
    settings: dict = {
        "data": torch.randn(2, 1, 5, 40, 48, generator=generator),
        "condition": {"field_position": torch.rand(2, 2, generator=generator)},
        **changes,
    }
    return Example(**settings)


def _save(path, *, overwrite: bool = False, **changes: object) -> Path:
    settings: dict = {
        "student": _student(),
        "contract": _contract(),
        "conditions": (FieldPosition(),),
        "input_normalizer": FieldStandardize(),
        "target_normalizer": Affine(offset=0.5, scale=2.0),
        "model": {"class": "Phase2D"},
        "record": {"steps": 3},
        "test": _example(),
        **changes,
    }
    return save_bundle(Bundle(**settings), path, overwrite=overwrite)


def test_a_bundle_round_trips_every_part_and_its_hash_is_that_of_bundle_json(tmp_path):
    student = _student().eval()
    position = FieldPosition(axes="xy", offset=0.5, scale=0.4375)  # the production convention
    path = _save(tmp_path / "bundle", student=student, conditions=(position,))
    assert sorted(item.name for item in path.iterdir()) == ["bundle.json", "test.pt", "weights.pt"]
    bundle = load_bundle(path)
    assert bundle.path == path
    assert bundle.sha256 == hashlib.sha256((path / "bundle.json").read_bytes()).hexdigest()
    assert json.loads((path / "bundle.json").read_text())["schema"] == SCHEMA == "waveorder.student/3"
    assert bundle.contract == _contract()
    (loaded,) = bundle.conditions
    assert type(loaded) is FieldPosition
    assert (
        arguments(loaded)
        == arguments(position)
        == {"axes": "xy", "offset": 0.5, "scale": 0.4375, "name": "field_position"}
    )
    assert (type(bundle.input_normalizer), type(bundle.target_normalizer)) == (FieldStandardize, Affine)
    assert (bundle.model, bundle.record) == ({"class": "Phase2D"}, {"steps": 3})
    assert not bundle.student.training
    example = _example()
    with torch.no_grad():
        expected = predict(
            student,
            example.data,
            example.condition,
            input_normalizer=FieldStandardize(),
            target_normalizer=Affine(offset=0.5, scale=2.0),
        )
        torch.testing.assert_close(bundle.predict(example.data, example.condition), expected)
    assert bundle.test is not None
    torch.testing.assert_close(bundle.test.expected, expected)  # computed at save time from the parts
    assert bundle.self_test() <= 1e-6
    for name in ("weights.pt", "test.pt"):
        torch.load(path / name, weights_only=True)  # no pickled code


def test_a_bundle_saved_from_views_of_a_large_tensor_stays_small(tmp_path):
    field = torch.randn(1, 1, 5, 1024, 1024)  # 20 MB; the test input is a 40 x 48 crop of it
    data = field[..., 100:140, 200:248]
    positions = torch.rand(64, 2)
    path = _save(tmp_path / "bundle", test=Example(data=data, condition={"field_position": positions[3:4]}))
    assert (path / "test.pt").stat().st_size < 200_000
    saved = torch.load(path / "test.pt", weights_only=True)
    torch.testing.assert_close(saved["data"], data)
    assert saved["data"].untyped_storage().nbytes() == data.numel() * data.element_size()
    assert load_bundle(path).self_test() <= 1e-6


def test_an_altered_missing_or_foreign_file_is_refused(tmp_path):
    path = _save(tmp_path / "bundle")
    weights = (path / "weights.pt").read_bytes()
    (path / "weights.pt").write_bytes(weights + b"\0")
    with pytest.raises(BundleError, match=r"weights\.pt was altered after the bundle was written"):
        load_bundle(path)
    (path / "weights.pt").write_bytes(weights)
    (path / "test.pt").unlink()
    with pytest.raises(BundleError, match=r"is incomplete: test\.pt is missing"):
        load_bundle(path)
    with pytest.raises(BundleError, match=r"has no bundle\.json"):
        load_bundle(tmp_path)


def test_another_schema_is_refused_with_what_to_do(tmp_path):
    path = _save(tmp_path / "bundle")
    document = json.loads((path / "bundle.json").read_text())
    (path / "bundle.json").write_text(json.dumps({**document, "schema": "waveorder.student/9"}))
    with pytest.raises(BundleError, match=r"has schema 'waveorder\.student/9', but this waveorder reads"):
        load_bundle(path)


def test_a_bundle_naming_a_class_no_path_imports_is_refused(tmp_path):
    path = _save(tmp_path / "bundle")
    document = json.loads((path / "bundle.json").read_text())
    document["student"]["class_path"] = "elsewhere.networks.UNet"
    (path / "bundle.json").write_text(json.dumps(document))
    with pytest.raises(ParameterError, match=r"no class is importable as 'elsewhere\.networks\.UNet'"):
        load_bundle(path)
    old = json.loads((SECOND / "bundle.json").read_text())
    old["student"] = {"elsewhere": old["student"]["unet"]}
    copy = shutil.copytree(SECOND, tmp_path / "second")
    (copy / "bundle.json").write_text(json.dumps(old))
    with pytest.raises(ParameterError, match=r"no network is named 'elsewhere'; choose one of"):
        load_bundle(copy)


def test_a_bundle_of_the_second_schema_loads_and_gives_the_output_it_was_saved_with():
    assert json.loads((SECOND / "bundle.json").read_text())["schema"] == "waveorder.student/2"
    bundle = load_bundle(SECOND)
    assert type(bundle.student) is UNet
    assert bundle.test is not None
    assert bundle.test.expected is not None
    with torch.no_grad():
        estimate = bundle.predict(bundle.test.data, bundle.test.condition)
    torch.testing.assert_close(estimate, bundle.test.expected, rtol=0, atol=1e-6)  # the output the /2 writer computed


def test_a_bundle_of_the_earlier_defocus_kernel_is_refused_naming_why(tmp_path):
    student = tiny.defocus(5)
    planes = PlaneOffsets(5, wavelength=0.532, numerical_aperture=0.9)
    example = Example(data=torch.randn(1, 1, 5, 32, 32), condition={"plane_offsets": torch.linspace(-2, 2, 5)[None]})
    path = _save(tmp_path / "bundle", student=student, conditions=(planes,), test=example)
    assert load_bundle(path).self_test() <= 1e-6
    weights = torch.load(path / "weights.pt", weights_only=True)
    key = "stem.kernel.0.weight"
    weights["student"][key] = weights["student"][key][:, :1].clone()  # one input per plane, as the earlier stem took
    torch.save(weights, path / "weights.pt")
    document = json.loads((path / "bundle.json").read_text())
    document["sha256"]["weights.pt"] = hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest()
    (path / "bundle.json").write_text(json.dumps(document))
    with pytest.raises(BundleError, match=r"takes 1 input\(s\) per plane: it was trained with the earlier defocus"):
        load_bundle(path)


OUTSIDE = """
import sys

from waveorder.io.bundle import load_bundle

bundle = load_bundle(sys.argv[1])
assert type(bundle.student.stem).__module__ == "tests.train.callbacks", type(bundle.student.stem)
print(bundle.self_test() <= bundle.test.tolerance)
"""


def test_a_bundle_with_a_part_from_outside_the_package_loads_in_a_fresh_process(tmp_path):
    from tests.train.callbacks import MeanThenConv

    head = PlaneHead()
    student = UNet(MeanThenConv(5), tiny.encoder(), head, decoder_channels=(32, 16, 8, 8, 8), norm="group")
    path = _save(tmp_path / "bundle", student=student, conditions=(), test=Example(data=torch.randn(1, 1, 5, 32, 32)))
    spec = json.loads((path / "bundle.json").read_text())["student"]["init_args"]["stem"]
    assert spec == {"class_path": "tests.train.callbacks.MeanThenConv", "init_args": {"planes": 5, "channels": 8}}
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [sys.executable, "-c", OUTSIDE, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "True"


def test_a_bundle_of_the_first_schema_loads_and_gives_the_output_it_was_saved_with(tmp_path):
    document = json.loads((FIRST / "bundle.json").read_text())
    assert document["schema"] == "waveorder.student/1"
    assert document["student"]["arguments"]["student"]["name"] == "unet"  # parts as {"name", "arguments"}
    bundle = load_bundle(FIRST)
    assert type(bundle.student) is SlidingPlanes
    assert type(bundle.student.student) is UNet
    assert bundle.student.condition_sizes == {"field_position": 2}
    (position,) = bundle.conditions
    assert arguments(position) == {
        "axes": "xy",
        "offset": [0.5, 0.5],
        "scale": [0.4375, 0.4375],
        "name": "field_position",
    }
    assert (type(bundle.input_normalizer), type(bundle.target_normalizer)) == (FieldStandardize, Affine)
    assert bundle.test is not None
    expected = bundle.test.expected
    assert expected is not None
    with torch.no_grad():
        estimate = bundle.predict(bundle.test.data, bundle.test.condition)
    torch.testing.assert_close(estimate, expected, rtol=0, atol=1e-6)  # the output the /1 writer computed
    solver = Distilled(load_bundle(FIRST))
    assert solver.bundle.sha256 == hashlib.sha256((FIRST / "bundle.json").read_bytes()).hexdigest()
    # Written again, it is a bundle of the current schema that gives the same output.
    test = Example(data=bundle.test.data, condition=bundle.test.condition)
    again = save_bundle(dataclasses.replace(bundle, test=test, path=None, sha256=None), tmp_path / "again")
    assert json.loads((again / "bundle.json").read_text())["schema"] == SCHEMA
    rewritten = load_bundle(again).test
    assert rewritten is not None
    assert rewritten.expected is not None
    torch.testing.assert_close(rewritten.expected, expected, rtol=0, atol=1e-6)


def test_the_plain_data_of_a_distilled_solver_is_rebuilt_by_reading_its_bundle():
    solver = Distilled(load_bundle(FIRST), precision="bf16")
    spec = registry.describe(solver)
    assert spec == {"distilled": {"path": str(FIRST), "precision": "bf16"}}
    rebuilt = registry.resolve(spec, "solver")
    assert isinstance(rebuilt, Distilled)
    assert rebuilt.precision == "bf16"
    assert rebuilt.bundle.sha256 == solver.bundle.sha256


def test_a_self_test_that_this_machine_does_not_reproduce_fails(tmp_path):
    student = _student()
    test = load_bundle(_save(tmp_path / "good", student=student)).test
    assert test is not None
    computed = test.expected
    assert computed is not None
    path = _save(tmp_path / "bad", student=student, test=_example(expected=computed + 1e-3, tolerance=1e-4))
    with pytest.raises(BundleError, match=r"differs from the expected output by 0\.001\d* on cpu, above the tolerance"):
        load_bundle(path).self_test()
    with pytest.raises(ShapeError, match="expected output has shape"):
        _save(tmp_path / "wrong", test=_example(expected=computed[:, :, :, :8]))
    with pytest.raises(ParameterError, match="the bundle has no test"):
        _save(tmp_path / "untested", test=None)


def test_a_bundle_is_replaced_only_when_asked_and_nothing_else_is_deleted(tmp_path):
    path = _save(tmp_path / "bundle")
    with pytest.raises(Rejected, match="exists; pass overwrite=True"):
        _save(path)
    _save(path, record={"steps": 4}, overwrite=True)
    assert load_bundle(path).record == {"steps": 4}
    other = tmp_path / "notes"
    other.mkdir()
    (other / "keep.txt").write_text("mine")
    with pytest.raises(Rejected, match="is not a student bundle; nothing is deleted"):
        _save(other, overwrite=True)
    assert (other / "keep.txt").read_text() == "mine"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["bundle", "notes"]  # no work directory left
    with pytest.raises(ParameterError, match="model and record must hold JSON-able values"):
        _save(tmp_path / "unjsonable", record={"when": object()})


def test_a_failed_replacement_keeps_the_old_bundle(tmp_path, monkeypatch):
    path = _save(tmp_path / "bundle", record={"steps": 3})
    before = load_bundle(path).sha256
    rename = Path.rename

    def failing(self: Path, target: Path) -> Path:
        if ".tmp-" in self.name:  # the new bundle, as it is renamed into place
            raise OSError("disk went away")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", failing)
    with pytest.raises(OSError, match="disk went away"):
        _save(path, record={"steps": 4}, overwrite=True)
    monkeypatch.undo()
    kept = load_bundle(path)
    assert (kept.sha256, kept.record) == (before, {"steps": 3})
    assert sorted(item.name for item in tmp_path.iterdir()) == ["bundle"]  # nothing aside, no work directory


def test_a_file_added_later_is_recorded_and_then_held_to_its_hash(tmp_path):
    path = _save(tmp_path / "bundle")
    before = load_bundle(path).sha256
    (path / "model.onnx").write_bytes(b"onnx bytes")
    digest = record_file(path, "model.onnx")
    assert digest == hashlib.sha256(b"onnx bytes").hexdigest()
    assert json.loads((path / "bundle.json").read_text())["sha256"]["model.onnx"] == digest
    assert load_bundle(path).sha256 != before
    (path / "model.onnx").write_bytes(b"other bytes")
    with pytest.raises(BundleError, match=r"model\.onnx was altered"):
        load_bundle(path)
    with pytest.raises(BundleError, match=r"holds no file missing\.onnx"):
        record_file(path, "missing.onnx")
    with pytest.raises(ParameterError, match="must be the name of a file in the bundle"):
        record_file(path, "bundle.json")
