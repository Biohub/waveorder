import importlib
from pathlib import Path

import pytest
import torch

from tests.data.plates import phase_plates, thin_phase_model
from tests.train.test_tasks import phase_data, phase_task
from waveorder.data import FovIndex, GroupSplit, index_from_plates
from waveorder.errors import IncompatibleModelError
from waveorder.io.bundle import load_bundle, save_bundle
from waveorder.models import Phase2D, _distill
from waveorder.nn import Detection, Distilled, Illumination
from waveorder.train import evaluate, ssim
from waveorder.train.metrics import dynamic_range_ratio


def phase_bundle(index: FovIndex, path: Path) -> Path:
    """Return an untrained bundle of the phase task: its normalizers fitted, its test the first validation batch."""
    task = phase_task()
    data = phase_data(index, task)
    task.fit_normalizers(data)
    return save_bundle(task.bundle(test=next(data.batches("val"))), path)


@pytest.fixture(scope="module")
def bundle(phase_index, tmp_path_factory) -> Path:
    return phase_bundle(phase_index, tmp_path_factory.mktemp("evaluate") / "bundle")


def test_the_student_is_scored_against_the_recorded_targets_of_the_split(bundle, phase_index):
    task = phase_task()
    data = phase_data(phase_index, task)
    scored = evaluate(load_bundle(bundle), data)
    assert scored["split"] == "test"
    assert scored["fields"] == [row.name for row in phase_index.split("test")]
    per_field = scored["per_field"]
    assert isinstance(per_field, dict)
    assert set(per_field) == {"ssim", "dynamic_range", "seconds"}
    loaded = load_bundle(bundle)
    fields = list(data.fields("test"))
    estimates = torch.cat([loaded.predict(field.data[None]) for field in fields])
    targets = torch.stack([field.target for field in fields if field.target is not None])
    torch.testing.assert_close(per_field["ssim"], ssim(estimates, targets), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        per_field["dynamic_range"], dynamic_range_ratio(estimates, targets), atol=1e-4, rtol=1e-4
    )
    assert bool((per_field["seconds"] > 0).all())
    assert scored["seconds_per_field"] == pytest.approx(float(per_field["seconds"][1:].mean()))
    summary = scored["summary"]
    assert isinstance(summary, dict)
    assert set(summary["ssim"]) == {"mean", "p10", "p50", "p90"}


def test_through_the_model_the_student_runs_as_its_solver_and_the_passband_is_scored(bundle, phase_index):
    task = phase_task()
    data = phase_data(phase_index, task)
    model = thin_phase_model()
    solver = Distilled(load_bundle(bundle))
    scored = evaluate(solver, data, imaging_model=model.imaging_model, limit_fields=1)
    per_field = scored["per_field"]
    assert isinstance(per_field, dict)
    assert {"passband_inside", "passband_outside"} <= set(per_field)
    direct = evaluate(load_bundle(bundle), data, limit_fields=1)["per_field"]
    assert isinstance(direct, dict)
    torch.testing.assert_close(per_field["ssim"], direct["ssim"], atol=1e-4, rtol=1e-4)


def test_through_a_reconstruction_model_the_students_system_is_checked_and_the_model_keeps_its_solver(
    bundle, phase_index
):
    task = phase_task()
    data = phase_data(phase_index, task)
    model = thin_phase_model()
    physics = model.solver
    scored = _distill.evaluate(bundle, data, model=model, limit_fields=1)
    assert model.solver is physics
    nn_level = evaluate(load_bundle(bundle), data, imaging_model=model.imaging_model, limit_fields=1)
    for name in ("ssim", "passband_inside"):
        torch.testing.assert_close(scored["per_field"][name], nn_level["per_field"][name])  # ty: ignore[not-subscriptable]
    other = Phase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.4),
        Detection(wavelength=0.532, numerical_aperture=0.8),
        index_of_refraction_media=1.33,
    )
    with pytest.raises(IncompatibleModelError, match="trained for another imaging system"):
        _distill.evaluate(bundle, data, model=other, limit_fields=1)


def test_through_a_reconstruction_model_auto_z_padding_pads_each_field_by_its_own_slices(bundle, tmp_path, monkeypatch):
    # Stacks of 5 and 7 slices of 0.4 um: auto pads 3 slices (5 + 2 * 3 = 11 has no fast length below 5) and 4
    # (7 + 2 * 4 = 15 = 3 * 5).
    shallow = phase_plates(tmp_path / "shallow", ("A/1/0",), shape=(5, 64, 64))
    deep = phase_plates(tmp_path / "deep", ("A/1/0",), shape=(7, 64, 64))
    index = index_from_plates(
        [shallow.inputs, deep.inputs],
        channels=shallow.channels,
        targets=[shallow.targets, deep.targets],
        target_channels=shallow.target_channels,
        split=GroupSplit(by="plate", fractions={"test": 1.0}),
    )
    model = thin_phase_model()
    model.z_padding = "auto"
    pixel_size = load_bundle(bundle).contract.pixel_size
    expected = {planes: model.z_padding_for((planes, 64, 64), pixel_size=pixel_size) for planes in (5, 7)}
    assert expected == {5: 3, 7: 4}
    padded = {}

    def through(imaging_model, _solver, field, *, z_padding):
        padded[field.data.shape[1]] = z_padding
        return field.target.expand(len(imaging_model.system.fields), -1, -1, -1), 0.0

    monkeypatch.setattr(importlib.import_module("waveorder.train.evaluate"), "_through", through)
    scored = _distill.evaluate(bundle, phase_data(index, phase_task()), model=model)
    fields = scored["fields"]
    assert isinstance(fields, list)
    assert len(fields) == 2
    assert padded == expected
    assert model.z_padding == "auto"  # the model keeps its setting
