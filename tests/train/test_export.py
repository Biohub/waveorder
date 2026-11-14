import json
import shutil
from pathlib import Path

import pytest
import torch

from tests.data.plates import PHASE_PIXEL_SIZE, thin_phase_model
from tests.nn.networks.tiny import tiny_unet
from tests.train.test_tasks import phase_data
from waveorder.errors import BundleError, ParameterError, ShapeError
from waveorder.io.bundle import load_bundle, save_bundle
from waveorder.models._distill import distillation, export
from waveorder.nn.networks import Affine, Bias, FieldPosition
from waveorder.train import export_onnx


def session(path: Path) -> object:
    import onnxruntime

    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def bundle(phase_index, tmp_path_factory) -> Path:
    """An untrained bundle whose student reads the field position, with fitted normalizers folded into the graph."""
    task = distillation(
        thin_phase_model(),
        tiny_unet(planes=5, conditioning=Bias({"field_position": 2})),
        pixel_size=PHASE_PIXEL_SIZE,
        conditions=(FieldPosition(),),
        input_normalizer=Affine.robust(),
        target_normalizer=Affine.robust(),
    )
    data = phase_data(phase_index, task)
    task.fit_normalizers(data)
    test = next(data.batches("val"))
    return save_bundle(task.bundle(test=test), tmp_path_factory.mktemp("export") / "bundle")


def test_the_exported_graph_reproduces_the_bundle_at_other_sizes_and_is_recorded(bundle, tmp_path):
    copy = shutil.copytree(bundle, tmp_path / "bundle")
    path = export(copy)  # through the model layer, which records the file in the bundle
    assert path == copy / "model.onnx"
    recorded = json.loads((copy / "bundle.json").read_text())["sha256"]
    assert "model.onnx" in recorded
    loaded = load_bundle(copy)  # verifies every recorded hash, the ONNX file's too

    runtime = session(path)
    assert [entry.name for entry in runtime.get_inputs()] == ["data", "field_position"]  # ty: ignore[missing-attribute]
    data = 1000 + 50 * torch.randn(3, 1, 5, 64, 96, generator=torch.Generator().manual_seed(0))
    position = torch.rand(3, 2, generator=torch.Generator().manual_seed(1))
    (estimate,) = runtime.run(["estimate"], {"data": data.numpy(), "field_position": position.numpy()})  # ty: ignore[missing-attribute]
    with torch.no_grad():
        expected = loaded.predict(data, {"field_position": position})
    torch.testing.assert_close(torch.from_numpy(estimate), expected, atol=1e-3 * float(expected.abs().max()), rtol=1e-4)

    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(BundleError):
        load_bundle(copy)


def test_a_student_without_conditions_exports_and_reproduces_the_torch_output(phase_index, tmp_path):
    task = distillation(thin_phase_model(), tiny_unet(planes=5), pixel_size=PHASE_PIXEL_SIZE)
    data = phase_data(phase_index, task)
    in_memory = task.bundle(test=next(data.batches("val")))
    with pytest.raises(ParameterError, match="and a path for a bundle not read from a directory"):
        export_onnx(in_memory)
    path = export_onnx(in_memory, tmp_path / "model.onnx")
    runtime = session(path)
    assert [entry.name for entry in runtime.get_inputs()] == ["data"]  # ty: ignore[missing-attribute]
    stack = 1000 + 50 * torch.randn(2, 1, 5, 64, 64, generator=torch.Generator().manual_seed(0))
    (estimate,) = runtime.run(["estimate"], {"data": stack.numpy()})  # ty: ignore[missing-attribute]
    with torch.no_grad():
        expected = in_memory.predict(stack, {})
    torch.testing.assert_close(torch.from_numpy(estimate), expected, atol=1e-3 * float(expected.abs().max()), rtol=1e-4)


def test_a_file_written_elsewhere_is_not_recorded_and_a_static_graph_is_checked(bundle, tmp_path):
    copy = shutil.copytree(bundle, tmp_path / "bundle")
    loaded = load_bundle(copy)
    assert loaded.test is not None
    _, _, planes, height, width = loaded.test.data.shape
    path = export_onnx(loaded, tmp_path / "student.onnx", dynamic=False, size=(height, width))
    assert path.is_file()
    assert "student.onnx" not in json.loads((copy / "bundle.json").read_text())["sha256"]
    assert list(session(path).get_inputs()[0].shape) == [1, 1, planes, height, width]  # ty: ignore[missing-attribute]


def test_export_refuses_sizes_and_plane_counts_the_student_does_not_take(bundle, tmp_path):
    loaded = load_bundle(bundle)
    with pytest.raises(ParameterError, match="multiple of the student's stride 32"):
        export_onnx(loaded, tmp_path / "a.onnx", size=(48, 48))
    with pytest.raises(ParameterError, match="cannot check a static graph"):
        export_onnx(loaded, tmp_path / "b.onnx", dynamic=False, size=(128, 128))
    with pytest.raises(ShapeError):
        export_onnx(loaded, tmp_path / "c.onnx", planes=3)
    assert not list(tmp_path.glob("*.onnx"))
