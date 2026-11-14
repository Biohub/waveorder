"""The bundle in memory: what a student computes around its network, its own test, and a solver of it."""

import pytest
import torch

from tests.models.plates import thin_phase_model
from tests.nn.networks import tiny
from waveorder.errors import BackendUnavailableError, BundleError, ParameterError, ShapeError
from waveorder.nn import Distilled
from waveorder.nn.networks import Affine, Bundle, Contract, Example, Identity, autocast, predict


def _bundle(**changes: object) -> Bundle:
    model = thin_phase_model().imaging_model
    contract = Contract.of(
        model, pixel_size=(0.4, 0.15, 0.15), model="Phase2D", prepare_scope="block", fields=("phase",)
    )
    settings: dict = {"student": tiny.strided(5).eval(), "contract": contract, **changes}
    return Bundle(**settings)


def test_a_bundle_in_memory_runs_its_test_and_has_no_plain_data_form():
    data = torch.randn(1, 1, 5, 32, 32, generator=torch.Generator().manual_seed(0))
    bundle = _bundle(target_normalizer=Affine(offset=0.5, scale=2.0))
    assert (bundle.path, bundle.sha256, bundle.test) == (None, None, None)
    with pytest.raises(BundleError, match="the bundle has no test input"):
        bundle.test_output()
    with pytest.raises(BundleError, match="the bundle's test has no expected output"):
        bundle.self_test()
    tested = _bundle(target_normalizer=Affine(offset=0.5, scale=2.0), test=Example(data=data))
    with torch.no_grad():
        expected = predict(tested.student, data, target_normalizer=Affine(offset=0.5, scale=2.0))
    torch.testing.assert_close(tested.test_output(), expected)
    solver = Distilled(tested)
    assert solver.bundle is tested
    with pytest.raises(ParameterError, match="runs a bundle built in memory, so it has no plain-data form"):
        solver.arguments()


def test_predict_pads_to_the_stride_by_reflection_and_crops():
    student = tiny.strided(5).eval()
    data = torch.randn(1, 1, 5, 40, 44, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        estimate = predict(student, data, target_normalizer=Identity())
        padded = torch.nn.functional.pad(data, (0, 20, 0, 24, 0, 0), mode="reflect")
        torch.testing.assert_close(estimate, student(padded)[..., :40, :44])
    with pytest.raises(ShapeError, match="too small to pad by reflection"):
        predict(student, data[..., :8, :8])


def test_bf16_runs_on_the_cpu_and_fp16_is_refused_there():
    student = tiny.strided(5).eval()
    data = torch.randn(1, 1, 5, 64, 64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = predict(student, data)
        half = predict(student, data, precision="bf16")
    assert half.dtype == torch.float32
    assert (half - full).norm() <= 0.05 * full.norm()  # bfloat16 keeps about 3 significant digits per layer
    with pytest.raises(BackendUnavailableError, match="'fp16' runs on CUDA only"):
        autocast("fp16", torch.device("cpu"))
    with pytest.raises(ParameterError, match="precision must be one of float32, bf16, fp16"):
        autocast("int8", torch.device("cpu"))  # ty: ignore[invalid-argument-type]
