import pickle
from collections.abc import Mapping, Sequence
from typing import Literal

import pytest
import torch

from tests.data.plates import PHASE_PIXEL_SIZE, thin_phase_model
from tests.nn.networks.tiny import tiny_unet
from waveorder.data import Batch, FovIndex, PatchData, RandomPatches, Transform
from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.io.bundle import load_bundle, save_bundle
from waveorder.models import ReconstructionModel
from waveorder.models._distill import distillation
from waveorder.nn import Distilled
from waveorder.nn.networks import (
    Affine,
    Bias,
    FieldPosition,
    FieldStandardize,
    Normalizer,
    ResNetEncoder,
    StridedVolume,
    SubPixelHead,
    UNet,
)
from waveorder.train import L1, SSIM, Distillation, Loss

PLANES = 5
"""Planes of the ``phase_pair`` stacks."""


def phase_task(
    model: ReconstructionModel | None = None,
    *,
    target_normalizer: Normalizer | Literal["robust"] | None = "robust",
    losses: Mapping[str, Loss | tuple[Loss, float]] | None = None,
    transforms: Sequence[Transform] = (),
) -> Distillation:
    """Return a task on a tiny UNet for ``phase_pair``: the phase field of a fresh ``thin_phase_model``."""
    return distillation(
        thin_phase_model() if model is None else model,
        tiny_unet(planes=PLANES),
        pixel_size=PHASE_PIXEL_SIZE,
        target_normalizer=Affine.robust() if target_normalizer == "robust" else target_normalizer,
        losses={"l1": L1(), "ssim": (SSIM(window=7), 0.5)} if losses is None else losses,
        transforms=transforms,
    )


def phase_data(
    index: FovIndex, task: Distillation, *, workers: int = 0, patches: RandomPatches | None = None
) -> PatchData:
    """Return the loader of ``index`` built from the task's parts: 32 px patches, 4 per field, batches of 4."""
    return PatchData(
        index,
        patches=RandomPatches((32, 32), per_field=4) if patches is None else patches,
        batch_size=4,
        workers=workers,
        stride=32,
        prepare=task.prepare,
        conditions=task.conditions,
        input_normalizer=task.input_normalizer,
        target_normalizer=task.target_normalizer,
    )


def first_batch(data: PatchData, split: str = "train") -> Batch:
    return next(data.batches(split))


def test_the_default_fields_are_the_first_of_the_system_and_make_the_contract():
    task = phase_task()
    assert task.fields == ("phase",)
    contract = task.contract
    assert contract.fields == ("phase",)
    assert contract.channels == 1
    assert contract.model == "Phase2D"
    assert contract.prepare_scope == "block"
    assert contract.pixel_size == PHASE_PIXEL_SIZE
    assert contract.system["class"] == "ThinPhase2D"
    assert isinstance(contract.system["sha256"], str)


def test_the_model_is_held_but_not_trained():
    task = phase_task()
    trained = {id(parameter) for parameter in task.parameters()}
    assert trained == {id(parameter) for parameter in task.student.parameters()}
    assert not {id(parameter) for parameter in task.imaging_model.parameters()} & trained
    assert not hasattr(task, "solver")  # the targets are recorded: the task holds no solver to make them


def test_no_step_runs_the_imaging_model(no_physics, phase_index, tmp_path):
    task = phase_task()
    data = phase_data(phase_index, task)
    task.fit_normalizers(data)
    task.step(first_batch(data)).loss.backward()
    with torch.no_grad():
        task.step(first_batch(data, "val"), stage="val")
    bundle = load_bundle(save_bundle(task.bundle(), tmp_path / "bundle"))
    assert bundle.test is not None
    assert bundle.self_test() <= bundle.test.tolerance
    assert no_physics == []


def custom_student(**changes) -> UNet:
    parts = {"stem": StridedVolume(PLANES), "head": SubPixelHead(1), "conditioning": None, **changes}
    return UNet(
        parts["stem"],
        ResNetEncoder(widths=(8, 8, 8, 8), depths=(1, 1, 1, 1)),
        parts["head"],
        decoder_channels=(8, 8, 8),
        conditioning=parts["conditioning"],
    )


@pytest.mark.parametrize(
    ("student", "settings", "error", "message"),
    [
        (custom_student(stem=StridedVolume(PLANES, data_channels=2)), {}, IncompatibleModelError, "channels=1"),
        (custom_student(), {"fields": ("phase", "absorption")}, IncompatibleModelError, "one field of the model's"),
        (custom_student(head=SubPixelHead(2)), {"fields": ("phase", "density")}, IncompatibleModelError, "distinct"),
        (
            custom_student(conditioning=Bias({"field_position": 2})),
            {},
            IncompatibleModelError,
            "reads the conditions {'field_position': 2}",
        ),
        (custom_student(), {"conditions": (FieldPosition(),)}, IncompatibleModelError, "but the task gives"),
        (custom_student(), {"target_normalizer": FieldStandardize()}, IncompatibleModelError, "has no inverse"),
        (custom_student(), {"losses": {"a/b": L1()}}, ParameterError, "without '.' or '/'"),
        (custom_student(), {"losses": {"l1": (L1(), float("nan"))}}, ParameterError, "finite number"),
        (custom_student(), {"losses": {}}, ParameterError, "at least one loss"),
    ],
)
def test_a_task_whose_parts_do_not_fit_is_refused_with_what_to_change(student, settings, error, message):
    with pytest.raises(error, match=message):
        distillation(thin_phase_model(), student, pixel_size=PHASE_PIXEL_SIZE, **settings)


def test_prepare_is_the_model_prepare_and_pickles(phase_model):
    task = phase_task(phase_model)
    prepare = pickle.loads(pickle.dumps(task.prepare))
    recorded = torch.randint(100, 4000, (1, PLANES, 16, 16)).to(torch.uint16)
    expected = phase_model.prepare(recorded).to(torch.float32)
    assert torch.equal(prepare(recorded), expected)
    assert prepare(recorded).dtype == torch.float32


def test_a_step_on_recorded_pairs_returns_each_term_and_trains_the_student(phase_index):
    task = phase_task()
    data = phase_data(phase_index, task)
    task.fit_normalizers(data)
    result = task.step(first_batch(data))
    assert set(result.terms) == {"l1", "ssim"}
    assert set(result.scalars()) == {"loss", "loss/l1", "loss/ssim"}
    expected = result.terms["l1"] + 0.5 * result.terms["ssim"]
    assert float(result.loss.detach()) == pytest.approx(float(expected))
    assert not result.metrics
    result.loss.backward()
    assert all(parameter.grad is not None for parameter in task.student.parameters())

    task.eval()
    with torch.no_grad():
        validated = task.step(first_batch(data, "val"), stage="val")
    assert set(validated.metrics) == {"ssim", "dynamic_range"}
    assert validated.scored == 4
    assert -1 <= float(validated.metrics["ssim"]) <= 1
    assert task.example is not None  # the first validation batch: the test tensors of a bundle


def test_targets_of_another_channel_count_are_refused_naming_the_target_channels():
    task = phase_task()
    batch = Batch(
        data=torch.zeros(1, 1, PLANES, 32, 32),
        target=torch.zeros(1, 2, 1, 32, 32),
        condition={},
        position=["A/1/0"],
        origin=torch.zeros(1, 2),
    )
    with pytest.raises(ShapeError, match="target_channels must name one channel per field"):
        task.step(batch)
    with pytest.raises(ParameterError, match="stage"):
        task.step(batch, stage="test")  # ty: ignore[invalid-argument-type]


class Recording(Transform):
    def __init__(self) -> None:
        self.generators: list[torch.Generator] = []

    def __call__(self, batch: Batch, *, generator: torch.Generator) -> Batch:
        self.generators.append(generator)
        return batch

    def arguments(self) -> dict[str, object]:
        return {}


def test_training_steps_apply_the_transforms_with_the_given_generator():
    transform = Recording()
    task = phase_task(transforms=(transform,))
    batch = Batch(
        data=torch.randn(2, 1, PLANES, 32, 32),
        target=torch.randn(2, 1, 1, 32, 32),
        condition={},
        position=["A/1/0", "A/1/1"],
        origin=torch.zeros(2, 2),
    )
    generator = torch.Generator()
    task.step(batch, generator=generator)
    task.step(batch, stage="val")
    task.step(batch)
    assert transform.generators[0] is generator
    assert len(transform.generators) == 2  # not in validation
    assert transform.generators[1] is not generator


def test_fit_normalizers_fits_the_target_statistics_on_the_train_split(phase_index):
    task = phase_task()
    data = phase_data(phase_index, task)
    before = {name: value.clone() for name, value in task.target_normalizer.state_dict().items()}
    task.fit_normalizers(data)
    after = task.target_normalizer.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)
    normalized = first_batch(phase_data(phase_index, task)).target
    assert normalized is not None
    assert float(normalized.min()) > -3
    assert float(normalized.max()) < 3


def test_the_bundle_loads_as_a_solver_that_runs_in_the_model(phase_index, tmp_path):
    task = phase_task()
    data = phase_data(phase_index, task)
    task.fit_normalizers(data)
    with pytest.raises(ParameterError, match="needs test tensors"):
        task.bundle()
    in_memory = task.bundle(test=first_batch(data, "val"), record={"note": "test"})
    assert in_memory.test is not None
    assert in_memory.self_test() <= in_memory.test.tolerance
    bundle = load_bundle(save_bundle(in_memory, tmp_path / "bundle"))
    assert bundle.test is not None
    assert bundle.self_test() <= bundle.test.tolerance
    assert bundle.record == {"note": "test"}
    assert bundle.model["model"] == "Phase2D"
    assert bundle.contract == task.contract

    model = thin_phase_model()
    model.solver = Distilled(bundle)
    field = next(data.fields("val"))
    estimate = model.reconstruct(field.data, pixel_size=PHASE_PIXEL_SIZE, prepared=True).estimate
    assert estimate.shape == (2, 1, 64, 64)
    assert torch.count_nonzero(estimate[1]) == 0  # absorption: not a field of the contract
    direct = bundle.predict(field.data[None])[0]
    torch.testing.assert_close(estimate[:1], direct, atol=1e-4, rtol=1e-4)


def test_predict_and_solver_give_the_model_units(phase_index):
    task = phase_task()
    data = phase_data(phase_index, task)
    task.fit_normalizers(data)
    batch = first_batch(data, "val")
    task.eval()
    with torch.no_grad():
        normalized = task.student(batch.data)
    torch.testing.assert_close(task.predict(batch), task.target_normalizer.inverse(normalized))
    solver = task.as_solver()
    with torch.no_grad():
        next(task.student.parameters()).add_(1.0)  # the solver holds copies
    torch.testing.assert_close(
        solver.target_normalizer.inverse(solver.student(batch.data)), task.target_normalizer.inverse(normalized)
    )


def test_a_batch_always_has_a_target(phase_index):
    data = phase_data(phase_index, phase_task())
    for split in ("train", "val"):
        batch = first_batch(data, split)
        assert batch.target.shape == (4, 1, 1, 32, 32)
