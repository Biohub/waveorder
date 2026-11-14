"""``RestorationNetwork.check_shapes``: the shape contract a network passes before it is trained."""

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn

from tests.nn.networks import tiny
from tests.nn.networks.tiny import tiny_unet
from waveorder import registry
from waveorder.errors import ParameterError, ShapeError
from waveorder.nn.networks import Bias, Case, RestorationNetwork

ARCHITECTURES = {
    "unet": (lambda: tiny.strided(5), (5,), (32, 32)),
    "sliding_planes": (lambda: tiny.sliding(3), (1, 3, 4), (32, 32)),
}
"""Per registered architecture: how to build a small instance, the plane counts it takes and a size it takes."""


def _built_in() -> list[str]:
    """The architectures waveorder registers; a plugin registered by another test module is left out."""
    names = registry.names("network")
    return [name for name in names if registry._classes[name][1].__module__.startswith("waveorder.")]  # noqa: SLF001


@pytest.mark.parametrize("name", _built_in())
def test_every_registered_architecture_passes_check_shapes(name):
    assert name in ARCHITECTURES, f"add a small {name} to ARCHITECTURES: every registered architecture is checked"
    build, planes, size = ARCHITECTURES[name]
    network = build()
    assert next(iter(registry.describe(network))) == name
    cases = network.check_shapes(planes=planes, sizes=[size])
    assert len(cases) == len(planes) * 3 * 2  # three sizes (the given one, a taller, a wider) and two batch sizes


class Plane(RestorationNetwork):
    """A one-convolution student of stride 2 that takes exactly ``planes`` planes, with switches that break it."""

    def __init__(self, planes: int = 3, *, fault: str = "") -> None:
        super().__init__()
        self.channels, self.fields, self.condition_sizes = 1, 1, {}
        self.planes, self.fault = planes, fault
        self.conv = nn.Conv2d(planes, 1 if fault != "extra_weight" else 2, 3, padding=1)
        if fault == "unsaved_setting":
            self.extra = nn.Linear(1, 1)

    @property
    def stride(self) -> int:
        return 2

    def halo(self) -> int:
        return 1

    def planes_out(self, planes_in: int) -> int:
        if planes_in != self.planes:
            raise ShapeError(f"Plane takes {self.planes} planes, got {planes_in}")
        return 1

    def arguments(self) -> dict[str, object]:
        if self.fault == "not_json":
            return {"planes": torch.tensor(self.planes)}
        return {"planes": self.planes}

    def forward(self, data: Tensor, condition: Mapping[str, Tensor] | None = None) -> Tensor:
        del condition
        if data.shape[2] != self.planes:
            if self.fault == "wrong_error":
                raise RuntimeError("bad planes")
            if self.fault != "accepts_any":
                raise ShapeError(f"Plane takes {self.planes} planes, got {data.shape[2]}")
            data = data[:, :, : self.planes] if data.shape[2] > self.planes else data.repeat(1, 1, self.planes, 1, 1)
        image = self.conv(data[:, 0, : self.planes])[:, :1]
        if self.fault == "batch_statistics":
            image = image - image.mean()
        if self.fault == "transposed" and image.shape[-1] != image.shape[-2]:
            image = image.transpose(-1, -2)
        if self.fault == "nan":
            image = image / torch.zeros(())
        return image[:, :, None]


def test_a_unet_passes_and_the_cases_cover_planes_batches_and_derived_sizes():
    student = tiny_unet()
    cases = student.check_shapes(planes=(3,), sizes=((32, 32),))
    stride = student.stride
    assert {case.size for case in cases} == {(32, 32), (32 + stride, 32), (32, 32 + 2 * stride)}
    assert {case.batch for case in cases} == {1, 2}
    assert all(isinstance(case, Case) and case.planes == 3 for case in cases)
    assert len(cases) == 6


def test_a_student_with_conditions_gets_one_vector_per_sample():
    student = tiny_unet(conditioning=Bias({"field_position": 2}))
    assert student.check_shapes(planes=(3,), sizes=((32, 32),), batch_sizes=(2,))
    given = {"field_position": torch.tensor([0.25, 0.75])}
    assert student.check_shapes(planes=(3,), sizes=((32, 32),), batch_sizes=(1,), conditions=given)
    with pytest.raises(ParameterError, match="one \\(K,\\) tensor for each condition"):
        student.check_shapes(planes=(3,), sizes=((32, 32),), conditions={"field_position": torch.zeros(3)})


def test_the_student_is_left_in_its_mode():
    student = tiny_unet().train()
    student.check_shapes(planes=(3,), sizes=((32, 32),), batch_sizes=(1,))
    assert student.training


class Rebuilt(Plane):  # arguments() leaves the fault out, as a student that forgets a setting would
    pass


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("batch_statistics", "another output alone than in its batch"),
        ("transposed", r"the output is \(1, 1, 1, 8, 10\), expected .* \(1, 1, 1, 10, 8\)"),
        ("nan", "non-finite values"),
        ("not_json", "are not JSON"),
        ("extra_weight", r"another state than the network"),
        ("accepts_any", "the forward pass accepted 1 planes"),
        ("wrong_error", "the forward pass raised RuntimeError"),
    ],
)
def test_a_broken_student_fails_with_the_case_named(fault, message):
    student = Rebuilt(3, fault=fault)
    with pytest.raises(AssertionError, match=message):
        student.check_shapes(planes=(3,), sizes=((8, 8),))


def test_batch_norm_in_training_statistics_is_caught_only_through_eval_mode():
    # In eval mode batch norm uses running statistics, so a batch-norm network given in train mode passes.
    assert tiny_unet().train().check_shapes(planes=(3,), sizes=((32, 32),), batch_sizes=(2,))


def test_a_plane_count_the_student_refuses_may_not_be_required():
    with pytest.raises(AssertionError, match=r"planes_out\(5\) raises ShapeError, but 5 is one of the plane counts"):
        Plane(3).check_shapes(planes=(3, 5), sizes=((8, 8),))


def test_sizes_must_be_multiples_of_the_stride():
    with pytest.raises(ParameterError, match="not a positive multiple of the network's stride 32"):
        tiny_unet().check_shapes(planes=(3,), sizes=((40, 32),))
    with pytest.raises(ParameterError, match="at least one"):
        tiny_unet().check_shapes(planes=(3,), sizes=())
