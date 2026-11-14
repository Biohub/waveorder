"""Students: the shapes they take and return, the refusals, and the rebuild by name a bundle relies on."""

import json

import pytest
import torch
from torch import Tensor

from tests.nn.networks import tiny
from tests.registration import temporarily
from waveorder import registry
from waveorder.errors import ParameterError, RegistrationError, ShapeError
from waveorder.nn.networks import (
    Bias,
    ConcatPlanes,
    PlaneHead,
    PlaneProjection,
    RestorationNetwork,
    SlidingPlanes,
    StridedVolume,
    SubPixelHead,
    UNet,
)


def _data(*shape: int, seed: int = 0) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _offsets(batch: int, planes: int) -> Tensor:
    return torch.linspace(-2.0, 2.0, planes).expand(batch, planes).clone()


LAYOUTS = {
    "strided": (tiny.strided, 5, {}),
    "projected": (tiny.projected, 3, {}),
    "defocus": (tiny.defocus, 5, {"plane_offsets": _offsets(2, 5)}),
}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("conditioning", [None, ConcatPlanes, Bias])
def test_every_layout_returns_one_plane_of_the_datas_size(layout, conditioning):
    build, planes, condition = LAYOUTS[layout]
    student = build(planes, conditioning=None if conditioning is None else conditioning(sizes={"where": 2}))
    condition = {**condition, **({} if conditioning is None else {"where": torch.rand(2, 2)})}
    assert student.stride == 32
    assert student.planes_out(planes) == 1
    assert student.condition_sizes == {name: value.shape[1] for name, value in condition.items()}
    with torch.no_grad():
        estimate = student.eval()(_data(2, 1, planes, 64, 96), condition)
    assert estimate.shape == (2, 1, 1, 64, 96)
    assert torch.isfinite(estimate).all()


def test_a_shape_the_network_does_not_take_is_refused_with_what_it_takes():
    student = tiny.strided(5).eval()
    with pytest.raises(ShapeError, match="multiples of its stride 32"):
        student(_data(1, 1, 5, 64, 48))
    with pytest.raises(ShapeError, match="StridedVolume takes stacks of 5 planes, got 4"):
        student.planes_out(4)
    with pytest.raises(ShapeError, match="StridedVolume takes stacks of 5 planes, got 4"):
        student(_data(1, 1, 4, 64, 64))
    with pytest.raises(ShapeError, match=r"expected \(B, 1, Z, Y, X\)"):
        student(_data(1, 2, 5, 64, 64))
    conditioned = tiny.strided(5, conditioning=Bias(sizes={"where": 2})).eval()
    with pytest.raises(ShapeError, match=r"reads the condition 'where' \(2 values per item\), but none was given"):
        conditioned(_data(1, 1, 5, 64, 64))
    with pytest.raises(ShapeError, match=r"the condition 'where' has shape \(1, 3\), expected \(1, 2\)"):
        conditioned(_data(1, 1, 5, 64, 64), {"where": torch.zeros(1, 3)})


def test_the_defocus_stem_takes_any_plane_count_up_to_its_size():
    student = tiny.defocus(5).eval()
    offsets = _offsets(1, 5)
    offsets[:, 3:] = float("nan")
    with torch.no_grad():
        three = student(_data(1, 1, 3, 64, 64), {"plane_offsets": offsets})
        padded = student(
            torch.cat([_data(1, 1, 3, 64, 64), torch.zeros(1, 1, 2, 64, 64)], 2), {"plane_offsets": offsets}
        )
    torch.testing.assert_close(three, padded)  # zero planes with NaN defocus are padding
    with pytest.raises(ShapeError, match="1 to 5 planes, got 6"):
        student.planes_out(6)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_network_is_rebuilt_from_its_description_by_name(layout):
    build, planes, condition = LAYOUTS[layout]
    student = build(planes, conditioning=ConcatPlanes(sizes=[("b", 1), ("a", 2)])).eval()
    description = json.loads(json.dumps(registry.describe(student)))
    assert list(description) == ["unet"]
    rebuilt = registry.resolve(description, "network")
    assert type(rebuilt) is UNet
    assert registry.arguments(rebuilt) == registry.arguments(student)
    rebuilt.load_state_dict(student.state_dict())
    condition = {**condition, "a": torch.rand(2, 2), "b": torch.rand(2, 1)}
    data = _data(2, 1, planes, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt.eval()(data, condition), student(data, condition))


def test_the_order_of_the_conditions_is_part_of_the_network():
    student = tiny.strided(5, conditioning=ConcatPlanes(sizes={"a": 1, "b": 1})).eval()
    assert registry.arguments(student)["conditioning"] == {"concat_planes": {"sizes": [["a", 1], ["b", 1]]}}
    data, low, high = _data(1, 1, 5, 64, 64), torch.zeros(1, 1), torch.ones(1, 1)
    with torch.no_grad():
        assert not torch.allclose(student(data, {"a": low, "b": high}), student(data, {"a": high, "b": low}))


def test_parts_that_do_not_fit_together_are_refused_with_the_reason():
    projection = PlaneProjection(planes=5, channels=3)
    with pytest.raises(ParameterError, match="use a PlaneHead with a stem of stride 1 and a SubPixelHead"):
        UNet(projection, tiny.encoder(), SubPixelHead(), decoder_channels=(32, 16, 8))
    with pytest.raises(ParameterError, match="decoder_channels has 2 widths, but the decoder has 3 blocks"):
        UNet(StridedVolume(planes=5), tiny.encoder(), SubPixelHead(), decoder_channels=(8, 8))
    # A strided stem may also feed a decoder that runs up to full resolution: two more blocks.
    assert len(UNet(StridedVolume(planes=5), tiny.encoder(), PlaneHead(), decoder_channels=(8,) * 5).decoder) == 5
    with pytest.raises(ParameterError, match="full_resolution_channels needs a stem of stride 1"):
        UNet(
            StridedVolume(planes=5),
            tiny.encoder(),
            SubPixelHead(),
            decoder_channels=(32, 16, 8),
            full_resolution_channels=8,
        )
    with pytest.raises(ParameterError, match="norm must be 'batch' or 'group'"):
        UNet(projection, tiny.encoder(), PlaneHead(), decoder_channels=(8,), norm="layer")  # ty: ignore[invalid-argument-type]


def test_with_group_norm_no_estimate_depends_on_its_batch_mates():
    student = tiny.strided(5).train()  # training mode: a batch norm would read the batch
    data = _data(3, 1, 5, 64, 64)
    alone = student(data[:1])
    together = student(data)[:1]
    torch.testing.assert_close(alone, together)


def test_sliding_planes_returns_one_plane_per_data_plane_from_the_window_around_it():
    volume = tiny.sliding(3).eval()
    plane = volume.student
    data = _data(2, 1, 4, 64, 64)
    with torch.no_grad():
        estimate = volume(data)
        assert estimate.shape == (2, 1, 4, 64, 64)
        torch.testing.assert_close(estimate[:, :, 1:2], plane(data[:, :, 0:3]))
        # At the ends the first and last planes repeat.
        torch.testing.assert_close(estimate[:, :, 0:1], plane(data[:, :, [0, 0, 1]]))
        torch.testing.assert_close(SlidingPlanes(plane, planes=3, chunk=3).eval()(data), estimate)
    assert (volume.halo_planes(), volume.halo(), volume.stride) == (1, plane.halo(), plane.stride)
    assert volume.planes_out(7) == 7
    with pytest.raises(ParameterError, match="positive odd int"):
        SlidingPlanes(plane, planes=2)
    with pytest.raises(ParameterError, match="SlidingPlanes planes = 5"):
        SlidingPlanes(plane, planes=5)
    with pytest.raises(ParameterError, match="returns one plane"):
        SlidingPlanes(volume, planes=3)


class Doubling(RestorationNetwork):
    """A plugin's student: twice its data, as one field; registering it is all a bundle needs to rebuild it."""

    channels, fields, condition_sizes = 1, 1, {}

    def __init__(self, scale: float = 2.0) -> None:
        super().__init__()
        self.scale = scale

    @property
    def stride(self) -> int:
        return 1

    def halo(self) -> int:
        return 0

    def planes_out(self, planes_in: int) -> int:
        return planes_in

    def forward(self, data: Tensor, condition=None) -> Tensor:
        del condition
        return self.scale * data


def test_a_plugin_registers_its_student_and_the_registry_rebuilds_it():
    with temporarily("doubling", Doubling, kind="network"):
        assert "doubling" in registry.names("network")
        rebuilt = registry.resolve(registry.describe(Doubling(3.0)), "network", RestorationNetwork)
        assert isinstance(rebuilt, Doubling)
        assert rebuilt.scale == 3.0
        with pytest.raises(RegistrationError, match="is taken by Doubling"):
            registry.register("doubling", kind="network")(UNet)
        with pytest.raises(ParameterError, match="no network is named 'missing'; choose one of"):
            registry.resolve("missing", "network")
        with pytest.raises(ParameterError, match="does not take these arguments"):
            registry.resolve({"doubling": {"width": 2}}, "network")

    class Unregistered(UNet):
        pass

    with pytest.raises(ParameterError, match="Unregistered is not registered"):
        registry.describe(Unregistered(StridedVolume(5), tiny.encoder(), SubPixelHead(), decoder_channels=(8, 8, 8)))
    assert {"unet", "sliding_planes"} <= set(registry.names("network"))
