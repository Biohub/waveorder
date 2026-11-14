import math

import pytest
import torch

from waveorder.data import Batch, Flip, FocusShift, PlaneCrop
from waveorder.errors import IncompatibleModelError, ParameterError
from waveorder.nn.networks import FieldPosition

OFFSETS = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0, math.nan])


def _batch(*, planes: int = 5, target_planes: int = 1, condition: dict[str, torch.Tensor] | None = None) -> Batch:
    data = torch.arange(4 * 1 * planes * 6 * 8, dtype=torch.float32).reshape(4, 1, planes, 6, 8)
    target = torch.arange(4 * 2 * target_planes * 6 * 8, dtype=torch.float32).reshape(4, 2, target_planes, 6, 8)
    return Batch(
        data=data,
        target=target,
        condition=condition if condition is not None else {"plane_offsets": OFFSETS.expand(4, -1).clone()},
        position=[f"p/A/1/{n}" for n in range(4)],
        origin=torch.zeros(4, 2, dtype=torch.int64),
    )


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _target(batch: Batch) -> torch.Tensor:
    assert batch.target is not None
    return batch.target


def test_plane_crop_keeps_a_run_and_its_offsets():
    batch = _batch()
    for seed in range(20):
        cropped = PlaneCrop(2, probability=1.0)(batch, generator=_generator(seed))
        kept = cropped.data.shape[2]
        assert 2 <= kept <= 5
        offsets = cropped.condition["plane_offsets"][0]
        start = int((offsets[0] + 4) / 2)
        assert torch.equal(cropped.data, batch.data[:, :, start : start + kept])
        assert torch.equal(offsets[:kept], OFFSETS[start : start + kept])
        assert offsets[kept:].isnan().all()
        assert torch.equal(_target(cropped), _target(batch))  # a one-plane target is left as it is


def test_plane_crop_pads_and_crops_a_volume_target():
    batch = _batch(target_planes=5)
    cropped = PlaneCrop(2, probability=1.0, pad=True)(batch, generator=_generator(1))
    assert cropped.data.shape == batch.data.shape
    assert _target(cropped).shape == _target(batch).shape
    kept = int((~cropped.condition["plane_offsets"][0].isnan()).sum())
    assert (cropped.data[:, :, kept:] == 0).all()
    assert (_target(cropped)[:, :, kept:] == 0).all()
    assert PlaneCrop(2, probability=0.0)(batch, generator=_generator()) is batch
    with pytest.raises(ParameterError):
        PlaneCrop(0)


def test_focus_shift_moves_each_sample_within_the_range():
    batch = _batch()
    shifted = FocusShift(1.5)(batch, generator=_generator())
    offsets = shifted.condition["plane_offsets"]
    shift = offsets[:, 0] - OFFSETS[0]
    assert (shift.abs() <= 1.5).all()
    assert len(set(shift.tolist())) == 4
    assert torch.allclose(offsets[:, :5], OFFSETS[:5] + shift[:, None], atol=1e-6)
    assert offsets[:, 5].isnan().all()
    assert shifted.data is batch.data
    with pytest.raises(IncompatibleModelError, match="plane_offsets"):
        FocusShift(1.0)(_batch(condition={}), generator=_generator())


def test_flip_mirrors_data_target_and_field_position():
    position = torch.tensor([[0.25, 0.75]]).expand(4, -1).clone()
    batch = _batch(condition={"field_position": position, "well": torch.ones(4, 2)})
    flipped = Flip(("x",), probability=1.0, invariant=["well"])(batch, generator=_generator())
    assert torch.equal(flipped.data, batch.data.flip(-1))
    assert torch.equal(_target(flipped), _target(batch).flip(-1))
    assert torch.equal(flipped.condition["field_position"], torch.tensor([[0.25, 0.25]]).expand(4, -1))
    assert torch.equal(flipped.condition["well"], batch.condition["well"])
    some = Flip(probability=0.5)(_batch(), generator=_generator(2))
    each = [
        next(
            (dims for dims in ((), (-2,), (-1,), (-2, -1)) if torch.equal(some.data[n], _batch().data[n].flip(dims))),
            None,
        )
        for n in range(4)
    ]
    assert None not in each
    assert len(set(each)) > 1
    with pytest.raises(IncompatibleModelError, match="well"):
        Flip()(batch, generator=_generator())


def test_flip_mirrors_a_field_position_in_its_own_convention():
    convention = FieldPosition(axes="xy", offset=0.5, scale=0.4375, name="where")
    values = torch.tensor([[0.8, -0.2]]).expand(4, -1).clone()  # (x, y), centred and scaled
    batch = _batch(condition={"where": values})
    flipped = Flip(("x",), probability=1.0, position=convention)(batch, generator=_generator())
    assert torch.allclose(flipped.condition["where"], torch.tensor([[-0.8, -0.2]]).expand(4, -1))
    flipped = Flip(("y",), probability=1.0, position=convention)(batch, generator=_generator())
    assert torch.allclose(flipped.condition["where"], torch.tensor([[0.8, 0.2]]).expand(4, -1))


def test_transforms_rebuild_from_their_arguments():
    flip = Flip(("y",), invariant=["well"], position=FieldPosition(axes="xy", offset=0.5))
    for transform in (PlaneCrop(3, 0.25, pad=True), FocusShift(2.0), flip):
        rebuilt = type(transform)(**transform.arguments())  # ty: ignore[invalid-argument-type]
        assert rebuilt.arguments() == transform.arguments()
        assert repr(rebuilt) == repr(transform)
