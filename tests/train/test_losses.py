import pytest
import torch
from torch import Tensor

from waveorder.data import Batch
from waveorder.errors import ParameterError, ShapeError
from waveorder.train import L1, MSE, SSIM


def batch(data: Tensor, target: Tensor) -> Batch:
    count = data.shape[0]
    return Batch(
        data=data,
        target=target,
        condition={},
        position=[f"A/1/{n}" for n in range(count)],
        origin=torch.zeros(count, 2),
    )


def smooth(*shape: int, seed: int = 0) -> Tensor:
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    return torch.nn.functional.avg_pool2d(noise.flatten(0, -3), 5, stride=1, padding=2).reshape(shape)


def test_l1_and_mse_are_the_mean_errors():
    target = torch.zeros(2, 1, 1, 4, 4)
    estimate = torch.full_like(target, 2.0)
    assert float(L1()(estimate, batch(torch.zeros(2, 1, 3, 4, 4), target))) == pytest.approx(2.0)
    assert float(MSE()(estimate, batch(torch.zeros(2, 1, 3, 4, 4), target))) == pytest.approx(4.0)


def test_a_target_loss_refuses_a_target_of_another_shape():
    estimate = torch.zeros(1, 1, 1, 4, 4)
    with pytest.raises(ShapeError, match=r"batch.target has \(1, 2, 1, 4, 4\)"):
        MSE()(estimate, batch(torch.zeros(1, 1, 3, 4, 4), torch.zeros(1, 2, 1, 4, 4)))


def test_ssim_is_zero_for_the_target_and_grows_with_the_error():
    target = smooth(2, 1, 1, 32, 32)
    data = torch.zeros(2, 1, 3, 32, 32)
    assert float(SSIM()(target.clone(), batch(data, target))) == pytest.approx(0.0, abs=1e-5)
    noisy = target + 0.5 * target.std() * torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
    assert 0.05 < float(SSIM()(noisy, batch(data, target))) < 1.0


def test_ssim_leaves_blank_target_planes_out():
    target = torch.stack([smooth(1, 1, 32, 32), torch.full((1, 1, 32, 32), 3.0)])  # the second is blank
    estimate = target + 0.1 * torch.randn(target.shape, generator=torch.Generator().manual_seed(2))
    both = SSIM()(estimate, batch(torch.zeros(2, 1, 3, 32, 32), target))
    first = SSIM()(estimate[:1], batch(torch.zeros(1, 1, 3, 32, 32), target[:1]))
    assert float(both) == pytest.approx(float(first))

    blank = torch.ones(2, 1, 1, 32, 32, requires_grad=True)
    value = SSIM()(blank, batch(torch.zeros(2, 1, 3, 32, 32), torch.ones(2, 1, 1, 32, 32)))
    value.backward()
    assert float(value.detach()) == 0.0
    assert blank.grad is not None
    assert float(blank.grad.abs().max()) == 0.0


@pytest.mark.parametrize(
    ("arguments", "message"), [({"window": 1}, "window"), ({"window": 7, "min_range": 0.0}, "min_range")]
)
def test_ssim_refuses_bad_settings(arguments, message):
    with pytest.raises(ParameterError, match=message):
        SSIM(**arguments)
