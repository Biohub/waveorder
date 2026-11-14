import math

import pytest
import torch
from torch import Tensor

from waveorder.errors import IncompatibleModelError, ParameterError, ShapeError
from waveorder.models import Fluorescence2D
from waveorder.nn import Detection, ImagingSystem, LinearOperator, Sampling
from waveorder.train.metrics import (
    dynamic_range_ratio,
    passband_error,
    plane_ssim,
    ssim,
    summarize,
    transfer_support,
)


def images(batch: int = 3, *, fields: int = 1, planes: int = 1, size: tuple[int, int] = (32, 40), seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((batch, fields, planes, *size), generator=generator)


def reference_ssim(estimate: Tensor, target: Tensor, window: int) -> float:
    """SSIM of two (Y, X) images written from its definition, one window at a time."""
    span = float(target.max() - target.min())
    c1, c2 = (0.01 * span) ** 2, (0.03 * span) ** 2
    values = []
    for y in range(target.shape[0] - window + 1):
        for x in range(target.shape[1] - window + 1):
            a = estimate[y : y + window, x : x + window].double().flatten()
            b = target[y : y + window, x : x + window].double().flatten()
            covariance = float(torch.cov(torch.stack((a, b)))[0, 1])
            luminance = (2 * a.mean() * b.mean() + c1) / (a.mean() ** 2 + b.mean() ** 2 + c1)
            structure = (2 * covariance + c2) / (a.var() + b.var() + c2)
            values.append(float(luminance * structure))
    return sum(values) / len(values)


def test_ssim_matches_the_definition_with_the_targets_range():
    target = images(1, size=(12, 14))
    estimate = target + 0.2 * images(1, size=(12, 14), seed=1)
    expected = reference_ssim(estimate[0, 0, 0], target[0, 0, 0], window=7)
    assert float(ssim(estimate, target, window=7)) == pytest.approx(expected, abs=1e-5)


def test_ssim_is_one_for_a_match_and_falls_with_noise():
    target = images()
    noise = images(seed=1) - 0.5
    assert torch.allclose(ssim(target, target), torch.ones(3), atol=1e-6)
    light, heavy = ssim(target + 0.05 * noise, target), ssim(target + 0.5 * noise, target)
    assert (heavy < light).all()
    assert (light < 1).all()


def test_ssim_does_not_change_when_both_images_are_scaled():
    target = images()
    estimate = target + 0.1 * images(seed=1)
    assert torch.allclose(ssim(1e3 * estimate, 1e3 * target), ssim(estimate, target), atol=1e-5)


def test_ssim_of_a_sample_does_not_depend_on_its_batch_mates():
    target = images()
    estimate = target + 0.1 * images(seed=1)
    assert torch.allclose(ssim(estimate, target)[1], ssim(estimate[1:2], target[1:2])[0], atol=1e-6)


def test_blank_target_planes_are_not_scored():
    target = images(2, planes=2)
    target[0] = 0.25  # a blank sample
    target[1, :, 1] = 1e-6 * target[1, :, 1]  # one blank plane of the other
    estimate = target + 0.1 * images(2, planes=2, seed=1)
    planes = plane_ssim(estimate, target)
    assert torch.isnan(planes[0]).all()
    assert torch.isnan(planes[1, 0, 1])
    values = ssim(estimate, target)
    assert torch.isnan(values[0])
    assert float(values[1]) == pytest.approx(float(planes[1, 0, 0]))


def test_ssim_gives_finite_gradients_with_blank_planes_in_the_batch():
    target = images(2)
    target[0] = 0.0
    estimate = (target + 0.1 * images(2, seed=1)).requires_grad_()
    ssim(estimate, target).nansum().backward()
    assert torch.isfinite(estimate.grad).all()
    assert estimate.grad[0].abs().sum() == 0
    assert estimate.grad[1].abs().sum() > 0


def test_ssim_refuses_a_window_larger_than_the_plane():
    with pytest.raises(ShapeError, match="window of 11 pixels does not fit"):
        ssim(images(size=(8, 8)), images(size=(8, 8)))


def test_metrics_refuse_shapes_that_differ():
    with pytest.raises(ShapeError, match=r"\(B, F, Zout, Y, X\)"):
        ssim(images(2), images(3))
    with pytest.raises(ShapeError, match=r"\(B, F, Zout, Y, X\)"):
        dynamic_range_ratio(images()[0], images()[0])


def test_dynamic_range_ratio_is_the_ratio_of_percentile_ranges():
    target = images(2, fields=2, planes=3)
    estimate = torch.stack((0.5 * target[0] + 3.0, 2.0 * target[1]))
    assert torch.allclose(dynamic_range_ratio(estimate, target), torch.tensor([0.5, 2.0]), atol=1e-5)


def test_dynamic_range_ratio_matches_torch_quantile():
    target, estimate = images(1), images(1, seed=1) * 0.7
    low, high = torch.quantile(target.flatten(), torch.tensor([0.05, 0.95]))
    estimate_low, estimate_high = torch.quantile(estimate.flatten(), torch.tensor([0.05, 0.95]))
    expected = float((estimate_high - estimate_low) / (high - low))
    assert float(dynamic_range_ratio(estimate, target, low=5, high=95)) == pytest.approx(expected, rel=1e-5)


def test_dynamic_range_ratio_of_a_flat_target_is_nan():
    assert torch.isnan(dynamic_range_ratio(images(1), torch.ones(1, 1, 1, 32, 40)))
    with pytest.raises(ParameterError, match="low < high"):
        dynamic_range_ratio(images(1), images(1), low=99, high=1)


@pytest.mark.parametrize("width", [32, 33])
def test_passband_error_splits_the_normalized_squared_error(width):
    target = images(2, fields=2, size=(16, width))
    estimate = target + 0.3 * images(2, fields=2, size=(16, width), seed=1)
    support = torch.zeros(2, 1, 16, width // 2 + 1, dtype=torch.bool)
    support[:, :, :4, :5] = True
    error = passband_error(estimate, target, support)
    total = (estimate - target).square().sum(dim=(1, 2, 3, 4)) / target.square().sum(dim=(1, 2, 3, 4))
    assert torch.allclose(error["inside"] + error["outside"], total, rtol=1e-4)
    assert (error["inside"] > 0).all()
    assert (error["outside"] > 0).all()


def test_passband_error_puts_an_error_wave_on_its_side_of_the_support():
    target = images(1, size=(32, 32))
    y = torch.arange(32.0)[:, None].expand(32, 32)
    support = torch.zeros(1, 1, 32, 17, dtype=torch.bool)
    support[:, :, :6] = True
    support[:, :, -5:] = True  # |fy| <= 5 cycles per field
    low = passband_error(target + 0.1 * torch.cos(2 * math.pi * 3 * y / 32), target, support)
    high = passband_error(target + 0.1 * torch.cos(2 * math.pi * 12 * y / 32), target, support)
    assert float(low["outside"]) == pytest.approx(0.0, abs=1e-8)
    assert float(low["inside"]) > 1e-3
    assert float(high["inside"]) == pytest.approx(0.0, abs=1e-8)
    assert float(high["outside"]) == pytest.approx(float(low["inside"]), rel=1e-4)


def test_passband_error_refuses_a_support_of_another_shape():
    with pytest.raises(ShapeError, match="transfer_support"):
        passband_error(images(1), images(1), torch.ones(1, 1, 32, 40, dtype=torch.bool))
    with pytest.raises(ShapeError, match="boolean mask"):
        passband_error(images(1), images(1), torch.ones(1, 1, 32, 21))


def test_transfer_support_is_the_passband_of_the_teacher():
    model = Fluorescence2D(Detection(wavelength=0.5, numerical_aperture=0.5), index_of_refraction_media=1.0)
    sampling = Sampling(shape=(3, 64, 64), pixel_size=(1.0, 0.1, 0.1))
    support = transfer_support(model.system, sampling)
    assert support.shape == (1, 1, 64, 33)
    assert support.dtype == torch.bool
    cutoff = 2 * 0.5 / 0.5  # 2 NA / wavelength, cycles per um
    fy = torch.fft.fftfreq(64, d=0.1)[:, None]
    fx = torch.fft.rfftfreq(64, d=0.1)[None, :]
    radius = torch.sqrt(fy**2 + fx**2)
    assert support[0, 0][radius < 0.5 * cutoff].all()
    assert not support[0, 0][radius > 1.05 * cutoff].any()


def test_transfer_support_fits_passband_error():
    model = Fluorescence2D(Detection(wavelength=0.5, numerical_aperture=0.5), index_of_refraction_media=1.0)
    sampling = Sampling(shape=(3, 32, 32), pixel_size=(1.0, 0.1, 0.1))
    data = images(1, planes=3, size=(32, 32))[0]
    teacher = model.reconstruct(data, pixel_size=sampling.pixel_size).estimate[None]
    error = passband_error(
        teacher + 0.1 * images(1, size=(32, 32), seed=1), teacher, transfer_support(model.system, sampling)
    )
    assert float(error["outside"]) > 0  # white noise has energy the transfer function does not pass
    assert float(error["inside"]) > 0


class _Weighting(LinearOperator):
    """The object times a weight that varies along x: no transfer function."""

    def __init__(self, sampling: Sampling) -> None:
        shape = (1, *sampling.shape)
        super().__init__(sampling, object_shape=shape, data_shape=shape)
        self.weights = torch.linspace(0.5, 1.0, sampling.shape[-1])

    @property
    def placement(self) -> tuple[torch.dtype, torch.device]:
        return self.weights.dtype, self.weights.device

    def apply(self, obj: Tensor) -> Tensor:
        return self.weights * obj

    def adjoint(self, data: Tensor) -> Tensor:
        return self.weights * data

    def norm(self) -> Tensor:
        return self.weights.max()


class Vignetting(ImagingSystem):
    """A shift-variant system: the object times a weight that varies over the field."""

    def __init__(self) -> None:
        super().__init__(fields=("density",), channel_count=1)

    def compute_operator(self, sampling: Sampling) -> LinearOperator:
        return _Weighting(sampling)


def test_transfer_support_refuses_a_shift_variant_system():
    with pytest.raises(IncompatibleModelError, match="shift-invariant"):
        transfer_support(Vignetting(), Sampling(shape=(1, 16, 16), pixel_size=(1.0, 0.1, 0.1)))


def test_summarize_leaves_out_samples_that_could_not_be_scored():
    values = torch.tensor([float("nan"), 0.0, 1.0, 2.0, 3.0, 4.0, float("inf")])
    summary = summarize(values)
    assert summary == pytest.approx({"mean": 2.0, "p10": 0.4, "p50": 2.0, "p90": 3.6})
    empty = summarize(torch.tensor([float("nan")]))
    assert set(empty) == {"mean", "p10", "p50", "p90"}
    assert all(math.isnan(value) for value in empty.values())
