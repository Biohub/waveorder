"""The data-driven object weight (wo-4.0 ``weight: {auto: {rule: ...}}``): rules, crop, apodization and the sweep."""

import dataclasses
import math

import pytest
import torch

from tests.nn.wo40_fixtures import GaussianTransfer, crop_volume, noisy_l_curve, otsu_fixture, sweep_volume
from waveorder.errors import CalibrationWarning
from waveorder.nn import Evidence, Free, Gaussian, ImagingModel, Sampling, Tikhonov, ZPadding, calibrate
from waveorder.nn.preprocess import apodize, most_structured_crop
from waveorder.nn.search import WeightSweep
from waveorder.nn.search.sweep import _slice_statistics, l_curve_corner, l_curve_curvature

# wo-4.0 autoreg.otsu_cnr_scores(otsu_fixture()), float32.
OLD_OTSU = [3.347858190536499, 3.4132089614868164, 3.385707139968872, 3.7188501358032227, 3.821417808532715,
            4.026024341583252, 5.29611349105835, 6.553287029266357, 7.2533135414123535]  # fmt: skip
# wo-4.0 autoreg.l_curve_curvature(*noisy_l_curve())[:6].
OLD_CURVATURE = [0.07537479303642164, 0.03999658312644922, 0.059593211427171215, 0.1904771334933425,
                 0.11915700419248529, 0.4474305068873722]  # fmt: skip
# wo-4.0 autoreg.select_regularization(sweep_volume(), gaussian_otf((12, 32, 32)), search_min=-4, search_max=1,
# num_samples=9, contrast="fluorescence", z_padding=2, apodization_rolloff=0.2), float32.
OLD_OTSU_SWEEP = {
    "index": 7,
    "scored_slice": 0,
    "scores": [3.780688762664795, 4.219932556152344, 3.9874427318573, 3.9649734497070312, 4.3324995040893555,
               5.504706859588623, 8.181089401245117, 8.350015640258789, 7.350679397583008],
}  # fmt: skip
OLD_L_CURVE_SWEEP = {
    "index": 2,
    "residual": [445.67242431640625, 465.9244079589844, 485.1563720703125, 506.76190185546875, 567.6475219726562,
                 962.74755859375, 2383.44677734375, 4464.96337890625, 5713.4453125],
    "solution": [13034.9267578125, 8476.9794921875, 6993.09765625, 6541.337890625, 6247.06982421875,
                 5593.68701171875, 3997.502197265625, 1836.3651123046875, 561.071044921875],
}  # fmt: skip


def test_the_fixtures_are_the_ones_wo40_scored() -> None:
    residual, solution = noisy_l_curve()
    assert float(otsu_fixture().double().sum()) == pytest.approx(64266.73559317742, rel=1e-12)
    assert (sum(residual), sum(solution)) == pytest.approx((138.33056305956794, 2898.802998058132), rel=1e-12)
    assert float(crop_volume().double().sum()) == pytest.approx(-236.795003288079, rel=1e-12)
    assert float(sweep_volume().double().sum()) == pytest.approx(458541.6957798004, rel=1e-12)


def test_otsu_scores_on_the_shared_slice_match_wo40() -> None:
    statistics = [_slice_statistics(volume) for volume in otsu_fixture()]
    spread = torch.stack([spread for _, spread in statistics])
    scored = int(spread.mean(dim=0).argmax())
    scores = torch.stack([cnr[scored] for cnr, _ in statistics])
    assert scores.tolist() == pytest.approx(OLD_OTSU, rel=1e-5)


def test_l_curve_curvature_and_corners_match_wo40() -> None:
    residual, solution = noisy_l_curve()
    assert l_curve_curvature(residual, solution)[:6] == pytest.approx(OLD_CURVATURE, rel=1e-12)
    assert l_curve_corner(residual, solution) == 12
    bent = [1 + 0.05 * i / 11 for i in range(12)] + [1.05 + (40 - 1.05) * i / 12 for i in range(13)]
    flat = [500 - 440 * i / 11 for i in range(12)] + [60 - 5 * i / 12 for i in range(13)]
    assert l_curve_corner(bent, flat) == 11  # a real L: the midpoint of its two turning points


def test_l_curve_corner_degenerate_inputs() -> None:
    assert l_curve_corner([1.0, 2.0], [3.0, 4.0]) == 0
    assert 0 <= l_curve_corner([1.0] * 9, [1.0] * 9) < 9


def test_the_crop_matches_wo40_and_reaches_the_far_edge() -> None:
    volume = crop_volume()
    assert most_structured_crop(volume, size=32) == (38, 48, 32)
    assert most_structured_crop(volume, size=64) == (6, 26, 64)
    assert most_structured_crop(volume, size=100) == (0, 20, 70)
    assert most_structured_crop(volume[:, :20, :20], size=32) == (0, 0, 20)
    edge = torch.zeros(1, 50, 50)
    edge[0, -10:, -10:] = torch.arange(100.0).reshape(10, 10)
    assert most_structured_crop(edge, size=16) == (34, 34, 16)


def test_apodization_keeps_low_frequencies_and_removes_nyquist() -> None:
    planes = torch.randn(2, 16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert apodize(planes, 0.0) is planes
    filtered = apodize(planes, 0.5)
    assert torch.allclose(filtered.mean(dim=(-2, -1)), planes.mean(dim=(-2, -1)))
    checkerboard = (-1.0) ** (torch.arange(16)[:, None] + torch.arange(16)[None, :]).double()
    assert apodize(checkerboard, 0.2).abs().max() < 1e-12


@pytest.mark.parametrize(("rule", "expected"), [("otsu_cnr", OLD_OTSU_SWEEP), ("l_curve", OLD_L_CURVE_SWEEP)])
def test_the_sweep_picks_what_wo40_picks(rule, expected) -> None:
    padding = ZPadding(2)
    data = padding.pad(sweep_volume()[None])
    sampling = padding.sampling(Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0)))
    model = ImagingModel(GaussianTransfer(), Tikhonov(1e-3), Gaussian(1.0))
    sweep = WeightSweep(rule=rule, search_min=-4, search_max=1, samples=9, rolloff=0.2, z_padding=2)
    result = sweep.sweep(data, model, sampling)
    assert result.index == expected["index"]
    assert result.transfer_peak == pytest.approx(1.0)
    assert result.weights == pytest.approx([10 ** (-4 + 5 * i / 8) for i in range(9)], rel=1e-12)
    assert result.weight == result.weights[result.index]
    if rule == "otsu_cnr":
        assert result.scored_slice == expected["scored_slice"]
        assert list(result.scores) == pytest.approx(expected["scores"], rel=1e-4)
    else:
        assert list(result.residual_norms) == pytest.approx(expected["residual"], rel=1e-5)
        assert list(result.solution_norms) == pytest.approx(expected["solution"], rel=1e-5)
    assert float(model.object_prior.quantity("weight")) == pytest.approx(1e-3)  # the model is not modified


def test_a_pick_on_the_edge_of_the_sweep_warns() -> None:
    data = sweep_volume()[None]
    model = ImagingModel(GaussianTransfer(), Tikhonov(1e-3), Gaussian(1.0))
    sweep = WeightSweep(rule="otsu_cnr", search_min=1, search_max=3, samples=5)  # picks 10^1, the lower end
    with pytest.warns(CalibrationWarning, match="end of the sweep"):
        result = sweep.sweep(data, model, Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0)))
    assert result.warnings


def test_as_a_search_stage_the_sweep_sets_the_weight_within_its_bounds() -> None:
    data = sweep_volume()[None]
    sampling = Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0))
    model = ImagingModel(GaussianTransfer(), Tikhonov(Free(1e-3, bounds=(1e-6, 1.0))), Gaussian(1.0))
    sweep = WeightSweep(search_min=-4, search_max=1, samples=9)
    picked = sweep.sweep(data, model, sampling).weight
    result = calibrate(data, model, sampling, score=Evidence(), search=[sweep])
    assert float(result.parameters["object_prior.weight"]) == pytest.approx(min(picked, 1.0))
    assert math.isfinite(float(result.objective))


@pytest.mark.parametrize("rule", ["otsu_cnr", "l_curve"])
def test_kept_slices_are_the_scored_slice_of_every_estimate_and_change_no_pick(rule) -> None:
    data = ZPadding(2).pad(sweep_volume()[None])
    sampling = ZPadding(2).sampling(Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0)))
    model = ImagingModel(GaussianTransfer(), Tikhonov(Free(1e-3, bounds=(1e-6, 1.0))), Gaussian(1.0))
    stage = WeightSweep(rule=rule, search_min=-4, search_max=1, samples=9, rolloff=0.2, z_padding=2)
    plain = stage.sweep(data, model, sampling)
    kept = dataclasses.replace(stage, keep_slices=True).sweep(data, model, sampling)
    assert (kept.index, kept.scores, kept.residual_norms) == (plain.index, plain.scores, plain.residual_norms)
    assert plain.slices is None
    assert kept.slices is not None
    assert kept.data_slice is not None
    assert kept.slices.shape == (9, 32, 32)
    assert (plain.scored_slice is None) == (rule == "l_curve")  # l_curve names a slice only to show it
    assert kept.scored_slice is not None
    torch.testing.assert_close(kept.data_slice, data[0, 2 + kept.scored_slice])
    assert torch.isfinite(kept.slices).all()
    assert not torch.equal(kept.slices[0], kept.slices[-1])
    result = calibrate(data, model, sampling, score=Evidence(), search=[stage], history=True)
    assert result.sweep is not None
    assert (result.sweep.weights, result.sweep.index) == (plain.weights, plain.index)
    assert calibrate(data, model, sampling, score=Evidence(), search=[stage]).sweep is None
