"""The weight sweep's JSON report and PNG montage (wo-4.0 ``report_path`` and ``plot_path``)."""

import dataclasses
import json

import pytest
from PIL import Image, ImageColor

from tests.nn.wo40_fixtures import GaussianTransfer, sweep_volume
from waveorder.io.sweep_report import save_sweep_montage, save_sweep_report
from waveorder.nn import Gaussian, ImagingModel, Sampling, Tikhonov
from waveorder.nn.search import WeightSweep
from waveorder.nn.search.base import SweepRule

# wo-4.0 AutoRegResult.to_dict keys.
OLD_KEYS = [
    "rule", "regularization_strength", "index", "lambda_over_h2max", "transfer_function_peak_squared", "crop",
    "scored_z_index", "regularization_strengths", "scores", "residual_norms", "solution_norms", "warnings",
]  # fmt: skip


def _sweep(rule: SweepRule, *, keep_slices: bool = True):
    model = ImagingModel(GaussianTransfer(), Tikhonov(1e-3), Gaussian(1.0))
    sweep = WeightSweep(rule=rule, search_min=-4, search_max=1, samples=9, keep_slices=keep_slices)
    result = sweep.sweep(sweep_volume()[None], model, Sampling(shape=(8, 32, 32), pixel_size=(1.0, 1.0, 1.0)))
    return dataclasses.replace(result, region=(3, 5, 32))


@pytest.mark.parametrize("rule", ["otsu_cnr", "l_curve"])
def test_the_report_has_wo40s_fields(tmp_path, rule):
    result = _sweep(rule)
    report = json.loads(save_sweep_report(result, tmp_path / "nested" / "sweep.json").read_text())
    assert list(report) == OLD_KEYS
    assert report["rule"] == rule
    assert report["regularization_strength"] == result.weight == report["regularization_strengths"][result.index]
    assert report["lambda_over_h2max"] == pytest.approx(result.weight / result.transfer_peak)
    assert report["crop"] == {"y0": 3, "x0": 5, "size": 32}
    assert report["scored_z_index"] == result.scored_slice
    assert len(report["scores" if rule == "otsu_cnr" else "residual_norms"]) == 9
    assert report["solution_norms" if rule == "otsu_cnr" else "scores"] == []


@pytest.mark.parametrize("rule", ["otsu_cnr", "l_curve"])
def test_the_montage_is_a_png_of_every_weight(tmp_path, rule):
    path = save_sweep_montage(_sweep(rule), tmp_path / "plots" / "sweep.png", title="A/1/0 t=0")
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size[0] >= 3 * 160  # 3 x 3 cells under a header row
        assert image.size[1] >= 4 * 160
    assert isinstance(ImageColor.colormap["white"], str)  # iohub reads PIL's colour names as hex strings


def test_a_montage_needs_the_kept_slices(tmp_path):
    with pytest.raises(ValueError, match="keep_slices=True"):
        save_sweep_montage(_sweep("otsu_cnr", keep_slices=False), tmp_path / "sweep.png")
