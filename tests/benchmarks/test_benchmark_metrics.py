"""benchmarks.metrics: the wo-4.0 metric definitions, checked against values wo-4.0 (85282ce) computed."""

import numpy as np
import pytest
import torch
from benchmarks.metrics import comparison_metrics, compute_metrics, distribution_stats, structural_similarity


def _metrics(volume, **comparisons):
    return compute_metrics(volume, NA_det=1.2, wavelength=0.532, pixel_size=0.1, **comparisons)


def _random_volume(shape=(16, 32, 32), seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen)


def _patterns():
    z, y, x = torch.meshgrid(torch.arange(8.0), torch.arange(24.0), torch.arange(20.0), indexing="ij")
    volume = torch.sin(0.7 * x + 0.3 * y * y / 10 + 0.5 * z) + 0.1 * torch.cos(1.3 * x * y / 7)
    target = torch.cos(0.5 * x - 0.2 * y + 0.4 * z)
    return volume, target


# compute_metrics of _patterns() in wo-4.0 (skimage SSIM, waveorder.optim.losses), histograms left out.
_WO40_3D = {
    "image_quality": {
        "max": 1.0999575853347778,
        "min": -1.0991069078445435,
        "mean": 0.007978840731084347,
        "midband_power": 28.117509841918945,
        "spectral_flatness": 0.703508198261261,
        "total_variation": 0.4396868944168091,
    },
    "with_phantom": {"mse": 1.012688159942627, "ssim": 0.03590793561687723, "correlation": -0.0050742304883897305},
}
_WO40_2D = {
    "image_quality": {
        "max": 1.0993093252182007,
        "min": -1.0974570512771606,
        "mean": 0.015089468099176884,
        "midband_power": 32.22210693359375,
        "spectral_flatness": 0.6414532661437988,
        "total_variation": 0.4404887557029724,
    },
    "with_phantom": {"mse": 1.0348503589630127, "ssim": 0.003184316687107668, "correlation": -0.0189313143491745},
}


class TestEqualToWo40:
    def test_volume(self):
        volume, target = _patterns()
        result = compute_metrics(volume, NA_det=1.2, wavelength=0.532, pixel_size=0.1, phantom=target)
        result["image_quality"].pop("histogram")
        assert result == _WO40_3D

    def test_plane(self):
        volume, target = _patterns()
        result = compute_metrics(volume[3], NA_det=0.55, wavelength=0.45, pixel_size=0.325, phantom=target[3])
        result["image_quality"].pop("histogram")
        assert result == _WO40_2D

    @pytest.mark.parametrize(
        ("index", "expected"), [(slice(None), 0.8473457863062491), (0, 0.6247562434427537)], ids=["3d", "2d"]
    )
    def test_ssim_equals_scikit_image(self, index, expected):
        # scikit-image 0.26 structural_similarity(a, b, data_range=b.max() - b.min()) on these arrays.
        rng = np.random.default_rng(0)
        a = rng.normal(size=(9, 20, 17)).astype(np.float32)
        b = (a + 0.5 * rng.normal(size=a.shape)).astype(np.float32)
        a, b = a[index], b[index]
        assert structural_similarity(a, b, data_range=b.max() - b.min()) == expected


class TestDistributionStats:
    def test_keys(self):
        stats = distribution_stats(_random_volume())
        assert {"max", "min", "mean", "histogram"} <= set(stats.keys())

    def test_histogram_bins(self):
        stats = distribution_stats(_random_volume(), n_bins=20)
        assert len(stats["histogram"]["counts"]) == 20
        assert len(stats["histogram"]["bin_edges"]) == 21

    def test_values_correct(self):
        stats = distribution_stats(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert stats["max"] == 5.0
        assert stats["min"] == 1.0
        assert stats["mean"] == 3.0


class TestComparisonMetrics:
    def test_identical_volumes(self):
        v = _random_volume()
        m = comparison_metrics(v, v)
        assert m["mse"] == 0.0
        assert abs(m["ssim"] - 1.0) < 1e-5
        assert abs(m["correlation"] - 1.0) < 1e-5

    def test_different_volumes(self):
        m = comparison_metrics(_random_volume(seed=0), _random_volume(seed=1))
        assert m["mse"] > 0
        assert -1 <= m["ssim"] <= 1
        assert -1 <= m["correlation"] <= 1


class TestComputeMetrics:
    def test_image_quality_always_present(self):
        result = _metrics(_random_volume())
        assert {"max", "midband_power", "spectral_flatness", "total_variation"} <= set(result["image_quality"])
        assert result["image_quality"]["midband_power"] > 0
        assert "with_reference" not in result
        assert "with_phantom" not in result

    def test_with_reference(self):
        v = _random_volume()
        assert "ssim" in _metrics(v, reference=v)["with_reference"]

    def test_with_phantom(self):
        v = _random_volume()
        assert "mse" in _metrics(v, phantom=v)["with_phantom"]
