"""``wo simulate``: the drawn truth, the three stores, determinism, and the refusals."""

import iohub
import pytest
import torch
from typer.testing import CliRunner

from waveorder.cli import app
from waveorder.io import TfStore

runner = CliRunner()

SIMULATION = """\
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.532
    numerical_aperture: 1.2
    pupil_aberrations:
      zernike_up_to_order: {order: 2, tolerance: 0.05, bounds: [-0.3, 0.3]}
      spherical: {nominal: 0.1, tolerance: 0.05}
      coma_x: 0.02
object:
  dimension: 3
  source: {gaussian_spectrum: {exponent: 2.0}}
noise: {gaussian: {snr_db: 20}}
simulate: {seed: 0, shape: [8, 32, 32]}
"""
PREFIX = "system.detection.pupil_aberrations."


def _simulate(tmp_path, text: str, name: str, *extra: str):
    config = tmp_path / f"{name}.yml"
    config.write_text(text)
    return runner.invoke(app, ["simulate", "-c", str(config), "-o", str(tmp_path / name), *extra])


def _volume(path) -> tuple[list[str], torch.Tensor, list[float]]:
    with iohub.open_ome_zarr(path / "0/0/0", mode="r") as position:
        return list(position.channel_names), torch.as_tensor(position["0"][:]), list(position.scale)


def test_simulate_writes_data_object_and_tf_stores(tmp_path):
    result = _simulate(tmp_path, SIMULATION, "sim")
    assert result.exit_code == 0, result.output
    output = tmp_path / "sim"
    assert f"wrote data.zarr, object.zarr, tf.zarr to {output}" in result.stderr
    names, data, scale = _volume(output / "data.zarr")
    assert (names, tuple(data.shape), data.dtype) == (["GFP"], (1, 1, 8, 32, 32), torch.float32)
    assert scale == pytest.approx([1, 1, 0.25, 0.1, 0.1])
    names, obj, _ = _volume(output / "object.zarr")
    assert (names, tuple(obj.shape)) == (["object"], (1, 1, 8, 32, 32))
    assert float(obj.mean()) == pytest.approx(0, abs=1e-5)
    assert float(obj.std()) == pytest.approx(1, abs=1e-4)

    # The truth: one draw per numeric tolerance from one generator, shorthand terms first in Noll order, clamped.
    generator = torch.Generator().manual_seed(0)
    expected = {
        name: min(max(0.05 * float(torch.randn((), generator=generator)), -0.3), 0.3)
        for name in ("defocus", "astigmatism_45", "astigmatism_0")
    }
    expected["spherical"] = 0.1 + 0.05 * float(torch.randn((), generator=generator))
    store = TfStore.load(output / "tf.zarr")
    assert store.provenance["kind"] == "simulated"
    assert store.provenance["seed"] == 0
    truth = store.provenance["true_coefficients_waves"]
    assert truth == pytest.approx({**expected, "coma_x": 0.02})
    assert store.row() == pytest.approx({PREFIX + name: value for name, value in expected.items()}, abs=1e-6)
    assert [list(item.shape) for item in store.transfers] == [[8, 32, 32]]
    for name, value in expected.items():
        assert f"true {name:16s} {value:+.4f} waves" in result.stdout.splitlines()


def test_the_same_seed_gives_the_same_stores_and_another_seed_others(tmp_path):
    assert _simulate(tmp_path, SIMULATION, "a").exit_code == 0
    assert _simulate(tmp_path, SIMULATION, "b").exit_code == 0
    assert _simulate(tmp_path, SIMULATION.replace("seed: 0", "seed: 1"), "c").exit_code == 0
    a, b, c = (_volume(tmp_path / name / "data.zarr")[1] for name in "abc")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_poisson_counts_are_non_negative_integers(tmp_path):
    result = _simulate(tmp_path, SIMULATION.replace("{gaussian: {snr_db: 20}}", "{poisson: {photons: 50}}"), "p")
    assert result.exit_code == 0, result.output
    data = _volume(tmp_path / "p" / "data.zarr")[1]
    assert bool((data >= 0).all())
    assert torch.equal(data, data.round())
    assert 25 < float(data.mean()) < 100


def test_simulate_refuses_a_configuration_without_a_simulate_block_and_existing_stores(tmp_path):
    plain = _simulate(tmp_path, SIMULATION.replace("noise: {gaussian: {snr_db: 20}}\n", "").split("simulate:")[0], "n")
    assert plain.exit_code == 2
    assert "has no simulate block; add simulate: {seed: 0, shape: [Z, Y, X]}" in plain.stderr
    assert not (tmp_path / "n").exists()
    assert _simulate(tmp_path, SIMULATION, "s").exit_code == 0
    again = _simulate(tmp_path, SIMULATION, "s")
    assert again.exit_code == 2
    assert f"-o {tmp_path / 's' / 'data.zarr'}: exists; pass --overwrite to replace it" in again.stderr
    assert _simulate(tmp_path, SIMULATION, "s", "--overwrite").exit_code == 0
