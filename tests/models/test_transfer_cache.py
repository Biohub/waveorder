"""The transfer functions a store holds: read by a model's reconstruction where they apply, else computed."""

import dataclasses
import logging
import pickle

import pytest
import torch

from tests.models.calibration_data import NA, PIXEL_SIZE, SHAPE, intensities, model_of, raw_config, simulate_phase
from waveorder.io import TfStore
from waveorder.io.tf_store import read_transfer
from waveorder.nn import ThickPhase3D, load_parameters

LOGGER = "waveorder.models._stores"


@pytest.fixture(scope="module")
def data() -> torch.Tensor:
    return intensities(simulate_phase())


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """The path of a store of the phase model's transfer function on :data:`SHAPE`, as ``wo compute-tf`` writes."""
    return model_of().to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE).save(tmp_path_factory.mktemp("tf") / "tf.zarr")


@pytest.fixture
def computed(monkeypatch) -> list[tuple[int, ...]]:
    """The grids the phase system computes a transfer function on."""
    grids: list[tuple[int, ...]] = []
    original = ThickPhase3D.compute_transfer

    def counting(self, sampling):
        grids.append(sampling.shape)
        return original(self, sampling)

    monkeypatch.setattr(ThickPhase3D, "compute_transfer", counting)
    return grids


def test_the_store_holds_the_transfer_with_what_it_was_computed_for(stored):
    (item,) = TfStore.load(stored).transfers
    assert item.shape == SHAPE
    assert item.attrs["pixel_size_zyx_um"] == list(PIXEL_SIZE)
    assert (item.attrs["dtype"], item.attrs["device"]) == ("float32", "cpu")
    assert "backend" not in item.attrs
    assert item.attrs["parameters_sha256"] == TfStore.load(stored).parameters_sha256


def test_a_stored_transfer_that_applies_is_read_not_computed_and_gives_the_same_estimate(
    stored, data, computed, caplog
):
    uncached = model_of().reconstruct(data).estimate
    computed.clear()
    model = model_of()
    model.load_calibration(stored)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        estimate = model.reconstruct(data).estimate
        again = model.reconstruct(data).estimate
    assert computed == []
    assert torch.equal(estimate, uncached)
    assert torch.equal(again, uncached)
    assert caplog.messages == [f"the transfer function on 8x32x32 is read from {stored}, not computed"]
    worker = pickle.loads(pickle.dumps(model))  # what a worker process receives: no transfer, the store's path
    assert worker._stored_transfers.last is None  # noqa: SLF001
    assert torch.equal(worker.reconstruct(data).estimate, uncached)
    assert computed == []


def test_the_grid_after_z_padding_is_the_one_stored_and_matched(data, computed, tmp_path):
    maker = model_of()
    maker.z_padding = 2
    path = maker.to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE).save(tmp_path / "tf.zarr")
    assert [item.shape for item in TfStore.load(path).transfers] == [(12, 32, 32)]
    uncached = maker.reconstruct(data).estimate
    model = model_of()
    model.z_padding = 2
    model.load_calibration(path)
    computed.clear()
    assert torch.equal(model.reconstruct(data).estimate, uncached)
    assert computed == []


@pytest.mark.parametrize(
    ("change", "pixel_size", "planes", "reason"),
    [
        (None, None, 4, "it holds them on 8x32x32 only"),
        (None, (0.5, 0.15, 0.15), 8, "pixel_size_zyx_um [0.4, 0.15, 0.15] (the model: [0.5, 0.15, 0.15])"),
        (lambda m: load_parameters(m.imaging_model, {NA: 0.5}), None, 8, "other parameter values"),
        (lambda m: m.to(dtype=torch.float64), None, 8, "dtype float32 (the model: float64)"),
    ],
    ids=["shape", "voxel_size", "parameter", "dtype"],
)
def test_a_stored_transfer_that_does_not_apply_is_computed_naming_why(
    stored, data, computed, caplog, change, pixel_size, planes, reason
):
    model, uncached = model_of(), model_of()
    model.load_calibration(stored)
    for each in (model, uncached) if change is not None else ():
        change(each)
    volume = data[:planes]
    expected = uncached.reconstruct(volume, pixel_size=pixel_size).estimate
    computed.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        estimate = model.reconstruct(volume, pixel_size=pixel_size).estimate
    assert len(computed) == 1
    assert torch.equal(estimate, expected)
    grid = f"{planes}x32x32"
    (message,) = caplog.messages
    assert message.startswith(f"the transfer function on {grid} is computed, not read from {stored}: ")
    assert reason in message  # a float64 model also holds its float32 values as float64 ones: other values too


def test_a_transfer_stored_for_another_imaging_system_is_computed(stored, data, computed, caplog, tmp_path):
    store = TfStore.load(stored)
    (item,) = store.transfers
    other = dataclasses.replace(item, attrs=item.attrs | {"imaging_system_sha256": "0" * 64})

    held = read_transfer(stored, SHAPE)
    assert held is not None
    path = dataclasses.replace(store, transfers=(dataclasses.replace(other, transfer=held.transfer),)).save(
        tmp_path / "other.zarr"
    )
    model = model_of()
    model.load_calibration(path)
    computed.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        estimate = model.reconstruct(data).estimate
    assert len(computed) == 1
    assert torch.equal(estimate, model_of().reconstruct(data).estimate)
    assert caplog.messages == [
        f"the transfer function on 8x32x32 is computed, not read from {path}: another imaging system"
    ]


def test_a_calibration_neither_reads_nor_mentions_the_stored_transfer(stored, data, caplog):
    raw = raw_config()
    raw["solver"]["calibration"]["search"] = [{"grid": {"points": 3}}]
    model = model_of(raw)
    model.load_calibration(stored)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        model.calibrate(data)
        model.reconstruct(data)
    assert caplog.messages == []
