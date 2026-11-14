import logging
import pickle
import shutil
from collections import Counter
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests.data.plates import PIXEL_SIZE, Plates, raw_field, target_field, write_plate
from waveorder.data import (
    Batch,
    DeviceLoader,
    FovIndex,
    FovRow,
    GridPatches,
    GroupSplit,
    PatchData,
    RandomPatches,
    Recorded,
    index_from_plates,
)
from waveorder.data.loader import _Stream
from waveorder.errors import BackendUnavailableError, ParameterError, ReadError, ShapeError
from waveorder.nn.networks import Attributes, FieldPosition, FieldStandardize, PlaneOffsets

ALL_TRAIN = GroupSplit(by="plate", fractions={"train": 1.0})


def _plane_offsets(planes: int = 5) -> PlaneOffsets:
    """The defocus of each plane in units of 1 um (a wavelength and an aperture of 1): planes 2 um apart."""
    return PlaneOffsets(planes, wavelength=1.0, numerical_aperture=1.0)


def _halve(data: torch.Tensor) -> torch.Tensor:
    """A picklable ``prepare``."""
    return data / 2


class _Counting(Recorded):
    """A recorded source that counts its reads (in this process: use it with ``workers=0``)."""

    def __init__(self) -> None:
        self.reads: Counter[str] = Counter()

    def read(self, row, *, channels, target_channels):
        self.reads[row.name] += 1
        return super().read(row, channels=channels, target_channels=target_channels)


def _data(index: FovIndex, **settings: object) -> PatchData:
    arguments: dict[str, object] = {"patches": RandomPatches((32, 32), per_field=8), "batch_size": 4, "workers": 0}
    return PatchData(index, **{**arguments, **settings})  # ty: ignore[invalid-argument-type]


def _position(batch: Batch, number: int) -> str:
    """The position of a sample within its plate: ``A/1/0`` of ``raw.zarr/A/1/0``."""
    return batch.position[number].split("/", 1)[1]


def _signature(batches: list[Batch]) -> list[tuple[str, tuple[int, int]]]:
    return [
        (name, tuple(origin))
        for batch in batches
        for name, origin in zip(batch.position, batch.origin.tolist(), strict=False)
    ]


def _rows(count: int, *, height: int = 64) -> FovIndex:
    """An index of ``count`` fields that is never read: for what the loader computes from the rows alone."""
    rows = [
        FovRow(input="p.zarr", position=f"A/1/{n}", target="r.zarr", plate="p", well="A/1", planes=5, height=height,
               width=64)
        for n in range(count)
    ]  # fmt: skip
    meta = {"pixel_size": list(PIXEL_SIZE), "dtype": "uint16"}
    return FovIndex.of(rows, channels=["BF"], target_channels=["Phase2D"], meta=meta)


def _recorded(directory: Path, raw: dict[str, torch.Tensor], *, channels: list[str]) -> FovIndex:
    """Write ``raw`` as a plate with a one-plane ``Phase2D`` target per position; return its index, all train."""
    inputs = write_plate(directory / "raw.zarr", raw, channels=channels, pixel_size=PIXEL_SIZE)
    targets = {name: target_field(name, (1, 1, volume.shape[-2], volume.shape[-1])) for name, volume in raw.items()}
    recorded = write_plate(directory / "rec.zarr", targets, channels=["Phase2D"], pixel_size=PIXEL_SIZE)
    return index_from_plates(
        inputs, channels=channels, targets=[recorded], target_channels=["Phase2D"], split=ALL_TRAIN
    )


def test_batches_hold_patches_of_the_recorded_pair_and_their_conditions(index):
    conditions = (FieldPosition(), _plane_offsets())
    data = _data(index, conditions=conditions, stride=8)
    loader = data.loader("train", epoch=0)
    batches = list(loader)
    assert len(batches) == len(loader) == 4 * 8 // 4
    for batch in batches:
        assert batch.data.shape == (4, 2, 5, 32, 32)
        assert batch.data.dtype == torch.uint16
        assert batch.target is not None
        assert batch.target.shape == (4, 2, 1, 32, 32)
        assert batch.target.dtype == torch.float32
        assert (batch.origin % 8 == 0).all()
        for number, (y, x) in enumerate(batch.origin.tolist()):
            name = _position(batch, number)
            assert torch.equal(batch.data[number], raw_field(name)[0, :, :, y : y + 32, x : x + 32])
            assert torch.equal(batch.target[number], target_field(name)[0, :, :, y : y + 32, x : x + 32])
            centre = torch.tensor([(y + 16) / 64, (x + 16) / 64])
            assert torch.equal(batch.condition["field_position"][number], centre)
        defocus = torch.tensor([4.0, 2.0, 0.0, -2.0, -4.0])  # 2 um between planes, plane 2 in focus, z downwards
        assert torch.equal(batch.condition["plane_offsets"], defocus.expand(4, -1))
    assert {name.split("/", 1)[1] for batch in batches for name in batch.position} == {
        row.position for row in index.split("train")
    }


def test_prepare_and_normalizer_see_the_whole_field(index):
    data = _data(index, prepare=_halve, input_normalizer=FieldStandardize())
    batch = next(iter(data.loader("train")))
    assert batch.data.dtype == torch.float32
    for number, (y, x) in enumerate(batch.origin.tolist()):
        field = FieldStandardize()(raw_field(_position(batch, number))[0].to(torch.float32) / 2)
        assert torch.allclose(batch.data[number], field[:, :, y : y + 32, x : x + 32], atol=1e-5)


def test_attributes_come_from_the_index(plates):
    index = index_from_plates(
        [plates.inputs],
        channels=["BF"],
        targets=[plates.targets],
        target_channels=["Phase2D"],
        split=ALL_TRAIN,
        attributes=lambda _, name: {"fov": float(name.split("/")[2])},
    )
    batch = next(iter(_data(index, conditions=[Attributes(["fov"], scale=2.0)]).loader("train")))
    expected = [float(_position(batch, number).split("/")[2]) / 2 for number in range(len(batch))]
    assert batch.condition["attributes"][:, 0].tolist() == expected
    assert batch.data.shape[1] == 1  # the index's one channel


def test_validation_is_the_same_every_epoch_and_reads_each_field_once_over_ranks(index):
    data = _data(index)
    grid = GridPatches((32, 32)).origins((64, 64)).tolist()
    expected = sorted((f"raw.zarr/{row.position}", tuple(origin)) for row in index.split("val") for origin in grid)
    for world_size in (1, 2, 3):
        seen = []
        for rank in range(world_size):
            shard = data.shard(rank=rank, world_size=world_size)
            loader = shard.loader("val", epoch=0)
            first = list(loader)
            shard.set_epoch(5)
            assert _signature(list(loader)) == _signature(first)
            assert len(first) == len(loader)
            seen.extend(_signature(first))
        assert sorted(seen) == expected


def test_training_shards_are_disjoint_and_cover_the_split():
    index = _rows(23)
    for world_size in (1, 2, 3, 4):
        for workers in (1, 2, 3):
            data = PatchData(index, patches=RandomPatches((32, 32), per_field=8), batch_size=4, workers=workers)
            lengths, epochs = set(), []
            for epoch in (0, 1):
                data.set_epoch(epoch)
                read: list[str] = []
                for rank in range(world_size):
                    shard = data.shard(rank=rank, world_size=world_size)
                    read.extend(row.name for w in range(workers) for row in shard.assignment("train", worker=w))
                    lengths.add(len(shard.loader("train")))
                assert len(read) == len(set(read)) == 23 // world_size * world_size  # once, but a tail < world_size
                epochs.append(read)
            assert len(lengths) == 1  # as many batches on every rank
            assert epochs[0] != epochs[1]  # a new permutation each epoch


def test_every_rank_yields_its_length_in_batches(index):
    data = _data(index, patches=RandomPatches((32, 32), per_field=6))
    counts = []
    for rank in range(2):
        shard = data.shard(rank=rank, world_size=2)
        loader = shard.loader("train", epoch=3)
        batches = list(loader)
        counts.append(len(batches))
        assert len(batches) == len(loader) == 2 * 6 // 4
        assert {_position(batch, n) for batch in batches for n in range(4)} == {
            row.position for row in shard.assignment("train")
        }
    assert counts[0] == counts[1]


def test_set_epoch_reaches_persistent_spawned_workers(index):
    data = _data(index, workers=2)
    loader = data.loader("train", epoch=0)
    assert isinstance(loader, DataLoader)
    first = _signature(list(loader))
    running = loader._iterator  # noqa: SLF001 (the iterator that keeps the workers alive)
    workers = [process.pid for process in running._workers]  # ty: ignore[missing-attribute]  # noqa: SLF001
    assert data.loader("train", epoch=1) is loader
    second = _signature(list(loader))
    assert loader._iterator is running  # noqa: SLF001
    assert [process.pid for process in running._workers] == workers  # ty: ignore[missing-attribute]  # noqa: SLF001
    assert second != first
    data.set_epoch(0)
    assert _signature(list(loader)) == first
    assert len(first) == 4 * len(loader) == 4 * 8
    data.close()


@pytest.fixture
def with_blank(tmp_path) -> FovIndex:
    names = ["A/1/0", "A/1/1", "A/1/2", "A/1/3"]
    raw = {name: torch.zeros_like(raw_field(name)) if name == "A/1/1" else raw_field(name) for name in names}
    return _recorded(tmp_path, raw, channels=["GFP", "BF"])


def test_blank_fields_are_skipped_and_counted(with_blank, caplog):
    data = _data(with_blank)
    with caplog.at_level(logging.WARNING, logger="waveorder.data.loader"):
        batches = list(data.loader("train", epoch=0))
    assert len(batches) == len(data.loader("train")) == 4 * 8 // 4  # the others give the blank field's batches
    assert "A/1/1" not in {_position(batch, n) for batch in batches for n in range(4)}
    assert data.skipped == 1
    assert "skipped 1 blank (all-zero) fields" in caplog.text
    assert "raw.zarr/A/1/1" in caplog.text
    assert len(list(data.loader("train", train=False))) == 3 * 4 // 4
    assert [field.row.position for field in data.fields("train")] == ["A/1/0", "A/1/2", "A/1/3"]
    assert data.skipped == 3


def _damaged(plates: Plates, directory: Path) -> FovIndex:
    raw = shutil.copytree(plates.inputs, directory / "raw.zarr")
    chunks = [path for path in (raw / "A/1/0/0").rglob("*") if path.is_file() and not path.name.startswith((".", "z"))]
    assert chunks
    for chunk in chunks:
        chunk.write_bytes(b"not a chunk")
    return index_from_plates(
        raw, channels=plates.channels, targets=[plates.targets], target_channels=["Phase2D"], split=ALL_TRAIN
    )


@pytest.mark.parametrize("workers", [0, 1])
def test_a_field_that_cannot_be_read_names_plate_position_and_time(plates, tmp_path, workers):
    index = _damaged(plates, tmp_path)
    data = _data(index, patches=GridPatches((64, 64)), workers=workers)
    with pytest.raises(ReadError, match=r"could not read time 0 of .*raw\.zarr/A/1/0"):
        list(data.loader("train", train=False))
    data.close()


def test_mixed_plane_counts_batch_apart_or_pad(tmp_path):
    names = ["A/1/0", "A/1/1", "A/1/2", "A/1/3"]
    raw = {name: raw_field(name, (1, 5 if number % 2 else 3, 64, 64)) for number, name in enumerate(names)}
    index = _recorded(tmp_path, raw, channels=["BF"])
    assert index.planes == (3, 5)
    data = _data(index, patches=RandomPatches((32, 32), per_field=6))
    batches = list(data.loader("train"))
    assert len(batches) == len(data.loader("train")) == 4 * 6 // 4
    for batch in batches:
        planes = batch.data.shape[2]
        assert {raw[_position(batch, n)].shape[2] for n in range(4)} == {planes}  # one plane count per batch
    padded = _data(index, conditions=[_plane_offsets()])
    for batch in padded.loader("train"):
        assert batch.data.shape[2] == 5
        for number in range(4):
            depth = raw[_position(batch, number)].shape[2]
            assert (batch.data[number, :, depth:] == 0).all()
            assert batch.condition["plane_offsets"][number, depth:].isnan().all()


def test_fields_limit_and_cache(index):
    data = _data(index, prepare=_halve)
    fields = list(data.fields("val"))
    assert [field.row.split for field in fields] == ["val", "val"]
    assert fields[0].data.dtype == torch.float32
    assert fields[0].data.shape == (2, 5, 64, 64)
    assert torch.equal(fields[0].data, raw_field(fields[0].row.position)[0].to(torch.float32) / 2)
    assert fields[0].target is not None
    assert torch.equal(fields[0].target, target_field(fields[0].row.position)[0])
    assert len(list(data.fields("train", limit=3))) == 3
    assert data.replace(limit_fields=3).describe()["splits"]["train"]["fields"] == 3  # ty: ignore[not-subscriptable]
    for cache_fields, reads in ((0, 2), (4, 1)):
        source = _Counting()
        cached = _data(index, source=source, cache_fields=cache_fields)
        for epoch in range(2):
            list(cached.loader("train", epoch=epoch))
        assert set(source.reads.values()) == {reads}
        assert len(source.reads) == 4


def test_describe_counts_what_each_rank_reads():
    data = PatchData(_rows(10, height=96), patches=RandomPatches((32, 32), per_field=8), batch_size=4, cache_fields=2)
    described = data.shard(rank=1, world_size=3).describe()
    assert described["splits"] == {
        "train": {"fields": 10, "fields_per_epoch": 3, "patches_per_epoch": 24, "batches_per_epoch": 6}
    }
    assert described["rank"] == 1
    assert described["world_size"] == 3
    assert described["planes"] == [5]
    assert described["bytes_per_field"] == 96 * 64 * (5 * 2 + 4)  # uint16 data, one float32 target plane
    assert described["cache_bytes_per_worker"] == 2 * 96 * 64 * (5 * 2 + 4)


def test_rank_and_world_size_are_found_or_given(monkeypatch):
    index = _rows(4)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "3")
    data = PatchData(index, patches=RandomPatches((32, 32)))
    assert (data.rank, data.world_size) == (2, 3)
    assert (PatchData(index, patches=RandomPatches((32, 32)), rank=0, world_size=1).rank, data.rank) == (0, 2)
    monkeypatch.setenv("RANK", "3")
    with pytest.raises(ParameterError, match="WORLD_SIZE"):
        _ = data.rank
    monkeypatch.delenv("RANK")
    monkeypatch.delenv("WORLD_SIZE")
    assert (data.rank, data.world_size) == (0, 1)
    with pytest.raises(ParameterError, match="together"):
        PatchData(index, patches=RandomPatches((32, 32)), rank=1)


def test_batch_transfer_and_copies(index):
    data = _data(index)
    batch = next(iter(data.loader("val")))
    moved = batch.to("cpu")
    assert torch.equal(moved.data, batch.data)
    assert moved.position == batch.position
    if torch.cuda.is_available():
        assert batch.pin_memory().data.is_pinned()
    copy = data.replace(batch_size=2)
    data.set_epoch(7)
    assert copy.epoch == 7
    assert data.shard(rank=0, world_size=2) is data.shard(rank=0, world_size=2)


def test_fields_smaller_than_a_patch_are_refused_when_built(index):
    with pytest.raises(ShapeError, match="smaller than a patch"):
        _data(index, patches=RandomPatches((128, 128)))


def test_a_loader_pickles_its_rows_as_a_handle_so_workers_start_together():
    index = _rows(3000)
    loader = PatchData(index, patches=RandomPatches((32, 32)), workers=2).loader("train")
    assert isinstance(loader, DataLoader)
    stream = loader.dataset
    assert isinstance(stream, _Stream)
    packed = ForkingPickler.dumps(stream)
    assert len(packed) < 16384 < len(pickle.dumps(list(index)))  # fits in a pipe, unlike the row objects
    unpacked = pickle.loads(packed)
    assert unpacked.rows.equals(index.split("train").table)
    assert list(unpacked.assignment(1, epoch=2)) == list(stream.assignment(1, epoch=2))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the device pipeline needs CUDA")


def _same_batches(device: list[Batch], host: list[Batch]) -> None:
    assert _signature(device) == _signature(host)  # the same patches of the same fields in the same order
    for on_device, on_host in zip(device, host, strict=True):
        assert on_device.data.is_cuda
        assert on_device.data.dtype == on_host.data.dtype
        assert torch.allclose(on_device.data.cpu().float(), on_host.data.float(), atol=1e-5)
        assert on_device.target is not None
        assert on_host.target is not None
        assert torch.allclose(on_device.target.cpu(), on_host.target, atol=1e-6)
        assert on_device.condition.keys() == on_host.condition.keys()
        for name, values in on_device.condition.items():
            assert values.is_cuda
            assert torch.allclose(values.cpu(), on_host.condition[name], equal_nan=True)


@cuda
@pytest.mark.parametrize("workers", [0, 2])
def test_the_device_pipeline_gives_the_cpu_pipelines_batches_on_the_device(index, workers):
    settings: dict[str, object] = {
        "conditions": (FieldPosition(), _plane_offsets()),
        "input_normalizer": FieldStandardize(),
        "prepare": _halve,
        "stride": 4,
    }
    host = _data(index, **settings)
    device = _data(index, workers=workers, device="cuda", **settings)
    loader = device.loader("train")
    assert isinstance(loader, DeviceLoader)
    for epoch in (0, 1):
        batches = list(device.loader("train", epoch=epoch))
        assert len(batches) == len(loader)
        _same_batches(batches, list(host.loader("train", epoch=epoch)))
        assert loader.stats["fields"] == 4
    _same_batches(list(device.loader("val")), list(host.loader("val")))
    uint16 = _data(index, device="cuda")
    _same_batches(list(uint16.loader("train")), list(_data(index).loader("train")))  # stored dtype, no prepare
    described = device.describe()
    assert described["device"] == f"cuda:{torch.cuda.current_device()}"
    assert isinstance(described["device_bytes_per_rank"], int)
    assert isinstance(described["pinned_bytes_per_rank"], int)
    assert described["device_bytes_per_rank"] > 0
    assert described["pinned_bytes_per_rank"] > 0
    device.close()


@cuda
def test_the_device_pipeline_skips_and_counts_blank_fields(with_blank):
    data = _data(with_blank, device="cuda")
    batches = list(data.loader("train", epoch=0))
    assert len(batches) == len(data.loader("train")) == 4 * 8 // 4
    assert "A/1/1" not in {_position(batch, n) for batch in batches for n in range(4)}
    assert data.skipped == 1
    _same_batches(batches, list(_data(with_blank).loader("train", epoch=0)))
    data.close()


@cuda
def test_set_epoch_reaches_the_device_pipelines_workers_and_the_cache_spares_reads(index):
    data = _data(index, workers=2, device="cuda", cache_fields=4)
    loader = data.loader("train", epoch=0)
    assert isinstance(loader, DeviceLoader)
    first = _signature(list(loader))
    assert loader.stats["fields"] == 4
    assert loader._loader is not None  # noqa: SLF001
    running = loader._loader._iterator  # noqa: SLF001 (the iterator that keeps the workers alive)
    workers = [process.pid for process in running._workers]  # ty: ignore[missing-attribute]  # noqa: SLF001
    data.set_epoch(1)
    second = _signature(list(loader))
    assert second != first
    assert (loader.stats["fields"], loader.stats["cached"]) == (0, 4)  # all four on the device already
    assert [process.pid for process in running._workers] == workers  # noqa: SLF001
    data.set_epoch(0)
    assert _signature(list(loader)) == first
    data.close()


@cuda
def test_lightning_moves_a_device_batch_without_copying_it(index):
    from lightning.fabric.utilities.apply_func import move_data_to_device as move

    data = _data(index, device="cuda")
    batch = next(iter(data.loader("val")))
    moved = move(batch, batch.data.device)
    assert moved.data.data_ptr() == batch.data.data_ptr()
    assert moved.target is not None
    assert batch.target is not None
    assert moved.target.data_ptr() == batch.target.data_ptr()
    data.close()


def test_device_names_are_checked(index, monkeypatch):
    assert _data(index, device="cpu").device is None
    with pytest.raises(ParameterError, match="device must be"):
        _data(index, device="tpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _data(index, device="auto").device is None  # the CPU pipeline where there is no GPU
    with pytest.raises(BackendUnavailableError, match="CUDA is not available"):
        _data(index, device="cuda")


@cuda
def test_a_device_loader_left_mid_epoch_starts_the_next_epoch_cleanly(index):
    data = _data(index, workers=2, device="cuda")
    loader = data.loader("train", epoch=0)
    whole = _signature(list(loader))
    left = iter(loader)
    assert _signature([next(left)]) == whole[:4]  # one batch, then the iteration is left open
    again = _signature(list(loader))  # a new iteration stops the open one's upload thread first
    assert again == whole
    del left  # closing the old iteration now must not disturb the loader
    assert _signature(list(loader)) == whole
    data.close()
