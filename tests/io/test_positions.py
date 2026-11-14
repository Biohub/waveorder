"""Input positions: expanding plates, positions and globs, a position's metadata, channel names and volumes."""

import re
import subprocess
import sys

import iohub
import pytest
import torch
from iohub.ngff.models import ChannelAxisMeta, SpaceAxisMeta, TimeAxisMeta

from tests.io.plates import PIXEL_SIZE, write_plate
from waveorder.errors import ReadError, Rejected, WaveorderError
from waveorder.io.positions import (
    SINGLE_POSITION,
    channel_indices,
    expand_positions,
    is_file_argument,
    pixel_size,
    plate_name,
    read_position_meta,
    read_volume,
)


@pytest.fixture
def plate(tmp_path):
    """A plate with eleven tiny positions, A/1/0 to A/1/10."""
    path = tmp_path / "in.zarr"
    write_plate(path, [torch.zeros(1, 1, 1, 4, 4)] * 11)
    return path


@pytest.fixture
def volume():
    """``(T, C, Z, Y, X)`` = (2, 3, 4, 6, 8) of distinct values."""
    return torch.arange(2 * 3 * 4 * 6 * 8, dtype=torch.float32).reshape(2, 3, 4, 6, 8)


@pytest.fixture
def position(tmp_path, volume):
    """A position with channels BF, GFP and RFP holding ``volume``."""
    (path,) = write_plate(tmp_path / "data.zarr", [volume], channels=("BF", "GFP", "RFP"))
    return path


def test_a_plate_expands_to_its_positions(plate):
    found = expand_positions([plate])
    assert found == [(plate / "A" / "1" / str(index)).resolve() for index in range(11)]
    assert all(path.is_absolute() for path in found)
    assert expand_positions(plate) == found
    assert expand_positions(str(plate)) == found


def test_a_glob_is_naturally_sorted_and_a_repeat_kept_once(plate):
    found = expand_positions([f"{plate}/*/*/*", str(plate / "A" / "1" / "3")])
    assert [path.name for path in found] == [str(index) for index in range(11)]


def test_existing_files_are_skipped(plate):
    assert is_file_argument(plate / "zarr.json")
    assert not is_file_argument(plate)
    found = expand_positions([str(plate / "zarr.json"), str(plate / "A" / "1" / "2")])
    assert [path.name for path in found] == ["2"]


def test_only_files_are_refused_naming_the_first(plate):
    with pytest.raises(Rejected, match="is a file") as error:
        expand_positions([str(plate / "zarr.json")])
    assert error.value.problems[0].location == f"positions {plate / 'zarr.json'}"


def test_a_pattern_matching_nothing_is_named(plate, tmp_path):
    with pytest.raises(Rejected, match="matches nothing on disk") as error:
        expand_positions([str(plate), str(tmp_path / "missing.zarr")])
    assert error.value.problems[0].location == f"positions {tmp_path / 'missing.zarr'}"


def test_a_pattern_matching_only_other_directories_is_named(plate, tmp_path):
    (tmp_path / "notes").mkdir()
    with pytest.raises(Rejected, match="not OME-Zarr positions or plates") as error:
        expand_positions([str(plate / "A" / "1" / "0"), f"{tmp_path}/note*"])
    assert error.value.problems[0].location == f"positions {tmp_path}/note*"


def test_the_plate_name_is_the_name_the_plate_lists(plate):
    assert plate_name(plate / "A" / "1" / "10") == "A/1/10"
    assert plate_name(str(plate / "A" / "1" / "0")) == "A/1/0"


def test_a_position_no_plate_lists_is_a_single_position_store(plate, tmp_path):
    def single(path):
        with iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=["BF"]) as node:
            node.create_zeros("0", shape=(1, 1, 1, 4, 4), dtype="float32")
        return path

    alone = single(tmp_path / "run-1" / "cli4" / "single.zarr")
    assert plate_name(alone) == SINGLE_POSITION
    assert plate_name(single(plate / "B" / "2" / "0")) == SINGLE_POSITION  # under a plate whose metadata omits it
    meta = read_position_meta(alone)
    assert (meta.name, meta.plate_root, meta.store_root) == (SINGLE_POSITION, None, alone)


def test_a_store_linked_into_a_plate_is_the_position_the_plate_lists(tmp_path):
    (listed,) = write_plate(tmp_path / "plate.zarr", [torch.zeros(1, 1, 1, 4, 4)])
    stored = tmp_path / "data" / "single.zarr"
    stored.parent.mkdir()
    listed.rename(stored)
    listed.symlink_to(stored, target_is_directory=True)
    assert plate_name(listed) == "A/1/0"
    assert plate_name(stored) == SINGLE_POSITION
    with iohub.open_ome_zarr(tmp_path / "plate.zarr", mode="r") as node:
        meta = read_position_meta(node["A/1/0"])
    assert (meta.name, meta.plate_root, meta.path) == ("A/1/0", tmp_path / "plate.zarr", stored)


def test_the_metadata_of_a_position(position, tmp_path):
    meta = read_position_meta(position)
    assert meta.path == position
    assert meta.name == "A/1/0"
    assert meta.shape == (2, 3, 4, 6, 8)
    assert meta.zyx == (4, 6, 8)
    assert meta.times == 2
    assert meta.dtype == "float32"
    assert meta.channel_names == ("BF", "GFP", "RFP")
    assert meta.scale == pytest.approx((1.0, 1.0, *PIXEL_SIZE))
    assert meta.pixel_size == pytest.approx(PIXEL_SIZE)
    assert meta.version == "0.5"
    assert len(meta.chunks) == 5
    assert meta.shards is None or len(meta.shards) == 5
    assert meta.plate_root == tmp_path / "data.zarr"
    assert meta.store_root == tmp_path / "data.zarr"
    assert read_position_meta(str(position)) == meta


def test_the_metadata_of_a_v04_position(tmp_path, volume):
    (path,) = write_plate(tmp_path / "old.zarr", [volume], channels=("BF", "GFP", "RFP"), version="0.4")
    meta = read_position_meta(path)
    assert meta.version == "0.4"
    assert meta.shards is None
    assert meta.shape == (2, 3, 4, 6, 8)


def test_the_pixel_size_is_in_micrometers(tmp_path):
    path = tmp_path / "nm.zarr"
    axes = [
        TimeAxisMeta(name="t", unit="second"),
        ChannelAxisMeta(name="c"),
        *(SpaceAxisMeta(name=name, unit="nanometer") for name in "zyx"),
    ]
    with iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=["BF"], axes=axes) as node:
        node.create_image("0", torch.zeros(1, 1, 2, 4, 4).numpy())
        for axis, size in zip("zyx", (400.0, 150.0, 150.0), strict=True):
            node.set_scale("0", axis, size)
    meta = read_position_meta(path)
    assert meta.scale == pytest.approx((1.0, 1.0, 400.0, 150.0, 150.0))
    assert meta.pixel_size == pytest.approx((0.4, 0.15, 0.15))
    assert meta.plate_root is None


@pytest.mark.parametrize(
    ("unit", "sizes", "expected"),
    [
        ("nanometer", (500.0, 100.0, 100.0), (0.5, 0.1, 0.1)),
        ("millimeter", (0.002, 0.001, 0.001), (2.0, 1.0, 1.0)),
        ("inch", (1.0, 1.0, 2.0), (25_400.0, 25_400.0, 50_800.0)),
    ],
)
def test_pixel_size_converts_ome_length_units_to_micrometers(tmp_path, unit, sizes, expected):
    axes = [
        TimeAxisMeta(name="t", unit="second"),
        ChannelAxisMeta(name="c"),
        *(SpaceAxisMeta(name=name, unit=unit) for name in "zyx"),
    ]
    with iohub.open_ome_zarr(tmp_path / "s.zarr", layout="fov", mode="w-", channel_names=["c"], axes=axes) as position:
        position.create_zeros("0", shape=(1, 1, 2, 4, 4), dtype="float32")
        for axis, size in zip("zyx", sizes, strict=True):
            position.set_scale("0", axis, size)
        assert pixel_size(position) == pytest.approx(expected)


def test_what_is_no_position_is_refused_at_its_path(plate, tmp_path):
    (tmp_path / "notes").mkdir()
    for path in (plate, tmp_path / "notes", tmp_path / "missing.zarr"):
        with pytest.raises(Rejected, match="is not an OME-Zarr position") as error:
            read_position_meta(path)
        assert error.value.problems[0].location == str(path)


def test_channel_names_resolve_in_the_order_given(position):
    meta = read_position_meta(position)
    assert channel_indices(meta, ["RFP", "BF"]) == (2, 0)
    assert channel_indices(meta, []) == ()


def test_a_missing_channel_is_refused_listing_the_available_ones(position):
    meta = read_position_meta(position)
    with pytest.raises(Rejected, match="does not have; it has BF, GFP, RFP") as error:
        channel_indices(meta, ["BF", "DAPI", "Cy5"])
    assert "DAPI, Cy5" in error.value.problems[0].message
    assert error.value.problems[0].location == str(position)


def test_a_volume_is_one_time_point_of_the_chosen_channels(position, volume):
    read = read_volume(position, time=1, channels=[2, 0])
    assert read.dtype == torch.float32
    assert torch.equal(read, volume[1, [2, 0]])
    assert torch.equal(read_volume(str(position), time=0, channels=(1,)), volume[0, 1:2])


def test_a_volume_that_cannot_be_read_names_the_position_and_the_time(position):
    chunks = [file for file in (position / "0").rglob("*") if file.is_file() and not file.name.startswith(("z", "."))]
    assert chunks
    for chunk in chunks:
        chunk.write_bytes(b"not a chunk")
    with pytest.raises(ReadError, match=rf"could not read time 1 of {re.escape(str(position))}: ") as error:
        read_volume(position, time=1, channels=[2, 0])
    with pytest.raises(ReadError, match=r"could not read time 0 of .*absent\.zarr: FileNotFoundError"):
        read_volume(position.parent / "absent.zarr", time=0, channels=[0])
    assert isinstance(error.value, WaveorderError)
    assert isinstance(error.value, OSError)
    assert error.value.__cause__ is not None  # what the store raised
    assert type(error.value.__cause__).__name__ in str(error.value)


def test_importing_the_module_imports_neither_torch_nor_iohub():
    code = "import sys; import waveorder.io.positions; assert not {'torch', 'iohub'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
