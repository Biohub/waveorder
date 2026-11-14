"""The output plate: 3.x provenance keys, array geometry, creating it, filling it through iohub, the estimate."""

import iohub
import pytest
import torch

import waveorder.io
from tests.io.plates import write_plate
from waveorder.errors import ReadError, Rejected
from waveorder.io.plate import (
    Geometry,
    OutputPosition,
    choose_geometry,
    create_plate,
    estimate_resources,
    finished,
    is_single,
    name_problems,
    process_time,
    provenance_key,
    read_plate,
    tracked_time,
)
from waveorder.io.positions import SINGLE_POSITION

SCALE = (1.0, 1.0, 0.4, 0.15, 0.15)
RECORD = {"config": {"version": 4, "object": {"weight": 1e-3}}, "channels": ["BF"], "times": [0, 1]}


def _position(name: str = "A/1/0", *, z: int = 4, geometry: Geometry | None = None) -> OutputPosition:
    return OutputPosition(name=name, shape=(2, 1, z, 8, 8), scale=SCALE, geometry=geometry or Geometry())


def _create(output, positions, *, names=("Phase3D",), record=RECORD, version="0.5", sources=()):
    records = {position.name: record for position in positions}
    return create_plate(output, positions, channel_names=list(names), records=records, version=version, sources=sources)


def _attrs(output, name: str = "A/1/0") -> tuple[list[str], dict]:
    with iohub.open_ome_zarr(output / name, mode="r") as position:
        return list(position.channel_names), dict(position.zattrs)


@pytest.fixture
def source(tmp_path):
    """An input plate with positions A/1/0 and A/1/1 of two time points, root and per-position attributes."""
    path = tmp_path / "in.zarr"
    volume = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 1, 4, 8, 8) + 1
    write_plate(path, [volume, volume])
    return path


@pytest.mark.parametrize(
    ("names", "key"),
    [
        (["Phase3D"], "waveorder-Phase3D"),
        (["Phase2D", "Absorption2D"], "waveorder-Phase2D"),
        (["BF_Density3D"], "waveorder-BF_Density3D"),
        (["BF_Density2D"], "waveorder-BF_Density2D"),
        (["GFP_Density3D", "RFP_Density3D"], "waveorder-GFP_Density3D,RFP_Density3D"),
        (["GFP_Deblurred3D"], "waveorder-GFP_Deblurred3D"),  # no 3.x name: every channel, joined with "-"
        (["Scatter3D", "Haze3D"], "waveorder-Scatter3D-Haze3D"),
        (["Phase3D", "Haze3D"], "waveorder-Phase3D"),  # a 3.x name keeps the 3.x key
    ],
)
def test_keys_follow_3x_and_other_channels_are_keyed_by_name(names, key):
    assert provenance_key(names) == key


def test_whole_volumes_take_the_default_geometry():
    assert choose_geometry(None, x=600) == Geometry()
    assert Geometry() == Geometry(chunks=None, shards=None)


def test_tiles_take_shards_one_tile_row_high_of_chunks_that_divide_them():
    geometry = choose_geometry((12, 300, 64), x=600)
    assert geometry.shards == (1, 1, 12, 300, 768)
    assert geometry.chunks == (1, 1, 12, 150, 256)
    small = choose_geometry((4, 4, 8), x=8)
    assert (small.chunks, small.shards) == ((1, 1, 4, 4, 8), (1, 1, 4, 4, 8))


def test_no_plate_reads_as_none_and_something_else_is_refused(tmp_path):
    assert read_plate(tmp_path / "out.zarr") is None
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("mine")
    with pytest.raises(Rejected, match=r"exists and is not an OME-Zarr plate .*; use another output") as error:
        read_plate(notes)
    assert error.value.problems[0].location == str(notes)


def test_a_plate_is_created_with_its_channels_scale_and_record(tmp_path, source):
    output = tmp_path / "out.zarr"
    assert _create(output, [_position()], sources=[source]) == {"A/1/0": (0,)}
    plate = read_plate(output)
    assert plate is not None
    assert (plate.version, plate.shapes) == ("0.5", {"A/1/0": (2, 1, 4, 8, 8)})
    names, attrs = _attrs(output)
    assert names == ["Phase3D"]
    assert attrs["waveorder-Phase3D"] == RECORD
    assert attrs["biahub-deskew"] == {"angle": 30}  # inherited from the input position
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert dict(node.zattrs)["acquisition"] == {"microscope": "test"}  # copied from the input plate root
        position = node["A/1/0"]
        assert position.get_effective_scale("0") == pytest.approx(SCALE)
        assert str(position["0"].dtype) == "float32"
        assert tuple(position["0"].shards)[:2] == (1, 1)  # one time point and channel per shard


def test_a_plate_is_created_in_the_version_asked_for(tmp_path):
    output = tmp_path / "out.zarr"
    _create(output, [_position()], version="0.4")
    plate = read_plate(output)
    assert plate is not None
    assert plate.version == "0.4"


def test_a_position_another_reconstruction_and_a_new_record_are_added_to_an_existing_plate(tmp_path):
    output = tmp_path / "out.zarr"
    tiled = choose_geometry((4, 4, 8), x=8)
    _create(output, [_position()])
    assert _create(output, [_position(), _position("A/1/1", geometry=tiled)]) == {"A/1/0": (0,), "A/1/1": (0,)}
    with iohub.open_ome_zarr(output / "A/1/1", mode="r") as position:
        assert tuple(position["0"].shards) == tiled.shards
    other = {"config": {"version": 4}}
    assert _create(output, [_position()], names=["GFP_Density3D"], record=other) == {"A/1/0": (1,)}
    names, attrs = _attrs(output)
    assert names == ["Phase3D", "GFP_Density3D"]
    assert (attrs["waveorder-Phase3D"], attrs["waveorder-GFP_Density3D"]) == (RECORD, other)
    changed = RECORD | {"times": [0]}
    _create(output, [_position()], record=changed)
    assert _attrs(output)[1]["waveorder-Phase3D"] == changed  # a reconstruction's own record is replaced


def test_position_names_a_plate_cannot_hold_are_named_at_positions():
    (problem,) = name_problems(["run-1/cli4/single.zarr", "A/1/0", "A/1"])
    assert problem.location == "positions"
    assert "letters and digits only" in problem.message
    assert "give run-1/cli4/single.zarr, A/1;" in problem.message
    assert name_problems(["A/1/0"]) == []


def test_single_position_stores_among_several_positions_are_refused():
    assert name_problems([SINGLE_POSITION]) == []
    assert is_single([SINGLE_POSITION])
    assert not is_single([SINGLE_POSITION, "A/1/0"])
    (problem,) = name_problems([SINGLE_POSITION, "A/1/0"])
    assert problem.location == "positions"
    assert "2 positions and one is a single-position store" in problem.message
    assert "reconstruct each single-position store in a run of its own" in problem.message
    assert "plate.zarr/<row>/<column>/<field>" in problem.message
    (problem,) = name_problems([SINGLE_POSITION, SINGLE_POSITION])
    assert "all 2 are single-position stores" in problem.message
    assert len(name_problems([SINGLE_POSITION, "run-1/cli4/x.zarr"])) == 2  # the plate name check stays


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_one_single_position_writes_a_single_position_store_with_a_plate_positions_arrays(tmp_path, version):
    single = tmp_path / "single.zarr"
    with iohub.open_ome_zarr(single, layout="fov", mode="w-", channel_names=["BF"]) as node:
        node.create_zeros("0", shape=(2, 1, 4, 8, 8), dtype="uint16")
        node.zattrs["biahub-deskew"] = {"angle": 30}
    output, plate = tmp_path / "out.zarr", tmp_path / "plate.zarr"
    assert _create(output, [_position(SINGLE_POSITION)], version=version, sources=[single]) == {SINGLE_POSITION: (0,)}
    _create(plate, [_position()], version=version)
    with iohub.open_ome_zarr(output, mode="r") as node, iohub.open_ome_zarr(plate / "A/1/0", mode="r") as like:
        assert isinstance(node, iohub.ngff.Position)
        assert (node.version, node.channel_names) == (version, ["Phase3D"])
        assert node.get_effective_scale("0") == pytest.approx(SCALE)
        array, expected = node["0"], like["0"]
        assert (array.shape, array.dtype, array.chunks, array.shards) == (
            expected.shape,
            expected.dtype,
            expected.chunks,
            expected.shards,
        )
        attrs = dict(node.zattrs)
    assert attrs["waveorder-Phase3D"] == RECORD
    assert attrs["biahub-deskew"] == {"angle": 30}  # inherited from the input store
    existing = read_plate(output, single=True)
    assert existing is not None
    assert (existing.version, existing.shapes) == (version, {SINGLE_POSITION: (2, 1, 4, 8, 8)})
    tiled = choose_geometry((4, 4, 8), x=8)
    other = {"config": {"version": 4}}
    added = _create(output, [_position(SINGLE_POSITION, geometry=tiled)], names=["GFP_Density3D"], record=other)
    assert added == {SINGLE_POSITION: (1,)}  # an existing store keeps its array and gains the channel
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert node.channel_names == ["Phase3D", "GFP_Density3D"]
        assert dict(node.zattrs)["waveorder-Phase3D"] == RECORD


def test_a_single_position_output_takes_tile_geometry_and_records_its_time_points_beside_it(tmp_path, source):
    output = tmp_path / "out.zarr"
    tiled = choose_geometry((4, 4, 8), x=8)
    _create(output, [_position(SINGLE_POSITION, geometry=tiled)])
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert (tuple(node["0"].chunks), tuple(node["0"].shards)) == (tiled.chunks, tiled.shards)
    position = output / SINGLE_POSITION
    assert position == output
    assert process_time(_solve, source=source / "A/1/0", output=position, inputs=[0], outputs=[0], time=1, token="a")
    assert finished(position, time=1, channels=[0], token="a")
    assert not finished(position, time=0, channels=[0], token="a")
    assert (tmp_path / ".iohub-progress" / "out.zarr").is_dir()


def test_an_output_of_the_other_kind_is_refused(tmp_path):
    plate, single = tmp_path / "plate.zarr", tmp_path / "single.zarr"
    _create(plate, [_position()])
    _create(single, [_position(SINGLE_POSITION)])
    with pytest.raises(Rejected, match="is not a single-position store") as error:
        read_plate(plate, single=True)
    assert error.value.problems[0].location == str(plate)
    with pytest.raises(Rejected, match="is not an HCS plate"):
        read_plate(single)


def _solve(czyx: object) -> object:
    return torch.as_tensor(czyx)[:1] * 2


def test_a_time_point_is_written_through_iohub_and_recorded_under_its_token(tmp_path, source):
    output = tmp_path / "out.zarr"
    _create(output, [_position()])
    position = output / "A/1/0"
    assert not finished(position, time=1, channels=[0], token="a")
    written = process_time(_solve, source=source / "A/1/0", output=position, inputs=[0], outputs=[0], time=1, token="a")
    assert written
    with iohub.open_ome_zarr(position, mode="r") as node:
        image = torch.as_tensor(node["0"][:])
    with iohub.open_ome_zarr(source / "A/1/0", mode="r") as node:
        assert torch.equal(image[1], 2 * torch.as_tensor(node["0"][1]))
    assert image[0].abs().max() == 0
    assert finished(position, time=1, channels=[0], token="a")
    assert not finished(position, time=1, channels=[0], token="b")  # another reconstruction finished nothing
    assert not finished(position, time=0, channels=[0], token="a")
    assert (tmp_path / ".iohub-progress" / "out.zarr" / "A/1/0").is_dir()  # beside the store, not in it


def test_an_all_zero_input_is_not_solved_and_a_failed_read_names_the_position_and_time(tmp_path):
    plate = tmp_path / "in.zarr"
    volume = torch.ones(2, 1, 4, 8, 8)
    volume[0] = 0
    write_plate(plate, [volume])
    output = tmp_path / "out.zarr"
    _create(output, [_position()])
    position = output / "A/1/0"

    def never(czyx: object) -> object:
        raise AssertionError(f"an empty input is not solved, got {czyx!r}")

    assert not process_time(never, source=plate / "A/1/0", output=position, inputs=[0], outputs=[0], time=0, token="a")
    for chunk in (file for file in (plate / "A/1/0/0").rglob("*") if file.is_file() and file.name != "zarr.json"):
        chunk.write_bytes(b"not a chunk")
    with pytest.raises(ReadError, match=r"could not read time 1 of .*in\.zarr/A/1/0: "):
        process_time(_solve, source=plate / "A/1/0", output=position, inputs=[0], outputs=[0], time=1, token="a")


def test_a_time_point_the_caller_writes_is_recorded_only_once_the_block_ends(tmp_path):
    output = tmp_path / "out.zarr"
    _create(output, [_position()])
    position = output / "A/1/0"

    def interrupted() -> None:
        with tracked_time(position, time=0, channels=[0], token="a"):
            with iohub.open_ome_zarr(position, mode="r+") as node:
                node["0"][0, 0] = torch.ones(4, 8, 8).numpy()
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted()
    assert not finished(position, time=0, channels=[0], token="a")
    with tracked_time(position, time=0, channels=[0], token="a"), iohub.open_ome_zarr(position, mode="r+") as node:
        node["0"][0, 0] = torch.ones(4, 8, 8).numpy()
    assert finished(position, time=0, channels=[0], token="a")


def test_estimate_resources_counts_32_float32_copies_of_the_largest_volume():
    resources = estimate_resources([(10, 100, 100), (64, 2048, 2048)])
    assert resources.shape_zyx == (64, 2048, 2048)
    assert resources.host_bytes == 64 * 2048 * 2048 * 4 * 32
    assert resources.gb_ram == 32
    assert estimate_resources((1, 4, 4)).gb_ram == 1  # at least 1 GB, as 3.x requests
    assert waveorder.io.estimate_resources is estimate_resources
    with pytest.raises(ValueError, match="at least one"):
        estimate_resources([])
