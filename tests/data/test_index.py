import hashlib
import json
from collections import Counter
from multiprocessing import get_context
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.data.plates import PIXEL_SIZE, raw_field, target_field, write_plate
from tests.data.workers import hold_assignment
from waveorder.data import FovIndex, FovRow, GroupSplit, Holdout, PatchData, RandomPatches, index_from_plates
from waveorder.data.index import MANIFEST, SCHEMA, field_id
from waveorder.data.loader import _staged_bytes
from waveorder.errors import ParameterError, Rejected

WELLS = ("A/1", "A/2", "B/1", "B/2")


def _plate(path: Path, positions: list[str], *, times: int = 1, pixel_size=PIXEL_SIZE) -> Path:
    volumes = {name: raw_field(name, (1, 2, 16, 16)).expand(times, -1, -1, -1, -1).clone() for name in positions}
    return write_plate(path, volumes, channels=["BF"], pixel_size=pixel_size)


def _targets(path: Path, shapes: dict[str, tuple[int, int, int, int]]) -> Path:
    """Write a target plate of one ``Phase2D`` volume per position, of shape ``(T, Z, Y, X)``."""
    volumes = {
        name: target_field(name, (1, *shape[1:])).expand(shape[0], -1, -1, -1, -1).clone()
        for name, shape in shapes.items()
    }
    return write_plate(path, volumes, channels=["Phase2D"], pixel_size=PIXEL_SIZE)


def _pair(directory: Path, name: str, positions: list[str], *, times: int = 1, pixel_size=PIXEL_SIZE) -> Path:
    """Write the plate ``<name>.zarr`` and its target plate ``<name>.rec/<name>.zarr``; return the input plate."""
    plate = _plate(directory / f"{name}.zarr", positions, times=times, pixel_size=pixel_size)
    (directory / f"{name}.rec").mkdir()
    _targets(directory / f"{name}.rec" / f"{name}.zarr", dict.fromkeys(positions, (times, 1, 16, 16)))
    return plate


def _target_of(plate: Path) -> Path:
    return plate.parent / f"{plate.stem}.rec" / plate.name


def _index(plates: Path | list[Path], **settings) -> FovIndex:
    listed = [plates] if isinstance(plates, Path) else plates
    return index_from_plates(
        listed,
        channels=["BF"],
        targets=[_target_of(plate) for plate in listed],
        target_channels=["Phase2D"],
        **settings,
    )


def _positions(per_well: int, wells=WELLS) -> list[str]:
    return [f"{well}/{fov}" for well in wells for fov in range(per_well)]


@pytest.fixture(scope="module")
def two_plates(tmp_path_factory):
    directory = tmp_path_factory.mktemp("index")
    return [_pair(directory, f"exp{n}", _positions(8)) for n in (1, 2)]


def test_one_row_per_position_and_time(tmp_path):
    plate = _pair(tmp_path, "p", _positions(2, ("A/1",)), times=3)
    index = _index(plate, split=GroupSplit(by="plate", fractions={"train": 1.0}))
    assert len(index) == 6
    assert [(row.position, row.time) for row in index] == [(f"A/1/{fov}", t) for fov in range(2) for t in range(3)]
    row = index[1]
    assert (row.plate, row.well, row.planes, row.height, row.width) == ("p.zarr", "A/1", 2, 16, 16)
    assert row.name == "p.zarr/A/1/0@1"
    assert row.target == str(_target_of(plate).resolve())
    assert index[0].name == "p.zarr/A/1/0"
    assert index.column("field_id")[1] == row.field_id == field_id(row.input, "A/1/0", 1)
    assert index.pixel_size == PIXEL_SIZE
    assert index.planes == (2,)
    assert index.meta["dtype"] == "uint16"
    assert index.meta["target_planes"] == 1


def test_split_by_position_is_stratified_per_well_and_independent_of_input_order(two_plates):
    split = GroupSplit(by="position", fractions={"train": 0.5, "val": 0.25, "test": 0.25}, seed=4)
    index = _index(two_plates, split=split)
    again = _index(list(reversed(two_plates)), split=split)
    assert index.sha256 == again.sha256
    per_well = Counter((row.plate, row.well, row.split) for row in index)
    for plate in ("exp1.zarr", "exp2.zarr"):
        for well in WELLS:
            assert [per_well[plate, well, name] for name in ("train", "val", "test")] == [4, 2, 2]
    other = _index(two_plates, split=GroupSplit(by="position", fractions=split.fractions))
    assert [row.split for row in other] != [row.split for row in index]  # another seed, another split


@pytest.mark.parametrize("by", ["well", "plate"])
def test_split_by_well_or_plate_keeps_groups_whole(two_plates, by):
    fractions = {"train": 0.5, "val": 0.5}
    index = _index(two_plates, split=GroupSplit(by=by, fractions=fractions))
    group = (lambda row: (row.plate, row.well)) if by == "well" else (lambda row: row.plate)
    splits: dict[object, set[str]] = {}
    for row in index:
        splits.setdefault(group(row), set()).add(row.split)
    assert all(len(names) == 1 for names in splits.values())
    assert {name for names in splits.values() for name in names} == {"train", "val"}


def test_holdout_names_plates_and_wells(two_plates):
    split = GroupSplit(by="position", fractions={"train": 1.0}, holdout=["exp2.zarr", "exp1.zarr/B/2"])
    index = _index(two_plates, split=split)
    for row in index:
        held = row.plate == "exp2.zarr" or (row.plate, row.well) == ("exp1.zarr", "B/2")
        assert row.split == ("holdout" if held else "train")
    with pytest.raises(ParameterError, match="no plate and no plate/well"):
        _index(two_plates, split=GroupSplit(holdout=["exp3.zarr"]))


def test_per_group_keeps_that_many_positions_of_each_well(two_plates):
    index = _index(two_plates, per_group=3, seed=1)
    assert set(Counter((row.plate, row.well) for row in index).values()) == {3}
    assert index.sha256 == _index(two_plates, per_group=3, seed=1).sha256
    assert index.sha256 != _index(two_plates, per_group=3, seed=2).sha256
    assert index.meta["per_group"] == 3


def test_an_index_round_trips_through_its_directory_of_parquet_files(tmp_path, index):
    path = index.save(tmp_path / "index")
    assert sorted(file.name for file in path.iterdir()) == [MANIFEST, "raw.zarr.parquet"]
    manifest = json.loads((path / MANIFEST).read_text())
    assert manifest["schema"] == SCHEMA
    assert manifest["channels"] == ["GFP", "BF"]
    assert manifest["target_channels"] == ["Phase2D", "Focus3D"]
    assert manifest["pixel_size"] == [2.0, 0.65, 0.65]
    assert manifest["rows"] == 8
    assert manifest["counts"] == {"test": 2, "train": 4, "val": 2}
    assert manifest["splits"] == ["split"]
    assert manifest["split_rules"]["split"]["by"] == "position"
    (entry,) = manifest["files"]
    assert (entry["name"], entry["plate"], entry["rows"]) == ("raw.zarr.parquet", "raw.zarr", 8)
    assert len(entry["sha256"]) == 64

    loaded = FovIndex.load(path)
    assert list(loaded) == list(index)
    assert loaded.sha256 == index.sha256
    assert loaded.split_sha256 == index.split_sha256
    assert dict(loaded.meta) == dict(index.meta)
    assert (loaded.channels, loaded.target_channels) == (index.channels, index.target_channels)
    assert loaded.table.schema.field("plate").type == pa.dictionary(pa.int32(), pa.string())

    with pytest.raises(Rejected, match="overwrite=True"):
        index.save(path)
    assert index.save(path, overwrite=True) == path
    parquet = path / "raw.zarr.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"\0")
    with pytest.raises(Rejected, match="does not match the SHA-256"):
        FovIndex.load(path)
    parquet.unlink()
    with pytest.raises(Rejected, match=r"lacks raw\.zarr\.parquet"):
        FovIndex.load(path)
    (path / MANIFEST).write_text(json.dumps({**manifest, "schema": "waveorder.fov_index/1"}))
    with pytest.raises(Rejected, match="index the plates again"):
        FovIndex.load(path)
    with pytest.raises(Rejected, match=r"has no manifest\.json"):
        FovIndex.load(tmp_path)


def test_adding_a_plate_writes_its_file_and_the_manifest_only(tmp_path, two_plates):
    split = GroupSplit(by="position", fractions={"train": 0.5, "val": 0.5})
    first, second = (_index(plate, split=split) for plate in two_plates)
    path = first.save(tmp_path / "index")
    written = path / "exp1.zarr.parquet"
    before = (written.stat().st_mtime_ns, written.read_bytes())
    second.save(path, append=True)
    assert (written.stat().st_mtime_ns, written.read_bytes()) == before
    manifest = json.loads((path / MANIFEST).read_text())
    assert [entry["plate"] for entry in manifest["files"]] == ["exp1.zarr", "exp2.zarr"]
    assert manifest["rows"] == len(first) + len(second)
    both = FovIndex.load(path)
    assert both.sha256 == _index(two_plates, split=split).sha256  # the same rows as indexing both at once
    assert both.counts == {"train": 32, "val": 32}

    with pytest.raises(Rejected, match=r"already indexes exp2\.zarr"):
        second.save(path, append=True)
    second.save(path, append=True, overwrite=True)  # indexed again: its file is replaced
    assert FovIndex.load(path).sha256 == both.sha256
    other = FovIndex(second.table, channels=["GFP"], target_channels=["Phase2D"], meta=second.meta)
    with pytest.raises(Rejected, match="indexes channels"):
        other.save(path, append=True, overwrite=True)


def _plate_splits(path: Path) -> dict[str, str]:
    """Return the split of each plate of the index at ``path``, by input path."""
    return {row.input: row.split for row in FovIndex.load(path)}


def test_plates_appended_one_at_a_time_approach_the_fractions_of_a_split_by_plate(tmp_path):
    plates = [_pair(tmp_path, f"p{n}", _positions(1, ("A/1",))) for n in range(6)]
    split = GroupSplit(by="plate", fractions={"train": 0.5, "val": 0.5})
    path = tmp_path / "index"
    _index(plates[0], split=split, append_to=path).save(path, append=True)
    with pytest.raises(Rejected, match="ignores; build them with index_from_plates"):
        _index(plates[1], split=split).save(path, append=True)  # split alone: train, beside a train plate
    seen: dict[str, str] = {}
    for plate in plates[1:]:
        _index(plate, split=split, append_to=path).save(path, append=True)
        now = _plate_splits(path)
        assert {name: now[name] for name in seen} == seen  # a recorded plate never moves
        seen = now
    assert [seen[str(plate)] for plate in plates] == ["train", "val"] * 3
    assert Counter(seen.values()) == {"train": 3, "val": 3}

    joint = GroupSplit(by="plate", fractions=split.fractions).assign([(f"p{n}.zarr", "A/1", "A/1/0") for n in range(6)])
    assert Counter(joint) == {"train": 3, "val": 3}  # a first build still cuts the ranked plates
    recorded = [("a", "A/1", "A/1/0", "train"), ("b", "A/1", "A/1/0", "train")]
    assert split.assign([("c", "A/1", "A/1/0"), ("a", "A/1", "A/1/0")], recorded) == ["val", "train"]
    with pytest.raises(ParameterError, match="assigns a to the splits train and val"):
        split.assign([("c", "A/1", "A/1/0")], [("a", "A/1", "A/1/0", "train"), ("a", "A/2", "A/2/0", "val")])


def test_plates_of_one_name_in_two_folders_are_appended_apart(tmp_path):
    folders = [tmp_path / name for name in ("experiment1", "experiment2")]
    for folder in folders:
        folder.mkdir()
    first, second = (_pair(folder, "raw", _positions(2, ("A/1",))) for folder in folders)
    split = GroupSplit(by="plate", fractions={"train": 0.5, "val": 0.5})
    path = tmp_path / "index"
    _index(first, split=split, append_to=path).save(path, append=True)
    with pytest.raises(Rejected, match=r"another plate named raw\.zarr"):
        _index(second, split=split).save(path, append=True)  # named alone: raw.zarr
    _index(second, split=split, append_to=path).save(path, append=True)
    both = FovIndex.load(path)
    assert {(row.plate, row.input, row.split) for row in both} == {
        ("raw.zarr", str(first), "train"),
        ("experiment2", str(second), "val"),
    }
    assert sorted(entry["name"] for entry in json.loads((path / MANIFEST).read_text())["files"]) == [
        "experiment2.parquet",
        "raw.zarr.parquet",
    ]

    again = _index(second, split=split, append_to=path)
    assert set(again.column("plate")) == {"experiment2"}  # a plate keeps the name the index gave it
    with pytest.raises(Rejected, match=r"already indexes experiment2"):
        again.save(path, append=True)
    with pytest.raises(Rejected, match=r"already indexes raw\.zarr"):
        _index(first, split=split).save(path, append=True)
    again.save(path, append=True, overwrite=True)
    assert FovIndex.load(path).sha256 == both.sha256


def test_an_index_written_before_plates_were_known_by_path_loads_and_takes_plates(tmp_path):
    first, second = (_pair(tmp_path, name, _positions(2, ("A/1",))) for name in ("a", "b"))
    split = GroupSplit(by="plate", fractions={"train": 0.5, "val": 0.5})
    built = _index(first, split=split)
    path = tmp_path / "index"
    path.mkdir()
    pq.write_table(built.table, path / "a.zarr.parquet")
    manifest = {
        "schema": "waveorder.fov_index/2",
        "channels": ["BF"],
        "target_channels": ["Phase2D"],
        "attributes": [],
        "splits": ["split"],
        "rows": 2,
        "counts": {"train": 2},
        "files": [
            {
                "name": "a.zarr.parquet",
                "plate": "a.zarr",
                "rows": 2,
                "counts": {"train": 2},
                "sha256": hashlib.sha256((path / "a.zarr.parquet").read_bytes()).hexdigest(),
            }
        ],
        **built.meta,
    }
    (path / MANIFEST).write_text(json.dumps(manifest))
    assert FovIndex.load(path).sha256 == built.sha256
    _index(second, split=split, append_to=path).save(path, append=True)
    assert FovIndex.load(path).counts == {"train": 2, "val": 2}


def test_named_splits_are_columns_a_run_picks(tmp_path, two_plates):
    by_position = GroupSplit(by="position", fractions={"train": 0.5, "val": 0.25, "test": 0.25})
    by_plate = GroupSplit(by="plate", fractions={"train": 0.5, "val": 0.5}, seed=3)
    index = _index(two_plates, split=by_position, splits={"by_plate": by_plate})
    assert index.splits == ("split", "split_by_plate")
    loaded = FovIndex.load(index.save(tmp_path / "index"))
    assert loaded.splits == ("split", "split_by_plate")
    assert json.loads((tmp_path / "index" / MANIFEST).read_text())["split_rules"]["by_plate"]["by"] == "plate"

    picked = loaded.assign("split_by_plate")
    assert picked.column("split") == loaded.column("split_by_plate")
    per_plate = {(row.plate, row.split) for row in picked}
    assert len(per_plate) == 2  # each plate whole in one split
    assert picked.split_sha256 != loaded.split_sha256
    assert loaded.assign().split_sha256 == loaded.split_sha256  # the default column changes nothing
    data = PatchData(picked, patches=RandomPatches((8, 8)), workers=0)
    assert {row.plate for row in data.assignment("train")} == {row.plate for row in picked if row.split == "train"}
    with pytest.raises(ParameterError, match="no split column 'split_other'"):
        loaded.assign("split_other")
    with pytest.raises(ParameterError, match="split names are identifiers"):
        _index(two_plates, splits={"by plate": by_plate})


def test_a_holdout_given_at_load_overrides_the_stored_split(tmp_path, two_plates):
    loaded = FovIndex.load(_index(two_plates).save(tmp_path / "index"))
    held = loaded.assign(holdout=Holdout(plates=["exp2.zarr"], wells=["exp1.zarr/B/2"]))
    for before, after in zip(loaded, held, strict=True):
        out = before.plate == "exp2.zarr" or (before.plate, before.well) == ("exp1.zarr", "B/2")
        assert after.split == ("holdout" if out else before.split)
    assert held.counts["holdout"] == 32 + 8
    assert held.split_sha256 != loaded.split_sha256
    named = loaded.assign("split", holdout=Holdout(wells=["exp1.zarr/A/1"]))
    assert named.counts["holdout"] == 8
    with pytest.raises(ParameterError, match=r"names exp3.zarr, which is no plate"):
        loaded.assign(holdout=Holdout(plates=["exp3.zarr"]))
    with pytest.raises(ParameterError, match=r"names exp1.zarr/C/1, which is no plate/row/column"):
        loaded.assign(holdout=Holdout(wells=["exp1.zarr/C/1"]))


def test_voxel_sizes_must_agree_unless_allowed(tmp_path):
    first = _pair(tmp_path, "a", _positions(1, ("A/1",)))
    second = _pair(tmp_path, "b", _positions(1, ("A/1",)), pixel_size=(2.0, 0.66, 0.66))
    split = GroupSplit(by="plate", fractions={"train": 1.0})
    with pytest.raises(Rejected, match="allow_pixel_size_mismatch"):
        _index([first, second], split=split)
    index = _index([first, second], split=split, allow_pixel_size_mismatch=True)
    assert len(index) == 2


def test_targets_must_exist_for_every_position(tmp_path):
    inputs = _plate(tmp_path / "raw.zarr", _positions(2, ("A/1",)))
    targets = _targets(tmp_path / "rec.zarr", {"A/1/0": (1, 1, 16, 16)})
    with pytest.raises(Rejected, match="1 of 2 positions have no target"):
        index_from_plates(inputs, channels=["BF"], targets=[targets], target_channels=["Phase2D"])
    with pytest.raises(ParameterError, match="one plate per entry of inputs"):
        index_from_plates(inputs, channels=["BF"], targets=[targets, targets], target_channels=["Phase2D"])


def test_every_target_position_is_checked_and_the_deepest_sizes_the_staging_buffers(tmp_path):
    inputs = _plate(tmp_path / "raw.zarr", _positions(2, ("A/1",)), times=2)

    def index(targets: Path) -> FovIndex:
        return index_from_plates(inputs, channels=["BF"], targets=[targets], target_channels=["Phase2D"])

    short = _targets(tmp_path / "short.zarr", {"A/1/0": (2, 1, 16, 16), "A/1/1": (1, 1, 16, 16)})
    with pytest.raises(Rejected, match=r"A/1/1: holds \(T, Z, Y, X\) = \(1, 1, 16, 16\), but its input"):
        index(short)
    narrow = _targets(tmp_path / "narrow.zarr", {"A/1/0": (2, 1, 16, 16), "A/1/1": (2, 1, 16, 8)})
    with pytest.raises(Rejected, match="a target has its input's Y and X"):
        index(narrow)
    deep = index(_targets(tmp_path / "deep.zarr", {"A/1/0": (2, 1, 16, 16), "A/1/1": (2, 3, 16, 16)}))
    assert deep.meta["target_planes"] == 3
    staged = _staged_bytes(deep.table, deep.meta, channels=1, target_channels=1)
    assert staged == (2 * 16 * 16 * 2, 3 * 16 * 16 * 4)  # the data in uint16; the deepest target, float32


def test_from_rows_sorts_naturally_and_reads_attributes_and_named_splits(tmp_path):
    plate = _pair(tmp_path, "p", ["A/1/2", "A/1/10", "A/2/0"])
    table = [
        {"input": str(plate), "target": str(_target_of(plate)), "position": name, "split": split, "dx": str(number)}
        | {"split_alt": "train" if name == "A/2/0" else "val"}
        for number, (name, split) in enumerate([("A/2/0", "val"), ("A/1/10", "train"), ("A/1/2", "train")])
    ]
    index = FovIndex.from_rows(table, channels=["BF"], target_channels=["Phase2D"], attributes=["dx"])
    assert [row.position for row in index] == ["A/1/2", "A/1/10", "A/2/0"]
    assert [row.attributes["dx"] for row in index] == [2.0, 1.0, 0.0]
    assert index.counts == {"train": 2, "val": 1}
    assert index.split("val")[0].position == "A/2/0"
    assert index.assign("split_alt").counts == {"train": 1, "val": 2}
    with pytest.raises(ParameterError, match="finite number"):
        FovIndex.from_rows(
            [{**table[0], "dx": "east"}], channels=["BF"], target_channels=["Phase2D"], attributes=["dx"]
        )
    with pytest.raises(ParameterError, match="lacks the column 'target'"):
        FovIndex.from_rows([{"input": str(plate), "position": "A/1/2"}], channels=["BF"], target_channels=["X"])
    with pytest.raises(ParameterError, match="no rows in split 'test'"):
        index.split("test")


def test_rows_must_be_distinct_and_have_a_target():
    row = FovRow(input="p.zarr", position="A/1/0", target="r.zarr", plate="p", well="A/1", planes=1, height=8, width=8)
    with pytest.raises(ParameterError, match="twice"):
        FovIndex.of([row, row], channels=["BF"], target_channels=["Phase2D"])
    untargeted = FovRow(input="p.zarr", position="A/1/0", target="", plate="p", well="A/1", planes=1, height=8, width=8)
    with pytest.raises(ParameterError, match="a row has no target plate"):
        FovIndex.of([untargeted], channels=["BF"], target_channels=["Phase2D"])
    with pytest.raises(ParameterError, match="lacks the columns split"):
        FovIndex(FovIndex.of([row], channels=["BF"], target_channels=["P"]).table.drop(["split"]), channels=["BF"],
                 target_channels=["P"])  # fmt: skip


def _synthetic_index(rows: int) -> FovIndex:
    """An index of ``rows`` fields with paths as long as a cluster's, built as columns (nothing is read)."""
    plates = [f"/hpc/projects/intracellular_dashboard/ops/2026_11_{n:02d}_ops{n:04d}/0-convert/phase.zarr" for n in
              range(40)]  # fmt: skip
    plate = [plates[number % 40] for number in range(rows)]
    fovs = [number // 40 for number in range(rows)]  # with the plate, one position per row
    position = [f"{'ABCDEFGH'[fov % 8]}/{fov // 8 % 12 + 1}/{fov // 96}" for fov in fovs]
    table = pa.table({
        "field_id": pa.array([field_id(p, q, 0) for p, q in zip(plate, position, strict=True)], pa.int64()),
        "input": pa.array(plate).dictionary_encode(),
        "target": pa.array([name.replace("0-convert", "1-reconstruct") for name in plate]).dictionary_encode(),
        "plate": pa.array([name.split("/")[-3] for name in plate]).dictionary_encode(),
        "well": pa.array(["/".join(name.split("/")[:2]) for name in position]).dictionary_encode(),
        "position": pa.array(position).dictionary_encode(),
        "time": pa.array([0] * rows, pa.int32()),
        "planes": pa.array([9] * rows, pa.int32()),
        "height": pa.array([512] * rows, pa.int32()),
        "width": pa.array([512] * rows, pa.int32()),
        "split": pa.array(["train"] * rows).dictionary_encode(),
    })  # fmt: skip
    return FovIndex(table, channels=["BF"], target_channels=["Phase2D"], meta={"pixel_size": [2.0, 0.65, 0.65]})


def test_a_loader_worker_holds_its_assignment_not_the_index():
    """At 1e5 rows a worker's private memory grows by a few MB; the spike measured 65 MB with the old design.

    The worker is a spawned process that imports what a loader worker imports, then receives the stream as a loader
    sends it (the columns as a handle to shared memory) and makes every row of its epoch's assignment: 50,000 rows,
    whose objects would take about 25 MB held at once.
    """
    rows = 100_000
    data = PatchData(_synthetic_index(rows), patches=RandomPatches((256, 256), per_field=4), workers=2)
    stream = data._stream("train", train=True)  # noqa: SLF001 (what a loader gives its workers)
    assert len(ForkingPickler.dumps(stream)) < 16384  # a handle, not the rows
    small = PatchData(_synthetic_index(100), patches=RandomPatches((256, 256)), workers=2)
    context = get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=hold_assignment, args=(child, 1, 3))
    worker.start()
    try:
        parent.send(small._stream("train", train=True))  # noqa: SLF001 (the warm-up)
        parent.send(stream)
        assert parent.poll(120)
        made, private, peak, arrow = parent.recv()
    finally:
        worker.join(30)
    assert worker.exitcode == 0
    measured = f"{made} rows, private +{private / 1e6:.1f} MB, Python peak {peak / 1e6:.1f} MB"
    assert made == rows // 2
    assert arrow == 0  # the columns are read in place: arrow allocated nothing
    assert peak < 8e6, measured  # rows are made a chunk at a time: measured 5.6 MB
    assert private < 16e6, measured  # measured 11.9 MB
