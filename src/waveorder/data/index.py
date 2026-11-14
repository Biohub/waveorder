"""The table of fields of view a network is trained and evaluated on, and how it is split.

A :class:`FovIndex` has one row per field of view and time point, held as columns (an Arrow table), never as one
object per row: :func:`index_from_plates` builds one from OME-Zarr HCS plates and :meth:`FovIndex.from_rows` from a
table of your own. Every row names its input plate and the plate of its recorded target. Splits are columns:
``split``, assigned by a :class:`GroupSplit` by position, well or plate, and any named split ``split_<name>``;
:meth:`FovIndex.assign` picks the column a run uses and holds out plates or wells (:class:`Holdout`).

An index is saved as a directory: one parquet file per plate and ``manifest.json``, which records the schema, the
channels, the target channels, the voxel size, the split columns and each file's SHA-256 and row count;
:meth:`FovIndex.load` checks them. ``save(..., append=True)`` adds plates by writing their files and the manifest;
``index_from_plates(..., append_to=path)`` builds them against the index at ``path``: a plate is known by its path
(the ``input`` column), keeps the name the index gives it, and a split by group counts the groups already assigned.
"""

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from waveorder.errors import ParameterError, Rejected
from waveorder.io.positions import (
    PositionMeta,
    channel_indices,
    expand_positions,
    plate_name,
    plate_of,
    read_position_meta,
)
from waveorder.nn.sampling import ZYX

SCHEMA = "waveorder.fov_index/2"
"""The ``schema`` entry of an index's manifest."""

MANIFEST = "manifest.json"
"""The file of an index directory that records its parquet files."""

SPLIT = "split"
"""The split column a run reads by default; named splits are the columns ``split_<name>``."""

HOLDOUT = "holdout"
"""The split of the rows a :class:`GroupSplit` or a :class:`Holdout` keeps out."""

PIXEL_SIZE_TOLERANCE = 0.01
"""Largest relative difference between the voxel sizes of two plates of one index."""

_STRINGS = ("input", "target", "plate", "well", "position")
_INTEGERS = ("time", "planes", "height", "width")
_DERIVED = ("schema", "rows", "channels", "target_channels", "attributes", "splits", "counts", "files")
"""The manifest entries written from the table itself; the index's ``meta`` may not use these names."""

type Level = Literal["position", "well", "plate"]

type AttributeFunction = Callable[[Path, str], Mapping[str, float]]
"""``(plate_path, position_name) -> {name: value}``: the numeric attributes of one position."""


def field_id(input_plate: str, position: str, time: int) -> int:
    """Return the 63-bit id of a field: a hash of its input plate's path, its position and its time."""
    digest = hashlib.sha256(f"{input_plate}/{position}@{time}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


@dataclass(frozen=True, slots=True, kw_only=True)
class FovRow:
    """One field of view at one time point: what a loader worker reads.

    Parameters
    ----------
    input : str
        Path of the OME-Zarr HCS plate the data is read from.
    position : str
        The position's name within the plate, ``row/column/fov``; the target is read from the same name in
        ``target``.
    target : str
        Path of the plate the recorded target is read from.
    time : int
        Index along the time axis.
    split : str
        The split the row belongs to.
    plate : str
        The plate's name, the group a split by plate assigns as a whole.
    well : str
        The well's name within the plate, ``row/column``.
    planes, height, width : int
        ``Z``, ``Y`` and ``X`` of the data.
    attributes : Mapping[str, float]
        Numeric values of the position by name, what an ``Attributes`` condition reads.
    """

    input: str
    position: str
    target: str
    time: int = 0
    split: str = "train"
    plate: str
    well: str
    planes: int
    height: int
    width: int
    attributes: Mapping[str, float] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """The input position on disk, ``input/position``."""
        return Path(self.input) / self.position

    @property
    def target_path(self) -> Path:
        """The target position on disk, ``target/position``."""
        return Path(self.target) / self.position

    @property
    def name(self) -> str:
        """The row's name in messages and in a batch's ``position``: ``plate/position``, with ``@t`` after time 0."""
        return f"{self.plate}/{self.position}" + (f"@{self.time}" if self.time else "")

    @property
    def field_id(self) -> int:
        """The row's id, :func:`field_id` of its input plate, position and time: the unit of sharding."""
        return field_id(self.input, self.position, self.time)


@dataclass(frozen=True, slots=True)
class GroupSplit:
    """A rule that assigns fields of view to splits by group, the same way for a seed whatever the order of the rows.

    Units (positions, wells or plates) are ranked by a hash of the seed and their name, and the ranked list is cut at
    the cumulative fractions, each cut rounded to the nearest unit. Every time point of a position is in one split.
    Units added to an index that assigns units of their stratum already (``recorded``) never move a recorded unit:
    in ranked order, each new unit goes to the split furthest below its fraction, the one whose count falls short of
    ``fraction * (units assigned + 1)`` by the most, the first of ``fractions`` on a tie.

    Parameters
    ----------
    by : {"position", "well", "plate"}
        What is assigned as a whole. ``"position"`` splits the positions within each well; ``"well"`` and
        ``"plate"`` put no well (plate) in two splits.
    fractions : Mapping[str, float]
        Split name to the fraction of units it gets, in the order the ranked units are cut; they sum to 1.
    holdout : Collection[str]
        Plate names, or ``plate/well`` names (``ops0032/A/1``), whose rows get the split ``holdout``.
    stratify : {"well", "plate"}, optional
        Split the units within each well or each plate separately; it must be coarser than ``by``. ``None`` is each
        well for ``by="position"`` and all units ranked together otherwise.
    seed : int
        Seed of the ranking.

    Raises
    ------
    ParameterError
        If ``stratify`` is not coarser than ``by``, or ``fractions`` is empty, names a split ``holdout``, holds a
        value outside [0, 1] or does not sum to 1.
    """

    by: Level = "well"
    fractions: Mapping[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.1, "test": 0.2})
    holdout: Collection[str] = ()
    stratify: Literal["well", "plate"] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        levels = ("position", "well", "plate")
        coarser = levels[levels.index(self.by) + 1 :]
        if self.stratify is not None and self.stratify not in coarser:
            allowed = ", ".join(coarser) or "nothing"
            raise ParameterError(
                f"GroupSplit stratify must be coarser than by={self.by!r} ({allowed}, or None), got {self.stratify!r}",
                path="stratify",
            )
        object.__setattr__(self, "fractions", _fractions(self.fractions))
        object.__setattr__(self, "holdout", tuple(sorted(str(name) for name in self.holdout)))

    def assign(
        self, fields: Iterable[tuple[str, str, str]], recorded: Iterable[tuple[str, str, str, str]] = ()
    ) -> list[str]:
        """Return the split of each ``(plate, well, position)`` field, in the order given.

        A field listed more than once (once per time point) gets one split. ``recorded`` lists the
        ``(plate, well, position, split)`` of the fields an index assigns already; their units keep their splits.
        Raises :class:`ParameterError` if a ``holdout`` name is no plate and no ``plate/well`` of the fields, or a
        recorded unit is in two splits.
        """
        fields = list(fields)
        held = [plate in self.holdout or f"{plate}/{well}" in self.holdout for plate, well, _ in fields]
        _expect_known(self.holdout, fields, where="GroupSplit holdout")
        known, tallies = self._recorded(recorded)
        strata: dict[str, set[str]] = {}
        for (plate, well, position), out in zip(fields, held, strict=True):
            if not out:
                strata.setdefault(self._stratum(plate, well), set()).add(self._unit(plate, well, position))
        assigned = dict(known)
        for stratum, units in strata.items():
            new = units - known.keys()
            assigned.update(self._fill(new, tallies[stratum]) if stratum in tallies else self._cut(new))
        return [
            HOLDOUT if out else assigned[self._unit(plate, well, position)]
            for (plate, well, position), out in zip(fields, held, strict=True)
        ]

    def arguments(self) -> dict[str, object]:
        """Return the rule as JSON-able constructor arguments: ``GroupSplit(**split.arguments())`` rebuilds it."""
        return {
            "by": self.by,
            "fractions": dict(self.fractions),
            "holdout": list(self.holdout),
            "stratify": self.stratify,
            "seed": self.seed,
        }

    def _unit(self, plate: str, well: str, position: str) -> str:
        return {"plate": plate, "well": f"{plate}/{well}", "position": f"{plate}/{position}"}[self.by]

    def _stratum(self, plate: str, well: str) -> str:
        """Return the name of the stratum a field's unit is ranked in; empty when all units are ranked together."""
        level = self.stratify or ("well" if self.by == "position" else None)
        return {"plate": plate, "well": f"{plate}/{well}", None: ""}[level]

    def _recorded(
        self, recorded: Iterable[tuple[str, str, str, str]]
    ) -> tuple[dict[str, str], dict[str, Counter[str]]]:
        """Return the split of each recorded unit but the held out ones, and the units per split of each stratum."""
        known: dict[str, str] = {}
        tallies: dict[str, Counter[str]] = {}
        for plate, well, position, name in recorded:
            if name == HOLDOUT:
                continue
            unit = self._unit(plate, well, position)
            if unit not in known:
                known[unit] = name
                tallies.setdefault(self._stratum(plate, well), Counter())[name] += 1
            elif known[unit] != name:
                raise ParameterError(
                    f"GroupSplit by={self.by!r}: the index assigns {unit} to the splits {known[unit]} and {name}, so "
                    "it was not split by this level; index the plates again together",
                    path="by",
                )
        return known, tallies

    def _ranked(self, units: Collection[str]) -> list[str]:
        return sorted(units, key=lambda unit: hashlib.sha256(f"{self.seed}:{unit}".encode()).digest())

    def _cut(self, units: set[str]) -> dict[str, str]:
        """Rank ``units`` by the seeded hash of their names and cut the list at the cumulative fractions."""
        ranked = self._ranked(units)
        names = list(self.fractions)
        assigned: dict[str, str] = {}
        start, cumulative = 0, 0.0
        for name in names:
            cumulative += self.fractions[name]
            stop = len(ranked) if name == names[-1] else min(len(ranked), math.floor(cumulative * len(ranked) + 0.5))
            assigned.update(dict.fromkeys(ranked[start:stop], name))
            start = max(start, stop)
        return assigned

    def _fill(self, units: set[str], tally: Counter[str]) -> dict[str, str]:
        """Give each of ``units``, in ranked order, the split furthest below its fraction of the units so far."""
        names = list(self.fractions)
        counts = Counter({name: tally[name] for name in names})
        assigned: dict[str, str] = {}
        for unit in self._ranked(units):
            total = counts.total() + 1
            short = [round(self.fractions[name] * total - counts[name], 9) for name in names]
            name = names[short.index(max(short))]
            assigned[unit] = name
            counts[name] += 1
        return assigned


@dataclass(frozen=True, slots=True)
class Holdout:
    """Plates and wells a run keeps out of its splits: their rows get the split ``holdout`` when the index is loaded.

    Parameters
    ----------
    plates : Sequence[str]
        Plate names, as the index's ``plate`` column holds them.
    wells : Sequence[str]
        Wells as ``plate/row/column`` (``ops0032/A/1``).
    """

    plates: Sequence[str] = ()
    wells: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plates", tuple(self.plates))
        object.__setattr__(self, "wells", tuple(self.wells))


def _fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    """Return the fractions of a split as a dict, or raise naming what is wrong."""
    if not fractions:
        raise ParameterError("GroupSplit fractions must map split names to fractions, got none", path="fractions")
    for name, value in fractions.items():
        if not name or name == HOLDOUT:
            raise ParameterError(
                f"GroupSplit fractions names a split {name!r}; a split name is a non-empty string other than "
                f"{HOLDOUT!r}, which is the split of the holdout groups",
                path="fractions",
            )
        if not 0 <= value <= 1:
            raise ParameterError(
                f"GroupSplit fractions.{name} must be a number between 0 and 1, got {value!r}",
                path=f"fractions.{name}",
            )
    checked = {name: float(value) for name, value in fractions.items()}
    if abs(sum(checked.values()) - 1.0) > 1e-6:
        raise ParameterError(
            f"GroupSplit fractions must sum to 1, got {sum(checked.values()):g} from {checked}", path="fractions"
        )
    return checked


def _expect_known(names: Collection[str], fields: Sequence[tuple[str, str, str]], *, where: str) -> None:
    """Refuse names that are no plate and no ``plate/well`` of the ``(plate, well, position)`` fields."""
    known = {plate for plate, _, _ in fields} | {f"{plate}/{well}" for plate, well, _ in fields}
    unknown = [name for name in names if name not in known]
    if unknown:
        plates = ", ".join(sorted({plate for plate, _, _ in fields}))
        raise ParameterError(
            f"{where} names {', '.join(unknown)}, which is no plate and no plate/well of the index; the plates are "
            f"{plates}",
            path="holdout",
        )


@dataclass(frozen=True, eq=False, repr=False)
class FovIndex:
    """A table of fields of view with the channels read from them: what a ``PatchData`` loads.

    The constructor takes the table and reads nothing; build an index with :func:`index_from_plates`,
    :meth:`from_rows`, :meth:`of` or :meth:`load`. Iterating it yields :class:`FovRow` objects, made one at a time.

    Parameters
    ----------
    table : pyarrow.Table
        One row per field and time point: ``field_id`` (int64), ``input``, ``target``, ``plate``, ``well``,
        ``position`` (strings), ``time``, ``planes``, ``height``, ``width`` (int32), ``split`` and any
        ``split_<name>`` (strings), and one float32 column per attribute.
    channels : Sequence[str]
        Names of the input channels read from every row's position, in order.
    target_channels : Sequence[str]
        Names of the channels read from every row's target position, in order.
    attributes : Sequence[str]
        Names of the numeric attributes every row carries.
    meta : Mapping[str, object]
        JSON-able facts saved in the manifest (read-only): ``pixel_size`` (``[z, y, x]`` in um), ``dtype``,
        ``target_planes``, ``target_dtype`` and the split rules when built by this module.

    Raises
    ------
    ParameterError
        If a column is missing, two rows have one field id, a row has no target, ``channels`` or
        ``target_channels`` is empty, or ``meta`` uses a name the manifest derives from the table.
    """

    table: pa.Table
    _: KW_ONLY
    channels: Sequence[str]
    target_channels: Sequence[str]
    attributes: Sequence[str] = ()
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("channels", tuple(self.channels)),
            ("target_channels", tuple(self.target_channels)),
            ("attributes", tuple(self.attributes)),
            ("meta", MappingProxyType(dict(self.meta))),
        ):
            object.__setattr__(self, name, value)
        for name in ("channels", "target_channels"):
            if not getattr(self, name):
                raise ParameterError(f"FovIndex {name} must name at least one channel", path=name)
        clash = sorted(set(self.meta) & set(_DERIVED))
        if clash:
            raise ParameterError(
                f"FovIndex meta may not use the names {', '.join(clash)}: the manifest writes them from the table",
                path="meta",
            )
        missing = [name for name in ("field_id", *_STRINGS, *_INTEGERS, SPLIT, *self.attributes) if name not in self]
        if missing:
            raise ParameterError(f"FovIndex table lacks the columns {', '.join(missing)}", path="table")
        if len(_compute("unique", self.table["field_id"])) != len(self):
            raise ParameterError("FovIndex rows list one field (plate, position and time) twice", path="table")
        lengths = _compute("utf8_length", _plain(self.table["target"]))
        if len(self) and not _compute("all", _compute("not_equal", lengths, 0)).as_py():
            raise ParameterError(
                "FovIndex: a row has no target plate; every row of an index is a recorded pair", path="table"
            )

    # --- the table ---------------------------------------------------------------------------------------------------

    @classmethod
    def of(
        cls,
        rows: Sequence[FovRow],
        *,
        channels: Sequence[str],
        target_channels: Sequence[str],
        attributes: Sequence[str] = (),
        meta: Mapping[str, object] | None = None,
        splits: Mapping[str, Sequence[str]] | None = None,
    ) -> Self:
        """Build an index of complete rows, in their order; ``splits`` adds named split columns (``split_<name>``)."""
        columns: dict[str, pa.Array] = {
            "field_id": pa.array([row.field_id for row in rows], pa.int64()),
            **{name: _strings([getattr(row, name) for row in rows]) for name in _STRINGS},
            **{name: pa.array([getattr(row, name) for row in rows], pa.int32()) for name in _INTEGERS},
            SPLIT: _strings([row.split for row in rows]),
        }
        for name, values in (splits or {}).items():
            columns[f"{SPLIT}_{name}"] = _strings(list(values))
        for name in attributes:
            columns[name] = pa.array([float(row.attributes[name]) for row in rows], pa.float32())
        return cls(
            pa.table(columns),
            channels=channels,
            target_channels=target_channels,
            attributes=attributes,
            meta=meta or {},
        )

    def __contains__(self, column: object) -> bool:
        return column in self.table.column_names

    def __len__(self) -> int:
        return self.table.num_rows

    def __iter__(self) -> Iterator[FovRow]:
        return iter_rows(self.table, range(len(self)), self.attributes)

    def __getitem__(self, item: int) -> FovRow:
        return self.take([item])[0]

    def take(self, positions: Iterable[int]) -> list[FovRow]:
        """Return the rows at ``positions`` (numbers of rows of the table), in that order."""
        return take_rows(self.table, positions, self.attributes)

    def column(self, name: str) -> list:
        """Return one column as a list of Python values."""
        return self.table[name].to_pylist()

    @property
    def splits(self) -> tuple[str, ...]:
        """The split columns: ``split`` and every named split ``split_<name>``."""
        return tuple(name for name in self.table.column_names if name == SPLIT or name.startswith(f"{SPLIT}_"))

    @property
    def pixel_size(self) -> ZYX[float]:
        """Voxel size of the data along z, y, x in um, from ``meta["pixel_size"]``; raise if it records none."""
        value = self.meta.get("pixel_size")
        if (
            not isinstance(value, list | tuple)
            or len(value) != 3
            or not all(isinstance(size, int | float) and not isinstance(size, bool) and size > 0 for size in value)
        ):
            raise ParameterError(
                f"the index records no voxel size (meta['pixel_size'] is {value!r}); build it with "
                "index_from_plates or FovIndex.from_rows, which read it from the plates, or pass "
                "meta={'pixel_size': [z, y, x]} in um",
                path="meta.pixel_size",
            )
        z, y, x = (float(size) for size in value)
        return (z, y, x)

    @property
    def planes(self) -> tuple[int, ...]:
        """The plane counts ``Z`` of the rows, each once, ascending."""
        return tuple(sorted(_compute("unique", self.table["planes"]).to_pylist()))

    @property
    def counts(self) -> dict[str, int]:
        """Number of rows per split, by split name in alphabetical order."""
        return _counts(self.table, SPLIT)

    @cached_property
    def sha256(self) -> str:
        """SHA-256 of the table's values, columns in order: two indexes of the same rows have the same hash."""
        return _digest(self.table)

    @property
    def split_sha256(self) -> str:
        """SHA-256 of the split assignment, the ``field_id`` and ``split`` of every row: what a run records."""
        return _digest(self.table.select(["field_id", SPLIT]))

    def split(self, name: str) -> Self:
        """Return the index of the rows whose split is ``name``; raise :class:`ParameterError` if there are none."""
        kept = self.table.filter(_compute("equal", self.table[SPLIT], name))
        if not kept.num_rows:
            have = ", ".join(f"{split} ({count})" for split, count in self.counts.items()) or "no rows"
            raise ParameterError(f"the index has no rows in split {name!r}; it has {have}", path="split")
        return replace(self, table=kept)

    def assign(self, column: str = SPLIT, *, holdout: Holdout | None = None) -> Self:
        """Return the index whose ``split`` is the split column ``column``, with ``holdout``'s rows held out.

        Raises :class:`ParameterError` if the index has no column ``column`` or a holdout name is no plate and no
        well of the index.
        """
        if column not in self.splits:
            raise ParameterError(
                f"the index has no split column {column!r}; it has {', '.join(self.splits)}", path="split"
            )
        assigned = self.table[column]
        if holdout is not None and (holdout.plates or holdout.wells):
            plates = _plain(self.table["plate"])
            wells = _compute("binary_join_element_wise", plates, _plain(self.table["well"]), "/")
            for given, known, what in ((holdout.plates, plates, "plate"), (holdout.wells, wells, "plate/row/column")):
                unknown = sorted(set(given) - set(_compute("unique", known).to_pylist()))
                if unknown:
                    raise ParameterError(
                        f"the holdout names {', '.join(unknown)}, which is no {what} of the index", path="holdout"
                    )
            out = _compute(
                "or",
                _compute("is_in", plates, options=pc.SetLookupOptions(pa.array(holdout.plates, pa.string()))),
                _compute("is_in", wells, options=pc.SetLookupOptions(pa.array(holdout.wells, pa.string()))),
            )
            assigned = _compute("if_else", out, pa.scalar(HOLDOUT), _plain(assigned))
        table = self.table.set_column(self.table.schema.get_field_index(SPLIT), SPLIT, _dictionary(assigned))
        return replace(self, table=table)

    def __repr__(self) -> str:
        counts = ", ".join(f"{name}={count}" for name, count in self.counts.items())
        return f"FovIndex({len(self)} rows: {counts}; channels={list(self.channels)})"

    # --- files -------------------------------------------------------------------------------------------------------

    def save(self, path: str | Path, *, overwrite: bool = False, append: bool = False) -> Path:
        """Write the index as the directory ``path``: one parquet file per plate and the manifest; return ``path``.

        With ``append`` the plates are added to the index already at ``path``: only their files and the manifest are
        written. A plate is known by its input path; build the plates with ``index_from_plates(..., append_to=path)``,
        which names them apart from the index's plates and splits them next to its groups. Raises
        :class:`~waveorder.errors.Rejected` if an index (or a plate of it, with ``append``) is there already and
        ``overwrite`` is false, ``path`` is a file, or the index to append to has other channels, target channels,
        attributes, split columns or voxel size, another plate of the same name, or groups the splits of the plates to
        add ignore.
        """
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise Rejected.one("is a file; an index is a directory", location=str(path))
        held = read_manifest(path) if append and (path / MANIFEST).is_file() else None
        if held is None and (path / MANIFEST).is_file() and not overwrite:
            raise Rejected.one(
                "holds an index; replace it with overwrite=True (--overwrite in wo distill index), or add plates to "
                "it with append=True (--append)",
                location=str(path),
            )
        files = {} if held is None else {entry["plate"]: entry for entry in held["files"]}
        plates = sorted(set(self.column("plate")))
        replaced: dict[str, set[str]] = {}
        if held is not None:
            self._expect_appendable(held, path=path)
            recorded, broken = _held(path, held)
            replaced = self._replaced(recorded, broken, path=path)
            self._expect_assigned(recorded, path=path)
        taken = [plate for plate in plates if replaced.get(plate)]
        if taken and not overwrite:
            raise Rejected.one(
                f"already indexes {', '.join(taken)}; index them again with overwrite=True (--overwrite in wo "
                "distill index)",
                location=str(path),
            )
        path.mkdir(parents=True, exist_ok=True)
        for plate in plates:
            for old in sorted(replaced.get(plate, ())):
                entry = files.pop(old, None)
                if entry is not None and entry["name"] != _file_name(plate):
                    (path / entry["name"]).unlink(missing_ok=True)
            files[plate] = self._write_plate(path, plate)
        entries = [files[plate] for plate in sorted(files)]
        counts: Counter[str] = Counter()
        for entry in entries:
            counts.update(entry["counts"])
        manifest = {
            "schema": SCHEMA,
            "channels": list(self.channels),
            "target_channels": list(self.target_channels),
            "attributes": list(self.attributes),
            "splits": list(self.splits),
            "rows": sum(entry["rows"] for entry in entries),
            "counts": dict(sorted(counts.items())),
            "files": entries,
            **({} if held is None else {k: v for k, v in held.items() if k not in _DERIVED}),
            **self.meta,
        }
        work = path / f"{MANIFEST}.tmp"
        work.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        work.replace(path / MANIFEST)
        return path

    def _write_plate(self, path: Path, plate: str) -> dict[str, object]:
        """Write the rows of one plate as its parquet file; return its manifest entry."""
        table = self.table.filter(_compute("equal", self.table["plate"], plate))
        name = _file_name(plate)
        work = path / f"{name}.tmp"
        pq.write_table(table, work, row_group_size=max(1, table.num_rows))
        work.replace(path / name)
        return {
            "name": name,
            "plate": plate,
            "rows": table.num_rows,
            "counts": _counts(table, SPLIT),
            "sha256": _file_sha256(path / name),
        }

    def _expect_appendable(self, held: Mapping[str, object], *, path: Path) -> None:
        """Refuse to add this index's plates to the index described by the manifest ``held``."""
        for name, mine in (
            ("channels", list(self.channels)),
            ("target_channels", list(self.target_channels)),
            ("attributes", list(self.attributes)),
            ("splits", list(self.splits)),
        ):
            if held.get(name) != mine:
                raise Rejected.one(
                    f"indexes {name} {held.get(name)}, but the plates to add have {mine}; index them separately",
                    location=str(path),
                )
        theirs, size = held.get("pixel_size"), self.meta.get("pixel_size")
        if (
            isinstance(theirs, list)
            and isinstance(size, list | tuple)
            and any(abs(a - b) > PIXEL_SIZE_TOLERANCE * b for a, b in zip(size, theirs, strict=True))
        ):
            raise Rejected.one(
                f"has the voxel size {theirs} um, but the plates to add have {list(size)}", location=str(path)
            )

    def _replaced(self, recorded: pa.Table | None, broken: Collection[str], *, path: Path) -> dict[str, set[str]]:
        """Return the plates of the index at ``path`` each plate to add replaces; refuse a name another plate has.

        A plate replaces those of its input paths, and that of its name if that plate's file cannot be read.
        """
        held = _plates_by_input(recorded)
        replaced: dict[str, set[str]] = {plate: set() for plate in self.column("plate")}
        for location, plate in _plates_by_input(self.table).items():
            if location in held:
                replaced[plate].add(held[location])
        for plate, old in replaced.items():
            if plate in broken:
                old.add(plate)
            elif plate in held.values() and plate not in old:
                other = next(location for location, name in held.items() if name == plate)
                raise Rejected.one(
                    f"already indexes another plate named {plate} ({other}); build the plates to add with "
                    f"index_from_plates(..., append_to={str(path)!r}), which names them apart (wo distill index "
                    "--append does)",
                    location=str(path),
                )
        return replaced

    def _expect_assigned(self, recorded: pa.Table | None, *, path: Path) -> None:
        """Refuse plates whose split rules (``meta["split_rules"]``) give them other splits next to ``recorded``."""
        rules = self.meta.get("split_rules")
        if recorded is None or not isinstance(rules, Mapping):
            return
        fields = _fields(self.table)
        for name, arguments in rules.items():
            column = SPLIT if name == SPLIT else f"{SPLIT}_{name}"
            if column not in self or not isinstance(arguments, Mapping):
                continue
            if GroupSplit(**arguments).assign(fields, _recorded_fields(recorded, column)) != self.column(column):
                raise Rejected.one(
                    f"assigns groups to splits that the split column {column} of the plates to add ignores; build "
                    f"them with index_from_plates(..., append_to={str(path)!r}) (wo distill index --append does)",
                    location=str(path),
                )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read an index saved by :meth:`save`, after checking every file against the manifest.

        Raises :class:`~waveorder.errors.Rejected` if the manifest is missing or has another schema, or a file is
        missing, does not have its recorded SHA-256 or row count.
        """
        path = Path(path)
        manifest = read_manifest(path)
        tables = [_read_file(path, entry) for entry in manifest["files"]]
        if not tables:
            raise Rejected.one("lists no parquet file: the index is empty", location=str(path))
        return cls(
            pa.concat_tables(tables, promote_options="permissive"),
            channels=manifest["channels"],
            target_channels=manifest["target_channels"],
            attributes=manifest.get("attributes", ()),
            meta={name: value for name, value in manifest.items() if name not in _DERIVED},
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        channels: Sequence[str],
        target_channels: Sequence[str],
        attributes: Sequence[str] = (),
        meta: Mapping[str, object] | None = None,
        allow_pixel_size_mismatch: bool = False,
    ) -> Self:
        """Build an index from a table of your own, reading each position's shape and voxel size from its plate.

        Each row is a mapping (a ``csv.DictReader`` line, say) with the columns of :class:`FovRow` but the shape.
        ``input``, ``position`` and ``target`` are required; without ``time`` the row stands for every time point,
        without ``split`` it is ``train``, without ``plate`` the plate directory's name, without ``well`` the first
        two parts of ``position``; one numeric column per name in ``attributes``; columns ``split_<name>`` are named
        splits; other columns are ignored. The rows are sorted by plate, position (naturally, ``A/2`` before
        ``A/10``) and time. ``meta`` gets ``pixel_size``, ``dtype``, ``target_planes`` (the deepest target) and
        ``target_dtype`` (the widest) from the plates.
        ``allow_pixel_size_mismatch`` indexes plates whose voxel sizes differ by more than 1 %, recording that of
        the first row.

        Raises
        ------
        ParameterError
            If a row lacks ``input``, ``position`` or ``target``, or a column holds no number where one is due.
        Rejected
            If a position cannot be read, lacks a channel or the time point; if a target position does not exist or
            does not fit its input; or if the voxel sizes of two input plates differ by more than 1 % and
            ``allow_pixel_size_mismatch`` is false.
        """
        listed = list(rows)
        named = sorted({key for row in listed for key in row if str(key).startswith(f"{SPLIT}_")})
        records = [_record(row, tuple(attributes), named) for row in listed]
        measured, splits, facts = _measure(
            records, channels, target_channels, allow_pixel_size_mismatch=allow_pixel_size_mismatch
        )
        return cls.of(
            measured,
            channels=channels,
            target_channels=target_channels,
            attributes=attributes,
            meta={**(meta or {}), **facts},
            splits={name.removeprefix(f"{SPLIT}_"): values for name, values in splits.items()},
        )


def take_rows(table: pa.Table, positions: Iterable[int], attributes: Sequence[str] = ()) -> list[FovRow]:
    """Return the rows at ``positions`` (numbers of rows) of an index's table as :class:`FovRow` objects, in order."""
    taken = table.take(pa.array(list(positions), pa.int64()))
    names = [*_STRINGS, *_INTEGERS, SPLIT]
    columns = {name: taken[name].to_pylist() for name in [*names, *attributes]}
    return [
        FovRow(
            **{name: columns[name][number] for name in names},
            attributes={name: columns[name][number] for name in attributes},
        )
        for number in range(taken.num_rows)
    ]


def iter_rows(
    table: pa.Table, positions: Sequence[int], attributes: Sequence[str] = (), *, chunk: int = 1024
) -> Iterator[FovRow]:
    """Yield the rows at ``positions`` of an index's table as :class:`FovRow` objects, in order.

    They are made ``chunk`` at a time: a caller that keeps none holds at most ``chunk`` of them, whatever the count.
    """
    for start in range(0, len(positions), chunk):
        yield from take_rows(table, positions[start : start + chunk], attributes)


def index_from_plates(  # noqa: PLR0913 (what is read, how it is split, where it goes and one flag)
    inputs: str | Path | Iterable[str | Path],
    *,
    channels: Sequence[str],
    targets: Sequence[str | Path],
    target_channels: Sequence[str],
    split: GroupSplit | None = None,
    splits: Mapping[str, GroupSplit] | None = None,
    per_group: int | None = None,
    seed: int = 0,
    attributes: AttributeFunction | None = None,
    allow_pixel_size_mismatch: bool = False,
    append_to: str | Path | None = None,
) -> FovIndex:
    """Build the index of the positions of OME-Zarr HCS plates, split by group; independent of the order of inputs.

    Parameters
    ----------
    inputs : str, Path or Iterable[str or Path]
        Plates, positions or globs, as :func:`~waveorder.io.positions.expand_positions` takes.
    channels, target_channels : Sequence[str]
        As :class:`FovIndex`.
    targets : Sequence[str or Path]
        One plate per entry of ``inputs``, holding the target of every position that entry expands to under the same
        position name: the reconstructions made beforehand (``wo rec``).
    split : GroupSplit, optional
        The rule that assigns the ``split`` column; ``GroupSplit()`` by default. A plate's group name is its
        directory name when that is unique among the plates, else the part of its path that tells it from the others.
    splits : Mapping[str, GroupSplit], optional
        Named splits: each rule assigns the column ``split_<name>``, which a run may read instead of ``split``.
    per_group : int, optional
        Keep that many positions of each well of each plate, chosen by ``seed``; ``None`` keeps all.
    seed : int
        Seed of the ``per_group`` choice (the split has its own).
    attributes : callable, optional
        ``(plate_path, position_name) -> {name: value}``: numeric attributes of a position; every position must give
        the same names.
    allow_pixel_size_mismatch : bool
        As :meth:`FovIndex.from_rows`.
    append_to : str or Path, optional
        An index directory the plates are to be added to (``save(..., append=True)``). A plate it holds (same input
        path) keeps its name and recorded splits; a new plate gets a name none of its plates has, and a split by
        group assigns new groups next to the recorded ones (:class:`GroupSplit`).

    Raises
    ------
    ParameterError
        If ``targets`` has another length than ``inputs``, ``per_group`` is below 1, a split name is not a name, the
        attribute function gives different names for two positions, a holdout name of a split matches nothing, or
        ``append_to`` assigns a group to two splits.
    Rejected
        As :func:`~waveorder.io.positions.expand_positions` and :meth:`FovIndex.from_rows`; also for a position not
        inside an HCS plate, or an entry of ``inputs`` that spans several plates.
    """
    split = GroupSplit() if split is None else split
    rules = {SPLIT: split, **{f"{SPLIT}_{name}": rule for name, rule in (splits or {}).items()}}
    for name in splits or {}:
        if not name.isidentifier():
            raise ParameterError(f"index_from_plates split names are identifiers, got {name!r}", path="splits")
    if per_group is not None and per_group < 1:
        raise ParameterError(f"index_from_plates per_group must be a positive int or None, got {per_group!r}")
    found = _positions(inputs, targets)
    held = None
    if append_to is not None and (Path(append_to) / MANIFEST).is_file():
        held = _held(Path(append_to), read_manifest(append_to))[0]
    recorded = {Path(location): plate for location, plate in _plates_by_input(held).items()}
    names = _plate_names({plate for plate, _ in found}, recorded)
    kept = sorted(found) if per_group is None else _sample(found, names, per_group=per_group, seed=seed)
    wells = ["/".join(position.split("/")[:2]) for _, position in kept]
    groups = [(names[plate], well, position) for (plate, position), well in zip(kept, wells, strict=True)]
    assigned = {column: rule.assign(groups, _recorded_fields(held, column)) for column, rule in rules.items()}
    values = [dict(attributes(plate, position)) if attributes else {} for plate, position in kept]
    columns = tuple(values[0]) if values else ()
    for (plate, position), value in zip(kept, values, strict=True):
        if set(value) != set(columns):
            raise ParameterError(
                f"index_from_plates attributes gave the names {sorted(value)} for {plate / position} and "
                f"{sorted(columns)} for {kept[0][0] / kept[0][1]}; every position must give the same names",
                path="attributes",
            )
    rows = [
        {
            "input": str(plate),
            "position": position,
            "target": str(found[plate, position]),
            "plate": names[plate],
            "well": well,
            **{column: values[number] for column, values in assigned.items()},
            **value,
        }
        for number, ((plate, position), well, value) in enumerate(zip(kept, wells, values, strict=True))
    ]
    described = {name.removeprefix(SPLIT).lstrip("_") or SPLIT: rule.arguments() for name, rule in rules.items()}
    return FovIndex.from_rows(
        rows,
        channels=channels,
        target_channels=target_channels,
        attributes=columns,
        meta={"split_rules": described, "per_group": per_group, "seed": seed},
        allow_pixel_size_mismatch=allow_pixel_size_mismatch,
    )


def _positions(
    inputs: str | Path | Iterable[str | Path], targets: Sequence[str | Path]
) -> dict[tuple[Path, str], Path]:
    """Expand the inputs to ``{(plate, position name): target plate}``, each position once."""
    patterns = [inputs] if isinstance(inputs, str | Path) else list(inputs)
    if len(targets) != len(patterns):
        raise ParameterError(
            f"index_from_plates targets must list one plate per entry of inputs ({len(patterns)}), got {targets!r}",
            path="targets",
        )
    found: dict[tuple[Path, str], Path] = {}
    for pattern, target in zip(patterns, targets, strict=True):
        positions = expand_positions(pattern)
        plate_of_position: dict[Path, Path] = {}
        for position in positions:
            plate = plate_of(position)
            if plate is None:
                raise Rejected.one(
                    f"{position} is not inside an OME-Zarr HCS plate (row/column/fov below a plate); an index names "
                    "positions within plates",
                    location=str(pattern),
                )
            plate_of_position[position] = plate
        plates = set(plate_of_position.values())
        if len(plates) > 1:
            raise Rejected.one(
                f"{pattern} spans {len(plates)} plates, but targets lists one target plate per entry of inputs; "
                "list each input plate with its target plate",
                location=str(pattern),
            )
        for position, plate in plate_of_position.items():
            found.setdefault((plate, plate_name(position)), Path(target).resolve())
    return found


def _plate_names(plates: Collection[Path], recorded: Mapping[Path, str]) -> dict[Path, str]:
    """Name each plate: its ``recorded`` name, else as :func:`_distinct` names it among all plates, else its path.

    ``recorded`` maps the plates of the index the plates are added to onto their names; a new plate's name is none
    of theirs, and falls back to the plate's path if the parts that tell it apart are one of them.
    """
    distinct = _distinct({*plates, *recorded})
    taken = set(recorded.values())
    return {
        plate: recorded[plate] if plate in recorded else distinct[plate] if distinct[plate] not in taken else str(plate)
        for plate in plates
    }


def _distinct(plates: Collection[Path]) -> dict[Path, str]:
    """Name each plate: its directory name when unique, else the path parts that tell the plates apart."""
    if len({plate.name for plate in plates}) == len(plates):
        return {plate: plate.name for plate in plates}
    parts = [plate.parts for plate in plates]
    shortest = min(len(part) for part in parts)
    head = 0
    while head < shortest and len({part[head] for part in parts}) == 1:
        head += 1
    tail = 0
    while tail < shortest - head and len({part[-1 - tail] for part in parts}) == 1:
        tail += 1
    return {plate: "/".join(plate.parts[head : len(plate.parts) - tail]) for plate in plates}


def _sample(
    found: Mapping[tuple[Path, str], Path], names: Mapping[Path, str], *, per_group: int, seed: int
) -> list[tuple[Path, str]]:
    """Keep ``per_group`` positions of each well of each plate: those with the smallest seeded hashes."""
    groups: dict[tuple[Path, str], list[tuple[Path, str]]] = {}
    for plate, position in found:
        groups.setdefault((plate, "/".join(position.split("/")[:2])), []).append((plate, position))
    kept: list[tuple[Path, str]] = []
    for members in groups.values():
        ranked = sorted(
            members, key=lambda member: hashlib.sha256(f"{seed}:{names[member[0]]}/{member[1]}".encode()).digest()
        )
        kept.extend(ranked[:per_group])
    return sorted(kept)


type _Record = tuple[FovRow, int | None, dict[str, str]]
"""A row of a user's table without its shape, its time (``None``: every time point) and its named splits."""


def _record(row: Mapping[str, object], attributes: Sequence[str], named: Sequence[str]) -> _Record:
    """Read one row of a user's table."""
    for column in ("input", "position", "target"):
        if not row.get(column):
            raise ParameterError(
                f"FovIndex.from_rows: a row lacks the column {column!r} ({dict(row)!r}); every row names the plate "
                "(input), the position within it (position) and the plate of its recorded target (target)",
                path=f"rows.{column}",
            )
    position = str(row["position"]).strip("/")
    time = row.get("time")
    template = FovRow(
        input=str(row["input"]),
        position=position,
        target=str(row["target"]),
        split=str(row.get(SPLIT) or "train"),
        plate=str(row.get("plate") or Path(str(row["input"])).name),
        well=str(row.get("well") or "/".join(position.split("/")[:2])),
        planes=0,
        height=0,
        width=0,
        attributes={name: _number(row.get(name), column=name, position=position) for name in attributes},
    )
    times = None if time is None or time == "" else int(_number(time, column="time", position=position))
    return template, times, {name: str(row.get(name) or "") for name in named}


def _number(value: object, *, column: str, position: str) -> float:
    """Return a table value as a float, or raise naming the column and the position."""
    try:
        number = float(value)  # ty: ignore[invalid-argument-type] (anything float() refuses is reported below)
    except (TypeError, ValueError):
        number = math.nan
    if isinstance(value, bool) or not math.isfinite(number):
        raise ParameterError(
            f"FovIndex.from_rows: the column {column!r} of position {position} must hold a finite number, got "
            f"{value!r}",
            path=f"rows.{column}",
        )
    return number


def _measure(
    records: Sequence[_Record],
    channels: Sequence[str],
    target_channels: Sequence[str],
    *,
    allow_pixel_size_mismatch: bool,
) -> tuple[list[FovRow], dict[str, list[str]], dict[str, object]]:
    """Read every record's position; return the complete rows, sorted, their named splits and the ``meta`` facts."""
    from natsort import natsorted

    if not records:
        raise Rejected.one("the index has no rows: the table is empty", location="rows")
    records = natsorted(
        records, key=lambda record: (record[0].plate, record[0].input, record[0].position, record[1] or 0)
    )
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:  # the reads wait on the file system
        metas = list(pool.map(lambda record: read_position_meta(record[0].path), records))
    rows: list[FovRow] = []
    splits: dict[str, list[str]] = {name: [] for name in records[0][2]}
    for (template, time, named), meta in zip(records, metas, strict=True):
        channel_indices(meta, channels)
        if time is not None and not 0 <= time < meta.times:
            raise Rejected.one(
                f"{template.path} has {meta.times} time points, so it has no time {time}",
                location=str(template.path),
            )
        planes, height, width = meta.zyx
        times = range(meta.times) if time is None else [time]
        rows.extend(replace(template, time=each, planes=planes, height=height, width=width) for each in times)
        for name, value in named.items():
            splits[name].extend([value] * len(times))
    facts: dict[str, object] = {"pixel_size": list(metas[0].pixel_size), "dtype": metas[0].dtype}
    facts["target_planes"], facts["target_dtype"] = _expect_targets(records, metas, target_channels)
    _expect_one_pixel_size(metas, allow=allow_pixel_size_mismatch)
    return rows, splits, facts


def _expect_targets(
    records: Sequence[_Record], metas: Sequence[PositionMeta], target_channels: Sequence[str]
) -> tuple[int, str]:
    """Refuse target positions that do not fit their inputs; return the deepest target's planes and widest dtype.

    A target must exist, hold the target channels, have its input's Y and X and every indexed time point.
    """
    missing = [template for template, _, _ in records if not template.target_path.is_dir()]
    if missing:
        first = missing[0].target_path
        raise Rejected.one(
            f"{len(missing)} of {len(records)} positions have no target: {first} does not exist. Leave them out of "
            "the inputs (list positions or globs instead of the plate), or reconstruct them first (wo rec)",
            location=str(first),
        )
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        targets = list(pool.map(lambda record: read_position_meta(record[0].target_path), records))
    for (template, time, _), meta, target in zip(records, metas, targets, strict=True):
        channel_indices(target, target_channels)
        needed = meta.times if time is None else time + 1
        if target.zyx[1:] != meta.zyx[1:] or target.times < needed:
            raise Rejected.one(
                f"holds (T, Z, Y, X) = {(target.times, *target.zyx)}, but its input {template.path} holds "
                f"{(meta.times, *meta.zyx)}: a target has its input's Y and X and every indexed time point. "
                "Reconstruct it again, or leave the position out of the inputs",
                location=str(target.path),
            )
    widest = max((target.dtype for target in targets), key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
    return max(target.zyx[0] for target in targets), widest


def _expect_one_pixel_size(metas: Sequence[PositionMeta], *, allow: bool) -> None:
    """Refuse positions whose voxel size differs from the first one's by more than the tolerance."""
    if allow:
        return
    reference = metas[0]
    for meta in metas:
        if any(
            abs(size - expected) > PIXEL_SIZE_TOLERANCE * expected
            for size, expected in zip(meta.pixel_size, reference.pixel_size, strict=True)
        ):
            raise Rejected.one(
                f"{meta.path} has the voxel size {meta.pixel_size} um (z, y, x) but {reference.path} has "
                f"{reference.pixel_size}: a student is trained at one sampling. Index them separately, or pass "
                "allow_pixel_size_mismatch=True to index them together",
                location=str(meta.path),
            )


# --- columns and files ------------------------------------------------------------------------------------------------


def _strings(values: list[str]) -> pa.Array:
    return pa.array(values, pa.string()).dictionary_encode()


def _plain(column: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """Return a string column without its dictionary encoding."""
    return column.cast(pa.string()) if pa.types.is_dictionary(column.type) else column


def _dictionary(column: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    return _plain(column).dictionary_encode()


def _compute(
    function: str, *args: object, options: pc.FunctionOptions | None = None
) -> pa.ChunkedArray | pa.Array | pa.Scalar:
    """Return the pyarrow compute function ``function`` (``"equal"``, ``"unique"``, ...) of ``args``.

    ``pyarrow.compute`` makes its module functions (``pc.equal``, ...) when it is imported, so a type checker sees
    none of them; ``call_function`` runs the same kernels by name. Python values among ``args`` become scalars.
    """
    return pc.call_function(function, list(args), options)


def _counts(table: pa.Table, column: str) -> dict[str, int]:
    counted = Counter(table[column].to_pylist())
    return {name: counted[name] for name in sorted(counted)}


def _digest(table: pa.Table) -> str:
    """Return the SHA-256 of a table's values: string columns without their dictionaries, one chunk per column."""
    plain = pa.table({name: _plain(table[name]) for name in table.column_names}).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, plain.schema) as writer:
        writer.write_table(plain)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _file_name(plate: str) -> str:
    """Return the parquet file name of a plate: its name with ``/`` written ``__``."""
    return plate.replace("/", "__") + ".parquet"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_file(path: Path, entry: Mapping[str, Any], columns: Sequence[str] | None = None) -> pa.Table:
    """Read the parquet file of the manifest entry ``entry`` of the index ``path``, after checking it against it."""
    file = path / entry["name"]
    if not file.is_file():
        raise Rejected.one(f"lacks {entry['name']}, which its manifest lists", location=str(path))
    if _file_sha256(file) != entry["sha256"]:
        raise Rejected.one(
            f"{entry['name']} does not match the SHA-256 its manifest records: the file was changed after it was "
            "written. Index the plate again (wo distill index --append --overwrite)",
            location=str(path),
        )
    table = pq.read_table(file, columns=columns)
    if table.num_rows != entry["rows"]:
        raise Rejected.one(f"{entry['name']} has {table.num_rows} rows, not {entry['rows']}", location=str(path))
    return table


def _held(path: Path, manifest: Mapping[str, Any]) -> tuple[pa.Table | None, list[str]]:
    """Return the plate, well, position, input and split columns of an index's files, and its unreadable plates.

    The table is ``None`` if no file can be read; a plate whose file is missing or changed is replaced by name.
    """
    tables: list[pa.Table] = []
    broken: list[str] = []
    for entry in manifest["files"]:
        try:
            tables.append(_read_file(path, entry, ["plate", "well", "position", "input", *manifest["splits"]]))
        except Rejected:
            broken.append(entry["plate"])
    return (pa.concat_tables(tables, promote_options="permissive") if tables else None), broken


def _plates_by_input(table: pa.Table | None) -> dict[str, str]:
    """Return the plate name of each input plate of an index's table, by absolute path: what identifies a plate."""
    if table is None:
        return {}
    pairs = pa.table({"input": _plain(table["input"]), "plate": _plain(table["plate"])})
    pairs = pairs.group_by(["input", "plate"]).aggregate([])
    return {
        os.path.abspath(location): plate
        for location, plate in zip(pairs["input"].to_pylist(), pairs["plate"].to_pylist(), strict=True)
    }


def _fields(table: pa.Table) -> list[tuple[str, str, str]]:
    """Return the ``(plate, well, position)`` of each row of an index's table."""
    return list(zip(*(table[name].to_pylist() for name in ("plate", "well", "position")), strict=True))


def _recorded_fields(table: pa.Table | None, column: str) -> list[tuple[str, str, str, str]]:
    """Return the ``(plate, well, position, split)`` of each row, the split from ``column``; none without it."""
    if table is None or column not in table.column_names:
        return []
    return list(zip(*(table[name].to_pylist() for name in ("plate", "well", "position", column)), strict=True))


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Return the manifest of the index directory ``path``, after checking its schema."""
    path = Path(path)
    file = path / MANIFEST
    if not file.is_file():
        raise Rejected.one(
            f"has no {MANIFEST}; an index is a directory of parquet files with a manifest (wo distill index)",
            location=str(path),
        )
    manifest = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise Rejected.one(
            f"{MANIFEST} is not the manifest of an index of schema {SCHEMA}; index the plates again with this "
            "waveorder (wo distill index)",
            location=str(path),
        )
    return manifest
