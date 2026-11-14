"""The ``tf.zarr`` transfer-function store: parameters, their provenance and the transfer functions they give.

Layout ``waveorder.tf/2`` (zarr v3 through ``tensordict.PersistentTensorDict``; a read never unpickles)::

    <name>.zarr/
      format, parameter_names (dotted, column order), tile_grid, provenance
      parameters             (T, P) float64: every free quantity of the model, one row per tile
      transfer_function/<Z>x<Y>x<X>/{transfer, metadata}   optional: the transfer on that grid, reused by later runs
      score_trace            optional (N,) float64: the objective at each gradient step of the calibration
      score_grid             optional (N, G + 1) float64: each grid point's G values, then its objective

Stores from #592 are read too; 3.x ``compute-tf`` stores (no parameters) are refused. :meth:`TfStore.save` writes
``<name>.tmp-<id>`` and renames it onto ``<name>``, so a killed process leaves a ``.tmp-`` or ``.old-`` directory.
"""

import contextlib
import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import torch
from torch import Tensor

from waveorder.errors import Rejected, named
from waveorder.nn import Free, ParameterPrior

FORMAT = "waveorder.tf/2"
"""The ``format`` entry of the stores this module writes."""

_MALFORMED = (ValueError, TypeError, KeyError, OSError, RuntimeError)
"""What reading a damaged store raises."""

_SHAPE_NAME = re.compile(r"[0-9]+x[0-9]+x[0-9]+")
"""Name of a ``transfer_function/<Z>x<Y>x<X>`` group."""

SCORES = ("score_trace", "score_grid")
"""The score arrays a store may hold (:attr:`TfStore.scores`), by wo-4.0's names."""

_ARRAYS_3X = frozenset(
    {
        "real_potential_transfer_function",
        "imaginary_potential_transfer_function",
        "optical_transfer_function",
        "intensity_to_stokes_matrix",
        "singular_system_U",
        "vector_singular_system_U",
    }
)
"""Arrays 3.x ``compute-tf`` wrote (``origin/main:waveorder/cli/compute_transfer_function.py``)."""

type Layout = Literal["waveorder.tf", "#592", "3.x"]

_PREFIX_592 = "system.detection.pupil_aberrations."

NAMES_592: dict[str, str] = {
    name: f"{_PREFIX_592}{name}"
    for name in (
        "defocus",
        "astigmatism_45",
        "astigmatism_0",
        "coma_y",
        "coma_x",
        "trefoil_y",
        "trefoil_x",
        "spherical",
    )
}
"""#592's named Zernike terms (``waveorder/calibrate/zernike_names.py`` there, Noll 4 to 11) and their 4.x names."""


def canonical_json(value: object) -> str:
    """Return ``value`` as canonical JSON: sorted keys, no spaces, non-ASCII kept; other scalars by ``str``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def parameters_sha256(names: Sequence[str], parameters: Tensor) -> str:
    """Return the hex sha256 of parameter names and ``(T, P)`` values (exact float64 bits)."""
    content = {"names": list(names), "parameters": [[float.hex(v) for v in row] for row in parameters.tolist()]}
    return hashlib.sha256(canonical_json(content).encode()).hexdigest()


def dotted_592_name(name: str) -> str:
    """Return the 4.x dotted parameter name of a #592 parameter name.

    A short name (:data:`NAMES_592`) or ``z<n>`` (Noll index) maps under ``system.detection.pupil_aberrations``:
    ``z4`` is ``system.detection.pupil_aberrations.defocus``, ``z12`` is ``system.detection.pupil_aberrations.z12``.

    Raises
    ------
    Rejected
        If ``name`` is neither.
    """
    from waveorder.zernike_names import name_to_noll, noll_to_name

    if name in NAMES_592:
        return NAMES_592[name]
    if name.startswith("z") and name[1:].isdigit() and int(name[1:]) >= 1:
        return f"{_PREFIX_592}{noll_to_name(name_to_noll(name))}"
    raise Rejected.one(f"#592 store parameter {name!r} is not a Zernike term name this version can map")


@dataclass(frozen=True, kw_only=True)
class TransferArray:
    """The model's transfer function on one grid, which a later run on that grid may reuse.

    Attributes
    ----------
    shape : tuple[int, int, int]
        The ``(Z, Y, X)`` grid it was computed on, after z padding.
    attrs : dict
        Layout, FFT convention and what it was computed for (``waveorder.models._stores.transfer_record``).
    transfer : Tensor, optional
        The ``(C, F, Z, Y, X // 2 + 1)`` half spectrum; ``None`` when listed by :meth:`TfStore.load`
        (:func:`read_transfer` reads it).
    """

    shape: tuple[int, int, int]
    attrs: dict[str, object]
    transfer: Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class TfStore:
    """A ``tf.zarr`` in memory.

    Attributes
    ----------
    parameter_names : tuple[str, ...]
        ``P`` dotted names of the model's free quantities, in column order.
    parameters : Tensor
        ``(T, P)`` float64 on the CPU, one row per tile (one row for a shift-invariant store).
    tile_grid : dict
        ``size_yx``, ``overlap_yx``, ``centers_yx`` (one ``[y, x]`` per tile, um), ``shape_zyx``, ``blend``, ``units``.
    provenance : dict
        How the store was made: ``kind``, ``sources``, ``position``, ``time_index``, ...
    transfers : tuple[TransferArray, ...]
        Transfer functions per grid.
    scores : dict[str, Tensor]
        Optional float64 ``score_trace`` ``(N,)`` and ``score_grid`` ``(N, G + 1)``, whose columns
        ``provenance["score_grid_columns"]`` names.
    path : Path, optional
        Where the store was read from.
    origin : {"waveorder.tf/2", "#592"}
        The layout it was read from.
    """

    parameter_names: tuple[str, ...]
    parameters: Tensor
    tile_grid: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    transfers: tuple[TransferArray, ...] = ()
    scores: dict[str, Tensor] = field(default_factory=dict)
    path: Path | None = None
    origin: Literal["waveorder.tf/2", "#592"] = FORMAT

    @property
    def tiles(self) -> int:
        """Rows of :attr:`parameters`."""
        return int(self.parameters.shape[0])

    @property
    def parameters_sha256(self) -> str:
        """Hex sha256 of the names and the parameters (exact float bits): the store's identity."""
        return parameters_sha256(self.parameter_names, self.parameters)

    def __repr__(self) -> str:
        """Return the store's kind, size, names and (for one row) values; never the tensors."""
        shown: object = list(self.parameter_names)
        if self.tiles == 1:
            shown = {name: float(f"{value:.6g}") for name, value in self.row().items()}
        parts = [
            f"kind={self.provenance.get('kind')!r}",
            f"tiles={self.tiles}",
            f"parameters={shown}",
            f"transfers={len(self.transfers)}",
        ]
        if self.path is not None:
            parts.append(f"path={str(self.path)!r}")
        return f"TfStore({', '.join(parts)})"

    def row(self, tile: int = 0) -> dict[str, float]:
        """Return the parameters of tile ``tile`` by name."""
        return dict(zip(self.parameter_names, self.parameters[tile].tolist(), strict=True))

    def to_prior(self, specs: Mapping[str, Free]) -> ParameterPrior:
        """Return the prior this store sets over the free quantities ``specs``, for a calibration that starts from it.

        The store's values become the prior centres of the quantities it names, with each spec's tolerance (flat
        without one); other quantities keep their own prior. One row is shared by every tile; ``T`` rows give one
        prior per tile.

        Parameters
        ----------
        specs : Mapping[str, Free]
            The model's free quantities by dotted name (:func:`waveorder.nn.free_specs`).

        Returns
        -------
        ParameterPrior
        """
        columns = {name: self.parameter_names.index(name) for name in specs if name in self.parameter_names}
        return ParameterPrior(dict(specs), {name: self.parameters[:, column] for name, column in columns.items()})

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read and validate a ``tf.zarr``.

        Validation: names are distinct, ``parameters`` is ``(T, P)`` and finite, and ``tile_grid`` lists one centre
        per tile if any. Whether the names cover a model's free quantities is the caller's check.

        Parameters
        ----------
        path : str or Path
            The store.

        Returns
        -------
        TfStore
            With :attr:`path` set and its transfers listed without arrays (:func:`read_transfer` reads one).

        Raises
        ------
        Rejected
            If ``path`` is missing, not a zarr group, a 3.x ``compute-tf`` store (recompute it with ``wo compute-tf``),
            not a transfer-function store, another format, malformed, or fails validation.
        """
        return _reading(Path(path), lambda at: _read(cls, at))

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Validate the store and publish it at ``path`` atomically.

        Parameters
        ----------
        path : str or Path
            Where to publish; missing parent directories are created.
        overwrite : bool
            Replace a transfer-function store at ``path``; anything else there is never deleted.

        Returns
        -------
        Path
            ``path``.

        Raises
        ------
        Rejected
            If ``path`` exists and ``overwrite`` is false, is a file, or is not a transfer-function store (nothing is
            written), or if the store fails validation.
        """
        validate(self)
        path = Path(path)
        _expect_publishable(path, overwrite=overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex[:12]
        work = path.with_name(f"{path.name}.tmp-{suffix}")
        old = path.with_name(f"{path.name}.old-{suffix}")
        try:
            _write(work, self)
            if overwrite and path.exists():
                path.rename(old)
            work.rename(path)
        finally:
            shutil.rmtree(work, ignore_errors=True)  # gone once renamed
        shutil.rmtree(old, ignore_errors=True)
        return path


def _rejected(path: Path, message: str) -> Rejected:
    return Rejected.one(message, location=str(path))


def store_layout(path: str | Path) -> Layout | None:
    """Return which transfer-function store ``path`` holds (what ``overwrite`` may replace), or ``None``.

    ``waveorder.tf`` (a ``format`` entry), ``#592`` (``parameters`` and ``parameter_names``, no format), ``3.x``
    (a ``settings`` attribute or 3.x transfer arrays).
    """
    import zarr

    path = Path(path)
    if not path.is_dir():
        return None
    try:
        group = zarr.open_group(str(path), mode="r")
        return _layout(group.__contains__, dict(group.attrs))
    except (ValueError, OSError, KeyError, TypeError):
        return None


def _layout(has: Callable[[str], bool], attrs: Mapping[str, object]) -> Layout | None:
    """Return the layout of an open zarr group from its attributes and ``has(name)``, whether it holds a member."""
    if "__tensordict__" in attrs and has("format"):  # a format entry tensordict wrote
        return "waveorder.tf"
    if "format" not in attrs and has("parameters") and "parameter_names" in attrs:
        return "#592"
    if isinstance(attrs.get("settings"), dict) or any(has(name) for name in _ARRAYS_3X):
        return "3.x"
    return None


def _reading[T](path: Path, read: Callable[[Path], T]) -> T:
    """Return ``read(path)``; a damaged store, on which zarr or tensordict fail, is refused at ``path``."""
    try:
        return read(path)
    except Rejected:
        raise
    except _MALFORMED as error:
        raise _rejected(path, f"is a malformed transfer-function store ({type(error).__name__}: {error})") from None


def _read[S: TfStore](kind: type[S], path: Path) -> S:
    """Read and validate the store at ``path`` as a ``kind``, by its layout."""
    import zarr

    if not path.exists():
        raise _rejected(path, "does not exist")
    try:
        group = zarr.open_group(str(path), mode="r")
        attrs = dict(group.attrs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise _rejected(path, f"is not a zarr group ({error})") from None
    layout = _layout(group.__contains__, attrs)
    if layout == "waveorder.tf":
        store = _read_v2(kind, path)
    elif layout == "#592":
        store = _read_592(kind, group, attrs, path=path)
    elif layout == "3.x":
        raise _rejected(path, "was made by waveorder 3.x, which stored no parameters; recompute it with wo compute-tf")
    else:
        raise _rejected(path, "is not a transfer-function store (no format entry and no parameters)")
    validate(store)
    return store


def _numbers(values: object, name: str, path: Path) -> Tensor:
    if values is None:
        raise _rejected(path, f"has no {name} array")
    if not isinstance(values, Tensor) or values.is_complex() or values.dtype == torch.bool:
        raise _rejected(path, f"{name} must be a numeric array")
    return values.to(torch.float64)


def _grid_name(key: str, path: Path) -> tuple[int, int, int]:
    if not _SHAPE_NAME.fullmatch(key):
        raise _rejected(path, f"transfer_function/{key} is not named <Z>x<Y>x<X>")
    z, y, x = (int(n) for n in key.split("x"))
    return z, y, x


def _metadata(value: object, key: str, path: Path) -> dict[str, object]:
    """Return ``value``, the ``transfer_function/<key>/metadata`` entry, as a mapping."""
    name = f"transfer_function/{key}/metadata"
    return _mapping({name: value}, name, path)


def _read_v2[S: TfStore](kind: type[S], path: Path) -> S:
    """Read a :data:`FORMAT` store, its transfers listed without their arrays."""
    from tensordict import PersistentTensorDict, TensorDictBase

    with contextlib.closing(PersistentTensorDict.from_zarr(path, mode="r", allow_pickle=False)) as data:
        if (tag := data.get_non_tensor("format", None)) != FORMAT:
            raise _rejected(path, f"has format {tag!r}; this waveorder reads {FORMAT}, so a newer waveorder wrote it")
        keys = set(data.keys())
        meta = {key: data.get_non_tensor(key) for key in ("parameter_names", "tile_grid", "provenance") if key in keys}
        grids = data.get("transfer_function", None)
        if grids is not None and not isinstance(grids, TensorDictBase):
            raise _rejected(path, "transfer_function must be a group of <Z>x<Y>x<X> groups")
        listed = sorted(() if grids is None else grids.keys())
        metadata = {key: data.get_non_tensor(("transfer_function", key, "metadata"), None) for key in listed}
        return kind(
            parameter_names=tuple(_names(meta, path)),
            parameters=_numbers(data.get("parameters", None), "parameters", path),
            tile_grid=_mapping(meta, "tile_grid", path),
            provenance=_mapping(meta, "provenance", path),
            transfers=tuple(
                TransferArray(shape=_grid_name(key, path), attrs=_metadata(value, key, path))
                for key, value in metadata.items()
            ),
            scores={name: _numbers(data.get(name), name, path) for name in SCORES if name in keys},
            path=path,
        )


def read_transfer(path: str | Path, shape: Sequence[int]) -> TransferArray | None:
    """Read the transfer function a ``tf.zarr`` holds on ``shape``; ``None`` if absent or a #592 store.

    Raises
    ------
    Rejected
        As :meth:`TfStore.load`.
    """
    key = "x".join(map(str, shape))

    def read(at: Path) -> TransferArray | None:
        from tensordict import PersistentTensorDict

        if not at.exists():
            raise _rejected(at, "does not exist")
        with contextlib.closing(PersistentTensorDict.from_zarr(at, mode="r", allow_pickle=False)) as data:
            v2 = data.get_non_tensor("format", None) == FORMAT
            transfer = data.get(("transfer_function", key, "transfer"), None) if v2 else None
            if transfer is None:
                return None
            if not isinstance(transfer, Tensor):
                raise _rejected(at, f"transfer_function/{key}/transfer must be an array")
            metadata = data.get_non_tensor(("transfer_function", key, "metadata"), None)
            return TransferArray(shape=_grid_name(key, at), attrs=_metadata(metadata, key, at), transfer=transfer)

    return _reading(Path(path), read)


def _array(group: object, name: str, path: Path) -> Tensor:
    if name not in group:  # ty: ignore[unsupported-operator]
        raise _rejected(path, f"has no {name} array")
    node = group[name]  # ty: ignore[not-subscriptable]
    if not hasattr(node, "dtype") or node.dtype.kind not in "fiu":
        raise _rejected(path, f"{name} must be a numeric array")
    return torch.as_tensor(node[...], dtype=torch.float64)


def _mapping(attrs: Mapping[str, object], name: str, path: Path) -> dict[str, object]:
    value = attrs.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _rejected(path, f"attribute {name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _names(attrs: Mapping[str, object], path: Path) -> list[str]:
    names = attrs.get("parameter_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise _rejected(path, "parameter_names must be a list of names")
    return names


def _read_592[S: TfStore](kind: type[S], group: object, attrs: Mapping[str, object], *, path: Path) -> S:
    """Read #592's layout: short names mapped, its ``transfer_function`` array listed, its score arrays."""
    try:
        names = tuple(dotted_592_name(name) for name in _names(attrs, path))
    except Rejected as error:
        raise _rejected(path, "; ".join(problem.message for problem in error.problems)) from None
    parameters = _array(group, "parameters", path)
    grid = _mapping(attrs, "tile_grid", path)
    grid = {"size_yx": None, "overlap_yx": [0, 0], "centers_yx": None, "shape_zyx": None, "blend": None} | grid
    transfers = []
    if "transfer_function" in group:  # ty: ignore[unsupported-operator]
        z, y, x = (int(n) for n in group["transfer_function"].shape[-3:])  # ty: ignore[not-subscriptable]
        transfers.append(TransferArray(shape=(z, y, x), attrs={"layout": "#592 transfer_function array"}))
    return kind(
        parameter_names=names,
        parameters=parameters,
        tile_grid=grid | {"units": "um"},
        provenance=_mapping(attrs, "provenance", path),
        transfers=tuple(transfers),
        scores={name: _array(group, name, path) for name in SCORES if name in group},  # ty: ignore[unsupported-operator]
        path=path,
        origin="#592",
    )


def validate(store: TfStore) -> None:
    """Check a store's consistency, as :meth:`TfStore.load` describes.

    Raises
    ------
    Rejected
        Naming the store and what is wrong.
    """
    path = store.path or Path("tf.zarr")
    names = store.parameter_names
    if len(set(names)) != len(names):
        raise _rejected(path, f"parameter_names repeats a name: {list(names)}")
    size = len(names)
    if store.parameters.dim() != 2 or store.parameters.shape[1] != size or store.parameters.shape[0] < 1:
        raise _rejected(path, f"parameters must be [tiles, {size}], got {list(store.parameters.shape)}")
    tiles = store.tiles
    if not torch.isfinite(store.parameters).all():
        raise _rejected(path, "parameters must be finite")
    _validate_grid(store.tile_grid, tiles=tiles, path=path)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_grid(grid: Mapping[str, object], *, tiles: int, path: Path) -> None:
    """Refuse a ``tile_grid`` with wrong entry types, or another number of centres than tiles."""
    centers = grid.get("centers_yx")
    if centers is not None:
        if not isinstance(centers, list) or not all(
            isinstance(center, list) and len(center) == 2 and all(map(_is_number, center)) for center in centers
        ):
            raise _rejected(path, "tile_grid.centers_yx must be a list of [y, x] numbers")
        if len(centers) not in (0, tiles):
            raise _rejected(path, f"tile_grid lists {len(centers)} centres for {tiles} tiles")
    for name, length in (("shape_zyx", 3), ("size_yx", 2), ("overlap_yx", 2)):
        value = grid.get(name)
        if value is not None and not (isinstance(value, list) and len(value) == length and all(map(_is_number, value))):
            raise _rejected(path, f"tile_grid.{name} must be a list of {length} numbers, got {value!r}")


def _expect_publishable(path: Path, *, overwrite: bool) -> None:
    """Refuse a ``path`` a store cannot be published at: taken without ``overwrite``, or not a store to replace."""
    if path.is_file():
        raise _rejected(path, f"is a file, not a transfer-function store; use another {named('output')}")
    if not path.is_dir():
        return
    if not overwrite:
        raise _rejected(path, f"exists; pass {named('overwrite')} to replace it")
    if store_layout(path) is None:
        raise _rejected(path, f"is not a transfer-function store; use another {named('output')}")


def _write(work: Path, store: TfStore) -> None:
    """Write ``store`` as the new store ``work`` through tensordict: tensors as arrays, the rest as JSON entries."""
    from tensordict import NonTensorData, PersistentTensorDict, TensorDict

    def plain(value: object) -> NonTensorData:
        # JSON values only (a NaN as the string "NaN"): tensordict pickles an entry JSON does not give back equal
        return NonTensorData(json.loads(json.dumps(value, default=str), parse_constant=str))

    entries: dict[str, object] = {
        "format": plain(FORMAT),
        "parameter_names": plain(list(store.parameter_names)),
        "tile_grid": plain(store.tile_grid),
        "provenance": plain(store.provenance),
        "parameters": store.parameters.to(torch.float64),
        **{name: values.to(torch.float64) for name, values in store.scores.items()},
    }
    grids = {
        "x".join(map(str, item.shape)): {"transfer": item.transfer.detach().cpu(), "metadata": plain(item.attrs)}
        for item in store.transfers
        if item.transfer is not None
    }
    if grids:
        entries["transfer_function"] = grids
    PersistentTensorDict.from_dict(TensorDict(entries, batch_size=[]), work, backend="zarr").close()
