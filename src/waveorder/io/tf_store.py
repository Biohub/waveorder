"""The ``tf.zarr`` transfer-function store: parameters, their provenance and the transfer functions they give.

Layout ``waveorder.tf/3``, a zarr v3 group read and written with zarr-python; its metadata is plain JSON::

    <name>.zarr/
      attrs: format, parameter_names (dotted, column order), tile_grid, provenance
      parameters             (T, P) float64: every free quantity of the model, one row per tile
      parameters_covariance  (T, P, P) float64: each tile's Laplace covariance of the calibrated system quantities;
                             zero outside them and where there is none (a sharpness score), NaN where the
                             calibration found none (a quantity on a bound, a Hessian not positive definite)
      transfer_function/<Z>x<Y>x<X>/transfer   optional: the transfer on that grid, reused by later runs, with its
                             metadata as the group's attributes (a thin system's plane transfers are for wo view;
                             reconstructions recompute them)
      score_trace            optional (N,) float64: the objective at each gradient step of the calibration
      score_grid             optional (N, G + 1) float64: each grid point's G values, then its objective

Stores from #592 and wo-4.0 are read too. Earlier 4.x stores (format ``waveorder.tf/2``) and 3.x
``compute-tf`` stores (no parameters) are refused. :meth:`TfStore.save` writes ``<name>.tmp-<id>`` and renames it
onto ``<name>``, so a killed process leaves a ``.tmp-`` or ``.old-`` directory.
"""

import hashlib
import json
import re
import shutil
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import torch
import zarr
from torch import Tensor
from zarr.core.common import JSON

from waveorder.errors import CalibrationWarning, Rejected, named
from waveorder.nn import Free, ParameterPrior

FORMAT = "waveorder.tf/3"
"""The ``format`` attribute of the stores this module writes."""

_MALFORMED = (ValueError, TypeError, KeyError, OSError, RuntimeError)
"""What reading a damaged store raises."""

_V2_MARK = "__tensordict__"
"""The root attribute of the earlier 4.x ``waveorder.tf/2`` stores, which this version refuses."""

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
        The ``(C, F, Z, Y, X // 2 + 1)`` half spectrum, or for a thin (2D) system its ``(Z, F, 1, Y, X // 2 + 1)``
        plane transfers (``attrs["layout"]`` says which); ``None`` when listed by :meth:`TfStore.load`
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
    parameters_covariance : Tensor, optional
        ``(T, P, P)`` float64 on the CPU: each tile's covariance of the parameters, zero where unknown, NaN where the
        calibration found none. ``None`` writes zeros; a loaded store always has it.
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
    origin : {"waveorder.tf/3", "#592"}
        The layout it was read from (``#592`` covers the stores of #592 and wo-4.0).
    """

    parameter_names: tuple[str, ...]
    parameters: Tensor
    parameters_covariance: Tensor | None = None
    tile_grid: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    transfers: tuple[TransferArray, ...] = ()
    scores: dict[str, Tensor] = field(default_factory=dict)
    path: Path | None = None
    origin: Literal["waveorder.tf/3", "#592"] = FORMAT

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

        The store's values become the prior centres of the quantities it names. As in wo-4.0, where the store's
        :attr:`parameters_covariance` over those quantities is usable, its inverse becomes their Gaussian prior, a
        full precision matrix that replaces their tolerances; the quantities outside it keep each spec's tolerance
        (flat without one), and those the store does not name keep their own prior. Rows and columns that are zero
        on every tile (quantities with no covariance, such as the noise level or the weight) are left out of the
        block, and an all-zero block leaves the tolerances in force. One row is shared by every tile; ``T`` rows
        give one prior per tile, their centres and precision both.

        Parameters
        ----------
        specs : Mapping[str, Free]
            The model's free quantities by dotted name (:func:`waveorder.nn.free_specs`).

        Returns
        -------
        ParameterPrior

        Raises
        ------
        Rejected
            If the block on some tile is singular, so it cannot serve as a Gaussian prior.

        Warns
        -----
        CalibrationWarning
            If the block holds NaN (the calibration that made it found no covariance, e.g. on a bound): the
            tolerances stay in force for the quantities it covers.
        """
        names = [name for name in specs if name in self.parameter_names]
        columns = [self.parameter_names.index(name) for name in names]
        centers = {name: self.parameters[:, column] for name, column in zip(names, columns, strict=True)}
        covariance = self.parameters_covariance
        if covariance is None or not columns:
            return ParameterPrior(dict(specs), centers)
        block = covariance[:, columns][:, :, columns]
        known = ((block != 0).any(dim=2) | (block != 0).any(dim=1)).any(dim=0)
        kept = [name for name, flag in zip(names, known.tolist(), strict=True) if flag]
        if not kept:
            return ParameterPrior(dict(specs), centers)
        block = block[:, known][:, :, known]
        if block.isnan().any():
            shown = ", ".join(kept)
            warnings.warn(
                f"the {named('prior')} store's parameters_covariance over {shown} holds NaN (the calibration that made "
                "it found no covariance there); their prior takes the model's tolerances instead",
                CalibrationWarning,
                stacklevel=2,
            )
            return ParameterPrior(dict(specs), centers)
        precision, info = torch.linalg.inv_ex(block)
        if (info != 0).any():
            message = (
                f"the {named('prior')} parameters_covariance block over {len(kept)} covered parameters is singular "
                "and cannot serve as a Gaussian prior; re-estimate it, or drop the singular parameters from the store"
            )
            raise Rejected.one(message) if self.path is None else _rejected(self.path, message)
        return ParameterPrior(dict(specs), centers, precision, tuple(kept))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read and validate a ``tf.zarr``.

        Validation: names are distinct, ``parameters`` is ``(T, P)`` and finite, ``parameters_covariance`` is
        ``(T, P, P)`` with no infinity, and ``tile_grid`` lists one centre per tile if any. Whether the names cover a
        model's free quantities is the caller's check.

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
            an earlier 4.x ``waveorder.tf/2`` store (recompute it with this version), not a transfer-function store,
            another format, malformed, or fails validation.
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

    ``waveorder.tf`` (a ``format`` attribute, or an earlier ``waveorder.tf/2`` store), ``#592``
    (``parameters`` and ``parameter_names``, no format), ``3.x`` (a ``settings`` attribute or 3.x transfer arrays).
    """
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        group = zarr.open_group(str(path), mode="r")
        return _layout(group, dict(group.attrs))
    except (ValueError, OSError, KeyError, TypeError):
        return None


def _layout(group: zarr.Group, attrs: Mapping[str, object]) -> Layout | None:
    """Return the layout of an open zarr group from its attributes and members."""
    if "format" in attrs or _V2_MARK in attrs:
        return "waveorder.tf"
    if "parameters" in group and "parameter_names" in attrs:
        return "#592"
    if isinstance(attrs.get("settings"), dict) or any(name in group for name in _ARRAYS_3X):
        return "3.x"
    return None


def _reading[T](path: Path, read: Callable[[Path], T]) -> T:
    """Return ``read(path)``; a damaged store, on which zarr fails, is refused at ``path``."""
    try:
        return read(path)
    except Rejected:
        raise
    except _MALFORMED as error:
        raise _rejected(path, f"is a malformed transfer-function store ({type(error).__name__}: {error})") from None


def _open(path: Path) -> tuple[zarr.Group, dict[str, object]]:
    """Return the zarr group at ``path`` and its attributes; refuse a missing path or one that is not a group."""
    if not path.exists():
        raise _rejected(path, "does not exist")
    try:
        group = zarr.open_group(str(path), mode="r")
        return group, dict(group.attrs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise _rejected(path, f"is not a zarr group ({error})") from None


def _read[S: TfStore](kind: type[S], path: Path) -> S:
    """Read and validate the store at ``path`` as a ``kind``, by its layout."""
    group, attrs = _open(path)
    layout = _layout(group, attrs)
    if layout == "waveorder.tf":
        store = _read_v3(kind, group, attrs, path=path)
    elif layout == "#592":
        store = _read_592(kind, group, attrs, path=path)
    elif layout == "3.x":
        raise _rejected(path, "was made by waveorder 3.x, which stored no parameters; recompute it with wo compute-tf")
    else:
        raise _rejected(path, "is not a transfer-function store (no format attribute and no parameters)")
    validate(store)
    return store


def _grid_name(key: str, path: Path) -> tuple[int, int, int]:
    if not _SHAPE_NAME.fullmatch(key):
        raise _rejected(path, f"transfer_function/{key} is not named <Z>x<Y>x<X>")
    z, y, x = (int(n) for n in key.split("x"))
    return z, y, x


def _grids(group: zarr.Group, path: Path) -> dict[str, zarr.Group]:
    """Return the ``transfer_function/<Z>x<Y>x<X>`` groups of a :data:`FORMAT` store by name."""
    if "transfer_function" not in group:
        return {}
    grids = group["transfer_function"]
    if not isinstance(grids, zarr.Group):
        raise _rejected(path, "transfer_function must be a group of <Z>x<Y>x<X> groups")
    found = {}
    for key, node in sorted(grids.members()):
        _grid_name(key, path)
        if not isinstance(node, zarr.Group):
            raise _rejected(path, f"transfer_function/{key} must be a group")
        found[key] = node
    return found


def _read_v3[S: TfStore](kind: type[S], group: zarr.Group, attrs: Mapping[str, object], *, path: Path) -> S:
    """Read a :data:`FORMAT` store, its transfers listed without their arrays."""
    if attrs.get("format") == "waveorder.tf/2" or _V2_MARK in attrs:
        raise _rejected(
            path,
            "was written through tensordict by an earlier waveorder 4.x (format waveorder.tf/2), which this version "
            "does not read; recompute it with this version (wo calibrate-tf, or a model's to_tf_store and save)",
        )
    if (tag := attrs.get("format")) != FORMAT:
        raise _rejected(path, f"has format {tag!r}; this waveorder reads {FORMAT}, so a newer waveorder wrote it")
    return kind(
        parameter_names=tuple(_names(attrs, path)),
        parameters=_array(group, "parameters", path),
        parameters_covariance=_array(group, "parameters_covariance", path),
        tile_grid=_mapping(attrs, "tile_grid", path),
        provenance=_mapping(attrs, "provenance", path),
        transfers=tuple(
            TransferArray(shape=_grid_name(key, path), attrs=dict(node.attrs))
            for key, node in _grids(group, path).items()
        ),
        scores={name: _array(group, name, path) for name in SCORES if name in group},
        path=path,
    )


def read_transfer(path: str | Path, shape: Sequence[int]) -> TransferArray | None:
    """Read the transfer function a ``tf.zarr`` holds on ``shape``; ``None`` if absent or not a :data:`FORMAT` store.

    Raises
    ------
    Rejected
        As :meth:`TfStore.load`.
    """
    key = "x".join(map(str, shape))

    def read(at: Path) -> TransferArray | None:
        group, attrs = _open(at)
        grid = _grids(group, at).get(key) if attrs.get("format") == FORMAT else None
        if grid is None or "transfer" not in grid:
            return None
        transfer = grid["transfer"]
        if not isinstance(transfer, zarr.Array):
            raise _rejected(at, f"transfer_function/{key}/transfer must be an array")
        values = torch.as_tensor(transfer[...])
        return TransferArray(shape=_grid_name(key, at), attrs=dict(grid.attrs), transfer=values)

    return _reading(Path(path), read)


def _array(group: zarr.Group, name: str, path: Path) -> Tensor:
    """Return the real array ``name`` of ``group`` as float64."""
    if name not in group:
        raise _rejected(path, f"has no {name} array")
    node = group[name]
    if not isinstance(node, zarr.Array) or node.dtype.kind not in "fiu":
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


def _read_592[S: TfStore](kind: type[S], group: zarr.Group, attrs: Mapping[str, object], *, path: Path) -> S:
    """Read a #592 or wo-4.0 store: names mapped, the covariance (zeros if absent), its transfer array listed."""
    try:
        names = tuple(dotted_592_name(name) for name in _names(attrs, path))
    except Rejected as error:
        raise _rejected(path, "; ".join(problem.message for problem in error.problems)) from None
    parameters = _array(group, "parameters", path)
    covariance = None
    if "parameters_covariance" in group:
        covariance = _array(group, "parameters_covariance", path)
    elif parameters.dim() == 2:
        covariance = torch.zeros(*parameters.shape, parameters.shape[1], dtype=torch.float64)
    grid = _mapping(attrs, "tile_grid", path)
    grid = {"size_yx": None, "overlap_yx": [0, 0], "centers_yx": None, "shape_zyx": None, "blend": None} | grid
    transfers = []
    if isinstance(node := group.get("transfer_function"), zarr.Array):
        z, y, x = (int(n) for n in node.shape[-3:])
        transfers.append(TransferArray(shape=(z, y, x), attrs={"layout": "#592 transfer_function array"}))
    return kind(
        parameter_names=names,
        parameters=parameters,
        parameters_covariance=covariance,
        tile_grid=grid | {"units": "um"},
        provenance=_mapping(attrs, "provenance", path),
        transfers=tuple(transfers),
        scores={name: _array(group, name, path) for name in SCORES if name in group},
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
    covariance = store.parameters_covariance
    if covariance is not None:
        if covariance.shape != (tiles, size, size):
            shown = list(covariance.shape)
            raise _rejected(path, f"parameters_covariance must be [tiles, {size}, {size}], got {shown}")
        if torch.isinf(covariance).any():
            raise _rejected(path, "parameters_covariance must hold no infinity (NaN marks an unknown covariance)")
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


def _attributes(values: Mapping[str, object]) -> dict[str, JSON]:
    """Return ``values`` as plain JSON: NaN and infinities as strings (``"NaN"``), other objects by ``str``."""
    return json.loads(json.dumps(values, default=str), parse_constant=str)


def _put(group: zarr.Group, name: str, values: Tensor) -> None:
    group.create_array(name, data=values.detach().cpu().numpy())


def _write(work: Path, store: TfStore) -> None:
    """Write ``store`` as the new zarr v3 group ``work``: tensors as arrays, the rest as JSON attributes."""
    metadata = {
        "format": FORMAT,
        "parameter_names": list(store.parameter_names),
        "tile_grid": store.tile_grid,
        "provenance": store.provenance,
    }
    group = zarr.open_group(str(work), mode="w-", zarr_format=3, attributes=_attributes(metadata))
    size = len(store.parameter_names)
    covariance = store.parameters_covariance
    if covariance is None:
        covariance = torch.zeros(store.tiles, size, size, dtype=torch.float64)
    _put(group, "parameters", store.parameters.to(torch.float64))
    _put(group, "parameters_covariance", covariance.to(torch.float64))
    for name, values in store.scores.items():
        _put(group, name, values.to(torch.float64))
    for item in store.transfers:
        if item.transfer is not None:
            key = "x".join(map(str, item.shape))
            grid = group.create_group(f"transfer_function/{key}", attributes=_attributes(item.attrs))
            _put(grid, "transfer", item.transfer)
