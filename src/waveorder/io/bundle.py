"""A student bundle on disk, written by :func:`save_bundle` and read by :func:`load_bundle`.

A bundle is a directory::

    bundle.json     schema, each part as {"class_path": ..., "init_args": {...}}, contract, model, record, sha256s
    weights.pt      {"student", "input_normalizer", "target_normalizer"}: state dicts
    test.pt         {"data", "condition", "expected", "tolerance"}: one input and the output it must give
    model.onnx      optional, written by an export and recorded with :func:`record_file`

Parts are imported by class path, so a bundle holds no code; a file whose sha256 differs from the record is refused.
Schemas ``waveorder.student/1`` and ``/2`` are read too. The solver spec ``{"distilled": {"path": ...}}`` loads one.
"""

import dataclasses
import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from waveorder import registry
from waveorder.errors import BundleError, ParameterError, Rejected, ShapeError, named
from waveorder.nn.networks import Bundle, Condition, Contract, Example, Normalizer, Precision, RestorationNetwork
from waveorder.nn.networks.stems import _DefocusKernel
from waveorder.nn.solvers import Distilled

SCHEMA = "waveorder.student/3"
"""The schema a bundle is written with; :func:`load_bundle` also reads :data:`_NAMED` and :data:`_FIRST`."""

_NAMED = "waveorder.student/2"
"""The schema of bundles that store each part by its registered name, ``{name: arguments}``."""

_FIRST = "waveorder.student/1"
"""The schema of bundles that store each part as ``{"name": ..., "arguments": {...}}``."""

_DOCUMENT, _WEIGHTS, _TEST = "bundle.json", "weights.pt", "test.pt"
_REQUIRED = (_WEIGHTS, _TEST)


def save_bundle(bundle: Bundle, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write ``bundle`` as a directory at ``path``, creating missing parents.

    The directory is written beside ``path`` and renamed onto it when complete. Weights are saved on the CPU. A test
    with ``expected=None`` gets the output the parts compute for its data on the CPU.

    Parameters
    ----------
    bundle : Bundle
        The student, its normalizers, conditions, contract and test; ``model`` and ``record`` must be JSON-able.
    path : str or Path
        Directory to write.
    overwrite : bool
        Replace a bundle already at ``path``; the old one is restored if the replacement fails.

    Returns
    -------
    Path
        ``path``.

    Raises
    ------
    Rejected
        If ``path`` exists and ``overwrite`` is false, or exists and is not a bundle; nothing is written or deleted.
    ParameterError
        If a part's arguments cannot be written, ``model`` or ``record`` is not JSON-able, or the bundle has no test.
    ShapeError
        If the test's expected output does not have the shape the parts return for its data.
    """
    path = Path(path)
    _expect_writable(path, overwrite=overwrite)
    document: dict[str, object] = {
        "schema": SCHEMA,
        "student": registry.class_spec(bundle.student),
        "contract": bundle.contract.as_dict(),
        "conditions": [registry.class_spec(condition) for condition in bundle.conditions],
        "input_normalizer": registry.class_spec(bundle.input_normalizer),
        "target_normalizer": registry.class_spec(bundle.target_normalizer),
        "model": dict(bundle.model),
        "record": dict(bundle.record),
    }
    try:
        json.dumps(document)
    except (TypeError, ValueError) as error:
        raise ParameterError(f"save_bundle: model and record must hold JSON-able values: {error}") from error
    modules = {"student": bundle.student, "input_normalizer": bundle.input_normalizer}
    modules["target_normalizer"] = bundle.target_normalizer
    weights = {
        name: {key: _own(value) for key, value in module.state_dict().items()} for name, module in modules.items()
    }
    recorded = _recorded_test(bundle)

    path.parent.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    work = path.with_name(f"{path.name}.tmp-{tag}")
    work.mkdir()
    try:
        torch.save(weights, work / _WEIGHTS)
        torch.save(recorded, work / _TEST)
        document["sha256"] = {name: _sha256(work / name) for name in _REQUIRED}
        (work / _DOCUMENT).write_text(json.dumps(document, indent=2) + "\n")
        _expect_writable(path, overwrite=overwrite)
        _publish(work, path, aside=path.with_name(f"{path.name}.old-{tag}"))
    finally:
        shutil.rmtree(work, ignore_errors=True)  # gone once renamed
    return path


def _publish(work: Path, path: Path, *, aside: Path) -> None:
    """Rename ``work`` to ``path``, moving an existing ``path`` to ``aside`` and restoring it if the rename fails."""
    if not path.exists():
        work.rename(path)
        return
    path.rename(aside)
    try:
        work.rename(path)
    except BaseException:
        aside.rename(path)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _expect_writable(path: Path, *, overwrite: bool) -> None:
    """Refuse a ``path`` a bundle cannot be written at: taken without ``overwrite``, or not a bundle to replace."""
    if not path.exists():
        return
    if not overwrite:
        raise Rejected.one(f"exists; pass {named('overwrite')}=True to replace it", location=str(path))
    if not (path.is_dir() and ((path / _DOCUMENT).is_file() or not any(path.iterdir()))):
        raise Rejected.one(
            "exists and is not a student bundle; nothing is deleted: choose another path", location=str(path)
        )


def _own(tensor: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """Return a contiguous CPU copy that owns its storage (``torch.save`` of a view saves the whole base)."""
    return tensor.detach().to(device="cpu", dtype=dtype or tensor.dtype).clone(memory_format=torch.contiguous_format)


def _recorded_test(bundle: Bundle) -> dict[str, object]:
    """Return the test as ``test.pt`` holds it, with the expected output computed when the example has none."""
    test = bundle.test
    if test is None:
        raise ParameterError("save_bundle: the bundle has no test; give it Bundle(test=Example(data=...))", path="test")
    data = _own(test.data, dtype=torch.float32)
    condition = {name: _own(value, dtype=torch.float32) for name, value in test.condition.items()}
    computed = dataclasses.replace(bundle, test=Example(data=data, condition=condition)).test_output("cpu")
    expected = _own(computed if test.expected is None else test.expected, dtype=torch.float32)
    if expected.shape != computed.shape:
        raise ShapeError(
            f"save_bundle: the test's expected output has shape {tuple(expected.shape)}, but the student returns "
            f"{tuple(computed.shape)} for its data"
        )
    return {"data": data, "condition": condition, "expected": expected, "tolerance": float(test.tolerance)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _document(path: Path) -> tuple[dict[str, Any], str]:
    """Return the parsed ``bundle.json`` of the bundle at ``path`` and its sha256, after checking its schema."""
    file = path / _DOCUMENT
    if not file.is_file():
        raise BundleError(f"{path} is not a student bundle: it has no {_DOCUMENT}")
    raw = file.read_bytes()
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise BundleError(f"{file} is not valid JSON: {error}") from error
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema not in (SCHEMA, _NAMED, _FIRST):
        raise BundleError(
            f"{file} has schema {schema!r}, but this waveorder reads {SCHEMA!r}, {_NAMED!r} and {_FIRST!r}: load it "
            "with the waveorder version that wrote it, or write the bundle again with this one"
        )
    return document, hashlib.sha256(raw).hexdigest()


def _spec(value: object) -> object:
    """Return a part of a :data:`_FIRST` document, ``{"name": ..., "arguments": {...}}`` at any depth, as a spec."""
    if isinstance(value, dict):
        if value.keys() == {"name", "arguments"}:
            return {value["name"]: _spec(value["arguments"] or {})}
        return {key: _spec(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_spec(item) for item in value]
    return value


def _verify(path: Path, document: Mapping[str, object]) -> None:
    """Raise unless every required file is present and every recorded file that is present has its recorded hash."""
    recorded = document.get("sha256")
    if not isinstance(recorded, dict) or not all(name in recorded for name in _REQUIRED):
        raise BundleError(f"{path / _DOCUMENT} does not record the sha256 of {' and '.join(_REQUIRED)}")
    for name, expected in recorded.items():
        file = path / name
        if not file.is_file():
            if name in _REQUIRED:
                raise BundleError(f"{path} is incomplete: {name} is missing")
            continue
        actual = _sha256(file)
        if actual != expected:
            raise BundleError(
                f"{file} was altered after the bundle was written: its sha256 is {actual}, the bundle records "
                f"{expected}"
            )


def load_bundle(path: str | Path) -> Bundle:
    """Read the bundle directory at ``path``: rebuild its parts, load the weights, verify the hashes.

    Every part is returned in evaluation mode on the CPU, in float32.

    Parameters
    ----------
    path : str or Path
        Bundle directory.

    Returns
    -------
    Bundle

    Raises
    ------
    BundleError
        If ``path`` holds no ``bundle.json``, the schema is unknown, a file is missing or does not have its recorded
        hash, or the weights do not fit the student the bundle describes.
    ParameterError
        If the bundle names a class that cannot be imported (or, in an earlier schema, a name that is not registered),
        or a part that is not of its kind.
    """
    path = Path(path)
    document, digest = _document(path)
    _verify(path, document)
    if document["schema"] == _FIRST:
        for key in ("student", "conditions", "input_normalizer", "target_normalizer"):
            if key in document:
                document[key] = _spec(document[key])
    named = document["schema"] != SCHEMA
    try:
        student = _part(document["student"], RestorationNetwork, kind="network", named=named)
        contract = Contract.from_dict(document["contract"])
        listed = tuple(_part(each, Condition, kind="condition", named=named) for each in document["conditions"])
        input_normalizer = _part(document["input_normalizer"], Normalizer, kind="normalizer", named=named)
        target_normalizer = _part(document["target_normalizer"], Normalizer, kind="normalizer", named=named)
        model, record = dict(document["model"]), dict(document["record"])
    except (KeyError, TypeError) as error:
        raise BundleError(f"{path / _DOCUMENT} is not a complete bundle description: {error!r}") from error
    weights = torch.load(path / _WEIGHTS, map_location="cpu", weights_only=True)
    _expect_this_defocus(student, weights.get("student", {}), path=path)
    modules = {"student": student, "input_normalizer": input_normalizer, "target_normalizer": target_normalizer}
    for name, module in modules.items():
        try:
            module.load_state_dict(weights[name])
        except (KeyError, RuntimeError) as error:
            raise BundleError(
                f"{path / _WEIGHTS} does not fit the {name} that {_DOCUMENT} describes: {error}"
            ) from error
        module.eval()
    saved = torch.load(path / _TEST, map_location="cpu", weights_only=True)
    test = Example(
        data=saved["data"], condition=saved["condition"], expected=saved["expected"], tolerance=saved["tolerance"]
    )
    return Bundle(
        student=student,
        contract=contract,
        conditions=listed,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        model=model,
        record=record,
        test=test,
        path=path,
        sha256=digest,
    )


def _part[T](spec: object, expected: type[T], *, kind: str, named: bool) -> T:
    """Build one part of a bundle: by its registered name (``named``, the earlier schemas) or its class path."""
    part: object = registry.resolve(spec, kind, expected) if named else registry.build(spec)
    if not isinstance(part, expected):
        raise ParameterError(f"the bundle's {kind} {type(part).__name__} is not a {expected.__name__}", path=kind)
    return part


def _expect_this_defocus(student: RestorationNetwork, weights: Mapping[str, Tensor], *, path: Path) -> None:
    """Refuse a ``DefocusKernel`` stem trained on the old defocus input (one number per plane, um from stack centre)."""
    for name, module in student.named_modules():
        if not isinstance(module, _DefocusKernel):
            continue
        key = f"{name}.kernel.0.weight"
        saved = weights.get(key)
        expected = module.kernel[0].in_features
        if saved is not None and saved.ndim == 2 and saved.shape[1] != expected:
            raise BundleError(
                f"{path} holds a DefocusKernel stem whose kernel network takes {saved.shape[1]} input(s) per plane: "
                "it was trained with the earlier defocus (um from the stack centre). This waveorder's DefocusKernel "
                f"reads the defocus from the focus plane in axial optical units, {expected} inputs per plane, so "
                "these weights cannot be used; train the student again, or load the bundle with the waveorder "
                "version that wrote it"
            )


def record_file(path: str | Path, name: str) -> str:
    """Record the sha256 of the file ``name`` in the bundle at ``path``, and return it.

    For a file added after the bundle was written (``model.onnx`` from an export): :func:`load_bundle` then refuses
    the bundle if the file changes. The bundle's own ``sha256`` changes with it.

    Parameters
    ----------
    path : str or Path
        Bundle directory.
    name : str
        File name inside the bundle.

    Returns
    -------
    str
        The file's sha256 hex digest.

    Raises
    ------
    BundleError
        If ``path`` is not a bundle, or holds no file ``name``.
    ParameterError
        If ``name`` is ``bundle.json`` or not a plain file name.
    """
    path = Path(path)
    if name == _DOCUMENT or Path(name).name != name:
        raise ParameterError(f"record_file name must be the name of a file in the bundle, got {name!r}", path="name")
    document, _ = _document(path)
    file = path / name
    if not file.is_file():
        raise BundleError(f"{path} holds no file {name} to record")
    digest = _sha256(file)
    recorded = document.setdefault("sha256", {})
    if not isinstance(recorded, dict):
        raise BundleError(f"{path / _DOCUMENT} does not record file hashes as a mapping")
    recorded[name] = digest
    work = path / f"{_DOCUMENT}.tmp-{uuid.uuid4().hex[:12]}"
    work.write_text(json.dumps(document, indent=2) + "\n")
    work.replace(path / _DOCUMENT)
    return digest


@registry.build_with(Distilled)
def _load_distilled(path: str | Path, precision: Precision = "float32") -> Distilled:
    """Build the :class:`~waveorder.nn.Distilled` solver of the bundle at ``path``: spec ``{"distilled": {...}}``."""
    return Distilled(load_bundle(path), precision=precision)
