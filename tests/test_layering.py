"""Each layer imports only the layers below it, at module level or inside a function; ``waveorder.nn`` touches no
file."""

import ast
from pathlib import Path

import pytest

import waveorder

ROOT = Path(waveorder.__file__).parent

UTILITIES = frozenset({"errors", "observe", "zernike_names", "registry"})
"""Top-level modules any layer may import; they import no layer themselves."""

LAYERS: dict[str, frozenset[str]] = {
    "nn": frozenset(),
    "io": frozenset({"nn"}),
    "data": frozenset({"io", "nn"}),
    "tiling": frozenset({"io", "nn"}),
    "train": frozenset({"data", "nn"}),
    "config": frozenset({"io", "nn"}),
    "models": frozenset({"config", "train", "data", "tiling", "io", "nn"}),
    "cli": frozenset({"config", "models", "train", "data", "tiling", "io", "nn", "integrations"}),
    "integrations": frozenset(),
    "testing": frozenset({"nn"}),
    "calibrate": frozenset({"models", "io", "nn"}),
    "distill": frozenset({"models", "train", "data", "nn"}),
    "simulate": frozenset({"models", "nn"}),
    "focus": frozenset({"nn"}),
}
"""What each package or front-door module may import from ``waveorder``, besides itself and :data:`UTILITIES`."""

FILE_MODULES = frozenset({"zarr", "iohub", "h5py", "tifffile", "shutil", "tempfile", "glob"})
"""Modules ``waveorder.nn`` never imports: what they are for is files."""

FILE_FUNCTIONS = frozenset(
    {"open", "io.open", "torch.save", "torch.load", "json.load", "json.dump", "pickle.load", "pickle.dump"}
    | {
        f"os.{name}"
        for name in (
            "open",
            "remove",
            "unlink",
            "rename",
            "replace",
            "mkdir",
            "makedirs",
            "rmdir",
            "removedirs",
            "listdir",
            "scandir",
            "walk",
            "stat",
            "chmod",
            "symlink",
            "link",
            "truncate",
            "path.exists",
            "path.isfile",
            "path.isdir",
            "path.getsize",
            "path.getmtime",
        )
    }
)
"""Functions, by qualified name, that read or write files."""

PATH_METHODS = frozenset(
    {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "mkdir",
        "rmdir",
        "unlink",
        "rename",
        "touch",
        "exists",
        "is_file",
        "is_dir",
        "iterdir",
        "glob",
        "rglob",
        "stat",
        "chmod",
        "symlink_to",
        "hardlink_to",
        "samefile",
    }
)
"""Methods of :class:`pathlib.Path` that read or write the filesystem; ``replace(target)`` is checked apart, as
``str.replace`` and ``dataclasses.replace`` share its name."""


def _files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _owner(path: Path) -> str:
    relative = path.relative_to(ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def _imported(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
            found += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return found


def _package(name: str) -> str | None:
    parts = name.split(".")
    return parts[1] if parts[0] == "waveorder" and len(parts) > 1 else None


def _bound(tree: ast.AST) -> tuple[dict[str, str], set[str]]:
    """Return the qualified name of every name an import binds (``{"Path": "pathlib.Path", "np": "numpy"}``), and
    the names bound to modules by ``import``."""
    names, modules = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                names[local] = alias.name if alias.asname else local
                modules.add(local)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names |= {alias.asname or alias.name: f"{node.module}.{alias.name}" for alias in node.names}
    return names, modules


def _dotted(node: ast.expr) -> tuple[str, list[str]] | None:
    """Return the head name and the attributes of ``a.b.c``, or ``None`` when it does not start with a name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return (node.id, parts[::-1]) if isinstance(node, ast.Name) else None


def file_io(source: str) -> list[tuple[int, str]]:
    """Return ``(line, what)`` for every import of a file library and every call that reads or writes a file.

    A call is one of :data:`FILE_FUNCTIONS` or a function of a :data:`FILE_MODULES` module by its qualified name,
    or else a method of :data:`PATH_METHODS` (or ``replace(target)``) called on anything but an imported module.
    """
    tree = ast.parse(source)
    names, modules = _bound(tree)
    found = [(line, name) for line, name in _imported(tree) if name.split(".")[0] in FILE_MODULES]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        name = None if dotted is None else ".".join([names.get(dotted[0], dotted[0]), *dotted[1]])
        if name is not None and (name in FILE_FUNCTIONS or name.split(".")[0] in FILE_MODULES):
            found.append((node.lineno, name))
        elif isinstance(node.func, ast.Attribute) and (dotted is None or dotted[0] not in modules):
            method = node.func.attr
            if method in PATH_METHODS or (method == "replace" and len(node.args) == 1 and not node.keywords):
                found.append((node.lineno, f".{method}()"))
    return sorted(found)


def test_every_package_has_a_layer():
    owners = {_owner(path) for path in _files()} - {"__init__", "__main__"}
    assert owners <= set(LAYERS) | UTILITIES, f"no layer rule for {sorted(owners - set(LAYERS) - UTILITIES)}"


def _reached(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(line, name)`` for every import, and every ``waveorder.<x>`` reached by attribute through a module."""
    names, modules = _bound(tree)
    found = _imported(tree)
    for node in ast.walk(tree):
        dotted = _dotted(node) if isinstance(node, ast.Attribute) else None
        if dotted is not None and dotted[0] in modules and names.get(dotted[0], dotted[0]).startswith("waveorder"):
            found.append((getattr(node, "lineno", 0), ".".join([names.get(dotted[0], dotted[0]), *dotted[1]])))
    return found


def layer_violations(owner: str, source: str) -> list[tuple[int, str]]:
    """Return ``(line, name)`` for every use of a ``waveorder`` layer that ``owner`` may not import."""
    allowed = (frozenset() if owner in UTILITIES else LAYERS[owner]) | UTILITIES | {owner}
    known = set(LAYERS) | UTILITIES
    return sorted(
        {
            (line, name)
            for line, name in _reached(ast.parse(source))
            if _package(name) not in allowed | {None} and _package(name) in known
        }
    )


@pytest.mark.parametrize("path", _files(), ids=lambda path: str(path.relative_to(ROOT)))
def test_imports_go_down_only(path):
    owner = _owner(path)
    if owner in ("__init__", "__main__"):
        return  # the package root re-exports the user API lazily
    source = path.read_text()
    above = [f"{path.relative_to(ROOT.parent)}:{line}: {name}" for line, name in layer_violations(owner, source)]
    assert not above, f"waveorder.{owner} may import only {sorted(LAYERS.get(owner, ()))}:\n" + "\n".join(above)
    if owner == "nn":
        files = [f"{line}: {name}" for line, name in file_io(source)]
        assert not files, "waveorder.nn reads and writes no files:\n" + "\n".join(files)


ATTRIBUTE_FORM = """
import waveorder
import waveorder.nn as wnn


def build():
    return waveorder.models.Phase3D(wnn.Detection())


def read():
    return waveorder.io.read_plate
"""
"""A lower layer reaching ``waveorder.models`` by attribute, without importing it by name."""


def test_the_layer_rule_catches_a_layer_reached_by_attribute():
    assert {name for _, name in layer_violations("io", ATTRIBUTE_FORM)} == {
        "waveorder.models",
        "waveorder.models.Phase3D",
    }
    assert layer_violations("models", ATTRIBUTE_FORM) == []  # models may use nn and io
    assert {name for _, name in layer_violations("nn", ATTRIBUTE_FORM)} >= {"waveorder.io", "waveorder.models"}


OLD_BUNDLE = """
import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch import load


def save_bundle(path, weights, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    work = path.with_name(f"{path.name}.tmp")
    torch.save(weights, work / "weights.pt")
    (work / "bundle.json").write_text(json.dumps(document, indent=2) + "\\n")
    if path.exists():
        shutil.rmtree(path)
    work.rename(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        digest.update(file.read())
    return digest.hexdigest()


def load_bundle(path):
    document = json.loads((path / "bundle.json").read_bytes())
    (path / "bundle.json.tmp").replace(path / "bundle.json")
    return document, load(path / "test.pt"), torch.load(path / "weights.pt", weights_only=True)
"""
"""The file operations of ``nn/networks/bundle.py`` before they moved to :mod:`waveorder.io.bundle`."""

PURE = """
import dataclasses
import json
from pathlib import Path

import torch


def describe(spec, name: str, path: Path | None) -> str:
    spec = dataclasses.replace(spec, nominal=1.0)
    text = json.dumps({"name": name.replace("_", " "), "path": None if path is None else str(path / "bundle")})
    module = torch.nn.Linear(2, 2)
    module.load_state_dict(module.state_dict())
    return json.loads(text)["name"] + Path(name).name
"""
"""Code that names paths and parses JSON, and touches no file."""


def test_the_file_io_rule_catches_the_old_bundle_code_and_passes_pure_code():
    assert [what for _, what in file_io(OLD_BUNDLE)] == [
        "shutil",
        ".mkdir()",
        "torch.save",
        ".write_text()",
        ".exists()",
        "shutil.rmtree",
        ".rename()",
        ".open()",
        ".read_bytes()",
        ".replace()",
        "torch.load",
        "torch.load",
    ]
    assert file_io(PURE) == []
