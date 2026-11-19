"""No type hint in the package is written as a string: every annotation names an imported or defined type."""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "waveorder"


def _strings(node: ast.expr) -> list[ast.Constant]:
    """Return the string constants of an annotation that stand for types: not ``Literal`` values or metadata."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node]
    if isinstance(node, ast.Subscript):
        name = node.value.attr if isinstance(node.value, ast.Attribute) else getattr(node.value, "id", "")
        if name == "Literal":
            return []
        if name == "Annotated":
            first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
            return _strings(first)
    return [found for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr) for found in _strings(child)]


def _annotations(tree: ast.AST) -> list[ast.expr]:
    """Return every argument, return and variable annotation of ``tree``, each ``type X = ...`` value, and the type of
    each ``cast``."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = node.args
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
            found += [argument.annotation for argument in every if argument is not None and argument.annotation]
            if node.returns is not None:
                found.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            found.append(node.annotation)
        elif isinstance(node, ast.TypeAlias):
            found.append(node.value)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if function == "cast":
                found.append(node.args[0])
    return found


def _files() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


@pytest.mark.parametrize("path", _files(), ids=lambda path: str(path.relative_to(SOURCE)))
def test_no_annotation_is_a_string(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    written = [f"line {node.lineno}: {node.value!r}" for node in _annotations(tree) for node in _strings(node)]
    assert written == []


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("def f(x: 'Model') -> 'Model': ...", 2),
        ("x: list['Model'] = []", 1),
        ("y = cast('Tensor', z)", 1),
        ("def f(x: Literal['auto', 'off']) -> None: ...", 0),
        ("def f(x: Annotated[int, 'metadata']) -> None: ...", 0),
        ("def f(x: Annotated['Model', 'metadata']) -> None: ...", 1),
        ("type Model = 'Thing'", 1),
        ("type Pair = tuple['Model', int] | None", 1),
        ("type Mode = Literal['auto', 'off']", 0),
        ("type Weight = Annotated[float, 'metadata']", 0),
    ],
)
def test_the_walk_finds_string_annotations_and_only_those(source, count):
    tree = ast.parse(source)
    assert sum(len(_strings(node)) for node in _annotations(tree)) == count
