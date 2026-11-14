"""Every command's help reads as running text at 80 columns: no line breaks where the docstring was wrapped."""

import itertools
import re

import pytest
import typer.main
from typer._click import Command
from typer.core import TyperGroup
from typer.testing import CliRunner

from waveorder.cli import app
from waveorder.cli.render import _unwrapped

WIDTH = 80
runner = CliRunner()


def _paths(command: Command, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every shown command and group below ``command`` whose help typer prints, by its words after ``wo``.

    ``wo distill fit``, ``validate`` and ``test`` hand ``--help`` to Lightning's parser (``add_help_option=False``),
    whose usage lists flags longer than 80 columns; ``tests/cli/test_distill.py`` checks their help.
    """
    found = [path]
    if isinstance(command, TyperGroup):
        for name, child in command.commands.items():
            if not child.hidden and child.add_help_option:
                found.extend(_paths(child, (*path, name)))
    return found


PATHS = _paths(typer.main.get_command(app))


def _description(text: str) -> list[str]:
    """The lines of the help between the usage line and the first panel: the docstring, as printed."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if "Usage:" in line) + 1
    end = next((index for index, line in enumerate(lines) if line.lstrip().startswith("╭")), len(lines))
    return [line.rstrip() for line in lines[start:end]]


def _hard_breaks(lines: list[str]) -> list[tuple[str, str]]:
    """Pairs of lines broken mid-sentence where the next word would have fit: a line break the docstring made.

    Rich wraps a paragraph greedily, so a break it makes always leaves too little room for the next word.
    """
    breaks = []
    for line, after in itertools.pairwise(lines):
        if not line.strip() or not after.strip() or re.search(r"[.,:;!?)\]]$", line):
            continue
        word = after.split()[0]
        if word[0].islower() and len(line) + 1 + len(word) < WIDTH:
            breaks.append((line, after))
    return breaks


def test_every_shown_command_is_checked():
    assert ("config", "check") in PATHS
    assert ("reconstruct",) in PATHS
    assert not any(path and path[-1] in {"rec", "tile-stitch", "ts"} for path in PATHS)


@pytest.mark.parametrize("path", PATHS, ids=lambda path: " ".join(("wo", *path)))
def test_help_has_no_mid_sentence_breaks(path, monkeypatch):
    monkeypatch.setattr("typer.rich_utils.MAX_WIDTH", WIDTH)
    result = runner.invoke(app, [*path, "--help"], env={"COLUMNS": str(WIDTH)})
    assert result.exit_code == 0, result.output
    assert max(len(line) for line in result.stdout.splitlines()) <= WIDTH
    description = _description(result.stdout)
    assert any(line.strip() for line in description)
    assert _hard_breaks(description) == []


@pytest.mark.parametrize("path", PATHS, ids=lambda path: " ".join(("wo", *path)))
def test_help_shows_plain_metavars(path, monkeypatch):
    monkeypatch.setattr("typer.rich_utils.MAX_WIDTH", WIDTH)
    result = runner.invoke(app, [*path, "--help"], env={"COLUMNS": str(WIDTH)})
    assert result.exit_code == 0, result.output
    usage = next(line for line in result.stdout.splitlines() if "Usage:" in line)
    assert "{" not in usage
    assert not re.search(r"<str>|<integer>|\[x>=|\[\d<=x", result.stdout)


def test_examples_and_indented_lines_stay_on_their_own_lines():
    text = """Summary line.

    A paragraph wrapped
    over two lines.

    Example: wo rec -i in.zarr
    -c phase.yml
      indented block
      kept as is

    \b
    a do-not-rewrap
    paragraph
    """
    assert _unwrapped(text).split("\n\n") == [
        "Summary line.",
        "A paragraph wrapped over two lines.",
        "Example: wo rec -i in.zarr\n-c phase.yml\n  indented block\n  kept as is",
        "\b\na do-not-rewrap\nparagraph",
    ]
