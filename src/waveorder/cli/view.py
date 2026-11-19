"""``wo view``: OME-Zarr datasets and transfer-function stores in napari (:mod:`waveorder.plugin.view`)."""

import sys
from pathlib import Path
from typing import Annotated

import typer


# the docstring is the command's help, where Rich would read an unescaped [visual] as markup
def view(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(metavar="PATHS", help="OME-Zarr datasets or transfer-function stores; more on stdin."),
    ] = None,
) -> None:
    r"""Open OME-Zarr datasets or transfer-function stores in napari.

    Paths come as arguments and/or on stdin, one per line. A plate or position gives one layer per channel (and per
    position); a tf.zarr the real and imaginary parts of its transfer functions on the centred full spectrum. With
    several paths, layer names start with each store's name. Needs pip install waveorder\[visual].

    Example: wo view ./input.zarr ./reconstruction.zarr ./tf.zarr
    """
    listed = list(paths or [])
    if not sys.stdin.isatty():
        listed += [Path(line.strip()) for line in sys.stdin if line.strip()]
    if not listed:
        raise typer.BadParameter("no paths given, as arguments or on stdin", param_hint="PATHS")

    from waveorder.plugin.view import dataset_layers, show

    show(dataset_layers(listed), title=f"wo view: {', '.join(map(str, listed))}", command="wo view")
