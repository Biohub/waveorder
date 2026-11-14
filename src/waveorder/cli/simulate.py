"""``wo simulate``: image a drawn object through the configuration's imaging system and write three stores.

The configuration's ``simulate: {seed, shape}`` block makes it a simulation. One generator seeded with ``seed``
draws, in this order: the truth of every imaging-system number written with a numeric tolerance (nominal plus
tolerance times a standard normal draw, clamped to its bounds; Zernike shorthand terms first in Noll order, then the
named terms), the object, and the noise (``model.simulate``). ``-o`` receives ``data.zarr``, ``object.zarr`` and
``tf.zarr``.
"""

from pathlib import Path
from typing import Annotated

import typer

from waveorder.cli.options import ConfigFilepath, Overwrite


def simulate(
    *,
    config_filepath: ConfigFilepath,
    output_dirpath: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dirpath",
            file_okay=False,
            metavar="DIR",
            rich_help_panel="Outputs",
            help="Directory that receives data.zarr, object.zarr and tf.zarr.",
        ),
    ],
    overwrite: Overwrite = False,
) -> None:
    """Simulate data from the configuration's imaging system and write data.zarr, object.zarr and tf.zarr.

    The simulate block gives the volume shape and the seed. Every imaging-system number written with a numeric
    tolerance is drawn from it (clamped to its bounds); the object is a Gaussian random field with the power
    spectrum of object.source; the noise is the noise block's (sigma or snr_db, or poisson photons). data.zarr holds
    the measurement in channel data.input_channel_names, object.zarr the object, and tf.zarr the drawn values, the
    transfer function and the provenance of the simulation.

    Example: wo sim -c simulate.yml -o simulated
    """
    from waveorder.cli import _simulate

    _simulate.simulate(config_filepath, output_dirpath, overwrite=overwrite)
