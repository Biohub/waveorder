"""Tiled reconstruction of volumes larger than one solve: split, reconstruct each tile, merge.

:mod:`~waveorder.tiling.plan` splits a volume (:class:`TilePlan`, :func:`halo_for`), :mod:`~waveorder.tiling.merge`
blends estimates into a sink, and :mod:`~waveorder.tiling.run` drives the tiles (:func:`reconstruct_series`). The
modules in :mod:`waveorder.nn` know nothing about where a tile sits: a tile is data on its own sampling.
"""

from waveorder.tiling.plan import TilePlan, halo_for
from waveorder.tiling.run import TileOptions, TileStats, estimate_memory, reconstruct_series, recover_zernikes

__all__ = [
    "TileOptions",
    "TilePlan",
    "TileStats",
    "estimate_memory",
    "halo_for",
    "reconstruct_series",
    "recover_zernikes",
]
