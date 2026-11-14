"""Recorded data read as a model prepares it: the lazy view the tile methods wrap their sources in.

A tile method reads its source block by block, so the model's ``prepare`` runs on each block as it is read
(:class:`Prepared`). What that needs follows the class's ``prepare_scope``: nothing more for a ``prepare`` that acts
voxel by voxel; the mean of the whole stack, computed once for the volume
(:func:`stack_means`), for one that normalizes by it, so every block shares one normalization; and a single block
per volume for one whose result depends on the block it is given.
"""

import threading

import torch
from torch import Tensor

from waveorder.models._base import ReconstructionModel
from waveorder.models._parts import PrepareScope
from waveorder.nn.parameters import placement
from waveorder.tiling.merge import ArrayLike, Index

_SLAB_VOXELS = 1 << 24
"""Voxels read at once for the mean of a source that declares no chunks."""


class Prepared:
    """A recorded ``(C, Z, Y, X)`` or ``(Z, Y, X)`` source read as the model prepares it, for the tile loader.

    Reads go to the source as they are, and each block is passed through the model's ``prepare`` on the host and
    cast to the model's dtype. By the class's ``prepare_scope``:

    * ``"voxel"``: nothing else is read, and the chunk shape is the source's, so the loader decodes each chunk once;
    * ``"stack_mean"``: the per-channel means of the whole stack are computed once, at the first read
      (:func:`stack_means`), and every block is prepared with them; the chunk shape is the source's;
    * ``"block"``: the view declares no chunks, so a tile is read and prepared in one block (the tile methods refuse
      a plan of several tiles before they build the view).

    Parameters
    ----------
    source : ArrayLike
        The recorded data: anything with ``shape`` and numpy-style slicing.
    model : ReconstructionModel
        The model whose ``prepare`` turns recorded data into what its system measures.

    Attributes
    ----------
    shape : tuple[int, ...]
        The source's shape.
    dtype : torch.dtype
        The dtype blocks are returned in: the model's.
    scope : {"voxel", "stack_mean", "block"}
        The scope of the model's ``prepare``.
    """

    def __init__(self, source: ArrayLike, *, model: ReconstructionModel) -> None:
        self.source = source
        self.model = model
        self.shape = tuple(int(n) for n in source.shape)
        self.dtype, _ = placement(model.system)
        self.scope: PrepareScope = type(model).prepare_scope
        self._means: Tensor | None = None
        self._lock = threading.Lock()

    @property
    def chunks(self) -> tuple[int, ...] | None:
        """The source's chunk shape when a chunk can be prepared by itself, else ``None``."""
        return None if self.scope == "block" else getattr(self.source, "chunks", None)

    @property
    def means(self) -> Tensor | None:
        """Each channel's mean over the whole stack, ``(C,)`` in float64; ``None`` for a model that takes none."""
        if self.scope != "stack_mean":
            return None
        with self._lock:  # the loader reads from several threads; the stack is read once
            if self._means is None:
                self._means = stack_means(self.source)
            return self._means

    def __getitem__(self, key: Index, /) -> Tensor:
        parts = key if isinstance(key, tuple) else (key,)
        full = (*parts, *(slice(None),) * (len(self.shape) - len(parts)))
        dropped = [axis for axis, part in enumerate(full) if isinstance(part, int)]
        slices = tuple(slice(part, part + 1 or None) if isinstance(part, int) else part for part in full)
        block = torch.as_tensor(self.source[slices])
        means = self.means
        channels = len(self.shape) == 4
        if channels and means is not None:
            means = means[slices[0]]
        block = block if channels else block[None]
        block = self.model.prepare(block, mean=means).to(self.dtype)
        block = block if channels else block[0]
        for axis in reversed(dropped):
            block = block.squeeze(axis)
        return block


def stack_means(source: ArrayLike) -> Tensor:
    """Return each channel's mean over ``(Z, Y, X)`` of a ``(C, Z, Y, X)`` or ``(Z, Y, X)`` source, ``(C,)`` in float64.

    The source is read slab by slab along z (one chunk of z at a time when it declares chunks) and summed in float64,
    so a volume larger than memory is never held whole.
    """
    *lead, z, y, x = (int(n) for n in source.shape)
    chunks = getattr(source, "chunks", None)
    step = max(1, int(chunks[-3])) if chunks is not None else max(1, _SLAB_VOXELS // (y * x))
    plane = (slice(None), slice(None))
    means = []
    for channel in range(lead[0] if lead else 1):
        total = 0.0
        for start in range(0, z, step):
            slab = (slice(start, min(start + step, z)), *plane)
            block = source[(slice(channel, channel + 1), *slab)] if lead else source[slab]
            total += float(torch.as_tensor(block).double().sum())
        means.append(total / (z * y * x))
    return torch.tensor(means, dtype=torch.float64)
