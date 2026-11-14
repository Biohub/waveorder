"""The tile planner: which region each tile reads, which it contributes to, with what weights, and how much halo.

Along each axis a tile has three nested regions: the **core**, one cell of a non-overlapping grid that is written
exactly once; the **blend region**, the core widened by ``overlap`` on each side, cross-faded with its neighbours; and
the **read region**, the blend region widened by ``halo``, aligned to storage chunks and lengthened to an FFT-friendly
size. A tile is reconstructed on its read region and only its blend region is kept. The halo does not cover the
volume's outer edge, where a whole-volume circular solve wraps: compare on an interior mask.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import Literal

import torch
from torch import Tensor

from waveorder.errors import ParameterError
from waveorder.nn.model import ImagingModel
from waveorder.nn.parameters import placement
from waveorder.nn.sampling import ZYX, Sampling
from waveorder.nn.solvers import Solver
from waveorder.nn.systems.base import ImagingSystem
from waveorder.observe import operation

type Blend = Literal["linear", "cosine"]
"""Cross-fade across an overlap: a linear ramp, or a raised-cosine (``sin^2``) ramp."""

type Span = tuple[int, int]
"""A half-open voxel interval ``[start, stop)`` along one axis."""

type Region = ZYX[Span]
"""A box of voxels: one span per axis, z, y, x."""

type ChunkKey = tuple[int, int, int, int]
"""A storage chunk: volume index, then chunk indices along z, y, x."""

_AXES = ("z", "y", "x")


def relative(inner: Region, outer: Region) -> tuple[slice, slice, slice]:
    """Return the slices of ``inner`` inside an array covering ``outer``."""
    z, y, x = (slice(a - o, b - o) for (a, b), (o, _) in zip(inner, outer, strict=True))
    return (z, y, x)


def intersect(first: Region, second: Region) -> Region | None:
    """Return the intersection of two regions, or ``None`` if they do not overlap."""
    z, y, x = ((max(a, c), min(b, d)) for (a, b), (c, d) in zip(first, second, strict=True))
    if any(a >= b for a, b in (z, y, x)):
        return None
    return (z, y, x)


@dataclass(frozen=True, slots=True)
class TileRegion:
    """The regions of one tile, in volume voxel coordinates.

    Attributes
    ----------
    index : int
        Position of the tile in :attr:`TilePlan.regions` (raster order, x fastest).
    grid : tuple[int, int, int]
        Position of the tile on the core grid along z, y, x.
    core, blend, read : Region
        The core it owns, the region it is stitched into with weights, and the region read and reconstructed.
    """

    index: int
    grid: ZYX[int]
    core: Region
    blend: Region
    read: Region

    @property
    def read_shape(self) -> ZYX[int]:
        """Voxels of the read region along z, y, x."""
        (z0, z1), (y0, y1), (x0, x1) = self.read
        return (z1 - z0, y1 - y0, x1 - x0)

    @property
    def crop(self) -> tuple[slice, slice, slice]:
        """Slices that cut the blend region out of an array covering the read region: what is kept of a tile."""
        return relative(self.blend, self.read)


@dataclass(frozen=True, slots=True)
class _Axis:
    cores: tuple[Span, ...]
    blends: tuple[Span, ...]
    reads: tuple[Span, ...]


def _plan_axis(n: int, *, tile: int, overlap: int, halo: int, chunk: int) -> _Axis:
    cores = tuple((start, min(start + tile, n)) for start in range(0, n, tile))
    if len(cores) == 1:
        whole = ((0, n),)
        return _Axis(cores, whole, whole)
    blends = tuple((max(a - overlap, 0), min(b + overlap, n)) for a, b in cores)
    wanted = [(max(a - halo, 0), min(b + halo, n)) for a, b in blends]
    aligned = [(a // chunk * chunk, min(-(-b // chunk) * chunk, n)) for a, b in wanted]
    # One read length for every tile, so one bound operator serves the whole plan; a tile near the high edge shifts
    # inward instead of shrinking, which only gives it more support.
    length = _smooth_length(max(b - a for a, b in aligned), limit=n, step=chunk)
    reads = tuple((min(a, n - length), min(a, n - length) + length) for a, _ in aligned)
    return _Axis(cores, blends, reads)


def _is_smooth(n: int) -> bool:
    for factor in (2, 3, 5, 7):
        while n % factor == 0:
            n //= factor
    return n == 1


def _smooth_length(n: int, *, limit: int, step: int) -> int:
    """Return the smallest ``n + k * step`` with only prime factors 2, 3, 5 and 7, capped at ``limit``."""
    if n % step == 0 and not _is_smooth(step):
        return min(n, limit)
    length = n
    while length < limit and not _is_smooth(length):
        length += step
    return min(length, limit)


def _ramp(positions: Tensor, *, start: int, width: int, blend: Blend) -> Tensor:
    """Weights rising from 0 to 1 across ``[start, start + width)``, at voxel centres."""
    t = ((positions - start + 0.5) / width).clamp(0, 1)
    return t if blend == "linear" else torch.sin(t * (math.pi / 2)) ** 2


def _axis_weights(axis: _Axis, *, overlap: int, blend: Blend) -> tuple[Tensor, ...]:
    """Per-tile 1D weights along one axis, normalized so they sum to exactly 1 at every voxel."""
    count = len(axis.cores)
    raw = []
    for j, ((ca, cb), (a, b)) in enumerate(zip(axis.cores, axis.blends, strict=True)):
        positions = torch.arange(a, b, dtype=torch.float64)
        weight = torch.ones_like(positions)
        if overlap:
            if j > 0:
                weight = weight * _ramp(positions, start=ca - overlap, width=2 * overlap, blend=blend)
            if j < count - 1:
                weight = weight * (1 - _ramp(positions, start=cb - overlap, width=2 * overlap, blend=blend))
        raw.append(weight)
    total = torch.zeros(axis.cores[-1][1], dtype=torch.float64)
    for (a, b), weight in zip(axis.blends, raw, strict=True):
        total[a:b] += weight
    return tuple(weight / total[a:b] for (a, b), weight in zip(axis.blends, raw, strict=True))


def _triple(value: ZYX[int], *, name: str, minimum: int) -> ZYX[int]:
    """Check a z, y, x triple of ints; error paths are the argument (``tile``) and its axis (``tile.y``)."""
    if len(value) != 3:
        raise ParameterError(f"TilePlan {name} must be three integers >= {minimum} (z, y, x), got {value}", path=name)
    for axis, v in zip(_AXES, value, strict=True):
        if v < minimum:
            raise ParameterError(f"TilePlan {name}.{axis} must be >= {minimum}, got {v}", path=f"{name}.{axis}")
    z, y, x = value
    return (z, y, x)


def _overlaps(first: Span, second: Span) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


@dataclass(frozen=True, kw_only=True)
class TilePlan:
    """Tiling of a ``(Z, Y, X)`` volume into cores, blend regions and read regions.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Voxels of the volume along z, y, x.
    tile : tuple[int, int, int]
        Core size along z, y, x in voxels. With several write threads, make z and y multiples of the sink's chunks
        (shards, if sharded).
    overlap : tuple[int, int, int]
        Voxels each blend region extends past its core on each side; at most ``tile // 2`` where an axis has several
        tiles.
    halo : tuple[int, int, int]
        Voxels of extra support read beyond the blend region on each side, then discarded; see :func:`halo_for`.
    chunks : tuple[int, int, int], optional
        Source chunk shape along z, y, x; read regions are aligned outward to it. ``None`` for an unchunked source.
    blend : {"linear", "cosine"}
        Cross-fade across overlaps; weights sum to 1 at every voxel.

    Raises
    ------
    ParameterError
        If a field is invalid; ``path`` is the argument and axis, e.g. ``tile.y``, ``overlap.x``.
    """

    shape: ZYX[int]
    tile: ZYX[int]
    overlap: ZYX[int] = (0, 0, 0)
    halo: ZYX[int] = (0, 0, 0)
    chunks: ZYX[int] | None = None
    blend: Blend = "cosine"
    _axes: tuple[_Axis, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        shape = _triple(self.shape, name="shape", minimum=1)
        tile = _triple(self.tile, name="tile", minimum=1)
        overlap = _triple(self.overlap, name="overlap", minimum=0)
        halo = _triple(self.halo, name="halo", minimum=0)
        chunks = None if self.chunks is None else _triple(self.chunks, name="chunks", minimum=1)
        # Lists are stored as tuples, so plans hash and compare as values.
        for name, value in (("shape", shape), ("tile", tile), ("overlap", overlap), ("halo", halo), ("chunks", chunks)):
            object.__setattr__(self, name, value)
        if self.blend not in ("linear", "cosine"):
            raise ParameterError(f"TilePlan blend must be 'linear' or 'cosine', got {self.blend!r}", path="blend")
        for name, n, t, o in zip(_AXES, shape, tile, overlap, strict=True):
            if t < n and 2 * o > t:
                raise ParameterError(
                    f"TilePlan overlap.{name} = {o} exceeds half of tile.{name} = {t}; blend regions of one tile "
                    "would cross",
                    path=f"overlap.{name}",
                )
        axes = tuple(
            _plan_axis(n, tile=t, overlap=o, halo=h, chunk=c)
            for n, t, o, h, c in zip(shape, tile, overlap, halo, chunks or (1, 1, 1), strict=True)
        )
        object.__setattr__(self, "_axes", axes)

    @property
    def grid(self) -> ZYX[int]:
        """Number of tiles along z, y, x."""
        z, y, x = (len(axis.cores) for axis in self._axes)
        return (z, y, x)

    @property
    def read_shape(self) -> ZYX[int]:
        """Voxels every tile reads along z, y, x: the shape its system binds to."""
        z, y, x = (axis.reads[0][1] - axis.reads[0][0] for axis in self._axes)
        return (z, y, x)

    @cached_property
    def regions(self) -> tuple[TileRegion, ...]:
        """Every tile, in raster order (x fastest), which is the order tiles are read in."""

        def region(kind: str, grid: ZYX[int]) -> Region:
            z, y, x = (getattr(axis, kind)[g] for axis, g in zip(self._axes, grid, strict=True))
            return (z, y, x)

        regions = []
        for index, (i, j, k) in enumerate(product(*(range(n) for n in self.grid))):
            grid = (i, j, k)
            regions.append(
                TileRegion(index, grid, region("cores", grid), region("blends", grid), region("reads", grid))
            )
        return tuple(regions)

    @cached_property
    def weights(self) -> ZYX[tuple[Tensor, ...]]:
        """Per-axis float64 blend weights: ``weights[axis][g]`` is 1D over the blend span of grid position ``g``.

        A tile's 3D weight is the outer product of its three 1D weights.
        """
        z, y, x = (
            _axis_weights(axis, overlap=o, blend=self.blend) for axis, o in zip(self._axes, self.overlap, strict=True)
        )
        return (z, y, x)

    def _neighbours(self, grid: ZYX[int], *, of: str, onto: str) -> tuple[int, ...]:
        """Return the tiles whose ``onto`` span overlaps the ``of`` span of grid position ``grid`` on every axis."""
        ranges = []
        for axis, g in zip(self._axes, grid, strict=True):
            inner, outer = getattr(axis, of), getattr(axis, onto)
            ranges.append([h for h in (g - 1, g, g + 1) if 0 <= h < len(inner) and _overlaps(outer[h], inner[g])])
        _, ny, nx = self.grid
        return tuple(sorted((i * ny + j) * nx + k for i, j, k in product(*ranges)))

    def contributors(self, block: TileRegion) -> tuple[int, ...]:
        """Return the indices of the tiles whose blend region overlaps ``block``'s core, ``block`` included."""
        return self._neighbours(block.grid, of="cores", onto="blends")

    def blocks(self, region: TileRegion) -> tuple[int, ...]:
        """Return the indices of the cores ``region``'s blend region overlaps: where its contribution goes."""
        return self._neighbours(region.grid, of="blends", onto="cores")

    @cached_property
    def peak_pending_blocks(self) -> int:
        """Most cores holding a partial sum at once when tiles finish in raster order: the accumulator count."""
        remaining = [len(self.contributors(region)) for region in self.regions]
        open_blocks: set[int] = set()
        peak = 0
        for region in self.regions:
            targets = self.blocks(region)
            open_blocks.update(targets)
            for block in targets:
                remaining[block] -= 1
            peak = max(peak, len(open_blocks))
            open_blocks -= {block for block in targets if remaining[block] == 0}
        return peak

    @cached_property
    def peak_open_rows(self) -> int:
        """Most rows of cores (same z and y grid position) holding some but not all finished cores at once."""
        spans: dict[tuple[int, int], list[int]] = {}
        for region in self.regions:
            done = max(self.contributors(region))
            span = spans.setdefault(region.grid[:2], [done, done])
            span[0], span[1] = min(span[0], done), max(span[1], done)
        # At equal times opens sort first: a row that opens and closes with one tile still needs a buffer.
        events = sorted([(a, -1) for a, _ in spans.values()] + [(b, 1) for _, b in spans.values()])
        live = peak = 0
        for _, change in events:
            live -= change
            peak = max(peak, live)
        return peak


def chunk_keys(volume: int, region: Region, chunks: ZYX[int]) -> tuple[ChunkKey, ...]:
    """Return the chunks ``region`` touches in volume ``volume``, z slowest."""
    z, y, x = (range(a // c, (b - 1) // c + 1) for (a, b), c in zip(region, chunks, strict=True))
    return tuple((volume, i, j, k) for i, j, k in product(z, y, x))


def peak_chunk_bytes(reads: Sequence[tuple[int, Region]], *, chunks: ZYX[int], chunk_bytes: int, ahead: int) -> int:
    """Return the most bytes of decoded chunks held when each chunk lives from its first to its last tile.

    ``reads`` holds ``(volume, read region)`` per tile in reading order; ``chunk_bytes`` is one decoded chunk (all
    channels); ``ahead`` counts the tiles read ahead of the oldest unfinished one.
    """
    first: dict[ChunkKey, int] = {}
    last: dict[ChunkKey, int] = {}
    for index, (volume, region) in enumerate(reads):
        for key in chunk_keys(volume, region, chunks):
            first.setdefault(key, index)
            last[key] = index
    starts, ends = Counter(first.values()), Counter(last.values())
    live = peak = 0
    for index in range(len(reads) + ahead):
        live += starts[index]
        peak = max(peak, live)
        live -= ends[index - ahead]
    return peak * chunk_bytes


def _responses(target: ImagingSystem | ImagingModel, sampling: Sampling, *, solver: Solver | None) -> Tensor:
    """Return the summed ``|response|`` ``(Z, Y, X)`` to a unit impulse at the origin of each input volume."""
    system = target.system if isinstance(target, ImagingModel) else target
    dtype, device = placement(system)
    total = torch.zeros(sampling.shape, dtype=torch.float64, device=device)

    def impulse(shape: tuple[int, ...], index: int) -> Tensor:
        delta = torch.zeros(shape, dtype=dtype, device=device)
        delta[index, 0, 0, 0] = 1
        return delta

    with torch.no_grad(), operation():  # probes, not reconstructions: they report no events
        if solver is None:
            operator = system.bind(sampling)
            for index in range(len(system.fields)):
                total += operator.apply(impulse(system.object_shape(sampling), index)).abs().sum(0)
            return total
        if not isinstance(target, ImagingModel):
            raise ParameterError(
                "halo_for: a solver needs the ImagingModel it reconstructs with, not only its system", path="solver"
            )
        for channel in range(system.channel_count):
            data = impulse(system.data_shape(sampling), channel)
            total += solver(data, target, sampling).estimate.abs().sum(0)
    return total


def _expect_energy(energy: float, *, where: str) -> None:
    if not 0 < energy < 1:
        raise ParameterError(f"{where}: energy must be in (0, 1), got {energy}", path="energy")


def halo_for(
    target: ImagingSystem | ImagingModel, sampling: Sampling, *, energy: float = 1 - 1e-3, solver: Solver | None = None
) -> ZYX[int]:
    """Return the halo, in voxels along z, y, x, that keeps a fraction ``energy`` of the kernel's L1 mass per axis.

    Without ``solver`` the kernel is the point-spread function of ``target``, a lower bound for a deconvolution. With
    ``solver`` it is the reconstruction's impulse response; the bound holds for a linear shift-invariant solver
    (:class:`~waveorder.nn.Wiener`). A solver that states its own halo (:meth:`~waveorder.nn.Solver.tile_halo`) gets it.

    Parameters
    ----------
    target : ImagingSystem or ImagingModel
        The system, or the model to reconstruct with (required with ``solver``).
    sampling : Sampling
        One tile's read region (the plan's ``read_shape``). The halo never exceeds half of it.
    energy : float
        Fraction of the kernel's L1 mass, in (0, 1), to keep within the halo along each axis.
    solver : Solver, optional
        Use the reconstruction's impulse response instead of the point-spread function.

    Returns
    -------
    tuple[int, int, int]
        Halo along z, y, x in voxels.

    Raises
    ------
    ParameterError
        If ``energy`` is not in (0, 1), or ``solver`` is given with a bare system; ``path`` names the argument.
    """
    _expect_energy(energy, where="halo_for")
    declared = None if solver is None else solver.tile_halo()
    if declared is not None:
        z, y, x = (min(h, n // 2) for h, n in zip(declared, sampling.shape, strict=True))
        return (z, y, x)
    magnitude = _responses(target, sampling, solver=solver)
    halo = []
    for axis, n in enumerate(sampling.shape):
        profile = magnitude.sum(tuple(a for a in range(3) if a != axis))
        index = torch.arange(n, device=profile.device)
        distance = torch.minimum(index, n - index)  # circular distance to the kernel centre at index 0
        mass = torch.zeros(n // 2 + 1, dtype=profile.dtype, device=profile.device).index_add_(0, distance, profile)
        reached = torch.nonzero(torch.cumsum(mass, 0) / profile.sum() >= energy)
        halo.append(int(reached[0]) if len(reached) else n // 2)
    z, y, x = halo
    return (z, y, x)
