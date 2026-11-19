"""``wo view`` and ``wo bm view``: OME-Zarr datasets and transfer-function stores as napari image layers.

The readers return layer data, ``(array, keyword arguments of add_image)``, and import no napari, so a command reads
every path before :func:`show` opens a window.
"""

from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import numpy as np

from waveorder.errors import Rejected

if TYPE_CHECKING:  # napari and torch load only when a command runs; the aliases below are evaluated when inspected
    from napari.components import ViewerModel

    from waveorder.io.tf_store import TransferArray

type _Viewer = ViewerModel
type _Transfer = TransferArray

type LayerData = tuple[np.ndarray, dict[str, Any]]
"""One image layer: its array and the keyword arguments of ``viewer.add_image``."""

type Crop = tuple[slice, slice, slice]
"""A box to read, along z, y and x."""


def dataset_layers(paths: Sequence[str | Path]) -> list[LayerData]:
    """Return the layers of OME-Zarr datasets and transfer-function stores; with several, names start with each name.

    Parameters
    ----------
    paths : sequence of str or Path
        Plates, single-position stores or ``tf.zarr`` stores.

    Returns
    -------
    list of LayerData

    Raises
    ------
    Rejected
        If a path is neither an OME-Zarr node nor a transfer-function store, or a store cannot be read or holds no
        transfer function to show.
    """
    from waveorder.io.tf_store import store_layout

    layers: list[LayerData] = []
    for path in paths:
        prefix = f"{PurePath(path).stem}: " if len(paths) > 1 else ""
        reader = transfer_layers if store_layout(path) is not None else ome_zarr_layers
        layers += reader(path, prefix=prefix)
    return layers


def ome_zarr_layers(
    path: str | Path,
    *,
    prefix: str = "",
    channels: Sequence[str] | None = None,
    time: int | None = None,
    crop: Crop | None = None,
    windowed: bool = True,
) -> list[LayerData]:
    """Return one layer per channel (and per position of a plate), singleton T and Z squeezed.

    A layer is scaled by the store's voxel size and, when ``windowed``, takes its contrast limits from the omero window.

    Parameters
    ----------
    path : str or Path
        A plate or a single-position store.
    prefix : str
        Start of every layer name; a plate's layers end with ``[<position>]``.
    channels : sequence of str, optional
        Only these channels; by default all.
    time : int, optional
        Only this time point; by default all.
    crop : (slice, slice, slice), optional
        Read only this z, y, x box.
    windowed : bool
        Take the contrast limits from the omero window; else napari computes them.

    Returns
    -------
    list of LayerData

    Raises
    ------
    Rejected
        If ``path`` is no OME-Zarr position or plate.
    """
    from waveorder.io.positions import open_positions

    times = slice(None) if time is None else slice(time, time + 1)
    layers: list[LayerData] = []
    with open_positions(Path(path)) as positions:
        several = len(positions) > 1
        for position in positions:
            data = np.asarray(position["0"][(times, slice(None), *(crop or ()))])  # (T, C, Z, Y, X)
            t_scale, _, *zyx_scale = position.scale
            omero = position.metadata.omero
            windows = (omero.channels or []) if windowed and omero is not None else []
            for index, channel in enumerate(position.channel_names):
                if channels is not None and channel not in channels:
                    continue
                volume, scale = data[:, index], [t_scale, *zyx_scale]
                if volume.shape[1] == 1:
                    volume, scale = volume[:, 0], [t_scale, *zyx_scale[1:]]
                if volume.shape[0] == 1:
                    volume, scale = volume[0], scale[1:]
                name = f"{prefix}{channel} [{position.zgroup.path}]" if several else f"{prefix}{channel}"
                kwargs: dict[str, Any] = {"name": name, "scale": tuple(scale)}
                limits = _window(windows[index].window) if index < len(windows) else None
                if limits is not None:
                    kwargs["contrast_limits"] = limits
                layers.append((volume, kwargs))
    return layers


def _window(window: object) -> tuple[float, float] | None:
    """Return the ``(start, end)`` of an omero channel window, or ``None`` for none or an empty one."""
    if window is None:
        return None
    if isinstance(window, dict):
        low, high = window.get("start", 0), window.get("end", 1)
    else:
        low, high = getattr(window, "start", 0), getattr(window, "end", 1)
    return (float(low), float(high)) if low < high else None


def transfer_layers(path: str | Path, *, prefix: str = "") -> list[LayerData]:
    """Return the real and imaginary parts of each transfer function a ``tf.zarr`` holds, on the centred full spectrum.

    One pair of layers per channel and field of a 3D transfer, per field of a thin (2D) system's plane transfers
    (planes along the first axis); each with the ``bwr`` colormap and limits symmetric about zero.

    Parameters
    ----------
    path : str or Path
        The store.
    prefix : str
        Start of every layer name.

    Returns
    -------
    list of LayerData

    Raises
    ------
    Rejected
        As :meth:`waveorder.io.TfStore.load`, or if the store holds no transfer function this waveorder reads.
    """
    from waveorder.io.tf_store import TfStore, read_transfer

    store = TfStore.load(path)
    layers: list[LayerData] = []
    for item in store.transfers:
        read = read_transfer(path, item.shape)
        if read is not None and read.transfer is not None:
            grid = f" {'x'.join(map(str, item.shape))}" if len(store.transfers) > 1 else ""
            layers += _spectrum_layers(read, f"{prefix}transfer_function{grid}")
    if not layers:
        message = "holds no transfer function: wo compute-tf stores one when the outputs list transfer_function"
        if store.origin == "#592":
            message = "is a #592 store, whose transfer_function array this waveorder does not read"
        raise Rejected.one(message, location=str(path))
    return layers


def _spectrum_layers(read: _Transfer, name: str) -> list[LayerData]:
    """Return the layers of one stored transfer, ``(C, F, Z, Y, X // 2 + 1)`` or a thin system's plane transfers."""
    values = read.transfer
    assert values is not None  # read_transfer reads the array
    if not values.is_complex() and read.attrs.get("real") is False and values.shape[-1] == 2:
        values = values[..., 0] + 1j * values[..., 1]
    half = values.resolve_conj().numpy()
    layout = read.attrs.get("layout")
    if isinstance(layout, list) and layout[0] == "part":  # [Hermitian, anti-Hermitian]: the forward model's first
        half, layout = half[0], layout[1:]
    planes = isinstance(layout, list) and layout[0] == "plane"
    axes = (-2, -1) if planes else (-3, -2, -1)
    spectrum = np.fft.fftshift(full_spectrum(half, read.shape[-1], axes=axes), axes=axes)
    if planes:
        spectrum = spectrum[:, :, 0].swapaxes(0, 1)[None]  # (1, F, P, Y, X): each field over the planes
    channels, fields = spectrum.shape[:2]
    layers: list[LayerData] = []
    for c in range(channels):
        for f in range(fields):
            index = [f"c{c}"] * (channels > 1) + [f"f{f}"] * (fields > 1)
            label = " ".join([name, *index])
            array = spectrum[c, f]
            limit = float(np.abs(array).max()) or 1.0
            shown = {"colormap": "bwr", "contrast_limits": (-limit, limit)}
            layers.append((np.ascontiguousarray(array.real), {"name": f"Re({label})", **shown}))
            if np.iscomplexobj(array):
                layers.append((np.ascontiguousarray(array.imag), {"name": f"Im({label})", **shown}))
    return layers


def full_spectrum(half: np.ndarray, x: int, *, axes: Sequence[int]) -> np.ndarray:
    """Return the full spectrum, FFT order, of an ``rfftn`` half spectrum: each missing frequency is ``conj(H(-k))``.

    Parameters
    ----------
    half : ndarray
        Half spectrum, ``x // 2 + 1`` along the last axis.
    x : int
        Length of the last axis of the full spectrum.
    axes : sequence of int
        The axes of the transform; the last is the halved one.

    Returns
    -------
    ndarray
        ``x`` long along the last axis.
    """
    mirrored = np.conj(half[..., 1 : x - x // 2][..., ::-1])
    for axis in axes[:-1]:
        mirrored = np.roll(np.flip(mirrored, axis), 1, axis)
    return np.concatenate([half, mirrored], axis=-1)


def add_layers(viewer: _Viewer, layers: Sequence[LayerData], *, linked: Sequence[tuple[str, str]] = ()) -> None:
    """Add ``layers`` to ``viewer`` in grid view, the contrast limits of each pair of layer names in ``linked`` shared.

    A linked pair starts at the range of both layers' values.
    """
    for data, kwargs in layers:
        viewer.add_image(data, **kwargs)
    if linked:
        from napari.experimental import link_layers
        from napari.layers import Image

        arrays = {kwargs["name"]: data for data, kwargs in layers}
        for names in linked:
            pair = [layer for layer in viewer.layers if isinstance(layer, Image) and layer.name in names]
            low = min(float(arrays[name].min()) for name in names)
            high = max(float(arrays[name].max()) for name in names)
            for layer in pair:
                layer.contrast_limits = (low, high)
            link_layers(pair, ("contrast_limits",))
    viewer.canvas.grid.enabled = True


def show(layers: Sequence[LayerData], *, title: str, command: str, linked: Sequence[tuple[str, str]] = ()) -> None:
    """Open a napari window with ``layers`` (:func:`add_layers`) and run its event loop until it closes.

    Raises
    ------
    Rejected
        At ``command``, if napari is not installed.
    """
    try:
        import napari
    except ImportError as error:
        raise Rejected.one(f"needs napari ({error.name}): pip install waveorder[visual]", location=command) from None
    viewer = napari.Viewer(title=title)
    add_layers(viewer, layers, linked=linked)
    napari.run()
