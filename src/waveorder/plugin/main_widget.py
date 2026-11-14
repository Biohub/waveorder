"""The waveorder dock widget: the LC Calibration, Reconstruction and Visualization tabs.

The calibration and visualization tabs are laid out in Qt Designer (``gui.ui``, loaded as :attr:`MainWidget.ui`); the
Reconstruction tab is :class:`waveorder.plugin.tab_recon.ReconTab`. Polarization calibration (Micro-Manager and the
liquid crystals) and birefringence are not in this version: the LC Calibration tab is shown, disabled, with why.
"""

import logging
import textwrap
from importlib.resources import files

import dask.array as da
import numpy as np
from napari.components import ViewerModel  # noqa: TID253 (napari loads this module, and passes the viewer by this type)
from napari.utils.events import Event  # noqa: TID253
from pyqtgraph import PlotWidget
from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qtpy.uic import loadUi

from waveorder.plugin.overlay import ret_ori_overlay
from waveorder.plugin.tab_recon import NOT_IN_THIS_VERSION, ReconTab

logger = logging.getLogger(__name__)

CALIBRATION_UNAVAILABLE = (
    f"Polarization calibration with Micro-Manager is for birefringence; {NOT_IN_THIS_VERSION}. "
    "This version reconstructs phase and fluorescence (Reconstruction tab)."
)

CALIBRATION_MODES = (
    ("MM-Retardance", "The retardance of the LC is set directly through the Micro-Manager LC device adapter."),
    ("MM-Voltage", "The LC voltage for a retardance, from the LC calibration data, is set through Micro-Manager."),
    ("DAC", "The LC voltage for a retardance is applied to the LC controller through the TriggerScope DAC outputs."),
)
"""The LC control modes wo-4.0 offered, shown in the disabled calibration tab."""

PLUGIN_LOGGER = "waveorder.plugin"
"""The logger the Advanced tab's log box shows: the commands of the jobs, and what the dock widget does."""


def _load_asset_pixmap(name: str) -> QPixmap:
    """Return a bundled image of ``waveorder/assets``.

    Raises
    ------
    ValueError
        If the image cannot be decoded.
    """
    data = (files("waveorder") / "assets" / name).read_bytes()
    pixmap = QPixmap()
    if not pixmap.loadFromData(data):
        raise ValueError(f"Unable to load bundled image: {name}")
    return pixmap


class _NoWheel(QObject):
    """Ignore wheel events, so scrolling the dock widget does not change a combo box under the pointer."""

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:  # noqa: N802 (Qt's names)
        """Swallow wheel events."""
        if a1 is not None and a1.type() == QEvent.Type.Wheel:
            return True
        return super().eventFilter(a0, a1)


class QtLogger(logging.Handler):
    """A logging handler that writes to the log box of the Advanced tab."""

    def __init__(self, widget: QPlainTextEdit) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        """Append ``record`` to the log box."""
        self.widget.appendPlainText(self.format(record))


class MainWidget(QWidget):
    """The waveorder dock widget.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The viewer the widget shows reconstructions and overlays in.
    """

    def __init__(self, napari_viewer: ViewerModel) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.ui = loadUi(str(files("waveorder.plugin") / "gui.ui"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ui)
        self.setObjectName("waveorder")
        self.ret_max = 25
        self.overlay_retardance: da.Array | None = None
        self.overlay_orientation: da.Array | None = None
        self.overlay_name = ""
        self.overlay_scale: tuple[float, ...] = ()

        self.reconstruction_tab = ReconTab(self)
        self.reconstruction_tab.set_viewer(napari_viewer)
        tabs = self._get(QTabWidget, "tabWidget")
        tabs.insertTab(1, self.reconstruction_tab, "Reconstruction")
        tabs.setTabText(tabs.indexOf(self._get(QWidget, "Display")), "Visualization")
        tabs.setCurrentWidget(self.reconstruction_tab)
        self._get(QTabWidget, "tabWidget_2").setCurrentIndex(0)

        self._init_visuals()
        self._disable_calibration()
        self._init_logging()
        # a reader or a reconstruction adds layers or moves them to the top, which (re)draws the overlay
        self.viewer.layers.events.moved.connect(self.handle_layers_updated)
        self.viewer.layers.events.inserted.connect(self.handle_layers_updated)
        self._get(QSlider, "retMaxSlider").sliderMoved.connect(self.handle_ret_max_slider_move)

    def _get[W: QObject](self, kind: type[W], name: str) -> W:
        widget = self.ui.findChild(kind, name)
        if widget is None:
            raise LookupError(f"gui.ui has no {kind.__name__} named {name}")
        return widget

    def _init_visuals(self) -> None:
        plot_widget = self._get(PlotWidget, "plot_widget")
        plot_widget.getPlotItem().setLabel("left", "Intensity")
        plot_widget.setBackground((32, 34, 40))
        self._get(QLabel, "label_orientation_image").setPixmap(_load_asset_pixmap("HSV_legend.png"))
        self._get(QLabel, "label_logo").setPixmap(_load_asset_pixmap("waveorder_plugin_logo.png"))
        for name in ("label_lca", "label_lcb"):
            self._get(QLabel, name).hide()
        for name in ("cb_lca", "cb_lcb"):
            self._get(QComboBox, name).hide()
        # the overlay's colormap and display options are not offered, as in wo-4.0
        self._get(QLabel, "label_orientation_legend").setHidden(True)
        self._get(QWidget, "DisplayOptions").setHidden(True)
        self._get(QLabel, "label_extinction").setText("Extinction Ratio")
        self._get(QPlainTextEdit, "te_log").setStyleSheet("background-color: rgb(32,34,40);")
        self.setStyleSheet("QTabWidget::tab-bar {alignment: center;}")
        modes = self._get(QComboBox, "cb_calib_mode")
        for index, (name, description) in enumerate(CALIBRATION_MODES):
            modes.addItem(name)
            modes.setItemData(index, "\n".join(textwrap.wrap(description, width=70)), Qt.ItemDataRole.ToolTipRole)
        self._no_wheel = _NoWheel(self)
        for combo in self.ui.findChildren(QComboBox):
            if combo.objectName().startswith("cb_"):
                combo.installEventFilter(self._no_wheel)

    def _disable_calibration(self) -> None:
        """Show the LC Calibration tab and Connect to MM disabled, with why; the calibration notes too."""
        calibration = self._get(QWidget, "Calibration")
        grid = calibration.layout()
        assert isinstance(grid, QGridLayout)
        scroll = self._get(QWidget, "scrollArea")
        grid.removeWidget(scroll)
        self.calibration_message = QLabel(CALIBRATION_UNAVAILABLE)
        self.calibration_message.setWordWrap(True)
        self.calibration_message.setStyleSheet("color: rgb(252,190,3);")
        grid.addWidget(self.calibration_message, 0, 0)
        grid.addWidget(scroll, 1, 0)
        scroll.setEnabled(False)
        scroll.setToolTip(CALIBRATION_UNAVAILABLE)
        connect = self._get(QPushButton, "qbutton_connect_to_mm")
        connect.setEnabled(False)
        connect.setToolTip(CALIBRATION_UNAVAILABLE)
        status = self._get(QLineEdit, "le_mm_status")
        status.setText("Not available")
        status.setToolTip(CALIBRATION_UNAVAILABLE)
        status.setStyleSheet("border: 1px solid rgb(200,0,0); color: rgb(200,0,0);")
        for name in ("qbutton_push_note", "le_notes_field"):
            widget = self._get(QWidget, name)
            widget.setEnabled(False)
            widget.setToolTip("Notes are written to a calibration's metadata. " + CALIBRATION_UNAVAILABLE)

    def _init_logging(self) -> None:
        handler = QtLogger(self._get(QPlainTextEdit, "te_log"))
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        plugin_logger = logging.getLogger(PLUGIN_LOGGER)
        plugin_logger.addHandler(handler)
        plugin_logger.setLevel(logging.INFO)
        self.destroyed.connect(lambda: plugin_logger.removeHandler(handler))
        self._get(QComboBox, "cb_loglevel").currentIndexChanged.connect(self.enter_log_level)

    def enter_log_level(self) -> None:
        """Show info, or debug, records in the log box."""
        level = logging.INFO if self._get(QComboBox, "cb_loglevel").currentIndex() == 0 else logging.DEBUG
        logging.getLogger(PLUGIN_LOGGER).setLevel(level)

    # --- layers and the overlay --------------------------------------------------------------------------------------

    def _add_or_update_image_layer(
        self, image: np.ndarray | da.Array, name: str, *, cmap: str = "gray", scale: tuple[float, ...] = (1,) * 5
    ) -> None:
        """Add the image layer ``name``, or replace its data and move it to the top; ``cmap="rgb"`` for RGB."""
        if image.shape[0] == 1:
            image = image[0]
            scale = scale[1:]
        scale = tuple(scale[-image.ndim :])
        if name in self.viewer.layers:
            self.viewer.layers[name].data = image
            logger.debug("Moving layer %s to the top.", name)
            self.viewer.layers.move(self.viewer.layers.index(name), -1)
        elif cmap == "rgb":
            self.viewer.add_image(image, name=name, rgb=True, scale=scale, cache=False)
        else:
            self.viewer.add_image(image, name=name, colormap=cmap, scale=scale, cache=False)

    def handle_layers_updated(self, event: Event) -> None:
        """Color a new top ``Orientation*`` layer in HSV, and overlay it with its ``Retardance*`` layer.

        The overlay is ``Birefringence Overlay*``, with the orientation layer's suffix.
        """
        layers = event.source
        if not len(layers) or not layers[-1].name.startswith("Orientation"):
            return
        orientation_name = layers[-1].name
        suffix = orientation_name.removeprefix("Orientation")
        retardance_name = "Retardance" + suffix
        if retardance_name in layers:
            logger.info("Detected updated birefringence layers: '%s', '%s'", retardance_name, orientation_name)
            self._draw_bire_overlay(
                retardance_name, orientation_name, "Birefringence Overlay" + suffix, scale=tuple(layers[-1].scale)
            )
        logger.info("Detected orientation layer in updated layer list. Setting its colormap to HSV.")
        self.viewer.layers[orientation_name].colormap = "hsv"

    def _layer_data(self, name: str) -> da.Array:
        data = self.viewer.layers[name].data
        if isinstance(data, da.Array):
            # the ome-zarr reader reads HCS plates as a nested dask graph ('get_tile' or 'get_field' in its keys),
            # which stays a dask array after compute()
            if any("get_" in str(key) for key in data.dask):
                data = data.compute()
            return data
        chunks = (data.ndim - 2) * (1,) + tuple(data.shape[-2:])
        return da.from_array(data, chunks=chunks)

    def _draw_bire_overlay(
        self, retardance_name: str, orientation_name: str, overlay_name: str, *, scale: tuple[float, ...]
    ) -> None:
        self.overlay_scale = scale
        self.overlay_name = overlay_name
        self.overlay_retardance = self._layer_data(retardance_name)
        self.overlay_orientation = self._layer_data(orientation_name)
        self.update_overlay_dask_array()

    def update_overlay_dask_array(self) -> None:
        """Recompute the birefringence overlay lazily, with the current retardance maximum."""
        if self.overlay_retardance is None or self.overlay_orientation is None:
            return
        retardance = self.overlay_retardance
        chunks = (3,) + (retardance.ndim - 2) * (1,) + tuple(retardance.shape[-2:])
        overlay = da.map_blocks(
            ret_ori_overlay,
            da.stack((retardance, self.overlay_orientation)),
            ret_max=self.ret_max,
            chunks=chunks,
            dtype=np.float32,
            drop_axis=0,
            new_axis=0,
        )
        overlay = da.moveaxis(overlay, source=0, destination=-1)
        self._add_or_update_image_layer(overlay, self.overlay_name, cmap="rgb", scale=self.overlay_scale)

    def handle_ret_max_slider_move(self, value: int) -> None:
        """Take the Overlay Retardance Maximum, and redraw the overlay."""
        self.ret_max = value
        self.update_overlay_dask_array()
