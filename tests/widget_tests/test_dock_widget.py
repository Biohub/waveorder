import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from qtpy.QtWidgets import QComboBox, QPushButton, QTabWidget, QWidget

from waveorder.plugin.main_widget import CALIBRATION_UNAVAILABLE, MainWidget
from waveorder.plugin.tab_recon import NOT_IN_THIS_VERSION


def test_dock_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(MainWidget(viewer))
    assert "waveorder" in next(iter(viewer.window.dock_widgets))


def test_tabs_and_the_disabled_calibration(main_widget):
    tabs = main_widget.ui.findChild(QTabWidget, "tabWidget")
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "LC Calibration",
        "Reconstruction",
        "Visualization",
    ]
    assert tabs.currentWidget() is main_widget.reconstruction_tab
    connect = main_widget.ui.findChild(QPushButton, "qbutton_connect_to_mm")
    assert not connect.isEnabled()
    assert connect.toolTip() == CALIBRATION_UNAVAILABLE
    assert not main_widget.ui.findChild(QWidget, "cb_calib_mode").isEnabled()
    assert not main_widget.ui.findChild(QWidget, "tab_plot").isEnabled()
    # the reconstructions log to the Advanced page of that tab, which stays reachable
    for name in ("tab_advanced", "te_log", "cb_loglevel"):
        assert main_widget.ui.findChild(QWidget, name).isEnabled(), name
    inner = main_widget.ui.findChild(QTabWidget, "tabWidget_2")
    assert inner.isEnabled()
    assert inner.isTabEnabled(inner.indexOf(main_widget.ui.findChild(QWidget, "tab_advanced")))
    assert NOT_IN_THIS_VERSION in main_widget.calibration_message.text()
    modes = main_widget.ui.findChild(QComboBox, "cb_calib_mode")
    assert [modes.itemText(index) for index in range(modes.count())] == ["MM-Retardance", "MM-Voltage", "DAC"]


def test_an_orientation_layer_gets_hsv_and_an_overlay_with_its_retardance(main_widget):
    import math

    import torch

    viewer = main_widget.viewer
    viewer.add_image(torch.full((2, 8, 8), 5.0).numpy(), name="Retardance")
    viewer.add_image(torch.full((2, 8, 8), math.pi / 2).numpy(), name="Orientation")
    assert viewer.layers["Orientation"].colormap.name == "hsv"
    overlay = viewer.layers["Birefringence Overlay"]
    assert overlay.rgb
    assert overlay.data.shape == (2, 8, 8, 3)
    assert overlay.data[0, 0, 0].compute().tolist() == pytest.approx([0, 1, 1])  # hue of 90 degrees: cyan
    main_widget.handle_ret_max_slider_move(10)
    assert main_widget.ret_max == 10
    assert viewer.layers["Birefringence Overlay"].data[0, 0, 0].compute().tolist() == pytest.approx([0, 1, 1])
