"""Fixtures of the napari plugin's tests, which need napari and pytest-qt (``pip install waveorder[visual]``)."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _fresh_tab_state(monkeypatch, tmp_path):
    """Each test starts in its own directory, with no paths kept from a previous tab."""
    monkeypatch.chdir(tmp_path)
    try:
        from waveorder.plugin import tab_recon
    except ImportError:  # napari or Qt is not installed: the plugin's tests skip
        return
    monkeypatch.setattr(tab_recon, "LAST_PATHS", {})


@pytest.fixture
def messages(monkeypatch):
    """The messages the Reconstruction tab shows, as ``(text, error)``; confirmations answer yes."""
    from waveorder.plugin import tab_recon

    shown: list[tuple[str, bool]] = []
    monkeypatch.setattr(tab_recon.ReconTab, "message_box", lambda _self, msg, *, error=True: shown.append((msg, error)))
    monkeypatch.setattr(tab_recon.ReconTab, "confirm_dialog", lambda _self, _msg="": True)
    return shown


@pytest.fixture
def main_widget(qtbot, messages):
    """The dock widget on a viewer model: offscreen Qt has no OpenGL to draw a napari canvas's layers."""
    del messages
    from napari.components import ViewerModel

    from waveorder.plugin.main_widget import MainWidget

    widget = MainWidget(ViewerModel())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def recon_tab(main_widget):
    """The Reconstruction tab of the dock widget."""
    return main_widget.reconstruction_tab


def set_input(tab, store: Path) -> None:
    """Choose ``store`` as the input, as typing it in the Input Store line does."""
    tab.data_input_LineEdit.value = str(store)
