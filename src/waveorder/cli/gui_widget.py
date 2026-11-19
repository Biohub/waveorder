"""``wo interactive``: the Reconstruction tab of the napari plugin in its own window."""

import sys

from waveorder.errors import Rejected

PLUGIN_NAME = "waveorder: Wave-optical simulation and reconstruction"
PLUGIN_ICON = "🔬"


# the docstring is the command's help, where Rich would read an unescaped [visual] as markup
def gui() -> None:
    r"""Open the reconstruction GUI in its own window, without napari's viewer.

    Build or load models, run them as wo reconstruct jobs and follow their progress. Needs the napari plugin's
    packages: pip install waveorder\[visual].
    """
    try:
        from qtpy.QtWidgets import QApplication, QStyle

        from waveorder.plugin.tab_recon import ReconTab
    except ImportError as error:
        raise Rejected.one(
            f"needs the napari plugin's packages ({error.name}): pip install waveorder[visual]",
            location="wo interactive",
        ) from None

    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setStyle("Fusion")
    window = ReconTab(stand_alone=True)
    window.setWindowTitle(f"{PLUGIN_ICON} {PLUGIN_NAME} {PLUGIN_ICON}")
    style = app.style()
    if style is not None:
        window.setWindowIcon(style.standardIcon(QStyle.StandardPixmap.SP_TitleBarMenuButton))
    window.show()
    raise SystemExit(app.exec())
