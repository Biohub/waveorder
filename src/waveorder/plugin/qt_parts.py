"""Small Qt widgets of the Reconstruction tab: a collapsible box, a scrollable label and drop targets for files."""

from collections.abc import Callable

from qtpy.QtCore import QAbstractAnimation, QParallelAnimationGroup, QPropertyAnimation, Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_DURATION_MS = 500


class CollapsibleBox(QWidget):
    """A box under a title button that expands and collapses its content.

    Parameters
    ----------
    title : str
        The text of the title button.
    parent : QWidget, optional
        The parent widget.
    expanded : bool
        Whether the box starts expanded.
    """

    def __init__(self, title: str = "", parent: QWidget | None = None, *, expanded: bool = False) -> None:
        super().__init__(parent)
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QParallelAnimationGroup(self)
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

        self._animations = [
            QPropertyAnimation(self, b"minimumHeight"),
            QPropertyAnimation(self, b"maximumHeight"),
            QPropertyAnimation(self.content_area, b"maximumHeight"),
        ]
        for animation in self._animations:
            self.toggle_animation.addAnimation(animation)
        self._set_heights(layout.sizeHint().height())
        if expanded:
            self.on_pressed()

    @property
    def title(self) -> str:
        """The text of the title button."""
        return self.toggle_button.text()

    def set_title(self, title: str) -> None:
        """Show ``title`` on the title button."""
        self.toggle_button.setText(title)

    def on_pressed(self) -> None:
        """Expand a collapsed box, or collapse an expanded one."""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if not checked else Qt.ArrowType.RightArrow)
        self.toggle_animation.setDirection(
            QAbstractAnimation.Direction.Forward if not checked else QAbstractAnimation.Direction.Backward
        )
        self.toggle_animation.start()

    def set_content_layout(self, layout: QLayout) -> None:
        """Show ``layout`` in the box."""
        self.content_area.setLayout(layout)
        content_height = layout.sizeHint().height()
        self._set_heights(content_height)
        content = self._animations[-1]
        content.setStartValue(0)
        content.setEndValue(content_height)

    def _set_heights(self, content_height: int) -> None:
        collapsed = self.sizeHint().height() - self.content_area.maximumHeight()
        for animation in self._animations:
            animation.setDuration(_DURATION_MS)
            animation.setStartValue(collapsed)
            animation.setEndValue(collapsed + content_height)


class ScrollableLabel(QScrollArea):
    """A scrollable, word-wrapped label: the text of a job."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setText(text)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setWidget(container)
        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignTop)

    def set_text(self, text: str) -> None:
        """Replace the text."""
        self.label.setText(text)

    def text(self) -> str:
        """Return the text."""
        return self.label.text()

    def append(self, line: str) -> None:
        """Add ``line`` at the end, and scroll to it."""
        self.label.setText(f"{self.label.text()}\n{line}")
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())


def _dropped_files(event: QDropEvent | None) -> list[str]:
    mime = None if event is None else event.mimeData()
    return [] if mime is None else [url.toLocalFile() for url in mime.urls()]


def _accept_files(event: QDragEnterEvent | None) -> None:
    mime = None if event is None else event.mimeData()
    if event is not None and mime is not None and mime.hasUrls():
        event.acceptProposedAction()


class DropButton(QPushButton):
    """A push button that also takes configuration files dropped on it."""

    def __init__(self, text: str, on_files: Callable[[list[str]], object], parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self._on_files = on_files

    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:  # noqa: N802 (Qt's names)
        """Accept a drag that carries files."""
        _accept_files(a0)

    def dropEvent(self, a0: QDropEvent | None) -> None:  # noqa: N802 (Qt's names)
        """Load the dropped files."""
        self._on_files(_dropped_files(a0))


class DropWidget(QWidget):
    """A container that takes configuration files dropped on it."""

    def __init__(self, on_files: Callable[[list[str]], object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._on_files = on_files

    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:  # noqa: N802 (Qt's names)
        """Accept a drag that carries files."""
        _accept_files(a0)

    def dropEvent(self, a0: QDropEvent | None) -> None:  # noqa: N802 (Qt's names)
        """Load the dropped files."""
        self._on_files(_dropped_files(a0))
