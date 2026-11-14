"""The Reconstruction tab: choose an input store and its positions, build or load models, run them, show the results.

A model is a version-4 configuration shown as a form (:mod:`waveorder.plugin.forms`). **New** starts one from the
``wo config new`` template of each checked contrast; **Load** reads ``.yml`` files, migrating 3.x files as
``wo config migrate`` does. **RUN** validates every model, writes each configuration next to its output, and runs
``wo reconstruct`` on it as a job (:mod:`waveorder.plugin.job_manager`), so every reconstruction can be repeated from
the command line. A job's output lines show in the Reconstruction Queue; a job that fails keeps its entry and reports
its error. A store still being acquired can be reconstructed on the fly, a few time points at a time.
"""

import datetime
import functools
import json
import logging
import subprocess
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml
from iohub import open_ome_zarr
from iohub.ngff import Plate, Position
from magicgui import widgets
from pydantic import ValidationError
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from waveorder.config import V4Config, dump, dump_yaml, load
from waveorder.config.loader import parse
from waveorder.config.migrate import migrate
from waveorder.errors import Rejected
from waveorder.plugin import forms
from waveorder.plugin.job_manager import WO, JobManager, shown
from waveorder.plugin.qt_parts import CollapsibleBox, DropButton, DropWidget, ScrollableLabel


class Viewer(Protocol):
    """What the tab uses of a napari viewer."""

    def open(self, path: str, *, plugin: str) -> object:
        """Open ``path`` with the reader of ``plugin``."""
        ...


logger = logging.getLogger(__name__)

_validate_alert = "⚠️"
_validate_ok = "✅"
_green_dot = "🟢"
_red_dot = "🔴"
_info_icon = "ℹ️"  # noqa: RUF001 (wo-4.0's info icon)

MODES = ("birefringence", "phase", "fluorescence")
"""The mode checkboxes, as wo-4.0 showed them; a model reconstructs the contrast of its mode."""

NOT_IN_THIS_VERSION = "birefringence is not in this version yet; use a 3.x release"
"""Why the birefringence checkbox and the LC Calibration tab are disabled."""

POLL_INTERVAL_MS = 10_000
"""How often an acquisition in progress is checked for new time points."""

LAST_PATHS: dict[str, str] = {}
"""The tab's paths, kept when the dock widget is closed and opened again."""

_ERROR_LINES = 20
"""How many of a failed job's last lines its error message shows."""


@functools.cache
def template(contrast: str) -> dict[str, Any]:
    """Return the configuration ``wo config new --contrast <contrast>`` writes, every default filled in.

    Raises
    ------
    subprocess.CalledProcessError
        If ``wo config new`` fails.
    """
    written = subprocess.run(
        [*WO, "config", "new", "--contrast", contrast], capture_output=True, text=True, check=True, timeout=120
    )
    return dump(load(parse(written.stdout)))


def with_times(config: V4Config, times: list[int]) -> V4Config:
    """Return ``config`` reconstructing the time points ``times``."""
    return config.model_copy(update={"data": config.data.model_copy(update={"time_indices": times})})


@functools.cache
def job_manager() -> JobManager:
    """Return the jobs of every Reconstruction tab: they keep running when the dock widget is closed."""
    return JobManager()


def read_attributes(path: str | Path) -> dict[str, Any]:
    """Return the root attributes of an OME-Zarr store, also while an acquisition writes to it; ``{}`` if unreadable."""
    try:
        with open_ome_zarr(path, mode="r") as dataset:
            return dict(dataset.zattrs)
    except (OSError, ValueError, KeyError, RuntimeError):
        pass
    for name, key in ((".zattrs", None), ("zarr.json", "attributes")):
        try:
            document = json.loads((Path(path) / name).read_text())
        except (OSError, ValueError):
            continue
        return dict(document if key is None else document.get(key, {}))
    return {}


def acquisition_running(attributes: dict[str, Any]) -> bool:
    """Whether an acquisition still writes the store: its ``CurrentDimensions`` are not its ``FinalDimensions``."""
    current, final = attributes.get("CurrentDimensions"), attributes.get("FinalDimensions")
    return current is not None and final is not None and current != final


@dataclass
class ModelEntry:
    """A model of the tab: its form and the widgets around it."""

    uid: str
    title: str
    mode: str
    form: widgets.Container
    box: CollapsibleBox
    output_line: widgets.LineEdit
    output_button: widgets.PushButton
    show: widgets.CheckBox
    widget: QWidget
    poll_data: bool = False


@dataclass
class JobEntry:
    """A job of the Reconstruction Queue."""

    uid: str
    title: str
    output: Path
    show: bool
    widget: QWidget
    info: ScrollableLabel
    cancel: widgets.PushButton
    lines: list[str] = field(default_factory=list)


@dataclass
class _Poll:
    path: str
    last: int
    timer: QTimer
    row: QWidget
    label: QLabel
    models: list[ModelEntry]
    configs: dict[str, V4Config]
    inputs: list[str]


class ReconTab(QWidget):
    """The Reconstruction tab of the dock widget, or the window of ``wo interactive`` (``stand_alone``).

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.
    stand_alone : bool
        Whether messages show in dialogs (no napari viewer) rather than as napari notifications.
    """

    def __init__(self, parent: QWidget | None = None, *, stand_alone: bool = False) -> None:
        super().__init__(parent)
        self.stand_alone = stand_alone
        self.viewer: Viewer | None = None
        cwd = str(Path.cwd())
        self.input_directory = LAST_PATHS.get("input_directory", cwd)
        self.output_directory = LAST_PATHS.get("output_directory", cwd)
        self.model_directory = LAST_PATHS.get("model_directory", cwd)
        self.input_channel_names: list[str] = []
        self.input_positions: list[str] = []
        self.poll_data = False
        self.models: list[ModelEntry] = []
        self.jobs: dict[str, JobEntry] = {}
        self.polls: dict[str, _Poll] = {}
        self._counts: Counter[str] = Counter()
        self._highlighted: list[tuple[widgets.Widget, str | None]] = []

        self.recon_tab_mainScrollArea = QScrollArea()
        self.recon_tab_mainScrollArea.setWidgetResizable(True)
        self.recon_tab_widget = QWidget()
        self.recon_tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.recon_tab_layout = QVBoxLayout()
        self.recon_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.recon_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.recon_tab_layout.setSpacing(0)
        self.recon_tab_widget.setLayout(self.recon_tab_layout)
        self.recon_tab_mainScrollArea.setWidget(self.recon_tab_widget)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.recon_tab_mainScrollArea)

        self.recon_tab_layout.addWidget(self._data_group())
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Vertical)
        splitter.addWidget(self._models_group())
        splitter.addWidget(self._queue_group())
        splitter.setSizes([600, 200])
        handle = splitter.handle(1)
        if handle is not None:
            handle.setStyleSheet("background: 1px rgb(128,128,128);")
        splitter.setStyleSheet("QSplitter::handle:pressed {background-color: #ca5;}")
        self.recon_tab_layout.addWidget(splitter)

        jobs = job_manager()
        jobs.output.connect(self._job_output)
        jobs.finished.connect(self._job_finished)
        self.job_manager = jobs

    # --- layout ------------------------------------------------------------------------------------------------------

    @staticmethod
    def _row(*parts: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        row.setLayout(layout)
        for part in parts:
            layout.addWidget(part)
        return row

    def _data_group(self) -> QGroupBox:
        group = QGroupBox("Data")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(0)
        group.setLayout(layout)

        self.data_input_Label = widgets.Label(value="Input Store")
        self.data_input_LineEdit = widgets.LineEdit(value=self.input_directory)
        self.data_input_PushButton = widgets.PushButton(label="Browse")
        self.data_input_PushButton.clicked.connect(self.browse_dir_path_input)
        self.data_input_LineEdit.changed.connect(self.read_and_set_input_path_on_validation)

        self.data_output_Label = widgets.Label(value="Output Directory")
        self.data_output_LineEdit = widgets.LineEdit(value=self.output_directory)
        self.data_output_PushButton = widgets.PushButton(label="Browse")
        self.data_output_PushButton.clicked.connect(self.browse_dir_path_output)
        self.data_output_LineEdit.changed.connect(self.read_and_set_out_path_on_validation)

        self.data_positions_Label = widgets.Label(value="Positions")
        self.data_positions_Select = widgets.Select(choices=[], value=[])
        self.data_positions_Select.tooltip = "The positions to reconstruct; all are selected when a store is chosen"
        self.data_positions_Select.native.setMaximumHeight(90)
        for label in (self.data_input_Label, self.data_output_Label, self.data_positions_Label):
            label.native.setMinimumWidth(115)

        layout.addWidget(
            self._row(self.data_input_Label.native, self.data_input_LineEdit.native, self.data_input_PushButton.native)
        )
        layout.addWidget(
            self._row(
                self.data_output_Label.native, self.data_output_LineEdit.native, self.data_output_PushButton.native
            )
        )
        self.data_positions_widget = self._row(self.data_positions_Label.native, self.data_positions_Select.native)
        self.data_positions_widget.setVisible(False)
        layout.addWidget(self.data_positions_widget)
        return group

    def _models_group(self) -> QGroupBox:
        group = QGroupBox("Models")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(0)
        group.setLayout(layout)

        self.modes_selected: dict[str, dict[str, widgets.CheckBox]] = {
            mode: {"Checkbox": widgets.Checkbox(name=mode, label=mode)} for mode in MODES
        }
        birefringence = self.modes_selected["birefringence"]["Checkbox"]
        birefringence.enabled = False
        birefringence.tooltip = NOT_IN_THIS_VERSION
        self.models_new_PushButton = widgets.PushButton(label="New")
        self.models_new_PushButton.clicked.connect(self.build_acq_contols)
        self.models_load_PushButton = DropButton("Load", self.open_model_files)
        self.models_load_PushButton.clicked.connect(self.browse_dir_path_model)
        self.models_clear_PushButton = widgets.PushButton(label="Clear")
        self.models_clear_PushButton.clicked.connect(lambda: self.clear_all_models())
        layout.addWidget(
            self._row(
                *(checkbox["Checkbox"].native for checkbox in self.modes_selected.values()),
                self.models_new_PushButton.native,
                self.models_load_PushButton,
                self.models_clear_PushButton.native,
            )
        )

        self.models_scrollArea = QScrollArea()
        self.models_scrollArea.setWidgetResizable(True)
        self.models_container_widget = DropWidget(self.open_model_files)
        self.models_container_widget_layout = QVBoxLayout()
        self.models_container_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.models_container_widget_layout.setSpacing(2)
        self.models_container_widget_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.models_container_widget.setLayout(self.models_container_widget_layout)
        self.models_scrollArea.setWidget(self.models_container_widget)
        layout.addWidget(self.models_scrollArea)
        return group

    def _queue_group(self) -> QGroupBox:
        group = QGroupBox("Reconstruction Queue")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 10, 5, 5)
        layout.setSpacing(2)
        group.setLayout(layout)

        self.reconstruction_run_PushButton = widgets.PushButton(name="RUN Model")
        self.reconstruction_run_PushButton.native.setMinimumWidth(100)
        self.reconstruction_run_PushButton.clicked.connect(self.build_model_and_run)
        layout.addWidget(self.reconstruction_run_PushButton.native)

        tabs = QTabWidget()
        layout.addWidget(tabs)
        processing = QWidget()
        processing_layout = QVBoxLayout()
        processing_layout.setContentsMargins(5, 5, 5, 5)
        processing_layout.setSpacing(2)
        processing.setLayout(processing_layout)
        self.proc_table_QFormLayout = QFormLayout()
        self.proc_table_QFormLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        table = QWidget()
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.setLayout(self.proc_table_QFormLayout)
        processing_layout.addWidget(table)
        clear_results = widgets.PushButton(label="Clear Results")
        clear_results.clicked.connect(self.clear_results_table)
        processing_layout.addWidget(clear_results.native)

        on_the_fly = QWidget()
        on_the_fly_layout = QVBoxLayout()
        on_the_fly_layout.setContentsMargins(0, 0, 0, 0)
        on_the_fly_layout.setSpacing(0)
        on_the_fly.setLayout(on_the_fly_layout)
        self.proc_OTF_table_QFormLayout = QFormLayout()
        self.proc_OTF_table_QFormLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        otf_table = QWidget()
        otf_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        otf_table.setLayout(self.proc_OTF_table_QFormLayout)
        on_the_fly_layout.addWidget(otf_table)
        on_the_fly.setMaximumHeight(100)

        tabs.addTab(processing, "Processing")
        tabs.addTab(on_the_fly, "On-The-Fly")
        return group

    # --- messages and dialogs ----------------------------------------------------------------------------------------

    def set_viewer(self, viewer: Viewer) -> None:
        """Show results in ``viewer``."""
        self.viewer = viewer

    def show_dataset(self, data_path: str | Path) -> None:
        """Open a reconstruction in the viewer (with napari-ome-zarr)."""
        if self.viewer is None:
            return
        try:
            self.viewer.open(str(data_path), plugin="napari-ome-zarr")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.message_box(f"Cannot show {data_path}: {error}")

    def confirm_dialog(self, msg: str = "Confirm your selection ?") -> bool:
        """Ask the user to confirm ``msg``."""
        buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        answer = QMessageBox.question(self.recon_tab_widget, "Confirm", msg, buttons)
        return answer == QMessageBox.StandardButton.Yes

    def message_box(self, msg: str, *, error: bool = True) -> None:
        """Show ``msg``: as a napari notification, or in a dialog when the tab stands alone."""
        if not msg:
            return
        (logger.error if error else logger.info)(msg)
        if self.stand_alone:
            box = QMessageBox(
                QMessageBox.Icon.Warning if error else QMessageBox.Icon.Information,
                "Message",
                msg,
                QMessageBox.StandardButton.Ok,
                self.recon_tab_widget,
            )
            box.exec()
            return
        from napari.utils.notifications import show_error, show_info

        (show_error if error else show_info)(msg)

    def open_file_dialog(self, default_path: str, kind: str, file_filter: str = "All Files (*)") -> str | list[str]:
        """Return the directory (``dir``), file (``file``, ``save``) or files (``files``) the user chooses."""
        options = QFileDialog.Option.DontUseNativeDialog
        if kind == "dir":
            return QFileDialog.getExistingDirectory(None, "select a directory", default_path, options=options)
        if kind == "files":
            return QFileDialog.getOpenFileNames(None, "select file(s)", default_path, file_filter, options=options)[0]
        if kind == "save":
            return QFileDialog.getSaveFileName(None, "Choose a save name", default_path, file_filter, options=options)[
                0
            ]
        return QFileDialog.getOpenFileName(None, "select a file", default_path, file_filter, options=options)[0]

    def save_last_paths(self) -> None:
        """Keep the tab's paths for the next time the dock widget opens."""
        LAST_PATHS.update(
            input_directory=self.input_directory,
            output_directory=self.output_directory,
            model_directory=self.model_directory,
        )

    # --- data --------------------------------------------------------------------------------------------------------

    def browse_dir_path_input(self) -> None:
        """Choose the input store in a dialog; existing models are reset."""
        if self.models and not self.confirm_dialog("Changing Input Data will reset your models. Continue ?"):
            return
        self.clear_all_models(silent=True)
        result = self.open_file_dialog(self.input_directory, "dir", "ZARR Storage (*.zarr)")
        if result:
            self.data_input_LineEdit.value = str(result)

    def browse_dir_path_output(self) -> None:
        """Choose the output directory in a dialog."""
        result = self.open_file_dialog(self.output_directory, "dir")
        if not result:
            return
        if not Path(str(result)).exists():
            self.message_box("Output Directory path must exist !")
            return
        self.data_output_LineEdit.value = str(result)

    def validate_input_data(self, input_data_folder: str) -> tuple[bool, str]:
        """Read the channels, positions and scale of an OME-Zarr store, and whether it is still being acquired.

        Returns
        -------
        tuple of bool and str
            Whether the store can be read, and why not.
        """
        self.input_channel_names, self.input_positions = [], []
        self.data_input_Label.value = "Input Store"
        try:
            with open_ome_zarr(input_data_folder, mode="r") as dataset:
                if not isinstance(dataset, Plate | Position):
                    return False, f"{input_data_folder} is neither an HCS plate nor a position"
                self.input_channel_names = list(dataset.channel_names)
                positions = list(dataset.positions()) if isinstance(dataset, Plate) else [(".", dataset)]
                self.input_positions = [name for name, _ in positions if name != "."]
                first = positions[0][1] if positions else None
                scale = (
                    ""
                    if first is None
                    else ", ".join(f"{axis}={size}" for axis, size in zip(first.axis_names, first.scale, strict=True))
                )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            return False, f"{input_data_folder}: dataset does not appear to be a valid ome-zarr storage ({error})"
        self.data_input_Label.value = f"Input Store {_info_icon}"
        self.data_input_Label.tooltip = (
            "Channel Names:\n- " + "\n- ".join(self.input_channel_names) + f"\n\nScale: {scale}"
            f"\n\nFOV: {', '.join(self.input_positions or ['.'])}"
        )
        self.data_positions_Select.choices = self.input_positions
        self.data_positions_Select.value = self.input_positions
        self.data_positions_widget.setVisible(bool(self.input_positions))
        self.poll_data = acquisition_running(read_attributes(input_data_folder)) and self.confirm_dialog(
            "This seems like an in-process Acquisition. Would you like to process data on-the-fly ?"
        )
        return True, ""

    def read_and_set_input_path_on_validation(self) -> None:
        """Check the input store typed or chosen, and point the output directory next to it."""
        value = self.data_input_LineEdit.value
        if not value:
            self.data_input_LineEdit.value = self.input_directory
            self.message_box("Input data path cannot be empty")
            return
        if not Path(value).exists():
            self.data_input_LineEdit.value = self.input_directory
            self.message_box("Input data path must point to a valid location")
            return
        valid, message = self.validate_input_data(value)
        if not valid:
            self.data_input_LineEdit.value = self.input_directory
            self.message_box(message)
            return
        self.input_directory = value
        self.save_last_paths()
        self.data_output_LineEdit.value = str(Path(value).parent.absolute())

    def read_and_set_out_path_on_validation(self) -> None:
        """Check the output directory typed or chosen, and point every model's output into it."""
        value = self.data_output_LineEdit.value
        if not value:
            self.data_output_LineEdit.value = self.output_directory
            self.message_box("Output data path cannot be empty")
            return
        if not Path(value).exists():
            self.data_output_LineEdit.value = self.output_directory
            self.message_box("Output data path must point to a valid location")
            return
        self.output_directory = value
        self.save_last_paths()
        for entry in self.models:
            self._mark_output(entry)

    def input_arguments(self) -> list[str]:
        """Return the ``-i`` paths of a run: the store, or each selected position of a plate."""
        selected = list(self.data_positions_Select.value)
        if not self.input_positions or sorted(selected) == sorted(self.input_positions):
            return [self.input_directory]
        return [str(Path(self.input_directory) / name) for name in selected]

    # --- models ------------------------------------------------------------------------------------------------------

    def build_acq_contols(self) -> None:
        """Add a model for each checked contrast (**New**), from the last model of that contrast or its template."""
        checked = [mode for mode in MODES if self.modes_selected[mode]["Checkbox"].value]
        if "birefringence" in checked:
            self.message_box(NOT_IN_THIS_VERSION)
            return
        if not checked:
            self.message_box("Select phase or fluorescence, then click New")
            return
        for mode in checked:
            values = self._previous_values(mode)
            if values is None:
                try:
                    values = template(mode)
                except subprocess.CalledProcessError as error:
                    self.message_box(f"wo config new --contrast {mode} failed:\n{error.stderr}")
                    return
                except (subprocess.TimeoutExpired, Rejected) as error:
                    self.message_box(f"Cannot start a {mode} model: {error}")
                    return
            self.add_model(mode, values)

    def _previous_values(self, mode: str) -> dict[str, Any] | None:
        for entry in reversed(self.models):
            if entry.mode == mode:
                config, _ = self.validated(entry)
                return None if config is None else dump(config)
        return None

    def browse_dir_path_model(self) -> None:
        """Choose configuration files in a dialog and load them (**Load**)."""
        results = self.open_file_dialog(self.model_directory, "files", "YAML Files (*.yml *.yaml)")
        if not results:
            return
        paths = [results] if isinstance(results, str) else results
        self.model_directory = str(Path(paths[0]).parent.absolute())
        self.save_last_paths()
        self.open_model_files(paths)

    def open_model_files(self, paths: list[str]) -> list[ModelEntry]:
        """Add a model for each configuration file; a 3.x file is migrated as ``wo config migrate`` does."""
        added = []
        for path in paths:
            read = self._read_model_file(Path(path))
            if read is not None:
                added.append(self.add_model(*read))
        return added

    def _read_model_file(self, path: Path) -> tuple[str, dict[str, Any]] | None:
        try:
            raw = parse(path.read_text())
        except (OSError, yaml.YAMLError) as error:
            self.message_box(f"{path}: cannot be read: {error}")
            return None
        result = migrate(raw)
        if result.config is None:
            self.message_box(f"{path} cannot be loaded:\n" + "\n".join(str(item) for item in result.rejections))
            return None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = load(result.config)
        except Rejected as error:
            self.message_box(f"{path}:\n{error}")
            return None
        notes = [*result.notes, *(str(warning.message) for warning in caught)]
        if result.dialect != "4" or notes:
            migrated = f"{path}: migrated from a {result.dialect} configuration" if result.dialect != "4" else str(path)
            self.message_box("\n".join([migrated, *notes]), error=False)
        return config.imaging_system.contrast, dump(config)

    def add_model(self, mode: str, values: dict[str, Any]) -> ModelEntry:
        """Add the form of a configuration of ``mode`` to the Models list, with its output and buttons."""
        self._counts[mode] += 1
        number = f"{self._counts[mode]:02d}"
        title = f"{mode} - {number}"
        uid = str(uuid.uuid4())
        form = forms.build_form(uid, values)
        system = form["imaging_system"]
        assert isinstance(system, widgets.Container)
        system["contrast"].enabled = False  # the model's contrast is its mode

        stem = Path(self.input_directory).name.removesuffix(".zarr")
        short = "".join(part[:3].capitalize() for part in mode.split("-and-"))
        output_line = widgets.LineEdit(value=f"{stem}_{short}_{number}.zarr")
        output_button = widgets.PushButton(text="Output Data:")
        show = widgets.CheckBox(name="Show after Reconstruction", value=True)
        show.max_width = 200
        validate = widgets.PushButton(name="Validate")
        delete = widgets.PushButton(name="Delete Model")

        content = QVBoxLayout()
        content.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.addWidget(form.native)
        content.addWidget(self._row(output_button.native, output_line.native))
        content.addWidget(self._row(show.native, validate.native, delete.native))
        inner = QWidget()
        inner.setLayout(content)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumHeight(content.sizeHint().height() + 20)
        box_layout = QVBoxLayout()
        box_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        box_layout.addWidget(scroll)
        box = CollapsibleBox(title, expanded=not self.models)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box.set_content_layout(box_layout)

        widget = QWidget()
        widget_layout = QVBoxLayout()
        widget_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        widget_layout.addWidget(box)
        widget.setLayout(widget_layout)
        widget.setToolTip(title)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.models_container_widget_layout.addWidget(widget)

        entry = ModelEntry(uid, title, mode, form, box, output_line, output_button, show, widget, self.poll_data)
        output_button.clicked.connect(lambda: self._browse_model_output(entry))
        output_line.changed.connect(lambda: self._mark_output(entry))
        validate.clicked.connect(lambda: self.validate_model(entry))
        delete.clicked.connect(lambda: self.delete_model(entry))
        self.models.append(entry)
        self._mark_output(entry)
        self._update_run_button()
        return entry

    def output_path(self, entry: ModelEntry) -> Path:
        """Return where ``entry`` writes its reconstruction."""
        return Path(self.output_directory).absolute() / entry.output_line.value

    def _mark_output(self, entry: ModelEntry) -> None:
        exists = self.output_path(entry).exists()
        mark = f"{_validate_alert} " if exists else ""
        entry.output_button.text = f"{mark}Output Data:"
        entry.output_button.tooltip = f"{_validate_alert} Output file exists" if exists else ""
        entry.output_line.tooltip = entry.output_button.tooltip

    def _browse_model_output(self, entry: ModelEntry) -> None:
        result = self.open_file_dialog(self.output_directory, "save")
        if result:
            entry.output_line.value = Path(str(result)).name

    def _update_run_button(self) -> None:
        count = len(self.models)
        self.reconstruction_run_PushButton.text = f"RUN {count} Models" if count > 1 else "RUN Model"

    def validated(self, entry: ModelEntry) -> tuple[V4Config | None, list[str]]:
        """Return the configuration of ``entry``'s form, or ``None`` and its problems, each field framed in yellow."""
        try:
            mapping = forms.read_form(entry.form)
        except forms.FormError as error:
            self._highlight(entry, error.path.split("."), str(error))
            return None, [str(error)]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = V4Config.model_validate(mapping)
        except ValidationError as error:
            problems = []
            for line in error.errors():
                path = ".".join(str(part) for part in line["loc"])
                problems.append(f"{path}: {line['msg']}" if path else line["msg"])
                self._highlight(entry, line["loc"], line["msg"])
            return None, problems
        for warning in caught:
            logger.warning("%s: %s", entry.title, warning.message)
        return config, []

    def _highlight(self, entry: ModelEntry, location: Any, message: str) -> None:  # noqa: ANN401 (a pydantic location)
        widget = forms.locate(entry.form, location)
        if widget is not None:
            self._highlighted.append((widget, widget.tooltip))
            forms.highlight(widget, message)

    def _reset_highlights(self) -> None:
        for widget, tooltip in self._highlighted:
            widget.native.setStyleSheet("")
            widget.tooltip = tooltip
        self._highlighted.clear()

    def validate_model(self, entry: ModelEntry) -> V4Config | None:
        """Check ``entry``'s configuration (**Validate**): its title gets ✅, or ⚠️ and a message of its problems."""
        self._reset_highlights()
        config, problems = self.validated(entry)
        entry.box.set_title(f"{entry.title} {_validate_ok if config is not None else _validate_alert}")
        if config is None:
            self.message_box(f"{entry.title}:\n" + "\n".join(problems))
        return config

    def delete_model(self, entry: ModelEntry, *, silent: bool = False) -> None:
        """Remove ``entry`` (**Delete Model**), after the user confirms."""
        if not silent and not self.confirm_dialog():
            return
        entry.widget.setParent(None)
        self.models.remove(entry)
        self._update_run_button()

    def clear_all_models(self, *, silent: bool = False) -> None:
        """Remove every model (**Clear**), after the user confirms."""
        if not silent and not self.confirm_dialog():
            return
        for entry in list(self.models):
            self.delete_model(entry, silent=True)
        self._counts.clear()

    # --- runs --------------------------------------------------------------------------------------------------------

    def build_model_and_run(self) -> None:
        """Validate every model, then write its configuration and run ``wo reconstruct`` on it (**RUN**)."""
        if not self.models:
            self.message_box("Please create a processing model first !")
            return
        self._reset_highlights()
        configs: dict[str, V4Config] = {}
        errors = []
        for entry in self.models:
            config, problems = self.validated(entry)
            entry.box.set_title(f"{entry.title} {_validate_ok if config is not None else _validate_alert}")
            if config is None:
                errors.append(f"{entry.title}\n" + "\n".join(problems))
            else:
                configs[entry.uid] = config
        if errors:
            self.message_box("\n\n".join(errors))
            return
        inputs = self.input_arguments()
        if not inputs:
            self.message_box("Select at least one position to reconstruct")
            return
        if any(entry.poll_data for entry in self.models):
            self._start_polling(list(self.models), configs, inputs)
            return
        stamp = _stamp()
        for index, entry in enumerate(self.models, start=1):
            self.submit(entry, configs[entry.uid], inputs=inputs, suffix=f"{stamp}-{index:02d}")

    def submit(self, entry: ModelEntry, config: V4Config, *, inputs: list[str], suffix: str) -> JobEntry:
        """Write ``config`` next to ``entry``'s output and queue its ``wo reconstruct`` job.

        The configuration is ``<output directory>/<mode>-<suffix>.yml``.
        """
        output = self.output_path(entry)
        config_path = output.parent / f"{entry.mode}-{suffix}.yml"
        config_path.write_text(dump_yaml(config))
        arguments = ["reconstruct", "-i", *inputs, "-c", str(config_path), "-o", str(output)]
        job = self._add_job_entry(f"{entry.title}: ({suffix})", output, show=entry.show.value)
        job.info.append(f"$ {shown(arguments)}")
        self.job_manager.run_job(job.uid, arguments)
        return job

    def _add_job_entry(self, title: str, output: Path, *, show: bool) -> JobEntry:
        uid = str(uuid.uuid4())
        cancel = widgets.PushButton(name="JobID", label="Cancel Job", enabled=True)
        info = ScrollableLabel(f"Updating {title}: Please wait...")
        info.setFixedHeight(300)
        content = QVBoxLayout()
        content.setAlignment(Qt.AlignmentFlag.AlignTop)
        content.addWidget(cancel.native)
        content.addWidget(info)
        box = CollapsibleBox(title)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box.set_content_layout(content)
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(box)
        widget.setLayout(layout)
        widget.setToolTip(title)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.proc_table_QFormLayout.insertRow(0, widget)  # the latest job on top
        job = JobEntry(uid, title, output, show, widget, info, cancel)
        cancel.clicked.connect(lambda: self.cancel_job(job))
        self.jobs[uid] = job
        return job

    def cancel_job(self, job: JobEntry) -> None:
        """Stop ``job`` (**Cancel Job**), after the user confirms."""
        if not self.confirm_dialog():
            return
        job.cancel.enabled = False
        job.info.append("Cancelled by User")
        self.job_manager.cancel_job(job.uid)

    def _job_output(self, uid: str, line: str) -> None:
        job = self.jobs.get(uid)
        if job is not None:
            job.lines.append(line)
            job.info.append(line)

    def _job_finished(self, uid: str, code: int) -> None:
        job = self.jobs.get(uid)
        if job is None:
            return
        job.cancel.enabled = False
        if code == 0:
            self._remove_job(job)
            if job.show and job.output.exists():
                self.show_dataset(job.output)
            return
        job.info.append(f"Process ended with return code {code}")
        if "Cancelled by User" not in job.info.text():
            last = "\n".join(job.lines[-_ERROR_LINES:])
            self.message_box(f"{job.title} failed with return code {code}:\n{last}")

    def _remove_job(self, job: JobEntry) -> None:
        self.jobs.pop(job.uid, None)
        self.proc_table_QFormLayout.removeRow(job.widget)

    def clear_results_table(self) -> None:
        """Remove every entry of the Processing table (**Clear Results**), after the user confirms."""
        if self.proc_table_QFormLayout.rowCount() < 1:
            self.message_box("There are no processing results to clear !")
            return
        if self.confirm_dialog():
            for job in list(self.jobs.values()):
                self._remove_job(job)
            while self.proc_table_QFormLayout.rowCount():
                self.proc_table_QFormLayout.removeRow(0)

    # --- on the fly --------------------------------------------------------------------------------------------------

    def _start_polling(self, models: list[ModelEntry], configs: dict[str, V4Config], inputs: list[str]) -> None:
        path = self.input_directory
        if path in self.polls:
            self.message_box(f"{path} is already reconstructed on the fly")
            return
        label = QLabel(f"{path} {_green_dot}")
        label.setWordWrap(True)
        stop = QPushButton("Stop")
        row = self._row(label, stop)
        row.setObjectName(path)
        self.proc_OTF_table_QFormLayout.insertRow(0, row)
        timer = QTimer(self)
        poll = _Poll(path, 0, timer, row, label, models, configs, inputs)
        self.polls[path] = poll
        stop.clicked.connect(lambda: self.stop_polling(path))
        timer.timeout.connect(lambda: self.poll(path))
        timer.start(POLL_INTERVAL_MS)
        self.poll(path)

    def poll(self, path: str) -> None:
        """Reconstruct the time points an acquisition has finished since the last check, then stop once it is done."""
        state = self.polls.get(path)
        if state is None:
            return
        attributes = read_attributes(path)
        current = attributes.get("CurrentDimensions", {})
        final = attributes.get("FinalDimensions", {})
        done = not acquisition_running(attributes)
        # the time point being acquired is incomplete until the next one starts or the acquisition ends
        end = int(final.get("time", current.get("time", 0))) if done else int(current.get("time", 0)) - 1
        if end > state.last:
            times = list(range(state.last, end))
            stamp = _stamp()
            for index, entry in enumerate(state.models, start=1):
                config = state.configs[entry.uid]
                self.submit(entry, with_times(config, times), inputs=state.inputs, suffix=f"{stamp}-{index:02d}")
            state.last = end
        if done:
            self.stop_polling(path)

    def stop_polling(self, path: str) -> None:
        """Stop reconstructing ``path`` on the fly (**Stop**)."""
        state = self.polls.pop(path, None)
        if state is None:
            return
        state.timer.stop()
        state.label.setText(f"{path} {_red_dot}")
        self.proc_OTF_table_QFormLayout.removeRow(state.row)


def _stamp() -> str:
    """Return a time stamp for configuration file names, to the millisecond (``2026_11_13_17_16_00_123``)."""
    now = datetime.datetime.now(tz=datetime.UTC)
    return now.strftime("%Y_%m_%d_%H_%M_%S_") + now.strftime("%f")[:3]
