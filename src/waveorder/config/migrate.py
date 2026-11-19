"""Translate a waveorder 3.x configuration, or a version-4 file of the #592 preview, into the current schema.

:func:`migrate` works on the raw mapping (no 3.x models exist in WaveOrder 4): every 3.x key in :data:`TABLE` is
converted, dropped with a note, or rejected with its key. The table is frozen against the 3.x settings
(``waveorder/cli/settings.py`` and the ``waveorder/api`` settings it imports, commit 4ea8027), including their
defaults: a key a 3.x file leaves out takes its 3.x default, which the output then states explicitly, because 4.x
defaults differ (pupil edge, input channels, condenser aperture).

Rules:

- rejected: birefringence, tile-stitch settings, and 3.x settings 4.x does not execute when they hold anything but
  their 3.x default (optimizer stopping rules, the auto-regularization report and plot);
- dropped with a note: those optimizer settings at their 3.x default, and blocks 3.x itself ignored;
- pinned to 3.x behaviour where 4.x has a knob: ``detection.edge: sigmoid``, Richardson-Lucy ``start: ones``.

:func:`render` writes the result as YAML under a header comment listing the notes. :func:`table_help` is the help
text of ``wo config migrate``. This module imports neither pydantic models nor torch; check the result with
:func:`waveorder.config.load` and :func:`waveorder.config.build.build`, as ``wo config check`` does.
"""

import copy
import math
import re
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import yaml

type Dialect = Literal["3.x", "592", "4"]
"""What :func:`migrate` read: a 3.x file, a version-4 file of the #592 preview (``pixel_size.yx``), or a 4.x file."""


@dataclass(frozen=True)
class Rejection:
    """A setting that has no WaveOrder 4 translation.

    Parameters
    ----------
    key : str
        Dotted path of the setting in the input, e.g. ``"phase.apply_inverse.auto_regularization"``.
    message : str
        Why, and what to do instead.
    """

    key: str
    message: str

    def __str__(self) -> str:
        return f"{self.key}: {self.message}"


@dataclass(frozen=True)
class MigrationResult:
    """What :func:`migrate` made of a configuration.

    Parameters
    ----------
    dialect : {"3.x", "592", "4"}
        What the input was. For ``"4"`` the input is returned unchanged.
    config : dict or None
        The WaveOrder 4 mapping; ``None`` when anything was rejected.
    notes : tuple of str
        What changed in meaning: dropped settings, pinned 3.x behaviour, new checks.
    rejections : tuple of Rejection
        Settings with no translation, keyed by their path in the input.
    """

    dialect: Dialect
    config: dict[str, Any] | None
    notes: tuple[str, ...] = ()
    rejections: tuple[Rejection, ...] = ()


NO_DEFAULT = object()
"""The default of a 3.x key that has none (a block the file must name)."""


@dataclass(frozen=True)
class Row:
    """One 3.x key: its 3.x default and what it becomes.

    Parameters
    ----------
    key : str
        Dotted 3.x path; ``<contrast>`` stands for ``phase`` or ``fluorescence``.
    default : object
        The 3.x default, used when the file leaves the key out.
    becomes : str
        The 4.x key it is written to, or what happens to it.
    """

    key: str
    default: object
    becomes: str


_C = "<contrast>"
_TF = f"{_C}.transfer_function"
_AI = f"{_C}.apply_inverse"
_OPT = "optimization"

_EDGE_NOTE = (
    "imaging_system.detection.edge: sigmoid keeps the 3.x pupil; delete it for the 4.x default (ramp, antialiased)"
)
_PIXEL_NOTE = (
    "data.pixel_size is used as written, as in 3.x: wo rec warns where it differs from the scale of the input by "
    "more than 5%; pixel_size: auto reads the scale instead"
)
_APERTURES = ("detection.numerical_aperture", "illumination.numerical_aperture")
_START_NOTE = (
    "solver.reconstruction.richardson_lucy.start: ones keeps the 3.x first estimate; delete it for the 4.x default "
    "(the data mean)"
)
_AUTO_FIELDS = ("rule", "search_min", "search_max", "num_samples", "search_scale")
_BIREFRINGENCE_MESSAGE = "birefringence is not in waveorder 4; use waveorder 3.x"
_TILE_STITCH_MESSAGE = (
    "tile-stitch ({keys}) is not in waveorder 4; migrate the recon block on its own and run wo rec --tile Z,Y,X "
    "(the output becomes a plate)"
)
_METHODS = ("adam", "nadam", "lbfgs", "nelder_mead", "grid_search")
_STEPPED = _METHODS[:3]
"""The 3.x optimizers that step along the gradient: they read ``use_gradients`` and stop on ``convergence_tol``."""
_RL_FIELDS = ("iterations", "background", "back_projector")

TABLE: tuple[Row, ...] = (
    Row("input_channel_names", ["State0", "State1", "State2", "State3"], "data.input_channel_names (one channel)"),
    Row("time_indices", "all", "data.time_indices"),
    Row("reconstruction_dimension", 3, "object.dimension"),
    Row("device", None, "solver.device"),
    Row("birefringence", None, f"rejected: {_BIREFRINGENCE_MESSAGE}"),
    Row("tile, blend, recon", NO_DEFAULT, "rejected: tile-stitch settings; use wo rec --tile"),
    Row(f"{_TF}.yx_pixel_size", 0.1, "data.pixel_size.y and .x; warned about if 5% off the data's scale"),
    Row(f"{_TF}.z_pixel_size", 0.25, "data.pixel_size.z; warned about if 5% off the data's scale"),
    Row(f"{_TF}.z_padding", 0, "solver.padding.z"),
    Row(
        f"{_TF}.z_focus_offset",
        0,
        "imaging_system.detection.z_focus_offset for 2D; 0 with a note for 3D (3.x ignored it there, 4.x applies it)",
    ),
    Row(f"{_TF}.index_of_refraction_media", 1.3, "imaging_system.index_of_refraction_media"),
    Row(f"{_TF}.numerical_aperture_detection", 1.2, "imaging_system.detection.numerical_aperture"),
    Row(
        f"{_TF}.tilt_angle_zenith",
        0.0,
        "phase: imaging_system.illumination.tilt.zenith; fluorescence: dropped with a note (3.x did not read it)",
    ),
    Row(
        f"{_TF}.tilt_angle_azimuth",
        0.0,
        "phase: imaging_system.illumination.tilt.azimuth; fluorescence: dropped with a note (3.x did not read it)",
    ),
    Row("phase.transfer_function.wavelength_illumination", 0.532, "imaging_system.illumination.wavelength"),
    Row(
        "phase.transfer_function.numerical_aperture_illumination",
        0.9,
        "imaging_system.illumination.numerical_aperture",
    ),
    Row(
        "phase.transfer_function.invert_phase_contrast",
        default=False,
        becomes="imaging_system.illumination.invert_phase_contrast",
    ),
    Row("fluorescence.transfer_function.wavelength_emission", 0.532, "imaging_system.detection.wavelength"),
    Row(
        "fluorescence.transfer_function.confocal_pinhole_diameter",
        None,
        "imaging_system.detection.pinhole: {diameter_um: ...}; null is pinhole: none",
    ),
    Row("(pupil edge)", NO_DEFAULT, "pinned: imaging_system.detection.edge: sigmoid (4.x defaults to ramp)"),
    Row(
        f"{_AI}.reconstruction_algorithm",
        "Tikhonov",
        "Tikhonov: object.prior tikhonov, closed_form; TV: prior tv, admm; RL or RLGC: prior none, noise poisson, "
        "richardson_lucy",
    ),
    Row(f"{_AI}.regularization_strength", 0.001, "object.weight (dropped with a note for RL and RLGC)"),
    Row(f"{_AI}.TV_rho_strength", 0.001, "solver.reconstruction.admm.rho (TV only)"),
    Row(f"{_AI}.TV_iterations", 1, "solver.reconstruction.admm.iterations (TV only)"),
    Row(f"{_AI}.apodization_rolloff", 0.0, "solver.apodization_rolloff"),
    Row(
        f"{_AI}.auto_regularization",
        None,
        "object.weight: {auto: ...} (the same rule, sweep, report_path and plot_path) for 3D Tikhonov; dropped with "
        "a note where 3.x ignored it",
    ),
    Row("fluorescence.apply_inverse.rl", None, "solver.reconstruction.richardson_lucy (RL and RLGC only)"),
    Row("fluorescence.apply_inverse.rl.iterations", 25, "richardson_lucy.iterations"),
    Row("fluorescence.apply_inverse.rl.background", 0.0, "richardson_lucy.background"),
    Row(
        "fluorescence.apply_inverse.rl.stopping_tolerance",
        None,
        "richardson_lucy.tolerance with check_every: 1 (3.x checked every iteration)",
    ),
    Row("fluorescence.apply_inverse.rl.back_projector", "matched", "richardson_lucy.back_projector"),
    Row("fluorescence.apply_inverse.rl.bp_alpha", None, "richardson_lucy.bp_alpha"),
    Row("fluorescence.apply_inverse.rl.bp_beta", None, "richardson_lucy.bp_beta"),
    Row("(RL start)", NO_DEFAULT, "pinned: richardson_lucy.start: ones (4.x defaults to the data mean)"),
    Row(
        "value with lr > 0",
        NO_DEFAULT,
        "{nominal: init, tolerance: free}; lr goes to solver.calibration.advanced.learning_rates",
    ),
    Row(
        "numerical aperture with lr > 0",
        NO_DEFAULT,
        "also bounds 0.01 to index_of_refraction_media (grid_search sets its own); a 2D phase illumination aperture "
        "stays at most the detection aperture, so the pair is split there",
    ),
    Row(
        "tilt zenith with lr > 0",
        NO_DEFAULT,
        "also bounds init +- max_iterations * lr (adam, nadam; unlimited otherwise) cut to |zenith| < asin(detection "
        "NA / n): 4.x needs a range that keeps the unscattered light inside the detection aperture",
    ),
    Row(
        _OPT,
        None,
        "solver.calibration with score: sharpness and one search stage; dropped with a note when no value has lr > 0",
    ),
    Row(
        f"{_OPT}.method",
        "adam",
        "adam, nadam: gradient stage; lbfgs: lbfgs stage; nelder_mead: nelder_mead stage; grid_search: grid stage "
        "over init +- (grid_points // 2) * lr",
    ),
    Row(f"{_OPT}.max_iterations", 10, "the stage's iterations (grid_search ignores it, as in 3.x)"),
    Row(
        f"{_OPT}.convergence_tol",
        None,
        "adam, nadam: gradient.tolerance; lbfgs: lbfgs.tolerance; nelder_mead: nelder_mead.tolerance; rejected for "
        "grid_search",
    ),
    Row(
        f"{_OPT}.convergence_patience",
        5,
        "gradient.patience or lbfgs.patience beside convergence_tol; otherwise dropped with a note (3.x read it only "
        "there)",
    ),
    Row(
        f"{_OPT}.use_gradients",
        None,
        "false under adam, nadam, lbfgs: the free values at their init and no calibration (3.x took no step); "
        "otherwise dropped with a note (3.x resolved it to what the method does)",
    ),
    Row(
        f"{_OPT}.grid_points", 7, "grid.points for grid_search; otherwise dropped with a note at 7, rejected otherwise"
    ),
    Row(f"{_OPT}.loss", {"type": "midband_power"}, "solver.calibration.advanced.loss"),
    Row(f"{_OPT}.log_dir", None, "solver.calibration.advanced.log_dir (needs the tensorboard extra)"),
    Row(
        "(2D phase gradient fit)",
        NO_DEFAULT,
        "adam, nadam, lbfgs on 2D Tikhonov phase: solver.calibration.reconstruction: {closed_form: {coupling: "
        "per_field}}, the per-channel estimate 3.x scored them with; the final reconstruction stays joint",
    ),
    Row(
        "(phase fit pupil edge)",
        NO_DEFAULT,
        "any 3.x phase optimizer: solver.calibration.edge_steepness: 100, the soft sigmoid edge 3.x fitted on; the "
        "final reconstruction keeps imaging_system.detection.edge_steepness 1e4",
    ),
    Row("version: 4 with data.pixel_size.yx (#592)", NO_DEFAULT, "pixel_size.yx -> y and x, the rest above"),
)
"""Every 3.x key and what :func:`migrate` makes of it; :func:`table_help` renders it."""

_DEFAULTS: dict[str, object] = {row.key: row.default for row in TABLE}


def _block_keys() -> dict[str, set[str]]:
    """Return the keys each 3.x block may hold (``""`` is the top level), read off the dotted keys of :data:`TABLE`."""
    blocks: dict[str, set[str]] = {}
    for row in TABLE:
        if not re.fullmatch(r"[\w<>.]+", row.key):
            continue  # a rule, not a key
        for contrast in ("phase", "fluorescence"):
            parts = row.key.replace(_C, contrast).split(".")
            for depth, part in enumerate(parts):
                blocks.setdefault(".".join(parts[:depth]), set()).add(part)
    return blocks


_BLOCK_KEYS = _block_keys()
"""The keys each 3.x block may hold; anything else is not a 3.x setting."""
_TILE_STITCH_KEYS = ("tile", "blend", "recon")

# The 3.x optimizer settings (OptimizationSettings), which #592 copied into solver.calibration.advanced.
_OPTIMIZER_DEFAULTS: dict[str, object] = {
    name: _DEFAULTS[f"{_OPT}.{name}"]
    for name in ("method", "max_iterations", "convergence_tol", "convergence_patience", "use_gradients", "grid_points")
}

_GRADIENT_STEPS = 200
"""The ``max_iter`` of a gradient stage that writes none."""

_MISSING = object()


def _lookup(raw: Mapping[str, Any], key: str) -> object:
    """Return the value at dotted ``key`` in ``raw``, or ``_MISSING``."""
    node: object = raw
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _shown(value: object) -> str:
    """Return a default as the help table prints it, without brackets (Rich markup)."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def table_help() -> str:
    """Return :data:`TABLE` as help text: one ``3.x key (3.x default): what it becomes`` line per key."""
    lines = []
    for row in TABLE:
        default = "" if row.default is NO_DEFAULT else f" ({_shown(row.default)})"
        lines.append(f"{row.key}{default}: {row.becomes}")
    return "\n".join(lines)


class _Migration:
    """The notes and rejections of one :func:`migrate` call, and the free values it found."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = raw
        self.notes: list[str] = []
        self.rejections: list[Rejection] = []
        # 4.x path under imaging_system -> (start, 3.x learning rate, the emitted free-parameter mapping).
        self.free: dict[str, tuple[float, float, dict[str, Any]]] = {}
        # 4.x path -> how far the 3.x optimizer could move the value from its start (adam, nadam only).
        self.reach: dict[str, float] = {}
        # The 3.x block being translated: phase or fluorescence.
        self.contrast = "phase"
        # A #592 file: a setting wo config check warns of and does not execute is dropped with a note, not rejected.
        self.preview = False
        # A 3.x gradient optimizer with use_gradients: false took no step: the free values stay at their init.
        self.frozen = False

    def reject(self, key: str, message: str) -> None:
        self.rejections.append(Rejection(key, message))

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    def unread(self, key: str, reason: str) -> None:
        """Reject a setting 4.x does not execute; in a #592 file, which wo config check loads, drop it with a note."""
        if self.preview:
            self.note(f"{key} dropped: {reason}; wo config check warns of it and runs without it")
        else:
            self.reject(key, f"{reason}; remove it")

    def result(self, dialect: Dialect, config: dict[str, Any] | None) -> MigrationResult:
        return MigrationResult(
            dialect=dialect,
            config=None if self.rejections else config,
            notes=tuple(self.notes),
            rejections=tuple(self.rejections),
        )

    # --- shared by 3.x files and #592 files ----------------------------------------------------------------------

    def richardson_lucy_tolerance(self, rl: dict[str, Any], tolerance: object, *, key: str) -> None:
        """Write the 3.x ``stopping_tolerance`` into ``rl``: checked every iteration, as 3.x did."""
        if tolerance is None:
            return
        if tolerance == 0:
            self.note(f"{key}: 0 dropped; 3.x never stopped early with it, and 4.x runs every iteration without one")
            return
        rl["tolerance"] = tolerance
        rl["check_every"] = 1

    def search(
        self, options: Mapping[str, Any], *, given: set[str], prefix: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Translate the 3.x optimizer settings into one search stage and the ``advanced`` block.

        Parameters
        ----------
        options : Mapping
            The settings with the 3.x defaults filled in.
        given : set of str
            The settings the file wrote, which get a note when dropped.
        prefix : str
            The dotted path of the settings in the input, for notes and rejections.

        Returns
        -------
        tuple of dict, or None
            ``(stage, advanced)``, or ``None`` after a rejection.
        """
        method = options["method"]
        if method not in _METHODS:
            self.reject(f"{prefix}.method", f"{method!r} is not a 3.x optimizer; use one of {', '.join(_METHODS)}")
            return None
        start = len(self.rejections)
        self._check_stopping(options, given=given, prefix=prefix)
        self._check_unread(options, given=given, prefix=prefix)
        if len(self.rejections) > start:
            return None
        return self._stage(options)

    def _check_stopping(self, options: Mapping[str, Any], *, given: set[str], prefix: str) -> None:
        """Reject ``convergence_tol`` where 3.x did not read it; note ``convergence_patience`` where it did nothing."""
        method, tolerance = options["method"], options["convergence_tol"]
        if tolerance is not None and method == "grid_search":
            self.unread(f"{prefix}.convergence_tol", "grid_search does not read it, nor did 3.x")
        stops = tolerance is not None and method in _STEPPED
        if not stops and "convergence_patience" in given and options["convergence_patience"] is not None:
            self.note(
                f"{prefix}.convergence_patience dropped: 3.x read it only beside convergence_tol under adam, nadam "
                "or lbfgs"
            )

    def _check_unread(self, options: Mapping[str, Any], *, given: set[str], prefix: str) -> None:
        """Resolve ``use_gradients`` as 3.x did; reject ``grid_points`` where 4.x would not read it."""
        method = options["method"]
        gradients = options["use_gradients"]
        if gradients is False and method in _STEPPED:
            if self.preview:  # wo rec's optimizer is held at nominal before (_hold_nominal); calibrate-tf ignored it
                self.unread(f"{prefix}.use_gradients", "wo-4.0's calibrate-tf did not read it")
            else:
                self.frozen = True
        elif "use_gradients" in given and gradients is not None:
            what = "steps along the gradient either way" if method in _STEPPED else "ignores it"
            self.note(f"{prefix}.use_gradients dropped: {method} {what}, as in 3.x")
        if method != "grid_search" and options["grid_points"] != _OPTIMIZER_DEFAULTS["grid_points"]:
            self.unread(f"{prefix}.grid_points", f"is read only by grid_search, not {method}")
        elif method != "grid_search" and "grid_points" in given:
            self.note(f"{prefix}.grid_points dropped: read only by grid_search")

    def _stage(self, options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the search stage the 3.x method stands for, and the ``advanced`` block it needs.

        ``max_iterations`` of ``None`` leaves the stage's iterations at its default, as ``wo config check`` runs a
        #592 optimizer beside a search that is not one gradient stage.
        """
        method, iterations = options["method"], options["max_iterations"]
        budget = {} if iterations is None else {"iterations": iterations}
        tolerance = options["convergence_tol"]
        stop = {}
        if tolerance is not None and method in _STEPPED:
            patience = options["convergence_patience"]
            default = _OPTIMIZER_DEFAULTS["convergence_patience"]
            stop = {"tolerance": tolerance, "patience": default if patience is None else patience}
        advanced: dict[str, Any] = {}
        if method in ("adam", "nadam"):
            gradient = {"optimizer": method} if iterations is None else {"max_iter": iterations, "optimizer": method}
            stage: dict[str, Any] = {"gradient": gradient | stop}
            steps = _GRADIENT_STEPS if iterations is None else iterations
            advanced["learning_rates"] = {path: rate for path, (_, rate, _) in self.free.items()}
            self.reach = {path: steps * rate for path, (_, rate, _) in self.free.items()}
        elif method == "lbfgs":
            stage = {"lbfgs": budget | stop}
            self.note(
                "lbfgs: 4.x L-BFGS uses a strong-Wolfe line search within bounds; 3.x took unit steps without one"
            )
        elif method == "nelder_mead":
            stage = {"nelder_mead": {**budget, "tolerance": 1e-4 if tolerance is None else tolerance}}
        else:
            points = options["grid_points"]
            for value, rate, parameter in self.free.values():
                parameter["bounds"] = [value - points // 2 * rate, value + points // 2 * rate]
            stage = {"grid": {"points": points}}
        loss = options.get("loss")
        if loss is not None:
            advanced["loss"] = dict(loss)
        if options.get("log_dir") is not None:
            advanced["log_dir"] = options["log_dir"]
        return stage, advanced


def migrate(raw: object) -> MigrationResult:
    """Translate a 3.x configuration, or a #592 version-4 file, into the current WaveOrder 4 schema.

    Parameters
    ----------
    raw : object
        The configuration as YAML gave it (``yaml.safe_load`` of the file); anything but a mapping is rejected.

    Returns
    -------
    MigrationResult
        The 4.x mapping with its notes, or the rejections keyed by their path in ``raw``. A current 4.x mapping is
        returned unchanged with ``dialect="4"``.
    """
    if not isinstance(raw, Mapping):
        return MigrationResult("3.x", None, rejections=(Rejection("(file)", "must hold a YAML mapping"),))
    if "version" not in raw:
        return _from_3x(_Migration(raw))
    if raw["version"] != 4:
        return MigrationResult(
            "4", None, rejections=(Rejection("version", f"{raw['version']!r} is not a known version; expected 4"),)
        )
    # #592 required data.pixel_size.yx, so it marks a #592 file; a {z, yx} file is read as one too.
    pixel_size = _lookup(raw, "data.pixel_size")
    if isinstance(pixel_size, Mapping) and "yx" in pixel_size:
        return _from_preview(_Migration(raw))
    return MigrationResult("4", dict(raw))


# --- 3.x -----------------------------------------------------------------------------------------------------------


def _unknown_keys(migration: _Migration, contrast: str) -> None:
    """Reject keys that are not 3.x settings, block by block."""
    for block, allowed in _BLOCK_KEYS.items():
        if block.split(".")[0] not in ("", contrast, "optimization"):
            continue
        node = _lookup(migration.raw, block) if block else migration.raw
        if not isinstance(node, Mapping):
            continue
        for key in node:
            if key not in allowed:
                migration.reject(f"{block}.{key}" if block else str(key), "is not a waveorder 3.x setting")


def _from_3x(migration: _Migration) -> MigrationResult:
    raw = migration.raw
    stitched = [key for key in _TILE_STITCH_KEYS if key in raw]
    if stitched:
        migration.reject(stitched[0], _TILE_STITCH_MESSAGE.format(keys=", ".join(stitched)))
    if raw.get("birefringence") is not None:
        migration.reject("birefringence", _BIREFRINGENCE_MESSAGE)
    contrasts = [name for name in ("phase", "fluorescence") if raw.get(name) is not None]
    if stitched or migration.rejections:
        return migration.result("3.x", None)
    if not contrasts:
        migration.reject("phase", "a 3.x configuration needs a phase or a fluorescence block")
        return migration.result("3.x", None)
    if len(contrasts) > 1:
        migration.reject("fluorescence", "cannot be present alongside phase (3.x refuses this too)")
        return migration.result("3.x", None)
    migration.contrast = contrasts[0]
    _unknown_keys(migration, migration.contrast)
    if migration.rejections:
        return migration.result("3.x", None)
    return migration.result("3.x", _translate(migration))


def _value(migration: _Migration, key: str) -> object:
    """Return the 3.x value at ``key`` (``<contrast>`` replaced), or its 3.x default."""
    found = _lookup(migration.raw, key.replace(_C, migration.contrast))
    return _DEFAULTS[key] if found is _MISSING else found


def _changed(migration: _Migration, key: str) -> bool:
    """Whether the file sets ``key`` to something other than its 3.x default."""
    found = _lookup(migration.raw, key.replace(_C, migration.contrast))
    return found is not _MISSING and found != _DEFAULTS[key]


def _calibratable(migration: _Migration, key: str, *, path: str) -> object:
    """Return a 3.x number or ``{init, lr}`` as a plain number, or as a free 4.x parameter when ``lr > 0``."""
    value = _value(migration, key)
    if not isinstance(value, Mapping):
        return value
    unknown = set(value) - {"init", "initial_value", "lr", "learning_rate"}
    if unknown:
        shown = key.replace(_C, migration.contrast)
        migration.reject(shown, f"{', '.join(sorted(unknown))}: not a setting of a 3.x optimizable value")
        return None
    start = value.get("initial_value", value.get("init", 0.0))
    rate = value.get("learning_rate", value.get("lr", 0.01))
    if not rate > 0:
        return start
    parameter: dict[str, Any] = {"nominal": start, "tolerance": "free"}
    migration.free[path] = (start, rate, parameter)
    return parameter


def _pixel_size(migration: _Migration) -> dict[str, Any]:
    yx = _value(migration, f"{_TF}.yx_pixel_size")
    y, x = (yx.get("y"), yx.get("x")) if isinstance(yx, Mapping) else (yx, yx)
    migration.note(_PIXEL_NOTE)
    return {"z": _value(migration, f"{_TF}.z_pixel_size"), "y": y, "x": x}


def _imaging_system(migration: _Migration) -> dict[str, Any]:
    contrast = migration.contrast
    detection: dict[str, Any] = {}
    if contrast == "fluorescence":
        detection["wavelength"] = _value(migration, "fluorescence.transfer_function.wavelength_emission")
    detection["numerical_aperture"] = _calibratable(
        migration, f"{_TF}.numerical_aperture_detection", path="detection.numerical_aperture"
    )
    if _value(migration, "reconstruction_dimension") == 3:
        _ignored_offset(migration, _value(migration, f"{_TF}.z_focus_offset"))
        detection["z_focus_offset"] = 0
    else:
        detection["z_focus_offset"] = _calibratable(migration, f"{_TF}.z_focus_offset", path="detection.z_focus_offset")
    if contrast == "fluorescence":
        pinhole = _value(migration, "fluorescence.transfer_function.confocal_pinhole_diameter")
        detection["pinhole"] = "none" if pinhole is None else {"diameter_um": pinhole}
    detection["edge"] = "sigmoid"
    migration.note(_EDGE_NOTE)
    system: dict[str, Any] = {
        "contrast": contrast,
        "index_of_refraction_media": _value(migration, f"{_TF}.index_of_refraction_media"),
        "detection": detection,
    }
    if contrast == "phase":
        system["illumination"] = {
            "wavelength": _value(migration, "phase.transfer_function.wavelength_illumination"),
            "numerical_aperture": _calibratable(
                migration,
                "phase.transfer_function.numerical_aperture_illumination",
                path="illumination.numerical_aperture",
            ),
            "tilt": {
                "zenith": _calibratable(migration, f"{_TF}.tilt_angle_zenith", path="illumination.tilt.zenith"),
                "azimuth": _calibratable(migration, f"{_TF}.tilt_angle_azimuth", path="illumination.tilt.azimuth"),
            },
            "invert_phase_contrast": _value(migration, "phase.transfer_function.invert_phase_contrast"),
        }
        return system
    for name in ("tilt_angle_zenith", "tilt_angle_azimuth"):
        key = f"fluorescence.transfer_function.{name}"
        if _lookup(migration.raw, key) not in (_MISSING, 0, 0.0):
            migration.note(f"{key} dropped: 3.x fluorescence did not read it")
    return system


def _ignored_offset(migration: _Migration, offset: object) -> None:
    """Note a 3D reconstruction's ``z_focus_offset``, which 3.x and wo-4.0 ignored and 4.x applies."""
    if offset not in (0, 0.0, _MISSING):
        migration.note(
            f"imaging_system.detection.z_focus_offset: {offset} written as 0: 3.x and wo-4.0 ignore the offset of a "
            "3D reconstruction and 4.x applies it, so 0 keeps the results equal"
        )


def _inverse(migration: _Migration) -> tuple[dict[str, Any], dict[str, Any], object] | None:
    """Return ``(object_block, noise, reconstruction)`` for the 3.x ``apply_inverse`` block, or ``None`` if rejected."""
    contrast = migration.contrast
    algorithm = _value(migration, f"{_AI}.reconstruction_algorithm")
    dimension = _value(migration, "reconstruction_dimension")
    algorithm_key = f"{contrast}.apply_inverse.reconstruction_algorithm"
    object_block: dict[str, Any] = {"dimension": dimension}
    noise: dict[str, Any] = {"gaussian": {"sigma": 1.0}}
    if algorithm in ("RL", "RLGC"):
        if contrast != "fluorescence" or dimension != 3:
            migration.reject(algorithm_key, f"{algorithm} is 3D fluorescence only (3.x refuses this too)")
            return None
        object_block["prior"] = "none"
        noise = {"poisson": {}}
        if _changed(migration, f"{_AI}.regularization_strength"):
            migration.note(f"{contrast}.apply_inverse.regularization_strength dropped: {algorithm} does not read it")
        reconstruction: object = {
            "richardson_lucy": _richardson_lucy(migration, gradient_consensus=algorithm == "RLGC")
        }
    elif algorithm == "TV":
        object_block |= {"prior": "tv", "weight": _value(migration, f"{_AI}.regularization_strength")}
        iterations, rho = _value(migration, f"{_AI}.TV_iterations"), _value(migration, f"{_AI}.TV_rho_strength")
        reconstruction = {"admm": {"iterations": iterations, "rho": rho}}
        migration.note("object.prior: tv is isotropic total variation by ADMM in 4.x; numbers differ from 3.x")
    elif algorithm == "Tikhonov":
        weight = _value(migration, f"{_AI}.regularization_strength")
        auto = _value(migration, f"{_AI}.auto_regularization")
        if auto is not None and dimension == 3:
            if _changed(migration, f"{_AI}.regularization_strength"):
                migration.note(f"{contrast}.apply_inverse.regularization_strength dropped: auto_regularization sets it")
            weight = _auto_weight(migration, auto, key=f"{contrast}.apply_inverse.auto_regularization")
        object_block |= {"prior": "tikhonov", "weight": weight}
        reconstruction = "closed_form"
    else:
        migration.reject(algorithm_key, f"{algorithm!r} is not a 3.x reconstruction_algorithm")
        return None
    _ignored_blocks(migration, algorithm=str(algorithm), dimension=dimension)
    return object_block, noise, reconstruction


def _ignored_blocks(migration: _Migration, *, algorithm: str, dimension: object) -> None:
    """Note the settings 3.x ignored for ``algorithm``, auto-regularization outside 3D Tikhonov among them."""
    contrast = migration.contrast
    if algorithm != "TV":
        for name in ("TV_rho_strength", "TV_iterations"):
            if _changed(migration, f"{_AI}.{name}"):
                migration.note(f"{contrast}.apply_inverse.{name} dropped: read only by TV")
    if algorithm not in ("RL", "RLGC") and _lookup(migration.raw, f"{contrast}.apply_inverse.rl") not in (
        _MISSING,
        None,
    ):
        migration.note(f"{contrast}.apply_inverse.rl dropped: 3.x ignored it for {algorithm}")
    if _value(migration, f"{_AI}.auto_regularization") is None or (algorithm == "Tikhonov" and dimension == 3):
        return
    auto_key = f"{contrast}.apply_inverse.auto_regularization"
    reason = f"reconstruction_algorithm is {algorithm}" if algorithm != "Tikhonov" else "the reconstruction is 2D"
    migration.note(f"{auto_key} dropped: 3.x ignored it ({reason})")


def _auto_weight(migration: _Migration, auto: object, *, key: str) -> dict[str, Any] | None:
    """Return the 3.x ``auto_regularization`` block as ``object.weight: {auto: ...}``: the same rule and sweep."""
    if not isinstance(auto, Mapping):
        migration.reject(key, "must be a mapping of the 3.x auto-regularization settings")
        return None
    unknown = set(auto) - {*_AUTO_FIELDS, "report_path", "plot_path"}
    if unknown:
        migration.reject(key, f"{', '.join(sorted(map(str, unknown)))}: not a 3.x auto-regularization setting")
    paths = {name: auto[name] for name in ("report_path", "plot_path") if auto.get(name) is not None}
    return {"auto": {name: auto[name] for name in _AUTO_FIELDS if name in auto} | paths}


def _richardson_lucy(migration: _Migration, *, gradient_consensus: bool) -> dict[str, Any]:
    prefix = "fluorescence.apply_inverse.rl"
    rl: dict[str, Any] = {name: _value(migration, f"{prefix}.{name}") for name in _RL_FIELDS}
    for name in ("bp_alpha", "bp_beta"):
        if _value(migration, f"{prefix}.{name}") is not None:
            rl[name] = _value(migration, f"{prefix}.{name}")
    rl["gradient_consensus"] = gradient_consensus
    tolerance_key = f"{prefix}.stopping_tolerance"
    migration.richardson_lucy_tolerance(rl, _value(migration, tolerance_key), key=tolerance_key)
    rl["start"] = "ones"
    migration.note(_START_NOTE)
    return rl


def _calibration(migration: _Migration) -> dict[str, Any] | None:
    """Return the 3.x ``optimization`` block as ``solver.calibration``; ``None`` when 3.x would not optimize."""
    given = migration.raw.get(_OPT)
    if not migration.free:
        if given is not None:
            migration.note("optimization dropped: no transfer-function value has lr > 0, so 3.x did not optimize")
        return None
    settings = dict(given) if isinstance(given, Mapping) else {}
    options = {**_OPTIMIZER_DEFAULTS, "loss": None, "log_dir": None, **settings}
    translated = migration.search(options, given=set(settings), prefix=_OPT)
    if translated is None:
        return None
    stage, advanced = translated
    if migration.frozen:
        log = " (and its log_dir: 4.x logs no calibration that does not run)" if options.get("log_dir") else ""
        migration.note(
            f"{_OPT}.use_gradients: false: 3.x's {options['method']} took no step with it, so "
            f"{', '.join(f'imaging_system.{path}' for path in migration.free)} are written at their init and the "
            f"optimization is dropped{log}"
        )
        return None
    migration.note(
        "solver.calibration: 3.x optimized before reconstructing and wrote <config>_optimized.yml; 4.x stores the "
        "calibrated values in the transfer-function store (wo config show FILE -t STORE prints them)"
    )
    calibration: dict[str, Any] = {"score": "sharpness", "search": [stage]}
    if advanced:
        calibration["advanced"] = advanced
    return calibration


def _hold_init(migration: _Migration, system: dict[str, Any]) -> None:
    """Write each free value of ``system`` as its 3.x init, which an optimizer that took no step left it at."""
    for path, (start, _, _) in migration.free.items():
        parent, _, key = path.rpartition(".")
        holder = _lookup(system, parent) if parent else system
        assert isinstance(holder, dict)
        holder[key] = start
    migration.free, migration.reach = {}, {}


def _translate(migration: _Migration) -> dict[str, Any] | None:
    channels = _value(migration, "input_channel_names")
    if not isinstance(channels, list) or len(channels) != 1:
        count = len(channels) if isinstance(channels, list) else channels
        migration.reject("input_channel_names", f"a {migration.contrast} reconstruction reads one channel, got {count}")
    system = _imaging_system(migration)
    inverse = _inverse(migration)
    calibration = _calibration(migration)
    if migration.frozen:
        _hold_init(migration, system)
    _bound_apertures(
        migration, system, thin=_value(migration, "reconstruction_dimension") == 2, low=_LINEAR_LOW_APERTURE
    )
    _bound_zenith(migration, system)
    if inverse is None or migration.rejections:
        return None
    object_block, noise, reconstruction = inverse
    thin_phase = migration.contrast == "phase" and object_block["dimension"] == 2
    gradient = calibration is not None and next(iter(calibration["search"][0])) in ("gradient", "lbfgs")
    if gradient and thin_phase and reconstruction == "closed_form":
        _score_per_field(migration, calibration)
    if calibration is not None and migration.contrast == "phase":
        _fit_soft_edge(migration, calibration)
    solver: dict[str, Any] = {
        "padding": {"z": _value(migration, f"{_TF}.z_padding")},
        "reconstruction": reconstruction,
    }
    if calibration is not None:
        solver["calibration"] = calibration
    solver["apodization_rolloff"] = _value(migration, f"{_AI}.apodization_rolloff")
    solver["device"] = _value(migration, "device")
    return {
        "version": 4,
        "data": {
            "input_channel_names": channels,
            "time_indices": _value(migration, "time_indices"),
            "pixel_size": _pixel_size(migration),
        },
        "imaging_system": system,
        "object": object_block,
        "noise": noise,
        "solver": solver,
    }


# --- #592 version-4 files -------------------------------------------------------------------------------------------


def _from_preview(migration: _Migration) -> MigrationResult:
    migration.preview = True
    config = copy.deepcopy(dict(migration.raw))
    object_block = config.get("object")
    if isinstance(object_block, dict):
        _preview_weight(migration, object_block)
    pixel_size = _lookup(config, "data.pixel_size")
    if isinstance(pixel_size, dict) and "yx" in pixel_size:
        yx = pixel_size.pop("yx")
        y, x = ((yx.get("y"), yx.get("x")) if yx else (0.1, 0.1)) if isinstance(yx, Mapping) else (yx, yx)
        config["data"]["pixel_size"] = {"z": pixel_size.pop("z", None), "y": y, "x": x, **pixel_size}
    migration.note(_PIXEL_NOTE)
    system = config.get("imaging_system")
    if isinstance(system, dict):
        detection = system.setdefault("detection", {})
        if isinstance(detection, dict) and "edge" not in detection:
            detection["edge"] = "sigmoid"
            migration.note(_EDGE_NOTE)
        if isinstance(detection, dict) and _lookup(config, "object.dimension") in (3, _MISSING):
            _preview_offset(migration, config, detection)
    solver = config.get("solver")
    if isinstance(solver, dict):
        per_field = _scored_per_field(config)  # read from the file as written, before its optimizer is translated
        soft = _lookup(config, "imaging_system.contrast") == "phase" and bool(_wo_rec_free(config))
        _preview_solver(migration, solver, system if isinstance(system, dict) else {})
        calibration = solver.get("calibration")
        if per_field and isinstance(calibration, dict):
            _score_per_field(migration, calibration)
        sigmoid = _lookup(config, "imaging_system.detection.edge") == "sigmoid"
        if soft and sigmoid and isinstance(calibration, dict):
            _fit_soft_edge(migration, calibration)
    if isinstance(system, dict):
        _bound_apertures(migration, system, thin=_lookup(config, "object.dimension") == 2, low=_LINEAR_LOW_APERTURE)
        _bound_zenith(migration, system)
    return migration.result("592", config)


def _preview_weight(migration: _Migration, object_block: dict[str, Any]) -> None:
    """Drop the weight wo-4.0 wrote for a prior that has none; keep a rule-based weight, without its unset paths."""
    weight = object_block.get("weight", _MISSING)
    if object_block.get("prior") == "none" and weight is not _MISSING:
        del object_block["weight"]
        migration.note(
            f"object.weight: {weight} dropped: object.prior: none has no weight (wo-4.0 wrote the 3.x "
            "regularization_strength, which Richardson-Lucy does not read)"
        )
    auto = weight.get("auto") if isinstance(weight, dict) else None
    if isinstance(auto, dict):
        for name in ("report_path", "plot_path"):
            if auto.get(name, _MISSING) is None:
                del auto[name]


def _preview_offset(migration: _Migration, config: dict[str, Any], detection: dict[str, Any]) -> None:
    """Write a 3D #592 file's ``z_focus_offset`` as 0; drop ``solver.calibration`` if it was the only free value."""
    offset = detection.get("z_focus_offset", _MISSING)
    _ignored_offset(migration, offset)
    detection["z_focus_offset"] = 0
    solver = config.get("solver")
    if not isinstance(offset, Mapping) or not isinstance(solver, dict):
        return
    calibration = solver.get("calibration")
    if not isinstance(calibration, dict):
        return
    advanced = calibration.get("advanced")
    if isinstance(advanced, dict) and isinstance(advanced.get("learning_rates"), dict):
        advanced["learning_rates"].pop("detection.z_focus_offset", None)
    if not _has_free(config.get("imaging_system")) and _lookup(config, "object.weight") != "auto":
        del solver["calibration"]
        migration.note("solver.calibration dropped: nothing else is free, so 3.x and wo-4.0 did not calibrate")


def _has_free(node: object) -> bool:
    """Whether ``node`` holds a free value: a mapping with a ``tolerance``."""
    if isinstance(node, Mapping):
        return "tolerance" in node or any(_has_free(value) for value in node.values())
    return isinstance(node, list) and any(_has_free(value) for value in node)


_LOW_APERTURE = 0.01
"""The lower bound a free aperture without bounds gets in a 4.x configuration."""

_LINEAR_LOW_APERTURE = 0.02
"""The lower bound it gets from 3.x or wo-4.0: within a factor of 100 of the index, so the span is searched linearly
and an Adam step stays in aperture units, as 3.x and wo-4.0 took it (a wider span is searched in log space)."""


def _bound_apertures(migration: _Migration, system: dict[str, Any], *, thin: bool, low: float) -> None:
    """Give a free numerical aperture without bounds the range its system takes, from ``low``.

    3.x optimized apertures without bounds; 4.x requires a positive range up to ``index_of_refraction_media``. A 2D
    phase system also needs the illumination aperture at most the detection aperture: a free one is bounded by the
    other's value or bound, or, both free, by the point halfway between their starts.
    """
    index = system.get("index_of_refraction_media", 1.3)
    pair = thin and system.get("contrast") == "phase"
    for path, other in zip(_APERTURES, reversed(_APERTURES), strict=True):
        parameter = _lookup(system, path)
        if not isinstance(parameter, dict) or "nominal" not in parameter or parameter.get("bounds") is not None:
            continue
        bounds = [low, index]
        split = _split(parameter["nominal"], _lookup(system, other), detection=path == _APERTURES[0]) if pair else None
        if split is not None:
            bounds[path == _APERTURES[1]] = split
            low_bound, high = bounds
            if not low_bound < high or not low_bound <= parameter["nominal"] <= high:
                bounds, split = [low, index], None  # the apertures conflict at their starts, which the system refuses
        parameter["bounds"] = bounds
        reason = (
            f"{low} to the index of refraction"
            if split is None
            else f"a 2D phase system keeps the illumination aperture at most the detection aperture, split at {split}"
        )
        if low == _LINEAR_LOW_APERTURE and bounds[0] == low:
            reason += ", within a factor of 100 so a gradient step stays in aperture units, as 3.x and wo-4.0 stepped"
        migration.note(
            f"imaging_system.{path} is free without bounds; 4.x needs a positive range, so it gets bounds {bounds} "
            f"({reason})"
        )


def _split(start: object, other: object, *, detection: bool) -> float | None:
    """Return where a 2D phase aperture starting at ``start`` meets ``other``: its value, bound, or halfway point."""
    if isinstance(other, Mapping):
        bounds = other.get("bounds")
        if bounds is not None:
            return bounds[1] if detection else bounds[0]
        other = other.get("nominal")
        if isinstance(other, int | float) and isinstance(start, int | float):
            return round((start + other) / 2, 6)
        return None
    return other if isinstance(other, int | float) and not isinstance(other, bool) else None


_ZENITH = "illumination.tilt.zenith"


def _bound_zenith(migration: _Migration, system: dict[str, Any]) -> None:
    """Give a free tilt zenith without bounds the range 3.x could reach that keeps brightfield.

    3.x optimized the raw angle without bounds or clamping. Adam and NAdam move a value by about its learning rate
    per step at most, so by ``max_iterations * lr`` from its start; the other methods have no such limit. 4.x needs
    bounds that keep the unscattered light inside the detection aperture, ``n sin(zenith)`` below the smallest
    detection numerical aperture, so the range is cut to ``|zenith| < asin(NA / n)`` (rounded down to 1e-3 rad).
    """
    parameter = _lookup(system, _ZENITH)
    if not isinstance(parameter, dict) or "nominal" not in parameter or parameter.get("bounds") is not None:
        return
    aperture = _lookup(system, "detection.numerical_aperture")
    if aperture is _MISSING:
        aperture = 1.2
    elif isinstance(aperture, Mapping):
        aperture = (aperture.get("bounds") or [aperture.get("nominal")])[0]
    index = system.get("index_of_refraction_media", 1.3)
    start = parameter["nominal"]
    if not all(isinstance(value, int | float) for value in (aperture, index, start)):
        return  # validation names the setting that is not a number
    edge = math.floor(math.asin(min(aperture / index, 1.0)) * 1000) / 1000
    reach = migration.reach.get(_ZENITH, math.inf)
    parameter["bounds"] = [round(max(start - reach, -edge), 6), round(min(start + reach, edge), 6)]
    migration.note(
        f"imaging_system.{_ZENITH} is free without bounds; 4.x needs bounds, so it gets {parameter['bounds']}: what "
        f"3.x could reach from {start}, inside |zenith| < asin({aperture} / {index}), where the unscattered light "
        "stays inside the detection aperture"
    )


def bounded(raw: Mapping[str, Any], *, wo40: bool = False) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return ``raw`` with the range a migration gives each free aperture and tilt zenith written without bounds.

    wo-4.0 took these unbounded, and 4.x needs a range, so :func:`waveorder.config.load` reads such a configuration
    with the range :func:`migrate` writes: for a file in wo-4.0's spelling (``wo40``), the one that keeps wo-4.0's
    steps. A wo-4.0 ``advanced.optimizer: grid_search`` gets wo-4.0's grid span as bounds. Returns the copy and one
    note per range, each ``imaging_system.<path> ...``.
    """
    config = copy.deepcopy(dict(raw))
    system = config.get("imaging_system")
    migration = _Migration(raw)
    if isinstance(system, dict):
        migration.reach = _reach(config)
        _grid_span(migration, config)
        low = _LINEAR_LOW_APERTURE if wo40 else _LOW_APERTURE
        _bound_apertures(migration, system, thin=_lookup(config, "object.dimension") == 2, low=low)
        _bound_zenith(migration, system)
    return config, tuple(migration.notes)


def wo40_differences(raw: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return what 4.x reads differently in the wo-4.0 file ``raw`` and ``wo config migrate`` writes as wo-4.0 ran it.

    Returns the defaults that differ, named for the file-level warning, and one note per located difference, each
    ``<path> <message>``.
    """
    defaults = []
    detection = _lookup(raw, "imaging_system.detection")
    if not isinstance(detection, Mapping) or "edge" not in detection:
        defaults.append("detection.edge: sigmoid")
    offset = detection.get("z_focus_offset", 0) if isinstance(detection, Mapping) else 0
    if _lookup(raw, "object.dimension") in (3, _MISSING) and offset != 0:
        defaults.append("a 3D z_focus_offset read as 0")
    rl = _lookup(raw, "solver.reconstruction.richardson_lucy")
    if isinstance(rl, Mapping) and "start" not in rl:
        defaults.append("Richardson-Lucy started from ones")
    if _lookup(raw, "imaging_system.contrast") == "phase" and _wo_rec_free(raw):
        defaults.append("a phase fit at pupil edge steepness 100")
    notes = []
    if _wo_rec_steps(raw) is not None:
        search = _lookup(raw, "solver.calibration.search")
        at = "solver.calibration.search" if search is _MISSING else "solver.calibration.search.0.gradient"
        notes.append(
            f"{at} wo-4.0's wo rec ran 10 adam steps for a sharpness calibration without advanced.optimizer, "
            f"whatever max_iter said; this runs the gradient stage's max_iter ({_wo_rec_steps(raw)}); wo config "
            "migrate writes max_iter: 10"
        )
    return tuple(defaults), tuple(notes)


_WO_REC_STEPS = 10
"""The adam steps wo-4.0's ``wo rec`` took without ``advanced.optimizer`` (3.x ``OptimizationSettings()``)."""


def _wo_rec_steps(config: Mapping[str, Any]) -> int | None:
    """Return the max_iter 4.x runs where wo-4.0's wo rec ran its default 10 adam steps instead; ``None`` elsewhere."""
    calibration = _lookup(config, "solver.calibration")
    advanced = calibration.get("advanced") if isinstance(calibration, Mapping) else None
    if (isinstance(advanced, Mapping) and advanced.get("optimizer") is not None) or not _wo_rec_free(config):
        return None
    search = calibration.get("search", _MISSING) if isinstance(calibration, Mapping) else None
    if search is _MISSING:
        return _GRADIENT_STEPS
    stage = search[0] if isinstance(search, list) and len(search) == 1 else None
    gradient = stage.get("gradient") if isinstance(stage, Mapping) else None
    if not isinstance(gradient, Mapping):
        return None  # wo-4.0's wo rec refused any other search
    steps = gradient.get("max_iter", _GRADIENT_STEPS)
    return None if steps == _WO_REC_STEPS else steps


def _free_values(node: object, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Return each free value under ``node``, a mapping with a ``tolerance``, by its dotted path."""
    if not isinstance(node, dict):
        return {}
    if "tolerance" in node:
        return {prefix: node}
    found: dict[str, dict[str, Any]] = {}
    for key, value in node.items():
        found |= _free_values(value, f"{prefix}.{key}" if prefix else str(key))
    return found


def _wo_rec_free(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the free values wo-4.0's ``wo rec`` calibrated with the 3.x optimizer, by path under imaging_system.

    That is a sharpness calibration with one gradient stage of a shift-invariant system whose free values all have
    ``tolerance: free`` and none is a pupil aberration (wo-4.0's ``calibrate-tf`` recovered those); none otherwise,
    as wo-4.0 refused the rest.
    """
    calibration = _lookup(config, "solver.calibration")
    if not isinstance(calibration, Mapping) or calibration.get("score", "evidence") != "sharpness":
        return {}
    search = calibration.get("search", [{"gradient": {}}])
    stage = search[0] if isinstance(search, list) and len(search) == 1 else None
    if not (isinstance(stage, Mapping) and "gradient" in stage):
        return {}
    if _lookup(config, "imaging_system.field_dependence") not in (_MISSING, "shift_invariant"):
        return {}
    free = _free_values(config.get("imaging_system"))
    pupil = any(path.startswith("detection.pupil_aberrations") for path in free)
    if pupil or any(value["tolerance"] != "free" for value in free.values()):
        return {}
    return free


def _scored_per_field(config: Mapping[str, Any]) -> bool:
    """Whether wo-4.0's ``wo rec`` scored the calibration of ``config`` on per-channel estimates.

    It did for a gradient fit (adam by default, nadam, lbfgs) of 2D phase under Tikhonov, where 3.x's
    ``isotropic_thin_3d.reconstruct`` chose the per-channel inverse because the transfer function carried gradients.
    """
    advanced = _lookup(config, "solver.calibration.advanced")
    method = advanced.get("optimizer") if isinstance(advanced, Mapping) else None
    return (
        _lookup(config, "imaging_system.contrast") == "phase"
        and _lookup(config, "object.dimension") == 2
        and _lookup(config, "object.prior") in (_MISSING, "tikhonov")
        and _lookup(config, "solver.reconstruction") in (_MISSING, "closed_form")
        and method in (None, "adam", "nadam", "lbfgs")
        and bool(_wo_rec_free(config))
    )


def _score_per_field(migration: _Migration, calibration: dict[str, Any]) -> None:
    """Score a gradient fit of 2D phase on each field's own estimate, as 3.x did; the final estimate stays joint."""
    calibration["reconstruction"] = {"closed_form": {"coupling": "per_field"}}
    migration.note(
        "solver.calibration.reconstruction: {closed_form: {coupling: per_field}}: 3.x scored a gradient fit of 2D "
        "phase on per-channel estimates (decomposition per_channel) and wrote the final reconstruction with the "
        "joint one (svd); this keeps both"
    )


_FIT_STEEPNESS = 100.0
"""The sigmoid pupil edge steepness wo-4.0's (and 3.x's) phase optimizer fitted with (``optim_steepness``)."""


def _fit_soft_edge(migration: _Migration, calibration: dict[str, Any]) -> None:
    """Fit phase on the soft pupil edge wo-4.0's optimizer fitted on; the reconstruction keeps the detection's 1e4."""
    calibration["edge_steepness"] = _FIT_STEEPNESS
    migration.note(
        "solver.calibration.edge_steepness: 100: the 3.x phase optimizer (wo-4.0's wo rec) fitted on a sigmoid pupil "
        "edge of steepness 100 and wrote the final reconstruction at 1e4 (imaging_system.detection.edge_steepness); "
        "this keeps both"
    )


def _grid_span(migration: _Migration, config: dict[str, Any]) -> None:
    """Bound each value a wo-4.0 ``grid_search`` calibrates by the grid wo-4.0 searched.

    wo-4.0 centred ``grid_points`` values on each nominal, its learning rate apart (0.01 by default), and read no
    bounds; 4.x grids over the bounds.
    """
    advanced = _lookup(config, _PREVIEW_ADVANCED)
    if not isinstance(advanced, Mapping) or advanced.get("optimizer") != "grid_search":
        return
    rates = advanced.get("learning_rates") or {}
    points = advanced.get("grid_points") or _OPTIMIZER_DEFAULTS["grid_points"]
    if not isinstance(points, int) or points < 2:
        return  # validation names grid_points
    for path, parameter in _wo_rec_free(config).items():
        nominal, rate = parameter.get("nominal"), rates.get(path, 0.01)
        if not all(isinstance(value, int | float) for value in (nominal, rate)) or rate <= 0:
            continue  # validation names the setting that is not a positive number
        half = points // 2 * rate
        bounds = [round(nominal - half, 9), round(nominal + half, 9)]
        if parameter.get("bounds") != bounds:
            parameter["bounds"] = bounds
            migration.note(
                f"imaging_system.{path} gets bounds {bounds}: wo-4.0's grid_search spanned {points} points "
                f"{rate} apart around its nominal, and read no bounds"
            )


def _hold_nominal(migration: _Migration, solver: dict[str, Any], system: dict[str, Any]) -> bool:
    """Write each free value at its nominal and drop the calibration where wo-4.0's optimizer took no step."""
    advanced = _lookup(solver, "calibration.advanced")
    config = {"solver": solver, "imaging_system": system}
    free = _wo_rec_free(config)
    if not isinstance(advanced, Mapping) or advanced.get("use_gradients") is not False or not free:
        return False
    if advanced.get("optimizer") not in ("adam", "nadam", "lbfgs"):
        return False
    for path, parameter in free.items():
        parent, _, key = path.rpartition(".")
        holder = _lookup(system, parent) if parent else system
        assert isinstance(holder, dict)
        holder[key] = parameter.get("nominal", 0.0)
    del solver["calibration"]
    migration.note(
        f"{_PREVIEW_ADVANCED}.use_gradients: false: wo-4.0's {advanced['optimizer']} took no step with it, so "
        f"{', '.join(f'imaging_system.{path}' for path in free)} are written at their nominal values and "
        "solver.calibration is dropped"
    )
    return True


def _reach(config: Mapping[str, Any]) -> dict[str, float]:
    """Return how far wo-4.0's adam or nadam could move each free value: max_iter times its learning rate."""
    calibration = _lookup(config, "solver.calibration")
    advanced = calibration.get("advanced") if isinstance(calibration, Mapping) else None
    if not isinstance(advanced, Mapping) or advanced.get("optimizer") not in ("adam", "nadam"):
        return {}
    search = calibration.get("search") if isinstance(calibration, Mapping) else None
    stage = search[0] if isinstance(search, list) and len(search) == 1 else None
    gradient = stage.get("gradient") if isinstance(stage, Mapping) else None
    iterations = gradient.get("max_iter", _GRADIENT_STEPS) if isinstance(gradient, Mapping) else _GRADIENT_STEPS
    rates = advanced.get("learning_rates")
    if not isinstance(rates, Mapping) or not isinstance(iterations, int | float):
        return {}
    return {path: iterations * rate for path, rate in rates.items() if isinstance(rate, int | float)}


_PREVIEW_ADVANCED = "solver.calibration.advanced"


def _preview_solver(migration: _Migration, solver: dict[str, Any], system: dict[str, Any]) -> None:
    """Upgrade the #592 ``solver`` block: Richardson-Lucy's 3.x fields, and the 3.x optimizer settings."""
    reconstruction = solver.get("reconstruction")
    rl = reconstruction.get("richardson_lucy") if isinstance(reconstruction, Mapping) else None
    if isinstance(rl, dict):
        key = "solver.reconstruction.richardson_lucy.stopping_tolerance"
        migration.richardson_lucy_tolerance(rl, rl.pop("stopping_tolerance", None), key=key)
        if "start" not in rl:
            rl["start"] = "ones"
            migration.note(_START_NOTE)
    calibration = solver.get("calibration")
    if not isinstance(calibration, dict):
        return
    _wo_rec_budget(migration, calibration, {"solver": solver, "imaging_system": system})
    advanced = calibration.get("advanced")
    if isinstance(advanced, dict) and not _hold_nominal(migration, solver, system):
        _preview_advanced(migration, solver, system=system, advanced=advanced)
        if not advanced:
            del calibration["advanced"]


def _wo_rec_budget(migration: _Migration, calibration: dict[str, Any], config: Mapping[str, Any]) -> None:
    """Write the 10 adam steps wo-4.0's ``wo rec`` took without ``advanced.optimizer``, whatever max_iter said."""
    steps = _wo_rec_steps(config)
    if steps is None:
        return
    search = calibration.get("search")
    if isinstance(search, list):
        search[0]["gradient"]["max_iter"] = _WO_REC_STEPS
    else:
        calibration["search"] = [{"gradient": {"max_iter": _WO_REC_STEPS}}]
    migration.note(
        f"solver.calibration.search: max_iter {steps} written as {_WO_REC_STEPS}: without advanced.optimizer "
        f"wo-4.0's wo rec ran {_WO_REC_STEPS} adam steps, whatever max_iter said"
    )


def _preview_advanced(
    migration: _Migration, solver: dict[str, Any], *, system: dict[str, Any], advanced: dict[str, Any]
) -> None:
    """Translate #592's ``solver.calibration.advanced``: the 3.x optimizer and the settings wo-4.0 read beside it."""
    calibration = solver["calibration"]
    method = advanced.get("optimizer")
    score = calibration.get("score", "evidence")
    if score != "sharpness" and advanced.get("learning_rates"):
        migration.note(
            f"{_PREVIEW_ADVANCED}.learning_rates dropped: wo-4.0's calibrate-tf did not read it under score {score} "
            "(Adam at lr 0.01 for every quantity, the step 4.x takes without it)"
        )
        del advanced["learning_rates"]
    if method == "grid_search":
        _grid_span(migration, {"solver": solver, "imaging_system": system})
    else:
        for path, rate in (advanced.get("learning_rates") or {}).items():
            parameter = _lookup(system, path)
            if isinstance(parameter, dict) and "nominal" in parameter:
                migration.free[path] = (parameter["nominal"], rate, parameter)
    migration.reach = _reach({"solver": solver})  # the bounds wo config check gives, whichever stage runs
    advanced.pop("optimizer", None)
    if method is not None and score != "sharpness":
        migration.note(
            f"{_PREVIEW_ADVANCED}.optimizer: {method} dropped: under score {score} the search stages run, as in "
            "wo-4.0; wo config check warns of it"
        )
    if method is None or score != "sharpness":
        _preview_without_optimizer(migration, advanced)
    else:
        _preview_optimizer(migration, calibration, advanced, method=method)


def _preview_without_optimizer(migration: _Migration, advanced: dict[str, Any]) -> None:
    """Without a 3.x optimizer the search stages run, and they read none of the 3.x settings."""
    for name in ("convergence_tol", "use_gradients", "log_dir", "convergence_patience", "grid_points"):
        value = advanced.pop(name, None)
        if value is None:
            continue
        if value == _OPTIMIZER_DEFAULTS.get(name):
            migration.note(f"{_PREVIEW_ADVANCED}.{name} dropped: the 3.x default, not executed by the search stages")
        else:
            migration.unread(f"{_PREVIEW_ADVANCED}.{name}", "a 3.x setting the search stages do not execute")


def _preview_optimizer(
    migration: _Migration, calibration: dict[str, Any], advanced: dict[str, Any], *, method: object
) -> None:
    """Replace #592's ``advanced.optimizer`` and its search by the one stage the 3.x method stands for.

    As ``wo config check`` reads it: the stage runs the ``max_iter`` of a lone gradient stage, and its own default
    beside any other search, which it replaces.
    """
    search = calibration.get("search")
    stage = search[0] if isinstance(search, list) and len(search) == 1 else None
    gradient = stage.get("gradient") if isinstance(stage, Mapping) else None
    iterations = None
    if search is None:  # the default search, one gradient stage
        iterations = _GRADIENT_STEPS
    elif isinstance(gradient, Mapping):
        iterations = gradient.get("max_iter", _GRADIENT_STEPS)
    else:
        migration.note(
            "solver.calibration.search dropped: under score sharpness advanced.optimizer is the search; wo config "
            "check warns of it"
        )
    given = {name for name, value in advanced.items() if value is not None}
    options = {**_OPTIMIZER_DEFAULTS, **advanced, "method": method, "max_iterations": iterations}
    translated = migration.search(options, given=given, prefix=_PREVIEW_ADVANCED)
    if translated is None:
        return
    new_stage, new_advanced = translated
    calibration["search"] = [new_stage]
    for name in ("convergence_tol", "convergence_patience", "use_gradients", "grid_points", "log_dir", "loss"):
        advanced.pop(name, None)
    if "learning_rates" not in new_advanced:
        advanced.pop("learning_rates", None)
    advanced.update(new_advanced)
    migration.note(f"{_PREVIEW_ADVANCED}.optimizer: {method} is now the search stage {next(iter(new_stage))}")


# --- output ------------------------------------------------------------------------------------------------------


def render(result: MigrationResult, *, source: str) -> str:
    """Return ``result.config`` as YAML under a header comment naming ``source`` and listing the notes.

    Raises
    ------
    ValueError
        If ``result`` holds rejections (it has no configuration).
    """
    if result.config is None:
        raise ValueError("a rejected migration has no configuration to render")
    origin = "waveorder 3.x" if result.dialect == "3.x" else "a #592 version-4 file"
    header = [f"# WaveOrder 4 configuration migrated by wo config migrate from {origin}: {source}"]
    if result.notes:
        header.append("# Notes:")
        for note in result.notes:
            header.extend(textwrap.wrap(note, width=118, initial_indent="# - ", subsequent_indent="#   "))
    body = yaml.safe_dump(result.config, sort_keys=False, default_flow_style=None, width=118)
    return "\n".join(header) + "\n\n" + body
