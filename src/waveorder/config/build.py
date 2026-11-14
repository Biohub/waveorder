"""Build the parts of a model from a configuration: the schema, then constructor arguments, then the registry.

:func:`build` returns the imaging system and the other arguments of a
:class:`~waveorder.models.ReconstructionModel`. A part the configuration does not write is ``None``, for the model
class to choose. A constructor's own :class:`~waveorder.errors.ParameterError` is reported as a
:class:`~waveorder.errors.ConfigurationError` at the config path of the argument it names. Whether the parts can run
together is for the parts to say when the model is built.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from waveorder import registry
from waveorder.config.schema import (
    CalibrationAdvanced,
    DetectionSettings,
    FieldDependenceTiled,
    FreeParameter,
    IlluminationSettings,
    LoopOptions,
    Pinhole,
    PriorBeads,
    ReconstructionAdmm,
    ReconstructionDistilled,
    ReconstructionFista,
    ReconstructionRichardsonLucy,
    SearchGradient,
    SearchGrid,
    SearchLBFGS,
    SearchNelderMead,
    SearchRandom,
    V4Config,
    WeightAuto,
    free_parameter_specs,
    nominal_of,
    parameter_name,
)
from waveorder.errors import BundleError, ConfigurationError, ParameterError, UnsupportedOptionError
from waveorder.io.bundle import load_bundle
from waveorder.nn import (
    Distilled,
    Free,
    ImagingSystem,
    Noise,
    ObjectPrior,
    Score,
    SearchStage,
    Solver,
    free_specs,
)
from waveorder.nn.solvers import Loop
from waveorder.zernike_names import canonical_name, noll_indices_up_to_order, noll_to_name

_AUTO_WEIGHT = Free(1e-3, bounds=(1e-6, 1.0))
"""The weight of ``object.weight: auto``: free over six decades up to 1, as wo-4.0 calibrates it."""

_SHARPNESS = {
    "midband_power": "midband_power_score",
    "spectral_flatness": "spectral_flatness_score",
    "total_variation": "total_variation_score",
    "laplacian_variance": "laplacian_variance_score",
    "normalized_variance": "normalized_variance_score",
}
"""The registered score of each 3.x sharpness loss type (``solver.calibration.advanced.loss.type``)."""


def build(config: V4Config) -> tuple[ImagingSystem, dict[str, Any]]:
    """Return the imaging system of ``config`` and the other keyword arguments of its model.

    The arguments are ``object_prior``, ``noise`` and ``solver`` (``None`` when the configuration does not write
    them), ``score`` and ``search`` (``None`` unless ``solver.calibration`` is set and a quantity is free),
    ``z_padding``, ``field_tiles`` (``None``, or the ``size_yx``, ``overlap_yx`` and ``blend`` of tiled field
    dependence) and ``device``.

    Raises
    ------
    ConfigurationError
        At the config path of a value the parts refuse or do not read.
    UnsupportedOptionError
        At a setting the schema takes and no part runs yet, with the reason.
    """
    _expect_run(config)
    system = _system(config)
    written = config.model_fields_set
    object_written = {"prior", "weight", "constraints"} & config.object.model_fields_set
    prior = _prior(config) if object_written else None
    noise = _noise(config) if "noise" in written else None
    solver = _solver(config) if "reconstruction" in config.solver.model_fields_set else None
    free = any(free_specs(part) for part in (system, prior, noise) if part is not None)
    calibration = config.solver.calibration
    calibrated = calibration is not None and free
    # The bead fit estimates sigma itself (calibrate_beads) and runs one gradient search: no profile grid before it.
    profile = () if isinstance(config.object.prior, PriorBeads) else _profile(prior, noise)
    search = _search(config, profile=profile) if calibrated else None
    weight = config.object.weight
    if isinstance(weight, WeightAuto):  # the 3.x sweep of the weight: a stage of its own, nothing else free
        if calibration is not None:
            raise ConfigurationError(
                "object.weight.auto",
                "sweeps the weight with every other quantity held fixed, so it does not run with "
                "solver.calibration; remove one of them",
            )
        rule = weight.auto
        sweep = {
            "rule": rule.rule,
            "search_min": rule.search_min,
            "search_max": rule.search_max,
            "samples": rule.num_samples,
            "scale": rule.search_scale,
            "rolloff": config.solver.apodization_rolloff,
            # auto resolves per stack: the model gives the sweep the slices it chose when it calibrates
            "z_padding": 0 if config.solver.padding.z == "auto" else config.solver.padding.z,
            "keep_slices": rule.plot_path is not None,
        }
        keys = {"samples": "object.weight.auto.num_samples", "scale": "object.weight.auto.search_scale"}
        search = (_resolve("search", "weight_sweep", sweep, at="object.weight.auto", keys=keys),)
    dependence = config.imaging_system.field_dependence
    tiles = dependence.tiled if isinstance(dependence, FieldDependenceTiled) else None
    return system, {
        "object_prior": prior,
        "noise": noise,
        "solver": solver,
        # A bead calibration scores by the bead fit's own likelihood (calibrate_beads), not by solver.calibration.score.
        "score": _score(config) if calibrated and not isinstance(config.object.prior, PriorBeads) else None,
        "search": search,
        "z_padding": config.solver.padding.z,
        "field_tiles": None
        if tiles is None
        else {"size_yx": tiles.size_yx, "overlap_yx": tiles.overlap_yx, "blend": tiles.tile_blend},
        "device": config.solver.device,
    }


def _resolve(
    kind: str, name: str, arguments: Mapping[str, object], *, at: str, keys: Mapping[str, str] | None = None
) -> Any:  # noqa: ANN401
    """Build a registered part; a constructor's ``ParameterError`` is located at the config path of its argument.

    An argument is at ``keys[argument]``, else at ``<at>.<argument>``. The error's path is followed from its first
    segment that names an argument, so ``illumination.tilt.zenith`` raised by the tilt built at
    ``imaging_system.illumination.tilt`` lands on ``...tilt.zenith``; any other path stays at ``at``.
    """
    try:
        return registry.resolve({name: dict(arguments)}, kind)
    except ParameterError as error:
        keys = keys or {}
        parts = (error.path or "").split(".")
        field = at
        for index, part in enumerate(parts):
            if part in keys or part in arguments:
                field = ".".join((keys.get(part, f"{at}.{part}"), *parts[index + 1 :]))
                break
        raise ConfigurationError(field, str(error)) from error


def _quantity(value: object, *, at: str) -> object:
    """Return a configured number, or ``Free`` for ``{nominal, tolerance, bounds}``."""
    if not isinstance(value, FreeParameter):
        return value
    tolerance = None if value.tolerance == "free" else value.tolerance
    try:
        return Free(value.nominal, tolerance=tolerance, bounds=value.bounds)
    except ParameterError as error:
        raise ConfigurationError(f"{at}.{error.path}" if error.path else at, str(error)) from error


def _unread(settings: object, default: object, *, at: str, by: str) -> None:
    """Refuse a value of ``settings`` that differs from ``default``, where no part reads it."""
    given = settings.model_dump() if isinstance(settings, BaseModel) else settings
    usual = default.model_dump() if isinstance(default, BaseModel) else default
    if isinstance(given, dict) and isinstance(usual, dict):
        for key, value in given.items():
            _unread(value, usual.get(key), at=f"{at}.{key}", by=by)
    elif given != usual:
        raise ConfigurationError(at, f"is set, but {by} does not read it; remove it")


def _expect_run(config: V4Config) -> None:
    """Refuse settings the schema takes and no part runs."""
    solver = config.solver
    if solver.padding.z and config.object.dimension == 2:
        raise UnsupportedOptionError(
            "solver.padding.z", str(solver.padding.z), reason="a thin object has one plane, nothing to pad; set 0"
        )


# --- the imaging system ----------------------------------------------------------------------------------------------


def _detection(settings: DetectionSettings, wavelength: float | None) -> object:
    at = "imaging_system.detection"
    arguments: dict[str, object] = {
        "wavelength": wavelength,
        "numerical_aperture": _quantity(settings.numerical_aperture, at=f"{at}.numerical_aperture"),
        "z_focus_offset": _quantity(settings.z_focus_offset, at=f"{at}.z_focus_offset"),
        "edge": settings.edge,
    }
    aberrations = settings.pupil_aberrations
    if aberrations is not None:
        field = f"{at}.pupil_aberrations"
        terms: dict[str, object] = {}
        shorthand = aberrations.zernike_up_to_order
        if shorthand is not None:
            free = FreeParameter(nominal=0.0, tolerance=shorthand.tolerance, bounds=shorthand.bounds)
            for j in noll_indices_up_to_order(shorthand.order):
                if noll_to_name(j) not in aberrations.terms:
                    terms[noll_to_name(j)] = _quantity(free, at=f"{field}.zernike_up_to_order")
        terms |= {name: _quantity(value, at=f"{field}.{name}") for name, value in aberrations.terms.items()}
        if terms:
            arguments["pupil_aberrations"] = _resolve("pupil_factor", "zernike", {"terms": terms}, at=field)
    return _resolve("detection", "detection", arguments, at=at)


def _pinhole(settings: DetectionSettings) -> object:
    at = "imaging_system.detection.pinhole"
    pinhole = settings.pinhole
    if not isinstance(pinhole, Pinhole):
        return 0.0  # point
    if pinhole.diameter_um is not None:
        return _quantity(pinhole.diameter_um, at=f"{at}.diameter_um")
    aperture = nominal_of(settings.numerical_aperture)
    assert settings.wavelength is not None
    assert aperture is not None
    assert pinhole.diameter_au is not None
    airy = 1.22 * settings.wavelength / aperture  # 1 AU at the nominal aperture: the pinhole does not move with it
    diameter = pinhole.diameter_au
    if isinstance(diameter, FreeParameter):
        tolerance = diameter.tolerance if diameter.tolerance == "free" else diameter.tolerance * airy
        bounds = None if diameter.bounds is None else (diameter.bounds[0] * airy, diameter.bounds[1] * airy)
        diameter = FreeParameter(nominal=diameter.nominal * airy, tolerance=tolerance, bounds=bounds)
        return _quantity(diameter, at=f"{at}.diameter_au")
    return float(diameter) * airy


def _system(config: V4Config) -> ImagingSystem:
    settings = config.imaging_system
    detection, at = settings.detection, "imaging_system"
    index = settings.index_of_refraction_media
    thin = config.object.dimension == 2
    if settings.contrast == "fluorescence":
        _unread(settings.illumination, IlluminationSettings(), at=f"{at}.illumination", by="fluorescence")
        if detection.wavelength is None:
            raise ConfigurationError(f"{at}.detection.wavelength", "the emission wavelength is required")
        if detection.pinhole != "none":
            if thin:
                raise UnsupportedOptionError(
                    f"{at}.detection.pinhole", "confocal", reason="confocal detection needs object.dimension: 3"
                )
            arguments = {
                "detection": _detection(detection, detection.wavelength),
                "pinhole_diameter": _pinhole(detection),
                "index_of_refraction_media": index,
            }
            pinhole, written = detection.pinhole, ""
            if isinstance(pinhole, Pinhole):
                written = ".diameter_um" if pinhole.diameter_um is not None else ".diameter_au"
            keys = {
                "pinhole_diameter": f"{at}.detection.pinhole{written}",
                "index_of_refraction_media": f"{at}.index_of_refraction_media",
            }
            return _resolve("system", "confocal_fluorescence_3d", arguments, at=at, keys=keys)
        name = "thin_fluorescence_2d" if thin else "widefield_fluorescence_3d"
        arguments = {"detection": _detection(detection, detection.wavelength), "index_of_refraction_media": index}
        return _resolve("system", name, arguments, at=at)
    illumination = settings.illumination
    field = f"{at}.illumination"
    if illumination.wavelength is None:
        raise ConfigurationError(f"{field}.wavelength", "the illumination wavelength is required")
    if illumination.numerical_aperture is None:
        raise ConfigurationError(f"{field}.numerical_aperture", "the condenser numerical aperture is required")
    if detection.pinhole != "none":
        raise UnsupportedOptionError(
            f"{at}.detection.pinhole", "confocal", reason="confocal detection needs contrast: fluorescence"
        )
    if detection.wavelength not in (None, illumination.wavelength):
        raise ConfigurationError(
            f"{at}.detection.wavelength",
            f"is {detection.wavelength} um, but brightfield detects the illumination's {illumination.wavelength} um; "
            "omit it",
        )
    tilt = _resolve(
        "tilt",
        "illumination_tilt",
        {
            "zenith": _quantity(illumination.tilt.zenith, at=f"{field}.tilt.zenith"),
            "azimuth": _quantity(illumination.tilt.azimuth, at=f"{field}.tilt.azimuth"),
        },
        at=f"{field}.tilt",
    )
    arm = _resolve(
        "illumination",
        "illumination",
        {
            "wavelength": illumination.wavelength,
            "numerical_aperture": _quantity(illumination.numerical_aperture, at=f"{field}.numerical_aperture"),
            "tilt": tilt,
        },
        at=field,
    )
    arguments = {
        "illumination": arm,
        "detection": _detection(detection, None),
        "index_of_refraction_media": index,
        "invert_phase_contrast": illumination.invert_phase_contrast,
    }
    return _resolve("system", "thin_phase_2d" if thin else "thick_phase_3d", arguments, at=at)


# --- the object prior and the noise ----------------------------------------------------------------------------------


def _prior(config: V4Config) -> ObjectPrior:
    settings = config.object
    prior, at = settings.prior, "object.prior"
    nonnegative = "nonneg" in settings.constraints
    weight: object = settings.weight
    if not isinstance(weight, float | int):  # auto: free; {auto: {rule}}: free for the sweep stage, which sets it
        weight = _AUTO_WEIGHT if weight == "auto" else Free(1e-3, bounds=(1e-12, 1e12))
    if isinstance(prior, PriorBeads):
        return _resolve("object_prior", "beads", prior.beads.model_dump(), at=f"{at}.beads")
    if prior == "none":
        if "weight" in settings.model_fields_set:
            raise ConfigurationError("object.weight", "is set, but object.prior: none has no weight; remove it")
        return _resolve("object_prior", "none", {"nonnegative": nonnegative}, at=at)
    return _resolve("object_prior", prior, {"weight": weight, "nonnegative": nonnegative}, at="object")


def _noise(config: V4Config) -> Noise:
    noise = config.noise
    name = noise.kind
    settings = getattr(noise, name)
    at = f"noise.{name}"
    arguments = settings.model_dump(exclude={"snr_db", "photons"})
    sigma = getattr(settings, "sigma", None)
    if sigma == "estimate" and name == "gaussian":
        # Estimated with the evidence or by beads; elsewhere sigma cancels out of the solve and the score.
        calibration = config.solver.calibration
        estimated = isinstance(config.object.prior, PriorBeads) or (
            calibration is not None and calibration.score == "evidence"
        )
        arguments["sigma"] = "estimate" if estimated else 1.0
    elif sigma == "estimate":
        raise UnsupportedOptionError(
            f"{at}.sigma", "estimate", reason="only noise.gaussian.sigma is estimated; give sigma as a number"
        )
    return _resolve("noise", name, arguments, at=at)


# --- the solver, the score and the search ----------------------------------------------------------------------------


def _loop(options: LoopOptions) -> Loop:
    fields = ("tolerance", "check_every")
    return Loop(options.iterations, **{name: getattr(options, name) for name in fields})


def _solver(config: V4Config) -> Solver:
    reconstruction, at = config.solver.reconstruction, "solver.reconstruction"
    if isinstance(reconstruction, ReconstructionFista):
        return _resolve("solver", "fista", {"loop": _loop(reconstruction.fista)}, at=f"{at}.fista")
    if isinstance(reconstruction, ReconstructionAdmm):
        options = reconstruction.admm
        return _resolve("solver", "admm", {"loop": Loop(options.iterations), "rho": options.rho}, at=f"{at}.admm")
    if isinstance(reconstruction, ReconstructionRichardsonLucy):
        options = reconstruction.richardson_lucy
        names = ("background", "back_projector", "bp_alpha", "bp_beta", "gradient_consensus", "start")
        arguments = {"loop": _loop(options), **{name: getattr(options, name) for name in names}}
        return _resolve("solver", "richardson_lucy", arguments, at=f"{at}.richardson_lucy")
    if isinstance(reconstruction, ReconstructionDistilled):
        options = reconstruction.distilled
        field = f"{at}.distilled.bundle"
        if not Path(options.bundle).is_dir():
            raise ConfigurationError(
                field, f"{options.bundle} is not a directory; give the bundle wo distill fit wrote"
            )
        try:
            return Distilled(load_bundle(options.bundle), precision=options.precision)
        except (BundleError, ParameterError) as error:
            raise ConfigurationError(field, str(error)) from error
    return _resolve("solver", "wiener", {}, at=at)


def _score(config: V4Config) -> Score:
    calibration = config.solver.calibration
    assert calibration is not None
    at = "solver.calibration.score"
    loss_at = "solver.calibration.advanced.loss"
    loss = None if calibration.advanced is None else calibration.advanced.loss
    if calibration.score != "sharpness":
        if loss is not None:
            raise ConfigurationError(loss_at, f"is a sharpness setting, and score is {calibration.score}; remove it")
        return _resolve("score", calibration.score, {}, at=at)
    settings = dict(loss or {"type": "midband_power"})
    kind = settings.pop("type", "midband_power")
    if kind not in _SHARPNESS:
        raise ConfigurationError(
            f"{loss_at}.type", f"{kind!r} is not a 3.x sharpness loss; use one of {list(_SHARPNESS)}"
        )
    arguments: dict[str, object] = {"fields": ("phase",) if config.imaging_system.contrast == "phase" else None}
    if kind in ("midband_power", "spectral_flatness"):
        arguments["fractions"] = tuple(settings.pop("midband_fractions", (0.125, 0.25)))
    if settings:
        raise ConfigurationError(
            f"{loss_at}.{next(iter(settings))}", f"is not a setting of loss type {kind}; remove it"
        )
    keys = {"fractions": f"{loss_at}.midband_fractions"}
    return _resolve("score", _SHARPNESS[kind], arguments, at=loss_at, keys=keys)


def _keyed(value: object, *, at: str, config: V4Config) -> object:
    """Return a per-quantity setting keyed by parameter names instead of the paths the configuration writes."""
    if not isinstance(value, Mapping):
        return value
    paths = free_parameter_specs(config.imaging_system, expand_shorthand=True)
    keyed = {}
    for key, entry in value.items():
        if key not in paths:
            prefix, _, term = key.rpartition(".")
            try:
                spelled = f"{prefix}.{canonical_name(term)}" if prefix.endswith("pupil_aberrations") else None
            except ValueError:
                spelled = None
            hint = f"write {spelled!r}" if spelled in paths else f"the free quantities are {sorted(paths)}"
            raise ConfigurationError(f"{at}.{key}", f"names no free quantity of imaging_system; {hint}")
        keyed[parameter_name(key)] = entry
    return keyed


def _profile(prior: ObjectPrior | None, noise: Noise | None) -> tuple[str, ...]:
    """Return the free noise sigma and object weight, which wo-4.0 grids over before the other quantities."""
    names = [f"noise.{name}" for name in free_specs(noise) if name == "sigma"] if noise is not None else []
    if prior is not None:
        names += [f"object_prior.{name}" for name in free_specs(prior) if name == "weight"]
    return tuple(names)


def _optimizer_stage(advanced: CalibrationAdvanced, rates: object) -> SearchStage:
    """Return the one stage a 3.x ``advanced.optimizer`` names, with the 3.x setting it reads."""
    at = "solver.calibration.advanced"
    optimizer = advanced.optimizer
    if optimizer in ("adam", "nadam"):
        arguments = {"learning_rates": rates, "optimizer": optimizer, "learning_rate": 0.01, "last": True}
        return _resolve(
            "search", "gradient", arguments, at=f"{at}.optimizer", keys={"learning_rates": f"{at}.learning_rates"}
        )
    if optimizer == "lbfgs":
        return _resolve("search", "lbfgs", {}, at=f"{at}.optimizer")
    if optimizer == "nelder_mead":
        arguments = {} if advanced.convergence_tol is None else {"tolerance": advanced.convergence_tol}
        return _resolve(
            "search", "nelder_mead", arguments, at=f"{at}.optimizer", keys={"tolerance": f"{at}.convergence_tol"}
        )
    arguments = {"width": 2.0} if advanced.grid_points is None else {"points": advanced.grid_points, "width": 2.0}
    return _resolve("search", "grid", arguments, at=f"{at}.optimizer", keys={"points": f"{at}.grid_points"})


def _search(config: V4Config, *, profile: tuple[str, ...]) -> tuple[SearchStage, ...]:
    """Build the search stages; with ``profile``, a grid over those quantities goes first and before each gradient.

    Gradient stages take wo-4.0's step (``learning_rate=0.01``) and end at their last point (``last=True``), and
    grids span wo-4.0's width of two tolerances, so a migrated file calibrates as wo-4.0 did.
    """
    calibration = config.solver.calibration
    assert calibration is not None
    advanced = calibration.advanced or CalibrationAdvanced()
    at = "solver.calibration.advanced"
    rate_at = f"{at}.learning_rates"
    rates = _keyed(advanced.learning_rates, at=rate_at, config=config)
    reads = {"nelder_mead": "convergence_tol", "grid_search": "grid_points"}
    for name in ("convergence_tol", "grid_points"):
        if getattr(advanced, name) is not None and reads.get(advanced.optimizer or "") != name:
            raise UnsupportedOptionError(
                f"{at}.{name}", str(getattr(advanced, name)), reason="it is not run; remove it"
            )
    if advanced.optimizer is not None:
        return (_optimizer_stage(advanced, rates),)
    grid = None
    if profile:
        grid = _resolve("search", "grid", {"points": 13, "names": profile, "width": 2.0}, at="solver.calibration")
    stages: list[SearchStage] = [] if grid is None else [grid]
    for index, stage in enumerate(calibration.search):
        if grid is not None and isinstance(stage, SearchGradient) and stages[-1] is not grid:
            stages.append(grid)
        stages.append(_stage(stage, at=f"solver.calibration.search.{index}", rates=rates, config=config))
    return tuple(stages)


def _stage(stage: object, *, at: str, rates: object, config: V4Config) -> SearchStage:
    """Build one entry of ``solver.calibration.search``."""
    if isinstance(stage, SearchGrid):
        points = _keyed(stage.grid.points, at=f"{at}.grid.points", config=config)
        return _resolve("search", "grid", {"points": points, "width": 2.0}, at=f"{at}.grid")
    if isinstance(stage, SearchGradient):
        options = stage.gradient
        arguments = {
            "iterations": options.max_iter,
            "learning_rates": rates,
            "optimizer": options.optimizer,
            "learning_rate": 0.01,
            "last": True,
        }
        keys = {"iterations": f"{at}.gradient.max_iter", "learning_rates": "solver.calibration.advanced.learning_rates"}
        return _resolve("search", "gradient", arguments, at=f"{at}.gradient", keys=keys)
    assert isinstance(stage, SearchRandom | SearchLBFGS | SearchNelderMead)
    ((name, options),) = stage.model_dump().items()
    return _resolve("search", name, options, at=f"{at}.{name}")
