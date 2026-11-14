"""A configuration builds the registered parts it writes, with its arguments, and is refused at the config path."""

import copy
from typing import Any

import pytest
import torch
from tests.registration import temporarily

from waveorder import registry
from waveorder.config import V4Config, load
from waveorder.config.build import build
from waveorder.errors import ConfigurationError, ConfigurationWarning, Rejected, UnsupportedOptionError
from waveorder.models import ReconstructionModel
from waveorder.models._stores import system_record, system_sha256
from waveorder.nn import (
    FISTA,
    CalibrationResult,
    Free,
    Gaussian,
    NoPrior,
    Poisson,
    PoissonGaussian,
    Quantity,
    RichardsonLucy,
    Sampling,
    ShiftInvariantSystem,
    Tikhonov,
    TotalVariation,
    Wiener,
    free_specs,
    load_parameters,
)
from waveorder.nn.functional import half_spectrum_frequencies
from waveorder.nn.solvers import Loop
from waveorder.tiling import TilePlan
from waveorder.zernike_names import noll_indices_up_to_order, noll_to_name

PIXEL_SIZE = (0.5, 0.1, 0.1)
BASE: dict[str, Any] = {
    "version": 4,
    "data": {"input_channel_names": ["c"], "pixel_size": {"z": 0.5, "y": 0.1, "x": 0.1}},
    "imaging_system": {
        "contrast": "fluorescence",
        "index_of_refraction_media": 1.33,
        "detection": {"wavelength": 0.52, "numerical_aperture": 1.2},
    },
}
PHASE: dict[str, Any] = {
    "contrast": "phase",
    "index_of_refraction_media": 1.33,
    "detection": {"numerical_aperture": 0.9},
    "illumination": {"wavelength": 0.532, "numerical_aperture": 0.5},
}
FREE_NA = {"nominal": 1.2, "tolerance": 0.05, "bounds": [1.0, 1.3]}
NA = "system.detection.numerical_aperture"


def _raw(*, detection: dict | None = None, imaging_system: dict | None = None, **blocks: object) -> dict:
    """BASE with ``detection`` merged into its detection arm, and the other blocks replaced."""
    raw = copy.deepcopy(BASE)
    if imaging_system is not None:
        raw["imaging_system"] = copy.deepcopy(imaging_system)
    raw["imaging_system"]["detection"].update(detection or {})
    raw.update(copy.deepcopy(blocks))
    return raw


def _free(**blocks: Any) -> dict:
    """BASE with a free aperture."""
    return _raw(detection={"numerical_aperture": FREE_NA}, **blocks)


def _loss(score: str, **loss: object) -> dict:
    """BASE with a free aperture, calibrated by ``score`` with the 3.x ``loss`` settings."""
    return _free(solver={"calibration": {"score": score, "advanced": {"loss": loss}}})


def _phase(**illumination: object) -> dict:
    system = copy.deepcopy(PHASE)
    system["illumination"].update(illumination)
    return _raw(imaging_system=system, data={"input_channel_names": ["BF"], "pixel_size": BASE["data"]["pixel_size"]})


def _parts(raw: dict) -> dict[str, Any]:
    system, parts = build(V4Config(**raw))
    return {"system": system, **parts}


def _spec(part: object) -> tuple[str, dict[str, object]]:
    ((name, arguments),) = registry.describe(part).items()
    return name, arguments


def _refusal(raw: dict) -> str:
    """Return the message of the one problem ``from_config`` refuses ``raw`` with."""
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(raw)
    (problem,) = rejected.value.problems
    assert problem.location == "the configuration"
    return problem.message


def _no_effect(raw: dict) -> dict[str, str]:
    """Return the located warnings ``from_config`` gives ``raw``, message by location; every warning is located."""
    with pytest.warns(ConfigurationWarning) as caught:
        ReconstructionModel.from_config(raw)
    located = [warning.message for warning in caught if isinstance(warning.message, ConfigurationWarning)]
    assert len(located) == len(caught)
    return {str(warning.location): str(warning) for warning in located}


# --- the imaging system ----------------------------------------------------------------------------------------------


DETECTION = {
    "wavelength": 0.52,
    "numerical_aperture": 1.2,
    "z_focus_offset": 0.0,
    "edge": "ramp",
    "pupil_aberrations": None,
}
ILLUMINATION = {
    "illumination": {
        "wavelength": 0.532,
        "numerical_aperture": 0.5,
        "tilt": {"illumination_tilt": {"zenith": 0.0, "azimuth": 0.0}},
    }
}


@pytest.mark.parametrize(
    ("raw", "name", "arguments"),
    [
        (_raw(), "widefield_fluorescence_3d", {"detection": {"detection": DETECTION}}),
        (_raw(object={"dimension": 2}), "thin_fluorescence_2d", {"detection": {"detection": DETECTION}}),
        (
            _raw(detection={"pinhole": {"diameter_um": 0.3}, "edge": "sigmoid", "z_focus_offset": 1.0}),
            "confocal_fluorescence_3d",
            {
                "detection": {"detection": DETECTION | {"edge": "sigmoid", "z_focus_offset": 1.0}},
                "pinhole_diameter": 0.3,
            },
        ),
        (_raw(detection={"pinhole": "point"}), "confocal_fluorescence_3d", {"pinhole_diameter": 0.0}),
        (
            _phase(),
            "thick_phase_3d",
            {
                "illumination": ILLUMINATION,
                # the phase system's detection holds the illumination's wavelength, and is described with it
                "detection": {"detection": DETECTION | {"wavelength": 0.532, "numerical_aperture": 0.9}},
                "invert_phase_contrast": False,
            },
        ),
        (
            _phase(invert_phase_contrast=True) | {"object": {"dimension": 2}},
            "thin_phase_2d",
            {"invert_phase_contrast": True},
        ),
    ],
    ids=["widefield", "thin_fluorescence", "confocal", "point_pinhole", "thick_phase", "thin_phase"],
)
def test_each_system_builds_its_class_with_the_configured_arguments(raw, name, arguments):
    built, given = _spec(_parts(raw)["system"])
    assert built == name
    assert given["index_of_refraction_media"] == 1.33
    assert {key: given[key] for key in arguments} == arguments


def test_a_free_quantity_is_free_with_its_nominal_tolerance_and_bounds():
    tilt = {"zenith": {"nominal": 0.1, "tolerance": 0.05, "bounds": [0.0, 0.3]}, "azimuth": 0.5}
    specs = free_specs(_parts(_phase(numerical_aperture={"nominal": 0.45, "bounds": [0.3, 0.7]}, tilt=tilt))["system"])
    assert specs["illumination.numerical_aperture"] == Free(0.45, bounds=(0.3, 0.7))
    assert specs["illumination.tilt.zenith"] == Free(0.1, tolerance=0.05, bounds=(0.0, 0.3))
    assert set(specs) == {"illumination.numerical_aperture", "illumination.tilt.zenith"}


def test_the_zernike_shorthand_frees_each_term_and_named_terms_override_it():
    aberrations = {
        "zernike_up_to_order": {"order": 4, "tolerance": 0.05},
        "spherical": {"nominal": 0.05, "tolerance": 0.02},
        "z12": 0.01,  # by Noll index: fixed
    }
    specs = free_specs(_parts(_raw(detection={"pupil_aberrations": aberrations}))["system"])
    prefix = "detection.pupil_aberrations"
    named = {f"{prefix}.{noll_to_name(j)}" for j in noll_indices_up_to_order(4)}
    assert set(specs) == named - {f"{prefix}.z12"}
    assert specs[f"{prefix}.spherical"] == Free(0.05, tolerance=0.02)
    assert specs[f"{prefix}.coma_x"] == Free(0.0, tolerance=0.05)


def test_an_airy_unit_pinhole_is_converted_at_the_nominal_aperture():
    unit = 1.22 * 0.52 / 1.1  # the aperture's nominal value, not where a calibration moves it
    fixed = _raw(
        detection={"numerical_aperture": {"nominal": 1.1, "bounds": [1.0, 1.3]}, "pinhole": {"diameter_au": 1}}
    )
    assert _spec(_parts(fixed)["system"])[1]["pinhole_diameter"] == pytest.approx(unit)
    free = {"diameter_au": {"nominal": 1.0, "tolerance": 0.2, "bounds": [0.5, 2.0]}}
    spec = free_specs(_parts(_raw(detection={"numerical_aperture": 1.1, "pinhole": free}))["system"])[
        "pinhole_diameter"
    ]
    assert (spec.nominal, spec.tolerance) == pytest.approx((unit, 0.2 * unit))
    assert spec.bounds == pytest.approx((0.5 * unit, 2.0 * unit))


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        (_raw(detection={"pinhole": {"diameter_um": -0.5}}), "imaging_system.detection.pinhole.diameter_um"),
        (
            _raw(detection={"pinhole": {"diameter_au": {"nominal": 1.0, "bounds": [-0.1, 2.0]}}}),
            "imaging_system.detection.pinhole.diameter_au",
        ),
        (_phase(tilt={"zenith": 0.8}), "imaging_system.illumination.tilt.zenith"),
        (
            _loss("sharpness", midband_fractions=[0.3, 0.1]),
            "solver.calibration.advanced.loss.midband_fractions",
        ),
    ],
    ids=["negative_um", "negative_au", "darkfield_zenith", "fractions"],
)
def test_a_constructor_error_is_located_at_the_config_path_of_its_argument(raw, field):
    with pytest.raises(ConfigurationError) as error:
        build(V4Config(**raw))
    assert error.value.field == field
    assert _refusal(raw) == str(error.value)
    assert str(error.value).startswith(f"{field}: ")


@pytest.mark.parametrize(
    ("raw", "path", "bounds"),
    [
        (_raw(detection={"numerical_aperture": {"nominal": 1.2}}), "detection.numerical_aperture", (0.01, 1.33)),
        (_phase(tilt={"zenith": {"nominal": 0.1, "tolerance": 0.05}}), "illumination.tilt.zenith", (-0.743, 0.743)),
    ],
    ids=["unbounded_aperture", "free_zenith"],
)
def test_a_free_value_without_the_range_its_part_needs_loads_with_the_range_a_migration_gives(raw, path, bounds):
    # The parts refuse it; wo-4.0 took it, so a loaded configuration gets the range wo config migrate writes.
    field = f"imaging_system.{path}"
    with pytest.raises(ConfigurationError) as error:
        build(V4Config(**raw))
    assert error.value.field == field
    with pytest.warns(ConfigurationWarning, match="^is free without bounds; 4.x needs") as caught:
        model = ReconstructionModel.from_config(raw)
    assert [getattr(warning.message, "location", None) for warning in caught] == [field]
    assert free_specs(model.imaging_model)[f"system.{path}"].bounds == pytest.approx(bounds)


# --- the object prior, the noise and the solver ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "name", "arguments"),
    [
        ({"prior": "tikhonov", "weight": 0.01}, "tikhonov", {"weight": 0.01, "nonnegative": False}),
        ({"prior": "tv", "weight": 0.1, "constraints": ["nonneg"]}, "tv", {"weight": 0.1, "nonnegative": True}),
        ({"prior": "none"}, "none", {"nonnegative": False}),
        ({"prior": {"beads": {"diameter": 0.2}}}, "beads", {"diameter": 0.2}),
    ],
    ids=["tikhonov", "tv_nonneg", "none", "beads"],
)
def test_each_prior_builds_its_class_with_the_configured_arguments(block, name, arguments):
    built, given = _spec(_parts(_free(object=block))["object_prior"])
    assert built == name
    assert {key: given[key] for key in arguments} == arguments


def test_weight_auto_is_a_free_weight_calibrated_with_the_system():
    parts = _parts(_free(object={"weight": "auto"}, solver={"calibration": {"score": "evidence"}}))
    weight = free_specs(parts["object_prior"])["weight"]
    assert weight.nominal == 1e-3
    assert weight.bounds is not None


@pytest.mark.parametrize(
    ("block", "name", "arguments"),
    [
        ({"gaussian": {"sigma": 2.0}}, "gaussian", {"sigma": 2.0}),
        ({"poisson": {"gain": 2.0, "offset": 100.0}}, "poisson", {"gain": 2.0, "offset": 100.0}),
        ({"poisson_gaussian": {"gain": 2.0, "offset": 1.0, "sigma": 3.0}}, "poisson_gaussian", {"sigma": 3.0}),
        ({"gaussian": {"sigma": "estimate"}}, "gaussian", {"sigma": 1.0}),  # nothing estimates it: it cancels out
    ],
    ids=["gaussian", "poisson", "poisson_gaussian", "unestimated_sigma"],
)
def test_each_noise_builds_its_class_with_the_configured_arguments(block, name, arguments):
    built, given = _spec(_parts(_raw(noise=block, object={"prior": "none"}))["noise"])
    assert built == name
    assert {key: given[key] for key in arguments} == arguments


def test_a_sigma_to_estimate_is_free_under_the_evidence():
    parts = _parts(_free(noise={"gaussian": {"sigma": "estimate"}}, solver={"calibration": {"score": "evidence"}}))
    assert "sigma" in free_specs(parts["noise"])


LOOP = Loop(9, tolerance=1e-3, check_every=3)
LOOP_SETTINGS = {name: getattr(LOOP, name) for name in ("iterations", "tolerance", "check_every")}
RICHARDSON_LUCY = {"background": 1.0, "back_projector": "gaussian", "start": "ones"}
SHORT = Loop(3, tolerance=1e-3, check_every=3)


@pytest.mark.parametrize(
    ("reconstruction", "blocks", "name", "loop", "arguments"),
    [
        ("closed_form", {}, "wiener", None, {}),
        ("fista", {}, "fista", Loop(FISTA().loop.iterations), {}),
        ({"fista": LOOP_SETTINGS}, {}, "fista", LOOP, {}),
        ({"admm": {"iterations": 3, "rho": 0.01}}, {"object": {"prior": "tv"}}, "admm", Loop(3), {"rho": 0.01}),
        (
            {"richardson_lucy": LOOP_SETTINGS | {"iterations": 3} | RICHARDSON_LUCY},
            {"object": {"prior": "none"}, "noise": {"poisson": {}}},
            "richardson_lucy",
            SHORT,
            RICHARDSON_LUCY,
        ),
    ],
    ids=["closed_form", "bare_fista", "fista", "admm", "richardson_lucy"],
)
def test_each_solver_builds_its_class_with_the_configured_arguments(reconstruction, blocks, name, loop, arguments):
    solver = _parts(_raw(solver={"reconstruction": reconstruction}, **blocks))["solver"]
    built, given = _spec(solver)
    assert built == name
    assert getattr(solver, "loop", None) == loop
    assert {key: given[key] for key in arguments} == arguments


# --- the score and the search ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("calibration", "name", "arguments"),
    [
        ({"score": "evidence"}, "evidence", {}),
        ({"score": "gcv"}, "gcv", {}),
        ({"score": "sharpness"}, "midband_power_score", {"fractions": [0.125, 0.25], "fields": None}),
        *(
            ({"score": "sharpness", "advanced": {"loss": {"type": loss}}}, f"{loss}_score", {"fields": None})
            for loss in ("midband_power", "spectral_flatness", "total_variation", "laplacian_variance")
        ),
        (
            {"score": "sharpness", "advanced": {"loss": {"type": "normalized_variance"}}},
            "normalized_variance_score",
            {},
        ),
        (
            {
                "score": "sharpness",
                "advanced": {"loss": {"type": "spectral_flatness", "midband_fractions": [0.1, 0.3]}},
            },
            "spectral_flatness_score",
            {"fractions": [0.1, 0.3]},
        ),
    ],
)
def test_each_score_builds_its_class_with_the_configured_arguments(calibration, name, arguments):
    built, given = _spec(_parts(_free(solver={"calibration": calibration}))["score"])
    assert built == name
    assert {key: given[key] for key in arguments} == arguments


def test_a_sharpness_score_of_a_phase_system_scores_the_phase_field():
    raw = _phase(numerical_aperture={"nominal": 0.5, "bounds": [0.3, 0.7]}) | {
        "solver": {"calibration": {"score": "sharpness"}}
    }
    assert _spec(_parts(raw)["score"])[1]["fields"] == ["phase"]


def _stages(**calibration: object) -> list[tuple[str, dict[str, object]]]:
    stages = _parts(_free(solver={"calibration": calibration}))["search"]
    return [_spec(stage) for stage in stages]


def test_each_search_stage_builds_its_class_in_the_configured_order():
    search = [
        {"grid": {"points": 3}},
        {"random": {"samples": 4, "seed": 2}},
        {"gradient": {"max_iter": 4, "optimizer": "nadam"}},
        {"lbfgs": {"iterations": 5, "history": 4}},
        {"nelder_mead": {"iterations": 9, "tolerance": 1e-3}},
    ]
    (grid, random, gradient, lbfgs, nelder_mead) = _stages(search=search)
    assert grid[0] == "grid"
    assert grid[1]["points"] == 3
    assert random == ("random", random[1] | {"samples": 4, "seed": 2})
    assert gradient[0] == "gradient"
    assert (gradient[1]["iterations"], gradient[1]["optimizer"]) == (4, "nadam")
    assert lbfgs == ("lbfgs", {"iterations": 5, "history": 4})
    assert nelder_mead == ("nelder_mead", {"iterations": 9, "tolerance": 1e-3})


@pytest.mark.parametrize(
    ("advanced", "name", "arguments"),
    [
        ({"optimizer": "adam"}, "gradient", {"optimizer": "adam", "iterations": 200}),
        (
            {"optimizer": "nadam", "learning_rates": {"detection.numerical_aperture": 0.1}},
            "gradient",
            {"optimizer": "nadam", "learning_rates": {NA: 0.1}},
        ),
        ({"optimizer": "lbfgs"}, "lbfgs", {"iterations": 200}),
        ({"optimizer": "nelder_mead", "convergence_tol": 1e-3}, "nelder_mead", {"tolerance": 1e-3, "iterations": 200}),
        ({"optimizer": "grid_search", "grid_points": 5}, "grid", {"points": 5}),
    ],
    ids=["adam", "nadam", "lbfgs", "nelder_mead", "grid_search"],
)
def test_a_3x_optimizer_is_the_one_stage_it_names(advanced, name, arguments):
    # Its budget is the max_iter of the default lone gradient stage, 200, as wo-4.0 read it.
    ((built, given),) = _stages(score="sharpness", advanced=advanced)
    assert built == name
    assert {key: given[key] for key in arguments} == arguments


@pytest.mark.parametrize("optimizer", ["adam", "nadam", "lbfgs", "nelder_mead"])
def test_wo_40_optimizer_beside_its_gradient_stage_runs_the_optimizer_for_the_stage_max_iter(optimizer):
    # wo-4.0's wo config migrate writes both; its sharpness calibration ran the optimizer for search.0's max_iter.
    raw = _free(
        solver={
            "calibration": {
                "score": "sharpness",
                "search": [{"gradient": {"max_iter": 50, "method": "split"}}],
                "advanced": {"optimizer": optimizer, "learning_rates": {"detection.numerical_aperture": 0.05}},
            }
        }
    )
    ((built, given),) = [_spec(stage) for stage in _parts(raw)["search"]]
    assert (built, given["iterations"]) == ({"adam": "gradient", "nadam": "gradient"}.get(optimizer, optimizer), 50)
    if built == "gradient":
        assert (given["optimizer"], given["learning_rates"]) == (optimizer, {NA: 0.05})


def test_wo_40_optimizer_beside_a_search_it_does_not_read_warns_at_what_has_no_effect():
    sharpness = {"score": "sharpness", "advanced": {"optimizer": "lbfgs"}}
    stages = _free(solver={"calibration": sharpness | {"search": [{"grid": {"points": 3}}, "gradient"]}})
    stages["imaging_system"]["detection"]["numerical_aperture"] = FREE_NA | {"tolerance": "free"}
    assert list(_no_effect(stages)) == ["solver.calibration.search"]
    with pytest.warns(ConfigurationWarning):
        assert [_spec(stage)[0] for stage in _parts(stages)["search"]] == ["lbfgs"]
    other = _free(solver={"calibration": sharpness | {"search": [{"gradient": {"optimizer": "adamw"}}]}})
    assert list(_no_effect(other)) == ["solver.calibration.search.0.gradient.optimizer"]
    evidence = _free(solver={"calibration": {"search": ["lbfgs"], "advanced": {"optimizer": "adam"}}})
    assert list(_no_effect(evidence)) == ["solver.calibration.advanced.optimizer"]
    with pytest.warns(ConfigurationWarning):
        assert [_spec(stage)[0] for stage in _parts(evidence)["search"]] == ["lbfgs"]  # evidence runs the search
    consistent = {"search": [{"gradient": {"max_iter": 7}}], "advanced": {"optimizer": "adam"}}
    ((_, given),) = _stages(**consistent)  # adam describes the adam stage: accepted silently
    assert given["iterations"] == 7


def test_name_keyed_settings_reach_the_stages_under_the_parameter_names():
    path = "detection.numerical_aperture"
    grid, gradient = _stages(
        search=[{"grid": {"points": {path: 3}}}, "gradient"], advanced={"learning_rates": {path: 0.1}}
    )
    assert grid[1]["points"] == {NA: 3}
    assert gradient[1]["learning_rates"] == {NA: 0.1}


@pytest.mark.parametrize(
    ("calibration", "field", "hint"),
    [
        (
            {"search": [{"grid": {"points": {"detection.pupil_aberrations.z11": 3}}}]},
            "solver.calibration.search.0.grid.points.detection.pupil_aberrations.z11",
            "write 'detection.pupil_aberrations.spherical'",
        ),
        (
            {"advanced": {"learning_rates": {"detection.numerical_apertur": 0.1}}},
            "solver.calibration.advanced.learning_rates.detection.numerical_apertur",
            "the free quantities are",
        ),
    ],
    ids=["noll_spelling", "typo"],
)
def test_a_name_keyed_setting_of_no_free_quantity_is_dropped_with_a_warning_at_its_key(calibration, field, hint):
    # wo-4.0 read these mappings by the free quantities' names only, so another key changed nothing.
    spherical = {"spherical": {"nominal": 0.0, "bounds": [-0.2, 0.2]}}
    raw = _free(solver={"calibration": {"score": "evidence", **calibration}})
    raw["imaging_system"]["detection"]["pupil_aberrations"] = spherical
    warned = _no_effect(raw)
    assert list(warned) == [field]
    assert "names no free quantity of imaging_system; " in warned[field]
    assert hint in warned[field]


def test_no_score_or_search_is_built_without_a_free_quantity_and_a_calibration_block():
    assert (lambda parts: (parts["score"], parts["search"]))(_parts(_free())) == (None, None)
    assert (lambda parts: (parts["score"], parts["search"]))(_parts(_raw(solver={"calibration": {}}))) == (None, None)


# --- settings nothing runs or reads ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "field"),
    [(_raw(detection={"pinhole": "point"}, object={"dimension": 2}), "imaging_system.detection.pinhole")],
)
def test_a_setting_nothing_runs_is_refused_with_its_reason(raw, field):
    with pytest.raises(UnsupportedOptionError) as error:
        build(V4Config(**raw))
    assert error.value.field == field
    assert error.value.reason
    assert _refusal(raw) == str(error.value)


SHARP = {"score": "sharpness"}


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        (_raw(solver={"padding": {"z": 2}}, object={"dimension": 2}), "solver.padding.z"),
        (
            _free(solver={"calibration": SHARP | {"advanced": {"optimizer": "adam", "convergence_tol": 1e-3}}}),
            "solver.calibration.advanced.convergence_tol",
        ),
        (
            _free(solver={"calibration": SHARP | {"advanced": {"optimizer": "lbfgs", "grid_points": 5}}}),
            "solver.calibration.advanced.grid_points",
        ),
        (
            _raw(noise={"poisson_gaussian": {"sigma": "estimate"}}, object={"prior": "none"}),
            "noise.poisson_gaussian.sigma",
        ),
        (_raw(outputs=["transfer_function"]), "outputs"),
        (_raw(outputs=["reconstruction", "psf", "tile_map"]), "outputs.1"),
        (_raw(object={"prior": {"denoiser": {"checkpoint": "x.pt"}}}), "object.prior.denoiser"),
        (_raw(noise={"gaussian": {"sigma": {"dark_frames": "dark.zarr"}}}), "noise.gaussian.sigma.dark_frames"),
        (_raw(object={"source": {"phantom": {"kind": "beads"}}}), "object.source.phantom"),
        (_raw(object={"source": {"from_recon": {"path": "r.zarr"}}}), "object.source.from_recon"),
        (_raw(solver={"padding": {"yx": [4, 4]}}), "solver.padding.yx"),
        (_raw(noise={"gaussian": {"snr_db": 20.0}}), "noise.gaussian.snr_db"),
        (
            _free(solver={"calibration": {"final_estimate": "average_near_best_H"}}),
            "solver.calibration.final_estimate",
        ),
        (
            _free(solver={"calibration": {"search": [{"gradient": {"method": "joint"}}]}}),
            "solver.calibration.search.0.gradient.method",
        ),
    ],
    ids=[
        "thin_padding",
        "tol_not_read",
        "points_not_read",
        "poisson_gaussian_estimate",
        "no_reconstruction",
        "unwritten_output",
        "denoiser",
        "dark_frames",
        "phantom",
        "from_recon",
        "yx_padding",
        "snr_db_outside_a_simulation",
        "average_near_best_H",
        "joint_gradient",
    ],
)
def test_a_wo_40_setting_nothing_runs_is_accepted_with_a_warning_at_its_key(raw, field):
    warned = _no_effect(raw)
    assert field in warned
    assert "has no effect: " in warned[field]


def test_wo_40_placeholders_build_what_runs_without_them():
    with pytest.warns(ConfigurationWarning, match="denoiser"):
        parts = _parts(_raw(object={"prior": {"denoiser": {"checkpoint": "x.pt"}}, "weight": 0.1}))
    assert isinstance(parts["object_prior"], NoPrior)
    with pytest.warns(ConfigurationWarning, match="dark frames"):
        parts = _parts(_raw(noise={"gaussian": {"sigma": {"dark_frames": "dark.zarr"}}}))
    assert parts["noise"].sigma == 1.0  # the default
    with pytest.warns(ConfigurationWarning, match="nothing to pad"):
        assert _parts(_raw(solver={"padding": {"z": 2}}, object={"dimension": 2}))["z_padding"] == 0


def test_an_output_that_is_no_wo_40_output_is_refused_at_load():
    with pytest.raises(Rejected) as error:
        load(_raw(outputs=["reconstruction", "score_map"]))
    (problem,) = error.value.problems
    assert problem.message.startswith("outputs.1: ")
    assert "not available in this version of WaveOrder" in problem.message


def test_a_score_grid_without_a_grid_and_score_arrays_of_a_tiled_field_are_accepted_with_a_warning():
    outputs = ["reconstruction", "score_trace", "score_grid"]
    assert (
        load(_free(outputs=outputs, solver={"calibration": {"search": [{"grid": {"points": 5}}]}})).outputs == outputs
    )
    load(_free(outputs=outputs, solver={"calibration": SHARP | {"advanced": {"optimizer": "grid_search"}}}))
    for solver in ({"calibration": {"search": [{"gradient": {}}]}}, {}):
        with pytest.warns(ConfigurationWarning, match="'score_grid' has no effect: the calibration runs no grid"):
            load(_free(outputs=outputs, solver=solver))
    tiled = BASE["imaging_system"] | {"field_dependence": {"tiled": {"size_yx": [16, 16]}}}
    with pytest.warns(ConfigurationWarning, match="a field-tiled calibration keeps no score arrays") as caught:
        load(_free(outputs=["reconstruction", "score_trace"], imaging_system=tiled))
    assert [getattr(warning.message, "location", None) for warning in caught] == ["outputs.1"]


def test_a_rule_based_weight_does_not_run_with_a_calibration():
    raw = _free(object={"weight": {"auto": {"rule": "l_curve"}}}, solver={"calibration": {"score": "evidence"}})
    assert _refusal(raw).startswith("object.weight.auto: sweeps the weight with every other quantity held fixed")


def test_a_bead_calibration_with_an_estimated_sigma_runs_one_gradient_search():
    raw = _free(
        object={"prior": {"beads": {"diameter": 0.2}}},
        noise={"gaussian": {"sigma": "estimate"}},
        solver={"calibration": {"score": "evidence", "search": [{"gradient": {"max_iter": 5}}]}},
    )
    (stage,) = _parts(raw)["search"]
    assert _spec(stage)[0] == "gradient"


UNREAD_ILLUMINATION = _raw(imaging_system=BASE["imaging_system"] | {"illumination": {"wavelength": 0.5}})


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        (_loss("sharpness", type="brenner"), "solver.calibration.advanced.loss.type"),
        (
            _loss("sharpness", type="total_variation", midband_fractions=[0.1, 0.3]),
            "solver.calibration.advanced.loss.midband_fractions",
        ),
    ],
    ids=["unknown_loss", "fractions_of_a_spatial_loss"],
)
def test_a_value_set_where_nothing_reads_it_is_refused_there(raw, field):
    with pytest.raises(ConfigurationError) as error:
        build(V4Config(**raw))
    assert error.value.field == field
    assert _refusal(raw).startswith(f"{field}: ")


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        (UNREAD_ILLUMINATION, "imaging_system.illumination.wavelength"),
        (_raw(object={"prior": "none", "weight": 0.1}), "object.weight"),
        (
            _phase() | {"imaging_system": PHASE | {"detection": {"wavelength": 0.6}}},
            "imaging_system.detection.wavelength",
        ),
        (
            _phase() | {"imaging_system": PHASE | {"detection": {"pinhole": "point"}}},
            "imaging_system.detection.pinhole",
        ),
        (_loss("evidence", type="total_variation"), "solver.calibration.advanced.loss"),
    ],
    ids=[
        "illumination_of_fluorescence",
        "weight_of_no_prior",
        "phase_detection_wavelength",
        "phase_pinhole",
        "loss_of_evidence",
    ],
)
def test_a_value_wo_40_accepted_where_nothing_reads_it_is_accepted_with_a_warning_there(raw, field):
    warned = _no_effect(raw)
    assert list(warned) == [field]
    assert "has no effect: " in warned[field]


def test_the_apodization_rolloff_builds_and_reaches_the_weight_sweep():
    _parts(_raw(solver={"apodization_rolloff": 0.2}))
    raw = _raw(object={"weight": {"auto": {"rule": "l_curve", "num_samples": 5}}}, solver={"apodization_rolloff": 0.2})
    parts = _parts(raw)
    ((name, sweep),) = (_spec(stage) for stage in parts["search"])
    assert name == "weight_sweep"
    assert (sweep["rule"], sweep["samples"], sweep["rolloff"], sweep["keep_slices"]) == ("l_curve", 5, 0.2, False)
    assert list(free_specs(parts["object_prior"])) == ["weight"]  # free, so the sweep stage sets it
    plotted = {"rule": "l_curve", "report_path": "sweep.json", "plot_path": "sweep.png"}
    ((_, sweep),) = (_spec(stage) for stage in _parts(_raw(object={"weight": {"auto": plotted}}))["search"])
    assert sweep["keep_slices"]  # the montage needs the scored slices


# --- what the model class chooses, and what the parts refuse --------------------------------------------------------


MINIMAL = {name: block for name, block in BASE.items() if name in ("version", "data", "imaging_system")}
"""A configuration that writes no object prior, no noise model and no solver."""


@pytest.mark.parametrize(
    ("block", "written"),
    [
        ({}, None),
        ({"object": {"prior": "tikhonov"}}, "object_prior"),
        ({"object": {"weight": 0.01}}, "object_prior"),
        ({"object": {"constraints": ["nonneg"]}}, "object_prior"),
        ({"noise": {}}, "noise"),
        ({"solver": {"reconstruction": "closed_form"}}, "solver"),
        ({"solver": {"padding": {"z": 2}, "device": None}}, None),
    ],
    ids=["nothing", "prior", "weight", "constraints", "empty_noise_block", "reconstruction", "other_solver_settings"],
)
def test_a_part_the_configuration_does_not_write_is_left_to_the_model_class(block, written):
    parts = _parts(MINIMAL | block)
    assert [name for name in ("object_prior", "noise", "solver") if parts[name] is not None] == [written] * bool(
        written
    )


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({}, (Tikhonov, Gaussian, Wiener)),
        ({"object": {"prior": "tv"}}, (TotalVariation, Gaussian, FISTA)),
        ({"noise": {"poisson": {}}}, (NoPrior, Poisson, RichardsonLucy)),
        ({"noise": {"poisson_gaussian": {"sigma": 2.0}}}, (NoPrior, PoissonGaussian, RichardsonLucy)),
    ],
    ids=["nothing", "total_variation", "shot_noise", "shot_and_read_noise"],
)
def test_the_model_class_chooses_the_unwritten_parts_as_for_a_model_built_in_python(block, expected):
    model = ReconstructionModel.from_config(MINIMAL | block)
    assert (type(model.object_prior), type(model.noise), type(model.solver)) == expected
    system, parts = build(V4Config(**(MINIMAL | block)))
    given = {name: parts[name] for name in ("object_prior", "noise") if parts[name] is not None}
    python = ReconstructionModel.for_system(system, **given)
    assert (type(python.object_prior), type(python.noise), type(python.solver)) == expected


def test_the_unwritten_parts_take_the_schemas_default_weight_and_sigma():
    model = ReconstructionModel.from_config(MINIMAL)
    assert float(model.imaging_model.object_prior.get_buffer("weight")) == pytest.approx(1e-3)
    assert float(model.imaging_model.noise.get_buffer("sigma")) == 1.0


@pytest.mark.parametrize(
    ("raw", "field", "reason"),
    [
        (
            _raw(noise={"poisson": {}}, object={"prior": "tikhonov"}, solver={"reconstruction": "closed_form"}),
            "solver.reconstruction",
            "Wiener needs white Gaussian noise, got Poisson",
        ),
        (
            _free(object={"prior": "tv"}, solver={"reconstruction": "fista", "calibration": {"score": "evidence"}}),
            "solver.calibration.score",
            "Evidence needs a Gaussian object prior, got TotalVariation",
        ),
        (
            _free(object={"weight": "auto"}, solver={"calibration": {"score": "sharpness"}}),
            "object.weight",
            "auto needs a score that estimates the weight",
        ),
    ],
    ids=["photon_noise_with_a_gaussian_solver", "evidence_with_tv", "sharpness_with_weight_auto"],
)
def test_parts_that_refuse_each_other_are_refused_at_the_part_with_its_reason(raw, field, reason):
    message = _refusal(raw)
    assert message.startswith(f"{field}: ")
    assert reason in message
    assert "solver.reconstruction to" not in message  # the part's own reason, not a list of other options


# --- device and the calibrated block ---------------------------------------------------------------------------------


def test_the_device_reaches_the_model():
    _, parts = build(V4Config(**_raw(solver={"device": "cpu"})))
    assert parts["device"] == "cpu"
    model = ReconstructionModel.from_config(_raw(solver={"device": "cpu"}))
    assert model.device.type == "cpu"
    assert ReconstructionModel.from_config(_raw(solver={"device": "auto"})).device.type in {"cpu", "cuda", "mps"}
    assert _refusal(_raw(solver={"device": "nonsense:7"})).startswith("solver.device: 'nonsense:7' is not a device")


def test_an_unavailable_mps_device_gets_a_short_message(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    message = _refusal(_raw(solver={"device": "mps"}))
    assert message.startswith("solver.device: 'mps' is not available here: no Apple GPU")
    assert "\n" not in message


def test_the_calibrated_block_is_applied_after_building():
    model = ReconstructionModel.from_config(_free(calibrated={NA: 1.1}))
    assert model.imaging_model.get_parameter(NA).item() == pytest.approx(1.1)
    assert free_specs(model.imaging_model)[NA].nominal == 1.2  # the prior is the configured one
    assert _refusal(_free(calibrated={"system.detection.numerical_apertur": 1.1})).startswith("calibrated: ")


# --- the provenance hash ---------------------------------------------------------------------------------------------


# The digests are of registry.describe(system) since the hash stopped writing an imaging_system block per built-in
# system (review B9), so a thin and a thick system now differ. An aberration-free detection's pupil_aberrations: None
# is left out, so the digests are those of the stores written before Detection took that argument.
CONFOCAL_DIGEST = "9325f61c1205be64a0a6f0b5c23a438360977b6b20c216f5a2db1914b1a67b6f"
TILED = _raw()
TILED["imaging_system"]["field_dependence"] = {"tiled": {"size_yx": [64, 64]}}
FREE_SYSTEM = _raw(
    detection={
        "numerical_aperture": FREE_NA,
        "pupil_aberrations": {"spherical": {"nominal": 0, "bounds": [-0.2, 0.2]}, "defocus": 0.01},
    }
)


@pytest.mark.parametrize(
    ("raw", "digest"),
    [
        (_raw(), "c1324af02f6f944a3d8931884e3e9d0d2234c2e5f00a174270e70fc66387c628"),
        (_raw(object={"dimension": 2}), "d7fe00e7d12a578149b8714e47e45d1eaa1f765ecc1f2732d75b9c80b40efeb5"),
        (FREE_SYSTEM, "c6015c5535919573c525fbd07baf8652e92f3991e4e08bef8f3020b6a22b4c9f"),
        (_raw(detection={"pinhole": {"diameter_um": 0.3}}), CONFOCAL_DIGEST),
        (_phase(), "c34678b5fea13e4fa142f64890d92c5c1937a29f233efb0cd860ea58f193f4a9"),
        (
            _phase(numerical_aperture={"nominal": 0.45, "bounds": [0.3, 0.7]}, tilt={"zenith": 0.1}),
            "91f86a1bfc7024a6dce9a38411cd92e678015a3560061aa90381851113bfa71d",
        ),
        (TILED, "f210c9b4f61a61b90e33924019a5fe1f61ee5f25fcf58291f423a1782dae3b38"),
    ],
    ids=["widefield", "thin", "free", "confocal", "phase", "phase_free", "tiled"],
)
def test_the_system_hash_of_a_configuration_does_not_change(raw, digest):
    model = ReconstructionModel.from_config(raw)
    assert system_sha256(model) == digest
    assert system_record(model)["imaging_system_sha256"] == digest


class _GainBlur(ShiftInvariantSystem):
    """A system of one's own: a Gaussian blur of a free width, times a fixed gain."""

    real_transfer = True

    def __init__(self, *, width: Quantity, gain: float = 1.0) -> None:
        super().__init__(fields=("density",), channel_count=1)
        self.register_quantity("width", width, positive=True)
        self.gain = gain

    def compute_transfer(self, sampling: Sampling) -> torch.Tensor:
        width = self.quantity("width")
        fz, fy, fx = half_spectrum_frequencies(sampling, device=width.device, dtype=width.dtype)
        return self.gain * torch.exp(-2 * torch.pi**2 * width**2 * (fz**2 + fy**2 + fx**2))[None, None]


def test_a_store_is_refused_by_a_registered_system_of_ones_own_with_other_settings():
    with temporarily("test_gain_blur_3d", _GainBlur, kind="system"):
        width = Free(0.3, bounds=(0.1, 0.6))
        made = ReconstructionModel.for_system(_GainBlur(width=width), pixel_size=(0.25, 0.1))
        same = ReconstructionModel.for_system(_GainBlur(width=Free(0.4, bounds=(0.2, 0.5))), pixel_size=(0.25, 0.1))
        other = ReconstructionModel.for_system(_GainBlur(width=width, gain=5.0), pixel_size=(0.25, 0.1))
        store = made.to_tf_store()
        assert store.provenance["imaging_system_sha256"] == system_sha256(made) == system_sha256(same)
        same.load_calibration(store)
        with pytest.raises(Rejected, match=r"another imaging_system \(imaging_system\.test_gain_blur_3d\.gain: 1\.0"):
            other.load_calibration(store)


# --- the built model computes ----------------------------------------------------------------------------------------


def _sampling(shape: tuple[int, int, int]) -> Sampling:
    return Sampling(shape=shape, pixel_size=PIXEL_SIZE)


def test_the_model_images_and_reconstructs_on_the_configured_grid():
    config = V4Config(**_raw(object={"weight": 1e-6}))
    model = ReconstructionModel.from_config(config)
    arguments = config.data.arguments()
    assert arguments["pixel_size"] == PIXEL_SIZE
    sampling = _sampling((8, 16, 16))
    model.check(sampling.shape, pixel_size=arguments["pixel_size"])
    with torch.no_grad():
        obj = torch.rand(1, *sampling.shape, generator=torch.Generator().manual_seed(0))
        data = model.imaging_model(obj, sampling)
        estimate = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
        residual = model.imaging_model(estimate, sampling) - data
    assert estimate.shape == (1, 8, 16, 16)
    assert residual.norm() < 1e-2 * data.norm()


def test_calibrate_moves_the_aperture_toward_the_truth():
    search = [{"grid": {"points": 4}}, {"gradient": {"max_iter": 5}}]
    raw = _raw(
        detection={"numerical_aperture": {"nominal": 1.05, "tolerance": "free", "bounds": [1.0, 1.3]}},
        object={"weight": 1e-2},
        noise={"gaussian": {"sigma": 0.05}},
        solver={"calibration": {"score": "evidence", "search": search}},
    )
    model = ReconstructionModel.from_config(raw)
    sampling = _sampling((8, 32, 32))
    truth = ReconstructionModel.from_config(raw).imaging_model.double()
    load_parameters(truth, {NA: torch.tensor(1.25, dtype=torch.float64)})
    generator = torch.Generator().manual_seed(0)
    obj = 0.5 * torch.randn(truth.system.object_shape(sampling), generator=generator, dtype=torch.float64)
    with torch.no_grad():
        data = truth.noise.sample(truth(obj, sampling), generator=generator)
    model.imaging_model.double()
    result = model.calibrate(data, pixel_size=PIXEL_SIZE, prepared=True)
    assert isinstance(result, CalibrationResult)
    estimate = result.parameters[NA].item()
    assert abs(estimate - 1.25) < abs(1.05 - 1.25) / 2
    assert model.imaging_model.get_parameter(NA).item() == pytest.approx(estimate)  # the model keeps the estimate


def test_reconstruct_tiles_with_one_tile_matches_reconstruct():
    model = ReconstructionModel.from_config(_raw(object={"weight": 1e-2}))
    sampling = _sampling((8, 16, 16))
    with torch.no_grad():
        data = model.imaging_model(torch.rand(1, *sampling.shape, generator=torch.Generator().manual_seed(0)), sampling)
        whole = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    sink = torch.zeros_like(whole)
    model.reconstruct_tiles(
        [(data, sink)], pixel_size=PIXEL_SIZE, plan=TilePlan(shape=sampling.shape, tile=sampling.shape)
    )
    assert (sink - whole).norm() <= 1e-6 * whole.norm()


def test_a_tile_plan_stitches_a_series_close_to_the_whole_reconstruction():
    model = ReconstructionModel.from_config(_raw(object={"weight": 1e-2}))
    shape = (8, 64, 64)
    sampling = _sampling(shape)
    with torch.no_grad():
        data = model.imaging_model(torch.rand(1, *shape, generator=torch.Generator().manual_seed(0)), sampling)
        whole = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    plan = model.tile_plan(shape, pixel_size=PIXEL_SIZE, tile=(8, 32, 32), overlap=(0, 4, 4))
    assert plan.halo[1] > 0  # the filter reaches beyond a core
    sinks = [torch.zeros_like(whole), torch.zeros_like(whole)]
    stats = model.reconstruct_tiles([(data, sinks[0]), (2 * data, sinks[1])], pixel_size=PIXEL_SIZE, plan=plan)
    assert stats.tiles == 8  # four tiles per volume
    assert (sinks[0] - whole).norm() <= 0.02 * whole.norm()  # a 99% halo keeps the stitching error near 1%
    assert (sinks[1] - 2 * whole).norm() <= 0.02 * 2 * whole.norm()
