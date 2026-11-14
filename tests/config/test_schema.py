"""The 4.0 schema parses the design document's examples and enforces its rules."""

import pytest
import yaml
from pydantic import ValidationError

from waveorder.config import load
from waveorder.config.schema import (
    DataSettings,
    FieldDependenceTiled,
    FreeParameter,
    ImagingSystemSettings,
    NoiseModel,
    ObjectSettings,
    PixelSize,
    PriorBeads,
    SearchGradient,
    SearchGrid,
    SolverSettings,
    V4Config,
    free_parameter_specs,
)
from waveorder.errors import ConfigurationWarning, Rejected
from waveorder.models import ReconstructionModel
from waveorder.nn import free_specs

# Design document example 4.1: blind, tiled, evidence-scored phase reconstruction.
EXAMPLE_4_1 = """
version: 4

data:
  input_channel_names: [BF]
  time_indices: all
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}

imaging_system:
  contrast: phase
  field_dependence:
    tiled: {size_yx: [256, 256], overlap_yx: [32, 32], tile_blend: linear}
  index_of_refraction_media: 1.3
  detection:
    numerical_aperture: {nominal: 1.2, tolerance: 0.05, bounds: [1.0, 1.3]}
    pupil_aberrations:
      zernike_up_to_order: {order: 4, tolerance: 0.02}
      spherical: {nominal: 0.05, tolerance: 0.02}
  illumination:
    wavelength: 0.532
    numerical_aperture: 0.9
    tilt: {zenith: {nominal: 0.0, tolerance: 0.02}, azimuth: 0.0}
    invert_phase_contrast: false

object:
  dimension: 3
  prior: tikhonov
  weight: auto
  constraints: []

noise:
  gaussian: {sigma: estimate}

solver:
  padding: {z: 5}
  reconstruction: closed_form
  calibration:
    score: evidence
    search:
      - grid: {points: 7}
      - gradient: {max_iter: 200}

outputs: [reconstruction, transfer_function]
"""

# Design document example 4.2: the same system, non-blind, as 3.x ran it.
EXAMPLE_4_2 = """
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.3
  detection: {numerical_aperture: 1.2}
  illumination: {wavelength: 0.532, numerical_aperture: 0.9}
object: {prior: tikhonov, weight: 1e-3}
"""

# Design document example 4.3: bead calibration (fluorescence).
EXAMPLE_4_3 = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  field_dependence: shift_invariant
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.520
    numerical_aperture: {nominal: 1.2, tolerance: free, bounds: [1.0, 1.3]}
    pinhole: none
    pupil_aberrations: {zernike_up_to_order: {order: 4, tolerance: free}}
object:
  prior: {beads: {diameter: 0.2}}
noise: {poisson_gaussian: {gain: 2.1, offset: 100, sigma: estimate}}
solver:
  calibration: {score: evidence, search: [{gradient: {max_iter: 100}}]}
outputs: [reconstruction, transfer_function]
"""


def _parse(text: str) -> V4Config:
    return V4Config(**yaml.safe_load(text))


def _blocks_4_1() -> dict:
    # The whole example names settings no module represents yet (tiled shift-variance, an estimated sigma), which the
    # builders reject at validation (see the test below); its blocks still parse as the design document writes them.
    raw = yaml.safe_load(EXAMPLE_4_1)
    return {
        "imaging_system": ImagingSystemSettings.model_validate(raw["imaging_system"]),
        "object": ObjectSettings.model_validate(raw["object"]),
        "noise": NoiseModel.model_validate(raw["noise"]),
        "solver": SolverSettings.model_validate(raw["solver"]),
    }


def test_example_4_1_validates_and_its_model_bounds_the_unbounded_free_tilt_as_a_migration_does():
    # The document is valid (its grid warns); the phase systems clamp a free zenith only with bounds, which wo-4.0
    # did not need, so the model takes the range wo config migrate gives it, with a warning at the key.
    raw = yaml.safe_load(EXAMPLE_4_1)
    with pytest.warns(ConfigurationWarning, match="the grid evaluates"):
        V4Config(**raw)
    with pytest.warns(ConfigurationWarning) as caught:
        model = ReconstructionModel.from_config(raw)
    assert "imaging_system.illumination.tilt.zenith" in {
        getattr(warning.message, "location", None) for warning in caught
    }
    assert free_specs(model.imaging_model)["system.illumination.tilt.zenith"].bounds == pytest.approx((-0.877, 0.877))


def test_example_4_1_parses():
    blocks = _blocks_4_1()
    field_dependence = blocks["imaging_system"].field_dependence
    assert isinstance(field_dependence, FieldDependenceTiled)
    assert field_dependence.tiled.size_yx == (256, 256)
    na = blocks["imaging_system"].detection.numerical_aperture
    assert isinstance(na, FreeParameter)
    assert na.tolerance == 0.05
    assert na.bounds == (1.0, 1.3)
    aberrations = blocks["imaging_system"].detection.pupil_aberrations
    assert aberrations is not None
    assert aberrations.zernike_up_to_order is not None
    assert aberrations.zernike_up_to_order.order == 4
    spherical = aberrations.terms["spherical"]
    assert isinstance(spherical, FreeParameter)
    assert spherical.nominal == 0.05
    calibration = blocks["solver"].calibration
    assert calibration is not None
    assert isinstance(calibration.search[0], SearchGrid)
    assert isinstance(calibration.search[1], SearchGradient)
    assert blocks["noise"].gaussian is not None
    assert blocks["noise"].gaussian.sigma == "estimate"
    assert blocks["object"].weight == "auto"


def test_example_4_2_parses_and_is_fixed():
    cfg = _parse(EXAMPLE_4_2)
    assert free_parameter_specs(cfg.imaging_system) == {}
    assert cfg.solver.reconstruction == "closed_form"
    assert cfg.noise.kind == "gaussian"
    assert cfg.noise.gaussian is not None
    assert cfg.noise.gaussian.sigma == 1.0  # default reproduces 3.x
    assert cfg.solver.padding.z == 0  # 3.x z_padding default
    assert cfg.solver.calibration is None


def test_example_4_3_parses():
    cfg = _parse(EXAMPLE_4_3)
    prior = cfg.object.prior
    assert isinstance(prior, PriorBeads)
    assert prior.beads.diameter == 0.2
    assert cfg.noise.kind == "poisson_gaussian"
    params = free_parameter_specs(cfg.imaging_system)
    assert "detection.numerical_aperture" in params
    assert "detection.pupil_aberrations.zernike_up_to_order" in params


def test_free_parameter_specs_collects_named_terms():
    params = free_parameter_specs(_blocks_4_1()["imaging_system"])
    assert params["detection.pupil_aberrations.spherical"].tolerance == 0.02
    assert params["illumination.tilt.zenith"].tolerance == 0.02
    assert "illumination.tilt.azimuth" not in params


def test_nominal_outside_bounds_rejected():
    with pytest.raises(ValueError, match="outside bounds"):
        FreeParameter(nominal=2.0, tolerance=0.1, bounds=(0.0, 1.0))


def test_anisotropic_pixel_size_keeps_each_axis():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["data"]["pixel_size"] = {"z": 0.4, "y": 0.108, "x": 0.116}
    assert V4Config(**raw).data.pixel_size == PixelSize(z=0.4, y=0.108, x=0.116)


@pytest.mark.parametrize(
    "pixel_size",
    [
        {"z": 0.25, "y": 0.1},  # missing x
        {"z": 0.25, "y": 0.1, "x": 0.0},
        {"z": float("inf"), "y": 0.1, "x": 0.1},
    ],
)
def test_invalid_pixel_size_rejected(pixel_size):
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["data"]["pixel_size"] = pixel_size
    with pytest.raises(ValueError, match="pixel_size"):
        V4Config(**raw)


def test_aberration_names_are_canonical_in_free_parameter_specs():
    raw = yaml.safe_load(EXAMPLE_4_3)
    raw["imaging_system"]["detection"]["pupil_aberrations"] = {
        "z11": {"nominal": 0.0, "tolerance": 0.02},
        "z15": {"nominal": 0.0, "tolerance": 0.02},
    }
    params = free_parameter_specs(V4Config(**raw).imaging_system)
    assert "detection.pupil_aberrations.spherical" in params
    assert "detection.pupil_aberrations.z15" in params


def test_same_aberration_under_two_names_rejected():
    raw = yaml.safe_load(EXAMPLE_4_3)
    raw["imaging_system"]["detection"]["pupil_aberrations"] = {"spherical": 0.05, "z11": 0.01}
    with pytest.raises(ValueError, match="twice"):
        V4Config(**raw)


@pytest.mark.parametrize("arm", ["detection", "illumination"])
def test_na_above_index_is_refused_at_the_aperture(arm):
    bad = yaml.safe_load(EXAMPLE_4_2)
    bad["imaging_system"].setdefault(arm, {})["numerical_aperture"] = 1.4
    V4Config(**bad)  # the document is valid: the system refuses the aperture
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(bad)
    (problem,) = rejected.value.problems
    assert problem.message.startswith(f"imaging_system.{arm}.numerical_aperture: ")
    assert "above index_of_refraction_media = 1.3" in problem.message


@pytest.mark.parametrize("aperture", [0.0, -0.5], ids=["zero", "negative"])
def test_a_non_positive_detection_na_is_refused_at_the_aperture(aperture):
    bad = yaml.safe_load(EXAMPLE_4_2)
    bad["imaging_system"]["detection"]["numerical_aperture"] = aperture
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(bad)
    (problem,) = rejected.value.problems
    assert problem.message.startswith("imaging_system.detection.numerical_aperture: ")
    assert "must be positive" in problem.message


@pytest.mark.parametrize(
    ("path", "location"),
    [
        (("imaging_system", "detection", "wavelength"), "imaging_system.detection.wavelength"),
        (("imaging_system", "index_of_refraction_media"), "imaging_system.index_of_refraction_media"),
        (("imaging_system", "detection", "numerical_aperture"), "imaging_system.detection.numerical_aperture"),
        (("noise", "gaussian", "sigma"), "noise.gaussian.sigma"),
        (("object", "weight"), "object.weight"),
    ],
    ids=["wavelength", "index", "aperture", "sigma", "weight"],
)
def test_a_yaml_bool_is_never_a_number(path, location):
    # Pydantic's lax mode would read `true` as 1; every number in the schema rejects it at its key.
    bad = yaml.safe_load(EXAMPLE_4_2)
    *parents, leaf = path
    node = bad
    for parent in parents:
        node = node.setdefault(parent, {})
    node[leaf] = True
    with pytest.raises(ValidationError, match="not a bool") as error:
        V4Config(**bad)
    locations = {".".join(map(str, detail["loc"])) for detail in error.value.errors()}
    assert locations == {location}


@pytest.mark.parametrize(
    ("path", "location"),
    [
        (("imaging_system", "detection", "wavelength"), "imaging_system.detection.wavelength"),
        (("noise", "gaussian", "sigma"), "noise.gaussian.sigma"),
    ],
    ids=["wavelength", "sigma"],
)
@pytest.mark.parametrize("text", [".nan", ".inf"])
def test_a_non_finite_positive_number_reads_finite_at_its_key(path, location, text):
    # The finiteness check runs before the sign, so `.nan` is not reported as "greater than 0".
    bad = yaml.safe_load(EXAMPLE_4_2)
    *parents, leaf = path
    node = bad
    for parent in parents:
        node = node.setdefault(parent, {})
    node[leaf] = yaml.safe_load(text)
    with pytest.raises(ValidationError, match="finite number") as error:
        V4Config(**bad)
    locations = {".".join(map(str, detail["loc"])) for detail in error.value.errors()}
    assert locations == {location}
    node[leaf] = -1.0
    with pytest.raises(ValidationError, match="greater than 0"):
        V4Config(**bad)


def test_a_yaml_bool_is_not_a_bound_or_a_seed():
    bad = yaml.safe_load(EXAMPLE_4_2)
    bad["imaging_system"]["detection"]["numerical_aperture"] = {"nominal": 1.0, "tolerance": 0.1, "bounds": [True, 1.2]}
    with pytest.raises(ValidationError, match="not a bool"):
        V4Config(**bad)
    seeded = yaml.safe_load(EXAMPLE_4_2)
    seeded.setdefault("solver", {})["calibration"] = {"search": [{"random": {"seed": True}}]}
    with pytest.raises(ValidationError, match="not a bool"):
        V4Config(**seeded)


def test_unknown_zernike_name_rejected():
    bad = yaml.safe_load(EXAMPLE_4_3)
    bad["imaging_system"]["detection"]["pupil_aberrations"] = {"comatose": 0.1}
    with pytest.raises(ValueError, match="Zernike"):
        V4Config(**bad)


def test_grid_stage_with_unbounded_free_parameter_rejected():
    bad = yaml.safe_load(EXAMPLE_4_3)
    bad["imaging_system"]["detection"]["numerical_aperture"] = {"nominal": 1.2, "tolerance": "free"}
    bad["imaging_system"]["detection"].pop("pupil_aberrations")
    bad["solver"]["calibration"]["search"] = [{"grid": {"points": 5}}]
    with pytest.raises(ValueError, match="bounds"):
        V4Config(**bad)


def _location(error: pytest.ExceptionInfo[ValidationError]) -> str:
    (detail,) = error.value.errors()
    return ".".join(str(part) for part in detail["loc"])


def _refused_at(raw: dict) -> str:
    """Return the config path of the one problem the model refuses ``raw`` with."""
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(raw)
    (problem,) = rejected.value.problems
    return problem.message.split(": ", 1)[0]


def test_rl_on_a_phase_system_is_refused_at_the_solver():
    # Phase transfers take negative values, so Richardson-Lucy cannot keep its estimate nonnegative.
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["solver"] = {"reconstruction": {"richardson_lucy": {"iterations": 10}}}
    raw["object"] = {"prior": "none"}
    raw["noise"] = {"poisson": {}}
    V4Config(**raw)  # the document is valid: the solver refuses the system
    assert _refused_at(raw) == "solver.reconstruction"


def test_prior_none_with_closed_form_is_refused_at_the_solver():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["object"] = {"prior": "none"}
    raw.setdefault("solver", {})["reconstruction"] = "closed_form"
    assert _refused_at(raw) == "solver.reconstruction"


@pytest.mark.parametrize("noise", [{"poisson_gaussian": {"gain": 2.1, "offset": 100, "sigma": 3.0}}, {"gaussian": {}}])
def test_weight_auto_requires_a_score_that_estimates_the_weight(noise):
    raw = yaml.safe_load(EXAMPLE_4_3)
    raw["object"] = {"prior": "tikhonov", "weight": "auto"}
    raw["noise"] = noise
    raw["solver"]["calibration"]["score"] = "sharpness"
    with pytest.raises(ValidationError) as error:
        V4Config(**raw)
    assert [".".join(map(str, line["loc"])) for line in error.value.errors()] == ["object.weight"]


def test_extra_fields_rejected():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["imaging_system"]["detection"]["numerical_apertur"] = 1.0
    with pytest.raises(ValueError, match="numerical_apertur"):
        V4Config(**raw)


def test_beads_prior_requires_a_free_parameter():
    raw = yaml.safe_load(EXAMPLE_4_3)
    raw["imaging_system"]["detection"]["numerical_aperture"] = 1.2
    raw["imaging_system"]["detection"].pop("pupil_aberrations")
    with pytest.raises(ValueError, match="beads"):
        V4Config(**raw)


def test_grid_point_count_warning():
    raw = yaml.safe_load(EXAMPLE_4_3)
    raw["imaging_system"]["detection"]["pupil_aberrations"] = {"zernike_up_to_order": {"order": 4, "tolerance": 0.05}}
    raw["solver"]["calibration"]["search"] = [{"grid": {"points": 7}}]
    with pytest.warns(ConfigurationWarning, match="the grid evaluates 96889010407 points"):
        V4Config(**raw)  # 7^13 points (12 Zernikes + free NA)


def test_option_errors_point_at_the_chosen_branch():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["solver"] = {"reconstruction": {"fista": {"iterationz": 3}}}
    with pytest.raises(ValueError, match=r"solver\.reconstruction\.fista\.iterationz") as error:
        V4Config(**raw)
    assert "closed_form" not in str(error.value)  # no noise from the other branches


@pytest.mark.parametrize(("section", "key", "name"), [("solver", "reconstruction", "fista")])
def test_bare_names_of_parameterized_options_take_their_defaults(section, key, name):
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw.setdefault(section, {})[key] = name
    bare = getattr(getattr(V4Config(**raw), section), key)
    raw[section][key] = {name: {}}
    assert bare == getattr(getattr(V4Config(**raw), section), key)
    assert getattr(bare, name) is not None


@pytest.mark.parametrize(
    ("block", "key", "value", "path"),
    [
        ("detection", "pinhole", {"diameter_au": 1.0, "diameter_um": 2.0}, "imaging_system.detection.pinhole"),
        ("detection", "numerical_aperture", {"nominal": 1.0, "tolerence": 0.1}, "imaging_system.detection"),
        ("detection", "numerical_aperture", {"nominal": 1.0, "tolerance": "fre"}, "imaging_system.detection"),
        (None, "field_dependence", {"tiled": {"size": 3}}, "imaging_system.field_dependence"),
    ],
    ids=["pinhole", "free_parameter_typo", "tolerance", "field_dependence"],
)
def test_untagged_union_errors_read_as_yaml_paths(block, key, value, path):
    # A number-or-mapping field reports each alternative at the YAML path, without pydantic's member labels
    # (float, FreeParameter, function-after[...]).
    raw = yaml.safe_load(EXAMPLE_4_2)
    (raw["imaging_system"] if block is None else raw["imaging_system"][block])[key] = value
    with pytest.raises(ValidationError) as error:
        V4Config(**raw)
    for detail in error.value.errors():
        location = ".".join(str(part) for part in detail["loc"])
        assert location.startswith(path), location
        assert all(str(part) == str(part).lower() and "[" not in str(part) for part in detail["loc"]), location


@pytest.mark.parametrize(
    ("edit", "path", "kind"),
    [
        (
            lambda raw: raw["solver"].update(calibration={"search": [{"grid": {"points": 0}}]}),
            "solver.calibration.search.0.grid.points",
            "greater_than_equal",
        ),
        (
            lambda raw: raw["imaging_system"]["detection"].update(
                numerical_aperture={"nominal": 1.31, "tolerance": 0.1, "bounds": [1.0, 1.3]}
            ),
            "imaging_system.detection.numerical_aperture",
            "value_error",
        ),
    ],
    ids=["grid_points", "nominal_out_of_bounds"],
)
def test_a_union_reports_only_the_alternative_that_matched_the_input_type(edit, path, kind):
    # `points: 0` is an int, so the dict alternative's dict_type error is noise; an out-of-bounds FreeParameter is a
    # mapping, so the float alternative's float_type error is noise.
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw.setdefault("solver", {})
    edit(raw)
    with pytest.raises(ValidationError) as error:
        V4Config(**raw)
    (detail,) = error.value.errors()
    assert ".".join(str(part) for part in detail["loc"]) == path
    assert detail["type"] == kind


@pytest.mark.parametrize(
    ("time_indices", "times"),
    [("all", None), (3, [3]), ([0, 2, 5], [0, 2, 5]), ([], [])],
    ids=["all", "one index", "a list", "an empty list"],
)
def test_the_data_block_gives_the_arguments_of_the_position_methods(time_indices, times):
    data = DataSettings(
        input_channel_names=["BF"], time_indices=time_indices, pixel_size=PixelSize(z=0.25, y=0.1, x=0.12)
    )
    arguments = data.arguments()
    assert arguments == {"channels": ["BF"], "pixel_size": (0.25, 0.1, 0.12), "times": times}
    assert arguments["channels"] is not data.input_channel_names  # a copy: the caller may change it


def test_a_free_quantity_needs_no_calibration_block():
    # Nothing refuses it: the model calibrates with its defaults (the evidence, and calibrate's default search).
    raw = yaml.safe_load(EXAMPLE_4_1)
    del raw["solver"]["calibration"]
    raw["noise"] = {"gaussian": {"sigma": 1.0}}  # a number, so the modules are built and judge the configuration
    raw["imaging_system"]["illumination"]["tilt"] = {"zenith": 0.0, "azimuth": 0.0}
    config = V4Config(**raw)
    assert config.solver.calibration is None
    assert config.object.weight == "auto"
    assert free_parameter_specs(config.imaging_system)


def _messages(**blocks: object) -> list[str]:
    raw = {"version": 4, "imaging_system": {"contrast": "fluorescence", "detection": {"wavelength": 0.5}}, **blocks}
    with pytest.raises(Rejected) as error:
        load(raw)
    return [problem.message for problem in error.value.problems]


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        (
            {"object": {"prior": "lasso"}},
            "object.prior: 'lasso' is not an option; choose one of tikhonov, tv, none, beads, denoiser",
        ),
        (
            {"solver": {"reconstruction": "newton"}},
            "solver.reconstruction: 'newton' is not an option; choose one of closed_form, fista, admm, "
            "richardson_lucy, distilled",
        ),
        (
            {"solver": {"calibration": {"search": ["gradiant"]}}},
            "solver.calibration.search.0: 'gradiant' is not an option; choose one of grid, random, gradient, lbfgs, "
            "nelder_mead",
        ),
    ],
)
def test_an_unknown_option_lists_every_option(blocks, message):
    assert _messages(**blocks) == [message]


@pytest.mark.parametrize(
    ("imaging_system", "key"),
    [
        (
            {"contrast": "fluorescence", "detection": {"wavelength": {"nominal": 0.5, "tolerance": "free"}}},
            "detection.wavelength",
        ),
        (
            {"contrast": "phase", "illumination": {"wavelength": {"nominal": 0.5}, "numerical_aperture": 0.5}},
            "illumination.wavelength",
        ),
        (
            {
                "contrast": "fluorescence",
                "index_of_refraction_media": {"nominal": 1.3},
                "detection": {"wavelength": 0.5},
            },
            "index_of_refraction_media",
        ),
    ],
)
def test_a_free_wavelength_or_index_is_refused_at_its_key(imaging_system, key):
    fixed = "a free value here is not available in this version of WaveOrder; give a number"
    assert _messages(imaging_system=imaging_system) == [f"imaging_system.{key}: {fixed}"]


def test_a_missing_bundle_names_the_command_that_writes_one(tmp_path):
    raw = {"version": 4, "imaging_system": {"contrast": "fluorescence", "detection": {"wavelength": 0.5}}}
    raw["solver"] = {"reconstruction": {"distilled": {"bundle": str(tmp_path / "missing")}}}
    with pytest.raises(Rejected, match="give the bundle wo distill fit wrote"):
        ReconstructionModel.from_config(raw)
