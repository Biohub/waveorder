"""``solver.reconstruction: {distilled: {bundle, precision}}``: a student bundle as the solver of a configuration."""

from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from tests.models.bundles import phase_bundle
from tests.nn.networks import tiny

from waveorder.config import V4Config
from waveorder.errors import ConfigurationError, Rejected
from waveorder.io.bundle import load_bundle
from waveorder.models import Phase2D, ReconstructionModel
from waveorder.nn import Detection, Distilled, Gaussian, Illumination, Tikhonov
from waveorder.nn.networks import Bundle

CONFIG = """
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.33
  detection: {numerical_aperture: 0.9}
  illumination: {wavelength: 0.532, numerical_aperture: 0.52}
object: {dimension: 2, weight: 1e-3}
noise: {gaussian: {sigma: 1e-3}}
"""
"""The system of :func:`tests.models.plates.thin_phase_model`, which the bundles of these tests are trained for."""


def _raw(reconstruction: object, **blocks: object) -> dict:
    raw = yaml.safe_load(CONFIG)
    raw.update(blocks)
    raw["solver"] = {**raw.get("solver", {}), "reconstruction": reconstruction}
    return raw


def _problem(source: object) -> str:
    """Return the message of the one problem ``from_config`` refuses ``source`` with."""
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(source)  # ty: ignore[invalid-argument-type] (a mapping or a path)
    (problem,) = rejected.value.problems
    return problem.message


def test_a_distilled_reconstruction_builds_from_its_configuration_and_keeps_it(tmp_path):
    path = phase_bundle(tmp_path / "student")
    raw = _raw({"distilled": {"bundle": str(path), "precision": "bf16"}})
    model = ReconstructionModel.from_config(V4Config(**raw))
    assert isinstance(model, Phase2D)
    solver = model.solver
    assert isinstance(solver, Distilled)
    assert solver.precision == "bf16"
    assert solver.bundle is not None
    assert solver.bundle.path == path
    written = model.to_config()
    assert isinstance(written, V4Config)
    assert written.model_dump(mode="json")["solver"]["reconstruction"] == raw["solver"]["reconstruction"]
    again = ReconstructionModel.from_config(written)
    assert isinstance(again.solver, Distilled)
    assert again.solver.bundle is not None
    assert again.solver.bundle.sha256 == solver.bundle.sha256 == load_bundle(path).sha256


def test_the_description_of_a_model_built_in_python_names_the_bundle_it_loaded(tmp_path):
    path = phase_bundle(tmp_path / "student")
    model = Phase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.52),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(weight=1e-3),
        noise=Gaussian(sigma=1e-3),
        solver=Distilled(load_bundle(path), precision="bf16"),
    )
    described: Any = model.to_config()
    assert isinstance(described, dict)
    assert described["parts"]["solver"] == {"distilled": {"path": str(path), "precision": "bf16"}}
    loaded = ReconstructionModel.load(model.save(tmp_path / "saved"))
    assert isinstance(loaded.solver, Distilled)
    assert loaded.solver.precision == "bf16"


def test_a_relative_bundle_path_is_read_from_the_working_directory(tmp_path, monkeypatch):
    phase_bundle(tmp_path / "student")
    monkeypatch.chdir(tmp_path)
    solver = ReconstructionModel.from_config(_raw({"distilled": {"bundle": "student"}})).solver
    assert isinstance(solver, Distilled)
    assert solver.precision == "float32"


def test_a_distilled_solver_of_a_bundle_in_memory_has_no_plain_data_form(tmp_path):
    bundle = load_bundle(phase_bundle(tmp_path / "student"))
    model = ReconstructionModel.from_config(_raw("closed_form"))
    model.solver = Distilled(Bundle(student=tiny.strided(8, halo=16), contract=bundle.contract))
    with pytest.raises(ConfigurationError, match="runs a bundle built in memory") as error:
        model.to_config()
    assert error.value.field == "solver"


def test_a_bare_distilled_needs_its_bundle():
    with pytest.raises(ValidationError, match="Field required") as error:
        V4Config(**_raw("distilled"))
    (detail,) = error.value.errors()
    assert ".".join(str(part) for part in detail["loc"]) == "solver.reconstruction.distilled.bundle"


def test_a_missing_bundle_is_a_problem_at_its_key_located_at_the_file(tmp_path):
    missing = tmp_path / "absent"
    raw = _raw({"distilled": {"bundle": str(missing)}})
    assert _problem(raw).startswith(f"solver.reconstruction.distilled.bundle: {missing} is not a directory")
    path = tmp_path / "distilled.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(Rejected) as located:
        ReconstructionModel.from_config(path)
    (problem,) = located.value.problems
    assert problem.location == str(path)
    assert problem.message.startswith("solver.reconstruction.distilled.bundle: ")


def test_a_directory_that_is_no_bundle_is_a_problem_at_its_key(tmp_path):
    (tmp_path / "empty").mkdir()
    message = _problem(_raw({"distilled": {"bundle": str(tmp_path / "empty")}}))
    assert message.startswith("solver.reconstruction.distilled.bundle: ")
    assert "is not a student bundle" in message


def test_calibration_with_a_distilled_reconstruction_is_refused(tmp_path):
    raw = _raw(
        {"distilled": {"bundle": str(phase_bundle(tmp_path / "student"))}},
        solver={"calibration": {"score": "evidence"}},
    )
    raw["imaging_system"]["illumination"]["numerical_aperture"] = {"nominal": 0.52, "tolerance": 0.05}
    with pytest.raises(ValidationError, match="needs a physics solver") as error:
        V4Config(**raw)
    (detail,) = error.value.errors()
    assert ".".join(str(part) for part in detail["loc"]) == "solver.calibration"


def test_the_object_prior_and_the_noise_do_not_judge_a_distilled_solve(tmp_path):
    # Refused for every physics solver but Richardson-Lucy, which needs no prior: a student reads neither.
    path = phase_bundle(tmp_path / "student")
    raw = _raw({"distilled": {"bundle": str(path)}}, object={"dimension": 2, "prior": "none"}, noise={"poisson": {}})
    assert isinstance(ReconstructionModel.from_config(raw).solver, Distilled)


def test_a_student_of_another_imaging_system_is_refused_at_the_reconstruction(tmp_path):
    other = Phase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.4),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
    )
    path = phase_bundle(tmp_path / "student", other)
    message = _problem(_raw({"distilled": {"bundle": str(path)}}))
    assert message.startswith("solver.reconstruction: ")
    assert "trained for another imaging system" in message
