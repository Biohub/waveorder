"""``waveorder.config.load`` validates a version-4 file or mapping, or refuses it with each problem at its path."""

import pytest
import yaml

from waveorder.config import V4Config, dump, load
from waveorder.config.schema import PixelSize
from waveorder.errors import ConfigurationWarning, Problem, Rejected
from waveorder.models import ReconstructionModel

V4_TEXT = """
version: 4
data:
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: phase
  detection: {numerical_aperture: 1.2}
  illumination: {wavelength: 0.532, numerical_aperture: 0.9}
"""
OLD_TEXT = "input_channel_names: [BF]\nreconstruction_dimension: 3\n"
NOT_A_KEY = (
    "imaging_system.detection.numerical_apperture: is not a configuration key; "
    "did you mean imaging_system.detection.numerical_aperture?"
)


def _problems(source: object) -> list[Problem]:
    with pytest.raises(Rejected) as rejected:
        load(source)  # ty: ignore[invalid-argument-type] (each test passes one of the accepted sources)
    return list(rejected.value.problems)


def test_a_configuration_is_loaded_from_a_path_a_mapping_or_itself(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(V4_TEXT)
    from_path = load(path)
    assert isinstance(from_path, V4Config)
    assert load(str(path)) == from_path
    assert load(yaml.safe_load(V4_TEXT)) == from_path
    assert load(from_path) is from_path


def test_a_3x_file_is_refused_naming_the_command_that_migrates_it(tmp_path):
    old = tmp_path / "old.yml"
    old.write_text(OLD_TEXT)
    expected = Problem(
        f"has no version, so it is a 3.x configuration; convert it with: wo config migrate {old}, "
        f"then use {tmp_path / 'old_v4.yml'}",
        str(old),
    )
    assert _problems(old) == [expected]
    with pytest.raises(Rejected) as rejected:
        ReconstructionModel.from_config(old)
    assert list(rejected.value.problems) == [expected]
    (problem,) = _problems(yaml.safe_load(OLD_TEXT))  # a mapping has no file to name
    assert problem.message.endswith("convert it with: wo config migrate FILE")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "is empty; print a configuration to start from with: wo config new"),
        ("{}\n", "is empty; print a configuration to start from with: wo config new"),
        ("- a\n", "must hold a YAML mapping, got list"),
    ],
    ids=["empty", "empty_mapping", "list"],
)
def test_a_document_that_is_no_configuration_is_refused_at_the_file(tmp_path, text, message):
    path = tmp_path / "config.yml"
    path.write_text(text)
    assert _problems(path) == [Problem(message, str(path))]


def test_a_missing_file_is_refused_saying_so(tmp_path):
    missing = tmp_path / "missing.yml"
    assert _problems(missing) == [
        Problem("cannot be read as a configuration file: No such file or directory", str(missing))
    ]


def test_a_key_given_twice_is_refused_at_its_line_and_merge_keys_may_be_overridden(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(V4_TEXT + "version: 4\n")
    (problem,) = _problems(path)
    assert problem.message == "while reading a mapping the key 'version' is given twice"
    assert problem.location == f"{path}:{len(V4_TEXT.splitlines()) + 1}:1"
    merged = "pixel_size: {<<: &base {z: 0.5, y: 0.1, x: 0.1}, z: 0.25}"
    path.write_text(V4_TEXT.replace("pixel_size: {z: 0.25, y: 0.1, x: 0.1}", merged))
    assert load(path).data.pixel_size == PixelSize(z=0.25, y=0.1, x=0.1)  # a merged key may be overridden


def test_a_yaml_syntax_error_keeps_its_line_and_column(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("version: 4\nimaging_system: {contrast: [\n")
    (problem,) = _problems(path)
    assert problem.location == f"{path}:3:1"


def test_each_problem_starts_with_its_dotted_path_and_is_located_at_the_file(tmp_path):
    path = tmp_path / "config.yml"
    text = V4_TEXT.replace("numerical_aperture: 1.2", "numerical_apperture: 1.2").replace("z: 0.25", "z: -1")
    path.write_text(text + "solver: {reconstruction: {fista: {iterations: 0}}}\n")
    problems = {problem.message: problem.location for problem in _problems(path)}
    assert problems == {
        NOT_A_KEY: str(path),
        "data.pixel_size.z: must be greater than 0, got -1.0": str(path),
        "solver.reconstruction.fista.iterations: Input should be greater than 0": str(path),
    }
    assert set(_problems(yaml.safe_load(text))) == {
        Problem(NOT_A_KEY, "the configuration"),
        Problem("data.pixel_size.z: must be greater than 0, got -1.0", "the configuration"),
    }


def test_the_yx_padding_wo_40_writes_loads_and_any_other_has_no_effect_at_its_key():
    raw = yaml.safe_load(V4_TEXT)
    loaded = load(raw | {"solver": {"padding": {"z": 5, "yx": 0}}})  # as wo-4.0's wo config migrate writes it
    assert (loaded.solver.padding.z, loaded.solver.padding.yx) == (5, 0)
    with pytest.warns(ConfigurationWarning, match="4 has no effect: yx padding is not in this version") as caught:
        load(raw | {"solver": {"padding": {"z": 5, "yx": 4}}})
    assert [getattr(warning.message, "location", None) for warning in caught] == ["solver.padding.yx"]


def test_wo_40_forms_of_a_pixel_size_and_a_tolerance_read_as_wo_40_read_them():
    raw = yaml.safe_load(V4_TEXT)
    wo40 = raw | {"data": raw["data"] | {"pixel_size": {"z": 0.25, "yx": {}}}}
    with pytest.warns(ConfigurationWarning, match=r"data\.pixel_size is written \{z, yx\}"):
        assert load(wo40).data.pixel_size == PixelSize(z=0.25, y=0.1, x=0.1)  # wo-4.0's default yx
    never = {"solver": {"reconstruction": {"richardson_lucy": {"stopping_tolerance": 0}}}}
    assert dump(load(raw | never))["solver"]["reconstruction"]["richardson_lucy"]["tolerance"] is None  # never stops


def test_wo_40_richardson_lucy_stopping_tolerance_is_the_tolerance():
    raw = yaml.safe_load(V4_TEXT)
    wo40 = dump(load(raw | {"solver": {"reconstruction": {"richardson_lucy": {"stopping_tolerance": 1e-3}}}}))
    ours = dump(load(raw | {"solver": {"reconstruction": {"richardson_lucy": {"tolerance": 1e-3}}}}))
    assert wo40 == ours
    assert ours["solver"]["reconstruction"]["richardson_lucy"]["tolerance"] == 1e-3


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("auto", "auto"),
        (0.2, PixelSize(z=0.2, y=0.2, x=0.2)),
        ({"z": 0.5, "y": 0.1, "x": 0.2}, PixelSize(z=0.5, y=0.1, x=0.2)),
    ],
    ids=["auto", "number", "zyx"],
)
def test_the_pixel_size_is_auto_a_number_or_one_per_axis(written, expected):
    raw = yaml.safe_load(V4_TEXT)
    raw["data"]["pixel_size"] = written
    assert load(raw).data.pixel_size == expected
    del raw["data"]["pixel_size"]
    assert load(raw).data.pixel_size == "auto"  # the default: read from the store


def test_a_wo40_pixel_size_loads_with_a_warning_naming_the_migration(tmp_path):
    raw = yaml.safe_load(V4_TEXT)
    raw["data"]["pixel_size"] = {"z": 0.5, "yx": 0.1}
    path = tmp_path / "wo40.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.warns(ConfigurationWarning, match=r"data.pixel_size is written \{z, yx\}, as wo-4.0 writes it") as got:
        loaded = load(path)
    assert loaded.data.pixel_size == PixelSize(z=0.5, y=0.1, x=0.1)
    assert str(got[0].message).endswith(f"convert it with: wo config migrate {path}")
    raw["data"]["pixel_size"] = {"z": 0.5, "yx": {"y": 0.1, "x": 0.2}}  # wo-4.0's {y, x} form of yx
    with pytest.warns(ConfigurationWarning, match="as wo-4.0 writes it"):
        assert load(raw).data.pixel_size == PixelSize(z=0.5, y=0.1, x=0.2)
    raw["data"]["pixel_size"] = {"z": -1, "yx": 0.1}
    size, *_, migrate = _problems(raw)
    assert size.message.startswith("data.pixel_size.z: ")
    assert migrate.message.endswith("convert it with: wo config migrate FILE")


@pytest.mark.parametrize(
    ("data", "arguments"),
    [
        ({}, {"channels": ["simulated"], "pixel_size": None, "times": None}),
        (
            {"input_channel_names": ["BF"], "pixel_size": 0.2, "time_indices": 3},
            {"channels": ["BF"], "pixel_size": (0.2, 0.2, 0.2), "times": [3]},
        ),
        (
            {"pixel_size": {"z": 0.5, "y": 0.1, "x": 0.2}, "time_indices": [0, 2]},
            {"channels": ["simulated"], "pixel_size": (0.5, 0.1, 0.2), "times": [0, 2]},
        ),
    ],
    ids=["defaults", "one_number", "per_axis"],
)
def test_the_data_block_gives_the_arguments_of_the_position_methods(data, arguments):
    raw = yaml.safe_load(V4_TEXT)
    raw["data"] = data
    assert load(raw).data.arguments() == arguments


def test_dump_leaves_out_unchosen_nulls_and_an_empty_calibrated_block():
    config = load(yaml.safe_load(V4_TEXT))
    shown = dump(config)
    assert "wavelength" not in shown["imaging_system"]["detection"]  # brightfield detects the illumination's
    assert "simulate" not in shown
    assert "calibrated" not in shown
    assert shown["noise"] == {"gaussian": {"sigma": 1.0}}
    assert load(shown) == config
    calibrated = config.model_copy(update={"calibrated": {"system.illumination.numerical_aperture": 0.85}})
    assert dump(calibrated)["calibrated"] == {"system.illumination.numerical_aperture": 0.85}
