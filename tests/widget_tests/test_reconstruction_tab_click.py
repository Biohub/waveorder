"""Integration tests that click the Reconstruction tab's buttons as a user does."""

import warnings

import pytest
import yaml

pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from waveorder.config import dump, load
from waveorder.errors import ConfigurationWarning
from waveorder.models import ReconstructionModel
from waveorder.plugin import forms
from waveorder.plugin.tab_recon import NOT_IN_THIS_VERSION, template


def _check(tab, *modes):
    for mode, checkbox in tab.modes_selected.items():
        checkbox["Checkbox"].value = mode in modes


def test_birefringence_is_shown_disabled_and_explains_why(recon_tab, messages):
    checkbox = recon_tab.modes_selected["birefringence"]["Checkbox"]
    assert not checkbox.enabled
    assert checkbox.tooltip == NOT_IN_THIS_VERSION
    _check(recon_tab, "birefringence")
    recon_tab.build_acq_contols()
    assert recon_tab.models == []
    assert messages == [(NOT_IN_THIS_VERSION, True)]


def test_click_new_button_phase(recon_tab, messages):
    _check(recon_tab, "phase")
    recon_tab.build_acq_contols()
    assert messages == []
    (entry,) = recon_tab.models
    assert entry.mode == "phase"
    assert entry.title == "phase - 01"
    assert entry.form["imaging_system"]["contrast"].value == "phase"
    assert not entry.form["imaging_system"]["contrast"].enabled
    assert recon_tab.reconstruction_run_PushButton.text == "RUN Model"


def test_click_new_button_phase_and_fluorescence(recon_tab):
    _check(recon_tab, "phase", "fluorescence")
    recon_tab.build_acq_contols()
    assert [entry.mode for entry in recon_tab.models] == ["phase", "fluorescence"]
    assert recon_tab.reconstruction_run_PushButton.text == "RUN 2 Models"


def test_multiple_models_created(recon_tab):
    _check(recon_tab, "phase")
    recon_tab.build_acq_contols()
    first = recon_tab.models[0]
    first.form["object"]["weight"].value = "0.05"
    recon_tab.build_acq_contols()  # a new model of a mode starts from the last one of that mode
    second = recon_tab.models[1]
    assert second.title == "phase - 02"
    assert second.form["object"]["weight"].value == "0.05"
    assert second.output_line.value.endswith("_Pha_02.zarr")

    recon_tab.delete_model(first)
    assert recon_tab.models == [second]
    recon_tab.clear_all_models()
    assert recon_tab.models == []
    assert recon_tab.reconstruction_run_PushButton.text == "RUN Model"


@pytest.mark.parametrize("contrast", ["phase", "fluorescence"])
def test_a_configuration_built_in_the_widget_equals_the_one_the_schema_loads(recon_tab, contrast):
    _check(recon_tab, contrast)
    recon_tab.build_acq_contols()
    (entry,) = recon_tab.models
    config, problems = recon_tab.validated(entry)
    assert problems == []
    assert config == load(template(contrast))
    assert config == load(forms.read_form(entry.form))


def test_validate_frames_a_wrong_field_and_reports_it(recon_tab, messages):
    _check(recon_tab, "phase")
    recon_tab.build_acq_contols()
    (entry,) = recon_tab.models
    aperture = entry.form["imaging_system"]["detection"]["numerical_aperture"]
    aperture.value = "[1, 2"
    assert recon_tab.validate_model(entry) is None
    assert "imaging_system.detection.numerical_aperture" in messages[-1][0]
    assert "255, 255, 0" in aperture.native.styleSheet()
    assert entry.box.title.endswith("⚠️")

    aperture.value = "high"
    assert recon_tab.validate_model(entry) is None
    assert "imaging_system.detection.numerical_aperture" in messages[-1][0]
    aperture.value = "0.9"
    assert recon_tab.validate_model(entry) is not None
    assert aperture.native.styleSheet() == ""
    assert entry.box.title.endswith("✅")


def test_load_migrates_a_3x_file_and_shows_its_notes(recon_tab, messages, tmp_path):
    old = tmp_path / "phase_3x.yml"
    old.write_text(
        "input_channel_names: [BF]\n"
        "time_indices: all\n"
        "reconstruction_dimension: 3\n"
        "phase:\n"
        "  transfer_function: {wavelength_illumination: 0.532, yx_pixel_size: 0.1, z_pixel_size: 0.25}\n"
    )
    (entry,) = recon_tab.open_model_files([str(old)])
    assert entry.mode == "phase"
    assert any("migrated from a 3.x configuration" in text and not error for text, error in messages)
    config, _ = recon_tab.validated(entry)
    assert config is not None
    assert config.imaging_system.illumination.wavelength == pytest.approx(0.532)


def test_load_shows_why_a_file_is_refused(recon_tab, messages, tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("version: 4\nimaging_system: {contrast: phase, nonsense: 1}\n")
    assert recon_tab.open_model_files([str(bad)]) == []
    assert "nonsense" in messages[-1][0]


def test_load_round_trips_a_v4_file(recon_tab, tmp_path):
    from tests.cli.plates import PHASE

    path = tmp_path / "phase.yml"
    path.write_text(PHASE)
    (entry,) = recon_tab.open_model_files([str(path)])
    config, problems = recon_tab.validated(entry)
    assert problems == []
    assert dump(config) == dump(load(path))


def test_validate_runs_the_build_as_wo_config_check_does(recon_tab, messages, tmp_path):
    # The schema takes object.prior: none with closed_form; the build refuses it, and Validate says so.
    raw = template("phase")
    raw["object"] = {"prior": "none", "dimension": raw["object"]["dimension"]}
    raw["solver"]["reconstruction"] = "closed_form"
    path = tmp_path / "none.yml"
    path.write_text(yaml.safe_dump(raw))
    (entry,) = recon_tab.open_model_files([str(path)])
    assert recon_tab.validated(entry)[1] == []  # the schema's check alone passes
    assert recon_tab.validate_model(entry) is None
    assert "solver.reconstruction: " in messages[-1][0]
    assert entry.box.title.endswith("⚠️")


WO40_BEADS = """\
version: 4
data: {input_channel_names: [GFP], pixel_size: {yx: 0.2, z: 0.4}}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.532
    numerical_aperture: 1.2
    pupil_aberrations: {astigmatism_0: {nominal: 0.0, tolerance: free}}
noise: {gaussian: {sigma: 1.0}}
object: {prior: {beads: {diameter: 0.5}}}
solver: {calibration: {score: evidence, search: [{gradient: {max_iter: 150}}]}}
"""
WO40_SIMULATION = """\
version: 4
data: {input_channel_names: [GFP], pixel_size: {yx: 0.2, z: 0.4}}
imaging_system: {contrast: fluorescence, index_of_refraction_media: 1.3, detection: {wavelength: 0.532}}
noise: {poisson: {photons: 50}}
object: {source: {gaussian_spectrum: {exponent: 2.0}}}
simulate: {seed: 0, shape: [16, 32, 32]}
"""


@pytest.mark.parametrize("text", [WO40_BEADS, WO40_SIMULATION], ids=["beads", "simulation"])
def test_a_wo40_file_loaded_into_the_form_builds(recon_tab, messages, tmp_path, text):
    # The form writes every field, so solver.reconstruction: closed_form too, as wo-4.0 wrote it.
    del messages  # a refused load would open a message box
    path = tmp_path / "wo40.yml"
    path.write_text(text)
    (entry,) = recon_tab.open_model_files([str(path)])
    config, problems = recon_tab.validated(entry)
    assert problems == []
    assert config is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfigurationWarning)
        ReconstructionModel.from_config(load(forms.read_form(entry.form)))
