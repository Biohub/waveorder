"""Integration tests that click the Reconstruction tab's buttons as a user does."""

import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from waveorder.config import dump, load
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


def test_click_new_button_fluorescence(recon_tab):
    _check(recon_tab, "fluorescence")
    recon_tab.build_acq_contols()
    (entry,) = recon_tab.models
    assert entry.form["imaging_system"]["contrast"].value == "fluorescence"


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
