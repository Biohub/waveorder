"""The form must not switch a feature on just because its settings block is optional.

``solver.calibration: null`` in a configuration means "do not calibrate". These tests drive the schema-to-form
traversals of :mod:`waveorder.plugin.forms` directly, without a napari viewer.
"""

import pytest

pytest.importorskip("magicgui")
pytest.importorskip("qtpy.QtWidgets")

from waveorder.config import dump, load
from waveorder.plugin import forms

PHASE = {
    "version": 4,
    "data": {"input_channel_names": ["BF"], "pixel_size": {"z": 0.4, "y": 0.15, "x": 0.15}},
    "imaging_system": {"contrast": "phase", "illumination": {"wavelength": 0.532, "numerical_aperture": 0.5}},
}


def _round_trip(values):
    """Build the form for ``values`` and collect what it would submit, untouched."""
    form = forms.build_form("model", values)
    return form, forms.read_form(form)


def _toggle(form):
    return form["solver"]["calibration"][forms.OPTIONAL_BLOCK_TOGGLE]


def test_untouched_optional_block_is_submitted_as_none(qapp):
    del qapp
    form, submitted = _round_trip(dump(load(PHASE)))
    assert _toggle(form).value is False
    assert submitted["solver"]["calibration"] is None
    assert load(submitted).solver.calibration is None  # and the submission parses with the block off


def test_optional_block_present_in_the_config_comes_back_populated(qapp):
    del qapp
    config = {
        **PHASE,
        "imaging_system": {
            "contrast": "phase",
            "illumination": {"wavelength": 0.532, "numerical_aperture": {"nominal": 0.5, "bounds": [0.3, 0.7]}},
        },
        "solver": {"calibration": {"score": "gcv", "search": [{"random": {"samples": 4}}]}},
    }
    form, submitted = _round_trip(dump(load(config)))
    assert _toggle(form).value is True
    block = submitted["solver"]["calibration"]
    assert block["score"] == "gcv"
    assert block["search"] == [{"random": {"samples": 4, "seed": 0}}]
    assert load(submitted) == load(config)


def test_switching_the_block_on_submits_its_defaults(qapp):
    del qapp
    form = forms.build_form("model", dump(load(PHASE)))
    _toggle(form).value = True
    calibration = load(forms.read_form(form)).solver.calibration
    assert calibration is not None
    assert calibration.score == "evidence"
