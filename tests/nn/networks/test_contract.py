"""The contract: what a student takes and returns, recorded from its model and replayed on every solve."""

import json

import pytest

from tests.models.plates import thin_phase_model
from tests.nn.conftest import make_model
from waveorder.errors import ParameterError
from waveorder.nn import Sampling
from waveorder.nn.networks import Contract

PIXEL_SIZE = (0.4, 0.15, 0.15)


def _contract(**changes: object) -> Contract:
    model = thin_phase_model().imaging_model
    settings: dict = {"pixel_size": PIXEL_SIZE, "model": "Phase2D", "prepare_scope": "block", **changes}
    return Contract.of(model, **settings)


def test_a_contract_reads_channels_fields_and_system_from_the_model():
    contract = _contract(planes=(5, 7), sha256="ab12")
    assert (contract.channels, contract.planes, contract.fields) == (1, (5, 7), ("phase", "absorption"))
    assert contract.system == {"class": "ThinPhase2D", "sha256": "ab12"}
    assert _contract(fields=("phase",)).fields == ("phase",)  # a recorded plate may hold a subset
    with pytest.raises(ParameterError, match=r"fields \['density'\] are not fields of ThinPhase2D"):
        _contract(fields=("density",))


def test_data_sampled_otherwise_than_the_training_data_is_named():
    contract = _contract(planes=(5,))
    assert contract.sampling_problems(Sampling(shape=(5, 64, 64), pixel_size=(0.41, 0.15, 0.156))) == []
    (planes, size) = contract.sampling_problems(Sampling(shape=(4, 64, 64), pixel_size=(0.4, 0.18, 0.15)))
    assert planes == "the data has 4 plane(s), but the student takes 5"
    assert size.startswith("the pixel size along y is 0.18 um, but the student was trained at 0.15 um (tolerance 5%)")


def test_a_model_the_student_was_not_distilled_from_is_named():
    contract = _contract(fields=("absorption",))
    assert contract.model_problems(thin_phase_model().imaging_model) == []
    problems = contract.model_problems(make_model())
    assert problems == [
        "the student returns the field(s) absorption, which WidefieldFluorescence3D does not have (its fields are "
        "fluorescence)",
        "the student was distilled from a ThinPhase2D system, but the model's system is a WidefieldFluorescence3D",
    ]


def test_a_contract_survives_json():
    contract = _contract(planes=(5,), fields=("phase",), pixel_size_tolerance=0.1)
    assert Contract.from_dict(json.loads(json.dumps(contract.as_dict()))) == contract
    with pytest.raises(ParameterError, match="not a contract"):
        Contract.from_dict({**contract.as_dict(), "colour": "red"})
    with pytest.raises(ParameterError, match="pixel_size must be three positive numbers"):
        Contract.from_dict({**contract.as_dict(), "pixel_size": [0.4, 0.15]})
    with pytest.raises(ParameterError, match="pixel_size_tolerance must be in"):
        Contract.from_dict({**contract.as_dict(), "pixel_size_tolerance": 1.5})
    with pytest.raises(ParameterError, match="not a contract"):  # a file's wrongly typed value is refused too
        Contract.from_dict({**contract.as_dict(), "pixel_size": ["0.4", "0.15", "0.15"]})
