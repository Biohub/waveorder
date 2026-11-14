"""A model whose solver is a distilled student: the system hash it is checked by, and what its positions record."""

import pytest
import torch
import yaml

from tests.models.bundles import phase_bundle
from tests.models.plates import PIXEL_SIZE, SHAPE, contrast, read_output, series, thin_phase_model, write_plate
from waveorder.config import V4Config
from waveorder.errors import IncompatibleModelError
from waveorder.io.bundle import load_bundle
from waveorder.models import FieldTiles, Phase2D, ReconstructionModel
from waveorder.models._stores import system_sha256
from waveorder.nn import Detection, Distilled, Illumination, Wiener

THIN_PHASE = """
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
"""The configuration of :func:`thin_phase_model`."""


def _other_system() -> Phase2D:
    """A thin phase model of another condenser aperture: the same class, channels and fields, another system."""
    return Phase2D(
        Illumination(wavelength=0.532, numerical_aperture=0.4),
        Detection(wavelength=0.532, numerical_aperture=0.9),
        index_of_refraction_media=1.33,
    )


def test_the_system_hash_is_the_same_however_the_model_was_built():
    built = ReconstructionModel.from_config(V4Config(**yaml.safe_load(THIN_PHASE)))
    assert system_sha256(built) == system_sha256(thin_phase_model()) != system_sha256(_other_system())
    assert system_sha256(thin_phase_model(solver=Wiener())) == system_sha256(thin_phase_model())  # parts aside


def test_a_student_of_another_imaging_system_is_refused_when_assigned(tmp_path):
    other = _other_system()
    path = phase_bundle(tmp_path / "student", other)
    model = thin_phase_model()
    before = model.solver
    with pytest.raises(IncompatibleModelError, match="the student was trained for another imaging system") as error:
        model.solver = Distilled(load_bundle(path))
    assert f"records the imaging system sha256 {system_sha256(other)}" in str(error.value)
    assert f"this model's imaging system hashes to {system_sha256(model)}" in str(error.value)
    assert model.solver is before
    with pytest.raises(IncompatibleModelError, match="trained for another imaging system"):
        thin_phase_model(solver=Distilled(load_bundle(path)))  # at construction


def test_a_student_of_this_imaging_system_is_taken_and_check_compares_the_hash_again(tmp_path):
    model = thin_phase_model(solver=Distilled(load_bundle(phase_bundle(tmp_path / "student"))))
    model.check(SHAPE, pixel_size=PIXEL_SIZE)
    model.field_tiles = FieldTiles((16, 16))  # imaging_system.field_dependence is part of the system's record
    with pytest.raises(IncompatibleModelError, match="trained for another imaging system"):
        model.check(SHAPE, pixel_size=PIXEL_SIZE)


def test_a_contract_that_records_no_system_hash_is_not_compared(tmp_path):
    path = phase_bundle(tmp_path / "student", _other_system(), hashed=False)
    model = thin_phase_model(solver=Distilled(load_bundle(path)))
    model.check(SHAPE, pixel_size=PIXEL_SIZE)


def test_output_positions_record_the_bundle_of_a_distilled_solver(tmp_path):
    path = phase_bundle(tmp_path / "student")
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(contrast(), 1)])
    model = thin_phase_model(solver=Distilled(load_bundle(path)))
    output = tmp_path / "out.zarr"
    report = model.reconstruct_positions(plate, output=output, channels=["BF"])
    assert (report.status, report.count("complete")) == ("finished", 1)
    names, image, attrs = read_output(output)
    assert names == ["Phase2D", "Absorption2D"]
    assert image.shape == (1, 2, 1, 32, 32)
    assert torch.equal(image[0, 1], torch.zeros_like(image[0, 1]))  # the student returns phase only
    record = attrs["waveorder-Phase2D"]
    assert set(record) == {"model", "bundle", "versions", "host"}  # a model built in Python: its description
    assert record["bundle"] == {"path": str(path.absolute()), "sha256": load_bundle(path).sha256}
    assert record["model"]["parts"]["solver"] == {"distilled": {"path": str(path), "precision": "float32"}}
