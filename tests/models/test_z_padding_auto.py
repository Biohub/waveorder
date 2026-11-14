"""``z_padding="auto"``: at least the axial extent of the point-spread function, grown to a fast FFT length.

It is resolved for each volume (``solver.padding.z: auto`` in a configuration); an integer pads as before.
"""

from typing import Any

import pydantic
import pytest
import torch
import yaml

from tests.models.plates import PIXEL_SIZE, SHAPE, contrast, phase_model, read_output, series, write_plate
from waveorder.config import V4Config
from waveorder.errors import CalibrationWarning, ParameterError, ShapeError
from waveorder.models import ReconstructionModel, _base
from waveorder.nn import Sampling, ZPadding, load_parameters
from waveorder.nn.search import sweep
from waveorder.nn.search.sweep import WeightSweep

FLUORESCENCE = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {z: 0.25, y: 0.1, x: 0.1}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection:
    wavelength: 0.52
    numerical_aperture: {nominal: 1.2, tolerance: free, bounds: [1.0, 1.3]}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
solver:
  padding: {z: auto}
"""

NA = "system.detection.numerical_aperture"


def _fluorescence(**object_settings: object) -> ReconstructionModel:
    raw = yaml.safe_load(FLUORESCENCE)
    raw["object"].update(object_settings)
    return ReconstructionModel.from_config(V4Config(**raw))


def test_auto_pads_the_axial_extent_then_grows_to_a_fast_fft_length():
    # phase_model: n = 1.33, wavelength 0.532 um, detection NA 0.9, so n * wavelength / NA**2 = 0.87353 um.
    model = phase_model(z_padding="auto")
    assert model.z_padding == "auto"
    # z step 0.1 um: 8.74 slices, at least 9. 20 + 2 * 9 = 38 = 2 * 19 is no fast length; 20 + 2 * 10 = 40 = 2**3 * 5.
    assert model.z_padding_for((20, 32, 32), pixel_size=(0.1, 0.15, 0.15)) == 10
    # z step 0.4 um: 2.18 slices, at least 3; 8 + 2 * 3 = 14 = 2 * 7 is fast already.
    assert model.z_padding_for(SHAPE, pixel_size=PIXEL_SIZE) == 3


def test_auto_reads_the_nominal_detection_aperture_of_a_configuration():
    # n = 1.33, emission 0.52 um, nominal NA 1.2: 1.33 * 0.52 / 1.44 = 0.48028 um, 1.92 slices of 0.25 um, at least 2.
    # 9 + 2 * 2 = 13 is prime; 9 + 2 * 3 = 15 = 3 * 5.
    model = _fluorescence()
    assert model.z_padding == "auto"
    assert model.config is not None
    assert model.config.solver.padding.z == "auto"
    assert model.z_padding_for((9, 16, 16)) == 3  # the voxel size of data.pixel_size
    load_parameters(model.imaging_model, {NA: 1.0})  # a calibrated value does not move the padding
    assert model.z_padding_for((9, 16, 16)) == 3


def test_auto_solves_as_the_integer_it_resolves_to():
    data = contrast()  # (8, 32, 32) at PIXEL_SIZE: auto is 3 slices
    auto = phase_model(z_padding="auto").reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    fixed = phase_model(z_padding=3).reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    unpadded = phase_model().reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    torch.testing.assert_close(auto, fixed, rtol=0, atol=0)
    assert (auto - unpadded).norm() > 1e-3 * unpadded.norm()


def test_an_integer_pads_as_before():
    model = phase_model(z_padding=3)
    assert model.z_padding == 3
    assert model.z_padding_for((5, 4, 4)) == 3  # an integer needs no voxel size and is the same on every stack
    data = contrast()
    padding = ZPadding(3)
    padded = padding.sampling(Sampling(shape=SHAPE, pixel_size=PIXEL_SIZE))
    operator = model.system.bind(padded) if model.solver.needs_operator else None
    expected = padding.crop(model.solver(padding.pad(data[None]), model.imaging_model, padded, operator=operator))
    estimate = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    torch.testing.assert_close(estimate, expected.estimate, rtol=0, atol=0)


def test_tiles_and_stored_transfers_take_the_resolved_slices():
    model = phase_model(z_padding="auto")
    data = contrast()[None]
    whole = model.reconstruct(data, pixel_size=PIXEL_SIZE, prepared=True).estimate
    sink = torch.zeros_like(whole)
    plan = model.tile_plan(SHAPE, pixel_size=PIXEL_SIZE, tile=SHAPE)
    model.reconstruct_tiles([(data, sink)], pixel_size=PIXEL_SIZE, plan=plan, prepared=True)
    assert (sink - whole).norm() <= 1e-5 * whole.norm()
    (item,) = model.to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE).transfers
    assert item.shape == (8 + 2 * 3, 32, 32)
    assert item.attrs["z_padding"] == 3


def test_the_output_record_holds_the_slices_auto_chose(tmp_path):
    volumes = [series(contrast(), 1), series(-contrast(), 1)]
    plate = tmp_path / "in.zarr"
    write_plate(plate, volumes)
    model = phase_model(z_padding="auto")
    model.reconstruct_positions(plate, output=tmp_path / "auto.zarr", channels=["BF"])
    _, image, attrs = read_output(tmp_path / "auto.zarr")
    assert attrs["waveorder-Phase3D"]["z_padding"] == 3
    fixed = phase_model(z_padding=3)
    torch.testing.assert_close(image[0], fixed.reconstruct(volumes[0][0][None], pixel_size=PIXEL_SIZE).estimate)
    fixed.reconstruct_positions(plate, output=tmp_path / "fixed.zarr", channels=["BF"])
    _, _, attrs = read_output(tmp_path / "fixed.zarr")
    assert "z_padding" not in attrs["waveorder-Phase3D"]  # an integer's record is as before


def test_a_weight_sweep_leaves_out_the_slices_auto_chose(monkeypatch):
    model = _fluorescence(weight={"auto": {"rule": "otsu_cnr", "num_samples": 3}})
    stages = model.search
    assert stages is not None
    (stage,) = stages
    assert isinstance(stage, WeightSweep)
    assert stage.z_padding == 0  # the configuration's stage holds no slices; the calibration gives them
    searched, depths = [], []
    real_calibrate, real_statistics = _base.calibrate, sweep._slice_statistics  # noqa: SLF001 (what Otsu scores)

    def spied_calibrate(*args: Any, search: Any, **kwargs: Any) -> object:
        searched.extend(search)
        return real_calibrate(*args, search=search, **kwargs)

    def spied_statistics(volume: torch.Tensor) -> Any:
        depths.append(volume.shape[0])
        return real_statistics(volume)

    monkeypatch.setattr("waveorder.models._base.calibrate", spied_calibrate)
    monkeypatch.setattr(sweep, "_slice_statistics", spied_statistics)
    with pytest.warns(CalibrationWarning) as caught:  # random data may also land the pick on a bound of the sweep
        model.calibrate(torch.rand(1, 9, 16, 16, generator=torch.Generator().manual_seed(0)), prepared=True)
    assert any("padding.z is auto" in str(warning.message) for warning in caught)
    (swept,) = searched
    assert isinstance(swept, WeightSweep)
    assert swept.z_padding == 3
    assert depths == [9] * 3  # each of the 3 reconstructions, on 9 + 2 * 3 slices, is scored on the measured 9


def test_auto_refuses_a_stack_shorter_than_its_padding_and_other_strings():
    model = phase_model(z_padding="auto")
    # 0.87353 um at a z step of 0.1 um is 9 slices, which a stack of 8 cannot mirror.
    with pytest.raises(ShapeError, match=r"z_padding 'auto' pads at least 9 slices .* the stack has 8"):
        model.reconstruct(contrast(), pixel_size=(0.1, 0.15, 0.15), prepared=True)
    with pytest.raises(ParameterError, match="z_padding must be at least 0, or 'auto', got 'big'") as error:
        model.z_padding = "big"  # ty: ignore[invalid-assignment]
    assert error.value.path == "z_padding"
    raw = yaml.safe_load(FLUORESCENCE)
    raw["solver"]["padding"]["z"] = "big"
    with pytest.raises(pydantic.ValidationError, match=r"padding\.z"):
        V4Config(**raw)
