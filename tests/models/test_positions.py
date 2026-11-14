"""``reconstruct_positions``: the checks, the output plate, the units, resume and the report."""

import re
from typing import Any, Literal

import iohub
import pytest
import torch
import yaml

from tests.models.plates import (
    PIXEL_SIZE,
    contrast,
    fluorescence_model,
    phase_model,
    read_output,
    series,
    thin_phase_model,
    write_plate,
)
from waveorder.config import V4Config
from waveorder.config.schema import PixelSize
from waveorder.errors import PositionWarning, Problems, Rejected, WaveorderError, argument_names
from waveorder.models import (
    FieldTiles,
    Fluorescence3D,
    ReconstructionFailed,
    ReconstructionModel,
    _pool,
    _positions,
)
from waveorder.models._checks import Arguments, Plan, check_positions
from waveorder.nn import (
    FISTA,
    Detection,
    Free,
    Gaussian,
    GridSearch,
    ThickPhase3D,
    Tikhonov,
    TotalVariation,
    free_specs,
)
from waveorder.observe import observe

FLUORESCENCE = """\
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {z: 0.4, y: 0.15, x: 0.15}
imaging_system:
  contrast: fluorescence
  index_of_refraction_media: 1.33
  detection: {wavelength: 0.52, numerical_aperture: 0.8}
object: {prior: tikhonov, weight: 1e-3}
noise: {gaussian: {sigma: 0.01}}
"""


@pytest.fixture(scope="module")
def volume() -> torch.Tensor:
    """Contrast of a simulated phase object, ``(Z, Y, X)``."""
    return contrast()


@pytest.fixture
def volumes(volume) -> list[torch.Tensor]:
    """The recorded intensities of two positions of three time points each."""
    return [series(volume, 3), series(-volume, 3)]


@pytest.fixture
def plate(tmp_path, volumes):
    path = tmp_path / "in.zarr"
    write_plate(path, volumes)
    return path


def _close(image: torch.Tensor, expected: torch.Tensor) -> bool:
    return torch.allclose(image, expected.to(image), rtol=1e-4, atol=1e-5 * float(expected.abs().max()))


def _plan(model: ReconstructionModel, positions: Any, **arguments: Any) -> Plan:
    """Return what the checks of ``reconstruct_positions`` decide, writing nothing."""
    problems = Problems()
    plan = check_positions(model, Arguments(positions=positions, **arguments), problems=problems)
    problems.raise_if_any()
    return plan


class Smooth(Tikhonov):
    """A prior defined here, which no configuration describes."""


class Offset(Fluorescence3D):
    """Fluorescence with its own ``prepare``, declared as depending on the block it is given."""

    prepare_scope = "block"

    def prepare(self, data, *, mean=None):
        del mean
        return data - 100.0


class VoxelOffset(Offset):
    """The same ``prepare``, declared as what it is: voxel by voxel, so tiles need nothing shared."""

    prepare_scope = "voxel"


def _offset(kind: type[Offset]) -> Offset:
    return kind(
        Detection(wavelength=0.52, numerical_aperture=0.8),
        index_of_refraction_media=1.33,
        object_prior=Tikhonov(weight=1e-3),
        noise=Gaussian(sigma=0.01),
    )


class _Events:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    def event(self, name, payload) -> None:
        self.seen.append((name, dict(payload)))


def test_a_plate_is_reconstructed_and_each_unit_equals_the_models_reconstruction(plate, volumes, tmp_path):
    model = phase_model()
    output = tmp_path / "out.zarr"
    report = model.reconstruct_positions(plate, output=output, channels=["BF"])
    assert (report.status, report.total, report.count("complete")) == ("finished", 6, 6)
    assert report.output == str(output)
    assert [result.unit for result in report.results] == [f"A/1/{p}/t{t}" for p in range(2) for t in range(3)]
    assert all(result.wall_s > 0 and result.host and result.device == "cpu" for result in report.results)
    assert all(result.solve_s is not None and result.solve_s > 0 for result in report.results)
    for index, name in enumerate(("A/1/0", "A/1/1")):
        names, image, _ = read_output(output, name)
        assert names == ["Phase3D"]
        assert image.shape == (3, 1, 8, 32, 32)
        for t in range(3):
            expected = model.reconstruct(volumes[index][t][None], pixel_size=PIXEL_SIZE).estimate
            assert _close(image[t], expected)


def test_open_iohub_nodes_reconstruct_as_their_paths_and_stay_open(plate, tmp_path):
    model = phase_model()

    def output_of(inputs: Any, name: str) -> list[torch.Tensor]:
        output = tmp_path / f"{name}.zarr"
        report = model.reconstruct_positions(inputs, output=output, channels=["BF"], times=[0])
        assert report.status == "finished"
        return [read_output(output, position)[1] for position in ("A/1/0", "A/1/1")]

    expected = output_of(plate, "path")
    with iohub.open_ome_zarr(plate, mode="r") as node, iohub.open_ome_zarr(plate / "A/1/1", mode="r") as position:
        given = {
            "plate": node,
            "positions": [node["A/1/0"], position],
            "mixed": [str(plate / "A/1/0"), position],
            "glob": f"{plate}/A/1/*",
        }
        for name, inputs in given.items():
            assert all(torch.equal(a, b) for a, b in zip(output_of(inputs, name), expected, strict=True)), name
        assert position["0"][0, 0].shape == (8, 32, 32)  # the caller's nodes are still open
        assert node["A/1/0"]["0"][0, 0].shape == (8, 32, 32)
        free = phase_model(object_prior=Tikhonov(weight=Free(4e-4, bounds=(1e-6, 1.0))), search=[GridSearch(points=5)])
        assert free.calibrate_position(node, channels=["BF"], dry_run=True) is None  # a plate: its first position
        from_node = free.calibrate_position(position, channels=["BF"])
        from_path = free.calibrate_position(plate / "A/1/1", channels=["BF"])
        assert from_node is not None
        assert from_path is not None
        assert from_node.row() == from_path.row()
        assert position["0"].shape == (3, 1, 8, 32, 32)


def test_output_positions_record_3x_names_provenance_and_the_inputs_metadata(plate, tmp_path):
    output = tmp_path / "out.zarr"
    phase_model().reconstruct_positions(plate, output=output, channels=["BF"], times=[0])
    _, _, attrs = read_output(output, "A/1/1")
    record = attrs["waveorder-Phase3D"]
    assert set(record) == {"model", "versions", "host"}  # a model built in Python records its description
    assert record["model"]["data"]["input_channel_names"] == ["BF"]
    assert record["model"]["data"]["pixel_size"] == {"z": 0.4, "y": 0.15, "x": 0.15}
    assert set(record["versions"]) == {"waveorder", "iohub", "torch"}
    assert attrs["biahub-deskew"] == {"angle": 30}  # the input position's non-OME attributes
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert node.zattrs["acquisition"] == {"microscope": "test"}  # the input plate root's
        assert node.version == "0.5"
        assert list(node["A/1/0"].scale[2:]) == pytest.approx(list(PIXEL_SIZE))


def test_a_model_with_a_part_no_configuration_describes_records_its_class_names(plate, tmp_path):
    model = fluorescence_model(object_prior=Smooth(weight=1e-3))
    output = tmp_path / "out.zarr"
    report = model.reconstruct_positions(str(plate / "A/1/0"), output=output, channels=["BF"], times=[0])
    assert report.count("complete") == 1
    record = read_output(output)[2]["waveorder-BF_Density3D"]
    assert set(record) == {"model", "versions", "host"}
    assert record["model"] == {
        "class": "Fluorescence3D",
        "system": "WidefieldFluorescence3D",
        "object_prior": "Smooth",
        "noise": "Gaussian",
        "solver": "Wiener",
    }


def test_a_model_built_from_a_configuration_reconstructs_and_records_it(plate, volumes, tmp_path):
    model = ReconstructionModel.from_config(V4Config(**yaml.safe_load(FLUORESCENCE)))
    output = tmp_path / "out.zarr"
    report = model.reconstruct_positions(str(plate / "A/1/0"), output=output, channels=["BF"])
    assert report.count("complete") == 3
    names, image, attrs = read_output(output)
    assert names == ["BF_Density3D"]
    assert attrs["waveorder-BF_Density3D"]["config"]["imaging_system"]["contrast"] == "fluorescence"
    assert _close(image[2], model.reconstruct(volumes[0][2][None], pixel_size=PIXEL_SIZE).estimate)


@pytest.mark.parametrize(
    ("build", "shape", "key"),
    [
        (thin_phase_model, (3, 2, 1, 32, 32), "waveorder-Phase2D"),
        (lambda: phase_model(z_padding=2), (3, 1, 8, 32, 32), "waveorder-Phase3D"),
    ],
)
def test_thin_and_z_padded_models_solve_whole_volumes(plate, volumes, tmp_path, build, shape, key):
    model = build()
    output = tmp_path / "out.zarr"
    planned = _plan(model, plate, output=output, channels=["BF"])
    assert {grid.path for grid in planned.grids.values()} == {"whole"}
    report = model.reconstruct_positions(plate, output=output, channels=["BF"])
    assert report.count("complete") == 6
    names, image, attrs = read_output(output)
    assert tuple(image.shape) == shape
    assert key in attrs
    assert names == (["Phase2D", "Absorption2D"] if shape[1] == 2 else ["Phase3D"])
    assert image.abs().amax(dim=(0, 2, 3, 4)).min() > 0
    assert _close(image[1], model.reconstruct(volumes[0][1][None], pixel_size=PIXEL_SIZE).estimate)


def test_tiles_reconstruct_edge_rows_and_stay_close_to_the_whole_volume(tmp_path, volume):
    tall = torch.cat([volume, volume[:, :8]], dim=1)  # Y = 40: rows of 16, 16 and 8
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(tall, 2), series(-tall, 2)])
    model = phase_model()
    tiled_output, whole_output = tmp_path / "tiled.zarr", tmp_path / "whole.zarr"
    report = model.reconstruct_positions(
        plate, output=tiled_output, channels=["BF"], tile=(8, 16, 16), tile_overlap=(0, 4, 4)
    )
    assert report.count("complete") == 4
    assert report.results[0].solve_s is not None
    model.reconstruct_positions(plate, output=whole_output, channels=["BF"])
    for position in ("A/1/0", "A/1/1"):
        _, tiled, _ = read_output(tiled_output, position)
        _, untiled, _ = read_output(whole_output, position)
        assert tiled.shape == (2, 1, 8, 40, 32)
        assert tiled[:, :, :, 32:].abs().max() > 0  # the short edge row is written
        assert torch.corrcoef(torch.stack([tiled.flatten(), untiled.flatten()]))[0, 1] > 0.9
    with iohub.open_ome_zarr(tiled_output / "A/1/0", mode="r") as position:
        assert tuple(position["0"].chunks)[:2] == (1, 1)  # one time point and channel per block
    with pytest.raises(Rejected, match="tile_overlap widens tile cores; give tile too"):
        model.reconstruct_positions(plate, output=tmp_path / "no.zarr", channels=["BF"], tile_overlap=(0, 4, 4))


def test_a_z_padded_model_tiles_in_y_and_x_when_its_tiles_span_z(plate, volumes, tmp_path):
    model = phase_model(z_padding=2)
    tile, overlap = (8, 16, 16), (0, 4, 4)
    output = tmp_path / "out.zarr"
    planned = _plan(model, plate, output=output, channels=["BF"], tile=tile)
    assert {grid.path for grid in planned.grids.values()} == {"tiles"}
    report = model.reconstruct_positions(plate, output=output, channels=["BF"], tile=tile, tile_overlap=overlap)
    assert report.count("complete") == 6
    _, tiled, _ = read_output(output)
    whole = model.reconstruct(volumes[0][1][None], pixel_size=PIXEL_SIZE).estimate
    assert torch.corrcoef(torch.stack([tiled[1].flatten(), whole.flatten()]))[0, 1] > 0.9
    with pytest.raises(Rejected, match=r"tile splits z \(4 of 8 slices\), but z_padding pads whole stacks"):
        model.reconstruct_positions(plate, output=tmp_path / "no.zarr", channels=["BF"], tile=(4, 16, 16))


@pytest.mark.parametrize("tiles", [{}, {"tile": (8, 16, 16), "tile_overlap": (0, 4, 4)}], ids=["whole", "tiles"])
def test_units_reuse_the_transfer_function_the_checks_computed(plate, tmp_path, monkeypatch, tiles):
    # Job 38077265: the tiled neuromast case computed its transfer function three times on the tile grid, and every
    # whole volume twice, at the checks' bind and again in the unit.
    model = phase_model(z_padding=2)
    system = ThickPhase3D
    assert isinstance(model.system, system)
    computed = []
    compute = system.compute_transfer
    monkeypatch.setattr(
        system, "compute_transfer", lambda self, sampling: computed.append(sampling) or compute(self, sampling)
    )
    model.reconstruct_positions(plate, output=tmp_path / "plan.zarr", channels=["BF"], dry_run=True, **tiles)
    checked = list(computed)
    report = model.reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"], **tiles)
    assert report.count("complete") == 6
    assert computed[len(checked) :] == checked  # the six units computed none
    assert model.system.cache == "off"
    assert model.system._cache_entry is None  # noqa: SLF001 (the run keeps no transfer function on the model)


def test_a_prepare_of_block_scope_cannot_tile_and_a_voxel_one_can(plate, tmp_path):
    tile, overlap = (8, 16, 16), (0, 4, 4)
    with pytest.raises(Rejected, match=r"Offset\.prepare is given one block at a time") as error:
        _offset(Offset).reconstruct_positions(
            plate, output=tmp_path / "no.zarr", channels=["BF"], tile=tile, tile_overlap=overlap, dry_run=True
        )
    message = str(error.value)
    assert "its prepare_scope is 'block'" in message
    assert "set the class attribute prepare_scope to 'voxel' when prepare acts voxel by voxel" in message
    whole = _offset(Offset).reconstruct_positions(plate, output=tmp_path / "whole.zarr", channels=["BF"])
    assert whole.count("complete") == 6  # one solve per volume: the block is the volume
    report = _offset(VoxelOffset).reconstruct_positions(
        plate, output=tmp_path / "tiled.zarr", channels=["BF"], tile=tile, tile_overlap=overlap
    )
    assert (report.status, report.count("complete")) == ("finished", 6)
    _, untiled, _ = read_output(tmp_path / "whole.zarr")
    _, split, _ = read_output(tmp_path / "tiled.zarr")
    assert torch.corrcoef(torch.stack([split.flatten(), untiled.flatten()]))[0, 1] > 0.9


def test_field_tiles_solve_each_tile_with_its_own_parameters(plate, tmp_path):
    detection = Detection(wavelength=0.52, numerical_aperture=Free(0.8, bounds=(0.6, 1.0)))
    model = fluorescence_model(detection, field_tiles=FieldTiles(size_yx=(16, 16), overlap_yx=(0, 0)))
    (name,) = free_specs(model.imaging_model)
    one = str(plate / "A/1/0")
    planned = _plan(model, one, output=tmp_path / "plain.zarr", channels=["BF"], times=[0])
    assert [grid.path for grid in planned.grids.values()] == ["field_tiles"]
    outputs = {}
    for label, table in (("plain", None), ("uniform", [0.8] * 4), ("varied", [0.65, 0.8, 0.8, 0.8])):
        model.tile_parameters = None if table is None else {name: torch.tensor(table)}
        report = model.reconstruct_positions(one, output=tmp_path / f"{label}.zarr", channels=["BF"], times=[0])
        assert report.count("complete") == 1
        outputs[label] = read_output(tmp_path / f"{label}.zarr")[1][0, 0]
    assert torch.allclose(outputs["uniform"], outputs["plain"], atol=1e-6)  # a uniform table is the plain run
    changed = (outputs["varied"] - outputs["plain"]).abs()
    assert changed[:, :16, :16].max() > 0  # the first tile took its own aperture
    assert changed[:, 16:, 16:].max() == 0  # the last one the same as before
    model.tile_parameters = {name: torch.full((3,), 0.8)}
    with pytest.raises(Rejected) as error:
        model.reconstruct_positions(one, output=tmp_path / "no.zarr", channels=["BF"], tile=(8, 16, 16))
    messages = [problem.message for problem in error.value.problems]
    assert any("tile_parameters hold 3 tiles" in message and "into 4" in message for message in messages)
    assert any("field tiles come from the model's field_tiles; drop tile" in message for message in messages)
    assert not (tmp_path / "no.zarr").exists()


def test_times_select_the_units(plate, tmp_path):
    output = tmp_path / "out.zarr"
    report = phase_model().reconstruct_positions(str(plate / "A/1/1"), output=output, channels=["BF"], times=[2, 0])
    assert [result.unit for result in report.results] == ["A/1/1/t2", "A/1/1/t0"]
    _, image, _ = read_output(output, "A/1/1")
    assert image[0].abs().max() > 0
    assert image[1].abs().max() == 0  # not selected: left as created
    assert image[2].abs().max() > 0
    with pytest.raises(Rejected, match="times selects 7, outside the 3 time points of the data"):
        phase_model().reconstruct_positions(plate, output=tmp_path / "no.zarr", channels=["BF"], times=[1, 7])


def test_one_time_index_and_the_schemas_pixel_size_are_taken_as_they_are(plate, tmp_path):
    model = phase_model()
    size = PixelSize(z=0.4, y=0.15, x=0.15)  # the form a configuration's data block holds
    accepted = (size, size.model_dump(), list(PIXEL_SIZE), torch.tensor(PIXEL_SIZE, dtype=torch.float64), "auto", None)
    for pixel_size in accepted:
        planned = _plan(model, plate, output=tmp_path / "plan.zarr", channels=["BF"], times=1, pixel_size=pixel_size)
        assert list(planned.times.values()) == [(1,), (1,)]
        assert tuple(planned.pixel_sizes["A/1/0"]) == pytest.approx(PIXEL_SIZE)
    output = tmp_path / "out.zarr"
    report = model.reconstruct_positions(str(plate / "A/1/0"), output=output, channels=["BF"], times=2, pixel_size=size)
    assert [result.unit for result in report.results] == ["A/1/0/t2"]
    refusals = ("0.4", {"z": 0.4, "yx": 0.15}, {"z": 0.4, "yx": 0.15, "x": 0.15}, (0.4, 0.15, 0.15, 1.0))
    for refused in refusals:
        with pytest.raises(Rejected, match=r"pixel_size must be positive numbers in um") as error:
            model.reconstruct_positions(
                plate,
                output=output,
                channels=["BF"],
                pixel_size=refused,
                dry_run=True,
            )
        assert [problem.location for problem in error.value.problems] == ["pixel_size"]


def test_an_isotropic_size_and_one_of_z_and_yx_are_taken(plate, tmp_path):
    model = phase_model()
    with pytest.warns(PositionWarning):  # 0.15 um along z too: not what the positions record
        isotropic = model.reconstruct_positions(plate, output=tmp_path / "a.zarr", channels=["BF"], pixel_size=0.15)
    assert read_output(tmp_path / "a.zarr")[1].abs().max() > 0
    assert isotropic.count("complete") == 6
    split = _plan(model, plate, output=tmp_path / "b.zarr", channels=["BF"], pixel_size=(0.4, 0.15))
    assert tuple(split.pixel_sizes["A/1/0"]) == pytest.approx(PIXEL_SIZE)


def test_a_model_with_its_own_voxel_size_reconstructs_positions_that_record_none(tmp_path, volume):
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(volume, 1)], pixel_size=(1.0, 1.0, 1.0))
    model = phase_model(pixel_size=PIXEL_SIZE)
    with pytest.warns(PositionWarning, match="records \\(1, 1, 1\\) um"):  # the stated size is used
        report = model.reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"])
    assert report.count("complete") == 1
    expected = model.reconstruct(series(volume, 1)[0][None]).estimate
    assert _close(read_output(tmp_path / "out.zarr")[1][0], expected)


def test_a_position_that_records_no_voxel_size_needs_one_stated(tmp_path, volume):
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(volume, 1)], pixel_size=(1.0, 1.0, 1.0))
    model = phase_model()
    for automatic in (None, "auto"):
        with pytest.raises(Rejected, match=r"records no voxel size \(its scale is 1, 1, 1\); set pixel_size") as error:
            model.reconstruct_positions(plate, output=tmp_path / "no.zarr", channels=["BF"], pixel_size=automatic)
        assert error.value.problems[0].location == str((plate / "A/1/0").resolve())
    assert not (tmp_path / "no.zarr").exists()
    with pytest.warns(PositionWarning):
        report = model.reconstruct_positions(
            plate, output=tmp_path / "out.zarr", channels=["BF"], pixel_size=PIXEL_SIZE
        )
    assert report.count("complete") == 1


def test_a_refused_request_is_a_waveorder_error(plate, tmp_path):
    with pytest.raises(WaveorderError) as error:
        phase_model().reconstruct_positions(plate, output=tmp_path / "no.zarr", channels=["DAPI"])
    assert isinstance(error.value, Rejected)
    assert isinstance(error.value, ValueError)  # as every other error about what a caller passed
    assert issubclass(ReconstructionFailed, WaveorderError)
    assert ReconstructionFailed.__module__ == "waveorder.errors"  # exceptions live there; models exports this one


def _corrupt(files) -> None:
    for file in files:
        file.write_bytes(b"not a chunk")


def _chunks(position) -> list:
    files = (file for file in (position / "0").rglob("*") if file.is_file())
    return sorted(file for file in files if not file.name.startswith(("z", ".")))  # not the array's metadata


def test_a_unit_whose_data_cannot_be_read_fails_as_a_read_error(plate, tmp_path):
    _corrupt(_chunks(plate / "A/1/0"))
    report = phase_model().reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"], keep_going=True)
    failed = [result for result in report.results if result.status == "failed"]
    assert [result.unit for result in failed] == ["A/1/0/t0", "A/1/0/t1", "A/1/0/t2"]
    assert {result.error_type for result in failed} == {"ReadError"}
    assert (failed[1].error or "").startswith(f"ReadError: could not read time 1 of {(plate / 'A/1/0').resolve()}: ")
    assert report.count("complete") == 3  # the other position


def test_a_read_that_fails_inside_the_tile_loader_is_a_read_error_too(tmp_path, volume):
    plate = tmp_path / "in.zarr"
    with iohub.open_ome_zarr(plate, layout="hcs", mode="w-", channel_names=["BF"]) as node:
        position = node.create_position("A", "1", "0")
        data = series(volume, 1)[:, None].to(torch.float32).numpy()
        position.create_image("0", data, chunks=(1, 1, 2, 32, 32))  # four chunks along z
        for axis, size in zip("zyx", PIXEL_SIZE, strict=True):
            position.set_scale("0", axis, size)
    chunks = _chunks(plate / "A/1/0")
    assert len(chunks) == 4
    _corrupt(chunks[1:])  # the first planes read, so the unit is not empty; the loader then meets the damage
    refused = r"unit A/1/0/t0 failed: ReadError: could not read time 0 of"
    with pytest.raises(ReconstructionFailed, match=refused) as error:
        fluorescence_model().reconstruct_positions(
            plate, output=tmp_path / "out.zarr", channels=["BF"], tile=(8, 16, 16)
        )
    (failed,) = error.value.report.results
    assert failed.error_type == "ReadError"


def _single_store(path, volume: torch.Tensor, version: Literal["0.4", "0.5"] = "0.5"):
    """Write a single-position store (no plate lists it) of one time point at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=["BF"], version=version) as position:
        position.create_image("0", series(volume, 1)[:, None].to(torch.float32).numpy())
        for axis, size in zip("zyx", PIXEL_SIZE, strict=True):
            position.set_scale("0", axis, size)
    return path


def test_a_single_position_store_among_other_inputs_is_refused_before_any_write(tmp_path, volume):
    single = _single_store(tmp_path / "run-1" / "cli4" / "single.zarr", volume)
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(volume, 1)])
    output = tmp_path / "out.zarr"
    model = phase_model()
    for dry_run in (False, True):
        with pytest.raises(Rejected, match="single-position store") as error:
            model.reconstruct_positions([single, plate], output=output, channels=["BF"], dry_run=dry_run)
        assert "positions" in [problem.location for problem in error.value.problems]
    assert not output.exists()  # nothing was begun, so a rerun gives the same answer
    old = tmp_path / "old.zarr"
    write_plate(old, [series(volume, 1)])
    with pytest.raises(Rejected, match="single-position store"):
        model.reconstruct_positions([single, plate], output=old, channels=["BF"], overwrite=True)
    assert (old / "A/1/0").exists()  # refused before the plate to replace was deleted


def test_messages_name_a_single_position_store_by_its_path(tmp_path, volume):
    single = _single_store(tmp_path / "single.zarr", volume)
    with pytest.warns(PositionWarning, match=re.escape(f"but {single.resolve()} records")):
        _plan(phase_model(), single, output=tmp_path / "out.zarr", channels=["BF"], pixel_size=(0.5, 0.15, 0.15))
    with pytest.raises(Rejected, match=re.escape(f"is the input position {single.resolve()} or lies inside it")):
        _plan(phase_model(), single, output=single / "out.zarr", channels=["BF"])


def test_an_all_zero_unit_is_not_solved_and_leaves_no_earlier_values(tmp_path, volume):
    stacks = series(volume, 3)
    plate = tmp_path / "in.zarr"
    write_plate(plate, [stacks])
    model = phase_model()
    output = tmp_path / "out.zarr"
    model.reconstruct_positions(plate, output=output, channels=["BF"])
    assert read_output(output)[1][1].abs().max() > 0
    stacks[1] = 0
    blank = tmp_path / "blank.zarr"
    write_plate(blank, [stacks])
    report = model.reconstruct_positions(blank, output=output, channels=["BF"])  # into the positions written above
    assert [result.status for result in report.results] == ["complete", "empty", "complete"]
    assert (report.status, report.count("empty")) == ("finished", 1)
    _, image, _ = read_output(output)
    assert image[1].abs().max() == 0  # the values written before did not survive
    assert image[0].abs().max() > 0


def test_a_model_no_evidence_can_score_still_reconstructs_positions(plate, tmp_path):
    detection = Detection(wavelength=0.52, numerical_aperture=Free(0.8, bounds=(0.6, 1.0)))
    model = fluorescence_model(detection, object_prior=TotalVariation(1e-3), solver=FISTA())
    output = tmp_path / "out.zarr"
    one = str(plate / "A/1/0")
    report = model.reconstruct_positions(one, output=output, channels=["BF"], times=[0])
    assert (report.status, report.count("complete")) == ("finished", 1)


def test_positions_and_a_second_reconstruction_are_appended_to_an_existing_plate(plate, tmp_path):
    output = tmp_path / "out.zarr"
    phase_model().reconstruct_positions(str(plate / "A/1/0"), output=output, channels=["BF"])
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert [name for name, _ in node.positions()] == ["A/1/0"]
    phase_model().reconstruct_positions(f"{plate}/*/*/1", output=output, channels=["BF"])  # a quoted glob
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert [name for name, _ in node.positions()] == ["A/1/0", "A/1/1"]
    assert read_output(output, "A/1/1")[1].abs().max() > 0
    fluorescence_model().reconstruct_positions(plate, output=output, channels=["BF"])
    names, image, attrs = read_output(output)
    assert names == ["Phase3D", "BF_Density3D"]
    assert {"waveorder-Phase3D", "waveorder-BF_Density3D"} <= set(attrs)
    assert image.abs().amax(dim=(0, 2, 3, 4)).min() > 0  # both reconstructions kept their channel


def test_an_existing_position_of_another_shape_is_refused(plate, tmp_path):
    output = tmp_path / "out.zarr"
    phase_model().reconstruct_positions(plate, output=output, channels=["BF"], times=[0])
    with pytest.raises(Rejected, match=r"holds \(T, Z, Y, X\) = 3x8x32x32, but this reconstruction writes 3x1x32x32"):
        thin_phase_model().reconstruct_positions(plate, output=output, channels=["BF"])
    assert read_output(output)[0] == ["Phase3D"]


def test_a_voxel_size_that_differs_from_the_stored_scale_is_used_with_a_warning(tmp_path, volume):
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(volume, 1)], pixel_size=(0.5, 0.15, 0.15))
    model = phase_model()
    output = tmp_path / "out.zarr"
    with pytest.warns(PositionWarning) as seen:
        model.reconstruct_positions(plate, output=output, channels=["BF"], pixel_size=PIXEL_SIZE)
    (warning,) = [item.message for item in seen if isinstance(item.message, PositionWarning)]
    assert warning.location is None  # the message names the argument
    assert str(warning) == (
        "pixel_size is (z, y, x) = (0.4, 0.15, 0.15) um, but A/1/0 records (0.5, 0.15, 0.15) um; reconstructing at "
        "pixel_size. Set it to the data's voxel size, or to auto to use the recorded one"
    )
    with iohub.open_ome_zarr(output / "A/1/0", mode="r") as position:
        assert list(position.scale[2:]) == pytest.approx([0.5, 0.15, 0.15])  # the input's scale, as 3.x writes
    written = tmp_path / "written.zarr"
    with pytest.warns(PositionWarning):
        model.reconstruct_positions(
            plate, output=written, channels=["BF"], pixel_size=PIXEL_SIZE, write_pixel_size=True
        )
    with iohub.open_ome_zarr(written / "A/1/0", mode="r") as position:
        assert list(position.scale[2:]) == pytest.approx([0.4, 0.15, 0.15])  # the stated size is written
    matching = tmp_path / "match.zarr"
    write_plate(matching, [series(volume, 1)])
    plain = tmp_path / "plain.zarr"
    model.reconstruct_positions(matching, output=plain, channels=["BF"], pixel_size=PIXEL_SIZE)  # no warning
    assert torch.equal(read_output(plain)[1], read_output(output)[1])  # the same numerics: the stated voxel size
    stored = tmp_path / "stored.zarr"
    model.reconstruct_positions(plate, output=stored, channels=["BF"])  # no size stated: the stored scale
    assert not torch.equal(read_output(stored)[1], read_output(output)[1])


def test_every_problem_is_reported_at_once_before_any_write(plate, tmp_path):
    output = tmp_path / "out.zarr"
    missing = tmp_path / "missing.zarr"
    with pytest.raises(Rejected) as error:
        thin_phase_model().reconstruct_positions(
            [plate, missing],
            output=output,
            channels=["DAPI"],
            times=[5],
            tile=(4, 16, 16),
            workers=0,
            resume=True,
            overwrite=True,
        )
    problems = error.value.problems
    locations = [problem.location or "" for problem in problems]
    assert f"positions {missing}" in locations  # the pattern matching nothing
    assert str((plate / "A/1/0").resolve()) in locations  # the position lacking DAPI, and time 5
    assert "tile" in locations  # a thin object cannot tile
    assert "workers" in locations
    messages = "\n".join(problem.message for problem in problems)
    assert "the input channel names list DAPI" in messages
    assert "times selects 5, outside the 3 time points" in messages
    assert "tile cannot split these volumes: a thin object reconstructs one plane" in messages
    assert "resume continues the output that overwrite deletes; give one of them" in messages
    assert len(set(problems)) == len(problems)
    assert not output.exists()


def test_messages_name_arguments_as_the_caller_spells_them(plate, tmp_path):
    missing = tmp_path / "missing.zarr"
    token = argument_names.set({"positions": "-i", "tile": "--tile", "pixel_size": "data.pixel_size"})
    try:
        with pytest.raises(Rejected) as error:
            thin_phase_model().reconstruct_positions(
                [plate, missing],
                output=tmp_path / "out.zarr",
                channels=["BF"],
                pixel_size=(0.5, 0.15, 0.15, 1.0),
                tile=(4, 16, 16),
            )
    finally:
        argument_names.reset(token)
    locations = [problem.location for problem in error.value.problems]
    assert locations[:2] == [f"-i {missing}", "data.pixel_size"]
    assert "--tile" in locations
    assert any("--tile cannot split these volumes" in problem.message for problem in error.value.problems)


def test_the_channels_must_number_what_the_system_measures(plate, tmp_path):
    with pytest.raises(Rejected, match=r"channels lists 2 .*measures 1") as error:
        phase_model().reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF", "BF2"])
    assert "channels" in [problem.location for problem in error.value.problems]


def test_an_output_inside_an_input_position_is_refused(plate):
    with pytest.raises(Rejected, match="is the input position A/1/0 or lies inside it"):
        phase_model().reconstruct_positions(plate, output=plate / "A/1/0" / "out.zarr", channels=["BF"])


def test_a_dry_run_writes_nothing_and_lists_the_pending_units(plate, tmp_path):
    output = tmp_path / "out.zarr"
    report = phase_model().reconstruct_positions(plate, output=output, channels=["BF"], times=[0, 2], dry_run=True)
    assert (report.status, report.total, report.results) == ("planned", 4, [])
    assert report.output == str(output)
    assert not output.exists()
    assert report.pending == ("A/1/0/t0", "A/1/0/t2", "A/1/1/t0", "A/1/1/t2")  # nothing ran


def test_overwrite_replaces_an_output_store_and_nothing_else(plate, tmp_path):
    model = phase_model()
    output = tmp_path / "out.zarr"
    model.reconstruct_positions(plate, output=output, channels=["BF"])
    report = model.reconstruct_positions(
        str(plate / "A/1/0"), output=output, channels=["BF"], times=[0], overwrite=True
    )
    assert report.count("complete") == 1
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert [name for name, _ in node.positions()] == ["A/1/0"]  # the store was replaced, not added to
    _, image, _ = read_output(output)
    assert image[0].abs().max() > 0
    assert image[1].abs().max() == 0
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("mine")
    with pytest.raises(Rejected, match="is not an OME-Zarr plate"):
        model.reconstruct_positions(plate, output=notes, channels=["BF"], overwrite=True)
    assert (notes / "keep.txt").read_text() == "mine"
    with pytest.raises(Rejected, match="holds the input position A/1/0, which overwrite would delete"):
        model.reconstruct_positions(plate, output=plate, channels=["BF"], overwrite=True)
    assert (plate / "A/1/0").exists()


def test_an_output_that_holds_an_input_is_refused_and_the_recorded_data_stays(plate, tmp_path):
    model = phase_model()
    refused = "holds the input position A/1/0: a reconstruction adds its channels to the positions of its output plate"
    with pytest.raises(Rejected, match=refused) as error:
        model.reconstruct_positions(plate, output=plate, channels=["BF"])
    assert f"output {plate}" in [problem.location for problem in error.value.problems]
    with pytest.raises(Rejected, match="holds the input position A/1/1: a reconstruction adds its channels"):
        model.reconstruct_positions(str(plate / "A/1/1"), output=plate, channels=["BF"])  # one position of it
    with pytest.raises(Rejected, match="holds the input position A/1/0"):
        model.reconstruct_positions(plate, output=tmp_path, channels=["BF"], dry_run=True)  # a folder around it
    for position in ("A/1/0", "A/1/1"):
        names, image, attrs = read_output(plate, position)
        assert (names, tuple(image.shape)) == (["BF"], (3, 1, 8, 32, 32))
        assert "waveorder-Phase3D" not in attrs


def _v04(path) -> None:
    """Create a v0.4 plate holding an empty Phase3D position A/1/0, as waveorder 3.x wrote them."""
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=["Phase3D"], version="0.4") as node:
        node.create_position("A", "1", "0").create_image("0", torch.zeros(3, 1, 8, 32, 32).numpy())


def test_v04_plates_keep_their_version_and_refuse_tiles(plate, tmp_path):
    model = phase_model()
    output = tmp_path / "out.zarr"
    _v04(output)
    with pytest.raises(Rejected, match=r"v0\.4 outputs have no"):
        model.reconstruct_positions(plate, output=output, channels=["BF"], tile=(8, 16, 16))
    assert model.reconstruct_positions(str(plate / "A/1/0"), output=output, channels=["BF"]).count("complete") == 3
    model.reconstruct_positions(str(plate / "A/1/1"), output=output, channels=["BF"])  # an append keeps the version
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert node.version == "0.4"
        assert [name for name, _ in node.positions()] == ["A/1/0", "A/1/1"]
    assert read_output(output, "A/1/1")[1].abs().max() > 0


def test_a_new_output_keeps_the_version_of_the_first_input(tmp_path, volume):
    old = tmp_path / "old.zarr"
    write_plate(old, [series(volume, 1)], version="0.4")
    new = tmp_path / "new.zarr"
    write_plate(new, [series(volume, 1)])
    single = _single_store(tmp_path / "single.zarr", volume, version="0.4")
    model = phase_model()
    cases = [(old, "plate.zarr", "0.4"), (new, "plate5.zarr", "0.5"), (single, "single.out.zarr", "0.4")]
    for source, name, version in cases:
        model.reconstruct_positions(source, output=tmp_path / name, channels=["BF"])
        with iohub.open_ome_zarr(tmp_path / name, mode="r") as node:
            assert node.version == version
    model.reconstruct_positions(new, output=tmp_path / "plate.zarr", channels=["BF"], overwrite=True)
    with iohub.open_ome_zarr(tmp_path / "plate.zarr", mode="r") as node:
        assert node.version == "0.5"  # a replaced store is new: its version is the input's, not the old store's
    with pytest.raises(Rejected, match=r"v0\.4 outputs have no shards"):
        _plan(model, old, output=tmp_path / "tiled.zarr", channels=["BF"], tile=(8, 16, 16))  # a v0.4 output too


def _failing_at(monkeypatch, time: int, error: BaseException | None = None, position: str = "") -> None:
    """Make iohub's processing of the time point ``time`` (of positions whose path ends with ``position``) raise."""
    real = _positions.process_time

    def failing(solve, **arguments):
        if arguments["time"] == time and str(arguments["source"]).endswith(position):
            raise error or OSError("disk full")
        return real(solve, **arguments)

    monkeypatch.setattr(_positions, "process_time", failing)


def test_a_failing_unit_stops_the_run_with_the_report_so_far(plate, tmp_path, monkeypatch):
    _failing_at(monkeypatch, 1)
    with pytest.raises(ReconstructionFailed, match=r"unit A/1/0/t1 failed: OSError: disk full") as error:
        phase_model().reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"])
    report = error.value.report
    assert (report.status, report.total) == ("failed", 6)
    assert [(result.unit, result.status) for result in report.results] == [
        ("A/1/0/t0", "complete"),
        ("A/1/0/t1", "failed"),
    ]
    assert report.results[1].error == "OSError: disk full"
    assert (report.results[0].error_type, report.results[1].error_type) == (None, "OSError")
    assert isinstance(error.value, WaveorderError)
    assert isinstance(error.value, RuntimeError)
    assert [report.count(status) for status in ("complete", "empty", "skipped", "failed")] == [1, 0, 0, 1]
    assert report.pending == ("A/1/0/t2", "A/1/1/t0", "A/1/1/t1", "A/1/1/t2")  # the units to resubmit


def test_keep_going_finishes_the_other_units_and_reports_the_failures(plate, tmp_path, monkeypatch):
    _failing_at(monkeypatch, 1)
    output = tmp_path / "out.zarr"
    report = phase_model().reconstruct_positions(plate, output=output, channels=["BF"], keep_going=True)
    assert (report.status, report.count("complete"), report.count("failed")) == ("failed", 4, 2)
    assert report.pending == ()  # every unit ran
    assert [result.unit for result in report.results if result.status == "failed"] == ["A/1/0/t1", "A/1/1/t1"]
    _, image, _ = read_output(output, "A/1/1")
    assert image[2].abs().max() > 0
    assert image[1].abs().max() == 0


# --- resume ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tile", [None, (8, 16, 16)], ids=["whole", "tiled"])
def test_an_interrupted_run_resumed_writes_what_an_uninterrupted_one_writes(plate, tmp_path, monkeypatch, tile):
    model = phase_model()
    whole = tmp_path / "whole.zarr"
    model.reconstruct_positions(plate, output=whole, channels=["BF"], tile=tile)
    output = tmp_path / "out.zarr"
    if tile is None:
        _failing_at(monkeypatch, 0, KeyboardInterrupt(), position="A/1/1")  # as a job killed half-way
    else:
        real = ReconstructionModel.reconstruct_tiles
        calls = []

        def interrupted(self, volumes, **arguments):
            calls.append(1)
            if len(calls) == 4:  # A/1/1 at t0
                raise KeyboardInterrupt
            return real(self, volumes, **arguments)

        monkeypatch.setattr(ReconstructionModel, "reconstruct_tiles", interrupted)
    with pytest.raises(KeyboardInterrupt):
        model.reconstruct_positions(plate, output=output, channels=["BF"], tile=tile)
    monkeypatch.undo()
    report = model.reconstruct_positions(plate, output=output, channels=["BF"], tile=tile, resume=True)
    statuses = {result.unit: result.status for result in report.results}
    assert [unit for unit, status in statuses.items() if status == "skipped"] == ["A/1/0/t0", "A/1/0/t1", "A/1/0/t2"]
    assert [unit for unit, status in statuses.items() if status == "complete"] == ["A/1/1/t0", "A/1/1/t1", "A/1/1/t2"]
    assert report.status == "finished"
    for position in ("A/1/0", "A/1/1"):
        assert torch.equal(read_output(output, position)[1], read_output(whole, position)[1])
    again = model.reconstruct_positions(plate, output=output, channels=["BF"], tile=tile)  # no resume: all again
    assert again.count("complete") == 6


def test_a_resumed_run_recomputes_what_another_reconstruction_wrote(plate, tmp_path):
    output = tmp_path / "out.zarr"
    phase_model().reconstruct_positions(plate, output=output, channels=["BF"], times=[0, 1])
    with iohub.open_ome_zarr(output / "A/1/0", mode="r+") as position:
        position["0"][0, 0, 0, 0, 0] = 123.0  # by hand: a skipped unit keeps it
    kept = phase_model().reconstruct_positions(plate, output=output, channels=["BF"], resume=True, devices=["cpu"])
    assert [result.status for result in kept.results] == ["skipped", "skipped", "complete"] * 2  # t2 never ran
    assert float(read_output(output)[1][0, 0, 0, 0, 0]) == 123.0
    stronger = phase_model(object_prior=Tikhonov(weight=1e-2))
    changed = stronger.reconstruct_positions(plate, output=output, channels=["BF"], resume=True)
    assert changed.count("complete") == 6  # another weight: another fingerprint, nothing finished
    assert float(read_output(output)[1][0, 0, 0, 0, 0]) != 123.0
    elsewhere = stronger.reconstruct_positions(plate, output=output, channels=["BF"], resume=True, times=[0])
    assert [result.status for result in elsewhere.results] == ["skipped", "skipped"]  # times are not settings
    tiled = stronger.reconstruct_positions(plate, output=output, channels=["BF"], resume=True, tile=(8, 16, 16))
    assert tiled.count("complete") == 6  # the tile arguments are


def test_resume_needs_a_v05_output_and_says_so(plate, tmp_path):
    model = phase_model()
    output = tmp_path / "out.zarr"
    _v04(output)
    model.reconstruct_positions(plate, output=output, channels=["BF"], times=[0])
    with pytest.warns(PositionWarning, match=re.escape(_positions.RESUME_UNTRACKED)):
        report = model.reconstruct_positions(plate, output=output, channels=["BF"], resume=True, times=[0])
    assert report.count("complete") == 2


def test_the_fingerprint_leaves_out_where_and_when_a_job_runs():
    configured = ReconstructionModel.from_config(V4Config(**yaml.safe_load(FLUORESCENCE)))
    for model, key, placement in ((configured, "config", "solver"), (phase_model(), "model", "settings")):
        record = _positions.provenance(model, ["BF"], pixel_size=PIXEL_SIZE, times=[1])
        token = _positions.fingerprint(model, record, tile=None, tile_overlap=None)
        described: Any = record[key]
        elsewhere = record | {"host": "elsewhere", "versions": {"torch": "0"}}
        elsewhere[key] = described | {
            "data": described["data"] | {"time_indices": [7]},
            placement: described[placement] | {"device": "cuda:3"},
        }
        assert _positions.fingerprint(model, elsewhere, tile=None, tile_overlap=None) == token
        assert _positions.fingerprint(model, record, tile=(8, 16, 16), tile_overlap=None) != token
        other = record | {key: described | {"data": described["data"] | {"input_channel_names": ["GFP"]}}}
        assert _positions.fingerprint(model, other, tile=None, tile_overlap=None) != token


# --- events and workers ----------------------------------------------------------------------------------------------


def test_units_start_and_unit_end_reach_an_attached_observer(plate, tmp_path):
    events = _Events()
    with observe(events):
        phase_model().reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"], times=[0])
    names = [name for name, _ in events.seen]
    assert names[0] == "units.start"
    assert events.seen[0][1] == {
        "output": str(tmp_path / "out.zarr"),
        "total": 2,
        "positions": 2,
        "devices": ["cpu"],
        "workers": 1,
    }
    ends = [payload for name, payload in events.seen if name == "unit.end"]
    assert [(payload["unit"], payload["status"]) for payload in ends] == [
        ("A/1/0/t0", "complete"),
        ("A/1/1/t0", "complete"),
    ]
    assert names.index("run.start") < names.index("unit.end")  # the model's own events arrive between


@pytest.mark.parametrize("tile", [None, (8, 16, 16)], ids=["whole", "tiled"])
def test_every_event_of_a_unit_run_in_this_process_names_the_unit(plate, tmp_path, tile):
    events = _Events()
    with observe(events):
        phase_model().reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"], times=[1], tile=tile)
    (first, _), *rest = events.seen
    assert first == "units.start"
    by_unit: dict[str, list[str]] = {}
    for name, payload in rest:
        assert "worker" not in payload  # no worker process ran it
        by_unit.setdefault(payload["unit"], []).append(name)
    assert list(by_unit) == ["A/1/0/t1", "A/1/1/t1"]
    for names in by_unit.values():
        assert (names[0], names[-2], names[-1]) == ("run.start", "run.end", "unit.end")
        assert len(names) > 3  # what the verb reports in between: tiles.end or solve.end
    units = [payload["unit"] for _, payload in rest]
    assert units == sorted(units)  # one unit after the other


def test_auto_devices_of_a_cpu_model_run_in_the_calling_process(plate, tmp_path, monkeypatch):
    def spawned(*arguments, **settings):
        del arguments, settings
        raise AssertionError("no worker process is started")

    monkeypatch.setattr(_pool, "_run_workers", spawned)
    model = phase_model()
    report = model.reconstruct_positions(plate, output=tmp_path / "out.zarr", channels=["BF"], devices="auto")
    assert report.count("complete") == 6
    assert {result.device for result in report.results} == {"cpu"}
    assert model.device == torch.device("cpu")
