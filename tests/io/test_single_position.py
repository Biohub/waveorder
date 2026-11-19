"""A single-position store (an OME-Zarr position no plate lists) in and out: Python, the tile route, resume, the CLI."""

from pathlib import Path

import iohub
import pytest
import torch
from typer.testing import CliRunner

from tests.cli.plates import FIXED_PHASE, FREE_PHASE, NA, simulate_phase
from tests.cli.plates import series as cli_series
from tests.models.plates import PIXEL_SIZE, contrast, phase_model, series, write_plate
from waveorder.cli import app
from waveorder.errors import Rejected
from waveorder.io import TfStore
from waveorder.models import _positions
from waveorder.nn import Free, GridSearch, Tikhonov

runner = CliRunner()


def _write_single(path: Path, stack: torch.Tensor) -> Path:
    """Write ``(T, Z, Y, X)`` as a single-position store of channel BF, as iohub writes one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=["BF"]) as node:
        node.create_image("0", stack[:, None].to(torch.float32).numpy())
        for axis, size in zip("zyx", PIXEL_SIZE, strict=True):
            node.set_scale("0", axis, size)
        node.zattrs["biahub-deskew"] = {"angle": 30}
    return path


def _image(path: Path) -> torch.Tensor:
    with iohub.open_ome_zarr(path, mode="r") as node:
        return torch.as_tensor(node["0"][:])


@pytest.fixture(scope="module")
def stack() -> torch.Tensor:
    """Three brightfield stacks ``(T, Z, Y, X)`` of a simulated phase object; Y = 40 for an edge row of tiles."""
    volume = contrast()
    return series(torch.cat([volume, volume[:, :8]], dim=1), 3)


@pytest.fixture
def inputs(tmp_path, stack) -> tuple[Path, Path]:
    """The same data as a single-position store (in a folder whose names no plate could hold) and as plate A/1/0."""
    single = _write_single(tmp_path / "01-correct" / "data" / "bright-780.zarr", stack)
    (position,) = write_plate(tmp_path / "in.zarr", [stack])
    return single, position


def test_a_single_position_store_writes_one_whose_voxels_equal_a_plate_positions(inputs, tmp_path):
    single, position = inputs
    model = phase_model()
    output, plate = tmp_path / "out.zarr", tmp_path / "plate.zarr"
    report = model.reconstruct_positions(single, output=output, channels=["BF"])
    assert (report.status, report.count("complete")) == ("finished", 3)
    assert [result.unit for result in report.results] == ["./t0", "./t1", "./t2"]
    model.reconstruct_positions(position, output=plate, channels=["BF"])
    with iohub.open_ome_zarr(output, mode="r") as node, iohub.open_ome_zarr(plate / "A/1/0", mode="r") as like:
        assert isinstance(node, iohub.ngff.Position)
        assert node.channel_names == like.channel_names == ["Phase3D"]
        assert node.version == like.version
        assert node.get_effective_scale("0") == like.get_effective_scale("0")
        assert (node["0"].chunks, node["0"].shards, node["0"].dtype) == (
            like["0"].chunks,
            like["0"].shards,
            like["0"].dtype,
        )
        assert torch.equal(torch.as_tensor(node["0"][:]), torch.as_tensor(like["0"][:]))
        attrs = dict(node.zattrs)
        assert attrs["waveorder-Phase3D"] == dict(like.zattrs)["waveorder-Phase3D"]  # the same record
    assert attrs["biahub-deskew"] == {"angle": 30}  # the input store's own attributes, as a plate position's
    assert (tmp_path / ".iohub-progress" / "out.zarr").is_dir()


def test_resume_finishes_a_half_written_single_position_store_without_redoing_its_time_points(
    inputs, tmp_path, monkeypatch
):
    single, position = inputs
    model = phase_model()
    whole = tmp_path / "whole.zarr"
    model.reconstruct_positions(single, output=whole, channels=["BF"])
    real = _positions.process_time

    def killed_at_t1(solve, **arguments):
        if arguments["time"] == 1:
            raise KeyboardInterrupt  # as a job killed half-way
        return real(solve, **arguments)

    monkeypatch.setattr(_positions, "process_time", killed_at_t1)
    output = tmp_path / "out.zarr"
    with pytest.raises(KeyboardInterrupt):
        model.reconstruct_positions(single, output=output, channels=["BF"])
    monkeypatch.undo()
    report = model.reconstruct_positions(single, output=output, channels=["BF"], resume=True)
    assert {result.unit: result.status for result in report.results} == {
        "./t0": "skipped",
        "./t1": "complete",
        "./t2": "complete",
    }
    assert torch.equal(_image(output), _image(whole))
    plate = tmp_path / "plate.zarr"
    model.reconstruct_positions(position, output=plate, channels=["BF"], times=[0])
    with pytest.raises(Rejected, match="is not a single-position store"):  # an existing plate holds no position
        model.reconstruct_positions(single, output=plate, channels=["BF"], resume=True)


@pytest.mark.parametrize("dry_run", [False, True])
def test_several_inputs_with_a_single_position_store_are_refused_before_any_write(inputs, tmp_path, stack, dry_run):
    single, position = inputs
    other = _write_single(tmp_path / "other.zarr", stack)
    output = tmp_path / "out.zarr"
    for given, which in (([single, position], "one is a single-position store"), ([single, other], "all 2 are")):
        with pytest.raises(Rejected, match=which) as error:
            phase_model().reconstruct_positions(given, output=output, channels=["BF"], dry_run=dry_run)
        (problem,) = [problem for problem in error.value.problems if "single-position" in problem.message]
        assert problem.location == "positions"
        assert "reconstruct each single-position store in a run of its own" in problem.message
        assert "plate.zarr/<row>/<column>/<field>" in problem.message
        assert len(error.value.problems) == 1
    assert not output.exists()


def test_the_tile_route_on_a_single_position_store_matches_the_untiled_route(inputs, tmp_path):
    single, position = inputs
    model = phase_model()
    tile, tile_overlap = (8, 16, 16), (0, 4, 4)
    tiled, whole, plate = tmp_path / "tiled.zarr", tmp_path / "whole.zarr", tmp_path / "plate.zarr"
    report = model.reconstruct_positions(single, output=tiled, channels=["BF"], tile=tile, tile_overlap=tile_overlap)
    assert report.count("complete") == 3
    model.reconstruct_positions(single, output=whole, channels=["BF"])
    model.reconstruct_positions(position, output=plate, channels=["BF"], tile=tile, tile_overlap=tile_overlap)
    image = _image(tiled)
    assert image.shape == (3, 1, 8, 40, 32)
    assert image[:, :, :, 32:].abs().max() > 0  # the short edge row is written
    assert torch.corrcoef(torch.stack([image.flatten(), _image(whole).flatten()]))[0, 1] > 0.9
    assert torch.equal(image, _image(plate / "A/1/0"))
    with iohub.open_ome_zarr(tiled, mode="r") as node:
        assert tuple(node["0"].chunks)[:2] == (1, 1)  # one time point and channel per block


def test_an_iohub_position_opened_from_a_single_position_store_works_from_python(inputs, tmp_path):
    single, _ = inputs
    model = phase_model()
    from_path = tmp_path / "path.zarr"
    model.reconstruct_positions(single, output=from_path, channels=["BF"], times=[0])
    free = phase_model(object_prior=Tikhonov(weight=Free(4e-4, bounds=(1e-6, 1.0))), search=[GridSearch(points=5)])
    with iohub.open_ome_zarr(single, mode="r") as node:
        from_node = tmp_path / "node.zarr"
        report = model.reconstruct_positions(node, output=from_node, channels=["BF"], times=[0])
        assert report.status == "finished"
        assert torch.equal(_image(from_node), _image(from_path))
        from_node_store = free.calibrate_position(node, channels=["BF"])
        from_path_store = free.calibrate_position(single, channels=["BF"])
        assert from_node_store is not None
        assert from_path_store is not None
        assert from_node_store.row() == from_path_store.row()
        assert node["0"].shape == (3, 1, 8, 40, 32)  # the caller's node stays open


@pytest.fixture(scope="module")
def cli_single(tmp_path_factory) -> Path:
    """A single-position store of two brightfield stacks of the CLI tests' simulated object."""
    return _write_single(tmp_path_factory.mktemp("cli") / "data" / "bright-780.zarr", cli_series(simulate_phase(), 2))


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "phase.yml"
    path.write_text(text)
    return path


def test_wo_rec_writes_a_single_position_store_iohub_opens_as_a_position(cli_single, tmp_path):
    output = tmp_path / "out.zarr"
    config = _config(tmp_path, FIXED_PHASE)
    result = runner.invoke(app, ["rec", "-i", str(cli_single), "-c", str(config), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{output}: finished; 2 units: 2 reconstructed, 0 failed, 0 left\n"
    with iohub.open_ome_zarr(output, mode="r") as node:
        assert isinstance(node, iohub.ngff.Position)
        assert node.channel_names == ["Phase3D"]
        assert node["0"].shape == (2, 1, 8, 32, 32)
        assert torch.as_tensor(node["0"][:]).abs().max() > 0
    rec = ["rec", "-i", str(cli_single), "-c", str(config), "-o", str(output)]
    resumed = runner.invoke(app, [*rec, "--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert "2 units: 0 reconstructed" in resumed.stdout
    overwritten = runner.invoke(app, [*rec, "--overwrite"])
    assert overwritten.exit_code == 0, overwritten.output
    assert "2 units: 2 reconstructed" in overwritten.stdout


def test_wo_calibrates_computes_and_applies_transfer_functions_on_a_single_position_store(cli_single, tmp_path):
    free = _config(tmp_path, FREE_PHASE)
    output = tmp_path / "out.zarr"
    calibrated = runner.invoke(app, ["rec", "-i", str(cli_single), "-c", str(free), "-o", str(output)])
    assert calibrated.exit_code == 0, calibrated.output
    assert f"calibrating on {cli_single} at t=0" in calibrated.stderr
    assert NA in TfStore.load(tmp_path / "out_transfer_function_phase.zarr").row()
    store = tmp_path / "cal.zarr"
    alone = runner.invoke(app, ["calibrate-tf", "-i", str(cli_single), "-c", str(free), "-o", str(store)])
    assert alone.exit_code == 0, alone.output
    fixed = tmp_path / "fixed.yml"
    fixed.write_text(FIXED_PHASE)
    tf = tmp_path / "tf.zarr"
    computed = runner.invoke(app, ["compute-tf", "-i", str(cli_single), "-c", str(fixed), "-o", str(tf)])
    assert computed.exit_code == 0, computed.output
    applied, direct = tmp_path / "applied.zarr", tmp_path / "direct.zarr"
    arguments = ["-i", str(cli_single), "-c", str(fixed)]
    result = runner.invoke(app, ["apply-inv-tf", *arguments, "-o", str(applied), "-t", str(tf), "-j", "2"])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["rec", *arguments, "-o", str(direct)]).exit_code == 0
    assert torch.equal(_image(applied), _image(direct))  # the worker pool reads the store and writes the position
