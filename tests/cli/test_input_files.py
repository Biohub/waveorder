"""``-i`` given only files (e.g. a plate's ``zarr.json``): refused naming the file, not skipped into a vague error."""

from typer.testing import CliRunner

from tests.cli.plates import series, write_plate
from waveorder.cli import app
from waveorder.cli.config import template

runner = CliRunner()


def test_a_lone_file_input_is_refused_naming_it(tmp_path, phase_volume):
    plate = tmp_path / "in.zarr"
    write_plate(plate, [series(phase_volume, 1)])
    metadata = plate / "zarr.json"
    assert metadata.is_file()
    config = tmp_path / "phase.yml"
    config.write_text(template("phase", 3))
    result = runner.invoke(app, ["rec", "-i", str(metadata), "-c", str(config), "-o", str(tmp_path / "out.zarr")])
    assert result.exit_code == 2, result.output
    message = f"-i {metadata}: {metadata} is a file, not an OME-Zarr position or plate"
    assert f"wo rec: error: {message}" in result.stderr.splitlines()
    assert not (tmp_path / "out.zarr").exists()
