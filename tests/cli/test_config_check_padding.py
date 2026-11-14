"""``wo config check`` shows what ``solver.padding.z: auto`` pads."""

from typer.testing import CliRunner

from tests.cli.plates import edited
from waveorder.cli import app
from waveorder.cli.config import template

runner = CliRunner()


def test_check_shows_the_least_auto_padding_at_the_configured_voxel_size(tmp_path):
    source = tmp_path / "phase.yml"
    source.write_text(template("phase", 3))
    path = edited(source, "solver.padding.z=auto")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 0, result.output
    # The template: n = 1.3, wavelength 0.532 um, detection NA 1.2; 1.3 * 0.532 / 1.44 = 0.4803 um, 1.92 slices of
    # 0.25 um.
    assert result.stdout == (
        "solver.padding.z: auto pads at least 2 slices at each end (n * wavelength / NA**2 = 0.4803 um at a z voxel "
        f"size of 0.25 um), then more for a fast FFT length of each stack\nok: {path}\n"
    )


def test_check_says_auto_is_resolved_per_position_without_a_voxel_size(tmp_path):
    source = tmp_path / "phase.yml"
    source.write_text(template("phase", 3))
    path = edited(source, "solver.padding.z=auto", "data.pixel_size=auto")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "solver.padding.z: auto pads at least n * wavelength / NA**2 = 0.4803 um at each end of z; the slices are "
        f"resolved per position from its z voxel size and stack\nok: {path}\n"
    )
