"""``wo view`` and ``wo bm view`` on napari's viewer model: the layers, names, scales and contrast limits they show."""

import dataclasses
import json
from pathlib import Path

import iohub
import pytest
import torch
import yaml
import zarr
from typer.testing import CliRunner

from tests.io.plates import write_plate
from tests.models.calibration_data import PHASE, model_of
from tests.models.plates import thin_phase_model
from waveorder.cli import app
from waveorder.io.tf_store import read_transfer

napari = pytest.importorskip("napari")

runner = CliRunner()

SHAPE = (6, 16, 15)  # odd x: the half spectrum has X // 2 + 1 columns and the centre is not the middle of an even axis
PIXEL_SIZE = (0.4, 0.15, 0.15)


@pytest.fixture
def opened(monkeypatch):
    """The viewers the commands open, as viewer models; their event loop returns at once."""
    from napari.components import ViewerModel

    viewers: list[ViewerModel] = []

    def viewer(*, title: str) -> ViewerModel:
        viewers.append(ViewerModel(title=title))
        return viewers[-1]

    monkeypatch.setattr(napari, "Viewer", viewer)
    monkeypatch.setattr(napari, "run", lambda: None)
    return viewers


def _layers(viewer) -> dict:
    return {layer.name: layer for layer in viewer.layers}


@pytest.fixture
def plate(tmp_path) -> Path:
    """Two positions of channels BF and GFP, ``(1, 2, 3, 8, 8)``; GFP has the omero window 2 to 7."""
    path = tmp_path / "plate.zarr"
    volumes = [torch.rand(1, 2, 3, 8, 8, generator=torch.Generator().manual_seed(i)) * 10 for i in range(2)]
    write_plate(path, volumes, channels=("BF", "GFP"))
    with iohub.open_ome_zarr(path, mode="r+") as node:
        for _, position in node.positions():
            position.set_contrast_limits("GFP", {"start": 2.0, "end": 7.0, "min": 0.0, "max": 10.0})
    return path


@pytest.fixture
def single(tmp_path) -> Path:
    """A single-position store of channel Phase2D, one plane ``(1, 1, 1, 8, 8)``."""
    path = tmp_path / "single.zarr"
    with iohub.open_ome_zarr(path, layout="fov", mode="w-", channel_names=["Phase2D"]) as position:
        position.create_image("0", torch.rand(1, 1, 1, 8, 8).numpy())
        for axis, size in zip("zyx", (2.0, 0.25, 0.5), strict=True):
            position.set_scale("0", axis, size)
    return path


def test_a_plate_gives_a_layer_per_channel_and_position_scaled_with_the_omero_window(plate, opened):
    result = runner.invoke(app, ["view", str(plate)])
    assert result.exit_code == 0, result.output
    (viewer,) = opened
    layers = _layers(viewer)
    assert list(layers) == ["BF [A/1/0]", "GFP [A/1/0]", "BF [A/1/1]", "GFP [A/1/1]"]
    gfp = layers["GFP [A/1/1]"]
    assert gfp.data.shape == (3, 8, 8)  # T squeezed
    assert list(gfp.scale) == pytest.approx(PIXEL_SIZE)
    assert list(gfp.contrast_limits) == [2.0, 7.0]
    with iohub.open_ome_zarr(plate / "A/1/1", mode="r") as position:
        assert torch.equal(torch.as_tensor(gfp.data), torch.as_tensor(position["0"][0, 1]))
    assert viewer.canvas.grid.enabled


def test_a_single_position_squeezes_z_and_two_stores_get_their_names_as_prefixes(plate, single, opened):
    result = runner.invoke(app, ["v", str(single), str(plate)])
    assert result.exit_code == 0, result.output
    layers = _layers(opened[0])
    assert list(layers) == [
        "single: Phase2D",
        "plate: BF [A/1/0]",
        "plate: GFP [A/1/0]",
        "plate: BF [A/1/1]",
        "plate: GFP [A/1/1]",
    ]
    phase = layers["single: Phase2D"]
    assert phase.data.shape == (8, 8)
    assert list(phase.scale) == [0.25, 0.5]


def test_paths_are_read_from_stdin(plate, single, opened):
    result = runner.invoke(app, ["view", str(plate)], input=f"{single}\n\n")
    assert result.exit_code == 0, result.output
    assert "single: Phase2D" in _layers(opened[0])
    nothing = runner.invoke(app, ["view"], input="")
    assert nothing.exit_code == 2
    assert "no paths given" in nothing.stderr


def _spectrum(layers: dict, label: str) -> torch.Tensor:
    """Return the layers ``Re(label)`` and ``Im(label)`` as one complex tensor, checking their colormap and limits."""
    real, imaginary = layers[f"Re({label})"], layers[f"Im({label})"]
    spectrum = torch.complex(torch.as_tensor(real.data), torch.as_tensor(imaginary.data))
    top = float(spectrum.abs().max())
    for layer in (real, imaginary):
        assert layer.colormap.name == "bwr"
        assert list(layer.contrast_limits) == pytest.approx([-top, top], rel=1e-6)
    return spectrum


def _negated(full: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """``full`` at ``-k`` along ``dims``, FFT order."""
    return torch.roll(torch.flip(full, dims), (1,) * len(dims), dims)


def _half(path: Path) -> torch.Tensor:
    """The half spectrum the store at ``path`` holds on :data:`SHAPE`, as the reader reads it."""
    read = read_transfer(path, SHAPE)
    assert read is not None
    assert read.transfer is not None
    return read.transfer


def test_a_3d_store_shows_its_transfer_function_on_the_centred_full_spectrum(tmp_path, opened):
    path = model_of(PHASE).to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE).save(tmp_path / "tf3d.zarr")
    result = runner.invoke(app, ["view", str(path)])
    assert result.exit_code == 0, result.output
    layers = _layers(opened[0])
    assert list(layers) == ["Re(transfer_function)", "Im(transfer_function)"]
    centred = _spectrum(layers, "transfer_function")
    half = _half(path)[0, 0]
    z, y, x = SHAPE
    assert centred.shape == SHAPE
    assert centred[z // 2, y // 2, x // 2] == half[0, 0, 0]  # the zero frequency lands at the centre
    full = torch.fft.ifftshift(centred, dim=(0, 1, 2))
    assert torch.equal(full[..., : x // 2 + 1], half)
    assert torch.equal(full, _negated(full, (0, 1, 2)).conj())  # Hermitian: H(-k) = conj(H(k))


def test_a_2d_store_shows_each_fields_plane_transfers_centred_laterally(tmp_path, opened):
    path = thin_phase_model().to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE).save(tmp_path / "tf2d.zarr")
    result = runner.invoke(app, ["view", str(path)])
    assert result.exit_code == 0, result.output
    layers = _layers(opened[0])
    labels = ["transfer_function f0", "transfer_function f1"]  # phase, absorption
    assert list(layers) == [f"{part}({label})" for label in labels for part in ("Re", "Im")]
    half = _half(path)  # (planes, fields, 1, Y, X // 2 + 1)
    _, y, x = SHAPE
    for field, label in enumerate(labels):
        centred = _spectrum(layers, label)
        assert centred.shape == SHAPE  # one plane per focal plane
        assert torch.equal(centred[:, y // 2, x // 2], half[:, field, 0, 0, 0])
        full = torch.fft.ifftshift(centred, dim=(1, 2))
        assert torch.equal(full[..., : x // 2 + 1], half[:, field, 0])
        assert torch.equal(full, _negated(full, (1, 2)).conj())


def test_a_store_without_transfer_function_or_unreadable_is_refused_with_the_readers_message(tmp_path, opened):
    store = thin_phase_model().to_tf_store(shapes=[SHAPE], pixel_size=PIXEL_SIZE)
    before = dataclasses.replace(store, transfers=()).save(tmp_path / "parameters.zarr")  # as stores were before
    refused = runner.invoke(app, ["view", str(before)])
    assert refused.exit_code == 2
    assert f"{before}: holds no transfer function" in refused.stderr
    old = tmp_path / "old.zarr"
    zarr.open_group(str(old), mode="w").attrs["settings"] = {"phase": {}}  # a 3.x compute-tf store
    refused = runner.invoke(app, ["view", str(old)])
    assert refused.exit_code == 2
    assert "was made by waveorder 3.x, which stored no parameters; recompute it with wo compute-tf" in refused.stderr
    assert opened == []  # nothing is opened before every path is read


# --- wo bm view ---------------------------------------------------------------------------------------------------


def _run(root: Path, name: str, *, reconstruction: float, cells: Path) -> Path:
    """Write a run in ``wo bm run``'s layout: a synthetic case ``beads`` and an hpc case ``cells`` on ``cells``."""
    run = root / name
    beads = run / "cases" / "beads"
    beads.mkdir(parents=True)
    write_plate(beads / "simulated.zarr", [torch.full((1, 1, 4, 8, 8), 3.0)], channels=("BF",))
    write_plate(beads / "phantom.zarr", [torch.ones(1, 1, 4, 8, 8)], channels=("BF_phantom",))
    write_plate(beads / "reconstruction.zarr", [torch.full((1, 1, 4, 8, 8), reconstruction)], channels=("Phase3D",))
    (beads / "config.yml").write_text(yaml.safe_dump({"input_channel_names": ["BF"]}))
    (run / "cases" / "cells").mkdir()
    (run / "cases" / "cells" / "config.yml").write_text(yaml.safe_dump({"input_channel_names": ["BF"]}))
    experiment = {
        "name": "regression",
        "cases": {
            "beads": {"type": "synthetic"},
            "cells": {"type": "hpc", "input": str(cells), "position": "A/1/0", "crop": {"z": [1, 3], "y": [0, 4]}},
        },
    }
    (run / "experiment.yml").write_text(yaml.safe_dump(experiment))
    (run / "summary.json").write_text(json.dumps({}))
    return run


@pytest.fixture
def runs(tmp_path, plate) -> Path:
    root = tmp_path / "bench"
    _run(root, "2026-01-01T00-00-00_regression", reconstruction=1.0, cells=plate)
    _run(root, "2026-01-02T00-00-00_regression", reconstruction=5.0, cells=plate)
    return root


def test_bm_view_loads_raw_phantom_and_reconstruction_of_the_latest_run(runs, plate, opened):
    result = runner.invoke(app, ["bm", "view", "-o", str(runs)])
    assert result.exit_code == 0, result.output
    assert "cells: no reconstruction.zarr" in result.stdout
    (viewer,) = opened
    assert viewer.title == "wo bm view: 2026-01-02T00-00-00_regression"
    layers = _layers(viewer)
    assert list(layers) == [
        "beads/raw/BF",
        "beads/phantom/BF_phantom",
        "beads/recon/Phase3D",
        "cells/raw/BF",  # the experiment's position, its input channels and its crop
    ]
    assert float(layers["beads/recon/Phase3D"].data.max()) == 5.0
    with iohub.open_ome_zarr(plate / "A/1/0", mode="r") as position:
        cropped = torch.as_tensor(position["0"][0, 0, 1:3, 0:4])
    assert torch.equal(torch.as_tensor(layers["cells/raw/BF"].data), cropped)


def test_bm_view_of_two_runs_loads_the_raw_input_once_and_links_the_reconstructions_contrast_limits(runs, opened):
    first, second = "2026-01-01T00-00-00_regression/beads", "2026-01-02T00-00-00_regression/beads"
    result = runner.invoke(app, ["bm", "view", first, second, "-o", str(runs)])
    assert result.exit_code == 0, result.output
    layers = _layers(opened[0])
    assert list(layers) == [
        "a/beads/raw/BF",
        "a/beads/phantom/BF_phantom",
        "a/beads/recon/Phase3D",
        "b/beads/phantom/BF_phantom",  # the raw input only once, from the first run
        "b/beads/recon/Phase3D",
    ]
    a, b = layers["a/beads/recon/Phase3D"], layers["b/beads/recon/Phase3D"]
    assert list(a.contrast_limits) == list(b.contrast_limits) == [1.0, 5.0]
    a.contrast_limits = (2.0, 4.0)
    assert list(b.contrast_limits) == [2.0, 4.0]


def test_bm_view_says_what_it_cannot_find(runs, tmp_path, opened):
    missing = runner.invoke(app, ["bm", "view", "nope/beads", "-o", str(runs)])
    assert missing.exit_code == 0
    assert f"Run not found: {runs / 'nope'}" in missing.stdout
    case = runner.invoke(app, ["bm", "view", "latest/none", "-o", str(runs)])
    assert "Case not found" in case.stdout
    empty = runner.invoke(app, ["bm", "view", "-o", str(tmp_path / "empty")])
    assert "No benchmark runs found." in empty.stdout
    assert opened == []
