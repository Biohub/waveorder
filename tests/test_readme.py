"""The README's examples run as written, and each prints what it claims: shapes, statuses and recovered values."""

import re
from pathlib import Path

import pytest
import yaml

README = Path(__file__).parents[1] / "README.md"
TEXT = README.read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", TEXT, flags=re.DOTALL)
CONFIGS = re.findall(r"```yaml\n(.*?)```", TEXT, flags=re.DOTALL)


def block(marker: str) -> str:
    """Return the one python block of the README that contains ``marker``."""
    found = [code for code in BLOCKS if marker in code]
    assert len(found) == 1, f"{len(found)} README python blocks contain {marker!r}, expected 1"
    return found[0]


def output_of(code: str, capsys: pytest.CaptureFixture[str]) -> str:
    """Run a block in a fresh namespace and return what it printed."""
    capsys.readouterr()
    exec(compile(code, str(README), "exec"), {"__name__": "readme"})
    return capsys.readouterr().out


def test_readme_has_examples():
    assert len(BLOCKS) >= 3, "README.md lost a python example"
    assert len(CONFIGS) == 3, "README.md has three configurations: the quick start's, base.yml and cytoland.yml"


def test_the_quick_start_calibrates_reconstructs_and_saves(capsys):
    out = output_of(block("model.calibrate(data, prepared=True)"), capsys)
    match = re.search(r"NA = ([\d.]+), estimate = \(1, 16, 64, 64\), direct", out)
    assert match, out
    assert abs(float(match[1]) - 0.52) < 0.05, out  # simulated at NA 0.52, started at 0.45
    assert f"loaded NA = {match[1]}" in out, out  # the saved folder keeps the calibrated value


def test_the_command_line_configuration_builds_the_quick_start_model():
    import waveorder as wo
    from waveorder.nn import free_specs

    model = wo.models.ReconstructionModel.from_config(yaml.safe_load(CONFIGS[0]))
    assert type(model).__name__ == "Phase3D"
    assert model.pixel_size is None  # auto: read from each position's scale
    assert list(free_specs(model.imaging_model)) == ["system.illumination.numerical_aperture"]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder as the README's training starts from: ``phase_2d.yml`` and the index of its recorded targets."""
    from tests.cli.plates import THIN_PHASE
    from tests.data.plates import phase_plates
    from waveorder.data import GroupSplit, index_from_plates

    folder = tmp_path_factory.mktemp("readme")
    plates = phase_plates(folder)
    split = GroupSplit(by="position", fractions={"train": 0.5, "val": 0.25, "test": 0.25})
    index = index_from_plates(
        [plates.inputs], channels=["BF"], targets=[plates.targets], target_channels=["Phase2D"], split=split
    )
    index.save(folder / "index")
    (folder / "phase_2d.yml").write_text(THIN_PHASE)
    return folder


def test_the_training_configuration_and_its_layer_train(recorded, monkeypatch):
    from waveorder.models._lightning import main

    monkeypatch.chdir(recorded)
    (recorded / "base.yml").write_text(CONFIGS[1])
    (recorded / "cytoland.yml").write_text(CONFIGS[2])
    main(["fit", "--config", "base.yml", "--config", "cytoland.yml", "--trainer.enable_progress_bar", "false"])
    run = recorded / "runs" / "lightning_logs" / "version_0"
    assert (run / "best" / "bundle.json").is_file()
    stem = yaml.safe_load((run / "config.yaml").read_text())["model"]["student"]["init_args"]["stem"]
    assert stem["class_path"] == "waveorder.nn.networks.StridedVolume"


def test_training_returns_a_model_whose_solver_is_the_network_and_exports_it(capsys, recorded, monkeypatch):
    monkeypatch.chdir(recorded)
    out = output_of(block('model.train("index"'), capsys)
    assert "solver = Distilled, bundle = best, onnx = model.onnx" in out, out
    assert (recorded / "model.onnx").is_file()


def test_a_registered_system_is_saved_and_loaded_as_its_model_class(capsys):
    out = output_of(block('@registry.register("gaussian_blur_3d"'), capsys)
    assert out.strip() == "Blur3D, solver = Wiener, estimate = (1, 8, 32, 32)", out


def test_every_example_is_asserted_on():
    markers = (
        "model.calibrate(data, prepared=True)",
        'model.train("index"',
        '@registry.register("gaussian_blur_3d"',
    )
    assert sorted(BLOCKS.index(block(marker)) for marker in markers) == list(range(len(BLOCKS)))
