"""``wo distill`` end to end on tiny CPU plates: index, fit, validate, test, evaluate, export, then ``wo rec``.

The runs start from the example configurations shipped in ``benchmarks/train/`` (bare class names), with the paths
and a one-epoch CPU run given as flags.
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.cli.plates import THIN_PHASE, read_output
from tests.data.plates import Plates, phase_plates
from waveorder.cli import app
from waveorder.data import FovIndex
from waveorder.io.bundle import load_bundle

runner = CliRunner()

REPOSITORY = Path(__file__).parents[2]
EXAMPLES = REPOSITORY / "benchmarks" / "train"
"""The example configurations: ``base.yml`` and the layers that make its variants."""

COMMANDS = ("index", "fit", "validate", "test", "evaluate", "export")

VARIANTS = {
    "cytoland": [
        "--model.student.init_args.stem", "StridedVolume", "--model.student.init_args.stem.planes", "5",
        "--model.student.init_args.head", "SubPixelHead", "--model.student.init_args.decoder_channels", "[8, 8, 8]",
    ],
    "position_bias": [
        "--model.conditions+", "FieldPosition", "--model.student.init_args.conditioning", "Bias",
        "--model.student.init_args.conditioning.sizes", "{field_position: 2}",
    ],
    "l1_ssim": [
        "--model.losses.ssim", "SSIM", "--model.losses.ssim.window", "7", "--model.weights", "{l1: 0.5, ssim: 0.5}",
    ],
}  # fmt: skip
"""Each layer of ``benchmarks/train/`` as the flags that make the same change."""

FULL_PATHS = {
    "UNet": "waveorder.nn.networks.UNet",
    "PlaneProjection": "waveorder.nn.networks.PlaneProjection",
    "ResNetEncoder": "waveorder.nn.networks.ResNetEncoder",
    "PlaneHead": "waveorder.nn.networks.PlaneHead",
    "FieldStandardize": "waveorder.nn.networks.FieldStandardize",
    "Affine": "waveorder.nn.networks.Affine",
    "L1": "waveorder.train.L1",
    "CSVLogger": "lightning.pytorch.loggers.CSVLogger",
}
"""The bare class names of ``base.yml`` and the import paths a configuration may give instead."""


def invoke(*arguments: object):
    return runner.invoke(app, [str(argument) for argument in arguments])


def fit(work: Path, version: str, *arguments: object, config: Path = EXAMPLES / "base.yml"):
    """Run ``wo distill fit`` from ``config`` and ``arguments`` for one epoch on the CPU into ``work/runs/version``."""
    return invoke(
        "distill", "fit", "--config", config, *arguments,
        "--model.reconstruction", work / "phase_2d.yml", "--data.index", work / "index",
        "--trainer.max_epochs", "1", "--trainer.accelerator", "cpu", "--trainer.enable_progress_bar", "false",
        "--trainer.logger.init_args.save_dir", work / "runs", "--trainer.logger.init_args.name", "",
        "--trainer.logger.init_args.version", version,
    )  # fmt: skip


def resolved(work: Path, *arguments: object) -> dict:
    """Return the configuration ``wo distill fit`` resolves from ``base.yml`` and ``arguments``, without running it."""
    result = fit(work, "printed", *arguments, "--print_config")
    assert result.exit_code == 0, result.output
    return yaml.safe_load(result.stdout)


@pytest.fixture(scope="module")
def pair(tmp_path_factory) -> Plates:
    """Brightfield stacks and their Phase2D reconstruction by the model of ``THIN_PHASE``: eight positions."""
    return phase_plates(tmp_path_factory.mktemp("plates"))


@pytest.fixture(scope="module")
def work(tmp_path_factory, pair) -> Path:
    """A folder with the model's configuration (``phase_2d.yml``) and the index of ``pair`` (``wo distill index``)."""
    folder = tmp_path_factory.mktemp("distill")
    (folder / "phase_2d.yml").write_text(THIN_PHASE)
    index = folder / "index"
    result = invoke(
        "distill", "index", pair.inputs, "--channel", "BF", "--targets", pair.targets, "--target-channel", "Phase2D",
        "--split-by", "position", "--fractions", "train=0.5,val=0.25,test=0.25",
        "--named-split", "wells=well:train=0.5,val=0.5", "-o", index,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{index}: 8 fields: test 2, train 4, val 2\n"
    return folder


@pytest.fixture(scope="module")
def run(work) -> Path:
    """The run folder of one epoch of ``base.yml``."""
    result = fit(work, "base")
    assert result.exit_code == 0, result.output
    return work / "runs" / "base"


def test_the_group_and_every_command_have_help():
    result = invoke("distill", "--help")
    assert result.exit_code == 0, result.output
    for command in COMMANDS:
        assert command in result.stdout
        shown = invoke("distill", command, "--help")
        assert shown.exit_code == 0, shown.output
        assert "Example: wo distill" in " ".join(shown.stdout.split())


def test_fit_help_lists_the_keys_of_the_module_and_the_data():
    shown = invoke("distill", "fit", "--help")
    assert shown.exit_code == 0, shown.output
    keys = ("--model.reconstruction", "--model.student", "--data.index", "--ckpt_path")
    assert all(key in shown.stdout for key in keys)


def test_index_writes_one_parquet_file_per_plate_and_the_manifest(work, pair):
    index = FovIndex.load(work / "index")
    assert index.channels == ("BF",)
    assert index.target_channels == ("Phase2D",)
    assert set(index.column("input")) == {str(pair.inputs.resolve())}
    assert index.splits == ("split", "split_wells")
    assert sorted(path.name for path in (work / "index").iterdir()) == ["bf.zarr.parquet", "manifest.json"]
    again = invoke(
        "distill", "index", pair.inputs, "--channel", "BF", "--targets", pair.targets, "--target-channel", "Phase2D",
        "-o", work / "index",
    )  # fmt: skip
    assert again.exit_code == 2
    assert "holds an index" in again.stderr


def test_index_refuses_a_malformed_fraction_and_a_bad_split(work, pair):
    given = (pair.inputs, "--channel", "BF", "--targets", pair.targets, "--target-channel", "Phase2D")
    malformed = invoke("distill", "index", *given, "--fractions", "train", "-o", work / "x")
    assert malformed.exit_code == 2
    assert "must read NAME=FRACTION" in malformed.stderr
    unsummed = invoke("distill", "index", *given, "--fractions", "train=0.5", "-o", work / "x")
    assert unsummed.exit_code == 2, unsummed.output
    assert "wo distill index: error: --fractions: " in unsummed.stderr
    assert not (work / "x").exists()


def test_fit_writes_bundles_and_a_run_folder_that_records_full_class_paths_and_absolute_paths(work, run):
    assert (run / "best" / "bundle.json").is_file()
    assert (run / "last" / "bundle.json").is_file()
    assert (run / "checkpoints" / "last.ckpt").is_file()
    assert (run / "index_manifest.json").read_text() == (work / "index" / "manifest.json").read_text()
    assert (run / "model.yml").read_text() == THIN_PHASE
    config = yaml.safe_load((run / "config.yaml").read_text())["model"]
    assert config["reconstruction"] == str(work / "phase_2d.yml")
    assert config["student"]["class_path"] == "waveorder.nn.networks.UNet"
    assert config["student"]["init_args"]["stem"]["class_path"] == "waveorder.nn.networks.PlaneProjection"
    assert config["losses"] == {"l1": {"class_path": "waveorder.train.L1"}}
    recorded = json.loads((run / "run.json").read_text())
    assert recorded["schema"] == "waveorder.distill_run/2"
    assert recorded["data"]["index"] == str(work / "index")
    student = json.loads((run / "best" / "bundle.json").read_text())["student"]
    assert student["class_path"] == "waveorder.nn.networks.unet.UNet"  # where the class is defined


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_layered_configuration_resolves_as_the_same_change_made_by_flags(work, variant):
    layered = resolved(work, "--config", EXAMPLES / f"{variant}.yml")
    flagged = resolved(work, *VARIANTS[variant])
    assert layered == flagged
    assert layered != resolved(work)


def test_a_variant_trains_from_its_layer(work):
    result = fit(work, "cytoland", "--config", EXAMPLES / "cytoland.yml")
    assert result.exit_code == 0, result.output
    config = yaml.safe_load((work / "runs" / "cytoland" / "config.yaml").read_text())
    assert config["model"]["student"]["init_args"]["stem"]["class_path"] == "waveorder.nn.networks.StridedVolume"
    assert (work / "runs" / "cytoland" / "best" / "bundle.json").is_file()


def test_bare_class_names_resolve_as_their_full_class_paths(work, tmp_path):
    text = (EXAMPLES / "base.yml").read_text()
    full = tmp_path / "full.yml"
    full.write_text(re.sub(rf"\b({'|'.join(FULL_PATHS)})\b", lambda found: FULL_PATHS[found[1]], text))
    assert "class_path: waveorder.nn.networks.UNet" in full.read_text()
    assert resolved(work, "--config", full) == resolved(work)


def test_a_stem_from_outside_the_package_trains_by_its_class_path_and_its_bundle_loads_in_a_fresh_process(work):
    outside = "tests.train.callbacks.MeanThenConv"  # not registered, not a waveorder class
    result = fit(work, "outside", "--model.student.init_args.stem", outside, "--model.student.init_args.stem.planes", 5)
    assert result.exit_code == 0, result.output
    bundle = work / "runs" / "outside" / "best"
    stem = json.loads((bundle / "bundle.json").read_text())["student"]["init_args"]["stem"]
    assert stem == {"class_path": outside, "init_args": {"planes": 5, "channels": 8}}
    code = (
        "import sys; from waveorder.io.bundle import load_bundle; bundle = load_bundle(sys.argv[1]); "
        "print(type(bundle.student.stem).__module__, bundle.self_test() <= bundle.test.tolerance)"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(bundle)], cwd=REPOSITORY, capture_output=True, text=True, check=False
    )
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.split() == ["tests.train.callbacks", "True"]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "decoder_channels",
            "decoder_chanels",
            "model.student: UNet takes no argument 'decoder_chanels'; did you mean decoder_channels?",
        ),
        (
            "batch_size: 4",
            "batch_size: four",
            "data.batch_size: input should be a valid integer, unable to parse string as an integer, got four",
        ),
        ("batch_size: 4", "batch_size: 0", "data.batch_size: input should be greater than 0"),
    ],
    ids=["unknown key", "wrong type", "out of range"],
)
def test_a_mistake_is_refused_when_the_configuration_is_parsed_in_one_located_line(work, tmp_path, old, new, message):
    path = tmp_path / "bad.yml"
    path.write_text((EXAMPLES / "base.yml").read_text().replace(old, new))
    result = fit(work, "bad", config=path)
    assert result.exit_code == 2, result.output
    assert result.stderr.splitlines() == [f"wo distill fit: error: {message}"]
    assert not (work / "runs" / "bad").exists()


def test_a_part_that_cannot_go_with_another_is_refused_at_its_key(work):
    result = fit(work, "bad_head", "--model.student.init_args.head", "SubPixelHead")
    assert result.exit_code == 2, result.output
    *logged, line = result.stderr.splitlines()  # before it, only Lightning's "Seed set to 0"
    assert line.startswith(
        "wo distill fit: error: model.student.head: ParameterError: SubPixelHead starts from 1/4 resolution, but the "
        "finest encoder map behind PlaneProjection is at 1/2"
    )
    assert all(entry.endswith("Seed set to 0") for entry in logged), result.stderr
    assert not (work / "runs" / "bad_head").exists()


def test_a_run_folder_runs_again_from_another_working_directory(work, run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # nothing the configuration names is here
    result = invoke("distill", "fit", "--config", run / "config.yaml", "--trainer.logger.init_args.version", "again")
    assert result.exit_code == 0, result.output
    again = work / "runs" / "again"  # the logger's folder is recorded as an absolute path too
    assert (again / "best" / "bundle.json").is_file()
    first, second = (yaml.safe_load((folder / "config.yaml").read_text()) for folder in (run, again))
    assert second["trainer"]["logger"]["init_args"].pop("version") == "again"
    first["trainer"]["logger"]["init_args"].pop("version")
    assert first == second
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("command", ["validate", "test"])
def test_validate_and_test_score_a_checkpoint(run, command):
    stage = {"validate": "val", "test": "test"}[command]
    result = invoke(
        "distill", command, "--config", run / "config.yaml", "--ckpt_path", run / "checkpoints" / "last.ckpt",
        "--trainer.logger.init_args.version", command,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    with (run.parent / command / "metrics.csv").open() as file:
        (row,) = list(csv.DictReader(file))
    assert {f"{stage}/loss", f"{stage}/ssim", f"{stage}/dynamic_range"} <= set(row)
    assert float(row[f"{stage}/loss"]) > 0


def test_evaluate_scores_the_test_split_alone_and_as_the_models_solver(work, run):
    bundle = run / "best"
    alone = invoke("distill", "evaluate", bundle, "--index", work / "index")
    assert alone.exit_code == 0, alone.output
    lines = alone.stdout.splitlines()
    assert lines[0].startswith("test: 2 fields, ")
    assert lines[1].split() == ["metric", "mean", "p10", "p50", "p90"]
    assert {line.split()[0] for line in lines[2:]} == {"ssim", "dynamic_range", "seconds"}
    through = invoke("distill", "evaluate", bundle, "--index", work / "index", "--model", work / "phase_2d.yml")
    assert through.exit_code == 0, through.output
    assert {"passband_inside", "passband_outside"} <= {line.split()[0] for line in through.stdout.splitlines()[2:]}


def test_export_writes_an_onnx_file_checked_against_the_bundle(run, tmp_path):
    bundle, path = run / "best", tmp_path / "model.onnx"
    result = invoke("distill", "export", bundle, "-o", path)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{path}: checked against the test tensors of {bundle}\n"
    assert path.is_file()


def test_rec_reconstructs_with_the_distilled_student(run, pair, tmp_path):
    bundle = run / "best"
    text = THIN_PHASE.replace("reconstruction: closed_form", f"reconstruction: {{distilled: {{bundle: {bundle}}}}}")
    config = tmp_path / "distilled.yml"
    config.write_text(text)
    output = tmp_path / "out.zarr"
    result = invoke("rec", "-i", f"{pair.inputs}/*/*/*", "-c", config, "-o", output)
    assert result.exit_code == 0, result.output
    assert "8 units: 8 reconstructed" in result.stdout
    names, image, attrs = read_output(output, "A/1/0")
    assert names == ["Phase2D", "Absorption2D"]
    assert image.shape == (1, 2, 1, 64, 64)
    assert image[:, 0].abs().max() > 0
    assert image[:, 1].abs().max() == 0  # the student returns phase only
    assert attrs["waveorder-Phase2D"]["bundle"]["sha256"] == load_bundle(bundle).sha256
