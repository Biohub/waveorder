"""Runs of ``wo distill fit`` on the tiny phase pair: what a run writes, resume, DDP, and the steps' purity.

The runs are built from a configuration as the command builds them (:func:`tests.train.runs.fit_here`), or run as
the command in a process of their own where a run is killed, resumed or launched on several ranks.
"""

import csv
import json
from pathlib import Path

import pytest
import torch

from tests.data.conftest import SPLIT
from tests.train.callbacks import traced
from tests.train.runs import (
    RESUMED,
    STUDENT,
    Inputs,
    checkpoint,
    configuration,
    differences,
    finished,
    fit_here,
    inputs,
    started,
)
from waveorder.data import FovIndex, GroupSplit, index_from_plates
from waveorder.errors import ParameterError, TrainingFailed
from waveorder.io.bundle import load_bundle
from waveorder.nn import Distilled
from waveorder.nn.systems.base import ImagingSystem


@pytest.fixture(scope="module")
def given(phase_index, tmp_path_factory) -> Inputs:
    return inputs(phase_index, tmp_path_factory.mktemp("given"))


@pytest.fixture(scope="module")
def recorded(given, tmp_path_factory) -> tuple[Path, list[str]]:
    """A run of two epochs with weight averaging, during which binding an imaging system records and raises."""
    run = tmp_path_factory.mktemp("recorded") / "run"
    tried: list[str] = []

    def bind(self: ImagingSystem, *args: object, **kwargs: object) -> None:
        del args, kwargs
        tried.append(type(self).__name__)
        raise AssertionError(f"the run bound the forward model of {type(self).__name__}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ImagingSystem, "bind", bind)
        fit_here(configuration(given, run))
    return run, tried


def metrics(run: Path, name: str) -> list[float]:
    with (run / "metrics.csv").open() as file:
        return [float(row[name]) for row in csv.DictReader(file) if row.get(name)]


def test_a_fit_writes_bundles_the_solver_loads_and_a_run_folder_that_says_what_made_it(recorded, given):
    run, _ = recorded
    solver = Distilled(load_bundle(run / "best"))
    assert solver.bundle.test is not None
    assert solver.bundle.self_test() <= solver.bundle.test.tolerance
    assert (run / "last" / "bundle.json").is_file()
    assert (run / "checkpoints" / "last.ckpt").is_file()
    assert not (run / "checkpoints" / "on_exception.ckpt").exists()
    assert len(metrics(run, "val/loss")) == len(metrics(run, "val/ssim")) == 2
    assert metrics(run, "val/loss/l1")
    assert metrics(run, "train/loss")

    index = FovIndex.load(given.index)
    record = json.loads((run / "run.json").read_text())
    assert record["schema"] == "waveorder.distill_run/2"
    assert record["stage"] == "fit"
    assert record["data"]["index"] == str(given.index)
    assert record["data"]["index_sha256"] == index.sha256
    assert record["data"]["split"] == {"column": "split", "holdout": None, "counts": {"test": 2, "train": 4, "val": 2}}
    assert record["data"]["split_sha256"] == index.split_sha256
    assert record["task"]["student"]["class_path"] == "waveorder.nn.networks.unet.UNet"
    assert {"waveorder", "torch", "lightning", "iohub"} <= set(record["software"])
    assert "git_commit" in record
    manifest = json.loads((given.index / "manifest.json").read_text())
    assert json.loads((run / "index_manifest.json").read_text()) == manifest
    assert (run / "model").is_dir()
    resolved = (run / "config.yaml").read_text()
    assert "class_path: waveorder.nn.networks.UNet" in resolved  # full class paths, given as bare names
    assert f"index: {given.index}" in resolved
    bundle = load_bundle(run / "best").record
    assert bundle["index_sha256"] == index.sha256
    assert bundle["splits"] == {"test": 2, "train": 4, "val": 2}


def test_no_step_of_a_fit_runs_the_imaging_model(recorded):
    _, tried = recorded
    assert tried == []


def test_the_bundles_hold_the_averaged_weights(recorded):
    run, _ = recorded
    saved = checkpoint(run)

    def student(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name.removeprefix("task.student."): value for name, value in state.items() if "task.student." in name}

    averaged, current = student(saved["state_dict"]), student(saved["current_model_state"])
    bundled = load_bundle(run / "last").student.state_dict()
    floating = [name for name, value in bundled.items() if value.is_floating_point()]
    assert all(torch.equal(bundled[name], averaged[name]) for name in floating)
    assert any(not torch.equal(bundled[name], current[name]) for name in floating)


def test_the_loss_falls_on_a_learnable_target(given, tmp_path):
    optimization = {"lr": 1e-2, "schedule": "constant", "warmup_epochs": 0, "ema": None}
    config = configuration(given, tmp_path / "run", epochs=15, model={"optimization": optimization, "check": False})
    fit_here(config)
    losses, similarities = metrics(tmp_path / "run", "val/loss"), metrics(tmp_path / "run", "val/ssim")
    assert len(losses) == 15
    assert losses[-1] < 0.5 * losses[0]
    assert similarities[-1] > similarities[0] + 0.2
    recorded_metrics = load_bundle(tmp_path / "run" / "best").record["metrics"]
    assert isinstance(recorded_metrics, dict)
    best = recorded_metrics["val/ssim"]
    assert best == pytest.approx(max(similarities), abs=1e-6)


def test_init_from_copies_the_matching_tensors_and_a_smoke_run_stops_at_max_steps(recorded, given, tmp_path):
    first, _ = recorded
    student = json.loads(json.dumps(STUDENT))
    student["init_args"]["decoder_channels"] = [8, 8, 16]  # another decoder: its tensors cannot be copied
    config = configuration(
        given,
        tmp_path / "run",
        model={"student": student, "init_from": str(first / "best")},
        data={"limit_fields": 2},
        trainer={"max_steps": 3},
    )
    cli = fit_here(config)
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert checkpoint(tmp_path / "run")["global_step"] == 3
    assert record["data"]["loader"]["splits"]["train"]["fields"] == 2
    copied = cli.model.initialized
    assert copied["path"] == str(first / "best")
    assert copied["copied"] > 0
    assert copied["skipped"]
    assert all(name.startswith("student.") for name in copied["skipped"])
    assert "student.stem.conv.weight" not in copied["skipped"]
    bundle = load_bundle(first / "best")
    for name, value in bundle.target_normalizer.state_dict().items():  # copied, and not fitted again
        torch.testing.assert_close(cli.model.task.target_normalizer.state_dict()[name], value, rtol=1e-6, atol=0)


def test_a_run_stopped_before_its_first_validation_still_writes_its_bundle_and_its_holdout(given, tmp_path):
    # Holding out well A/2 leaves two train fields: two batches an epoch, so one step stops the run mid-epoch.
    config = configuration(
        given, tmp_path / "run", data={"holdout": {"wells": ["bf.zarr/A/2"]}}, trainer={"max_steps": 1}
    )
    fit_here(config)
    bundle = load_bundle(tmp_path / "run" / "last")
    assert not (tmp_path / "run" / "best").exists()
    assert bundle.test is not None
    assert bundle.self_test() <= bundle.test.tolerance
    assert tuple(bundle.test.data.shape[-2:]) == (32, 32)  # a validation patch
    split = json.loads((tmp_path / "run" / "run.json").read_text())["data"]["split"]
    assert split["holdout"] == {"plates": [], "wells": ["bf.zarr/A/2"]}
    assert split["counts"] == {"holdout": 4, "test": 1, "train": 2, "val": 1}
    assert bundle.record["splits"] == split["counts"]


def test_a_non_finite_loss_stops_the_run(given, tmp_path):
    broken = {"losses": {"broken": "tests.train.callbacks.NotANumber"}, "weights": {}, "check": False}
    with pytest.raises(TrainingFailed, match="non-finite at step 0"):
        fit_here(configuration(given, tmp_path / "run", model=broken))


def test_a_monitor_validation_does_not_log_is_refused(given, tmp_path):
    with pytest.raises(ParameterError, match="no such metric"):
        fit_here(configuration(given, tmp_path / "run", model={"monitor": "val/psnr", "check": False}))


# --- in processes of their own ------------------------------------------------------------------------------------


def _preempt(**at: int) -> list[dict[str, object]]:
    return [{"class_path": "tests.train.callbacks.Preempt", "init_args": at}]


@pytest.mark.parametrize("workers", [0, 2])
def test_a_run_resumed_at_an_epoch_boundary_or_mid_epoch_is_the_uninterrupted_run(given, tmp_path, workers):
    """Four epochs of four steps; one run killed as epoch 2 starts, one killed before step 6 (epoch 1, batch 2)."""
    runs = {name: tmp_path / name for name in ("straight", "boundary", "middle")}
    stops = {"straight": [], "boundary": _preempt(at_epoch=2), "middle": _preempt(at_step=6)}
    first = {
        name: started("fit", configuration(given, run, epochs=4, workers=workers, callbacks=stops[name]),
                      tmp_path / f"{name}.json")
        for name, run in runs.items()
    }  # fmt: skip
    finished(first["straight"])
    for name, steps in (("boundary", 8), ("middle", 6)):
        finished(first[name], code=17)  # killed with os._exit: no teardown, no checkpoint on the way out
        assert checkpoint(runs[name])["global_step"] == steps
    resumed = [
        started(
            "fit",
            configuration(given, runs[name], epochs=4, workers=workers),
            tmp_path / f"{name}.resumed.json",
            "--ckpt_path",
            str(runs[name] / "checkpoints" / "last.ckpt"),
        )
        for name in ("boundary", "middle")
    ]
    for process in resumed:
        finished(process)
    straight = checkpoint(runs["straight"])
    batches = traced(runs["straight"] / "trace")
    assert sum(1 for key in batches if key[0] == "train") == 16
    for name in ("boundary", "middle"):
        again = checkpoint(runs[name])
        assert (again["epoch"], again["global_step"]) == (straight["epoch"], straight["global_step"]) == (3, 16)
        for part in RESUMED:
            assert list(differences(straight[part], again[part], part)) == [], name
        assert torch.equal(straight["transforms_generator"], again["transforms_generator"])
        assert traced(runs[name] / "trace") == batches, name  # every batch: its fields and its patch origins


@pytest.fixture(scope="module")
def four_val_fields(phase_pair, tmp_path_factory) -> Inputs:
    """The phase pair split by position half and half: four train and four validation fields."""
    index = index_from_plates(
        [phase_pair.inputs],
        channels=phase_pair.channels,
        targets=[phase_pair.targets],
        target_channels=phase_pair.target_channels,
        split=GroupSplit(by="position", fractions={"train": 0.5, "val": 0.5}, seed=SPLIT.seed),
    )
    return inputs(index, tmp_path_factory.mktemp("four"))


def test_ranks_read_disjoint_shards_and_three_ranks_validate_the_mean_of_one(four_val_fields, tmp_path):
    """Two DDP processes on the CPU fit; then the checkpoint is validated on one rank and on three.

    Three ranks share four validation fields as 2, 1 and 1: 8, 4 and 4 patches, a total no multiple of 3, which is
    where Lightning 2.6.6 averaged integer counts wrongly. The run logs float sums and counts.
    """
    run = tmp_path / "ddp"
    ddp = {"strategy": "ddp", "devices": 2}
    finished(started("fit", configuration(four_val_fields, run, workers=2, trainer=ddp), tmp_path / "fit.json"))
    index = FovIndex.load(four_val_fields.index)
    names = {split: {row.name for row in index if row.split == split} for split in ("train", "val")}
    batches = traced(run / "trace")
    for epoch in (0, 1):
        for stage, split in (("train", "train"), ("val", "val")):
            read = [
                {name for (s, rank, e, _), (positions, _) in batches.items() if (s, rank, e) == (stage, r, epoch)
                 for name in positions}
                for r in (0, 1)
            ]  # fmt: skip
            assert not read[0] & read[1], (stage, epoch)
            assert read[0] | read[1] == names[split], (stage, epoch)
            assert read[0], (stage, epoch)

    means = {}
    for devices in (1, 3):
        folder = tmp_path / f"validate{devices}"
        trainer = {"devices": devices, **({"strategy": "ddp"} if devices > 1 else {})}
        config = configuration(four_val_fields, folder, workers=0, trainer=trainer)
        checkpointed = str(run / "checkpoints" / "last.ckpt")
        finished(started("validate", config, tmp_path / f"validate{devices}.json", "--ckpt_path", checkpointed))
        means[devices] = {name: metrics(folder, name)[-1] for name in ("val/loss", "val/ssim", "val/dynamic_range")}
        if devices == 3:
            patches = [
                sum(len(positions) for (s, rank, _, _), (positions, _) in traced(folder / "trace").items()
                    if s == "validate" and rank == r)
                for r in range(3)
            ]  # fmt: skip
            assert sorted(patches) == [4, 4, 8]
    for name, value in means[1].items():
        assert means[3][name] == pytest.approx(value, rel=1e-6), name
