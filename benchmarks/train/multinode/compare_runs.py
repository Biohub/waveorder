"""Compare the two runs of the two-node test, and check how each sharded its fields.

python compare_runs.py ROOT/straight ROOT/requeue

Each run folder holds ``checkpoints/last.ckpt`` and ``trace/rank<N>.jsonl`` (``callbacks.Trace``). It prints, for the
final checkpoints, whether the weights, the weight average, the optimizer, the scheduler and the loops agree; whether
every train batch (rank, epoch, step: its fields and patch origins) agrees, a requeued step counting its last
attempt; and, per run and epoch, the fields each rank read and their overlap. It exits 1 when anything differs.
"""

import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import torch

PARTS = ("state_dict", "current_model_state", "averaging_state", "optimizer_states", "lr_schedulers")
"""What the final checkpoints of the two runs must hold identically."""


def differences(a: object, b: object, path: str = "") -> Iterator[str]:
    """Yield the paths where two nested states differ (with the largest difference, for tensors)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or not torch.equal(a, b):
            gap = float((a.double() - b.double()).abs().max()) if a.shape == b.shape else float("nan")
            yield f"{path}: shapes {tuple(a.shape)}, {tuple(b.shape)}, max |difference| {gap:.3g}"
        return
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        a, b = dict(enumerate(a)), dict(enumerate(b))
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            inside = f"{path}.{key}"
            yield from differences(a[key], b[key], inside) if key in a and key in b else [f"{inside}: in one only"]
    elif a != b:
        yield f"{path}: {a!r} vs {b!r}"


def traced(run: Path) -> list[dict]:
    """Return every traced batch of a run, in the order each rank wrote them."""
    return [json.loads(line) for file in sorted((run / "trace").glob("rank*.jsonl")) for line in file.open()]


def train_batches(rows: list[dict]) -> dict[tuple[int, int, int], tuple[list, list]]:
    """(rank, epoch, step) -> (fields, patch origins) of each train batch; a step traced twice keeps its last."""
    return {(r["rank"], r["epoch"], r["step"]): (r["positions"], r["origins"]) for r in rows if r["stage"] == "train"}


def shards(name: str, rows: list[dict]) -> bool:
    """Print the fields each rank read per stage and epoch; return whether the ranks' fields never overlap."""
    read: dict[tuple[str, int], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        read[row["stage"], row["epoch"]][row["rank"]].update(row["positions"])
    disjoint = True
    for (stage, epoch), ranks in sorted(read.items()):
        fields = [ranks[rank] for rank in sorted(ranks)]
        overlap = set.intersection(*fields) if len(fields) > 1 else set()
        disjoint &= not overlap
        counts = {rank: len(ranks[rank]) for rank in sorted(ranks)}
        print(f"{name} {stage} epoch {epoch}: fields per rank {counts}, overlap {sorted(overlap) or 'none'}")
    return disjoint


def main(first: str, second: str) -> int:
    """Print the comparison of the run folders ``first`` and ``second``; return 0 when they agree, else 1."""
    a, b = Path(first), Path(second)
    ca, cb = (torch.load(run / "checkpoints" / "last.ckpt", weights_only=False, map_location="cpu") for run in (a, b))
    same = ca["epoch"] == cb["epoch"] and ca["global_step"] == cb["global_step"]
    print(f"epoch {ca['epoch']} vs {cb['epoch']}; global_step {ca['global_step']} vs {cb['global_step']}")
    for part in PARTS:
        found = list(differences(ca.get(part), cb.get(part), part))
        same &= not found
        print(f"{part}: {'identical' if not found else f'{len(found)} differences, e.g. ' + '; '.join(found[:3])}")
    ra, rb = traced(a), traced(b)
    ta, tb = train_batches(ra), train_batches(rb)
    agreeing = sum(1 for key, value in ta.items() if tb.get(key) == value)
    same &= agreeing == len(ta) == len(tb)
    print(f"train batches: {len(ta)} vs {len(tb)}; identical (rank, epoch, step, fields, origins): {agreeing}")
    differing = sorted(key for key in ta if tb.get(key) != ta[key])
    if differing:
        print("first differing train batches (rank, epoch, step):", differing[:6])
    attempts = sorted({row["attempt"] for row in rb})
    print(f"{b.name}: attempts traced {attempts}")
    same &= shards(a.name, ra) & shards(b.name, rb)
    print("PASS" if same else "FAIL")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
