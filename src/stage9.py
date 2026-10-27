"""Arms B-offset and B-ridge, registered 27 October 2026.

Tests whether Arm C1's cold-start advantage over Arm B is about capacity or
about being the only arm able to decline to adapt. Writes to its own file so it
cannot interleave with a Stage 7 job appending to the main one; merged in
afterwards by `python -m src.stage9 merge`.
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.arms import SHRINKAGE_B, arm_a, arm_b_offset, arm_b_ridge, check_degenerate
from src.effective_n import effective_n
from src.harness import EXPERIMENTS, Run, append, load
from src.provenance import exec_digest
from src.ridge import mae, pearson, rmse
from src.splits import budget_prefix, load_index, make_split
from src.stage4 import embeddings

OUT = Path("results/experiments_stage9.csv")
BUDGETS = (2, 5, 10, 20, 40, None)
PROTOCOLS = ("activity", "temporal")
BACKBONES = ("p", "s")


def main() -> None:
    index = load_index()
    y, subject, starts = index["heart_rate"], index["subject"], index["start_second"]
    subjects = np.unique(subject)

    x_check = np.asarray(embeddings("p", "asis"), dtype=np.float64)
    check_degenerate(x_check, y, make_split(index, int(subjects[0])), subject)
    print("degenerate assertions pass, including B-ridge at both endpoints\n")

    started, rows = time.time(), []
    picked = collections.defaultdict(list)
    for backbone in BACKBONES:
        x = np.asarray(embeddings(backbone, "asis"), dtype=np.float64)
        for protocol in PROTOCOLS:
            for target in subjects:
                split = make_split(index, int(target), protocol=protocol)
                base = arm_a(x, y, split, subject)
                y_test = y[split.test]
                for budget in BUDGETS:
                    chunk = budget_prefix(split, index, budget)
                    for name, fn in (("B-offset", arm_b_offset), ("B-ridge", arm_b_ridge)):
                        fitted = fn(x, y, split, subject, chunk, base)
                        extra = dict(fitted.extra or {})
                        extra["effective_n"] = effective_n(starts[chunk])
                        extra["n_windows"] = int(len(chunk))
                        extra["src_digest"] = exec_digest("stage9")
                        if name == "B-ridge" and isinstance(extra.get("shrinkage_b"), float):
                            picked[(backbone, protocol, budget)].append(extra["shrinkage_b"])
                        rows.append(Run(
                            arm=name, backbone=backbone, polarity="asis", protocol=protocol,
                            subject=int(target), budget_minutes=budget,
                            usable_minutes=round(len(chunk) * 2 / 60, 1),
                            elapsed_minutes=round(len(chunk) * 2 / 60, 1),
                            trainable_parameters=fitted.trainable_parameters,
                            stored_bytes=fitted.stored_bytes,
                            mae=round(mae(y_test, fitted.predictions), 4),
                            rmse=round(rmse(y_test, fitted.predictions), 4),
                            pearson=round(pearson(y_test, fitted.predictions), 4),
                            n_test_windows=int(len(split.test)),
                            alpha=fitted.alpha, hyperparameters=json.dumps(extra),
                        ))
            print(f"  {backbone}/{protocol} done, {len(rows):,} rows, "
                  f"{time.time() - started:.0f}s", flush=True)

    append(rows, path=OUT)
    print(f"\nwrote {len(rows):,} rows to {OUT} in {time.time() - started:.0f}s\n")

    print("B-ridge's selected shrinkage, median log10, by budget")
    print("  (prediction 4: it should weaken as the window becomes representative)")
    for backbone in BACKBONES:
        for protocol in PROTOCOLS:
            line = []
            for budget in BUDGETS:
                v = picked.get((backbone, protocol, budget), [])
                line.append(f"{np.median(np.log10(v)):+5.1f}" if v else "    -")
            print(f"  {backbone}/{protocol:9s} " + " ".join(line))
    print("  " + " " * 21 + " ".join(f"{str(b or 'all'):>5s}" for b in BUDGETS))
    print(f"  grid spans {np.log10(SHRINKAGE_B[0]):+.0f} to {np.log10(SHRINKAGE_B[-1]):+.0f}")


def merge() -> None:
    """Fold the stage 9 rows into the main record, once nothing is appending."""
    rows = load(OUT)
    existing = {(r["arm"], r["backbone"], r["protocol"], r["subject"], r["budget_minutes"])
                for r in load(EXPERIMENTS)}
    fresh = [r for r in rows
             if (r["arm"], r["backbone"], r["protocol"], r["subject"],
                 r["budget_minutes"]) not in existing]
    if not fresh:
        print("nothing to merge")
        return
    with EXPERIMENTS.open("a", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=list(load(EXPERIMENTS)[0]))
        writer.writerows(fresh)
    print(f"merged {len(fresh)} rows into {EXPERIMENTS}")


if __name__ == "__main__":
    merge() if len(sys.argv) > 1 and sys.argv[1] == "merge" else main()
