"""Rerun Arms C1 and C1-scratch after the selection defect of 26 October 2026.

Only these two arms are affected. Arms A, B and B2 do not select a shrinkage
and their rows stand. The two diagnostics registered with the fix are reported
here rather than asserted: whether the selected penalty is still pinned to a
grid endpoint for every subject, and whether it strengthens as the budget
shrinks. Both are empirical predictions registered on the same day, so a
failure is evidence about the data and not grounds to stop the run.
"""

from __future__ import annotations

import collections
import json
import time

import numpy as np

from src.arms import SHRINKAGE, arm_a, arm_c1, check_degenerate
from src.effective_n import effective_n
from src.harness import Run, append
from src.ridge import mae, pearson, rmse
from src.splits import budget_prefix, load_index, make_split
from src.stage4 import embeddings

BUDGETS = (2, 5, 10, 20, 40, None)
PROTOCOLS = ("activity", "temporal")
BACKBONES = ("p", "s")


def main() -> None:
    index = load_index()
    y, subject, starts = index["heart_rate"], index["subject"], index["start_second"]
    subjects = np.unique(subject)

    x_check = np.asarray(embeddings("p", "asis"), dtype=np.float64)
    check_degenerate(x_check, y, make_split(index, int(subjects[0])), subject)
    print("degenerate assertions pass for every arm\n")

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
                    eff = effective_n(starts[chunk])
                    for name, warm in (("C1", True), ("C1-scratch", False)):
                        fitted = arm_c1(x, y, split, subject, chunk, base, warm=warm)
                        extra = dict(fitted.extra or {})
                        extra["effective_n"] = eff
                        extra["n_windows"] = int(len(chunk))
                        if name == "C1" and isinstance(extra.get("shrinkage"), float):
                            picked[(backbone, protocol, budget)].append(extra["shrinkage"])
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
                            alpha=fitted.alpha,
                            hyperparameters=json.dumps(extra),
                        ))
            print(f"  {backbone}/{protocol} done, {len(rows):,} rows, "
                  f"{time.time() - started:.0f}s", flush=True)

    append(rows)
    print(f"\nwrote {len(rows):,} rows in {time.time() - started:.0f}s\n")

    print("DIAGNOSTIC 1: is the selected penalty still pinned to a grid endpoint?")
    weakest, strongest = SHRINKAGE[0], SHRINKAGE[-1]
    pinned = []
    for key, values in sorted(picked.items(), key=lambda kv: str(kv[0])):
        at_weak = sum(1 for v in values if v == weakest)
        at_strong = sum(1 for v in values if v == strongest)
        flag = " <- all at one endpoint" if at_weak == len(values) or at_strong == len(values) else ""
        if flag:
            pinned.append(key)
        print(f"  {key[0]}/{key[1]:9s} budget {str(key[2] or 'all'):>3s}: "
              f"{at_weak:2d}/15 weakest, {at_strong:2d}/15 strongest{flag}")
    print(f"\n  cells pinned to a single endpoint for all fifteen subjects: {len(pinned)} of {len(picked)}")

    print("\nDIAGNOSTIC 2: does the penalty strengthen as the budget shrinks?")
    for backbone in BACKBONES:
        for protocol in PROTOCOLS:
            line = []
            for budget in BUDGETS:
                values = picked.get((backbone, protocol, budget), [])
                line.append(f"{np.median(np.log10(values)):+5.1f}" if values else "    -")
            print(f"  {backbone}/{protocol:9s} median log10 penalty: " + " ".join(line))
    print("  " + " " * 37 + " ".join(f"{str(b or 'all'):>5s}" for b in BUDGETS))


if __name__ == "__main__":
    main()
