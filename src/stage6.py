"""Stage 6: Arms B, B2 and C1 across the budget sweep.

Runs the registered grid and appends one row per run to the experiment record.
Before anything is fitted, every arm is checked against its degenerate setting:
if an arm cannot reproduce Arm A when it is doing nothing, the run stops.
"""

from __future__ import annotations

import time

import numpy as np

from src.arms import arm_a, arm_b, arm_b2, arm_c1, check_degenerate
from src.harness import Run, append
from src.ridge import mae, pearson, rmse
from src.splits import budget_prefix, load_index, make_split
from src.stage4 import embeddings

BUDGETS = (2, 5, 10, 20, 40, None)  # None is all available
PROTOCOLS = ("activity", "temporal")
BACKBONES = ("p", "s")


def main() -> None:
    index = load_index()
    y, subject = index["heart_rate"], index["subject"]
    subjects = np.unique(subject)

    print("checking that every arm reduces to Arm A in its degenerate setting")
    x_check = np.asarray(embeddings("p", "asis"), dtype=np.float64)
    check_degenerate(x_check, y, make_split(index, int(subjects[0])), subject)
    print("  all degenerate checks pass\n")

    started, rows = time.time(), []
    for backbone in BACKBONES:
        x = np.asarray(embeddings(backbone, "asis"), dtype=np.float64)
        for protocol in PROTOCOLS:
            for target in subjects:
                split = make_split(index, int(target), protocol=protocol)
                base = arm_a(x, y, split, subject)
                y_test = y[split.test]

                def record(arm, fit, budget, extra=None):
                    rows.append(Run(
                        arm=arm, backbone=backbone, polarity="asis", protocol=protocol,
                        subject=int(target), budget_minutes=budget,
                        usable_minutes=None if budget is None else budget,
                        trainable_parameters=fit.trainable_parameters,
                        stored_bytes=fit.stored_bytes,
                        mae=round(mae(y_test, fit.predictions), 4),
                        rmse=round(rmse(y_test, fit.predictions), 4),
                        pearson=round(pearson(y_test, fit.predictions), 4),
                        n_test_windows=len(y_test), alpha=fit.alpha,
                        hyperparameters=str(fit.extra or extra or {}),
                    ))

                record("A", base, None)
                for budget in BUDGETS:
                    chunk = budget_prefix(split, index, budget)
                    record("B", arm_b(x, y, split, subject, chunk, base), budget)
                    record("B2", arm_b2(x, y, split, subject, chunk, base, passes=1), budget)
                    record("C1", arm_c1(x, y, split, subject, chunk, base, warm=True), budget)
                    record("C1-scratch", arm_c1(x, y, split, subject, chunk, base, warm=False), budget)
            print(f"  {backbone}/{protocol} done, {len(rows):,} rows, {time.time()-started:.0f}s",
                  flush=True)

    append(rows)
    print(f"\nwrote {len(rows):,} rows in {time.time()-started:.0f}s")

    # Headline: each arm against Arm A and against Arm B, at all-available data.
    import collections
    best = collections.defaultdict(list)
    for r in rows:
        if r.protocol == "activity" and r.backbone == "p" and r.budget_minutes is None:
            best[r.arm].append(r.mae)
    baseline = np.mean(best["A"]) if best["A"] else float("nan")
    print(f"\nPaPaGei-P, activity-stratified, all available data, mean per-subject MAE")
    for arm in ("A", "B", "B2", "C1", "C1-scratch"):
        if best[arm]:
            value = np.mean(best[arm])
            print(f"  {arm:12s} {value:6.2f}   against Arm A {value - baseline:+6.2f}")


if __name__ == "__main__":
    main()
