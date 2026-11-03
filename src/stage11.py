"""Stage 11: does the finding survive a stronger backbone?

Registered 3 November 2026. The objection the floor baselines create is that the
estimator is weak, so of course cheap corrections beat capacity. MOMENT is the
strongest frozen probe in the reference table at a reported 8.82, and every
closed-form arm reads a cached table without knowing which encoder wrote it, so
the whole comparison transfers unchanged.

Two halves, run in order:

    python -m src.stage11 reproduce   # 455 triples against the published 8.82
    python -m src.stage11 arms        # A, B, B-offset, B-ridge, B2, C1, Z0, Z1

The reproduction is the gate. If MOMENT's embeddings do not land near 8.82 under
PaPaGei's own protocol then the table is wrong and the arms would be measuring
the mistake. Arms C2 and D are architecture-specific and are not run.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.arms import (arm_a, arm_b, arm_b2, arm_b_offset, arm_b_ridge, arm_c1,
                      arm_z0, arm_z1, check_degenerate, check_floors)
from src.harness import Run, append
from src.ridge import mae, pearson, rmse
from src.splits import budget_prefix, load_index, make_split
from src.stage4 import PUBLISHED, embeddings, reproduction

BACKBONE = "moment"
BUDGETS = (2, 5, 10, 20, 40, None)
PROTOCOLS = ("activity", "temporal")


def reproduce() -> None:
    """PaPaGei's own probe protocol on the MOMENT table, over all 455 triples."""
    index = load_index()
    runs = reproduction(index, BACKBONE)
    append(runs)

    values = np.array([r.mae for r in runs])
    published = PUBLISHED[BACKBONE]
    inside = values.min() <= published <= values.max()
    print(f"\n  MOMENT-1-large  published {published}")
    print(f"    our 455 triples: median {np.median(values):.2f}, "
          f"range {values.min():.2f} to {values.max():.2f}")
    print(f"    central 95%: {np.percentile(values, 2.5):.2f} to "
          f"{np.percentile(values, 97.5):.2f}")
    print(f"    published figure sits at the {float((values < published).mean() * 100):.1f}"
          f"th percentile of our distribution")
    print(f"    inside our range: {inside}")
    for name in ("p", "s"):
        print(f"    for comparison, PaPaGei-{name.upper()} published {PUBLISHED[name]}")
    if not inside:
        print("\n    MOMENT does NOT reproduce. Read the registration before running the")
        print("    arms: the table, not the arms, is what needs explaining.")


def arms() -> None:
    """Every closed-form arm on the MOMENT table, both protocols, all budgets."""
    index = load_index()
    y, subject = index["heart_rate"], index["subject"]
    subjects = np.unique(subject)
    x = np.asarray(embeddings(BACKBONE), dtype=np.float64)
    print(f"MOMENT table: {x.shape}, {x.dtype}")

    print("\nchecking every arm against its registered assertions on this table")
    first = make_split(index, int(subjects[0]))
    check_degenerate(x, y, first, subject)
    check_floors(x, y, first, subject)
    print("  all degenerate and floor checks pass\n")

    started, rows = time.time(), []
    for protocol in PROTOCOLS:
        for target in subjects:
            split = make_split(index, int(target), protocol=protocol)
            base = arm_a(x, y, split, subject)
            y_test = y[split.test]

            def record(arm, fit, budget):
                extra = fit.extra or {}
                rows.append(Run(
                    arm=arm, backbone=BACKBONE, polarity="asis", protocol=protocol,
                    subject=int(target), budget_minutes=budget,
                    usable_minutes=None if budget is None else budget,
                    trainable_parameters=fit.trainable_parameters,
                    stored_bytes=fit.stored_bytes,
                    mae=round(mae(y_test, fit.predictions), 4),
                    rmse=round(rmse(y_test, fit.predictions), 4),
                    pearson=round(pearson(y_test, fit.predictions), 4),
                    n_test_windows=len(y_test), alpha=fit.alpha,
                    fitted_scale=extra.get("scale"),
                    fitted_offset=extra.get("offset"),
                    hyperparameters=str(extra),
                ))

            record("A", base, None)
            record("Z0", arm_z0(x, y, split, subject), None)
            for budget in BUDGETS:
                chunk = budget_prefix(split, index, budget)
                record("Z1", arm_z1(x, y, split, subject, chunk), budget)
                record("B", arm_b(x, y, split, subject, chunk, base), budget)
                record("B-offset", arm_b_offset(x, y, split, subject, chunk, base), budget)
                record("B-ridge", arm_b_ridge(x, y, split, subject, chunk, base), budget)
                record("B2", arm_b2(x, y, split, subject, chunk, base, passes=1), budget)
                record("C1", arm_c1(x, y, split, subject, chunk, base, warm=True), budget)
        print(f"  {protocol} done, {len(rows):,} rows, {time.time()-started:.0f}s",
              flush=True)

    # The floors read no embedding, so MOMENT's must equal the ones Stage 10
    # wrote for PaPaGei on the same splits. If they do not, the splits moved.
    import csv
    from pathlib import Path
    record_path = Path(__file__).resolve().parent.parent / "results" / "experiments.csv"
    previous = {}
    with open(record_path, newline="") as handle:
        for old in csv.DictReader(handle):
            if old["arm"] in ("Z0", "Z1") and old["backbone"] == "p":
                previous[(old["arm"], old["protocol"], old["subject"],
                          old["budget_minutes"])] = float(old["mae"])
    checked = 0
    for r in rows:
        if r.arm not in ("Z0", "Z1"):
            continue
        key = (r.arm, r.protocol, str(r.subject),
               "" if r.budget_minutes is None else str(r.budget_minutes))
        if key in previous:
            assert abs(previous[key] - r.mae) < 1e-9, \
                f"{key}: floor moved between backbones, {previous[key]} vs {r.mae}"
            checked += 1
    print(f"  floors match Stage 10's in all {checked} shared cells: the splits "
          f"are identical\n")

    append(rows)
    print(f"wrote {len(rows):,} rows in {time.time()-started:.0f}s")

    import collections
    best = collections.defaultdict(list)
    for r in rows:
        if r.protocol == "activity" and r.budget_minutes is None:
            best[r.arm].append(r.mae)
        elif r.protocol == "activity" and r.arm in ("A", "Z0"):
            best[r.arm].append(r.mae)
    baseline = np.mean(best["A"])
    print("\nMOMENT, activity-stratified, all available data, mean per-subject MAE")
    for arm in ("Z0", "Z1", "A", "B-offset", "B", "B-ridge", "B2", "C1"):
        if best[arm]:
            value = np.mean(best[arm])
            print(f"  {arm:10s} {value:6.2f}   against Arm A {baseline - value:+6.2f}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "reproduce":
        reproduce()
    elif step == "arms":
        arms()
    else:
        raise SystemExit(__doc__)
