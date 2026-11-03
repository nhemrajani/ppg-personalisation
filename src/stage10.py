"""Stage 10: the floor baselines, Arms Z0 and Z1.

Registered in docs/pre-registered-decisions.md on 3 November 2026 and committed
before this ran. Neither arm consults an embedding or a model output: Z0 predicts
the population's mean heart rate for every test window, Z1 the target's own mean
over its budget.

The point is not that either arm is good. It is that the study had measured six
adaptation methods against the frozen population model and never against not
having a model, which leaves every improvement figure unanchored. Arm Z1 costs
exactly what Arm B-offset costs, one parameter and four bytes, so the gap between
them is what the foundation model buys with the per-person mean held constant.

    python -m src.stage10
"""

from __future__ import annotations

import collections
import time

import numpy as np

from src.arms import arm_z0, arm_z1, check_floors
from src.harness import Run, append
from src.ridge import mae, pearson, rmse
from src.splits import budget_prefix, load_index, make_split

BUDGETS = (2, 5, 10, 20, 40, None)
PROTOCOLS = ("activity", "temporal")
BACKBONES = ("p", "s")


def main() -> None:
    index = load_index()
    y, subject = index["heart_rate"], index["subject"]
    subjects = np.unique(subject)

    print("checking the floor arms against their registered assertions")
    for protocol in PROTOCOLS:
        for target in subjects:
            check_floors(None, y, make_split(index, int(target), protocol=protocol), subject)
    print(f"  constant, correctly valued and correctly priced for all "
          f"{len(subjects)} subjects in both protocols\n")

    started, rows = time.time(), []
    for backbone in BACKBONES:
        for protocol in PROTOCOLS:
            for target in subjects:
                split = make_split(index, int(target), protocol=protocol)
                y_test = y[split.test]

                def record(arm, fit, budget):
                    rows.append(Run(
                        arm=arm, backbone=backbone, polarity="asis", protocol=protocol,
                        subject=int(target), budget_minutes=budget,
                        usable_minutes=None if budget is None else budget,
                        trainable_parameters=fit.trainable_parameters,
                        stored_bytes=fit.stored_bytes,
                        mae=round(mae(y_test, fit.predictions), 4),
                        rmse=round(rmse(y_test, fit.predictions), 4),
                        pearson=round(pearson(y_test, fit.predictions), 4),
                        n_test_windows=len(y_test),
                        hyperparameters=str(fit.extra or {}),
                        notes="floor baseline, no model output used",
                    ))

                # Z0 does not depend on the budget, so it is recorded once per
                # cell with the budget blank, exactly as Arm A is. Stated as a
                # deviation in the registration rather than left implicit.
                record("Z0", arm_z0(None, y, split, subject), None)
                for budget in BUDGETS:
                    chunk = budget_prefix(split, index, budget)
                    record("Z1", arm_z1(None, y, split, subject, chunk), budget)

    # Both floors ignore the embedding entirely, so the two backbones must agree
    # exactly. Asserted rather than assumed, because it is the property that lets
    # a change in "improvement over the floor" between backbones be attributed to
    # the encoder alone.
    by_cell = collections.defaultdict(dict)
    for r in rows:
        by_cell[(r.arm, r.protocol, r.subject, r.budget_minutes)][r.backbone] = r.mae
    for key, got in by_cell.items():
        if len(got) == 2:
            assert got["p"] == got["s"], f"{key}: floor differs by backbone, {got}"
    print(f"floors agree exactly across both backbones in all {len(by_cell)} cells, "
          f"as they must: neither reads an embedding\n")

    append(rows)
    print(f"wrote {len(rows):,} rows in {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
