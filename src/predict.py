"""The prediction task: how much does heart rate change over the next H seconds?

Registered 3 November 2026, with the data cost measured before the design was
fixed. The target is `hr[t+H] - hr[t]` for H in {60, 120}, and the baseline that
matters is persistence, which predicts no change at all.

The arms are not reimplemented. A split whose blocks are restricted to windows
whose target is available, over a target vector holding the change rather than
the level, makes every closed-form arm in `src/arms.py` work unchanged. That is
worth more than the brevity: those arms carry degenerate assertions that have
been passing since Gate 6, and reimplementing them here would throw that away.

Two rules close the leakage trap, both asserted rather than assumed:

  - every buffer widens by the full horizon, so a window ending at the cut has
    its target ending a full window before the test block starts;
  - a window is usable only if its target lies in the same contiguous region,
    not merely the same block, so no target jumps over an intervening block.

    python -m src.predict 60,120 temporal,activity
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np

from src.arms import arm_a, arm_b, arm_b_offset, arm_c1, arm_z1, Fitted
from src.harness import Run, append
from src.preprocess import SHIFT_SECONDS
from src.ridge import mae, pearson, rmse
from src.splits import BUFFER_SECONDS, budget_prefix, load_index, make_split
from src.stage4 import embeddings

HORIZONS = (60, 120)
BACKBONE = "p"  # the primary estimation backbone; MOMENT is a later extension

# Registered 3 November 2026: S6 has 27.0 usable adaptation minutes at H=60 on
# the activity protocol and 17.5 at H=120, so the larger budgets are out of
# reach there. A budget no subject can reach is not a budget, and dropping
# subjects from the larger cells would break the paired comparison.
BUDGETS = {
    ("activity", 60): (2, 5, 10, 20, None),
    ("activity", 120): (2, 5, 10, None),
    ("temporal", 60): (2, 5, 10, 20, 40, None),
    ("temporal", 120): (2, 5, 10, 20, 40, None),
}


def deltas(index: dict[str, np.ndarray], horizon: int) -> np.ndarray:
    """`hr[t+H] - hr[t]` per row, NaN where the target does not exist.

    Computed over each subject's own recording, which is the physical
    contiguity. Whether a row is *usable* additionally depends on its block,
    and that is `usable` below.
    """
    starts, subject, hr = index["start_second"], index["subject"], index["heart_rate"]
    out = np.full(len(hr), np.nan)
    for s in np.unique(subject):
        rows = np.flatnonzero(subject == s)
        at = {float(t): i for t, i in zip(starts[rows], rows)}
        for t, i in zip(starts[rows], rows):
            j = at.get(float(t) + horizon)
            if j is not None:
                out[i] = hr[j] - hr[i]
    return out


def usable(rows: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    """Rows whose target lies in the same contiguous stretch of this block.

    The looser same-block rule would let a window at the end of one activity
    run take its target from a later run, jumping an intervening test block.
    Registered as the stricter rule; measured to differ by under 2 per cent.
    """
    if len(rows) == 0:
        return rows
    order = np.argsort(starts[rows], kind="stable")
    rows = rows[order]
    s = starts[rows]
    breaks = np.flatnonzero(np.diff(s) > SHIFT_SECONDS) + 1
    keep = np.zeros(len(rows), dtype=bool)
    for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, len(rows)]):
        have = set(s[lo:hi].tolist())
        keep[lo:hi] = [(t + horizon) in have for t in s[lo:hi]]
    return np.sort(rows[keep])


def restrict(split, index, horizon: int, delta: np.ndarray):
    """The split with every block cut down to rows that have a target."""
    starts, subject = index["start_second"], index["subject"]
    population = []
    for s in np.unique(subject[split.population]):
        rows = split.population[subject[split.population] == s]
        population.append(usable(rows, starts, horizon))
    restricted = replace(
        split,
        population=np.sort(np.concatenate(population)) if population else split.population,
        adaptation=usable(split.adaptation, starts, horizon),
        test=usable(split.test, starts, horizon),
    )
    for name in ("population", "adaptation", "test"):
        rows = getattr(restricted, name)
        assert np.isfinite(delta[rows]).all(), f"{name} holds a row with no target"
    return restricted


def check_no_leakage(split, restricted, index, horizon: int) -> None:
    """The assertion the whole task turns on: no target reaches the test block.

    For every adaptation window, the window H seconds later must not be a test
    window. Checked by start time rather than by construction, so that a change
    to the splitter cannot silently invalidate it.
    """
    starts = index["start_second"]
    test_starts = set(starts[split.test].tolist())
    for t in starts[restricted.adaptation]:
        assert (t + horizon) not in test_starts, (
            f"S{split.target} {split.protocol} H={horizon}: an adaptation window at "
            f"t={t:.0f} has its target at t={t + horizon:.0f}, which is a test window"
        )


def arm_persistence(split) -> Fitted:
    """Predict no change. Zero parameters, no labels, no model."""
    return Fitted(np.zeros(len(split.test)), 0, 0, None, {"constant": 0.0})


def main(horizons, protocols) -> None:
    index = load_index()
    subject, starts = index["subject"], index["start_second"]
    subjects = np.unique(subject)
    x = np.asarray(embeddings(BACKBONE), dtype=np.float64)

    started, rows = time.time(), []
    for horizon in horizons:
        delta = deltas(index, horizon)
        print(f"\nH = {horizon}s: {int(np.isfinite(delta).sum()):,} of {len(delta):,} "
              f"rows have a target; mean |change| "
              f"{np.nanmean(np.abs(delta)):.2f} BPM", flush=True)
        for protocol in protocols:
            for target in subjects:
                split = make_split(index, int(target), protocol=protocol,
                                   buffer_seconds=BUFFER_SECONDS + horizon)
                restricted = restrict(split, index, horizon, delta)
                check_no_leakage(split, restricted, index, horizon)
                if len(restricted.test) == 0 or len(restricted.adaptation) == 0:
                    raise SystemExit(f"S{target} {protocol} H={horizon} has an empty block")

                base = arm_a(x, delta, restricted, subject)
                y_test = delta[restricted.test]

                def record(arm, fit, budget):
                    extra = fit.extra or {}
                    rows.append(Run(
                        arm=arm, backbone=BACKBONE, polarity="asis",
                        protocol=f"{protocol}-h{horizon}", subject=int(target),
                        budget_minutes=budget,
                        usable_minutes=round(len(restricted.adaptation)
                                             * SHIFT_SECONDS / 60, 2)
                        if budget is None else budget,
                        trainable_parameters=fit.trainable_parameters,
                        stored_bytes=fit.stored_bytes,
                        mae=round(mae(y_test, fit.predictions), 4),
                        rmse=round(rmse(y_test, fit.predictions), 4),
                        pearson=round(pearson(y_test, fit.predictions), 4),
                        n_test_windows=len(y_test), alpha=fit.alpha,
                        fitted_scale=extra.get("scale"),
                        fitted_offset=extra.get("offset"),
                        hyperparameters=str(extra),
                        notes=f"prediction task, horizon {horizon}s",
                    ))

                record("A", base, None)
                record("persistence", arm_persistence(restricted), None)
                for budget in BUDGETS[(protocol, horizon)]:
                    chunk = budget_prefix(restricted, index, budget)
                    record("Z1", arm_z1(x, delta, restricted, subject, chunk), budget)
                    record("B", arm_b(x, delta, restricted, subject, chunk, base), budget)
                    record("B-offset",
                           arm_b_offset(x, delta, restricted, subject, chunk, base), budget)
                    record("C1", arm_c1(x, delta, restricted, subject, chunk, base,
                                        warm=True), budget)
            print(f"  {protocol} H={horizon} done, {len(rows):,} rows, "
                  f"{time.time()-started:.0f}s", flush=True)

    append(rows)
    print(f"\nwrote {len(rows):,} rows in {time.time()-started:.0f}s")

    import collections
    best = collections.defaultdict(list)
    for r in rows:
        if r.budget_minutes is None:
            best[(r.protocol, r.arm)].append(r.mae)
    for protocol in sorted({p for p, _ in best}):
        base = np.mean(best[(protocol, "persistence")])
        print(f"\n{protocol}, all available data, mean per-subject MAE on the change")
        for arm in ("persistence", "Z1", "A", "B-offset", "B", "C1"):
            if best[(protocol, arm)]:
                v = np.mean(best[(protocol, arm)])
                print(f"  {arm:12s} {v:6.2f}   against persistence {base - v:+6.2f}")


if __name__ == "__main__":
    hs = [int(h) for h in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(HORIZONS)
    ps = sys.argv[2].split(",") if len(sys.argv) > 2 else ["temporal", "activity"]
    main(hs, ps)
