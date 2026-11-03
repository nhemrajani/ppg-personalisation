"""What the floor baselines show, and what Arm B's fitted scale shows.

Reads results/experiments.csv only. Two questions, both registered on
3 November 2026 before Arms Z0 and Z1 were fitted:

  - What does the foundation model contribute over knowing nothing (Arm Z0) and
    over knowing a person's own average heart rate (Arm Z1)?
  - How far below 1 does Arm B's fitted scale sit, and what does that mean?

Every comparison is paired per subject and bootstrapped over subjects, which is
the study's convention: the arms share a test block, so a difference of means
would discard the pairing that makes fifteen subjects usable.

    python -m src.floors
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "results" / "experiments.csv"
RESAMPLES = 5000
# Backbones in the order they entered the study; a cell appears when its rows
# do, so MOMENT joined these tables without this file changing.
BACKBONE_ORDER = ("p", "s", "moment")
CELLS = tuple((b, p) for b in BACKBONE_ORDER for p in ("activity", "temporal"))
BUDGETS = ("2", "5", "10", "20", "40", "")
BUDGET_LABEL = {"": "all", "2": "2", "5": "5", "10": "10", "20": "20", "40": "40"}
# Arms with no budget dimension, broadcast across the sweep for comparison.
FLAT = ("A", "Z0")


def load() -> dict:
    """MAE keyed by (arm, backbone, protocol, budget, subject)."""
    table = {}
    with open(RECORD, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mae"] == "" or row["rank"] not in ("", "16"):
                continue  # the C2 rank sweep enters only at its best rank
            arm = row["arm"] if row["arm"] != "C2" else "C2-r16"
            budget = "" if row["arm"] in FLAT else row["budget_minutes"]
            table[(arm, row["backbone"], row["protocol"], budget,
                   row["subject"])] = float(row["mae"])
    return table


def series(table, arm, backbone, protocol, budget):
    """One MAE per subject, or None if the cell is not populated."""
    want = "" if arm in FLAT else budget
    got = [(int(s), v) for (a, b, p, g, s), v in table.items()
           if a == arm and b == backbone and p == protocol and g == want]
    if len(got) != 15:
        return None
    return np.array([v for _, v in sorted(got)])


def paired(better: np.ndarray, worse: np.ndarray, seed: int = 0):
    """Mean per-subject advantage of `better` over `worse`, with an interval.

    Positive means `better` has the lower error. Subjects are resampled, not
    windows: the claim is about people, and with fifteen of them the subject is
    the unit the uncertainty lives in.
    """
    diff = worse - better
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, len(diff), (RESAMPLES, len(diff)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high), int((diff > 0).sum())


def detectable(low: float, high: float) -> bool:
    return low > 0 or high < 0


def main() -> None:
    table = load()
    arms = ("Z0", "Z1", "A", "B-offset", "B", "B-ridge", "B2", "C1", "C2-r16", "D")

    print("=" * 78)
    print("THE FLOOR, AND WHAT THE MODEL ADDS TO IT")
    print("=" * 78)
    print("\nMean per-subject MAE in BPM, at all available personal data.\n")
    head = f"{'arm':10s}" + "".join(f"{b}/{p[:4]:>12s}" for b, p in CELLS)
    print(head)
    print("-" * len(head))
    for arm in arms:
        cells = [series(table, arm, b, p, "") for b, p in CELLS]
        if all(c is None for c in cells):
            continue
        line = f"{arm:10s}"
        for c in cells:
            line += f"{c.mean():>13.2f}" if c is not None else f"{'-':>13s}"
        print(line)

    print("\n" + "=" * 78)
    print("THE CONTRASTS THAT THE FLOOR WAS REGISTERED TO SETTLE")
    print("=" * 78)
    print("\nPaired per-subject, 5,000-resample bootstrap over subjects, all available data.")
    print("Positive means the first arm has the lower error. '*' is detectable.\n")

    contrasts = (("A", "Z0"), ("A", "Z1"), ("Z1", "Z0"),
                 ("B-offset", "Z1"), ("B", "Z1"), ("C1", "Z1"), ("C2-r16", "Z1"))
    rows = []
    for better, worse in contrasts:
        print(f"  {better} vs {worse}")
        for b, p in CELLS:
            x, z = series(table, better, b, p, ""), series(table, worse, b, p, "")
            if x is None or z is None:
                continue
            d, lo, hi, won = paired(x, z)
            mark = "*" if detectable(lo, hi) else " "
            print(f"    {b}/{p:9s} {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]{mark}  "
                  f"better on {won:2d}/15")
            rows.append({"contrast": f"{better} vs {worse}", "backbone": b,
                         "protocol": p, "budget_minutes": "all",
                         "mean_advantage_bpm": round(d, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "subjects_better": won, "detectable": detectable(lo, hi)})
        print()

    print("=" * 78)
    print("THE FLOOR ACROSS THE BUDGET SWEEP")
    print("=" * 78)
    print("\nMean per-subject MAE, PaPaGei-P, activity-stratified. Arm A and Arm Z0")
    print("have no budget dimension and are repeated for reading.\n")
    header = f"{'arm':10s}" + "".join(f"{BUDGET_LABEL[g]:>8s}" for g in BUDGETS)
    print(header)
    print("-" * len(header))
    for arm in arms:
        line, any_value = f"{arm:10s}", False
        for g in BUDGETS:
            c = series(table, arm, "p", "activity", g)
            line += f"{c.mean():>8.2f}" if c is not None else f"{'-':>8s}"
            any_value = any_value or c is not None
        if any_value:
            print(line)

    # Z1 against A at every budget, which is where prediction 6 is scored.
    print("\nArm Z1 against Arm A at every budget, paired, all four cells.")
    print("Positive means Arm A is better, so a negative entry is the probe losing")
    print("to a per-person constant.\n")
    print(f"{'cell':14s}" + "".join(f"{BUDGET_LABEL[g]:>9s}" for g in BUDGETS))
    for b, p in CELLS:
        line = f"{b}/{p:11s}"
        for g in BUDGETS:
            a, z = series(table, "A", b, p, g), series(table, "Z1", b, p, g)
            if a is None or z is None:
                line += f"{'-':>9s}"
                continue
            d, lo, hi, _ = paired(a, z)
            line += f"{d:>+8.2f}" + ("*" if detectable(lo, hi) else " ")
            rows.append({"contrast": "A vs Z1", "backbone": b, "protocol": p,
                         "budget_minutes": BUDGET_LABEL[g],
                         "mean_advantage_bpm": round(d, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4), "subjects_better": int((z - a > 0).sum()),
                         "detectable": detectable(lo, hi)})
        print(line)

    out = ROOT / "results" / "floor_comparison.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)}, {len(rows)} contrasts")

    scale_summary()


def scale_summary() -> None:
    """Arm B's and Arm B-ridge's fitted scale, now that both are columns."""
    print("\n" + "=" * 78)
    print("THE FITTED CORRECTION: HOW MUCH OF THE MODEL ARM B KEEPS")
    print("=" * 78)
    print("\nA scale of 1 keeps the model's prediction and shifts it. A scale of 0")
    print("discards it and substitutes a constant, which is Arm Z1 exactly.\n")

    by = collections.defaultdict(list)
    with open(RECORD, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["fitted_scale"] == "":
                continue
            by[(row["arm"], row["backbone"], row["protocol"],
                row["budget_minutes"])].append((int(row["subject"]),
                                                float(row["fitted_scale"]),
                                                float(row["fitted_offset"])))

    rows = []
    for arm in ("B", "B-ridge"):
        print(f"  Arm {arm}")
        print(f"    {'cell':14s}{'budget':>8s}{'mean':>8s}{'median':>8s}"
              f"{'min':>8s}{'max':>8s}{'below 0.5':>11s}{'max offset':>12s}")
        for b, p in CELLS:
            for g in BUDGETS:
                got = by.get((arm, b, p, g))
                if not got:
                    continue
                scales = np.array([s for _, s, _ in got])
                offsets = np.array([o for _, _, o in got])
                print(f"    {b}/{p:11s}{BUDGET_LABEL[g]:>8s}{scales.mean():>8.3f}"
                      f"{np.median(scales):>8.3f}{scales.min():>8.3f}{scales.max():>8.3f}"
                      f"{int((scales < 0.5).sum()):>8d}/15"
                      f"{offsets[np.argmax(np.abs(offsets))]:>+12.1f}")
                rows.append({"arm": arm, "backbone": b, "protocol": p,
                             "budget_minutes": BUDGET_LABEL[g],
                             "scale_mean": round(float(scales.mean()), 4),
                             "scale_median": round(float(np.median(scales)), 4),
                             "scale_min": round(float(scales.min()), 4),
                             "scale_max": round(float(scales.max()), 4),
                             "subjects_below_half": int((scales < 0.5).sum()),
                             "offset_largest_absolute": round(
                                 float(offsets[np.argmax(np.abs(offsets))]), 4)})
            print()

    out = ROOT / "results" / "scale_distribution.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}, {len(rows)} rows")


if __name__ == "__main__":
    main()
