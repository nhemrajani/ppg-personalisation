"""Why Arm B's fitted scale sits below 1, decomposed exactly.

The scale was read as the correction discarding the model: at 0.207 on S5 it
keeps a fifth of the prediction and supplies the rest as a constant. The reading
is arithmetically right and the implication does not follow, because Arm B's
scale is ordinary least squares and OLS has a closed form for it.

Regressing truth on a predictor gives

    scale = r * sd(truth) / sd(prediction)

so the scale is the model's correlation with truth, times the ratio of their
spreads. Neither factor is a free choice. A scale of 0.69 at a correlation of
0.65 is the statistically optimal amount of shrinkage for a predictor that
explains 42 per cent of the variance, not a sign that the correction has given
up on the model. Arm Z1 is what giving up actually looks like, and it is 4.59
BPM worse.

The identity is asserted here rather than argued, and the two factors are
reported separately so the paper can say which one moves.

    python -m src.scale_mechanism
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.arms import _adaptation_predictions, arm_a
from src.splits import budget_prefix, load_index, make_split
from src.stage4 import embeddings

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = (2, 5, 10, 20, 40, None)
CELLS = (("p", "activity"), ("p", "temporal"), ("s", "activity"), ("s", "temporal"))
TOLERANCE = 1e-10


def main() -> None:
    index = load_index()
    y, subject = index["heart_rate"], index["subject"]
    subjects = np.unique(subject)
    rows = []

    print("Arm B's fitted scale, decomposed as r * sd(truth) / sd(prediction).")
    print("The identity is exact, so the question is only which factor moves.\n")
    print(f"{'cell':14s}{'budget':>8s}{'scale':>8s}{'=':>3s}{'r':>8s}"
          f"{'x':>3s}{'sd ratio':>10s}{'worst error':>14s}")
    print("-" * 68)

    for backbone, protocol in CELLS:
        x = np.asarray(embeddings(backbone, "asis"), dtype=np.float64)
        bases = {}
        for budget in BUDGETS:
            scales, correlations, ratios, errors = [], [], [], []
            for target in subjects:
                split = make_split(index, int(target), protocol=protocol)
                if target not in bases:
                    bases[target] = arm_a(x, y, split, subject)
                chunk = budget_prefix(split, index, budget)
                pred, _ = _adaptation_predictions(x, y, split, chunk, bases[target])
                truth = y[chunk]
                scale = float(np.polyfit(pred, truth, 1)[0])
                r = float(np.corrcoef(pred, truth)[0, 1])
                ratio = float(np.std(truth) / np.std(pred))
                errors.append(abs(scale - r * ratio))
                scales.append(scale); correlations.append(r); ratios.append(ratio)
                rows.append({"backbone": backbone, "protocol": protocol,
                             "budget_minutes": "all" if budget is None else budget,
                             "subject": int(target),
                             "fitted_scale": round(scale, 6),
                             "adaptation_r": round(r, 6),
                             "spread_ratio": round(ratio, 6)})
            worst = max(errors)
            assert worst < TOLERANCE, f"the OLS identity fails by {worst:.2e}"
            print(f"{backbone}/{protocol:11s}{'all' if budget is None else budget:>8}"
                  f"{np.mean(scales):>8.3f}{'=':>3s}{np.mean(correlations):>8.3f}"
                  f"{'x':>3s}{np.mean(ratios):>10.3f}{worst:>14.1e}")
        print()

    out = ROOT / "results" / "scale_mechanism.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}, {len(rows)} rows")

    # The subject the reading was built on. Its scale is low because the model
    # has almost no correlation with its heart rate, which is a statement about
    # the model on that subject rather than about the correction.
    worst = [r for r in rows if r["backbone"] == "p" and r["protocol"] == "activity"
             and r["budget_minutes"] == "all"]
    worst.sort(key=lambda r: r["fitted_scale"])
    print("\nThe three lowest scales at all available data, PaPaGei-P, activity:")
    for r in worst[:3]:
        print(f"  S{r['subject']:<3d} scale {r['fitted_scale']:.3f} "
              f"= r {r['adaptation_r']:.3f} x spread ratio {r['spread_ratio']:.3f}")
    print("\nA low scale is the model's correlation being low on that person.")
    print("Least squares is diagnosing the model, not discarding it.")


if __name__ == "__main__":
    main()
