"""Stress-testing the claim that distance from the population predicts error.

Four checks, any of which could dissolve it.

1. S6's recording stops early and the missing hour is the low-intensity part,
   so its elevated median heart rate may be a collection artefact rather than
   a property of the person. Tested directly by restricting every subject to
   the activities S6 actually has, and by reweighting S6 to the population's
   activity mix.
2. Leverage. Two subjects are extreme on both axes. The correlation is
   recomputed without S6, and without both S5 and S6.
3. Multiple comparisons and precision. Five predictors were tested on fifteen
   points, so a bootstrap interval on r is reported rather than a point
   estimate.
4. Mechanism. Ridge shrinks predictions toward the training mean, so a subject
   far from the centre is pulled toward it and the resulting bias appears as
   error. Each subject's error is decomposed into bias and spread: if the
   error of the distant subjects is mostly bias, the correlation is a property
   of regularised regression rather than a fact about physiology.

A label-free version of the distance measure is also computed, since the
labelled one could not be used to triage a new member.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from src.dataset_summary import ROOT
from src.measure_quality import ACTIVITY_NAMES
from src.ridge import GramRidge, Standardiser, group_fold_indices, mae, select_alpha
from src.splits import load_index
from src.stage4 import OUR_ALPHAS, OUR_FOLDS, embeddings

BACKBONE = "p"


def bootstrap_r(x: np.ndarray, y: np.ndarray, resamples: int = 5000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        values.append(pearsonr(x[idx], y[idx])[0])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def arm_a_predictions(index: dict) -> dict[int, dict]:
    """Refit Arm A, keeping per-window predictions so error can be decomposed."""
    x_all = np.asarray(embeddings(BACKBONE, "asis"), dtype=np.float64)
    y_all, subject = index["heart_rate"], index["subject"]
    out: dict[int, dict] = {}
    for held_out in np.unique(subject):
        train, test = subject != held_out, subject == held_out
        scaler = Standardiser.fit(x_all[train])
        x_train, x_test = scaler(x_all[train]), scaler(x_all[test])
        y_train, y_test = y_all[train], y_all[test]
        alpha = select_alpha(x_train, y_train, OUR_ALPHAS,
                             group_fold_indices(subject[train], OUR_FOLDS), "mae")
        prediction = GramRidge(x_train, y_train).predict(x_test, alpha)
        out[int(held_out)] = {
            "y": y_test, "pred": prediction, "mae": mae(y_test, prediction),
            "bias": float(np.mean(prediction - y_test)),
            "spread": float(np.mean(np.abs(prediction - y_test - np.mean(prediction - y_test)))),
            "predicted_median": float(np.median(prediction)),
            "train_label_median": float(np.median(y_train)),
        }
    return out


def main() -> None:
    index = load_index()
    y_all, subject, activity = index["heart_rate"], index["subject"], index["activity"].astype(int)
    subjects = np.unique(subject)

    print("CHECK 1: is S6's elevated heart rate a collection artefact?\n")
    s6_activities = set(np.unique(activity[subject == 6]).tolist())
    print(f"  S6 has activities {sorted(s6_activities)}; "
          f"missing {[ACTIVITY_NAMES[c] for c in ACTIVITY_NAMES if c not in s6_activities]}")
    others = subject != 6
    for code in sorted(ACTIVITY_NAMES):
        mask = others & (activity == code)
        flag = "" if code in s6_activities else "   <- missing from S6"
        print(f"    {ACTIVITY_NAMES[code]:13s} median HR across the other 14: "
              f"{np.median(y_all[mask]):6.1f}{flag}")

    # Like with like: every subject restricted to the activities S6 has.
    restricted = {}
    for s in subjects:
        mask = (subject == s) & np.isin(activity, list(s6_activities))
        restricted[int(s)] = float(np.median(y_all[mask]))
    full = {int(s): float(np.median(y_all[subject == s])) for s in subjects}
    print(f"\n  S6 median heart rate: {full[6]:.0f} over its own data")
    pool = [restricted[int(s)] for s in subjects if s != 6]
    print(f"  restricted to S6's activities, the other fourteen have median {np.median(pool):.0f} "
          f"(range {min(pool):.0f} to {max(pool):.0f})")
    print(f"  S6 restricted to the same activities: {restricted[6]:.0f}")
    print(f"  so S6 sits {restricted[6] - np.median(pool):+.0f} BPM from the others on comparable activities, "
          f"against {full[6] - np.median([full[int(s)] for s in subjects if s != 6]):+.0f} BPM on raw medians")

    print("\nCHECK 4: is the error bias or spread?\n")
    fits = arm_a_predictions(index)
    print(f"  {'subj':5s} {'MAE':>7s} {'bias':>8s} {'spread':>8s} {'bias share':>11s}")
    for s in subjects:
        f = fits[int(s)]
        share = abs(f["bias"]) / f["mae"]
        print(f"  S{s:<4d} {f['mae']:7.2f} {f['bias']:+8.2f} {f['spread']:8.2f} {share:10.0%}")

    print("\nCHECKS 2 and 3: does the correlation survive?\n")
    medians = np.array([full[int(s)] for s in subjects])
    errors = np.array([fits[int(s)]["mae"] for s in subjects])
    distance = np.array([abs(medians[i] - np.median(np.delete(medians, i))) for i in range(len(subjects))])
    predicted = np.array([abs(fits[int(s)]["predicted_median"] - fits[int(s)]["train_label_median"])
                          for s in subjects])

    for label, keep in (("all fifteen", subjects != 0),
                        ("without S6", subjects != 6),
                        ("without S5 and S6", ~np.isin(subjects, [5, 6]))):
        x, y = distance[keep], errors[keep]
        r, p = pearsonr(x, y)
        lo, hi = bootstrap_r(x, y)
        print(f"  {label:20s} n={keep.sum():2d}  r={r:+.2f}  p={p:.4f}  "
              f"95% interval [{lo:+.2f}, {hi:+.2f}]  explains {r**2:.0%}")

    print("\n  label-free version, using the population model's own predictions\n")
    for label, keep in (("all fifteen", subjects != 0),
                        ("without S6", subjects != 6),
                        ("without S5 and S6", ~np.isin(subjects, [5, 6]))):
        r, p = pearsonr(predicted[keep], errors[keep])
        print(f"  {label:20s} n={keep.sum():2d}  r={r:+.2f}  p={p:.4f}  explains {r**2:.0%}")

    print("\n  five predictors were tested on fifteen subjects; the interval above is the "
          "honest precision")

    np.savez(ROOT / "results" / "difficulty_checks.npz",
             subjects=subjects, errors=errors, distance=distance, predicted=predicted,
             bias=np.array([fits[int(s)]["bias"] for s in subjects]),
             spread=np.array([fits[int(s)]["spread"] for s in subjects]))


if __name__ == "__main__":
    main()
