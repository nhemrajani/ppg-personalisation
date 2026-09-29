"""Does the population model saturate in the number of subjects?

Stage 4 left a loose observation: the reproduction trains on nine subjects and
Arm A on fourteen, yet their central values are close. The aggregations differ,
so that comparison cannot carry weight. This measures it directly, holding
everything else fixed.

For each held-out subject, the population model is fitted on a random subset of
the remaining fourteen, at sizes from two upwards, with the same standardiser,
the same committed alpha grid and the same grouped selection as Arm A. If error
stops falling well before fourteen, the binding constraint is not the amount of
population data. It is individual variation, which is what per-person
adaptation exists to address, so this speaks directly to the study's premise.
"""

from __future__ import annotations

import numpy as np

from src.harness import Run, append
from src.ridge import GramRidge, Standardiser, group_fold_indices, mae, pearson, rmse, select_alpha
from src.splits import load_index
from src.stage4 import OUR_ALPHAS, OUR_FOLDS, embeddings

SIZES = (2, 4, 6, 8, 10, 12, 14)
SEEDS = tuple(range(25))  # 25 draws per size; three was too few to separate sizes


def main() -> None:
    index = load_index()
    y_all, subject = index["heart_rate"], index["subject"]
    subjects = np.unique(subject)

    for backbone in ("p", "s"):
        x_all = np.asarray(embeddings(backbone, "asis"), dtype=np.float64)
        print(f"\nPaPaGei-{backbone.upper()}")
        curve: dict[int, list[float]] = {n: [] for n in SIZES}
        curve_subjects: dict[int, list[int]] = {n: [] for n in SIZES}

        for held_out in subjects:
            pool = np.array([s for s in subjects if s != held_out])
            test = subject == held_out
            y_test = y_all[test]

            for size in SIZES:
                seeds = (0,) if size == len(pool) else SEEDS
                for seed in seeds:
                    chosen = pool if size == len(pool) else np.random.default_rng(
                        seed * 100 + int(held_out)
                    ).choice(pool, size, replace=False)
                    train = np.isin(subject, chosen)

                    scaler = Standardiser.fit(x_all[train])
                    x_train, x_test = scaler(x_all[train]), scaler(x_all[test])
                    y_train = y_all[train]

                    folds = min(OUR_FOLDS, size)
                    alpha = select_alpha(
                        x_train, y_train, OUR_ALPHAS,
                        group_fold_indices(subject[train], folds), "mae",
                    )
                    prediction = GramRidge(x_train, y_train).predict(x_test, alpha)
                    score = mae(y_test, prediction)
                    curve[size].append(score)
                    curve_subjects[size].append(int(held_out))

                    append(Run(
                        arm="A-population-size", backbone=backbone, polarity="asis",
                        protocol="leave-one-subject-out", subject=int(held_out),
                        mae=round(score, 4), rmse=round(rmse(y_test, prediction), 4),
                        pearson=round(pearson(y_test, prediction), 4),
                        n_test_windows=int(test.sum()), alpha=alpha, seed=seed,
                        hyperparameters=f'{{"population_subjects": {size}, "folds": {folds}}}',
                    ))
            print(f"  S{held_out} done", flush=True)

        print(f"  {'subjects':>9s} {'mean MAE':>9s} {'sem':>7s} {'change':>8s}")
        previous = None
        for size in SIZES:
            values = np.array(curve[size])
            value = float(values.mean())
            sem = float(values.std(ddof=1) / np.sqrt(len(values)))
            change = "" if previous is None else f"{value - previous:+8.3f}"
            print(f"  {size:9d} {value:9.2f} {sem:7.3f} {change:>8s}")
            previous = value

        # The six-versus-fourteen comparison, paired within subject rather than as
        # two means with overlapping error bars.
        per_subject = {}
        for size in (6, 14):
            by_subject: dict[int, list[float]] = {}
            for value, subj in zip(curve[size], curve_subjects[size]):
                by_subject.setdefault(subj, []).append(value)
            per_subject[size] = np.array([np.mean(by_subject[s]) for s in sorted(by_subject)])
        difference = per_subject[6] - per_subject[14]
        rng = np.random.default_rng(0)
        boot = np.array([difference[rng.integers(0, len(difference), len(difference))].mean()
                         for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n  paired per-subject difference, six against fourteen population subjects:")
        print(f"    mean {difference.mean():+.3f} BPM, 95% interval [{lo:+.3f}, {hi:+.3f}], "
              f"worse with six on {int((difference > 0).sum())}/15 subjects")


if __name__ == "__main__":
    main()
