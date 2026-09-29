"""Stage 4: the reproduction, and Arm A with the polarity control.

Two experiments, deliberately kept apart.

The reproduction follows PaPaGei's protocol exactly, on unflipped signal only:
a fixed subject-level split of nine training, three validation and three test
subjects, a StandardScaler fitted on train, ridge over their four alphas
selected by four-fold cross-validation on negative mean squared error, and MAE
reported. Their split file is not public, so every one of the 455 possible
test triples is run and the published figure is asked to fall inside the
resulting distribution. PaPaGei-P is the primary gate and PaPaGei-S the
secondary, per the registered 2x2.

Arm A is the study's own baseline: leave-one-subject-out over all fifteen
subjects, our own committed alpha grid, alpha selected by GroupKFold grouped
by subject. It runs on both checkpoints in both polarities, which gives the
paired comparison Sub Question 3 needs.

Every run appends a row to results/experiments.csv.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from src.embed import EMBEDDINGS
from src.harness import Run, append
from src.ridge import (
    GramRidge,
    Standardiser,
    bootstrap_interval,
    group_fold_indices,
    kfold_indices,
    mae,
    pearson,
    rmse,
    select_alpha,
)
from src.splits import load_index

# Registered before any fit. See docs/pre-registered-decisions.md.
PAPAGEI_ALPHAS = np.array([0.1, 1.0, 10.0, 100.0])
OUR_ALPHAS = np.logspace(-3, 6, 10)
PAPAGEI_FOLDS = 4
OUR_FOLDS = 7
PUBLISHED = {"s": 11.53, "p": 10.92}
PUBLISHED_INTERVAL = {"s": (11.40, 11.66), "p": (10.80, 11.04)}


def embeddings(backbone: str, polarity: str) -> np.ndarray:
    return np.load(EMBEDDINGS / f"papagei_{backbone}_{polarity}.npy", mmap_mode="r")


def reproduction(index: dict[str, np.ndarray], backbone: str) -> list[Run]:
    """PaPaGei's protocol over all 455 possible test triples."""
    x_all = np.asarray(embeddings(backbone, "asis"), dtype=np.float64)
    y_all, subject = index["heart_rate"], index["subject"]
    subjects = np.unique(subject)

    runs: list[Run] = []
    started = time.time()
    for n, triple in enumerate(itertools.combinations(subjects, 3), 1):
        remaining = [s for s in subjects if s not in triple]
        validation = remaining[:3]  # the three lowest remaining ids, registered in advance
        train_subjects = remaining[3:]  # the other nine; validation is held out unused

        train = np.isin(subject, train_subjects)
        test = np.isin(subject, triple)

        scaler = Standardiser.fit(x_all[train])
        x_train, x_test = scaler(x_all[train]), scaler(x_all[test])
        y_train, y_test = y_all[train], y_all[test]

        alpha = select_alpha(
            x_train, y_train, PAPAGEI_ALPHAS, kfold_indices(len(x_train), PAPAGEI_FOLDS), "mse"
        )
        prediction = GramRidge(x_train, y_train).predict(x_test, alpha)
        low, high = bootstrap_interval(y_test, prediction)

        runs.append(
            Run(
                arm="reproduction",
                backbone=backbone,
                polarity="asis",
                protocol="papagei-9-3-3",
                subject="+".join(f"S{s}" for s in triple),
                mae=round(mae(y_test, prediction), 4),
                rmse=round(rmse(y_test, prediction), 4),
                pearson=round(pearson(y_test, prediction), 4),
                n_test_windows=int(test.sum()),
                alpha=alpha,
                hyperparameters=f'{{"cv": {PAPAGEI_FOLDS}, "criterion": "mse", '
                                f'"validation_held_out": "{"+".join(f"S{s}" for s in validation)}"}}',
                notes=f"bootstrap_mae_ci=[{low:.4f},{high:.4f}]",
            )
        )
        if n % 100 == 0:
            print(f"    {n}/455 triples, {time.time() - started:.0f}s", flush=True)
    return runs


def arm_a(index: dict[str, np.ndarray], backbone: str, polarity: str) -> list[Run]:
    """Leave-one-subject-out, our own grid and grouped selection."""
    x_all = np.asarray(embeddings(backbone, polarity), dtype=np.float64)
    y_all, subject = index["heart_rate"], index["subject"]

    runs: list[Run] = []
    for held_out in np.unique(subject):
        train = subject != held_out
        test = ~train

        scaler = Standardiser.fit(x_all[train])
        x_train, x_test = scaler(x_all[train]), scaler(x_all[test])
        y_train, y_test = y_all[train], y_all[test]

        alpha = select_alpha(
            x_train, y_train, OUR_ALPHAS,
            group_fold_indices(subject[train], OUR_FOLDS), "mae",
        )
        prediction = GramRidge(x_train, y_train).predict(x_test, alpha)

        runs.append(
            Run(
                arm="A",
                backbone=backbone,
                polarity=polarity,
                protocol="leave-one-subject-out",
                subject=int(held_out),
                usable_minutes=round(float(test.sum()) * 2 / 60, 1),
                elapsed_minutes=round(float(test.sum()) * 2 / 60, 1),
                trainable_parameters=0,
                stored_bytes=0,
                mae=round(mae(y_test, prediction), 4),
                rmse=round(rmse(y_test, prediction), 4),
                pearson=round(pearson(y_test, prediction), 4),
                n_test_windows=int(test.sum()),
                alpha=alpha,
                hyperparameters=f'{{"folds": {OUR_FOLDS}, "grouped_by": "subject", "criterion": "mae"}}',
            )
        )
    return runs


def main() -> None:
    index = load_index()
    print("Stage 4\n")

    print("Reproduction, PaPaGei's protocol, 455 test triples per checkpoint")
    verdicts = {}
    for backbone in ("p", "s"):  # P first: it is the primary gate
        runs = reproduction(index, backbone)
        append(runs)
        values = np.array([r.mae for r in runs])
        published = PUBLISHED[backbone]
        inside = values.min() <= published <= values.max()
        percentile = float((values < published).mean() * 100)
        verdicts[backbone] = inside
        print(f"\n  PaPaGei-{backbone.upper()}  published {published}")
        print(f"    our 455 triples: median {np.median(values):.2f}, "
              f"range {values.min():.2f} to {values.max():.2f}")
        print(f"    central 90%: {np.percentile(values, 5):.2f} to {np.percentile(values, 95):.2f}")
        print(f"    published figure sits at the {percentile:.1f}th percentile of our distribution")
        print(f"    inside our range: {inside}")

    print("\n  registered 2x2 reading")
    if verdicts["p"] and verdicts["s"]:
        print("    both reproduce: everything is fine")
    elif verdicts["p"] and not verdicts["s"]:
        print("    P reproduces, S misses: pipeline exonerated by P, model selection stands for S")
    elif not verdicts["p"] and not verdicts["s"]:
        print("    both miss: pipeline bug, the shared machinery is the only common cause")
    else:
        print("    S reproduces, P misses: not expected, look again")

    print("\nArm A, leave-one-subject-out, both checkpoints and both polarities")
    summary = {}
    for backbone in ("p", "s"):
        for polarity in ("asis", "flipped"):
            runs = arm_a(index, backbone, polarity)
            append(runs)
            values = np.array([r.mae for r in runs])
            summary[(backbone, polarity)] = values
            print(f"  PaPaGei-{backbone.upper():1s} {polarity:8s}  "
                  f"mean per-subject MAE {values.mean():6.2f}  "
                  f"median {np.median(values):6.2f}  "
                  f"worst subject {values.max():6.2f}")

    print("\n  polarity, paired across all fifteen subjects")
    for backbone in ("p", "s"):
        difference = summary[(backbone, "asis")] - summary[(backbone, "flipped")]
        better = int((difference > 0).sum())
        print(f"    PaPaGei-{backbone.upper()}: flipped better on {better}/15 subjects, "
              f"mean change {difference.mean():+.3f} BPM")

    print("\n  identity comparison, P minus S, on each polarity")
    for polarity in ("asis", "flipped"):
        difference = summary[("p", polarity)] - summary[("s", polarity)]
        print(f"    {polarity:8s}: {difference.mean():+.3f} BPM, "
              f"P better on {int((difference < 0).sum())}/15 subjects")


if __name__ == "__main__":
    main()
