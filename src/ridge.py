"""Ridge regression on cached embeddings, fast enough for thousands of fits.

Stage 4 alone fits around a thousand ridges, each over tens of thousands of
512-dimensional rows, and each with a grid search inside it. Done naively that
is hours; done through the Gram matrix it is minutes, because the expensive
part depends only on the training rows and can be shared across every value of
alpha and reused across cross-validation folds.

For a design matrix X and target y, centred, the ridge solution is

    w = (XᵀX + alpha·I)⁻¹ Xᵀy

so XᵀX (512 by 512) and Xᵀy are computed once per training set and every alpha
is a 512-dimensional solve. Cross-validation folds reuse the same quantities:
the Gram of a fold's training portion is the total Gram minus that fold's own.

This matches scikit-learn's Ridge with fit_intercept=True to floating point,
which is verified in the self-check below rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardiser:
    """Feature standardisation, fitted on training rows only.

    PaPaGei's linear probing fits a StandardScaler on train and applies it to
    test. The paper never mentions it; it exists only in their code, and it is
    the likeliest single cause of a reproduction miss.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardiser":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        return cls(mean, np.where(scale == 0, 1.0, scale))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


class GramRidge:
    """Ridge over a fixed training set, solvable at any alpha."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_mean = x.mean(axis=0)
        self.y_mean = float(y.mean())
        centred = x - self.x_mean
        self.gram = centred.T @ centred
        self.xty = centred.T @ (y - self.y_mean)
        self.n = len(x)

    def weights(self, alpha: float) -> np.ndarray:
        size = self.gram.shape[0]
        return np.linalg.solve(self.gram + alpha * np.eye(size, dtype=self.gram.dtype), self.xty)

    def predict(self, x: np.ndarray, alpha: float) -> np.ndarray:
        return (x - self.x_mean) @ self.weights(alpha) + self.y_mean


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if y.std() == 0 or pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def bootstrap_interval(
    y: np.ndarray,
    pred: np.ndarray,
    metric=mae,
    resamples: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval, resampling rows with replacement.

    Rows are windows, which is what PaPaGei's own bootstrap resamples. For
    cross-subject claims the study resamples subjects instead; this function
    is used where the comparison is against their published interval.
    """
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        values[i] = metric(y[idx], pred[idx])
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def kfold_indices(n: int, folds: int) -> list[np.ndarray]:
    """Contiguous k-fold, as scikit-learn's KFold does without shuffling."""
    sizes = np.full(folds, n // folds)
    sizes[: n % folds] += 1
    bounds = np.concatenate(([0], np.cumsum(sizes)))
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(folds)]


def group_fold_indices(groups: np.ndarray, folds: int) -> list[np.ndarray]:
    """Folds that never split a group across the boundary.

    Groups are subjects, so a subject's windows sit wholly inside one fold.
    Groups are dealt to folds in order of size, largest first, which is what
    scikit-learn's GroupKFold does.
    """
    unique, counts = np.unique(groups, return_counts=True)
    order = unique[np.argsort(-counts)]
    load = np.zeros(folds)
    assignment: dict[int, int] = {}
    for group, count in zip(order, np.sort(counts)[::-1]):
        target = int(np.argmin(load))
        assignment[int(group)] = target
        load[target] += count
    return [np.flatnonzero([assignment[int(g)] == f for g in groups]) for f in range(folds)]


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    fold_indices: list[np.ndarray],
    criterion: str = "mse",
) -> float:
    """Choose alpha by cross-validation, scoring on MSE or MAE.

    PaPaGei score on negative mean squared error while reporting MAE, so the
    reproduction passes criterion="mse"; our own arms use "mae", matching the
    metric the study reports.
    """
    score = np.zeros(len(alphas))
    for fold in fold_indices:
        mask = np.ones(len(x), dtype=bool)
        mask[fold] = False
        model = GramRidge(x[mask], y[mask])
        for i, alpha in enumerate(alphas):
            pred = model.predict(x[fold], alpha)
            score[i] += rmse(y[fold], pred) ** 2 if criterion == "mse" else mae(y[fold], pred)
    return float(alphas[int(np.argmin(score))])


def _self_check() -> None:
    """Confirm the Gram solution matches scikit-learn before it is relied on."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 64))
    y = x @ rng.normal(size=64) + rng.normal(scale=0.5, size=2000) + 7.0

    for alpha in (0.1, 1.0, 10.0, 100.0):
        ours = GramRidge(x, y).predict(x, alpha)
        theirs = Ridge(alpha=alpha, fit_intercept=True).fit(x, y).predict(x)
        difference = np.abs(ours - theirs).max()
        assert difference < 1e-8, f"alpha {alpha}: differs from scikit-learn by {difference:.2e}"
    print("GramRidge matches scikit-learn Ridge to under 1e-8 at every alpha in their grid")

    groups = np.repeat(np.arange(14), 100)
    folds = group_fold_indices(groups, 7)
    assert sum(len(f) for f in folds) == len(groups), "folds do not cover the data"
    assert all(len(np.unique(groups[f])) == 2 for f in folds), "expected two subjects per fold"
    print("GroupKFold over fourteen subjects gives seven folds of two, no subject split")


if __name__ == "__main__":
    _self_check()
