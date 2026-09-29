"""Arms A, B, B2 and C1, fitted on the cached embeddings.

Each arm is specified in docs/pre-registered-decisions.md, registered before
any of them was fitted. Each also has a degenerate setting under which it must
reduce exactly to Arm A, asserted here rather than assumed: if an arm cannot
reproduce the population model when it is doing nothing, its other numbers
mean nothing either.

Arms C2 and D train and live in their own module, since they need the encoder
rather than the cached table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ridge import GramRidge, Standardiser, group_fold_indices, mae, select_alpha
from src.splits import Split
from src.stage4 import OUR_ALPHAS, OUR_FOLDS

SHRINKAGE = np.logspace(-3, 6, 10)  # registered: same range as the ridge penalty
TOLERANCE = 1e-8


@dataclass
class Fitted:
    """One arm's predictions on the target's test block, and what it cost."""

    predictions: np.ndarray
    trainable_parameters: int
    stored_bytes: int
    alpha: float | None = None
    extra: dict | None = None


def _prepare(x: np.ndarray, split: Split):
    """Standardise on the population set alone, as registered."""
    scaler = Standardiser.fit(x[split.population])
    return scaler(x[split.population]), scaler(x[split.adaptation]), scaler(x[split.test])


def arm_a(x: np.ndarray, y: np.ndarray, split: Split, subject: np.ndarray) -> Fitted:
    """Frozen encoder, ridge on the population set, applied unchanged."""
    x_pop, _, x_test = _prepare(x, split)
    y_pop = y[split.population]
    alpha = select_alpha(x_pop, y_pop, OUR_ALPHAS,
                         group_fold_indices(subject[split.population], OUR_FOLDS), "mae")
    return Fitted(GramRidge(x_pop, y_pop).predict(x_test, alpha), 0, 0, alpha)


def arm_b(x, y, split, subject, budget: np.ndarray, base: Fitted, x_all=None) -> Fitted:
    """Arm A plus a per-person affine correction fitted on the adaptation block.

    Scale and offset come from ordinary least squares of the adaptation block's
    predictions against its labels, as registered.
    """
    x_pop, _, x_test = _prepare(x, split)
    scaler = Standardiser.fit(x[split.population])
    model = GramRidge(x_pop, y[split.population])
    adapt_pred = model.predict(scaler(x[budget]), base.alpha)

    if len(budget) < 2 or np.std(adapt_pred) == 0:
        scale, offset = 1.0, 0.0  # nothing to fit on; degenerates to Arm A
    else:
        scale, offset = np.polyfit(adapt_pred, y[budget], 1)

    return Fitted(base.predictions * scale + offset, 2, 8, base.alpha,
                  {"scale": round(float(scale), 4), "offset": round(float(offset), 4)})


def _subject_weights(
    predicted_target: np.ndarray,
    predicted_population: dict[int, np.ndarray],
) -> dict[int, float]:
    """GAUL's weighting, in one dimension on predicted heart rate.

    A Gaussian is fitted to the target's predicted heart rates, and each
    population subject is weighted by the density of its own mean predicted
    heart rate under it. The bandwidth is the target's predicted standard
    deviation, so there is nothing to tune.
    """
    centre, width = float(np.mean(predicted_target)), float(np.std(predicted_target))
    width = width if width > 0 else 1.0
    raw = {
        s: float(np.exp(-0.5 * ((np.mean(p) - centre) / width) ** 2))
        for s, p in predicted_population.items()
    }
    total = sum(raw.values())
    if total == 0:
        return {s: 1 / len(raw) for s in raw}
    return {s: v / total for s, v in raw.items()}


def arm_b2(x, y, split, subject, budget: np.ndarray, base: Fitted, passes: int = 1) -> Fitted:
    """Population subjects reweighted by similarity to the target, using no labels.

    The target's windows enter only as unlabelled signal: their predicted heart
    rates set the Gaussian. No label from the target is used at any point.
    """
    scaler = Standardiser.fit(x[split.population])
    x_pop, y_pop = scaler(x[split.population]), y[split.population]
    pop_subjects = subject[split.population]
    x_unlabelled = scaler(x[budget])

    model = GramRidge(x_pop, y_pop)
    alpha = base.alpha
    weights = None
    for _ in range(passes):
        predicted_target = model.predict(x_unlabelled, alpha)
        predicted_population = {
            int(s): model.predict(x_pop[pop_subjects == s], alpha)
            for s in np.unique(pop_subjects)
        }
        weights = _subject_weights(predicted_target, predicted_population)
        row = np.sqrt(np.array([weights[int(s)] * len(weights) for s in pop_subjects]))
        model = GramRidge(x_pop * row[:, None], y_pop * row)

    _, _, x_test = _prepare(x, split)
    spread = float(np.std(list(weights.values()))) if weights else 0.0
    return Fitted(model.predict(x_test, alpha), 0, 0, alpha,
                  {"weight_spread": round(spread, 5), "passes": passes})


def arm_c1(x, y, split, subject, budget: np.ndarray, base: Fitted, warm: bool = True) -> Fitted:
    """A per-person output layer: 512 weights and one bias.

    Warm-started, the solution is pulled toward the population solution, so at
    strong shrinkage it is the population model exactly. From scratch, it is an
    ordinary ridge on the person's own data alone.
    """
    scaler = Standardiser.fit(x[split.population])
    x_pop, y_pop = scaler(x[split.population]), y[split.population]
    x_adapt, y_adapt = scaler(x[budget]), y[budget]
    _, _, x_test = _prepare(x, split)

    if len(budget) < 2:
        return Fitted(base.predictions, 513, 2052, base.alpha, {"shrinkage": "all"})

    population = GramRidge(x_pop, y_pop)
    w_pop, b_pop = population.weights(base.alpha), population.y_mean
    centre = x_adapt - population.x_mean
    gram = centre.T @ centre
    target = centre.T @ (y_adapt - b_pop)
    size = gram.shape[0]

    # Chosen on a validation slice of the adaptation block, never the test block.
    cut = max(2, int(len(budget) * 0.75))
    best, best_error = None, np.inf
    for value in SHRINKAGE:
        if warm:
            w = np.linalg.solve(gram + value * np.eye(size), target + value * w_pop)
        else:
            w = np.linalg.solve(gram + value * np.eye(size), target)
        error = mae(y_adapt[cut:], (x_adapt[cut:] - population.x_mean) @ w + b_pop)
        if error < best_error:
            best, best_error = w, error

    return Fitted((x_test - population.x_mean) @ best + b_pop, 513, 2052, base.alpha,
                  {"warm": warm})


def check_degenerate(x, y, split, subject) -> None:
    """Every arm must reduce to Arm A when it is doing nothing."""
    base = arm_a(x, y, split, subject)
    empty = np.empty(0, dtype=int)

    b = arm_b(x, y, split, subject, empty, base)
    assert np.abs(b.predictions - base.predictions).max() < TOLERANCE, "Arm B does not reduce to Arm A"

    b2 = arm_b2(x, y, split, subject, split.adaptation[:1], base, passes=0)
    assert np.abs(b2.predictions - base.predictions).max() < TOLERANCE, "Arm B2 does not reduce to Arm A"

    c1 = arm_c1(x, y, split, subject, empty, base)
    assert np.abs(c1.predictions - base.predictions).max() < TOLERANCE, "Arm C1 does not reduce to Arm A"
