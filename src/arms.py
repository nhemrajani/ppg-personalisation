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
SHRINKAGE_B = np.logspace(-4, 4, 9)  # registered 27 October 2026 for Arm B-ridge
BUFFER_WINDOWS = 4  # 8 s at a 2 s shift: the overlap between a window and its neighbours
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
        return Fitted(base.predictions, 513, 2052, base.alpha,
                      {"shrinkage": "degenerate", "n_fit": 0, "n_validation": 0})

    population = GramRidge(x_pop, y_pop)
    w_pop, b_pop = population.weights(base.alpha), population.y_mean
    size = x_adapt.shape[1]

    def solve(gram, target, value):
        rhs = target + value * w_pop if warm else target
        return np.linalg.solve(gram + value * np.eye(size), rhs)

    # Selected on a slice held OUT of the fit, never the test block. Scoring on
    # rows that are inside the fit measures in-sample error, which falls as the
    # penalty weakens, so the weakest penalty always wins and the arm runs
    # unregularised. That was the defect of 26 October 2026; see the registration.
    cut = max(2, int(len(budget) * 0.75))
    if cut > len(budget) - 1:
        cut = len(budget) - 1

    # Windows overlap by six of their eight seconds, so the four windows either
    # side of the cut share signal. Discard them, as every other block boundary
    # in this study does, or the held-out slice is not held out.
    fit_rows, validation_rows = slice(0, cut), slice(cut + BUFFER_WINDOWS, None)
    fit_x, fit_y = x_adapt[fit_rows], y_adapt[fit_rows]
    val_x, val_y = x_adapt[validation_rows], y_adapt[validation_rows]

    if len(fit_x) < 2 or len(val_x) < 1:
        selected = SHRINKAGE[-1]  # not enough data to select on; take the strongest
    else:
        centred_fit = fit_x - population.x_mean
        gram_fit = centred_fit.T @ centred_fit
        target_fit = centred_fit.T @ (fit_y - b_pop)
        selected, best_error = None, np.inf
        for value in SHRINKAGE:
            w = solve(gram_fit, target_fit, value)
            error = mae(val_y, (val_x - population.x_mean) @ w + b_pop)
            if error < best_error:
                selected, best_error = value, error

    # Refit on the whole budget at the selected strength.
    centre = x_adapt - population.x_mean
    best = solve(centre.T @ centre, centre.T @ (y_adapt - b_pop), selected)

    return Fitted((x_test - population.x_mean) @ best + b_pop, 513, 2052, base.alpha,
                  {"warm": warm, "shrinkage": float(selected),
                   "shrinkage_index": int(np.argmin(np.abs(SHRINKAGE - selected))),
                   "n_fit": int(len(fit_x)), "n_validation": int(len(val_x))})


def _adaptation_predictions(x, y, split, budget, base: Fitted):
    """The population model's predictions on the budget, and on the test block.

    Shared by Arms B, B-offset and B-ridge: all three correct the same numbers.
    """
    scaler = Standardiser.fit(x[split.population])
    model = GramRidge(scaler(x[split.population]), y[split.population])
    return model.predict(scaler(x[budget]), base.alpha), base.predictions


def arm_b_offset(x, y, split, subject, budget: np.ndarray, base: Fitted) -> Fitted:
    """One parameter: a per-person offset, with the slope fixed at 1.

    Registered 27 October 2026. The proposal's mechanism for per-person error is
    ridge shrinkage toward the training mean, which predicts a bias and not a
    gain, so this is the arm that mechanism actually points at. With no slope
    there is no extrapolation risk: however unrepresentative the calibration
    window, the worst case is an unhelpful constant shift.
    """
    if len(budget) < 1:
        return Fitted(base.predictions, 1, 4, base.alpha, {"offset": 0.0})

    adapt_pred, test_pred = _adaptation_predictions(x, y, split, budget, base)
    offset = float(np.mean(y[budget] - adapt_pred))
    return Fitted(test_pred + offset, 1, 4, base.alpha,
                  {"offset": round(offset, 4)})


def arm_b_ridge(x, y, split, subject, budget: np.ndarray, base: Fitted) -> Fitted:
    """Arm B's two parameters, penalised toward the null correction.

    Minimises squared error plus `lam * n * [sigma^2 (a-1)^2 + b^2]`, so the
    arm nests both of its neighbours: at the strongest shrinkage it is Arm A
    exactly, and at the weakest it is Arm B exactly. The offset's penalty is
    scaled by the prediction variance so both terms carry units of BPM squared
    and one lambda is dimensionless.
    """
    if len(budget) < 2:
        return Fitted(base.predictions, 2, 8, base.alpha,
                      {"scale": 1.0, "offset": 0.0, "shrinkage_b": "degenerate"})

    adapt_pred, test_pred = _adaptation_predictions(x, y, split, budget, base)
    y_adapt = y[budget]

    def solve(pred, target, lam):
        """Penalised least squares for (slope, offset) toward (1, 0)."""
        n = len(pred)
        design = np.column_stack([pred, np.ones(n)])
        variance = float(np.var(pred))
        penalty = np.diag([variance if variance > 0 else 1.0, 1.0]) * lam * n
        left = design.T @ design + penalty
        right = design.T @ target + penalty @ np.array([1.0, 0.0])
        return np.linalg.solve(left, right)

    # Selected on a slice held out of the fit, with the four-window discard at
    # the cut, exactly as Arm C1 does after the fix of 26 October 2026.
    cut = max(2, int(len(budget) * 0.75))
    if cut > len(budget) - 1:
        cut = len(budget) - 1
    fit_p, fit_y = adapt_pred[:cut], y_adapt[:cut]
    val_p, val_y = adapt_pred[cut + BUFFER_WINDOWS:], y_adapt[cut + BUFFER_WINDOWS:]

    if len(fit_p) < 2 or len(val_p) < 1:
        selected = SHRINKAGE_B[-1]
    else:
        selected, best_error = None, np.inf
        for lam in SHRINKAGE_B:
            scale, offset = solve(fit_p, fit_y, lam)
            error = mae(val_y, val_p * scale + offset)
            if error < best_error:
                selected, best_error = lam, error

    scale, offset = solve(adapt_pred, y_adapt, selected)
    return Fitted(test_pred * scale + offset, 2, 8, base.alpha,
                  {"scale": round(float(scale), 4), "offset": round(float(offset), 4),
                   "shrinkage_b": float(selected),
                   "n_fit": int(len(fit_p)), "n_validation": int(len(val_p))})


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

    bo = arm_b_offset(x, y, split, subject, empty, base)
    assert np.abs(bo.predictions - base.predictions).max() < TOLERANCE, "Arm B-offset does not reduce to Arm A"

    # Arm B-ridge nests both neighbours, so it carries two assertions. The real
    # budget is used here rather than an empty one, because the point is that
    # the penalty alone moves it between Arm A and Arm B.
    full = split.adaptation
    adapt_pred, test_pred = _adaptation_predictions(x, y, split, full, base)
    design = np.column_stack([adapt_pred, np.ones(len(adapt_pred))])
    variance = float(np.var(adapt_pred))
    for lam, name, expected in (
        (1e12, "Arm A", base.predictions),
        (0.0, "Arm B", arm_b(x, y, split, subject, full, base).predictions),
    ):
        penalty = np.diag([variance, 1.0]) * lam * len(adapt_pred)
        scale, offset = np.linalg.solve(design.T @ design + penalty,
                                        design.T @ y[full] + penalty @ np.array([1.0, 0.0]))
        got = test_pred * scale + offset
        drift = float(np.abs(got - expected).max())
        assert drift < 1e-6, f"Arm B-ridge at lambda={lam:g} does not equal {name}: off by {drift:.2e}"
