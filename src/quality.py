"""Signal quality measures, replicating PaPaGei's flatline test exactly.

PaPaGei rejects a window when more than 25 per cent of it is flat. Their test
returns only pass or fail, which cannot say how close the data sits to the
threshold, so the flatness fraction is reproduced here directly.

The reproduction follows biobss.sqatools.detect_flatline_segments and
PaPaGei's is_signal_flat_lined sample for sample, including two quirks that
matter:

  * A sample equal to the window's maximum or minimum is never counted as
    flat. A perfectly constant window therefore scores zero, not one.
  * The run-length filter uses duration + 1 while the total uses duration,
    so a run of L flat steps counts as L towards the total but must satisfy
    L + 1 >= min_duration to count at all.

Returning run lengths rather than a single fraction means the minimum
flatline duration, which PaPaGei never publishes, can be swept without
recomputing anything.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

from src.data import WRIST_BVP_HZ

CHANGE_THRESHOLD = 0.01  # PaPaGei's default
SQI_WINDOW_SECONDS = 5  # PaPaGei's signal quality index, section 3


def flat_run_lengths(window: np.ndarray, change_threshold: float = CHANGE_THRESHOLD) -> np.ndarray:
    """Lengths of maximal runs of consecutive flat steps, in samples."""
    x = np.asarray(window, dtype=float).ravel()
    if len(x) < 2:
        return np.empty(0, dtype=int)

    change = np.abs(np.diff(x))
    interior = (x[1:] != x.max()) & (x[1:] != x.min())
    flat = (change <= change_threshold) & interior

    if not flat.any():
        return np.empty(0, dtype=int)

    edges = np.diff(np.concatenate(([0], flat.view(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def flatness_fraction(
    run_lengths: np.ndarray,
    window_samples: int,
    min_duration_samples: int,
) -> float:
    """Proportion of a window that counts as flat, as PaPaGei computes it."""
    if len(run_lengths) == 0:
        return 0.0
    qualifying = run_lengths[run_lengths + 1 >= min_duration_samples]
    return float(qualifying.sum() / window_samples)


def sqi_skewness(window: np.ndarray, fs: int = WRIST_BVP_HZ) -> float:
    """PaPaGei's signal quality index: mean skewness over five-second windows.

    Used as a second opinion on the flatline test. A clean pulse is strongly
    positively skewed because systolic peaks are sharp and tall; noise and
    motion artefact drive skewness towards zero.
    """
    x = np.asarray(window, dtype=float).ravel()
    width = SQI_WINDOW_SECONDS * fs
    if len(x) < width:
        return float(skew(x))
    chunks = [x[i : i + width] for i in range(0, len(x) - width + 1, width)]
    return float(np.mean([skew(c) for c in chunks]))
