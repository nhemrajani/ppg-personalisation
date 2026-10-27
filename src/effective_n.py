"""Effective sample size of a set of windows.

Windows are eight seconds long on a two-second shift, so neighbours share six
of their eight seconds and a count of windows overstates how much independent
signal is present by about a factor of four. This matters for Arm C1, which
fits 513 coefficients: at every budget below all-available it has more
parameters than independent samples, which is the cleanest statement of why it
cannot be fitted without regularisation.

Computed from the union of the windows' time spans rather than approximated,
because activity-stratified budgets are not contiguous.
"""

from __future__ import annotations

import numpy as np

from src.preprocess import WINDOW_SECONDS


def effective_n(start_seconds: np.ndarray) -> int:
    """Independent eight-second spans covered by these windows."""
    if len(start_seconds) == 0:
        return 0
    starts = np.sort(np.asarray(start_seconds, dtype=float))
    covered, end = 0.0, -np.inf
    for s in starts:
        stop = s + WINDOW_SECONDS
        covered += stop - max(s, end) if stop > end else 0.0
        end = max(end, stop)
    return int(round(covered / WINDOW_SECONDS))
