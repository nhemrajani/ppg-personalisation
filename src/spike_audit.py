"""How often does a single spike dominate a window?

Per-window z-scoring sets the scale from the window's standard deviation. If a
window contains one large artefact, that artefact sets the scale and the real
pulse is compressed towards zero, so the encoder receives something close to a
flat line with a spike where the raw signal held usable data.

Measured as the largest absolute deviation from the median, in units of the
window's interquartile range, which a single outlier cannot inflate.
"""

from __future__ import annotations

import csv

import numpy as np

from src.data import available_subjects, load_subject
from src.measure_quality import RESULTS
from src.preprocess import filter_signal, segment

THRESHOLDS = (10, 20, 50, 100)


def spike_ratio(windows: np.ndarray) -> np.ndarray:
    """Largest deviation from the median, in interquartile ranges."""
    median = np.median(windows, axis=1, keepdims=True)
    q1, q3 = np.percentile(windows, [25, 75], axis=1)
    iqr = np.where((q3 - q1) == 0, np.nan, q3 - q1)
    return np.abs(windows - median).max(axis=1) / iqr


def pulse_fraction_after_zscore(windows: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """What share of the z-scored range the ordinary pulse occupies.

    Small means the spike owns the scale and the pulse has been flattened.
    """
    q1, q3 = np.percentile(windows, [25, 75], axis=1)
    spread = windows.max(axis=1) - windows.min(axis=1)
    return np.where(spread == 0, np.nan, (q3 - q1) / spread)


def main() -> None:
    all_ratios, all_fractions, rows = [], [], []
    for subject in available_subjects():
        raw = np.asarray(load_subject(subject)["signal"]["wrist"]["BVP"]).ravel()
        windows, _ = segment(filter_signal(raw))
        ratios = spike_ratio(windows)
        fractions = pulse_fraction_after_zscore(windows, ratios)
        all_ratios.append(ratios)
        all_fractions.append(fractions)
        rows.append(
            {"subject": subject, "windows": len(windows),
             **{f"over_{t}_iqr": int(np.nansum(ratios > t)) for t in THRESHOLDS}}
        )
        print(f"  {subject}: {rows[-1]}", flush=True)

    ratios = np.concatenate(all_ratios)
    fractions = np.concatenate(all_fractions)
    with open(RESULTS / "spike_audit.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(ratios)
    print(f"\n{total:,} windows")
    print(f"median spike ratio {np.nanmedian(ratios):.1f} IQR, "
          f"95th percentile {np.nanpercentile(ratios, 95):.1f}, max {np.nanmax(ratios):.0f}")
    for t in THRESHOLDS:
        n = int(np.nansum(ratios > t))
        print(f"  beyond {t:3d} IQR: {n:6,d} windows ({n/total:.2%})")
    severe = ratios > 20
    if severe.any():
        print(f"\nin those beyond 20 IQR, the interquartile pulse occupies a median of "
              f"{np.nanmedian(fractions[severe]):.1%} of the window's range")
        print(f"elsewhere it occupies {np.nanmedian(fractions[~severe]):.1%}")


if __name__ == "__main__":
    main()
