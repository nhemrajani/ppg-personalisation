"""Measure signal quality across PPG-DaLiA, and check the zero-rejection result.

PaPaGei's flatline test rejected nothing on this dataset. Pass or fail cannot
say whether that is a property of the data or of the parameters, so this
records the underlying flatness fraction for every window, sweeps the one
parameter PaPaGei never publishes, and reports quality by activity.

Writes results/quality_windows.npz and results/flatness_summary.csv, and
draws the two figures that go with them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data import WRIST_BVP_HZ, available_subjects, load_subject
from src.preprocess import (
    FLAT_THRESHOLD,
    WINDOW_SECONDS,
    activity_per_window,
    filter_signal,
    segment,
)
from src.quality import flat_run_lengths, flatness_fraction, sqi_skewness

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

# PaPaGei takes the minimum flatline duration as an argument and publishes no
# default. Our pipeline uses 2 s; the others test whether the result depends on it.
MIN_DURATIONS_S = (0.5, 1.0, 2.0, 4.0)

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
    5: "driving", 6: "lunch", 7: "walking", 8: "working",
}


def measure() -> dict[str, np.ndarray]:
    """Flatness, quality and activity for every window in the dataset."""
    window_samples = WINDOW_SECONDS * WRIST_BVP_HZ
    columns: dict[str, list] = {
        "subject": [], "window": [], "activity": [], "sqi": [],
        "step_raw": [], "step_filtered": [], "step_zscored": [],
        **{f"flat_filtered_{d}s": [] for d in MIN_DURATIONS_S},
        **{f"flat_raw_{d}s": [] for d in MIN_DURATIONS_S},
        **{f"flat_zscored_{d}s": [] for d in MIN_DURATIONS_S},
    }

    for subject in available_subjects():
        data = load_subject(subject)
        raw = np.asarray(data["signal"]["wrist"]["BVP"]).ravel()
        raw_windows, starts = segment(raw)
        filtered_windows, _ = segment(filter_signal(raw))
        activities = activity_per_window(data["activity"], starts)

        for index, (raw_window, filtered_window) in enumerate(zip(raw_windows, filtered_windows)):
            raw_runs = flat_run_lengths(raw_window)
            filtered_runs = flat_run_lengths(filtered_window)
            sd = filtered_window.std()
            zscored_runs = flat_run_lengths(
                (filtered_window - filtered_window.mean()) / (sd if sd else 1.0)
            )
            columns["subject"].append(int(subject[1:]))
            columns["window"].append(index)
            columns["activity"].append(activities[index])
            columns["sqi"].append(sqi_skewness(filtered_window))
            columns["step_raw"].append(float(np.median(np.abs(np.diff(raw_window)))))
            columns["step_filtered"].append(float(np.median(np.abs(np.diff(filtered_window)))))
            columns["step_zscored"].append(
                float(np.median(np.abs(np.diff((filtered_window - filtered_window.mean()) / (sd if sd else 1.0)))))
            )
            for duration in MIN_DURATIONS_S:
                samples = int(duration * WRIST_BVP_HZ)
                columns[f"flat_filtered_{duration}s"].append(
                    flatness_fraction(filtered_runs, window_samples, samples)
                )
                columns[f"flat_raw_{duration}s"].append(
                    flatness_fraction(raw_runs, window_samples, samples)
                )
                columns[f"flat_zscored_{duration}s"].append(
                    flatness_fraction(zscored_runs, window_samples, samples)
                )
        print(f"  {subject}: {len(raw_windows)} windows measured", flush=True)

    return {k: np.asarray(v) for k, v in columns.items()}


def summarise(measured: dict[str, np.ndarray]) -> list[dict]:
    """Per-subject maxima, which is what decides whether zero rejection is safe."""
    rows = []
    for subject in np.unique(measured["subject"]):
        mask = measured["subject"] == subject
        row = {"subject": f"S{subject}", "windows": int(mask.sum())}
        for duration in MIN_DURATIONS_S:
            flat = measured[f"flat_filtered_{duration}s"][mask]
            row[f"max_flat_{duration}s"] = round(float(flat.max()), 4)
            row[f"rejected_{duration}s"] = int((flat > FLAT_THRESHOLD).sum())
        row["max_flat_raw_2.0s"] = round(float(measured["flat_raw_2.0s"][mask].max()), 4)
        row["median_sqi"] = round(float(np.median(measured["sqi"][mask])), 3)
        rows.append(row)
    return rows


def draw_figures(measured: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    flat = measured["flat_filtered_2.0s"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(flat * 100, bins=50, color="#4a6fa5")
    axes[0].axvline(FLAT_THRESHOLD * 100, color="#b23a48", linestyle="--",
                    label=f"rejection threshold, {FLAT_THRESHOLD:.0%}")
    axes[0].set_xlabel("window flatness (%)")
    axes[0].set_ylabel("windows")
    axes[0].set_title(f"Flatness of all {len(flat):,} windows")
    axes[0].set_xlim(0, 100)
    axes[0].legend()

    order = [a for a in sorted(set(measured["activity"].astype(int))) if a in ACTIVITY_NAMES]
    axes[1].boxplot([measured["sqi"][measured["activity"] == a] for a in order],
                    tick_labels=[ACTIVITY_NAMES[a] for a in order], showfliers=False)
    axes[1].set_ylabel("signal quality index (skewness)")
    axes[1].set_title("Quality by activity")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate2_quality.png", dpi=150)
    plt.close(fig)

    # PaPaGei's measure is identically zero here, so the visual check uses the
    # signal quality index instead: the twenty windows with the lowest skewness.
    worst = np.argsort(measured["sqi"])[:20]
    fig, axes = plt.subplots(4, 5, figsize=(16, 9), sharex=True)
    for ax, idx in zip(axes.ravel(), worst):
        subject = f"S{measured['subject'][idx]}"
        window = int(measured["window"][idx])
        raw = np.asarray(load_subject(subject)["signal"]["wrist"]["BVP"]).ravel()
        start = window * 2 * WRIST_BVP_HZ
        ax.plot(np.arange(WINDOW_SECONDS * WRIST_BVP_HZ) / WRIST_BVP_HZ,
                raw[start : start + WINDOW_SECONDS * WRIST_BVP_HZ], linewidth=0.8)
        activity = ACTIVITY_NAMES.get(int(measured["activity"][idx]), "?")
        ax.set_title(f"{subject} w{window}, SQI {measured['sqi'][idx]:+.2f}, {activity}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("The twenty lowest-quality windows in PPG-DaLiA, raw wrist BVP")
    fig.tight_layout()
    fig.savefig(FIGURES / "gate2_worst_windows.png", dpi=130)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    measured = measure()
    np.savez_compressed(RESULTS / "quality_windows.npz", **measured)

    rows = summarise(measured)
    with open(RESULTS / "flatness_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flat = measured["flat_filtered_2.0s"]
    print(f"\nwindows measured: {len(flat):,}")
    print(f"flattest window, PaPaGei's order : {flat.max():.4%}")
    print(f"flattest window, raw signal      : {measured['flat_raw_2.0s'].max():.4%}")
    print(f"rejection threshold              : {FLAT_THRESHOLD:.0%}")
    print("\nsensitivity to the unpublished minimum flatline duration:")
    for duration in MIN_DURATIONS_S:
        column = measured[f"flat_filtered_{duration}s"]
        print(f"  {duration:>4} s  max flatness {column.max():8.4%}  "
              f"windows rejected {int((column > FLAT_THRESHOLD).sum())}")

    print("\nwhy: the change threshold of 0.01 is absolute, so it must be compared")
    print("to the size of a typical step between consecutive samples.")
    for name in ("raw", "filtered", "zscored"):
        steps = measured[f"step_{name}"]
        print(f"  {name:9s} median step {np.median(steps):10.4f}   "
              f"1st percentile {np.percentile(steps, 1):10.4f}   "
              f"below 0.01: {np.mean(steps < 0.01):.4%}")
    zero = measured["flat_zscored_2.0s"]
    print(f"\nif the z-scored signal were tested, as their own function appears to intend:")
    print(f"  max flatness {zero.max():.2%}, windows rejected {int((zero > FLAT_THRESHOLD).sum())}")

    draw_figures(measured)
    print(f"\nfigures written to {FIGURES}")


if __name__ == "__main__":
    main()
