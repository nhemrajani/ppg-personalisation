"""What PPG-DaLiA contains as a file, rather than as a cohort.

`src/dataset_summary.py` already reports the cohort: demographics, durations,
heart-rate ranges and minutes per activity, in results/subject_summary.csv.
This module deliberately does not repeat any of that. It records the three
things that summary does not:

1. **Which channels carry information.** The chest device ships six channels
   and three of them are flatlined in every subject.
2. **The ground truth's own consistency**, by comparing the supplied labels
   against the R-peak count in the ECG they were derived from.
3. **The protocol as a timeline**: one row per contiguous activity run, which
   is what the splitting protocol operates on and what a timeline figure needs.

Two outputs, both regenerable:

    results/dataset_inventory.csv    one row per subject, new columns only
    results/dataset_activities.csv   one row per activity run

Run with ``python -m src.describe_dataset``. It loads one subject at a time and
discards it, because the chest array alone is about 300 MB per subject.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from src.data import DATA_DIR, available_subjects, load_subject
from src.measure_quality import ACTIVITY_NAMES
from src.preprocess import SHIFT_SECONDS as SHIFT_S, WINDOW_SECONDS as WINDOW_S

CHEST_HZ = 700
ACTIVITY_HZ = 4


def read_protocol(subject: str) -> list[tuple[str, int]]:
    """Activity onsets in seconds, from the subject's activity CSV."""
    path = DATA_DIR / "PPG_FieldStudy" / subject / f"{subject}_activity.csv"
    out = []
    for line in path.read_text().splitlines():
        match = re.match(r"#\s*([A-Z_]+),\s*(\d+)", line.strip())
        if match and match.group(1) != "SUBJECT_ID":
            out.append((match.group(1).lower(), int(match.group(2))))
    return out


def dead(channel: np.ndarray) -> bool:
    """A channel that never changes carries no information."""
    return bool(np.nanmin(channel) == np.nanmax(channel))


def runs(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs of one activity code as (code, start index, length)."""
    flat = codes.ravel().astype(int)
    edges = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], edges, [len(flat)]])
    return [(int(flat[a]), int(a), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:])]


def inventory_one(subject: str) -> tuple[dict, list[dict]]:
    d = load_subject(subject)
    chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
    label = np.asarray(d["label"], dtype=float)
    activity = np.asarray(d["activity"])
    seconds = len(chest["ECG"]) / CHEST_HZ

    bvp = np.asarray(wrist["BVP"], dtype=float).ravel()
    row = {
        "subject": subject,
        # Does the ground truth agree with the signal it came from?
        "n_labels": len(label),
        "n_labels_predicted": int((seconds - WINDOW_S) // SHIFT_S + 1),
        "n_rpeaks": int(len(d["rpeaks"])),
        "ecg_mean_hr": round(60 * len(d["rpeaks"]) / seconds, 1),
        "label_mean_hr": round(float(label.mean()), 1),
        # Wrist signal scale and sign, which the polarity finding rests on.
        "bvp_sd": round(float(bvp.std()), 1),
        "bvp_abs_max": round(float(np.abs(bvp).max()), 1),
        "bvp_skew": round(float(((bvp - bvp.mean()) ** 3).mean() / bvp.std() ** 3), 3),
        "label_nans": int(np.isnan(label).sum()),
        "bvp_nans": int(np.isnan(bvp).sum()),
        # Which channels carry information at all.
        "n_activity_runs": len(runs(activity)),
        "dead_chest_channels": "|".join(
            sorted(k for k, v in chest.items() if dead(np.asarray(v, dtype=float)))),
        "live_chest_channels": "|".join(
            sorted(k for k, v in chest.items() if not dead(np.asarray(v, dtype=float)))),
        "protocol_csv_activities": len(read_protocol(subject)),
    }

    # Per-activity rows: duration and heart rate inside each contiguous run.
    label_per_s = len(label) / seconds
    acts = []
    for code, start, length in runs(activity):
        t0, t1 = start / ACTIVITY_HZ, (start + length) / ACTIVITY_HZ
        lo, hi = int(t0 * label_per_s), int(t1 * label_per_s)
        chunk = label[lo:hi]
        acts.append({
            "subject": subject, "code": code, "activity": ACTIVITY_NAMES.get(code, "?"),
            "start_s": round(t0, 1), "duration_s": round(t1 - t0, 1),
            "n_labels": len(chunk),
            "hr_mean": round(float(chunk.mean()), 1) if len(chunk) else "",
            "hr_sd": round(float(chunk.std()), 2) if len(chunk) else "",
            "hr_min": round(float(chunk.min()), 1) if len(chunk) else "",
            "hr_max": round(float(chunk.max()), 1) if len(chunk) else "",
        })
    del d
    return row, acts


def main() -> None:
    rows, acts = [], []
    for subject in available_subjects():
        row, a = inventory_one(subject)
        rows.append(row)
        acts.extend(a)
        print(f"  {subject:4s} {row['n_labels']:5d} labels "
              f"(predicted {row['n_labels_predicted']:5d})  "
              f"ECG {row['ecg_mean_hr']:5.1f} vs label {row['label_mean_hr']:5.1f} BPM  "
              f"dead: {row['dead_chest_channels'] or 'none'}", flush=True)

    out = Path("results")
    out.mkdir(exist_ok=True)
    for name, data in (("dataset_inventory.csv", rows), ("dataset_activities.csv", acts)):
        with open(out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        print(f"wrote results/{name}, {len(data)} rows")


if __name__ == "__main__":
    main()
