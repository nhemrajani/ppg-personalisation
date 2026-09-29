"""Verify PaPaGei's analysis window length for PPG-DaLiA.

PaPaGei's paper states two window lengths. Section 4.1 preprocesses signals
"across all datasets" into 10-second windows, but that section covers
pre-training and its Table 1 lists only the three pre-training corpora. The
PPG-DaLiA appendix separately specifies "a 8s window with 6s and 2s overlap and
shift", followed by resampling and padding.

PPG-DaLiA supplies one heart-rate label per analysis window, so counting labels
settles which window PaPaGei used: their reported sample count for the dataset
should equal the number of windows the label grid implies.

Reads from data/ when the dataset has been extracted, otherwise straight from
the downloaded archive.
"""

from __future__ import annotations

import math
import pickle
import zipfile
from pathlib import Path

import numpy as np

from src.data import DATA_DIR, available_subjects, load_subject

WRIST_PPG_HZ = 64
SHIFT_SECONDS = 2
PAPAGEI_REPORTED_WINDOWS = 64697
ARCHIVE = Path.home() / "Desktop" / "ppg+dalia" / "data.zip"


def windows_expected(duration_s: float, window_s: float) -> int:
    """Windows of the given length at a 2-second shift, as PaPaGei specifies."""
    return math.floor((duration_s - window_s) / SHIFT_SECONDS) + 1


def _subjects_from_archive():
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(
            (n for n in archive.namelist() if n.endswith(".pkl")),
            key=lambda n: int(n.split("/S")[1].split("/")[0]),
        )
        for name in names:
            with archive.open(name) as handle:
                yield name.split("/")[-1][:-4], pickle.load(handle, encoding="latin1")


def _subjects_from_disk():
    for subject in available_subjects():
        yield subject, load_subject(subject)


def main() -> None:
    source = _subjects_from_disk if available_subjects() else _subjects_from_archive
    where = DATA_DIR if available_subjects() else ARCHIVE
    print(f"Reading from {where}\n")

    totals = {"labels": 0, 8: 0, 10: 0}
    print(f"{'subj':5s} {'minutes':>8s} {'labels':>8s} {'8 s':>8s} {'10 s':>8s}")
    for subject, data in source():
        labels = np.asarray(data["label"]).ravel()
        duration = len(np.asarray(data["signal"]["wrist"]["BVP"]).ravel()) / WRIST_PPG_HZ
        counts = {w: windows_expected(duration, w) for w in (8, 10)}
        totals["labels"] += len(labels)
        for w in (8, 10):
            totals[w] += counts[w]
        print(f"{subject:5s} {duration/60:8.1f} {len(labels):8d} {counts[8]:8d} {counts[10]:8d}")

    print(f"\n{'total':5s} {'':8s} {totals['labels']:8d} {totals[8]:8d} {totals[10]:8d}")
    print(f"PaPaGei reports {PAPAGEI_REPORTED_WINDOWS} windows for PPG-DaLiA.")

    matches = [w for w in (8, 10) if totals[w] == PAPAGEI_REPORTED_WINDOWS]
    assert totals["labels"] == PAPAGEI_REPORTED_WINDOWS, (
        f"label total {totals['labels']} does not match PaPaGei's {PAPAGEI_REPORTED_WINDOWS}"
    )
    assert matches == [8], f"expected only the 8 s window to match, got {matches}"
    print("\nThe label grid and PaPaGei's count both give 8-second windows.")


if __name__ == "__main__":
    main()
