"""PaPaGei's preprocessing pipeline, applied to PPG-DaLiA.

PaPaGei's public repository contains the preprocessing functions but not the
step that turns PPG-DaLiA into windows, so that step is assembled here from
their functions in the order their paper gives:

    filter at 64 Hz -> segment -> reject flatlines -> z-score -> resample -> pad

Two parameters are not stated anywhere in the paper or the code, and are set
here as explicit choices rather than buried defaults:

  FLAT_MIN_SECONDS   the shortest run that counts as a flatline. Their code
                     takes it as an argument and publishes no default.
  INPUT_LENGTH       the padded length. The paper says only "resample and pad"
                     for this dataset. 1,250 samples matches the ten-second
                     pre-training window and their own example notebook, which
                     centre pads a shorter recording to that length.

Both are recorded with the output so that a later mismatch against PaPaGei's
published error can be traced to them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data import WRIST_BVP_HZ, load_subject
from src.papagei import INPUT_LENGTH, PAPAGEI_DIR, SAMPLE_RATE_HZ

if str(PAPAGEI_DIR) not in sys.path:
    sys.path.insert(0, str(PAPAGEI_DIR))

from linearprobing.utils import resample_batch_signal  # noqa: E402
from preprocessing.flatline import is_signal_flat_lined  # noqa: E402
from preprocessing.ppg import preprocess_one_ppg_signal  # noqa: E402

WINDOW_SECONDS = 8  # PaPaGei's PPG-DaLiA appendix; confirmed by src.verify_windows
SHIFT_SECONDS = 2
FLAT_THRESHOLD = 0.25  # "more than 25% of the data is flat"
FLAT_MIN_SECONDS = 2  # not published; our choice
ACTIVITY_HZ = 4  # PPG-DaLiA activity labels


@dataclass
class SubjectWindows:
    """One subject's preprocessed windows and everything aligned to them."""

    subject: str
    windows: np.ndarray  # (kept, INPUT_LENGTH) float32, model ready
    heart_rate: np.ndarray  # (kept,) BPM, the regression target
    activity: np.ndarray  # (kept,) activity code
    start_second: np.ndarray  # (kept,) window start, for temporal splitting
    kept: np.ndarray  # (total,) bool, False where rejected as flatline
    parameters: dict = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        return float(1 - self.kept.mean())

    @property
    def usable_minutes(self) -> float:
        """Distinct time covered by surviving windows, not window count times eight."""
        return float(self.kept.sum() * SHIFT_SECONDS / 60)

    @property
    def elapsed_minutes(self) -> float:
        return float(len(self.kept) * SHIFT_SECONDS / 60)


def filter_signal(raw: np.ndarray, fs: int = WRIST_BVP_HZ) -> np.ndarray:
    """PaPaGei's Chebyshev Type II bandpass, applied at the native rate.

    Filtering before resampling is deliberate. pyPPG adds a 50 ms smoothing
    pass at rates of 75 Hz and above, so filtering after the move to 125 Hz
    would silently apply a step that filtering at 64 Hz does not.
    """
    filtered, _, _, _ = preprocess_one_ppg_signal(waveform=np.asarray(raw).ravel(), frequency=fs)
    return filtered


def segment(signal: np.ndarray, fs: int = WRIST_BVP_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Eight-second windows at a two-second shift, and their start times."""
    width, step = WINDOW_SECONDS * fs, SHIFT_SECONDS * fs
    count = (len(signal) - width) // step + 1
    if count <= 0:
        raise ValueError(f"signal of {len(signal)} samples is shorter than one window")
    starts = np.arange(count) * step
    windows = np.stack([signal[s : s + width] for s in starts])
    return windows, starts / fs


def is_flat(window: np.ndarray, fs: int = WRIST_BVP_HZ) -> bool:
    """PaPaGei's flatline test, unmodified."""
    return bool(
        is_signal_flat_lined(
            sig=window,
            fs=fs,
            flat_time=FLAT_MIN_SECONDS,
            signal_time=WINDOW_SECONDS,
            flat_threshold=FLAT_THRESHOLD,
        )
    )


def zscore(windows: np.ndarray) -> np.ndarray:
    """Per window, as step 4. Constant windows would divide by zero."""
    mean = windows.mean(axis=1, keepdims=True)
    sd = windows.std(axis=1, keepdims=True)
    return (windows - mean) / np.where(sd == 0, 1.0, sd)


def resample_and_pad(windows: np.ndarray, fs: int = WRIST_BVP_HZ) -> np.ndarray:
    """To 125 Hz, then centre padded to the encoder's input length."""
    resampled = resample_batch_signal(windows, fs_original=fs, fs_target=SAMPLE_RATE_HZ, axis=-1)
    shortfall = INPUT_LENGTH - resampled.shape[1]
    if shortfall < 0:
        raise ValueError(f"window of {resampled.shape[1]} samples exceeds {INPUT_LENGTH}")
    left = shortfall // 2
    return np.pad(resampled, ((0, 0), (left, shortfall - left))).astype(np.float32)


def activity_per_window(activity: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """The activity a window sits in, taken as the most common code within it."""
    codes = np.asarray(activity).ravel()
    out = np.empty(len(starts))
    for i, start in enumerate(starts):
        lo = int(start * ACTIVITY_HZ)
        hi = min(lo + WINDOW_SECONDS * ACTIVITY_HZ, len(codes))
        span = codes[lo:hi]
        out[i] = np.bincount(span.astype(int)).argmax() if len(span) else -1
    return out


def preprocess_subject(subject: str, root: Path | None = None) -> SubjectWindows:
    """One subject, from raw pickle to model-ready windows."""
    data = load_subject(subject, root)
    raw = np.asarray(data["signal"]["wrist"]["BVP"]).ravel()
    labels = np.asarray(data["label"]).ravel()

    filtered = filter_signal(raw)
    windows, starts = segment(filtered)

    # PPG-DaLiA defines one heart-rate label per window on the same grid, so the
    # counts must agree. If they do not, the grid is wrong and nothing downstream
    # is meaningful.
    if len(windows) != len(labels):
        raise ValueError(
            f"{subject}: {len(windows)} windows but {len(labels)} labels. "
            "The segmentation grid does not match the dataset's own."
        )

    kept = np.array([not is_flat(w) for w in windows])
    prepared = resample_and_pad(zscore(windows[kept]))

    return SubjectWindows(
        subject=subject,
        windows=prepared,
        heart_rate=labels[kept],
        activity=activity_per_window(data["activity"], starts)[kept],
        start_second=starts[kept],
        kept=kept,
        parameters={
            "window_seconds": WINDOW_SECONDS,
            "shift_seconds": SHIFT_SECONDS,
            "flat_threshold": FLAT_THRESHOLD,
            "flat_min_seconds": FLAT_MIN_SECONDS,
            "target_hz": SAMPLE_RATE_HZ,
            "input_length": INPUT_LENGTH,
        },
    )


def check_synthetic(fs: int = WRIST_BVP_HZ, hz: float = 1.2, seconds: int = 30) -> int:
    """Filter and peak detect a signal whose answer is known.

    A 1.2 Hz sinusoid over thirty seconds has 36 cycles. If the filter and the
    peak detector cannot recover that, they cannot be trusted on real data.
    """
    from scipy.signal import find_peaks

    t = np.arange(fs * seconds) / fs
    filtered = filter_signal(np.sin(2 * np.pi * hz * t))
    peaks, _ = find_peaks(filtered, distance=int(fs / (hz * 1.5)))
    return len(peaks)


if __name__ == "__main__":
    expected = 36
    found = check_synthetic()
    print(f"synthetic 1.2 Hz check: {found} peaks, expected {expected}")
    assert abs(found - expected) <= 1, "filter or peak detection is wrong"

    subject = sys.argv[1] if len(sys.argv) > 1 else "S1"
    result = preprocess_subject(subject)
    print(
        f"\n{result.subject}: {len(result.kept)} windows, {result.kept.sum()} kept, "
        f"{result.rejection_rate:.1%} rejected"
    )
    print(f"  shape {result.windows.shape}, dtype {result.windows.dtype}")
    print(f"  heart rate {result.heart_rate.min():.0f} to {result.heart_rate.max():.0f} BPM")
    print(f"  activities present: {sorted(set(result.activity.astype(int)))}")
    print(f"  usable {result.usable_minutes:.1f} min of {result.elapsed_minutes:.1f} min elapsed")
