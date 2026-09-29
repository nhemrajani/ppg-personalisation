"""Stage 3: the splitting protocol.

This is the part of the design most likely to be wrong, and an error here
invalidates every number in the study, so it lives in one module that does
nothing else and checks itself.

For each target subject in turn:

  population set    every window from the other fourteen subjects
  adaptation block  a contiguous early portion of the target's recording
  buffer            discarded entirely, at least one window long
  test block        a later contiguous portion, where every arm is evaluated

Two protocols. The primary one is activity-stratified: within each contiguous
run of a single activity, an initial portion goes to adaptation and the
remainder to test. The secondary is a single naive temporal cut, reported as a
realism check. PPG-DaLiA runs its activities in a fixed order, so a single cut
puts different activities on either side and confounds adaptation with
activity transfer.

Transient periods between activities carry code 0 and are 27.1 per cent of all
windows, the second largest category. They are treated as a stratum like any
other rather than discarded.

Why the buffer is needed: windows are 8 s long on a 2 s grid, so consecutive
windows share 6 s. Adjacent windows either side of a cut would be near
duplicates. The buffer is specified in seconds of discarded recording between
the end of the last adaptation window and the start of the first test window,
which also covers the bandpass filter's reach, measured at under 0.1 per cent
of peak beyond 3.8 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocess import SHIFT_SECONDS, WINDOW_SECONDS

EMBEDDING_INDEX = Path(__file__).resolve().parent.parent / "results" / "embeddings" / "index.npz"

BUFFER_SECONDS = WINDOW_SECONDS  # one window, as the methodology specifies
ADAPTATION_FRACTION = 0.5


@dataclass(frozen=True)
class Split:
    """Row indices into the embedding table, for one target subject."""

    target: int
    protocol: str
    population: np.ndarray
    adaptation: np.ndarray
    buffer: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Split(S{self.target}, {self.protocol}, population={len(self.population):,}, "
            f"adaptation={len(self.adaptation):,}, buffer={len(self.buffer):,}, "
            f"test={len(self.test):,})"
        )


def load_index() -> dict[str, np.ndarray]:
    if not EMBEDDING_INDEX.exists():
        raise FileNotFoundError(f"{EMBEDDING_INDEX} not found. Run `python -m src.embed` first.")
    with np.load(EMBEDDING_INDEX) as data:
        return {k: data[k] for k in data.files}


def _runs(order: np.ndarray, activity: np.ndarray, protocol: str) -> list[np.ndarray]:
    """Contiguous stretches to split independently, in time order."""
    if protocol == "temporal":
        return [order]
    codes = activity[order]
    boundaries = np.flatnonzero(np.diff(codes) != 0) + 1
    return [r for r in np.split(order, boundaries) if len(r)]


def _regions(
    order: np.ndarray,
    start_second: np.ndarray,
    activity: np.ndarray,
    protocol: str,
    fraction: float,
    buffer_seconds: float,
) -> list[tuple[float, float, str]]:
    """Time intervals labelled adaptation or test, separated by the buffer.

    Regions are built per run, then the buffer is enforced again at every
    boundary between differently labelled regions. That second step matters:
    activities alternate with transient periods, and a window straddles the
    boundary between two runs, so the last test window of one run would
    otherwise overlap the first adaptation window of the next.
    """
    regions: list[tuple[float, float, str]] = []
    for run in _runs(order, activity, protocol):
        starts = start_second[run]
        span_start, span_end = starts.min(), starts.max() + WINDOW_SECONDS
        cut = span_start + (span_end - span_start) * fraction
        regions.append((span_start, cut, "adaptation"))
        regions.append((cut + buffer_seconds, span_end, "test"))

    regions.sort(key=lambda r: r[0])
    enforced: list[tuple[float, float, str]] = []
    for lo, hi, label in regions:
        if enforced and enforced[-1][2] != label:
            lo = max(lo, enforced[-1][1] + buffer_seconds)
        enforced.append((lo, hi, label))
    return [r for r in enforced if r[1] > r[0]]


def _assign(
    order: np.ndarray,
    start_second: np.ndarray,
    regions: list[tuple[float, float, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A window joins a block only if it lies wholly inside one region."""
    starts = start_second[order]
    ends = starts + WINDOW_SECONDS
    labels = np.array(["buffer"] * len(order), dtype=object)
    for lo, hi, label in regions:
        inside = (starts >= lo) & (ends <= hi)
        labels[inside] = label
    return (
        order[labels == "adaptation"],
        order[labels == "buffer"],
        order[labels == "test"],
    )


def make_split(
    index: dict[str, np.ndarray],
    target: int,
    protocol: str = "activity",
    fraction: float = ADAPTATION_FRACTION,
    buffer_seconds: float = BUFFER_SECONDS,
) -> Split:
    """Build one target subject's split over the cached embedding rows."""
    if protocol not in {"activity", "temporal"}:
        raise ValueError(f"unknown protocol {protocol!r}")

    subject = index["subject"]
    if target not in np.unique(subject):
        raise ValueError(f"no subject S{target} in the index")

    population = np.flatnonzero(subject != target)
    rows = np.flatnonzero(subject == target)
    order = rows[np.argsort(index["start_second"][rows], kind="stable")]
    start_second = index["start_second"]

    regions = _regions(order, start_second, index["activity"], protocol, fraction, buffer_seconds)
    adaptation, buffer, test = _assign(order, start_second, regions)

    return Split(int(target), protocol, population, np.sort(adaptation), np.sort(buffer), np.sort(test))


def budget_prefix(
    split: Split,
    index: dict[str, np.ndarray],
    minutes: float | None,
) -> np.ndarray:
    """The first `minutes` of usable adaptation data, as a prefix in time.

    Budgets are nested: a smaller budget is a strict subset of a larger one.
    Reported in usable minutes after rejection, which on this dataset equals
    elapsed minutes because nothing is rejected. `None` means all available.
    """
    if minutes is None:
        return split.adaptation
    order = split.adaptation[np.argsort(index["start_second"][split.adaptation], kind="stable")]
    keep = int(round(minutes * 60 / SHIFT_SECONDS))
    return np.sort(order[:keep])


def verify(index: dict[str, np.ndarray]) -> None:
    """Assert the properties every split must have, for every subject.

    These are the guarantees the whole study rests on, so they are checked
    rather than assumed.
    """
    subjects = np.unique(index["subject"])
    starts, activity = index["start_second"], index["activity"]

    for protocol in ("activity", "temporal"):
        for target in subjects:
            split = make_split(index, int(target), protocol=protocol)
            blocks = {"adaptation": split.adaptation, "buffer": split.buffer, "test": split.test}

            # The population model must never have seen the target.
            assert target not in index["subject"][split.population], "target leaked into population"
            assert len(split.population) + sum(len(b) for b in blocks.values()) == len(index["subject"])

            # No row may appear in two blocks.
            for a, b in (("adaptation", "buffer"), ("adaptation", "test"), ("buffer", "test")):
                assert not (set(blocks[a].tolist()) & set(blocks[b].tolist())), f"{a} overlaps {b}"

            assert len(split.adaptation) and len(split.test), f"S{target} {protocol}: empty block"

            # No adaptation window may share a sample with any test window.
            adapt_spans = np.stack([starts[split.adaptation], starts[split.adaptation] + WINDOW_SECONDS])
            test_spans = np.stack([starts[split.test], starts[split.test] + WINDOW_SECONDS])
            overlap = (adapt_spans[0][:, None] < test_spans[1][None, :]) & (
                test_spans[0][None, :] < adapt_spans[1][:, None]
            )
            assert not overlap.any(), f"S{target} {protocol}: adaptation and test windows overlap in time"

            if protocol == "temporal":
                assert starts[split.test].min() >= starts[split.adaptation].max() + WINDOW_SECONDS + BUFFER_SECONDS, (
                    f"S{target}: test block is not a full buffer later than adaptation"
                )
            else:
                # Test must follow adaptation within each contiguous run. Not per
                # activity code: transient periods recur throughout the recording,
                # so an early run's test block legitimately precedes a later run's
                # adaptation block.
                rows = np.flatnonzero(index["subject"] == target)
                order = rows[np.argsort(starts[rows], kind="stable")]
                adaptation_set, test_set = set(split.adaptation.tolist()), set(split.test.tolist())
                for run in _runs(order, activity, "activity"):
                    a = [i for i in run if i in adaptation_set]
                    t = [i for i in run if i in test_set]
                    if a and t:
                        assert starts[t].min() >= starts[a].max() + WINDOW_SECONDS, (
                            f"S{target}: within a run of activity "
                            f"{activity[run[0]]:.0f}, test precedes adaptation"
                        )

            # Budgets must nest.
            previous = None
            for minutes in (2, 5, 10, 20, 40):
                current = budget_prefix(split, index, minutes)
                if previous is not None:
                    assert set(previous.tolist()) <= set(current.tolist()), "budgets do not nest"
                previous = current
            assert set(previous.tolist()) <= set(split.adaptation.tolist())


def main() -> None:
    index = load_index()
    verify(index)
    print("all split invariants hold, for both protocols and all fifteen subjects\n")

    for protocol in ("activity", "temporal"):
        print(f"{protocol} protocol")
        print(f"  {'subj':5s} {'adapt':>7s} {'buffer':>7s} {'test':>7s} {'adapt min':>10s} {'test min':>9s}")
        totals = np.zeros(3)
        for target in np.unique(index["subject"]):
            s = make_split(index, int(target), protocol=protocol)
            totals += [len(s.adaptation), len(s.buffer), len(s.test)]
            print(f"  S{target:<4d} {len(s.adaptation):7,d} {len(s.buffer):7,d} {len(s.test):7,d} "
                  f"{len(s.adaptation)*SHIFT_SECONDS/60:10.1f} {len(s.test)*SHIFT_SECONDS/60:9.1f}")
        print(f"  {'mean':5s} {totals[0]/15:7,.0f} {totals[1]/15:7,.0f} {totals[2]/15:7,.0f}\n")

    # The budget sweep has to be reachable for every subject, including S6.
    print("usable adaptation minutes against the budget sweep, activity protocol")
    for target in np.unique(index["subject"]):
        s = make_split(index, int(target))
        available = len(s.adaptation) * SHIFT_SECONDS / 60
        reachable = [m for m in (2, 5, 10, 20, 40) if len(budget_prefix(s, index, m)) * SHIFT_SECONDS / 60 >= m]
        print(f"  S{target:<4d} {available:6.1f} min available, budgets met: {reachable}")


if __name__ == "__main__":
    main()
