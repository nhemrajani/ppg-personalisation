"""Figures describing PPG-DaLiA itself, before any model touches it.

Six figures, regenerated from the cached window index and the two inventory
tables. Nothing here depends on an experiment result, so these stay valid
whatever happens to the arms.

    python -m src.describe_dataset     # writes the two inventory tables
    python -m src.figures_dataset      # writes figures/dataset_*.png
"""

from __future__ import annotations

import csv
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.measure_quality import ACTIVITY_NAMES, FIGURES, RESULTS  # noqa: E402
from src.splits import load_index  # noqa: E402

# Activity colours, ordered by the median heart rate each one carries, so the
# timeline reads as an intensity map rather than an arbitrary categorical.
COLOURS = {
    "sitting": "#2c5f8a", "working": "#4a89b8", "lunch": "#7fb2d4",
    "driving": "#9ec6a0", "table soccer": "#d4b96a", "transient": "#c9c9c9",
    "walking": "#e09351", "stairs": "#c4603d", "cycling": "#8c2f27",
}
SUBJECT_ORDER = [f"S{i}" for i in range(1, 16)]


def read(name: str) -> list[dict]:
    with open(RESULTS / name) as handle:
        return list(csv.DictReader(handle))


def label(ax, title: str, note: str = "", width: int = 95) -> None:
    """Title above, wrapped caption beneath it, both left aligned and clear of the axes."""
    lines = textwrap.wrap(note, width) if note else []
    ax.set_title(title, fontsize=11, loc="left", pad=12 + 12 * len(lines))
    if lines:
        ax.annotate("\n".join(lines), xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8.5, color="#555", va="bottom", linespacing=1.45)


def fig_timeline(acts: list[dict]) -> None:
    """The protocol as it was actually recorded, one row per subject."""
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for row in acts:
        y = 15 - int(row["subject"][1:])
        t0, dur = float(row["start_s"]) / 60, float(row["duration_s"]) / 60
        ax.barh(y, dur, left=t0, height=0.72,
                color=COLOURS.get(row["activity"], "#999"), edgecolor="none")
    ax.set_yticks(range(15))
    ax.set_yticklabels(SUBJECT_ORDER[::-1], fontsize=9)
    ax.set_xlabel("minutes from the start of the recording")
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.6, 14.6)
    ax.annotate("S6 stops at 87 minutes, losing lunch, walking and working",
                xy=(88, 15 - 6), xytext=(96, 15 - 6 - 0.52), fontsize=9, color="#b23a48",
                va="center", arrowprops=dict(arrowstyle="->", color="#b23a48", lw=1.2))
    order = ["sitting", "working", "lunch", "driving", "table soccer",
             "transient", "walking", "stairs", "cycling"]
    ax.legend(handles=[Patch(facecolor=COLOURS[a], label=a) for a in order],
              ncol=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=8.5)
    label(ax, "Every subject runs the same activities in the same order",
          "Colour runs cool to hot with the heart rate each activity carries, so the palette "
          "reads as intensity while the axis reads as time. Grey is the transient periods "
          "between activities: 27 per cent of all recorded time, more than any named "
          "activity except lunch. The fixed order is why a naive temporal split would "
          "confound personalisation with activity transfer.", 150)
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_protocol_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_subject_hr(index: dict) -> None:
    """Who has a fast heart and who has a slow one. This drives everything later."""
    hr, subj = index["heart_rate"], index["subject"]
    per = {s: hr[subj == s] for s in np.unique(subj)}
    order = sorted(per, key=lambda s: np.median(per[s]))
    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot([per[s] for s in order], positions=range(len(order)),
                          vert=False, widths=0.85, showextrema=False, showmedians=True)
    for body, s in zip(parts["bodies"], order):
        extreme = np.median(per[s]) > 115
        body.set_facecolor("#b23a48" if extreme else "#4a6fa5")
        body.set_alpha(0.75)
    parts["cmedians"].set_color("#222")
    ax.axvline(np.median(hr), color="#222", ls="--", lw=1.2,
               label=f"population median, {np.median(hr):.0f} BPM")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"S{s}" for s in order], fontsize=9)
    ax.set_xlabel("heart rate (BPM)")
    ax.legend(fontsize=9, loc="lower right")
    label(ax, "Two people sit almost entirely outside everyone else's range",
          "Each shape is one subject's whole recording. S5 and S6, in red, have median "
          "heart rates above 120 BPM where the population median is 85. They are the two "
          "the shared model serves worst, and the two personalisation helps most.")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_subject_heart_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_activity_hr(index: dict) -> None:
    """Heart rate by activity, which is why a naive temporal split is a confound."""
    hr, act = index["heart_rate"], index["activity"].astype(int)
    names = [ACTIVITY_NAMES[c] for c in sorted(ACTIVITY_NAMES)]
    data = {n: hr[act == c] for c, n in zip(sorted(ACTIVITY_NAMES), names)}
    order = sorted(data, key=lambda n: np.median(data[n]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    box = ax.boxplot([data[n] for n in order], positions=range(len(order)),
                     widths=0.62, patch_artist=True, showfliers=False,
                     medianprops=dict(color="#222", lw=1.4))
    for patch, n in zip(box["boxes"], order):
        patch.set_facecolor(COLOURS[n]); patch.set_edgecolor("none")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("heart rate (BPM)")
    label(ax, "Activity moves the median heart rate by 61 BPM, from 59 sitting to 120 cycling",
          "All fifteen subjects pooled. Boxes are quartiles, whiskers 1.5 IQR.")

    minutes = {n: len(v) * 2 / 60 for n, v in data.items()}
    ax2.barh(range(len(order)), [minutes[n] for n in order],
             color=[COLOURS[n] for n in order], height=0.68)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=9)
    ax2.set_xlabel("minutes recorded, all subjects")
    label(ax2, "and the dataset is unbalanced across them",
          "Transient periods alone outweigh every named activity.")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_activity_heart_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cohort(rows: list[dict]) -> None:
    """Who the fifteen people are, and which ranges the dataset does not cover."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    age = [int(r["age"]) for r in rows]
    sex = [r["sex"] for r in rows]

    ax = axes[0]
    for s, colour in (("F", "#b23a48"), ("M", "#4a6fa5")):
        vals = [a for a, g in zip(age, sex) if g == s]
        ax.scatter(vals, np.random.default_rng(0).normal(0, 0.06, len(vals)),
                   s=70, color=colour, alpha=0.85, label=f"{s} (n={len(vals)})")
    ax.set_yticks([]); ax.set_xlabel("age (years)"); ax.legend(fontsize=8.5)
    ax.set_ylim(-0.4, 0.4)
    label(ax, "Age 21 to 55, median 28", "Eight women, seven men. Only three are over 35.", 46)

    ax = axes[1]
    counts = [sum(1 for r in rows if int(r["fitzpatrick_skin_type"]) == t) for t in range(1, 7)]
    bars = ax.bar(range(1, 7), counts,
                  color=["#4a6fa5" if c else "#e8e8e8" for c in counts], edgecolor="none")
    for t, c in zip(range(1, 7), counts):
        if not c:
            ax.text(t, 0.3, "absent", rotation=90, ha="center", fontsize=8, color="#b23a48")
    ax.set_xlabel("Fitzpatrick skin type"); ax.set_ylabel("subjects")
    label(ax, "Skin type 2 to 4 only",
          "Types 1, 5 and 6 are absent, and 11 of 15 are type 3, so the study cannot "
          "speak to personalisation across skin tone.", 46)

    ax = axes[2]
    fit = [sum(1 for r in rows if int(r["fitness_level"]) == f) for f in range(1, 7)]
    ax.bar(range(1, 7), fit, color="#4a6fa5", edgecolor="none")
    ax.set_xlabel("self-reported exercise, 1 rarely to 6 almost daily")
    ax.set_ylabel("subjects")
    label(ax, "Mostly very active",
          "Nine of fifteen report exercising 3 to 5 times a week.", 46)

    ax = axes[3]
    dur = sorted((float(r["duration_min"]), r["subject"]) for r in rows)
    ax.barh(range(len(dur)), [d for d, _ in dur],
            color=["#b23a48" if d < 100 else "#4a6fa5" for d, _ in dur], height=0.7)
    ax.set_yticks(range(len(dur)))
    ax.set_yticklabels([s for _, s in dur], fontsize=8)
    ax.set_xlabel("recording length (minutes)")
    label(ax, "Two hours each, except S6",
          "S6 is 87 minutes and is the binding case for the 40-minute budget, with "
          "48 seconds to spare.", 46)
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_cohort.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_channels(rows: list[dict], inv: list[dict]) -> None:
    """What the dataset ships against what carries any information."""
    seconds = sum(float(r["duration_min"]) for r in rows) * 60
    channels = [
        ("chest ECG", 700, 8, True), ("chest ACC (3 axis)", 700, 24, True),
        ("chest respiration", 700, 8, True), ("chest EMG", 700, 8, False),
        ("chest EDA", 700, 8, False), ("chest temperature", 700, 4, False),
        ("wrist BVP (the PPG)", 64, 8, True), ("wrist ACC (3 axis)", 32, 24, True),
        ("wrist EDA", 4, 8, True), ("wrist temperature", 4, 8, True),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    gb = [seconds * hz * b / 1e9 for _, hz, b, _ in channels]
    live = [l for *_, l in channels]
    ax.barh(range(len(channels)), gb,
            color=["#4a6fa5" if l else "#b23a48" for l in live], height=0.7)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([n for n, *_ in channels], fontsize=9)
    ax.set_xlabel("gigabytes across all fifteen subjects")
    for i, (g, l) in enumerate(zip(gb, live)):
        ax.text(g + 0.04, i, f"{g:.2f}" + ("" if l else "  flatlined"),
                va="center", fontsize=8, color="#222" if l else "#b23a48")
    ax.set_xlim(0, max(gb) * 1.45)
    label(ax, "Three of the ten channels are flatlined in every subject",
          "EMG is a constant, chest EDA is all zeros and chest temperature is a constant "
          "-273.15, which is the sensor reporting absolute zero. Bars are the size of the "
          "arrays themselves.", 100)

    wasted = sum(g for g, l in zip(gb, live) if not l)
    total = sum(gb)
    ax2.pie([total - wasted, wasted], labels=["carries signal", "carries nothing"],
            colors=["#4a6fa5", "#b23a48"], autopct="%1.0f%%",
            textprops=dict(fontsize=10), startangle=90,
            wedgeprops=dict(edgecolor="white", lw=2))
    used = seconds * 64 * 8 / 1e9
    label(ax2, f"and this study reads {used * 1000:.0f} MB of the {total:.1f} GB",
          f"One channel, the wrist PPG, is the entire model input; everything else is "
          f"context or ground truth. The {total:.1f} GB of arrays occupies 20.3 GB on "
          f"disk because the files are protocol 0 pickles, the Python 2 ASCII format, "
          f"which escapes every byte as text. That is also why they need latin1 to "
          f"load at all.", 78)
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_channels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_signal(inv: list[dict]) -> None:
    """Wrist signal scale and sign, which the polarity finding rests on."""
    subs = [r["subject"] for r in inv]
    skew = [float(r["bvp_skew"]) for r in inv]
    sd = [float(r["bvp_sd"]) for r in inv]
    peak = [float(r["bvp_abs_max"]) for r in inv]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    o = np.argsort(skew)
    ax.barh(range(len(subs)), [skew[i] for i in o],
            color=["#4a6fa5" if skew[i] < -0.05 else "#b23a48" for i in o], height=0.7)
    ax.axvline(0, color="#222", lw=1)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([subs[i] for i in o], fontsize=9)
    ax.set_xlabel("skewness of the raw wrist signal")
    clear = sum(1 for v in skew if v < -0.05)
    label(ax, f"The wrist signal is negatively skewed on {clear} of 15 subjects",
          "Conventional PPG has upward systolic peaks and positive skew. This channel is "
          "inverted, which matters because the model's own quality index is skewness, so "
          "its sign is flipped here. S11 sits at zero and only S9 is positive, by 0.02.",
          96)

    ratio = [p / s for p, s in zip(peak, sd)]
    ax2.scatter(sd, ratio, s=80, color="#4a6fa5", alpha=0.85)
    for s, x, y in zip(subs, sd, ratio):
        ax2.annotate(s, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax2.set_xlabel("signal standard deviation (device units)")
    ax2.set_ylabel("largest absolute sample, in standard deviations")
    label(ax2, "The largest artefact is 10 to 19 times the signal's own spread",
          "Per-window normalisation takes its scale from the window, so a single spike "
          "can crush the real pulse. It does so severely on 0.11 per cent of windows "
          "and noticeably on 1.1 per cent.")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_signal_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    index = load_index()
    rows = [r for r in read("subject_summary.csv") if r["subject"].startswith("S")]
    inv = read("dataset_inventory.csv")
    acts = read("dataset_activities.csv")
    fig_timeline(acts)
    fig_subject_hr(index)
    fig_activity_hr(index)
    fig_cohort(rows)
    fig_channels(rows, inv)
    fig_signal(inv)
    for name in sorted(p.name for p in FIGURES.glob("dataset_*.png")):
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
