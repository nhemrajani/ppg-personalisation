"""Stage 4 figures, regenerated from results/experiments.csv."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.harness import load  # noqa: E402
from src.measure_quality import FIGURES  # noqa: E402
from src.stage4 import PUBLISHED, PUBLISHED_INTERVAL  # noqa: E402


def main() -> None:
    rows = load()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    for ax, backbone in zip(axes[:2], ("p", "s")):
        values = np.array([float(r["mae"]) for r in rows
                           if r["arm"] == "reproduction" and r["backbone"] == backbone])
        ax.hist(values, bins=45, color="#4a6fa5", label=f"our 455 test triples (n={len(values)})")
        lo, hi = PUBLISHED_INTERVAL[backbone]
        ax.axvspan(lo, hi, color="#b23a48", alpha=0.85, label="their published 95% interval")
        ax.axvline(PUBLISHED[backbone], color="#b23a48", lw=2)
        ax.set_title(f"PaPaGei-{backbone.upper()}: published {PUBLISHED[backbone]} BPM "
                     f"sits at the {(values < PUBLISHED[backbone]).mean()*100:.0f}th percentile")
        ax.set_xlabel("MAE (BPM)")
        ax.set_ylabel("test triples")
        ax.legend(fontsize=8)

    ax = axes[2]
    width = 0.35
    for offset, backbone, colour in ((-width/2, "p", "#4a6fa5"), (width/2, "s", "#7b4b2a")):
        by = {p: np.array([float(r["mae"]) for r in rows
                           if r["arm"] == "A" and r["backbone"] == backbone and r["polarity"] == p])
              for p in ("asis", "flipped")}
        subjects = np.arange(len(by["asis"]))
        delta = by["flipped"] - by["asis"]
        ax.bar(subjects + offset, delta, width, color=colour, label=f"PaPaGei-{backbone.upper()}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels([f"S{i+1}" for i in range(15)], fontsize=7)
    ax.set_ylabel("MAE change when flipped (BPM)")
    ax.set_title("Polarity, paired by subject. Above zero means flipping hurt")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_reproduction.png", dpi=150)
    print(f"wrote {FIGURES / 'gate4_reproduction.png'}")

    for backbone in ("p", "s"):
        values = np.array([float(r["mae"]) for r in rows
                           if r["arm"] == "reproduction" and r["backbone"] == backbone])
        lo, hi = PUBLISHED_INTERVAL[backbone]
        spread = np.percentile(values, 95) - np.percentile(values, 5)
        print(f"  PaPaGei-{backbone.upper()}: our central 90% spans {spread:.2f} BPM, "
              f"their published interval spans {hi - lo:.2f} BPM, a factor of {spread / (hi - lo):.0f}")


if __name__ == "__main__":
    main()
