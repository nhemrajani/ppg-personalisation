"""Stage 4 figures, regenerated from results/experiments.csv."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.harness import load  # noqa: E402
from src.measure_quality import FIGURES  # noqa: E402
from src.stage4 import PUBLISHED, PUBLISHED_INTERVAL  # noqa: E402


def _ordinal(value: float) -> str:
    n = int(round(value))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
                     f"sits at the {_ordinal((values < PUBLISHED[backbone]).mean()*100)} percentile")
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
        # Compare like with like: their interval is a 95% interval, so ours must be too.
        spread95 = np.percentile(values, 97.5) - np.percentile(values, 2.5)
        spread90 = np.percentile(values, 95) - np.percentile(values, 5)
        print(f"  PaPaGei-{backbone.upper()}: their 95% interval spans {hi - lo:.2f} BPM; "
              f"ours spans {spread95:.2f} at 95% (factor {spread95 / (hi - lo):.0f}) "
              f"and {spread90:.2f} at 90% (factor {spread90 / (hi - lo):.0f})")




def population_curve() -> None:
    """Mean per-subject MAE against the number of population subjects."""
    rows = [r for r in load() if r["arm"] == "A-population-size"]
    if not rows:
        return
    import json

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for backbone, colour in (("p", "#4a6fa5"), ("s", "#7b4b2a")):
        sizes, means, errors = [], [], []
        by_size: dict[int, list[float]] = {}
        for r in rows:
            if r["backbone"] != backbone:
                continue
            n = json.loads(r["hyperparameters"])["population_subjects"]
            by_size.setdefault(n, []).append(float(r["mae"]))
        for n in sorted(by_size):
            sizes.append(n)
            means.append(np.mean(by_size[n]))
            errors.append(np.std(by_size[n]) / np.sqrt(len(by_size[n])))
        ax.errorbar(sizes, means, yerr=errors, marker="o", color=colour,
                    capsize=3, label=f"PaPaGei-{backbone.upper()}")
        ax.axhline(means[-1], color=colour, ls=":", lw=0.8)
    ax.set_xlabel("population subjects used to fit the model")
    ax.set_ylabel("mean per-subject MAE (BPM)")
    ax.set_title("The population model saturates at about six subjects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_population_curve.png", dpi=150)
    print(f"wrote {FIGURES / 'gate4_population_curve.png'}")


if __name__ == "__main__":
    main()
    population_curve()
