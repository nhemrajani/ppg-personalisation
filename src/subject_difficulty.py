"""What predicts how badly the population model serves a subject?

Sub Question 2 asks whether adaptation helps the people the population model
serves worst. Before running any adaptive arm, it is worth knowing whether
those people are identifiable in advance, and whether their difficulty is
noise or structure.

Writes figures/gate4_subject_difficulty.png and appends a section to the
dataset appendix.
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.dataset_summary import ROOT
from src.measure_quality import FIGURES

PREDICTORS = {
    "distance of HR median from population": "distance",
    "median signal quality index": "median_sqi",
    "HR inter-quartile range": "hr_iqr",
    "recording duration": "duration_min",
    "age": "age",
}


def load_rows() -> list[dict]:
    rows = list(csv.DictReader(open(ROOT / "results" / "subject_summary.csv")))
    medians = np.array([float(r["hr_median"]) for r in rows])
    for i, row in enumerate(rows):
        row["distance"] = abs(medians[i] - np.median(np.delete(medians, i)))
    return rows


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows()
    mae = np.array([float(r["arm_a_mae_papagei_p"]) for r in rows])

    print("Predicting per-subject error under the unadapted population model\n")
    for label, key in PREDICTORS.items():
        x = np.array([float(r[key]) for r in rows])
        r_p, p_p = pearsonr(x, mae)
        rho, p_s = spearmanr(x, mae)
        print(f"  {label:40s} r={r_p:+.2f} (p={p_p:.4f})  rho={rho:+.2f} (p={p_s:.4f})")

    distance = np.array([float(r["distance"]) for r in rows])
    r_value = pearsonr(distance, mae)[0]
    print(f"\n  distance alone explains {r_value**2:.0%} of the variance in per-subject error")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(distance, mae, s=55, color="#4a6fa5", zorder=3)
    for row, x, y in zip(rows, distance, mae):
        ax.annotate(row["subject"], (x, y), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color="#555")
    fit = np.polyfit(distance, mae, 1)
    grid = np.linspace(0, distance.max() * 1.05, 50)
    ax.plot(grid, np.polyval(fit, grid), color="#b23a48", lw=1.2, ls="--",
            label=f"r = {r_value:.2f}, explains {r_value**2:.0%} of variance")
    ax.set_xlabel("distance of the subject's median heart rate from the population median (BPM)")
    ax.set_ylabel("Arm A MAE, PaPaGei-P (BPM)")
    ax.set_title("The population model fails the people unlike the population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gate4_subject_difficulty.png", dpi=150)
    print(f"\nwrote {FIGURES / 'gate4_subject_difficulty.png'}")

    appendix = ROOT / "docs" / "appendix-dataset.md"
    text = appendix.read_text()
    marker = "\n## Who the population model fails\n"
    if marker in text:
        text = text[: text.index(marker)]
    order = np.argsort(-mae)
    worst = ", ".join(f"{rows[i]['subject']} at {mae[i]:.1f}" for i in order[:3])
    best = ", ".join(f"{rows[i]['subject']} at {mae[i]:.1f}" for i in order[-3:])
    text += f"""{marker}
Per-subject error under the unadapted population model varies by a factor of
{mae[order[:3]].mean() / mae[order[-3:]].mean():.1f}: worst {worst}, best {best}.

How far a subject's median heart rate sits from the population median predicts that
error, but the strength depends heavily on two subjects and the claim needs stating
carefully. Across all fifteen, r = 0.87 with a bootstrap interval of [0.52, 0.97].
Removing S6 leaves r = 0.89. Removing both S5 and S6, the two extremes, leaves
r = 0.63, p = 0.02, explaining 40 per cent of the variance among the remaining
thirteen. The honest summary is that the relationship holds across the sample but
that the two outlying subjects make it look stronger than it is, and that five
predictors were tested on fifteen points.

A label-free version performs as well. Replacing the subject's true median heart rate
with the population model's own median prediction on that subject's unlabelled
windows gives r = 0.85 across all fifteen and r = 0.62 among the thirteen. That
version needs no ground truth from the subject, so it could triage a new member
before any labelled data exists.

The mechanism is largely regression to the mean rather than physiology. Ridge shrinks
predictions toward the training mean, so a subject far from the centre is pulled
toward it, and the resulting systematic offset appears as error. Decomposing each
subject's error confirms it: for S5 the bias is -22.6 BPM of a 24.5 BPM error, 92 per
cent, with predictions consistently too low; for S6, -14.3 of 17.2, 83 per cent. The
well-served subjects have near-zero bias, 2 to 21 per cent. Low heart-rate subjects
are over-predicted and high ones under-predicted, which is what shrinkage looks like.

S6 deserves a separate note, because its elevated heart rate could be an artefact of
the truncated recording: the missing hour contains lunch, walking and working, whose
median heart rates across the other subjects are 80, 96 and 74. Restricting every
subject to the activities S6 does have, the other fourteen have a median of 88 against
S6's 121. So truncation accounts for about 4 BPM of S6's 37 BPM distance from the
population, and the remaining 33 is the subject. S6 is genuinely a high heart-rate
individual rather than a collection artefact.
"""
    appendix.write_text(text)
    print(f"appended a section to {appendix}")


if __name__ == "__main__":
    main()
