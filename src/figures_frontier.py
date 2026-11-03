"""The two figures the paper rests on, regenerated from results/experiments.csv.

  figures/frontier_cost_accuracy.png   cost per person against error, one point
                                       per arm, with the floor baselines marked
  figures/frontier_sample_efficiency.png
                                       personal-data budget against improvement
                                       over doing nothing, one line per arm

Both read the record only. A backbone appears as a panel when its rows exist, so
MOMENT joins the figures as soon as Stage 11 has run without this file changing.

The cost axis is trainable parameters per person and is logarithmic, which has no
room for zero. Zero-cost arms are drawn in a separate strip at the left, divided
by a dashed rule, rather than being nudged to a small positive number: a
population model and a one-parameter correction differ in kind, not by a factor.
"""

from __future__ import annotations

import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.floors import load, paired, series  # noqa: E402
from src.measure_quality import FIGURES  # noqa: E402

PROTOCOL = "activity"
BUDGETS = ("2", "5", "10", "20", "40", "")
BUDGET_TICK = {"2": "2", "5": "5", "10": "10", "20": "20", "40": "40", "": "all"}
BACKBONES = (("p", "PaPaGei-P"), ("s", "PaPaGei-S"), ("moment", "MOMENT-1-large"))

# Cost in trainable parameters per person, and the colour each arm carries in
# both figures. Ordered by cost so the legend reads as the cost axis does.
ARMS = (
    ("Z0", "population constant", 0, "#8d8d8d"),
    ("A", "population model", 0, "#1b1b1b"),
    ("B2", "unlabelled reweighting", 0, "#7a9e3f"),
    ("Z1", "per-person constant", 1, "#c2884a"),
    ("B-offset", "offset only", 1, "#2f7d95"),
    ("B", "affine", 2, "#b23a48"),
    ("B-ridge", "shrunk affine", 2, "#d4868f"),
    ("C1", "output layer", 513, "#4a6fa5"),
    ("C2-r16", "low rank, r=16", 32768, "#6b4a9e"),
    ("D", "full fine-tuning", 4993024, "#3d3d3d"),
)
ZERO_X = 0.35          # where the zero-cost strip sits on the log axis
ZERO_DIVIDE = 0.62     # the dashed rule between the strip and the real axis
# Long names on adjacent costs run into each other, so these are labelled on
# their left. Cosmetic only: the marker is unmoved.
LABEL_ON_LEFT = {"B-offset"}


def caption(ax, title: str, note: str, width: int = 88) -> None:
    lines = textwrap.wrap(note, width)
    ax.set_title(title, fontsize=11, loc="left", pad=12 + 12 * len(lines))
    ax.annotate("\n".join(lines), xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 6), textcoords="offset points",
                fontsize=8.5, color="#555", va="bottom", linespacing=1.45)


def present(table, backbone) -> bool:
    return series(table, "A", backbone, PROTOCOL, "") is not None


def cost_accuracy(table, backbones) -> None:
    """Cost per person against error, with the Pareto staircase drawn."""
    fig, axes = plt.subplots(1, len(backbones), figsize=(7.2 * len(backbones), 5.6),
                             squeeze=False)
    for ax, (backbone, pretty) in zip(axes[0], backbones):
        found = [(arm, name, cost, colour, series(table, arm, backbone, PROTOCOL, ""))
                 for arm, name, cost, colour in ARMS]
        found = [f for f in found if f[4] is not None]

        # Several arms cost the same, and two of them differ by 0.05 BPM, so they
        # would land on top of each other. Arms sharing a cost are nudged apart
        # multiplicatively, which is said in the caption; the tick each sits
        # under is still its true cost.
        groups: dict[int, list] = {}
        for item in found:
            groups.setdefault(item[2], []).append(item)
        points = []
        for cost, members in groups.items():
            centre = ZERO_X if cost == 0 else float(cost)
            members = sorted(members, key=lambda m: m[4].mean())
            for i, (arm, name, _, colour, values) in enumerate(members):
                spread = i - (len(members) - 1) / 2
                # Arms B and B-ridge differ by 0.05 BPM at the same cost, so an
                # x nudge alone leaves their labels on top of each other. The
                # label is staggered vertically as well.
                points.append((cost, centre * 1.26 ** spread, values.mean(),
                               arm, name, colour, spread * 14))

        baseline = series(table, "A", backbone, PROTOCOL, "")
        for cost, x, value, arm, name, colour, dy in points:
            err = None
            if arm != "A":
                other = series(table, arm, backbone, PROTOCOL, "")
                _, lo, hi = paired(other, baseline)[:3]
                # The interval is on the difference, so recentre it on the arm.
                err = [[abs(value - (baseline.mean() - hi))],
                       [abs((baseline.mean() - lo) - value)]]
            ax.errorbar(x, value, yerr=err, fmt="o", ms=8, color=colour,
                        ecolor=colour, elinewidth=1.1, capsize=3, alpha=0.9, zorder=3)
            side = -1 if arm in LABEL_ON_LEFT else 1
            ax.annotate(arm, (x, value), fontsize=9, color=colour, va="center",
                        ha="right" if side < 0 else "left", weight="bold",
                        xytext=(9 * side, dy), textcoords="offset points", zorder=4)

        # The Pareto staircase: arms no other arm beats at equal or lower cost.
        # It is extended to the right edge on purpose. The whole finding is that
        # it stops at two parameters, and a line that simply ended there would
        # leave the domination of everything costlier to be inferred.
        ordered = sorted(points, key=lambda p: (p[0], p[2]))
        frontier, best = [], np.inf
        for cost, x, value, arm, name, colour, _dy in ordered:
            if value < best - 1e-12:
                frontier.append((x, value))
                best = value
        if len(frontier) > 1:
            fx, fy = zip(*frontier)
            ax.step(list(fx) + [2e7], list(fy) + [fy[-1]], where="post",
                    color="#b23a48", lw=1.3, ls="--", alpha=0.65, zorder=1,
                    label="Pareto frontier (flat: nothing costlier beats it)")

        for arm, style, note in (("Z0", (0, (1, 2)), "knowing nothing"),
                                 ("Z1", (0, (4, 2)), "knowing only the person's own mean")):
            values = series(table, arm, backbone, PROTOCOL, "")
            if values is not None:
                ax.axhline(values.mean(), color="#8d8d8d", lw=1, ls=style, zorder=0)
                # Placed over the empty middle of the axes rather than at the
                # right edge, where the text was being clipped.
                ax.annotate(f"{arm}: {note}", (1e5, values.mean()), fontsize=8,
                            color="#8d8d8d", va="bottom", ha="center",
                            xytext=(0, 3), textcoords="offset points",
                            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

        ax.axvline(ZERO_DIVIDE, color="#bbb", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.18, 2e7)
        ticks = [ZERO_X, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", "1", "10", "100", "1k", "10k", "100k", "1M", "10M"])
        ax.set_xlabel("trainable parameters per person  (log scale; 0 in its own strip)")
        ax.set_ylabel("mean per-subject MAE (BPM)")
        knee = series(table, "B", backbone, PROTOCOL, "")
        caption(ax, f"{pretty}, activity-stratified, all available personal data",
                f"Two stored values reach {knee.mean():.2f} BPM, and the frontier is "
                f"flat from there to five million parameters. Bars are 95 per cent "
                f"paired bootstrap intervals on each arm's difference from Arm A, so "
                f"Arm A itself carries none. Arms of equal cost are nudged apart to "
                f"separate them; each sits under its true cost."
                if knee is not None else "")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = FIGURES / "frontier_cost_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES.parent)}")


def sample_efficiency(table, backbones) -> None:
    """How much personal data each arm needs before it earns its place."""
    fig, axes = plt.subplots(1, len(backbones), figsize=(7.2 * len(backbones), 5.6),
                             squeeze=False)
    x = np.arange(len(BUDGETS))
    for ax, (backbone, pretty) in zip(axes[0], backbones):
        baseline = series(table, "A", backbone, PROTOCOL, "")
        for arm, name, cost, colour in ARMS:
            if arm in ("A", "Z0"):
                continue
            gains, lows, highs = [], [], []
            for budget in BUDGETS:
                values = series(table, arm, backbone, PROTOCOL, budget)
                if values is None:
                    gains.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                    continue
                d, lo, hi, _ = paired(values, baseline)
                gains.append(d); lows.append(lo); highs.append(hi)
            if np.all(np.isnan(gains)):
                continue
            ax.plot(x, gains, "-o", ms=4.5, color=colour, lw=1.6,
                    label=f"{arm} ({cost:,} param)" if cost else f"{arm} (0 param)")
            ax.fill_between(x, lows, highs, color=colour, alpha=0.08, lw=0)

        ax.axhline(0, color="#1b1b1b", lw=1.4)
        # At the left edge, where no arm has risen to the line yet.
        ax.annotate("Arm A: doing nothing", (0, 0), fontsize=8.5, color="#1b1b1b",
                    va="bottom", ha="left", xytext=(2, 4),
                    textcoords="offset points",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels([BUDGET_TICK[b] for b in BUDGETS])
        ax.set_xlabel("personal data used for adaptation (usable minutes)")
        ax.set_ylabel("improvement over Arm A (BPM; positive is better)")
        caption(ax, f"{pretty}, activity-stratified",
                "Below the black line an arm is worse than not adapting. Budgets are "
                "prefixes in time, so the small ones are mostly sitting still and are "
                "unrepresentative of the test block; that composition, not the sample "
                "size, is what makes every arm with a slope fail there. Bands are 95 "
                "per cent paired bootstrap intervals.")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower right", ncol=2)

    fig.tight_layout()
    path = FIGURES / "frontier_sample_efficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES.parent)}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    table = load()
    backbones = [(b, pretty) for b, pretty in BACKBONES if present(table, b)]
    print(f"backbones in the record: {', '.join(b for b, _ in backbones)}")
    cost_accuracy(table, backbones)
    sample_efficiency(table, backbones)


if __name__ == "__main__":
    main()
