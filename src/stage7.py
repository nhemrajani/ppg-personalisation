"""Stage 7: Arm C2, low-rank adaptation of one convolution, and Arm D.

Registered before running: the adapted layer is the second convolution of the
final residual block, a single 512-channel convolution where a rank-r
correction costs 2,048r. Ranks 1, 2 and 4 are the primary sweep and sit inside
the stated band; ranks 8 and 16 are a labelled extension beyond it. Arm D
trains every parameter on the embedding path for a fixed 200 steps with no
early stopping, its head initialised from Arm A's ridge solution.

Both arms run under the activity-stratified protocol, with a single-budget
check under the naive temporal protocol, as registered.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from torch import nn

from src.arms import arm_a
from src.effective_n import effective_n
from src.embed import device_name
from src.harness import Run, append
from src.lora import LoRAConv1d, attach_lora, freeze_except
from src.papagei import EMBEDDING_DIM, load_backbone
from src.preprocess import preprocess_subject
from src.ridge import GramRidge, Standardiser, mae, pearson, rmse
from src.splits import budget_prefix, load_index, make_split
from src.stage4 import embeddings

RANKS = (1, 2, 4, 8, 16)
IN_BAND = (1, 2, 4)
BUDGETS = (2, 5, 10, 20, 40, None)
BATCH, SEED = 32, 0

# Amended 26 October 2026: training amount is fixed in passes over the budget's
# own data, not in steps. A fixed 200 steps gave the two-minute budget about 107
# passes and the all-available budget about two, which confounded the sweep.
EPOCHS, MIN_STEPS = 10, 10

# Amended 26 October 2026: C2's rate is selected per rank from this grid, once,
# on the held-out tail of two subjects' adaptation blocks. Arm D keeps the 1e-5
# of the 29 September amendment, which was selected the same way.
LR_GRID = (3e-3, 1e-3, 3e-4, 1e-4)
LR_SELECTION_SUBJECTS = (1, 7)
LR = {"D": 1e-5}

TARGET_LAYER = "basicblock_list.17.conv2.conv"
BUFFER_WINDOWS = 4  # windows overlap by six of eight seconds
TOLERANCE = 1e-4


def step_count(n_windows: int) -> int:
    """Steps giving EPOCHS passes over n_windows at batch size BATCH."""
    return max(MIN_STEPS, int(round(EPOCHS * n_windows / BATCH)))


def _windows(subject: int) -> np.ndarray:
    return preprocess_subject(f"S{subject}").windows


def _head_from_ridge(x_raw, y_pop, alpha, device, scaler: Standardiser):
    """Arm A's ridge solution as a linear layer over RAW encoder output.

    The ridge is fitted on standardised embeddings, but the encoder emits raw
    ones, so the standardiser is folded into the layer's weights. Without this
    the head is applied to features on a different scale from the ones it was
    fitted on, which silently produces predictions far worse than Arm A.
    """
    model = GramRidge(scaler(x_raw), y_pop)
    w = model.weights(alpha)
    weight = w / scaler.scale
    bias = float(model.y_mean - (scaler.mean / scaler.scale + model.x_mean) @ w)
    head = nn.Linear(EMBEDDING_DIM, 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight, dtype=torch.float32).view(1, -1))
        head.bias.fill_(bias)
    return head, model


def run_one(subject: int, budget, rank: int | None, protocol: str, index, device,
            backbone: str = "p", lr: float | None = None, select: bool = False):
    """One C2 or D run. rank=None means Arm D, full fine-tuning.

    With select=True the arm trains on the first three quarters of the budget
    and is scored on the held-out tail instead of the test block, which is how
    the learning rate is chosen. The test block is not touched in that mode.
    """
    split = make_split(index, subject, protocol=protocol)
    y, subj = index["heart_rate"], index["subject"]
    x_cached = np.asarray(embeddings(backbone, "asis"), dtype=np.float64)
    base = arm_a(x_cached, y, split, subj)

    scaler = Standardiser.fit(x_cached[split.population])
    head, ridge = _head_from_ridge(x_cached[split.population], y[split.population],
                                   base.alpha, device, scaler)

    model = load_backbone(backbone, device)
    adapters: list[LoRAConv1d] = []
    if rank is not None:
        adapters = attach_lora(model, [TARGET_LAYER], rank=rank)
        model.to(device)
        trainable = [p for a in adapters for p in (a.lora_a, a.lora_b)]
        freeze_except(model, trainable)
        cost = sum(a.trainable_parameters for a in adapters)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = [p for n, p in model.named_parameters()
                     if not n.startswith(("expert_layers", "gating_network"))]
        cost = sum(p.numel() for p in trainable)

    # Targets are standardised on population statistics, with the scaling folded
    # into the head so the arm still starts exactly at Arm A while the loss is O(1).
    mu = float(y[split.population].mean())
    sd = float(y[split.population].std()) or 1.0
    with torch.no_grad():
        head.weight.div_(sd)
        head.bias.copy_((head.bias - mu) / sd)

    windows = _windows(subject)
    local = {int(v): i for i, v in enumerate(np.flatnonzero(subj == subject))}
    chunk = budget_prefix(split, index, budget)

    if select:
        # Train on the first three quarters, score on the held-out tail, with a
        # buffer at the cut because neighbouring windows share six seconds.
        cut = max(BATCH, int(len(chunk) * 0.75))
        if cut + BUFFER_WINDOWS >= len(chunk):
            return None
        score_on = chunk[cut + BUFFER_WINDOWS:]
        chunk = chunk[:cut]
    else:
        score_on = split.test

    rows = np.array([local[int(i)] for i in chunk])
    test_rows = np.array([local[int(i)] for i in score_on])
    if len(rows) < BATCH:
        return None

    steps = step_count(len(rows))
    rate = LR["D"] if rank is None else (lr if lr is not None else LR_GRID[1])

    x_adapt = torch.from_numpy(windows[rows]).unsqueeze(1)
    y_adapt = torch.from_numpy(((y[chunk] - mu) / sd).astype(np.float32))
    optimiser = torch.optim.Adam(trainable + list(head.parameters()), lr=rate)

    # Registered assertion: before any training, the arm must reproduce Arm A.
    # The low-rank correction is zero at initialisation and Arm D has not moved,
    # so any disagreement here is a bug in the head or the encoder path.
    model.eval()
    with torch.inference_mode():
        assert_rows = np.array([local[int(i)] for i in split.test[:256]])
        start = head(model(torch.from_numpy(windows[assert_rows]).unsqueeze(1).to(device))[0])
    drift = float(np.abs(start.squeeze(-1).cpu().numpy() * sd + mu - base.predictions[:256]).max())
    assert drift < 1e-2, (
        f"S{subject} {('C2 rank ' + str(rank)) if rank else 'D'}: at initialisation the arm "
        f"differs from Arm A by {drift:.3f} BPM. It must reproduce Arm A before training."
    )

    def adaptation_errors() -> tuple[float, float]:
        """MAE and MSE in BPM on the training block, for the record."""
        model.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(x_adapt), 256):
                out.append(head(model(x_adapt[i:i+256].to(device))[0]).squeeze(-1).cpu().numpy())
        got = np.concatenate(out) * sd + mu
        want = y[chunk]
        return float(np.mean(np.abs(got - want))), float(np.mean((got - want) ** 2))

    before_mae, before_mse = adaptation_errors()

    # Amended 26 October 2026: the loss is MAE, the metric the study reports.
    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    trace: list[float] = []
    marks = {0, max(1, steps // 4), max(1, steps // 2), steps - 1}
    model.train()
    for step in range(steps):
        pick = generator.choice(len(x_adapt), BATCH, replace=False)
        pred = head(model(x_adapt[pick].to(device))[0]).squeeze(-1)
        loss = nn.functional.l1_loss(pred, y_adapt[pick].to(device))
        optimiser.zero_grad(); loss.backward(); optimiser.step()
        if step in marks:
            trace.append(round(float(loss.detach().cpu()), 4))

    after_mae, after_mse = adaptation_errors()

    model.eval()
    x_test = torch.from_numpy(windows[test_rows]).unsqueeze(1)
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x_test), 256):
            preds.append(head(model(x_test[i:i+256].to(device))[0]).squeeze(-1).cpu().numpy())
    prediction = np.concatenate(preds) * sd + mu  # back to BPM before scoring
    y_test = y[score_on]

    return Run(
        arm="D" if rank is None else "C2", backbone=backbone, polarity="asis", protocol=protocol,
        subject=subject, budget_minutes=budget, trainable_parameters=cost,
        stored_bytes=cost * 4, mae=round(mae(y_test, prediction), 4),
        rmse=round(rmse(y_test, prediction), 4), pearson=round(pearson(y_test, prediction), 4),
        n_test_windows=len(y_test), rank=rank, alpha=base.alpha, seed=SEED,
        hyperparameters=json.dumps({
            "steps": steps, "epochs": EPOCHS, "n_adapt_windows": int(len(rows)),
            "effective_n": effective_n(index["start_second"][chunk]),
            "lr": rate, "loss": "l1", "layer": TARGET_LAYER,
            "adapt_mae_before": round(before_mae, 3), "adapt_mae_after": round(after_mae, 3),
            "adapt_mse_before": round(before_mse, 1), "adapt_mse_after": round(after_mse, 1),
            "loss_trace": trace, "selection_mode": select,
        }),
        notes="in-band" if rank in IN_BAND else ("extension" if rank else "upper bound"),
    )


def select_rates(index, device, backbone: str) -> dict[int, float]:
    """One learning rate per rank, chosen on held-out adaptation data only.

    Amended 26 October 2026. Scored on the final quarter of the adaptation block
    of two subjects at the all-available budget, which the test block plays no
    part in. Selection happens once per rank rather than per row, following the
    precedent of the Arm D amendment of 29 September.
    """
    print(f"selecting C2's learning rate per rank on S{LR_SELECTION_SUBJECTS}, "
          f"held-out adaptation data only")
    chosen: dict[int, float] = {}
    for rank in RANKS:
        scores = {}
        for rate in LR_GRID:
            values = [run_one(s, None, rank, "activity", index, device,
                              backbone=backbone, lr=rate, select=True)
                      for s in LR_SELECTION_SUBJECTS]
            values = [r.mae for r in values if r]
            scores[rate] = float(np.mean(values)) if values else float("inf")
        chosen[rank] = min(scores, key=scores.get)
        detail = "  ".join(f"{r:.0e}:{v:6.2f}" for r, v in scores.items())
        print(f"  rank {rank:2d}  {detail}   chosen {chosen[rank]:.0e}", flush=True)
    return chosen


def main() -> None:
    index = load_index()
    device = device_name()
    subjects = [int(s) for s in np.unique(index["subject"])]
    backbones = sys.argv[1].split(",") if len(sys.argv) > 1 else ["p"]
    protocols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["activity", "temporal"]
    print(f"device {device}, backbones {backbones}, protocols {protocols}, "
          f"ranks {RANKS}, budgets {BUDGETS}, {EPOCHS} epochs\n")

    started = time.time()
    for backbone in backbones:
        rates = select_rates(index, device, backbone)
        print()
        for protocol in protocols:
            for rank in list(RANKS) + [None]:
                label = f"C2 rank {rank}" if rank else "D full fine-tune"
                for budget in BUDGETS:
                    rows = []
                    for subject in subjects:
                        row = run_one(subject, budget, rank, protocol, index, device,
                                      backbone=backbone,
                                      lr=rates.get(rank) if rank else None)
                        if row:
                            rows.append(row)
                    append(rows)
                    print(f"  {backbone}/{protocol} {label}, budget "
                          f"{str(budget or 'all'):>3s}: {len(rows):2d} rows, "
                          f"{time.time() - started:.0f}s", flush=True)
    print(f"\ndone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
