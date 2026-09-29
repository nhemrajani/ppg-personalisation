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

import time

import numpy as np
import torch
from torch import nn

from src.arms import arm_a
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
STEPS, BATCH, LR, SEED = 200, 32, 1e-3, 0
TARGET_LAYER = "basicblock_list.17.conv2.conv"
TOLERANCE = 1e-4


def _windows(subject: int) -> np.ndarray:
    return preprocess_subject(f"S{subject}").windows


def _head_from_ridge(x_pop, y_pop, alpha, device):
    """Arm A's ridge solution as a linear layer, so training starts from it."""
    model = GramRidge(x_pop, y_pop)
    head = nn.Linear(EMBEDDING_DIM, 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(model.weights(alpha), dtype=torch.float32).view(1, -1))
        head.bias.fill_(float(model.y_mean - model.x_mean @ model.weights(alpha)))
    return head, model


def run_one(subject: int, budget, rank: int | None, protocol: str, index, device):
    """One C2 or D run. rank=None means Arm D, full fine-tuning."""
    split = make_split(index, subject, protocol=protocol)
    y, subj = index["heart_rate"], index["subject"]
    x_cached = np.asarray(embeddings("p", "asis"), dtype=np.float64)
    base = arm_a(x_cached, y, split, subj)

    scaler = Standardiser.fit(x_cached[split.population])
    x_pop, y_pop = scaler(x_cached[split.population]), y[split.population]
    head, ridge = _head_from_ridge(x_pop, y_pop, base.alpha, device)

    model = load_backbone("p", device)
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

    windows = _windows(subject)
    local = {int(v): i for i, v in enumerate(np.flatnonzero(subj == subject))}
    chunk = budget_prefix(split, index, budget)
    rows = np.array([local[int(i)] for i in chunk])
    test_rows = np.array([local[int(i)] for i in split.test])
    if len(rows) < BATCH:
        return None

    x_adapt = torch.from_numpy(windows[rows]).unsqueeze(1)
    y_adapt = torch.from_numpy(y[chunk].astype(np.float32))
    optimiser = torch.optim.Adam(trainable + list(head.parameters()), lr=LR)

    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    model.train()
    for _ in range(STEPS):
        pick = generator.choice(len(x_adapt), BATCH, replace=False)
        pred = head(model(x_adapt[pick].to(device))[0]).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_adapt[pick].to(device))
        optimiser.zero_grad(); loss.backward(); optimiser.step()

    model.eval()
    x_test = torch.from_numpy(windows[test_rows]).unsqueeze(1)
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x_test), 256):
            preds.append(head(model(x_test[i:i+256].to(device))[0]).squeeze(-1).cpu().numpy())
    prediction = np.concatenate(preds)
    y_test = y[split.test]

    return Run(
        arm="D" if rank is None else "C2", backbone="p", polarity="asis", protocol=protocol,
        subject=subject, budget_minutes=budget, trainable_parameters=cost,
        stored_bytes=cost * 4, mae=round(mae(y_test, prediction), 4),
        rmse=round(rmse(y_test, prediction), 4), pearson=round(pearson(y_test, prediction), 4),
        n_test_windows=len(y_test), rank=rank, alpha=base.alpha, seed=SEED,
        hyperparameters=f'{{"steps": {STEPS}, "lr": {LR}, "layer": "{TARGET_LAYER}"}}',
        notes="in-band" if rank in IN_BAND else ("extension" if rank else "upper bound"),
    )


def main() -> None:
    index = load_index()
    device = device_name()
    subjects = [int(s) for s in np.unique(index["subject"])]
    print(f"device {device}, ranks {RANKS}, budgets {BUDGETS}\n")

    started, rows = time.time(), []
    for rank in list(RANKS) + [None]:
        label = f"C2 rank {rank}" if rank else "D full fine-tune"
        for budget in BUDGETS:
            for subject in subjects:
                row = run_one(subject, budget, rank, "activity", index, device)
                if row:
                    rows.append(row)
            print(f"  {label}, budget {budget}: {len(rows):,} rows, {time.time()-started:.0f}s",
                  flush=True)
            append([r for r in rows[-len(subjects):]])
            rows = []

    # Registered single-budget check under the naive temporal protocol.
    check = [run_one(s, None, 4, "temporal", index, device) for s in subjects]
    check += [run_one(s, None, None, "temporal", index, device) for s in subjects]
    append([r for r in check if r])
    print(f"\ndone in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
