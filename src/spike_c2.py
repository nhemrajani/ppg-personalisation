"""Arm C2 feasibility spike: does low-rank adaptation of the trunk train at all?

Not whether it helps. Whether it runs, whether gradients reach the adapters,
and whether the loss falls. The arm's fallback, adapting the final projection,
is known to work, so the question is only whether the more expressive variant
is reachable before committing to it in November.

Trains two variants on one subject for the same number of steps: a head alone,
and a head plus low-rank corrections on the final residual block. If the loss
falls no further with the adapters attached, the adapters are doing nothing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.embed import device_name
from src.lora import LoRAConv1d, adaptable_convolutions, attach_lora, freeze_except
from src.papagei import EMBEDDING_DIM, load_backbone
from src.preprocess import preprocess_subject

SUBJECT = "S1"
WINDOWS = 512  # a few minutes of one person, the scale the arm actually runs at
STEPS, BATCH, SEED = 60, 32, 0


def run(use_lora: bool, windows: torch.Tensor, target: torch.Tensor, device: str) -> dict:
    torch.manual_seed(SEED)
    model = load_backbone("s", device)
    head = nn.Linear(EMBEDDING_DIM, 1).to(device)

    trainable = list(head.parameters())
    adapters: list[LoRAConv1d] = []
    if use_lora:
        adapters = attach_lora(model, adaptable_convolutions(model, last_n_blocks=1), rank=4)
        model.to(device)
        trainable += [p for a in adapters for p in (a.lora_a, a.lora_b)]
    freeze_except(model, [p for a in adapters for p in (a.lora_a, a.lora_b)])

    adapter_params = sum(a.trainable_parameters for a in adapters)
    optimiser = torch.optim.Adam(trainable, lr=1e-3)
    before = [a.lora_b.detach().clone() for a in adapters]

    model.train()
    losses = []
    generator = np.random.default_rng(SEED)
    for _ in range(STEPS):
        idx = generator.choice(len(windows), BATCH, replace=False)
        batch, y = windows[idx].to(device), target[idx].to(device)
        prediction = head(model(batch.unsqueeze(1))[0]).squeeze(-1)
        loss = nn.functional.mse_loss(prediction, y)
        optimiser.zero_grad()
        loss.backward()
        if adapters:  # confirm gradient actually reaches the trunk adapters.
            # B starts at zero, so A's gradient is legitimately zero on the first
            # step and only becomes non-zero once B has moved. B is the one to check.
            grads = [a.lora_b.grad for a in adapters if a.lora_b.grad is not None]
            assert grads and any(g.abs().sum() > 0 for g in grads), "no gradient reached the adapters"
        optimiser.step()
        losses.append(loss.detach().item())

    moved = (
        max(float((a.lora_b - b).abs().max()) for a, b in zip(adapters, before))
        if adapters else 0.0
    )
    return {
        "first": float(np.mean(losses[:5])),
        "last": float(np.mean(losses[-5:])),
        "adapter_params": adapter_params,
        "moved": moved,
        "adapters": len(adapters),
    }


def main() -> None:
    device = device_name()
    prepared = preprocess_subject(SUBJECT)
    windows = torch.from_numpy(prepared.windows[:WINDOWS])
    hr = prepared.heart_rate[:WINDOWS]
    target = torch.from_numpy(((hr - hr.mean()) / hr.std()).astype(np.float32))
    print(f"{SUBJECT}, {len(windows)} windows, {STEPS} steps of batch {BATCH}, device {device}\n")

    head_only = run(False, windows, target, device)
    with_lora = run(True, windows, target, device)

    print(f"{'variant':22s} {'loss start':>11s} {'loss end':>10s} {'trainable in trunk':>20s}")
    print(f"{'head only':22s} {head_only['first']:11.4f} {head_only['last']:10.4f} {0:20,d}")
    print(f"{'head + LoRA rank 4':22s} {with_lora['first']:11.4f} {with_lora['last']:10.4f} "
          f"{with_lora['adapter_params']:20,d}")
    print(f"\nadapters attached: {with_lora['adapters']} convolutions in the final block")
    print(f"largest change in an adapter weight: {with_lora['moved']:.4f}")

    fell = with_lora["last"] < with_lora["first"]
    better = with_lora["last"] < head_only["last"]
    print(f"\nloss fell with adapters attached: {fell}")
    print(f"lower than head alone at the same step count: {better}")
    print("\nGate 7 fallback remains the final projection layer either way.")


if __name__ == "__main__":
    main()
