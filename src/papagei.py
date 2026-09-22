"""Loading the PaPaGei encoders.

PaPaGei's model code is vendored as a git submodule at external/papagei, pinned
to a known commit, and imported from there rather than copied. Weights are
downloaded separately from Zenodo into weights/ and never committed.

The two checkpoints use different classes, but the difference sits outside the
embedding. PaPaGei-S is a ResNet1DMoE and PaPaGei-P a plain ResNet1D; the
embedding path in each is the same convolutional trunk, global average pool and
512 -> 512 dense projection, with identical tensor shapes. PaPaGei-S adds
mixture-of-experts heads that branch off after pooling, used as auxiliary
targets in pretraining and never feeding the embedding. So on the path that
matters here the backbones share an architecture and differ only in their
pretraining objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
PAPAGEI_DIR = ROOT / "external" / "papagei"
WEIGHTS_DIR = ROOT / "weights"

if not (PAPAGEI_DIR / "models" / "resnet.py").exists():
    raise ImportError(
        "PaPaGei source not found. Run: git submodule update --init"
    )
if str(PAPAGEI_DIR) not in sys.path:
    sys.path.insert(0, str(PAPAGEI_DIR))

from models.resnet import ResNet1D, ResNet1DMoE  # noqa: E402

SAMPLE_RATE_HZ = 125
INPUT_LENGTH = 1250  # ten seconds at 125 Hz; eight-second windows are padded to this
EMBEDDING_DIM = 512

# Configuration as published in PaPaGei's example notebook.
_CONFIG = {
    "base_filters": 32,
    "kernel_size": 3,
    "stride": 2,
    "groups": 1,
    "n_block": 18,
    "n_classes": EMBEDDING_DIM,
}

# Parameters that exist only in PaPaGei-S and never reach the embedding.
_PRETRAINING_HEADS = ("expert_layers", "gating_network")

BACKBONES = {
    "s": {
        "file": "papagei_s.pt",
        "md5": "a4cdb32392e2a7b25999128af92813b5",
        "build": lambda: ResNet1DMoE(in_channels=1, n_experts=3, **_CONFIG),
    },
    "p": {
        "file": "papagei_p.pt",
        "md5": "052b50807465fae61e08e2b7acbb5c53",
        "build": lambda: ResNet1D(in_channels=1, **_CONFIG),
    },
}


def load_backbone(name: str, device: str | torch.device = "cpu") -> nn.Module:
    """Build a PaPaGei encoder and load its pretrained weights, in eval mode.

    Mirrors PaPaGei's load_model_without_module_prefix, which strips the
    DataParallel "module." prefix and loads strictly, with two additions:
    map_location, so checkpoints saved on a GPU load on any machine, and
    weights_only, so the checkpoint cannot execute code when unpickled.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; expected one of {sorted(BACKBONES)}")
    spec = BACKBONES[name]
    path = WEIGHTS_DIR / spec["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            f'  curl -L -o {path} "https://zenodo.org/records/13983110/files/{spec["file"]}?download=1"'
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = {k.removeprefix("module."): v for k, v in checkpoint.items()}

    model = spec["build"]()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """512-dimensional embeddings for a batch shaped (batch, 1, INPUT_LENGTH).

    Both classes return a tuple whose first element is the dense projection of
    the pooled trunk output. That is the representation PaPaGei's own linear
    probing uses, so it is the one used here.
    """
    with torch.inference_mode():
        return model(x)[0]


def embedding_path_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Trainable parameters that influence the embedding.

    Excludes PaPaGei-S's pretraining heads, which do not affect the embedding,
    so that full fine-tuning (Arm D) costs the same on both backbones and is
    not credited with parameters that change nothing.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if not name.startswith(_PRETRAINING_HEADS)
    ]


def count_parameters(params: list[tuple[str, nn.Parameter]]) -> int:
    return sum(param.numel() for _, param in params)


def _gate_one() -> None:
    """Gate 1: both encoders load, and random noise produces 512 numbers.

    Meaningless as a prediction. It proves the weights load strictly into the
    architecture, that the forward pass runs, and that the output has the
    expected shape on every available device.
    """
    import platform

    torch.manual_seed(0)
    noise = torch.randn(4, 1, INPUT_LENGTH)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

    print(f"python {platform.python_version()} | torch {torch.__version__} | devices: {', '.join(devices)}\n")
    for name in BACKBONES:
        model = load_backbone(name)
        out = embed(model, noise)
        assert out.shape == (4, EMBEDDING_DIM), f"unexpected shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "non-finite values in embedding"

        total = sum(p.numel() for p in model.parameters())
        path = count_parameters(embedding_path_parameters(model))
        dense = model.dense.weight.numel() + model.dense.bias.numel()
        print(f"PaPaGei-{name.upper()}  {type(model).__name__}")
        print(f"  output {tuple(out.shape)}, finite, mean {out.mean():+.4f}, std {out.std():.4f}")
        print(f"  trainable parameters: {total:,} total, {path:,} on the embedding path")
        print(f"  dense projection: {dense:,}")

        for device in devices[1:]:
            other = embed(load_backbone(name, device), noise.to(device)).cpu()
            print(f"  {device} vs cpu, max absolute difference: {(other - out).abs().max():.2e}")
        print()

    print("Gate 1 passed.")


if __name__ == "__main__":
    _gate_one()
