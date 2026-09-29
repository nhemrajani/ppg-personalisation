"""Low-rank adaptation for PaPaGei's convolutional encoder.

The parameter-efficient fine-tuning literature was written for transformers,
where the adapted weights are matrices. PaPaGei's encoder has no matrices to
adapt: all 37 of its convolutions use a kernel of size 3, so the option of
adapting 1x1 convolutions, which would be matrix multiplications, does not
exist here.

What remains is to treat a convolution kernel of shape (out, in, k) as a
matrix of shape (out, in * k) and learn a low-rank correction to it. A rank r
correction on such a kernel costs r * (out + in * k) parameters, against
out * in * k for the kernel itself.

The correction starts at exactly zero, so an unadapted model is identical to
the population model rather than merely close to it. That mirrors the
warm-start in Arm C1 and means the cost axis begins at a real zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LoRAConv1d(nn.Module):
    """Wraps a frozen Conv1d with a trainable low-rank correction."""

    def __init__(self, conv: nn.Conv1d, rank: int = 4, alpha: float | None = None):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self.conv = conv
        for parameter in self.conv.parameters():
            parameter.requires_grad_(False)

        out_channels, in_channels, kernel = conv.weight.shape
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.kernel_shape = (out_channels, in_channels, kernel)

        # B starts at zero, so the correction is exactly zero at initialisation.
        self.lora_a = nn.Parameter(torch.empty(rank, in_channels * kernel))
        self.lora_b = nn.Parameter(torch.zeros(out_channels, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    @property
    def trainable_parameters(self) -> int:
        return self.lora_a.numel() + self.lora_b.numel()

    def delta(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a).view(self.kernel_shape) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight + self.delta().to(self.conv.weight.dtype)
        return F.conv1d(
            x,
            weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def attach_lora(model: nn.Module, targets: list[str], rank: int = 4) -> list[LoRAConv1d]:
    """Replace named Conv1d layers with LoRA-wrapped copies, in place.

    `targets` are dotted module paths as `named_modules` reports them.
    """
    attached = []
    for path in targets:
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        conv = getattr(parent, leaf)
        if not isinstance(conv, nn.Conv1d):
            raise TypeError(f"{path} is {type(conv).__name__}, not Conv1d")
        wrapped = LoRAConv1d(conv, rank=rank)
        setattr(parent, leaf, wrapped)
        attached.append(wrapped)
    if not attached:
        raise ValueError("no layers were adapted")
    return attached


def freeze_except(model: nn.Module, trainable: list[nn.Parameter]) -> None:
    """Freeze the whole model, then unfreeze exactly the given parameters."""
    keep = {id(p) for p in trainable}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in keep)


def adaptable_convolutions(model: nn.Module, last_n_blocks: int = 1) -> list[str]:
    """Convolution paths in the final residual blocks, the widest and latest."""
    blocks = [n for n, _ in model.named_modules() if n.startswith("basicblock_list.")]
    indices = sorted({int(n.split(".")[1]) for n in blocks})[-last_n_blocks:]
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv1d)
        and name.startswith("basicblock_list.")
        and int(name.split(".")[1]) in indices
    ]
