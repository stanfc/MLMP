"""LayerNorm gradient helpers."""
from __future__ import annotations
import torch
import torch.nn as nn


def flatten_grads(ln_params: list[nn.Parameter], strict: bool = False) -> torch.Tensor:
    """Concatenate .grad of each param into one fp32 1-D tensor.

    Iteration order is the order given.

    A param may have .grad=None if it is in the leaf set (requires_grad=True)
    but does not actually participate in the loss's computation graph — for
    example NA-CLIP's `ln_post` is not on the path that produces patch-token
    logits. By default we treat None as a zero gradient (mathematically the
    same: the loss is constant in that param so ∂L/∂p = 0).

    Set strict=True to raise on None instead — useful when debugging.
    """
    chunks = []
    for i, p in enumerate(ln_params):
        if p.grad is None:
            if strict:
                raise RuntimeError(f"ln_params[{i}] has .grad=None")
            chunks.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            chunks.append(p.grad.detach().reshape(-1).float())
    return torch.cat(chunks)


def cosine(g1: torch.Tensor, g2: torch.Tensor) -> tuple[float, float, float]:
    """Return (cosine_similarity, ||g1||_2, ||g2||_2) as python floats.

    All computation is fp32 on the input device. Returns floats so the result
    can go straight into a CSV row.
    """
    g1 = g1.float()
    g2 = g2.float()
    n1 = torch.linalg.vector_norm(g1).item()
    n2 = torch.linalg.vector_norm(g2).item()
    if n1 == 0.0 or n2 == 0.0:
        return 0.0, n1, n2
    cos = torch.dot(g1, g2).item() / (n1 * n2)
    return float(cos), float(n1), float(n2)
