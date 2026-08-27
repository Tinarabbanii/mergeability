from __future__ import annotations
import torch
import torch.nn.functional as F

### Mean activation vectors
def activation_pair(act_a: torch.Tensor, act_b: torch.Tensor) -> dict[str, float]:
    na, nb = act_a.norm().item(), act_b.norm().item()
    return {
        "act_l2": (act_a - act_b).norm().item(),
        "act_cosine": F.cosine_similarity(act_a, act_b, dim=0).item(),
        "act_magnitude_ratio": (min(na, nb) / max(na, nb)) if max(na, nb) > 0 else 0.0,
        "act_dot": torch.dot(act_a, act_b).item(),
    }

### Gradient Aligment
def gradient_pair(grad_a: torch.Tensor, grad_b: torch.Tensor) -> dict[str, float]:
    return {
        "grad_cosine": F.cosine_similarity(grad_a, grad_b, dim=0).item(),
        "grad_l2": (grad_a - grad_b).norm().item(),
        "grad_dot": torch.dot(grad_a, grad_b).item(),
    }
