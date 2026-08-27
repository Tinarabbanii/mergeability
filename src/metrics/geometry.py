from __future__ import annotations
import torch
import torch.nn.functional as F

def compute_pair(flat_a: torch.Tensor, flat_b: torch.Tensor) -> dict[str, float]:
    cos = F.cosine_similarity(flat_a, flat_b, dim=0).item() # Cosine similarity
    l2 = (flat_a - flat_b).norm().item() # Euclidean distance
    dot = torch.dot(flat_a, flat_b).item() # Dot product
    na, nb = flat_a.norm().item(), flat_b.norm().item() # Norm Ratio
    ratio = min(na, nb) / max(na, nb) if max(na, nb) > 0 else 0.0 # Norm mean
    return {
        "tv_cosine": cos,
        "tv_l2": l2,
        "tv_dot": dot,
        "tv_norm_ratio": ratio,
        "tv_norm_mean": 0.5 * (na + nb),}