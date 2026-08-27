from __future__ import annotations
import torch

def _effective_rank(mat: torch.Tensor, eps: float) -> float:
    if mat.shape[0] < 2:
        return 1.0
    sv = torch.linalg.svdvals(mat)
    total = sv.sum()
    if total <= eps:
        return 1.0
    p = sv / total
    entropy = -(p * torch.log(p.clamp(min=eps))).sum()
    return torch.exp(entropy).item()

def compute_subset(flats: list[torch.Tensor], eps: float = 1e-12) -> dict[str, float]:
    mat = torch.stack(flats)  # (k, D)
    return {"eff_rank_global": _effective_rank(mat, eps)}

def compute_subset_layerwise(per_layer: list[dict[str, torch.Tensor]], eps: float = 1e-12) -> dict[str, float]:
    if not per_layer:
        return {"eff_rank_layerwise": 1.0}
    names = sorted(per_layer[0])
    ranks: list[float] = []
    weights: list[float] = []
    for name in names:
        rows = [d[name] for d in per_layer]
        mat = torch.stack(rows)
        mag = mat.norm().item()
        if mag <= eps:
            continue
        ranks.append(_effective_rank(mat, eps))
        weights.append(mag)
    if not ranks:
        return {"eff_rank_layerwise": 1.0}
    w = torch.tensor(weights)
    r = torch.tensor(ranks)
    return {"eff_rank_layerwise": ((r * w).sum() / w.sum()).item()}
