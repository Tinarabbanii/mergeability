
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class LayerDecomp:
    u_top: torch.Tensor      # Top-k left singular vectors
    v_top: torch.Tensor      # Top-k right singular vectors
    v_bot: torch.Tensor      # Bottom-k right singular vectors
    sv_norm: torch.Tensor    # Normalised top singular values
    shape: tuple[int, int]

def _svd(mat: torch.Tensor):
    try:
        return torch.linalg.svd(mat, full_matrices=False)
    except Exception:  # Fails to converge on real weights
        return torch.linalg.svd(mat + 1e-6 * torch.randn_like(mat), full_matrices=False)

def decompose(mats: list[torch.Tensor], top_k: int = 10, bottom_k: int = 10,
              sv_top: int = 100) -> list[LayerDecomp]: # Results per task
    out: list[LayerDecomp] = []
    for m in mats:
        u, s, vh = _svd(m)
        v = vh.T
        r = min(u.shape[1], v.shape[1])
        kt, kb = min(top_k, r), min(bottom_k, r)
        n = min(sv_top, s.numel())
        out.append(LayerDecomp(
            u_top=u[:, :kt].contiguous(),
            v_top=v[:, :kt].contiguous(),
            v_bot=v[:, -kb:].contiguous(),
            sv_norm=(s[:n] / s[:n].sum().clamp(min=1e-12)).contiguous(),
            shape=tuple(m.shape),
        ))
    return out

def _alignment(a: torch.Tensor, b: torch.Tensor) -> float:
    k = min(a.shape[1], b.shape[1]) # 1 = Coincide
    if k == 0:
        return 0.0
    return ((a[:, :k].T @ b[:, :k]).norm() ** 2 / k).item()

def _interaction(a: torch.Tensor, b: torch.Tensor) -> float:
    k = min(a.shape[1], b.shape[1])
    if k == 0:
        return 0.0
    inter = a[:, :k].T @ b[:, :k] # Mean squared singular value of A^T B
    if inter.numel() == 0:
        return 0.0
    return (torch.linalg.svdvals(inter) ** 2).mean().item()
# how aligned are these spans

def compute_pair(dec_a: list[LayerDecomp], dec_b: list[LayerDecomp]) -> dict[str, float]:
    """All subspace metrics for one pair, averaged across matched layers.

    Takes PRE-DECOMPOSED tasks, so no SVD happens here.
    """
    acc: dict[str, list[float]] = {
        "sv_overlap": [], "left_top": [], "right_top": [],
        "right_bot": [], "interact_top": [], "interact_bot": [],
    }

    for da, db in zip(dec_a, dec_b):
        if da.shape != db.shape:
            continue
        n = min(da.sv_norm.numel(), db.sv_norm.numel())
        acc["sv_overlap"].append(
            F.cosine_similarity(da.sv_norm[:n], db.sv_norm[:n], dim=0).item())
        acc["left_top"].append(_alignment(da.u_top, db.u_top))
        acc["right_top"].append(_alignment(da.v_top, db.v_top))
        acc["right_bot"].append(_alignment(da.v_bot, db.v_bot))
        acc["interact_top"].append(_interaction(da.v_top, db.v_top))
        acc["interact_bot"].append(_interaction(da.v_bot, db.v_bot))
    out: dict[str, float] = {}
    for name, values in acc.items():
        out[f"sub_{name}"] = float(sum(values) / len(values)) if values else 0.0
    return out