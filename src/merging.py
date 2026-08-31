### Merging methods

from __future__ import annotations
import torch
from .utils import StateDict, sd_clone, sd_mean, sd_sum

def weight_averaging(theta_pre: StateDict, taus: list[StateDict], **kw) -> StateDict:
    merged = sd_clone(theta_pre)
    mean_tau = sd_mean(taus)
    for name in merged:
        merged[name] = merged[name] + mean_tau[name]
    return merged

def task_arithmetic(theta_pre: StateDict, taus: list[StateDict], alpha: float = 0.3, **kw) -> StateDict:
    merged = sd_clone(theta_pre)
    total = sd_sum(taus)
    for name in merged:
        merged[name] = merged[name] + alpha * total[name]
    return merged

def ties(theta_pre: StateDict, taus: list[StateDict], alpha: float = 0.3, density: float = 0.2, **kw) -> StateDict:
    merged = sd_clone(theta_pre)
    layer_names = sorted(merged)
    thresholds = []
    for t in taus:
        v = torch.cat([t[n].float().flatten() for n in layer_names])
        n_keep = max(1, int(round(density * v.numel())))
        thresholds.append(torch.tensor(0.0) if n_keep >= v.numel()
                          else v.abs().kthvalue(v.numel() - n_keep + 1).values)
    thr_vec = torch.stack(thresholds)
    for name in merged:
        stack = torch.stack([t[name].float() for t in taus])  # (k, *shape)
        thr = thr_vec.reshape(-1, *([1] * (stack.dim() - 1)))  # broadcast over the layer
        trimmed = torch.where(stack.abs() >= thr, stack, torch.zeros_like(stack))
### ELECT
        elected = torch.sign(trimmed.sum(dim=0))

        agrees = (torch.sign(trimmed) == elected.unsqueeze(0)) & (trimmed != 0)
        kept = trimmed * agrees
        count = agrees.sum(dim=0).clamp(min=1)
        tau_merged = kept.sum(dim=0) / count
### MERGE
        merged[name] = merged[name] + alpha * tau_merged.to(merged[name].dtype)
    return merged

METHODS = {
    "weight_averaging": weight_averaging,
    "task_arithmetic": task_arithmetic,
    "ties": ties,
}

def merge(theta_pre: StateDict, taus: list[StateDict], method: str, cfg: dict | None = None) -> StateDict:
    if method not in METHODS:
        raise ValueError(f"unknown merge method {method!r}; have {sorted(METHODS)}")
    kwargs = dict((cfg or {}).get(method, {}) or {})
    return METHODS[method](theta_pre, taus, **kwargs)