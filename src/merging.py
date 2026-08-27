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

    for name in merged:
        stack = torch.stack([t[name].float() for t in taus])  # (k, ...)
### TRIM
        flat = stack.reshape(stack.shape[0], -1)
        n_keep = max(1, int(round(density * flat.shape[1])))
        if n_keep < flat.shape[1]:
            thresh = flat.abs().kthvalue(flat.shape[1] - n_keep + 1, dim=1, keepdim=True).values
            flat = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
        trimmed = flat.reshape(stack.shape)
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