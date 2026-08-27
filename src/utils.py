from __future__ import annotations
import random
from typing import Iterable
from itertools import combinations
import numpy as np
import torch

StateDict = dict[str, torch.Tensor]
### Set up
# numpy for the bootstrap, torch for model init, random for subset sampling
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

### State_dict algebra
def sd_sub(a: StateDict, b: StateDict)-> StateDict: # Extracts task vectors
    out: StateDict = {}
    for name in a:
        out[name] = a[name] - b[name]
    return out

def sd_add(a: StateDict, b: StateDict) -> StateDict:
    out: StateDict = {}
    for name in a:
        out[name] = a[name] + b[name]
    return out

def sd_scale(a: StateDict, factor: float) -> StateDict:
    out: StateDict = {}
    for name in a:
        out[name] = a[name] * factor
    return out

def sd_sum(dicts: list[StateDict]) -> StateDict: # Merging k models
    if not dicts:
        raise ValueError("sd_sum got an empty list")
    out: StateDict = {}
    for name in dicts[0]:
        total = torch.zeros_like(dicts[0][name]) # Tensor of 0s with same shape as our task vec
        for d in dicts:
            total = total total + d[name]
        out[name] = total
    return out

def sd_mean(dicts: list[StateDict]) -> StateDict:
    if not dicts:
        raise ValueError("sd_mean got an empty list")
    summed = sd_sum(dicts)
    return sd_scale(summed, 1 / len(dicts))

def sd_clone(a: StateDict) -> StateDict:
    out: StateDict = {}
    for name in a:
        out[name] = a[name].clone()
    return out

def sd_flatten(a: StateDict) -> torch.Tensor:
    pieces = []
    for name in sorted(a):
        pieces.append(a[name].flatten().float())
    return torch.cat(pieces)

def sd_matrices(a: StateDict, max_count: int = 0) -> list[torch.Tensor]:
    mats = []
    for name in sorted(a):
        t = a[name]
        if t.dim() == 2 and min(t.shape) >= 2:
            mats.append(t.float())
    if max_count and len(mats) > max_count:
        mats.sort(key=lambda m: m.numel(), reverse=True)
        mats = mats[:max_count]
    return mats

def subsets(items: Iterable[str], k: int, limit: int = 0, seed: int = 0) -> list[tuple[str, ...]]:
    all_subsets = [tuple(c) for c in combinations(sorted(items), k)]
    if limit and len(all_subsets) > limit:
        rng = random.Random(seed)
        all_subsets = sorted(rng.sample(all_subsets, limit))
    return all_subsets