from __future__ import annotations
from itertools import combinations
import torch

from ..config import Config
from ..utils import StateDict, sd_flatten, sd_matrices, sd_sub
from . import functional, geometry, rank, subspace

AGG_MIN = "agg_min_"
AGG_MAX = "agg_max_"

class MetricComputer:
    def __init__(self, cfg: Config, backend) -> None:
        self.cfg = cfg
        self.backend = backend
        self.families = cfg.metrics["families"]
        self.sub_cfg = cfg.metrics["subspace"]
        self.rank_eps = float(cfg.metrics["rank"]["eps"])

        self._theta_pre: StateDict | None = None
        self._tv: dict[str, StateDict] = {}
        self._flat: dict[str, torch.Tensor] = {}
        self._mats: dict[str, list[torch.Tensor]] = {}
        self._act: dict[str, torch.Tensor] = {}
        self._pair_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._grad: dict[str, torch.Tensor] = {}
        self._per_layer: dict[str, dict[str, torch.Tensor]] = {}
        self._decomp: dict[str, list] = {}

    def _enabled(self, name: str) -> bool:
        spec = self.families.get(name)
        return bool(spec and spec.get("enabled", True))

### Data-free families
    def data_free_metric_names(self) -> list[str]:
        names: list[str] = []
        if self._enabled("geometry"):
            names += ["tv_cosine", "tv_l2", "tv_dot", "tv_norm_ratio", "tv_norm_mean"]
        if self._enabled("rank"):
            names += ["eff_rank_global", "eff_rank_layerwise"]
        if self._enabled("subspace"):
            # sub_interact_* are excluded from the predictor. In THIS implementation
            # _interaction() and _alignment() come out numerically equivalent
            # (r = 0.998 across all task pairs, CLIP ViT-B/32), so including both
            # spends two of only ~21 degrees of freedom on one piece of information.
            # Whether that equivalence reflects the paper's intent or a misreading
            # of section 3.3 is an open question, put to the authors. The columns
            # remain in metrics.csv either way, so nothing needs recomputing if the
            # answer is that the interaction matrix should be formed differently.
            names += ["sub_sv_overlap", "sub_left_top", "sub_right_top",
                      "sub_right_bot"]
        return names
### Data-dependent
    def data_dependent_metric_names(self) -> list[str]:
        names: list[str] = []
        if self._enabled("activation"):
            names += ["act_l2", "act_cosine", "act_magnitude_ratio", "act_dot"]
        if self._enabled("gradient"):
            names += ["grad_cosine", "grad_l2", "grad_dot"]
        return names

    def all_metric_names(self) -> list[str]:
        return self.data_free_metric_names() + self.data_dependent_metric_names()

    def family_of(self, metric: str) -> str:
        if metric.startswith("tv_"):
            return "geometry"
        if metric.startswith("eff_rank"):
            return "rank"
        if metric.startswith("sub_"):
            return "subspace"
        if metric.startswith("act_"):
            return "activation"
        if metric.startswith("grad_"):
            return "gradient"
        return "other"

    def task_vector(self, task: str) -> StateDict:
        """tau_i = theta_i - theta_0.  Equation 1 of Task Arithmetic."""
        if self._theta_pre is None:
            self._theta_pre = self.backend.pretrained()
        if task not in self._tv:
            self._tv[task] = sd_sub(self.backend.finetuned(task), self._theta_pre)
        return self._tv[task]

    def _flat_tv(self, task: str) -> torch.Tensor:
        if task not in self._flat:
            self._flat[task] = sd_flatten(self.task_vector(task))
        return self._flat[task]

    def _mats_tv(self, task: str) -> list[torch.Tensor]:
        if task not in self._mats:
            self._mats[task] = sd_matrices(
                self.task_vector(task), int(self.sub_cfg.get("max_matrices", 0))
            )
        return self._mats[task]

    def _decomp_tv(self, task: str) -> list:
        if task not in self._decomp:
            self._decomp[task] = subspace.decompose(
                self._mats_tv(task),
                top_k=int(self.sub_cfg["top_k"]),
                bottom_k=int(self.sub_cfg["bottom_k"]),
                sv_top=int(self.sub_cfg["singular_value_top"]),)
        return self._decomp[task]
    
    def _per_layer_tv(self, task: str) -> dict[str, torch.Tensor]:
        if task not in self._per_layer:
            tv = self.task_vector(task)
            self._per_layer[task] = {k: v.flatten().float() for k, v in tv.items()}
        return self._per_layer[task]

    def _activation(self, task: str) -> torch.Tensor:
        if task not in self._act:
            self._act[task] = self.backend.activations(
                self.backend.finetuned(task), task
            ).flatten().float()
        return self._act[task]

    def _gradient(self, task: str) -> torch.Tensor:
        if task not in self._grad:
            g = self.backend.gradients(self.backend.finetuned(task), task)
            self._grad[task] = sd_flatten(g)
        return self._grad[task]

    def _pairwise(self, a: str, b: str) -> dict[str, float]:
### Pairwise-defined metrics for pairs
        key = (a, b) if a <= b else (b, a)
        if key in self._pair_cache:
            return self._pair_cache[key]
        out: dict[str, float] = {}
        if self._enabled("geometry"):
            out.update(geometry.compute_pair(self._flat_tv(a), self._flat_tv(b)))
        if self._enabled("subspace"):
            out.update(subspace.compute_pair(self._decomp_tv(a), self._decomp_tv(b)))
        if self._enabled("activation"):
            out.update(functional.activation_pair(self._activation(a), self._activation(b)))
        if self._enabled("gradient"):
            out.update(functional.gradient_pair(self._gradient(a), self._gradient(b)))
        self._pair_cache[key] = out
        return out

    def compute(self, tasks: tuple[str, ...]) -> dict[str, float]:
### k-ary metric
        if len(tasks) < 2:
            raise ValueError("a subset needs at least 2 tasks")
### Pairwise-defined metrics
        per_pair = [self._pairwise(a, b) for a, b in combinations(tasks, 2)]
        keys = sorted(per_pair[0])
        out: dict[str, float] = {}
        for key in keys:
            vals = [p[key] for p in per_pair]
            out[key] = sum(vals) / len(vals)
            out[AGG_MIN + key] = min(vals)
            out[AGG_MAX + key] = max(vals)
### k-ary metrics
        if self._enabled("rank"):
            flats = [self._flat_tv(t) for t in tasks]
            out.update(rank.compute_subset(flats, self.rank_eps))
            per_layer = [self._per_layer_tv(t) for t in tasks]
            out.update(rank.compute_subset_layerwise(per_layer, self.rank_eps))
        return out
