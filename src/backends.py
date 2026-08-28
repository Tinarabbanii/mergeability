from __future__ import annotations
from typing import Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config
from .utils import StateDict, get_device, set_seed

class Backend(Protocol):
    task_names: list[str]
    def pretrained(self) -> StateDict: ...
    def finetuned(self, task: str) -> StateDict: ...
    def evaluate(self, encoder_sd: StateDict, task: str) -> float: ...
    def activations(self, encoder_sd: StateDict, task: str) -> torch.Tensor: ...
    def gradients(self, encoder_sd: StateDict, task: str) -> StateDict: ...

### Synthetic
# Synthetic CLIP image encoder
class _Encoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class SyntheticBackend:
    def __init__(self, cfg: Config):
        spec = cfg.tasks["synthetic"]
        self.cfg = cfg
        self.spec = spec
        self.task_names = list(spec["tasks"])
        self.device = torch.device("cpu") # cpu works fine

        self.d_in = int(spec["d_in"])
        self.d_hidden = int(spec["d_hidden"])
        self.n_classes = int(spec["n_classes"])

        set_seed(cfg.seed)
        g = torch.Generator().manual_seed(cfg.seed)
        self._shared = torch.randn(self.n_classes, self.d_in, generator=g)

        self._data: dict[str, dict[str, torch.Tensor]] = {}
        for task in self.task_names:
            self._data[task] = self._make_task(task, g)

        self._heads: dict[str, nn.Linear] = {}
        self._pre: StateDict | None = None
        self._ft: dict[str, StateDict] = {}

    def _make_task(self, task: str, g: torch.Generator) -> dict[str, torch.Tensor]:
        spec = self.spec
        overlap = float(spec["task_overlap"][task])
        private = torch.randn(self.n_classes, self.d_in, generator=g)
        centers = overlap * self._shared + (1.0 - overlap) * private

        def draw(n: int) -> tuple[torch.Tensor, torch.Tensor]:
            y = torch.randint(0, self.n_classes, (n,), generator=g)
            x = centers[y] + torch.randn(n, self.d_in, generator=g) * float(spec.get("noise", 1.0))
            return x, y

        xtr, ytr = draw(int(spec["n_train"]))
        xte, yte = draw(int(spec["n_test"]))
        return {"x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte}

    def _new_model(self, encoder_sd: StateDict | None, task: str) -> tuple[_Encoder, nn.Linear]:
        enc = _Encoder(self.d_in, self.d_hidden)
        if encoder_sd is not None:
            enc.load_state_dict(encoder_sd)
        return enc, self._heads[task]
# Simulation of one pretrained backbone, N fine-tuned descendants
    def _ensure_trained(self) -> None:
        if self._pre is not None:
            return

        set_seed(self.cfg.seed)
        spec = self.spec

# Pooled
        enc = _Encoder(self.d_in, self.d_hidden)
        pre_head = nn.Linear(self.d_hidden, self.n_classes)
        x_all = torch.cat([self._data[t]["x_train"] for t in self.task_names])
        y_all = torch.cat([self._data[t]["y_train"] for t in self.task_names])

        opt = torch.optim.SGD(list(enc.parameters()) + list(pre_head.parameters()), lr=0.05)
        for _ in range(int(spec["pretrain_steps"])):
            opt.zero_grad()
            loss = F.cross_entropy(pre_head(enc(x_all)), y_all)
            loss.backward()
            opt.step()

        self._pre = {k: v.detach().clone() for k, v in enc.state_dict().items()}

        for task in self.task_names:
            head = nn.Linear(self.d_hidden, self.n_classes)
            head.load_state_dict(pre_head.state_dict())
            for p in head.parameters():
                p.requires_grad_(False)
            self._heads[task] = head

        for task in self.task_names:
            enc_t = _Encoder(self.d_in, self.d_hidden)
            enc_t.load_state_dict(self._pre)
            head = self._heads[task]
            d = self._data[task]
            opt = torch.optim.SGD(enc_t.parameters(), lr=float(spec["finetune_lr"]))
            for _ in range(int(spec["finetune_steps"])):
                opt.zero_grad()
                loss = F.cross_entropy(head(enc_t(d["x_train"])), d["y_train"])
                loss.backward()
                opt.step()
            self._ft[task] = {k: v.detach().clone() for k, v in enc_t.state_dict().items()}

    def pretrained(self) -> StateDict:
        self._ensure_trained()
        assert self._pre is not None
        return self._pre

    def finetuned(self, task: str) -> StateDict:
        self._ensure_trained()
        return self._ft[task]

    @torch.no_grad()
    def evaluate(self, encoder_sd: StateDict, task: str) -> float:
        self._ensure_trained()
        enc, head = self._new_model(encoder_sd, task)
        enc.eval()
        d = self._data[task]
        preds = head(enc(d["x_test"])).argmax(dim=1)
        return (preds == d["y_test"]).float().mean().item()

    @torch.no_grad()
    def activations(self, encoder_sd: StateDict, task: str) -> torch.Tensor:
        self._ensure_trained()
        enc, _ = self._new_model(encoder_sd, task)
        enc.eval()
        n = int(self.cfg.metrics["calibration"]["samples_per_task"])
        x = self._data[task]["x_train"][:n]
        return enc(x).mean(dim=0)

    def gradients(self, encoder_sd: StateDict, task: str) -> StateDict:
        self._ensure_trained()
        enc, head = self._new_model(encoder_sd, task)
        enc.train()
        n = int(self.cfg.metrics["calibration"]["samples_per_task"])
        d = self._data[task]
        loss = F.cross_entropy(head(enc(d["x_train"][:n])), d["y_train"][:n])
        enc.zero_grad()
        loss.backward()
        out: StateDict = {}
        for name, p in enc.named_parameters():
            out[name] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        return out
### Get backend
def get_backend(cfg: Config) -> Backend:
    if cfg.backend == "synthetic":
        return SyntheticBackend(cfg)
    if cfg.backend == "clip":
        return ClipBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")