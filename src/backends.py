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

### CLIP bechmark
class ClipBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        spec = cfg.tasks[cfg.backend]
        self.task_names = list(spec["tasks"])
        self.device = get_device()
        self.ckpt_dir = (__import__("pathlib").Path(cfg.tasks[cfg.backend]["checkpoint_dir"]))
        if not self.ckpt_dir.is_absolute():
            from .config import ROOT
            self.ckpt_dir = ROOT / self.ckpt_dir
        self._cache: dict[str, StateDict] = {}
        self._eval_cache: dict[str, tuple] = {}
        self._encoder = None

    def _load(self, filename: str) -> StateDict:
        if filename in self._cache:
            return self._cache[filename]
        path = self.ckpt_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"missing checkpoint {path}\n"
                f"run: python scripts/download_checkpoints.py")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
        sd = {k: v.float() for k, v in sd.items() if torch.is_floating_point(v)}
        prefix = self.cfg.tasks[self.cfg.backend].get("param_prefix", "")
        if prefix:
            sd = {k: v for k, v in sd.items() if k.startswith(prefix)}
            if not sd:
                raise RuntimeError(
                    f"no parameters matched prefix {prefix!r} in {path.name}; "
                    f"check configs/tasks.yaml:clip.param_prefix")
        self._cache[filename] = sd
        return sd

    def pretrained(self) -> StateDict:
        return self._load("zeroshot.pt")

    def finetuned(self, task: str) -> StateDict:
        return self._load(f"{task}.pt")

    def _encoder_once(self):
        if self._encoder is None:
            from .clip_assets import build_encoder
            self._encoder = build_encoder(self.cfg, self.device)
        return self._encoder

    def _eval_assets(self, task: str):
        encoder = self._encoder_once()
        if task not in self._eval_cache:
            from .clip_assets import build_eval_assets
            self._eval_cache[task] = build_eval_assets(
                self.cfg, task, self.ckpt_dir, self.device, encoder)
        head, loader = self._eval_cache[task]
        return encoder, head, loader

    def _load_into(self, encoder, encoder_sd: StateDict) -> None:
        result = encoder.load_state_dict(encoder_sd, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"{len(result.unexpected_keys)} checkpoint keys have no match in the "
                f"encoder (e.g. {result.unexpected_keys[:3]}). Nothing was loaded.")

    @torch.no_grad()
    def evaluate(self, encoder_sd: StateDict, task: str) -> float:
        encoder, head, loader = self._eval_assets(task)
        self._load_into(encoder, encoder_sd)
        encoder.eval().to(self.device)
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = head(encoder(x))
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)

    @torch.no_grad()
    def activations(self, encoder_sd: StateDict, task: str) -> torch.Tensor:
        encoder, _, loader = self._eval_assets(task)
        self._load_into(encoder, encoder_sd)
        encoder.eval().to(self.device)
        n = int(self.cfg.metrics["calibration"]["samples_per_task"])
        feats = []
        seen = 0
        for x, _ in loader:
            x = x.to(self.device)
            feats.append(encoder(x).float().cpu())
            seen += x.shape[0]
            if seen >= n:
                break
        return torch.cat(feats)[:n].mean(dim=0)

    def gradients(self, encoder_sd: StateDict, task: str) -> StateDict:
        encoder, head, loader = self._eval_assets(task)
        self._load_into(encoder, encoder_sd)
        encoder.train().to(self.device)
        n = int(self.cfg.metrics["calibration"]["samples_per_task"])
        xs, ys = [], []
        seen = 0
        for x, y in loader:
            xs.append(x); ys.append(y); seen += x.shape[0]
            if seen >= n:
                break
        x = torch.cat(xs)[:n].to(self.device)
        y = torch.cat(ys)[:n].to(self.device)
        loss = F.cross_entropy(head(encoder(x)), y)
        encoder.zero_grad()
        loss.backward()
        prefix = self.cfg.tasks[self.cfg.backend].get("param_prefix", "")
        out: StateDict = {}
        for name, p in encoder.named_parameters():
            if prefix and not name.startswith(prefix):
                continue
            out[name] = (p.grad.detach().float().cpu().clone()
                         if p.grad is not None else torch.zeros_like(p).cpu())
        return out

### Get backend
def get_backend(cfg: Config) -> Backend:
    if cfg.backend == "synthetic":
        return SyntheticBackend(cfg)
    if cfg.backend.startswith("clip"):
        return ClipBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")
