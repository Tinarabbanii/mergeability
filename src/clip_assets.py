from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from .config import ROOT

### Encoder
class ClipImageEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: str | None = None):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("the clip backend needs open_clip:  pip install open_clip_torch") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model
        self.preprocess = preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)

def build_encoder(cfg, device) -> ClipImageEncoder:
    return ClipImageEncoder(cfg.tasks[cfg.backend]["model"]).to(device)

### Head per task
def build_head(task: str, ckpt_dir: Path, device) -> nn.Linear:
    path = ckpt_dir / f"head_{task}.pt"
    if path.exists():
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        weight = sd.get("weight", sd.get("fc.weight"))
        if weight is None:
            raise KeyError(f"{path} has no recognisable weight tensor; keys={list(sd)[:5]}")
        bias = sd.get("bias", sd.get("fc.bias"))
        head = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        head.weight.data.copy_(weight.float())
        if bias is not None:
            head.bias.data.copy_(bias.float())
    else:
        raise FileNotFoundError(
            f"missing classification head: {path}\n"
            f"The heads are zero-shot classifiers built from CLIP text prompts.\n"
            f"Export them from the task_vectors repo alongside the encoders, then\n"
            f"convert with scripts/download_checkpoints.py --convert.")
    for p in head.parameters():
        p.requires_grad_(False)
    return head.to(device)

### Data
TORCHVISION_TASKS = {
    "MNIST": ("MNIST", dict(train=False, download=True)),
    "SVHN": ("SVHN", dict(split="test", download=True)),
    "DTD": ("DTD", dict(split="test", download=True)),
    "EuroSAT": ("EuroSAT", dict(download=True)),
    "GTSRB": ("GTSRB", dict(split="test", download=True)),}

HF_TASKS = {
    "RESISC45": "hf://datasets/tanganke/resisc45/data/test-*.parquet",
    "Cars":     "hf://datasets/tanganke/stanford_cars/data/test-*.parquet",
    "SUN397":   "hf://datasets/tanganke/sun397/data/test-*.parquet",}
class _HFDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds, self.transform = ds, transform
    def __len__(self) -> int:
        return len(self.ds)
    def __getitem__(self, i):
        row = self.ds[i]
        return self.transform(row["image"].convert("RGB")), row["label"]

def build_loader(cfg, task: str, preprocess, batch_size: int = 64):
    import torchvision.datasets as tvd
    n = int(cfg.eval["eval_samples_per_task"])
    configured = cfg.tasks[cfg.backend].get("data_root", "data")
    data_root = Path(configured)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    data_root.mkdir(parents=True, exist_ok=True)
    if task in HF_TASKS:
        local = data_root / "hf" / task
        if local.exists():
            from datasets import load_from_disk
            hf = load_from_disk(str(local))
        else:
            from datasets import load_dataset
            hf = load_dataset("parquet", data_files=HF_TASKS[task], split="train")
        ds = _HFDataset(hf, preprocess)
    elif task in TORCHVISION_TASKS:
        name, kwargs = TORCHVISION_TASKS[task]
        ds = getattr(tvd, name)(root=str(data_root), transform=preprocess, **kwargs)
    else:
        raise NotImplementedError(
            f"no loader for {task!r}. torchvision: {sorted(TORCHVISION_TASKS)}; "
            f"HuggingFace: {sorted(HF_TASKS)}.")
    if len(ds) > n:
        g = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        ds = Subset(ds, idx)
    import sys
    workers = 0 if sys.platform == "darwin" else 2
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

def build_eval_assets(cfg, task: str, ckpt_dir: Path, device, encoder):
    head = build_head(task, ckpt_dir, device)
    loader = build_loader(cfg, task, encoder.preprocess)
# Check
    with torch.no_grad():
        sample = next(iter(loader))[0][:1].to(device)
        feature_dim = encoder(sample).shape[-1]
    if feature_dim != head.in_features:
        raise RuntimeError(
            f"{task}: encoder outputs {feature_dim} features but head_{task}.pt "
            f"expects {head.in_features}. Wrong CLIP variant?")
    return head, loader
