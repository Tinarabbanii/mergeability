from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
CHECKPOINTS = ROOT / "checkpoints"

def _load(name: str) -> dict[str, Any]:
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)

@dataclass
class Config:

    tasks: dict[str, Any] = field(default_factory=lambda: _load("tasks"))
    metrics: dict[str, Any] = field(default_factory=lambda: _load("metrics"))
    merging: dict[str, Any] = field(default_factory=lambda: _load("merging"))
    eval: dict[str, Any] = field(default_factory=lambda: _load("eval"))

    backend: str = "synthetic"  # "synthetic" | "clip"

    @property
    def task_names(self) -> list[str]:
        return list(self.tasks[self.backend]["tasks"])

    @property
    def seed(self) -> int:
        return int(self.eval["seed"])

    @property
    def merge_methods(self) -> list[str]:
        return list(self.merging["methods"])

    @property
    def k_values(self) -> list[int]:
        return list(self.eval["k_values"])

    def artifact(self, name: str) -> Path:
        d = ARTIFACTS / self.backend
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def figure(self, name: str) -> Path:
        FIGURES.mkdir(parents=True, exist_ok=True)
        return FIGURES / name


def load_config(backend: str = "synthetic") -> Config:
    if backend not in ("synthetic", "clip"):
        raise ValueError(f"unknown backend {backend!r}; use 'synthetic' or 'clip'")
    return Config(backend=backend)
