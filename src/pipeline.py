from __future__ import annotations
import time
import pandas as pd
from .config import Config
from .metrics import MetricComputer
from .merging import merge
from .utils import subsets

def expert_accuracies(cfg: Config, backend) -> dict[str, float]:
    out: dict[str, float] = {}
    for task in cfg.task_names:
        out[task] = backend.evaluate(backend.finetuned(task), task)
    return out

def enumerate_subsets(cfg: Config) -> list[tuple[str, ...]]:
    all_subsets: list[tuple[str, ...]] = []
    for k in cfg.k_values:
        if k > len(cfg.task_names):
            continue
        all_subsets += subsets(
            cfg.task_names, k,
            limit=int(cfg.eval["max_subsets_per_k"]),
            seed=cfg.seed,
        )
    return all_subsets

def build_metrics(cfg: Config, backend, verbose: bool = True) -> pd.DataFrame:
    mc = MetricComputer(cfg, backend)
    todo = enumerate_subsets(cfg)
    rows = []
    t0 = time.time()

    for i, tasks in enumerate(todo, 1):
        row: dict = {"tasks": "|".join(tasks), "k": len(tasks)}
        row.update(mc.compute(tasks))
        rows.append(row)
        if verbose and (i % max(1, len(todo) // 10) == 0 or i == len(todo)):
            print(f"    metrics {i}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    path = cfg.artifact("metrics.csv")
    df.to_csv(path, index=False)
    if verbose:
        print(f"  -> {path}  ({len(df)} subsets, {len(df.columns) - 2} metric columns)")
    return df


def build_results(cfg: Config, backend, verbose: bool = True) -> pd.DataFrame:
    mc = MetricComputer(cfg, backend)
    theta_pre = backend.pretrained()
    experts = expert_accuracies(cfg, backend)
    if verbose:
        print("  expert accuracies: " +
              "  ".join(f"{t}={a:.3f}" for t, a in experts.items()))

    todo = enumerate_subsets(cfg)
    rows = []
    t0 = time.time()
    total = len(todo) * len(cfg.merge_methods)
    done = 0

    for tasks in todo:
        taus = [mc.task_vector(t) for t in tasks]
        for method in cfg.merge_methods:
            merged = merge(theta_pre, taus, method, cfg.merging)

            per_task = {}
            ratios = []
            for t in tasks:
                acc = backend.evaluate(merged, t)
                per_task[t] = acc
                ratios.append(acc / experts[t] if experts[t] > 0 else 0.0)

            rows.append({
                "tasks": "|".join(tasks),
                "k": len(tasks),
                "method": method,
                "normalized_accuracy": sum(ratios) / len(ratios),
                "mean_raw_accuracy": sum(per_task.values()) / len(per_task),
            })
            done += 1
            if verbose and (done % max(1, total // 10) == 0 or done == total):
                print(f"    merges {done}/{total}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = cfg.artifact("results.csv")
    df.to_csv(path, index=False)
    exp = pd.DataFrame([{"task": t, "expert_accuracy": a} for t, a in experts.items()])
    exp.to_csv(cfg.artifact("experts.csv"), index=False)
    if verbose:
        print(f"  -> {path}  ({len(df)} merges)")
    return df

def load_joined(cfg: Config) -> pd.DataFrame:
    m = pd.read_csv(cfg.artifact("metrics.csv"))
    r = pd.read_csv(cfg.artifact("results.csv"))
    return r.merge(m.drop(columns=["k"]), on="tasks", how="left")


def feature_columns(df: pd.DataFrame, kind: str, mc: MetricComputer) -> list[str]:
    if kind == "data_free":
        names = mc.data_free_metric_names()
    elif kind == "full":
        names = mc.all_metric_names()
    else:
        raise ValueError(f"unknown feature kind {kind!r}")
    return [c for c in names if c in df.columns]