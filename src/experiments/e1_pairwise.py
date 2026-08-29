from __future__ import annotations
import pandas as pd
from ..config import Config
from ..pipeline import build_metrics, build_results

def run(cfg: Config, backend, skip_metrics: bool = False,
        skip_merges: bool = False) -> pd.DataFrame:
    print(f"e1: ground truth  [backend={cfg.backend}]")
    print(f"  tasks: {cfg.task_names}")
    print(f"  k values: {cfg.k_values}   methods: {cfg.merge_methods}")

    if not skip_metrics:
        print("\n  computing metrics...")
        build_metrics(cfg, backend)
    if not skip_merges:
        print("\n  merging and evaluating...")
        build_results(cfg, backend)

    res = pd.read_csv(cfg.artifact("results.csv"))
    print("\n  post-merge normalised accuracy by k and method:")
    summary = (res.groupby(["k", "method"])["normalized_accuracy"]
                  .agg(["mean", "std", "min", "max", "count"]).round(4))
    print(summary.to_string())

    spread = res[res.k == 2]["normalized_accuracy"]
    if len(spread) > 1:
        print(f"\n  pairwise spread: {spread.min():.3f} to {spread.max():.3f} "
              f"(std {spread.std():.3f})")
        if spread.std() < 0.01:
            print("  WARNING: almost no variation in mergeability. There is "
                  "nothing to predict -- adjust task similarity or merge alpha.")
    return res
