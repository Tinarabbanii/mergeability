from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.config import load_config
from src.backends import get_backend
from src.merging import merge
from src.metrics import MetricComputer
from src.pipeline import expert_accuracies
from src.utils import subsets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--densities", default="0.05,0.1,0.2,0.4,0.8,1.0")
    args = ap.parse_args()

    cfg = load_config(args.backend)
    backend = get_backend(cfg)
    mc = MetricComputer(cfg, backend)
    theta_pre = backend.pretrained()
    experts = expert_accuracies(cfg, backend)
    densities = [float(d) for d in args.densities.split(",")]
    base = dict(cfg.merging.get("ties", {}) or {})

    subs = subsets(cfg.task_names, args.k,
                   limit=int(cfg.eval["max_subsets_per_k"]), seed=cfg.seed)
    print(f"density sweep  [backend={cfg.backend}, k={args.k}]")
    print(f"  densities: {densities}   (config default = {base.get('density')})")
    print(f"  {len(subs)} subsets x {len(densities)} densities = "
          f"{len(subs) * len(densities)} merges, "
          f"{len(subs) * len(densities) * args.k} evaluations")

    rows, t0 = [], time.time()
    for d in densities:
        for i, tasks in enumerate(subs, 1):
            taus = [mc.task_vector(t) for t in tasks]
            params = dict(base); params["density"] = d
            merged = merge(theta_pre, taus, "ties", {"ties": params})
            accs = {t: backend.evaluate(merged, t) for t in tasks}
            rows.append({
                "density": d, "tasks": "|".join(tasks), "k": args.k, "method": "ties",
                "normalized_accuracy": sum(accs[t] / experts[t] for t in tasks) / len(tasks),
                "mean_raw_accuracy": sum(accs.values()) / len(accs),
            })
            if i % 7 == 0:
                print(f"    density={d}  {i}/{len(subs)}  ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    path = cfg.artifact("density_sweep.csv")
    out.to_csv(path, index=False)
    print("\n  post-merge normalised accuracy by density:")
    print(out.groupby("density").normalized_accuracy
             .agg(["mean", "std", "min", "max"]).round(4).to_string())
    print(f"\n  -> {path}")
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
