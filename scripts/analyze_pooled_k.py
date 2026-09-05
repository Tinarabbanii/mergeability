from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.config import load_config
from src.metrics import MetricComputer
from src.pipeline import feature_columns, load_joined
from src.predict import loto_evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip", "clip16"])
    ap.add_argument("--kind", default="data_free", choices=["data_free", "full"])
    args = ap.parse_args()

    cfg = load_config(args.backend)
    mc = MetricComputer(cfg, None)
    df = load_joined(cfg)
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]), lr=float(p["lr"]),
              seed=cfg.seed, n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")), lambda_grid=p.get("lambda_grid"))

    print(f"pooled across k  [{cfg.backend}, {args.kind}]")
    rows = []
    for method in cfg.merge_methods:
        sub = df[df.method == method].reset_index(drop=True)
        if len(sub) < 6:
            continue
        cols = feature_columns(sub, args.kind, mc)
        subsets = [tuple(s.split("|")) for s in sub["tasks"]]
        x = np.nan_to_num(sub[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = sub["normalized_accuracy"].to_numpy(float)

        y_z = y_raw.copy()
        for k in sub.k.unique():
            m = (sub.k == k).to_numpy()
            s = y_raw[m].std()
            y_z[m] = (y_raw[m] - y_raw[m].mean()) / (s if s > 1e-12 else 1.0)

        naive = loto_evaluate(subsets, x, y_raw, cols, cfg.task_names, **kw)["pooled_r"]
        fair = loto_evaluate(subsets, x, y_z, cols, cfg.task_names, **kw)["pooled_r"]

        per_k = {}
        for k in sorted(sub.k.unique()):
            b = sub[sub.k == k].reset_index(drop=True)
            s2 = [tuple(t.split("|")) for t in b["tasks"]]
            x2 = np.nan_to_num(b[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            per_k[k] = loto_evaluate(s2, x2, b["normalized_accuracy"].to_numpy(float),
                                     cols, cfg.task_names, **kw)["pooled_r"]

        row = {"method": method, "n_pooled": len(sub), "pooled_raw_r": naive,
               "pooled_within_k_z_r": fair}
        row.update({f"k{k}_r": v for k, v in per_k.items()})
        rows.append(row)
        print(f"  {method:<18} pooled(z within k)={fair:+.3f}   naive pooled={naive:+.3f}   "
              f"per-k " + " ".join(f"k{k}={v:+.3f}" for k, v in per_k.items()))

    out = pd.DataFrame(rows)
    path = cfg.artifact(f"pooled_k_{args.kind}.csv")
    out.to_csv(path, index=False)
    print("\n  'naive pooled' does not correct for merges getting worse as k grows, so it")
    print("  can score well by detecting subset SIZE. Report the within-k standardised one.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
