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

    print(f"univariate baseline  [{cfg.backend}, {args.kind}]")
    rows = []
    for k in sorted(df.k.unique()):
        for method in cfg.merge_methods:
            sub = df[(df.k == k) & (df.method == method)].reset_index(drop=True)
            if len(sub) < 6:
                continue
            subsets = [tuple(s.split("|")) for s in sub["tasks"]]
            y = sub["normalized_accuracy"].to_numpy(float)
            cols = feature_columns(sub, args.kind, mc)
            x = np.nan_to_num(sub[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)

            multi = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)["pooled_r"]

            best_r, best_c = -np.inf, None
            for j, c in enumerate(cols):
                xi = x[:, [j]]
                r = loto_evaluate(subsets, xi, y, [c], cfg.task_names, **kw)["pooled_r"]
                if not np.isnan(r) and r > best_r:
                    best_r, best_c = r, c

            rows.append({"k": k, "method": method, "n": len(sub), "n_features": len(cols),
                         "multivariate_r": multi, "best_single_r": best_r,
                         "best_single_metric": best_c, "gain": multi - best_r})
            print(f"  k={k} {method:<18} multi={multi:+.3f}  best single={best_r:+.3f} "
                  f"({best_c})  gain={multi-best_r:+.3f}")

    out = pd.DataFrame(rows)
    path = cfg.artifact(f"univariate_{args.kind}.csv")
    out.to_csv(path, index=False)
    w = int((out.gain > 0).sum())
    print(f"\n  the multi-metric model beats the best single metric in {w} of {len(out)} settings")
    print(f"  mean gain: {out.gain.mean():+.3f}")
    print("\n  CAVEAT: the single metric is chosen AFTER seeing which scored best, so its")
    print("  score is optimistically biased. The multivariate score has no such advantage.")
    print("  A negative gain therefore understates the multivariate model.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
