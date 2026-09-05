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

            oracle_r, oracle_c = -np.inf, None
            for j, c in enumerate(cols):
                r = loto_evaluate(subsets, x[:, [j]], y, [c], cfg.task_names, **kw)["pooled_r"]
                if not np.isnan(r) and r > oracle_r:
                    oracle_r, oracle_c = r, c

            held = np.full(len(y), np.nan)
            picked = []
            for t in cfg.task_names:
                va = np.array([i for i, ss in enumerate(subsets) if t in ss])
                tr = np.array([i for i, ss in enumerate(subsets) if t not in ss])
                if len(va) < 1 or len(tr) < 3:
                    continue
                scores = []
                for j in range(x.shape[1]):
                    col = x[tr, j]
                    scores.append(0.0 if np.std(col) < 1e-12 or np.std(y[tr]) < 1e-12
                                  else abs(np.corrcoef(col, y[tr])[0, 1]))
                j_star = int(np.nanargmax(scores))
                sign = np.sign(np.corrcoef(x[tr, j_star], y[tr])[0, 1]) or 1.0
                held[va] = sign * x[va, j_star]
                picked.append(cols[j_star])
            ok = ~np.isnan(held)
            nested_r = (float(np.corrcoef(held[ok], y[ok])[0, 1])
                        if ok.sum() >= 3 and np.std(held[ok]) > 1e-12 else float("nan"))
            from collections import Counter
            mode_c = Counter(picked).most_common(1)[0][0] if picked else None

            rows.append({"k": k, "method": method, "n": len(sub), "n_features": len(cols),
                         "multivariate_r": multi,
                         "best_single_nested_r": nested_r, "nested_metric_mode": mode_c,
                         "best_single_oracle_r": oracle_r, "oracle_metric": oracle_c,
                         "gain": multi - nested_r,
                         "selection_bias": oracle_r - nested_r})
            print(f"  k={k} {method:<18} multi={multi:+.3f}  single(nested)={nested_r:+.3f} "
                  f"({mode_c})  gain={multi-nested_r:+.3f}   [oracle {oracle_r:+.3f}, "
                  f"bias {oracle_r-nested_r:+.3f}]")

    out = pd.DataFrame(rows)
    path = cfg.artifact(f"univariate_{args.kind}.csv")
    out.to_csv(path, index=False)
    w = int((out.gain > 0).sum())
    print(f"\n  the multi-metric model beats the best single metric in {w} of {len(out)} settings")
    print(f"  mean gain: {out.gain.mean():+.3f}")
    print("\n  'nested' picks the metric inside each fold using only the training tasks,")
    print("  so it is an honest held-out score. 'oracle' picks after seeing all results;")
    print("  the difference between them is the selection bias.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
