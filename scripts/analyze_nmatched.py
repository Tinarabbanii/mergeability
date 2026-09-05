from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.config import load_config
from src.metrics import MetricComputer
from src.pipeline import feature_columns, load_joined
from src.predict import loto_evaluate


def _score(subsets, x, y, cols, tasks, idx, kw):
    s = [subsets[i] for i in idx]
    out = loto_evaluate(s, x[idx], y[idx], cols, tasks, **kw)
    return out["pooled_r"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip", "clip16"])
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--kind", default="data_free", choices=["data_free", "full"])
    args = ap.parse_args()

    cfg = load_config(args.backend)
    mc = MetricComputer(cfg, None)
    df = load_joined(cfg)
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]), lr=float(p["lr"]),
              seed=cfg.seed, n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")), lambda_grid=p.get("lambda_grid"))

    ks = sorted(df.k.unique())
    n_target = int(df[df.k == min(ks)].tasks.nunique())
    print(f"n-matched subsampling  [{cfg.backend}]  matching every k to n={n_target}")

    rows = []
    for method in cfg.merge_methods:
        base = {}
        for k in ks:
            sub = df[(df.k == k) & (df.method == method)].reset_index(drop=True)
            subsets = [tuple(s.split("|")) for s in sub["tasks"]]
            cols = feature_columns(sub, args.kind, mc)
            x = np.nan_to_num(sub[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            y = sub["normalized_accuracy"].to_numpy(float)
            full = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)["pooled_r"]
            base[k] = full

            if len(sub) <= n_target:
                rows.append({"method": method, "k": k, "n_full": len(sub), "n_matched": len(sub),
                             "r_full": full, "median_r": full, "p05": full, "p95": full,
                             "draws_used": 0})
                print(f"  {method:<18} k={k}  n={len(sub):3d}  r={full:+.3f}  (reference)")
                continue

            rng = np.random.default_rng(cfg.seed)
            idxs = [rng.choice(len(sub), n_target, replace=False) for _ in range(args.draws)]
            vals = Parallel(n_jobs=-1)(
                delayed(_score)(subsets, x, y, cols, cfg.task_names, i, kw) for i in idxs)
            vals = np.array([v for v in vals if not np.isnan(v)])
            rows.append({"method": method, "k": k, "n_full": len(sub), "n_matched": n_target,
                         "r_full": full, "median_r": float(np.median(vals)),
                         "p05": float(np.percentile(vals, 5)),
                         "p95": float(np.percentile(vals, 95)),
                         "draws_used": len(vals)})
            print(f"  {method:<18} k={k}  n={len(sub):3d}  r={full:+.3f}   "
                  f"cut to {n_target}: median {np.median(vals):+.3f} "
                  f"[{np.percentile(vals,5):+.3f},{np.percentile(vals,95):+.3f}]  "
                  f"vs k={min(ks)} {base[min(ks)]:+.3f}  ({len(vals)}/{args.draws} usable)")

    out = pd.DataFrame(rows)
    path = cfg.artifact(f"nmatched_{args.kind}.csv")
    out.to_csv(path, index=False)

    print("\n  VERDICT")
    for method in cfg.merge_methods:
        m = out[out.method == method]
        ref = m[m.k == min(ks)]["r_full"]
        if ref.empty:
            continue
        ref = float(ref.iloc[0])
        for _, r in m[m.k > min(ks)].iterrows():
            if r.draws_used == 0:
                continue
            verdict = ("still better" if r.p05 > ref else
                       "explained by sample size" if r.median_r <= ref else "inconclusive")
            print(f"  {method:<18} k={int(r.k)}: {verdict}"
                  f"  (matched median {r.median_r:+.3f} vs k={min(ks)} {ref:+.3f})")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
