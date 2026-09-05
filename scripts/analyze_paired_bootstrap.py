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


def _diff(subsets, xf, xa, y, cf, ca, tasks, idx, kw):
    s = [subsets[i] for i in idx]
    a = loto_evaluate(s, xf[idx], y[idx], cf, tasks, **kw)["pooled_r"]
    b = loto_evaluate(s, xa[idx], y[idx], ca, tasks, **kw)["pooled_r"]
    return a - b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip", "clip16"])
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.backend)
    mc = MetricComputer(cfg, None)
    df = load_joined(cfg)
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]), lr=float(p["lr"]),
              seed=cfg.seed, n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")), lambda_grid=p.get("lambda_grid"))

    print(f"paired bootstrap on (data-free minus full)  [{cfg.backend}, {args.boot} resamples]")
    rows = []
    for k in sorted(df.k.unique()):
        for method in cfg.merge_methods:
            sub = df[(df.k == k) & (df.method == method)].reset_index(drop=True)
            if len(sub) < 6:
                continue
            subsets = [tuple(s.split("|")) for s in sub["tasks"]]
            y = sub["normalized_accuracy"].to_numpy(float)
            cf = feature_columns(sub, "data_free", mc)
            ca = feature_columns(sub, "full", mc)
            xf = np.nan_to_num(sub[cf].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            xa = np.nan_to_num(sub[ca].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)

            obs = (loto_evaluate(subsets, xf, y, cf, cfg.task_names, **kw)["pooled_r"]
                   - loto_evaluate(subsets, xa, y, ca, cfg.task_names, **kw)["pooled_r"])

            rng = np.random.default_rng(cfg.seed)
            idxs = [rng.integers(0, len(sub), len(sub)) for _ in range(args.boot)]
            d = Parallel(n_jobs=-1)(
                delayed(_diff)(subsets, xf, xa, y, cf, ca, cfg.task_names, i, kw) for i in idxs)
            d = np.array([v for v in d if not np.isnan(v)])
            lo, hi = np.percentile(d, [2.5, 97.5])
            crosses = lo <= 0 <= hi
            rows.append({"k": k, "method": method, "n": len(sub), "observed_diff": obs,
                         "ci_lo": lo, "ci_hi": hi, "crosses_zero": bool(crosses),
                         "n_boot": len(d)})
            verdict = "no real difference" if crosses else ("data-free BETTER" if lo > 0 else "full BETTER")
            print(f"  k={k} {method:<18} diff={obs:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  {verdict}")

    out = pd.DataFrame(rows)
    path = cfg.artifact("paired_bootstrap.csv")
    out.to_csv(path, index=False)
    n_tie = int(out.crosses_zero.sum())
    print(f"\n  {n_tie} of {len(out)} comparisons show no significant difference")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
