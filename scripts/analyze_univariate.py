from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.config import load_config
from src.metrics import MetricComputer
from src.pipeline import feature_columns, load_joined
from src.predict import loto_evaluate, minmax_apply, minmax_fit


def _single_metric_r(subsets, x, y, tasks, fixed_j=None):
    held_sum = np.zeros(len(y), dtype=float)
    held_cnt = np.zeros(len(y), dtype=float)
    picked = []
    for t in tasks:
        va = np.array([i for i, ss in enumerate(subsets) if t in ss])
        tr = np.array([i for i, ss in enumerate(subsets) if t not in ss])
        if len(va) < 2 or len(tr) < 3:
            continue
        lo, hi = minmax_fit(x[tr])
        xtr = minmax_apply(x[tr], lo, hi)
        xva = minmax_apply(x[va], lo, hi)
        if fixed_j is None:
            scores = []
            for j in range(xtr.shape[1]):
                col = xtr[:, j]
                scores.append(0.0 if np.std(col) < 1e-12 or np.std(y[tr]) < 1e-12
                              else abs(np.corrcoef(col, y[tr])[0, 1]))
            j_star = int(np.nanargmax(scores))
        else:
            j_star = fixed_j
        if np.std(xtr[:, j_star]) < 1e-12 or np.std(y[tr]) < 1e-12:
            continue
        sign = np.sign(np.corrcoef(xtr[:, j_star], y[tr])[0, 1]) or 1.0
        held_sum[va] += sign * xva[:, j_star]
        held_cnt[va] += 1
        picked.append(j_star)
    ok = held_cnt > 0
    if ok.sum() < 3:
        return float("nan"), picked
    held = held_sum[ok] / held_cnt[ok]
    if np.std(held) < 1e-12 or np.std(y[ok]) < 1e-12:
        return float("nan"), picked
    return float(np.corrcoef(held, y[ok])[0, 1]), picked


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
                r, _ = _single_metric_r(subsets, x, y, cfg.task_names, fixed_j=j)
                if not np.isnan(r) and r > oracle_r:
                    oracle_r, oracle_c = r, c

            nested_r, picked_idx = _single_metric_r(subsets, x, y, cfg.task_names)
            picked = [cols[j] for j in picked_idx]
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
    print("\n  Both single-metric scores use the same folds, the same min-max scaling")
    print("  fitted on training tasks only, and the same averaging of predictions")
    print("  over folds as the multivariate model.")
    print("  'nested' chooses the metric inside each fold from the training tasks, so")
    print("  it is an honest held-out score. 'oracle' fixes one metric chosen after")
    print("  seeing every fold. They differ ONLY in when the metric is chosen, so")
    print("  oracle minus nested isolates the selection bias.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
