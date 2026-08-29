### Important
# My research question 1
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import Config
from ..metrics import MetricComputer
from ..pipeline import feature_columns, load_joined
from ..predict import loto_evaluate

def run(cfg: Config, backend, k: int = 2) -> pd.DataFrame:
    print(f"e2: RQ1 -- data-free prediction  [backend={cfg.backend}, k={k}]")
    df = load_joined(cfg)
    df = df[df.k == k].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"no k={k} rows; run e1 first")

    mc = MetricComputer(cfg, backend)
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]),
              lr=float(p["lr"]), seed=cfg.seed,
              n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")))

    rows = []
    for method in cfg.merge_methods:
        sub = df[df.method == method].reset_index(drop=True)
        if len(sub) < 6:
            print(f"  {method}: only {len(sub)} rows, skipping")
            continue

        subsets = [tuple(s.split("|")) for s in sub["tasks"]]
        y = sub["normalized_accuracy"].to_numpy(dtype=float)

        result = {"method": method, "n": len(sub)}
        for kind in ("data_free", "full"):
            cols = feature_columns(sub, kind, mc)
            x = sub[cols].to_numpy(dtype=float)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            out = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)
            result[f"{kind}_r"] = out["pooled_r"]
            result[f"{kind}_fold_r"] = out["fold_r_mean"]
            result[f"{kind}_n_features"] = len(cols)

        free_r, full_r = result["data_free_r"], result["full_r"]
        result["retention"] = (free_r / full_r) if (full_r and abs(full_r) > 1e-9) else np.nan
        rows.append(result)

        print(f"  {method:<18} data-free r={free_r:+.3f}  "
              f"full r={full_r:+.3f}  retention={result['retention']:.1%}")

    out = pd.DataFrame(rows)
    path = cfg.artifact(f"e2_datafree_k{k}.csv")
    out.to_csv(path, index=False)

    if not out.empty:
        print(f"\n  MEAN RETENTION ACROSS METHODS: {out['retention'].mean():.1%}")
        print("  (this is the headline number for research question 1)")
    print(f"  -> {path}")
    return out
