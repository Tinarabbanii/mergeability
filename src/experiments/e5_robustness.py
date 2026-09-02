
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import Config
from ..metrics import MetricComputer
from ..pipeline import feature_columns, load_joined
from ..predict import (bootstrap_r, loto_evaluate, null_random_features, null_shuffled_target)

def run(cfg: Config, backend, k: int = 2) -> pd.DataFrame:
    print(f"e5: robustness and null baselines  [k={k}]")
    df = load_joined(cfg)
    df = df[df.k == k].reset_index(drop=True)
    mc = MetricComputer(cfg, backend)

    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]),
              lr=float(p["lr"]), seed=cfg.seed,
              n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")),
              lambda_grid=p.get("lambda_grid"))
    n_trials = int(cfg.eval["nulls"]["n_random_trials"])
    cols = feature_columns(df, "data_free", mc)

    rows = []
    for method in cfg.merge_methods:
        sub = df[df.method == method].reset_index(drop=True)
        if len(sub) < 6:
            continue
        subsets = [tuple(s.split("|")) for s in sub["tasks"]]
        x = np.nan_to_num(sub[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        y = sub["normalized_accuracy"].to_numpy(float)
        observed = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)["pooled_r"]
        n1 = null_random_features(subsets, y, cfg.task_names, len(cols), n_trials=n_trials, **dict(kw))
        n2 = null_shuffled_target(subsets, x, y, cols, cfg.task_names, n_trials=n_trials, **dict(kw))
        clears = (observed > n1["p95"]) and (observed > n2["p95"])

# Bootsrap CI
        ci = bootstrap_r(subsets, x, y, cols, cfg.task_names, n_boot=int(cfg.eval["nulls"].get("n_bootstrap", 200)), **dict(kw))
        groups = [mc.family_of(c) for c in cols]
        g_obs = loto_evaluate(subsets, x, y, cols, cfg.task_names, groups=groups, **kw)["pooled_r"]
        g_n1 = null_random_features(subsets, y, cfg.task_names, len(cols),
                                    n_trials=n_trials, groups=groups, **dict(kw))
        g_n2 = null_shuffled_target(subsets, x, y, cols, cfg.task_names,
                                    n_trials=n_trials, groups=groups, **dict(kw))
        g_null_p95 = max(g_n1["p95"], g_n2["p95"])
        g_clears = g_obs > g_null_p95

        rows.append({
            "method": method, "n_features": len(cols), "observed_r": observed,
            "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "null_random_mean": n1["mean"], "null_random_p95": n1["p95"],
            "null_shuffled_mean": n2["mean"], "null_shuffled_p95": n2["p95"],
            "clears_both_nulls": bool(clears),
            "grouped_n_features": len(set(groups)),
            "grouped_r": g_obs, "grouped_null_p95": g_null_p95,
            "grouped_clears": bool(g_clears),
        })
        print(f"  {method:<18} observed={observed:+.3f} "
              f"[{ci['lo']:+.2f},{ci['hi']:+.2f}]   "
              f"nulls p95={n1['p95']:+.3f}/{n2['p95']:+.3f}   "
              f"{'CLEARS' if clears else 'AT CHANCE'}")
        print(f"  {'':<18} grouped ({len(set(groups))} params): r={g_obs:+.3f}   "
              f"null p95={g_null_p95:+.3f}   "
              f"{'CLEARS' if g_clears else 'AT CHANCE'}")

    out = pd.DataFrame(rows)
    path = cfg.artifact(f"e5_nulls_k{k}.csv")
    out.to_csv(path, index=False)
    if not out.empty:
        n_clear = int(out["clears_both_nulls"].sum())
        n_g = int(out["grouped_clears"].sum())
        print(f"\n  {n_clear}/{len(out)} methods clear both nulls with "
              f"{out.n_features.iloc[0]} individual metrics.")
        print(f"  {n_g}/{len(out)} clear with "
              f"{out.grouped_n_features.iloc[0]} family-level features.")
        if n_g > n_clear:
            print("  -> grouping raised the number of methods clearing chance.")
        elif n_g == n_clear:
            print("  -> grouping did not change which methods clear; compare the "
                  "margins in the CSV to see whether it widened them.")
        if n_clear == 0:
            print(f"  Every result is at chance. Report that plainly, it is a "
                  f"finding about how little signal {len(df)} points can carry, "
                  f"not a bug.")
    print(f"  -> {path}")
    return out
