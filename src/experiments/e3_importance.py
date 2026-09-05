# I wanna know which data-free metrics carry the signal, and is that ordering real?
# Importance & Reliability

from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import Config
from ..metrics import MetricComputer
from ..pipeline import feature_columns, load_joined
from ..predict import loto_evaluate, split_half_reliability

def run(cfg: Config, backend, k: int = 2) -> pd.DataFrame:
    print(f"e3: metric importance and reliability  [k={k}]")
    df = load_joined(cfg)
    df = df[df.k == k].reset_index(drop=True)
    mc = MetricComputer(cfg, backend)

    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]),
              lr=float(p["lr"]), seed=cfg.seed,
              n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")),
              lambda_grid=p.get("lambda_grid"))

    cols = feature_columns(df, "data_free", mc)
    coef_rows, rel_rows = [], []

    for method in cfg.merge_methods:
        sub = df[df.method == method].reset_index(drop=True)
        if len(sub) < 6:
            continue
        subsets = [tuple(s.split("|")) for s in sub["tasks"]]
        x = np.nan_to_num(sub[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y = sub["normalized_accuracy"].to_numpy(dtype=float)

        out = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)
        w = out["mean_weights"]
        for name, weight in zip(cols, w):
            coef_rows.append({"method": method, "metric": name, "coefficient": float(weight)})
        rel = split_half_reliability(
            subsets, x, y, cols, cfg.task_names,
            n_splits=int(cfg.eval["reliability"]["n_splits"]), **kw
        )
        rel_rows.append({"method": method, **{k2: v for k2, v in rel.items() if k2 != "all"}})

        order = np.argsort(-np.abs(w))[:3]
        top = ", ".join(f"{cols[i]}({w[i]:+.2f})" for i in order)
        print(f"  {method:<18} top: {top}")
        print(f"  {'':<18} split-half r={rel['split_half_r']:+.3f}  "
              f"Spearman-Brown={rel['spearman_brown']:+.3f}")

    coefs = pd.DataFrame(coef_rows)
    rels = pd.DataFrame(rel_rows)
    coefs.to_csv(cfg.artifact(f"e3_coefficients_k{k}.csv"), index=False)
    rels.to_csv(cfg.artifact(f"e3_reliability_k{k}.csv"), index=False)
# Method's agreements
    if not coefs.empty and coefs.method.nunique() > 1:
        piv = coefs.pivot(index="metric", columns="method", values="coefficient")

        def _agreement(methods) -> float | None:
            sub = piv[[m for m in methods if m in piv.columns]]
            if sub.shape[1] < 2:
                return None
            signs = np.sign(sub.to_numpy())
            agree = []
            for i in range(signs.shape[1]):
                for j in range(i + 1, signs.shape[1]):
                    both = (signs[:, i] != 0) & (signs[:, j] != 0)
                    if both.sum():
                        agree.append((signs[both, i] == signs[both, j]).mean())
            return float(np.mean(agree)) if agree else None

        a_all = _agreement(list(piv.columns))
        if a_all is not None:
            print(f"\n  cross-method SIGN AGREEMENT (all methods): {a_all:.1%}")
            print("  (paper reports 79.3% on their 20-task benchmark)")

        nulls = cfg.artifact(f"e5_nulls_k{k}.csv")
        sources = [cfg.artifact("results.csv"), cfg.artifact("metrics.csv")]
        reason = None
        if nulls.exists():
            probe = pd.read_csv(nulls)
            if "n_tasks" in probe.columns:
                if int(probe.n_tasks.iloc[0]) != len(cfg.task_names):
                    reason = (f"it was computed on {int(probe.n_tasks.iloc[0])} tasks, "
                              f"config has {len(cfg.task_names)}")
            elif any(src.exists() and nulls.stat().st_mtime < src.stat().st_mtime
                     for src in sources):
                reason = "it predates task-count recording and is older than the data"
        stale = reason is not None
        if stale:
            print(f"  SKIPPED the null-restricted agreement: {nulls.name} is stale -- "
                  f"{reason}. Re-run e5 for k={k}, then e3 again.")
        elif nulls.exists():
            nl = pd.read_csv(nulls)
            p95 = nl[["null_random_p95", "null_shuffled_p95"]].max(axis=1)
            clearing = nl.loc[nl.observed_r > p95, "method"].tolist()
            a_clear = _agreement(clearing)
            if a_clear is not None:
                print(f"  restricted to methods clearing their null "
                      f"({', '.join(clearing)}): {a_clear:.1%}")
            else:
                print(f"  restricted to methods clearing their null: UNDEFINED -- only "
                      f"{len(clearing)} ({', '.join(clearing) or 'none'}) clears, and "
                      f"agreement needs two. The {a_all:.1%} above is therefore carried "
                      f"by methods whose coefficients are noise.")
        else:
            print("  (run e5 to see this restricted to methods that clear their null)")

    print(f"  -> {cfg.artifact(f'e3_coefficients_k{k}.csv')}")
    return coefs
