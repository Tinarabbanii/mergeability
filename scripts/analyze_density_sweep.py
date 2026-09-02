from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.config import load_config
from src.metrics import MetricComputer
from src.pipeline import feature_columns
from src.predict import loto_evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip"])
    args = ap.parse_args()

    cfg = load_config(args.backend)
    sweep = pd.read_csv(cfg.artifact("density_sweep.csv"))
    metrics = pd.read_csv(cfg.artifact("metrics.csv")).drop(columns=["k"])

                                                                             
                                                                      
    mc = MetricComputer(cfg, None)
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]),
              lr=float(p["lr"]), seed=cfg.seed,
              n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")),
              lambda_grid=p.get("lambda_grid"))

    rows = []
    for density in sorted(sweep.density.unique()):
        sub = sweep[sweep.density == density].merge(metrics, on="tasks", how="left")
        sub = sub.reset_index(drop=True)
        subsets = [tuple(s.split("|")) for s in sub["tasks"]]
        y = sub["normalized_accuracy"].to_numpy(dtype=float)

        row = {"density": density, "n": len(sub),
               "mean_acc": float(y.mean()), "std_y": float(y.std(ddof=1))}
        for kind in ("data_free", "full"):
            cols = feature_columns(sub, kind, mc)
            x = np.nan_to_num(sub[cols].to_numpy(dtype=float),
                              nan=0.0, posinf=0.0, neginf=0.0)
            out = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)
            row[f"{kind}_r"] = out["pooled_r"]
            row[f"{kind}_fold_r"] = out["fold_r_mean"]
            row[f"{kind}_fold_sd"] = out["fold_r_std"]
            row[f"{kind}_nfeat"] = len(cols)
        rows.append(row)
        print(f"  density={density:<5} std_y={row['std_y']:.4f}  "
              f"data-free r={row['data_free_r']:+.3f}  full r={row['full_r']:+.3f}")

    out = pd.DataFrame(rows)
    path = cfg.artifact("density_predictability.csv")
    out.to_csv(path, index=False)

    print("\n  held-out r by density (pooled over LOTO folds):")
    print(out[["density", "mean_acc", "std_y", "data_free_r", "full_r"]]
          .round(4).to_string(index=False))

    lo, hi = out.iloc[0], out.iloc[-1]
    print(f"\n  data-free r: {lo['data_free_r']:+.3f} at density {lo['density']} "
          f"-> {hi['data_free_r']:+.3f} at density {hi['density']}")
    print(f"  target std : {lo['std_y']:.4f} -> {hi['std_y']:.4f}  "
          f"(narrowing spread would DEPRESS r, so a rise is conservative)")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
