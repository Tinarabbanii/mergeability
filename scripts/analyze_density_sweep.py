"""Does PREDICTABILITY rise with TIES density, or only merge quality?

run_density_sweep.py showed post-merge accuracy climbing from density 0.05 to a
plateau at 0.4-0.8. That is a statement about how WELL the merge works. Item 3
asks something different: does the data-free predictor track the outcome better
as the trim gets less aggressive? If it does, TIES's unpredictability is caused
by the trim -- the one clearly non-linear step -- and not by TIES as a whole.

For each density slice this refits the same LOTO predictor used in e2 and reports
held-out r, so the numbers are comparable to e2_datafree_k2.csv by construction.

CAUTION when reading the output: the spread of the target narrows as density
rises (std 0.090 -> 0.062). Correlation is scale-free but NOT immune to range
restriction -- a shrinking spread in y can depress r even when the underlying
relationship is unchanged. std_y is printed beside r so the two are read
together; a rising r ALONGSIDE a shrinking std_y is the strong result, since
range restriction would push the other way.
"""
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

    # names only -- data_free_metric_names() reads config, never the backend,
    # so no checkpoints are loaded and this stays a CPU-only analysis.
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
