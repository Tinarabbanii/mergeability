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
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.backend)
    m10 = pd.read_csv(cfg.artifact("metrics_cal10.csv"))
    m100 = pd.read_csv(cfg.artifact("metrics_cal100.csv"))
    results = pd.read_csv(cfg.artifact("results.csv"))
    mc = MetricComputer(cfg, None)

    if len(m10) != len(m100) or not (m10["tasks"] == m100["tasks"]).all():
        raise RuntimeError("cal10 and cal100 disagree on rows; not comparable")

    free = [c for c in mc.data_free_metric_names() if c in m10.columns]
    dep = [c for c in mc.all_metric_names()
           if c in m10.columns and c not in free]

                                                                           
    drift = {c: float(np.nanmax(np.abs(m10[c] - m100[c]))) for c in free}
    worst = max(drift.values()) if drift else 0.0
    print(f"  control: {len(free)} data-free columns, max drift = {worst:.2e}")
    if worst > 1e-9:
        bad = sorted(drift.items(), key=lambda kv: -kv[1])[:3]
        raise RuntimeError(
            f"data-free metrics differ between runs (worst: {bad}). These do not "
            f"depend on calibration data, so the two files are not comparable.")
    print("  -> identical, the runs differ only in calibration size\n")

    print(f"  STABILITY of the {len(dep)} data-dependent metrics (10 vs 100 samples):")
    stab = []
    for c in dep:
        r = m10[c].corr(m100[c])
        stab.append({"metric": c, "corr_10_vs_100": r,
                     "mean_10": m10[c].mean(), "mean_100": m100[c].mean()})
        print(f"    {c:30s} r = {r:+.3f}")
    stab_df = pd.DataFrame(stab)

    print(f"\n  PREDICTION at k={args.k} (does a less noisy full set predict better?):")
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]),
              lr=float(p["lr"]), seed=cfg.seed,
              n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")),
              lambda_grid=p.get("lambda_grid"))

    rows = []
    res_k = results[results.k == args.k]
    for method in cfg.merge_methods:
        r_m = res_k[res_k.method == method]
        if len(r_m) < 6:
            continue
        row = {"method": method, "n": len(r_m)}
        for label, mdf in (("cal10", m10), ("cal100", m100)):
            sub = r_m.merge(mdf.drop(columns=["k"]), on="tasks", how="left").reset_index(drop=True)
            subsets = [tuple(s.split("|")) for s in sub["tasks"]]
            y = sub["normalized_accuracy"].to_numpy(dtype=float)
            for kind in ("data_free", "full"):
                cols = feature_columns(sub, kind, mc)
                x = np.nan_to_num(sub[cols].to_numpy(dtype=float),
                                  nan=0.0, posinf=0.0, neginf=0.0)
                out = loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)
                row[f"{kind}_{label}"] = out["pooled_r"]
        row["full_delta"] = row["full_cal100"] - row["full_cal10"]
        rows.append(row)
        print(f"    {method:<18} full: {row['full_cal10']:+.3f} -> {row['full_cal100']:+.3f} "
              f"({row['full_delta']:+.3f})   data-free: {row['data_free_cal10']:+.3f}")

    out = pd.DataFrame(rows)
    stab_df.to_csv(cfg.artifact("calibration_stability.csv"), index=False)
    out.to_csv(cfg.artifact("calibration_prediction.csv"), index=False)
    print(f"\n  -> {cfg.artifact('calibration_stability.csv')}")
    print(f"  -> {cfg.artifact('calibration_prediction.csv')}")


if __name__ == "__main__":
    main()
