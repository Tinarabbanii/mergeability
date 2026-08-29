import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.backends import get_backend
from src.config import load_config
from src.metrics import MetricComputer
from src.pipeline import feature_columns, load_joined
from src.predict import loto_evaluate

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
STEPS = [150, 300, 600, 1200, 2400]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="synthetic", choices=["synthetic", "clip"])
    ap.add_argument("--solver", default=None, choices=["lasso", "pearson"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.backend)
    p = cfg.eval["predictor"]
    solver = args.solver or str(p.get("solver", "lasso"))

    mc = MetricComputer(cfg, get_backend(cfg))
    df = load_joined(cfg)
    df = df[df.k == args.k]
    cols = feature_columns(df, "data_free", mc)

    sweep = ALPHAS if solver == "lasso" else STEPS
    label = "alpha" if solver == "lasso" else "steps"
    print(f"predictor sweep  [backend={cfg.backend}, k={args.k}, "
          f"{len(cols)} features, solver={solver}, varying {label}]\n")
    print(f"{'method':<20}" + "".join(f"{s:>9}" for s in sweep) + f"{'range':>10}{'  plateau ' + label:>14}")
    print("-" * (20 + 9 * len(sweep) + 24))

    rows, ok = [], True
    for method in cfg.merge_methods:
        sub = df[df.method == method].reset_index(drop=True)
        if len(sub) < 6:
            continue
        subsets = [tuple(t.split("|")) for t in sub["tasks"]]
        x = np.nan_to_num(sub[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        y = sub["normalized_accuracy"].to_numpy(float)

        vals = []
        for s in sweep:
            kw = dict(solver=solver, seed=cfg.seed, lr=float(p["lr"]))
            if solver == "lasso":
                kw["l1_lambda"] = s
            else:
                kw.update(steps=s, l1_lambda=float(p["l1_lambda"]),
                          n_restarts=int(p.get("n_restarts", 5)))
            vals.append(loto_evaluate(subsets, x, y, cols, cfg.task_names, **kw)["pooled_r"])

        finite = [v for v in vals if not np.isnan(v)]
        spread = (max(finite) - min(finite)) if finite else float("nan")
        lo = hi = 0
        best_len, best_lo, best_hi = 0, 0, 0
        for lo in range(len(vals)):
            for hi in range(lo + 1, len(vals) + 1):
                run = [v for v in vals[lo:hi] if not np.isnan(v)]
                if len(run) == hi - lo and (max(run) - min(run)) <= args.tol:
                    if hi - lo > best_len:
                        best_len, best_lo, best_hi = hi - lo, lo, hi
        plateau = sweep[best_lo:best_hi] if best_len >= 2 else []
        mid = plateau[len(plateau) // 2] if plateau else None
        ok &= best_len >= 2
        flag = "" if best_len >= 2 else "  <-- NO PLATEAU"
        print(f"{method:<20}" + "".join(f"{v:>+9.3f}" for v in vals)
              + f"{spread:>10.3f}{str(mid):>14}{flag}")
        rows.append({"method": method, "solver": solver,
                     **{f"{label}_{s}": v for s, v in zip(sweep, vals)},
                     "range": spread,
                     "plateau_from": plateau[0] if plateau else None,
                     "plateau_to": plateau[-1] if plateau else None,
                     "plateau_mid": mid, "stable": best_len >= 2})

    pd.DataFrame(rows).to_csv(cfg.artifact(f"predictor_sweep_{solver}.csv"), index=False)
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'every method has a plateau at tolerance ' + str(args.tol) if ok else 'no plateau found -- widen the sweep'}")
    if rows:
        mids = [r["plateau_mid"] for r in rows if r["plateau_mid"] is not None]
        if mids:
            print(f"Plateau midpoints: {mids}. Set {label} from the plateau, not "
                  f"from the peak -- the peak is fitted to this sample and will "
                  f"not transfer.")
            cur = (p.get("l1_lambda") if solver == "lasso" else p.get("steps"))
            inside = all(r["plateau_from"] <= cur <= r["plateau_to"]
                         for r in rows if r["plateau_from"] is not None)
            print(f"Configured {label}={cur}: "
                  f"{'INSIDE every plateau -- good' if inside else 'OUTSIDE at least one plateau -- reconsider'}")
    print(f"-> {cfg.artifact(f'predictor_sweep_{solver}.csv')}")
if __name__ == "__main__":
    main()
