### Important
# My research question 2
# ORACLE AGGREGATION, THE ADDITIVE BASELINE, METRIC TRANSFER

from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from ..config import Config
from ..metrics import AGG_MAX, AGG_MIN, MetricComputer
from ..pipeline import feature_columns, load_joined
from ..predict import fit_linear_l1, minmax_apply, minmax_fit

def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# Can we predict k-way accuracy from pairwise accuracy?
def test_a_oracle(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in cfg.merge_methods:
        sub = df[df.method == method]
        pair_acc = {tuple(s.split("|")): a for s, a in
                    zip(sub[sub.k == 2]["tasks"], sub[sub.k == 2]["normalized_accuracy"])}

        for k in [v for v in cfg.k_values if v > 2]:
            block = sub[sub.k == k]
            if block.empty:
                continue
            actual, agg = [], {"mean": [], "min": [], "max": []}
            for tasks_str, acc in zip(block["tasks"], block["normalized_accuracy"]):
                tasks = tuple(tasks_str.split("|"))
                vals = [pair_acc[tuple(sorted(p))] for p in combinations(tasks, 2)
                        if tuple(sorted(p)) in pair_acc]
                if len(vals) < len(list(combinations(tasks, 2))):
                    continue
                actual.append(acc)
                agg["mean"].append(float(np.mean(vals)))
                agg["min"].append(float(np.min(vals)))
                agg["max"].append(float(np.max(vals)))

            if len(actual) < 3:
                continue
            row = {"method": method, "k": k, "n": len(actual)}
            for name, vals in agg.items():
                row[f"r_{name}"] = _corr(vals, actual)
            row["mean_actual"] = float(np.mean(actual))
            rows.append(row)
            best = max(agg, key=lambda a: (row[f"r_{a}"] if not np.isnan(row[f"r_{a}"]) else -9))
            print(f"  [A] {method:<18} k={k}  "
                  f"mean r={row['r_mean']:+.3f}  min r={row['r_min']:+.3f}  "
                  f"max r={row['r_max']:+.3f}   best={best}")
    return pd.DataFrame(rows)


# Is it just additive in task?
def test_c_additive(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in cfg.merge_methods:
        sub = df[df.method == method]
        pairs = sub[sub.k == 2]
        pair_acc = {tuple(s.split("|")): a for s, a in
                    zip(pairs["tasks"], pairs["normalized_accuracy"])}
        solo: dict[str, float] = {}
        for t in cfg.task_names:
            vals = [a for p, a in pair_acc.items() if t in p]
            if vals:
                solo[t] = float(np.mean(vals))

        for k in [v for v in cfg.k_values if v > 2]:
            block = sub[sub.k == k]
            if block.empty:
                continue
            actual, additive, pairmean = [], [], []
            for tasks_str, acc in zip(block["tasks"], block["normalized_accuracy"]):
                tasks = tuple(tasks_str.split("|"))
                if not all(t in solo for t in tasks):
                    continue
                vals = [pair_acc[tuple(sorted(p))] for p in combinations(tasks, 2)
                        if tuple(sorted(p)) in pair_acc]
                if len(vals) < len(list(combinations(tasks, 2))):
                    continue
                actual.append(acc)
                additive.append(float(np.mean([solo[t] for t in tasks])))
                pairmean.append(float(np.mean(vals)))

            if len(actual) < 4:
                continue
            r_add = _corr(additive, actual)
            r_pair = _corr(pairmean, actual)
            a_arr = np.asarray(additive, float)
            y_arr = np.asarray(actual, float)
            p_arr = np.asarray(pairmean, float)
            def _resid(v, on):
                on_c = on - on.mean()
                if np.dot(on_c, on_c) < 1e-12:
                    return v - v.mean()
                beta = np.dot(v - v.mean(), on_c) / np.dot(on_c, on_c)
                return (v - v.mean()) - beta * on_c
            increment = _corr(_resid(p_arr, a_arr), _resid(y_arr, a_arr))

            rows.append({"method": method, "k": k, "n": len(actual),
                         "r_additive": r_add, "r_pairwise_mean": r_pair,
                         "increment_r": increment})
            print(f"  [C] {method:<18} k={k}  "
                  f"additive r={r_add:+.3f}  pairwise r={r_pair:+.3f}  "
                  f"increment={increment:+.3f}")
    return pd.DataFrame(rows)

def test_b_transfer(cfg: Config, df: pd.DataFrame, mc: MetricComputer) -> pd.DataFrame:
    p = cfg.eval["predictor"]
    kw = dict(l1_lambda=float(p["l1_lambda"]), steps=int(p["steps"]),
              lr=float(p["lr"]), seed=cfg.seed,
              n_restarts=int(p.get("n_restarts", 5)),
              solver=str(p.get("solver", "lasso")))
    base_cols = feature_columns(df, "data_free", mc)

    rows = []
    for method in cfg.merge_methods:
        sub = df[df.method == method]
        train = sub[sub.k == 2]
        if len(train) < 6:
            continue

        xtr_raw = np.nan_to_num(train[base_cols].to_numpy(float), nan=0.0,
                                posinf=0.0, neginf=0.0)
        lo, hi = minmax_fit(xtr_raw)
        fit = fit_linear_l1(minmax_apply(xtr_raw, lo, hi), train["normalized_accuracy"].to_numpy(float), base_cols, **kw)
        for k in [v for v in cfg.k_values if v > 2]:
            block = sub[sub.k == k]
            if len(block) < 3:
                continue
            y = block["normalized_accuracy"].to_numpy(float)
            row = {"method": method, "k": k, "n": len(block), "train_r": fit.train_r}

            for agg_name, prefix in (("mean", ""), ("min", AGG_MIN), ("max", AGG_MAX)):
                cols = [prefix + c if prefix + c in block.columns else c for c in base_cols]
                x = np.nan_to_num(block[cols].to_numpy(float), nan=0.0,
                                  posinf=0.0, neginf=0.0)
                pred = minmax_apply(x, lo, hi) @ fit.weights
                row[f"r_{agg_name}"] = _corr(pred, y)
            rows.append(row)
            print(f"  [B] {method:<18} k={k}  "
                  f"mean r={row['r_mean']:+.3f}  min r={row['r_min']:+.3f}  "
                  f"max r={row['r_max']:+.3f}")
    return pd.DataFrame(rows)

def run(cfg: Config, backend) -> dict[str, pd.DataFrame]:
    print(f"e4: RQ2 -- does pairwise predict k-way?  [backend={cfg.backend}]")
    df = load_joined(cfg)
    mc = MetricComputer(cfg, backend)

    if not any(v > 2 for v in cfg.k_values):
        raise RuntimeError("no k > 2 in configs/eval.yaml:k_values")

    print("\n  TEST A -- aggregating MEASURED pairwise accuracy:")
    a = test_a_oracle(cfg, df)
    print("\n  TEST C -- additive task-level baseline (the control):")
    c = test_c_additive(cfg, df)
    print("\n  TEST B -- transferring the k=2 data-free predictor:")
    b = test_b_transfer(cfg, df, mc)

    a.to_csv(cfg.artifact("e4_oracle.csv"), index=False)
    b.to_csv(cfg.artifact("e4_transfer.csv"), index=False)
    c.to_csv(cfg.artifact("e4_additive.csv"), index=False)

    if not a.empty:
        best = {n: a[f"r_{n}"].mean() for n in ("mean", "min", "max")}
        winner = max(best, key=lambda n: best[n] if not np.isnan(best[n]) else -9)
        print(f"\n  AGGREGATOR RANKING (test A, averaged): " +
              "  ".join(f"{n}={v:+.3f}" for n, v in best.items()))
        print(f"  Best aggregator: {winner}")
        if winner == "min":
            print("  -> consistent with the WEAKEST-LINK hypothesis: a group is "
                  "only as mergeable as its worst pair.")
        else:
            wins = int((a["r_mean"] > a["r_min"]).sum())
            print(f"  -> the weakest-link hypothesis is not supported in this "
                  f"benchmark; '{winner}' fits better, and mean beat min in "
                  f"{wins} of {len(a)} settings (no significance test).")

    if not c.empty:
        print(f"\n  ADDITIVITY CHECK   additive r={c.r_additive.mean():+.3f}   "
              f"pairwise r={c.r_pairwise_mean.mean():+.3f}   "
              f"increment={c.increment_r.mean():+.3f}")
        gain = float((c.r_pairwise_mean - c.r_additive).mean())
        print(f"     absolute gain over the additive baseline: {gain:+.3f}")
        by_k = c.groupby("k").apply(
            lambda g: float((g.r_pairwise_mean - g.r_additive).mean()))
        print("     gain by k: " + "  ".join(f"k={k}:{v:+.3f}" for k, v in by_k.items()))
        if c.increment_r.mean() < 0.2:
            print("  -> k-way merging is essentially ADDITIVE in the tasks: knowing")
            print("     which specific pairs are present adds almost nothing beyond")
            print("     knowing which tasks are present.")
        else:
            print("  -> pairwise structure explains variance the additive baseline")
            print("     leaves over, though the absolute gain above is small.")

    print(f"\n  -> {cfg.artifact('e4_oracle.csv')}, {cfg.artifact('e4_transfer.csv')}, "
          f"{cfg.artifact('e4_additive.csv')}")
    return {"oracle": a, "transfer": b, "additive": c}
