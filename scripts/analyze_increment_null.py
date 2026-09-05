from __future__ import annotations
import argparse, sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from src.config import load_config
from src.pipeline import load_joined


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _resid(v, on):
    on_c = on - on.mean()
    if np.dot(on_c, on_c) < 1e-12:
        return v - v.mean()
    beta = np.dot(v - v.mean(), on_c) / np.dot(on_c, on_c)
    return (v - v.mean()) - beta * on_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip", "clip16"])
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.backend)
    df = load_joined(cfg)
    print(f"permutation null on the pairwise increment  [{cfg.backend}, {args.trials} shuffles]")

    rows = []
    for method in cfg.merge_methods:
        m = df[df.method == method]
        pair_acc = {tuple(s.split("|")): a for s, a in
                    zip(m[m.k == 2]["tasks"], m[m.k == 2]["normalized_accuracy"])}
        solo = {}
        for t in cfg.task_names:
            v = [a for p, a in pair_acc.items() if t in p]
            if v:
                solo[t] = float(np.mean(v))

        for k in [v for v in cfg.k_values if v > 2]:
            block = m[m.k == k]
            actual, additive, pairmean = [], [], []
            for s, acc in zip(block["tasks"], block["normalized_accuracy"]):
                tasks = tuple(s.split("|"))
                vals = [pair_acc[tuple(sorted(p))] for p in combinations(tasks, 2)
                        if tuple(sorted(p)) in pair_acc]
                if len(vals) < len(list(combinations(tasks, 2))) or not all(t in solo for t in tasks):
                    continue
                actual.append(acc)
                additive.append(float(np.mean([solo[t] for t in tasks])))
                pairmean.append(float(np.mean(vals)))
            if len(actual) < 6:
                continue
            y = np.array(actual); a = np.array(additive); pm = np.array(pairmean)
            obs = _corr(_resid(pm, a), _resid(y, a))

            rng = np.random.default_rng(cfg.seed)
            null = []
            for _ in range(args.trials):
                v = _corr(_resid(rng.permutation(pm), a), _resid(y, a))
                if not np.isnan(v):
                    null.append(v)
            null = np.array(null)
            p95 = float(np.percentile(null, 95))
            pval = float((null >= obs).mean())
            rows.append({"method": method, "k": k, "n": len(y), "observed_increment": obs,
                         "null_p95": p95, "p_value": pval, "clears": bool(obs > p95)})
            print(f"  {method:<18} k={k}  increment={obs:+.3f}  null p95={p95:+.3f}  "
                  f"p={pval:.4f}  {'REAL' if obs > p95 else 'at chance'}")

    out = pd.DataFrame(rows)
    path = cfg.artifact("increment_null.csv")
    out.to_csv(path, index=False)
    print(f"\n  {int(out.clears.sum())} of {len(out)} increments beat their permutation null")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
