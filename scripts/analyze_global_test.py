from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scipy.stats import binomtest
from src.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clip", choices=["synthetic", "clip", "clip16"])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.backend)
    frames = []
    for k in cfg.k_values:
        p = cfg.artifact(f"e5_nulls_k{k}.csv")
        if p.exists():
            frames.append(pd.read_csv(p).assign(k=k))
    if not frames:
        raise SystemExit("no e5_nulls_k*.csv found; run e5 first")
    d = pd.concat(frames, ignore_index=True)

    print(f"global test  [{cfg.backend}]  per-comparison false-positive rate {args.alpha}")
    rows = []
    for label, col in (("individual metrics", "clears_both_nulls"),
                       ("family-grouped", "grouped_clears")):
        if col not in d.columns:
            continue
        n, s = len(d), int(d[col].astype(bool).sum())
        bt = binomtest(s, n, args.alpha, alternative="greater")
        rows.append({"variant": label, "n_comparisons": n, "n_clearing": s,
                     "expected_by_chance": args.alpha * n, "p_value": bt.pvalue})
        print(f"  {label:<20} {s}/{n} clear   expected by chance {args.alpha*n:.2f}   "
              f"p = {bt.pvalue:.2e}")

    print("\n  per comparison:")
    for _, r in d.sort_values(["k", "method"]).iterrows():
        print(f"    k={int(r.k)} {r.method:<18} r={r.observed_r:+.3f}  "
              f"clears={bool(r.clears_both_nulls)}")

    out = pd.DataFrame(rows)
    path = cfg.artifact("global_test.csv")
    out.to_csv(path, index=False)
    print("\n  NOTE: the comparisons share tasks and metrics, so they are not fully")
    print("  independent. The binomial p-value is therefore optimistic; report it as")
    print("  a sanity check, not as a formal joint test.")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
