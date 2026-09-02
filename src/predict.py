from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
import torch
from joblib import Parallel, delayed

# Null and bootstrap trials are independent, so they run across cores. The
# random draws are generated up front in the SERIAL order (see the null
# functions below), so results are identical whatever N_JOBS is set to.
# Override with MERGEABILITY_N_JOBS=1 to reproduce the single-process path.
N_JOBS = int(os.environ.get("MERGEABILITY_N_JOBS", "-1"))

L1_FITS = {"fits": 0, "fallbacks": 0}
def fallback_report(label: str = "") -> str:
    n, f = L1_FITS["fits"], L1_FITS["fallbacks"]
    if not f:
        return f"  L1 fallback: never fired ({n} fits) -- OK"
    pct = 100.0 * f / max(n, 1)
    flag = "  <-- ABOVE 0.5%, INVESTIGATE" if pct > 0.5 else "  (baseline 0.033%, fine)"
    return f"  L1 fallback fired {f}/{n} fits ({pct:.3f}%){flag}{label}"

### Pearson correlation
def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return (a * b).sum() / denom

def minmax_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.min(axis=0), x.max(axis=0)


def minmax_apply(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = hi - lo
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    out = 2.0 * (x - lo) / span - 1.0
    out = np.where(np.abs(hi - lo)[None, :] < 1e-12, 0.0, out)
    return np.clip(out, -3.0, 3.0)

@dataclass
class FitResult:
    weights: np.ndarray
    train_r: float
    feature_names: list[str]
    used_fallback: bool = False

def _fit_lasso(x: np.ndarray, y: np.ndarray, feature_names: list[str], l1_lambda: float) -> FitResult:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso

    ys = y - y.mean()
    sd = ys.std()
    ys = ys / sd if sd > 1e-12 else ys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=l1_lambda, fit_intercept=True, max_iter=50000, tol=1e-4)
        model.fit(x, ys)
    w = np.asarray(model.coef_, dtype=float)
    fell_back = False
    L1_FITS["fits"] += 1

    if np.linalg.norm(w) < 1e-12:
        L1_FITS["fallbacks"] += 1
        fell_back = True
        import warnings
        warnings.warn(
            f"Lasso(alpha={l1_lambda}) zeroed all {x.shape[1]} coefficients; "
            f"falling back to marginal correlations, which is a different "
            f"estimator. Lower l1_lambda if you see this.",
            RuntimeWarning, stacklevel=2,
        )
        w = np.array([
            np.corrcoef(x[:, j], ys)[0, 1] if np.std(x[:, j]) > 1e-12 else 0.0
            for j in range(x.shape[1])
        ])
        w = np.nan_to_num(w)

    norm = np.linalg.norm(w)
    w = w / norm if norm > 1e-12 else w

    pred = x @ w
    r = (float(np.corrcoef(pred, y)[0, 1])
         if np.std(pred) > 1e-12 and np.std(y) > 1e-12 else 0.0)
    return FitResult(w, r, feature_names, fell_back)

def fit_linear_l1(x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    l1_lambda: float = 0.01,
    steps: int = 600,
    lr: float = 0.05,
    seed: int = 0,
    n_restarts: int = 5,
    solver: str = "lasso",) -> FitResult:
    if solver == "lasso":
        return _fit_lasso(x, y, feature_names, l1_lambda)
    if solver != "pearson":
        raise ValueError(f"unknown solver {solver!r}; use 'lasso' or 'pearson'")

    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    best_w, best_obj = None, float("inf")
    for restart in range(n_restarts):
        g = torch.Generator().manual_seed(seed * 1000 + restart)
        w = torch.randn(xt.shape[1], generator=g)
        w = (w / w.norm().clamp(min=1e-8)).requires_grad_(True)

        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            w_dir = w / w.norm().clamp(min=1e-8)      # direction only
            pred = xt @ w_dir
            loss = -pearson(pred, yt) + l1_lambda * w_dir.abs().sum()
            loss.backward()
            opt.step()

        with torch.no_grad():
            w_dir = w / w.norm().clamp(min=1e-8)
            obj = (-pearson(xt @ w_dir, yt) + l1_lambda * w_dir.abs().sum()).item()
        if obj < best_obj:
            best_obj, best_w = obj, w_dir.detach().clone()

    assert best_w is not None
    with torch.no_grad():
        r = pearson(xt @ best_w, yt).item()
    return FitResult(best_w.numpy(), float(r), feature_names)

def collapse_groups(x: np.ndarray, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    names = sorted(set(groups))
    cols = []
    for g in names:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        cols.append(x[:, idx].mean(axis=1))
    return np.stack(cols, axis=1), names

def select_lambda_cv(subsets, x, y, feature_names, tasks,
                     grid: list[float], **kw) -> float:
    kw.pop("l1_lambda", None)          # set per-candidate below
    kw.pop("lambda_grid", None)        # never recurse
    scores, ses = [], []
    for lam in grid:
        res = loto_evaluate(subsets, x, y, feature_names, tasks,
                            l1_lambda=lam, lambda_grid=None, **kw)
        m, s, n = res["fold_r_mean"], res["fold_r_std"], res["n_folds"]
        degenerate = res.get("n_fallback", 0) > 0
        scores.append(-np.inf if (degenerate or np.isnan(m)) else m)
        ses.append(s / np.sqrt(n) if n and not np.isnan(s) else 0.0)

    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        return grid[len(grid) // 2]
    threshold = scores[best] - ses[best]
    ok = [i for i, sc in enumerate(scores) if sc >= threshold]
    return grid[max(ok)] if ok else grid[best]


def _select_lambda(subsets, x, y, feature_names, tasks, held_out,
                   grid: list[float], **kw) -> float:
    tr = [i for i, s in enumerate(subsets) if held_out not in s]
    inner_tasks = [t for t in tasks if t != held_out]
    if len(tr) < 4 or len(inner_tasks) < 3:
        return grid[len(grid) // 2]
    idx = np.asarray(tr)
    return select_lambda_cv([subsets[i] for i in tr], x[idx], y[idx],
                            feature_names, inner_tasks, grid, **kw)
def loto_evaluate(subsets: list[tuple[str, ...]],
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    tasks: list[str],
    l1_lambda: float = 0.01,
    steps: int = 600,
    lr: float = 0.05,
    seed: int = 0,
    groups: list[str] | None = None,
    n_restarts: int = 5,
    solver: str = "lasso",
    lambda_grid: list[float] | None = None,) -> dict:
    held_sum = np.zeros(len(y), dtype=float)
    held_cnt = np.zeros(len(y), dtype=float)
    fold_r: list[float] = []
    fold_weights: list[np.ndarray] = []
    chosen_lambdas: list[float] = []
    n_fallback = 0

    for t in tasks:
        val_idx = np.array([i for i, s in enumerate(subsets) if t in s])
        tr_idx = np.array([i for i, s in enumerate(subsets) if t not in s])
        if len(val_idx) < 2 or len(tr_idx) < 3:
            continue

        lo, hi = minmax_fit(x[tr_idx])
        xtr = minmax_apply(x[tr_idx], lo, hi)
        xva = minmax_apply(x[val_idx], lo, hi)
        names = feature_names
        if groups is not None:
            xtr, names = collapse_groups(xtr, groups)
            xva, _ = collapse_groups(xva, groups)

        lam = l1_lambda
        if lambda_grid:
            lam = _select_lambda(subsets, x, y, feature_names, tasks, t,
                                 lambda_grid, steps=steps, lr=lr, seed=seed,
                                 groups=groups, n_restarts=n_restarts, solver=solver)
        chosen_lambdas.append(lam)
        fit = fit_linear_l1(xtr, y[tr_idx], names, lam, steps, lr,
                            seed, n_restarts, solver)
        pred = xva @ fit.weights
        held_sum[val_idx] += pred
        held_cnt[val_idx] += 1
        fold_weights.append(fit.weights)
        n_fallback += int(fit.used_fallback)

        if len(val_idx) >= 3 and np.std(pred) > 1e-12 and np.std(y[val_idx]) > 1e-12:
            fold_r.append(float(np.corrcoef(pred, y[val_idx])[0, 1]))

    mask = held_cnt > 0
    held_pred = np.full(len(y), np.nan)
    held_pred[mask] = held_sum[mask] / held_cnt[mask]
    if mask.sum() >= 3 and np.std(held_pred[mask]) > 1e-12:
        pooled = float(np.corrcoef(held_pred[mask], y[mask])[0, 1])
    else:
        pooled = float("nan")
    n_feat = len(set(groups)) if groups is not None else x.shape[1]
    mean_w = np.mean(fold_weights, axis=0) if fold_weights else np.zeros(n_feat)

    return {
        "pooled_r": pooled,
        "n_features": n_feat,
        "fold_r_mean": float(np.mean(fold_r)) if fold_r else float("nan"),
        "fold_r_std": float(np.std(fold_r)) if fold_r else float("nan"),
        "n_folds": len(fold_r),
        "held_pred": held_pred,
        "mean_weights": mean_w,
        "feature_names": feature_names,
        "chosen_lambdas": chosen_lambdas,
        "n_fallback": n_fallback,
    }

def _trial(fn, *args, **kw):
    """Run one trial and report this worker's L1 fit counts back to the parent.

    In a loky worker L1_FITS starts at zero and its increments would otherwise
    be discarded when the process exits, silently breaking fallback_report().
    """
    before = dict(L1_FITS)
    res = fn(*args, **kw)
    return res, {k: L1_FITS[k] - before[k] for k in ("fits", "fallbacks")}


def _run_trials(jobs) -> list:
    out = Parallel(n_jobs=N_JOBS)(jobs)
    for _, counts in out:
        L1_FITS["fits"] += counts["fits"]
        L1_FITS["fallbacks"] += counts["fallbacks"]
    return [res for res, _ in out]


def null_random_features(subsets, y, tasks, n_features: int, n_trials: int = 20, **kw) -> dict:
    seed = int(kw.pop("seed", 0))
    rng = np.random.default_rng(seed)
    names = [f"noise_{i}" for i in range(n_features)]
    # drawn in the serial loop's order, so parallelism cannot shift the stream
    xs = [rng.standard_normal((len(y), n_features)) for _ in range(n_trials)]
    out = _run_trials(
        delayed(_trial)(loto_evaluate, subsets, xs[t], y, names, tasks,
                        seed=seed + t, **kw)
        for t in range(n_trials))
    scores = [r["pooled_r"] for r in out if not np.isnan(r["pooled_r"])]
    return {
        "mean": float(np.mean(scores)) if scores else float("nan"),
        "std": float(np.std(scores)) if scores else float("nan"),
        "p95": float(np.percentile(scores, 95)) if scores else float("nan"),
        "scores": scores,
    }

def null_shuffled_target(
    subsets, x, y, feature_names, tasks, n_trials: int = 20, **kw
) -> dict:
    seed = int(kw.pop("seed", 0))
    rng = np.random.default_rng(seed)
    ys = [rng.permutation(y) for _ in range(n_trials)]
    out = _run_trials(
        delayed(_trial)(loto_evaluate, subsets, x, ys[t], feature_names, tasks,
                        seed=seed + t, **kw)
        for t in range(n_trials))
    scores = [r["pooled_r"] for r in out if not np.isnan(r["pooled_r"])]
    return {
        "mean": float(np.mean(scores)) if scores else float("nan"),
        "std": float(np.std(scores)) if scores else float("nan"),
        "p95": float(np.percentile(scores, 95)) if scores else float("nan"),
        "scores": scores,
    }


### Metric importance
def split_half_reliability(
    subsets, x, y, feature_names, tasks, n_splits: int = 50, **kw
) -> dict:
    del subsets, tasks
    from scipy.stats import spearmanr
    seed = int(kw.get("seed", 0))
    rng = np.random.default_rng(seed)
    rs: list[float] = []
    n = len(y)
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        if len(a) < 4 or len(b) < 4:
            continue
        wa = _fit_normalised(x[a], y[a], feature_names, **kw)
        wb = _fit_normalised(x[b], y[b], feature_names, **kw)
        rho = spearmanr(wa, wb).statistic
        if not np.isnan(rho):
            rs.append(float(rho))

    r = float(np.mean(rs)) if rs else float("nan")
    sb = (2 * r / (1 + r)) if (rs and r > -1) else float("nan")
    return {"split_half_r": r, "spearman_brown": sb, "n_splits": len(rs), "all": rs}

def _fit_normalised(x, y, feature_names, **kw) -> np.ndarray:
    kw = {k: v for k, v in kw.items() if k not in ("seed", "lambda_grid")}
    lo, hi = minmax_fit(x)
    return fit_linear_l1(minmax_apply(x, lo, hi), y, feature_names, **kw).weights

def bootstrap_r(
    subsets, x, y, feature_names, tasks, n_boot: int = 200, **kw
) -> dict:
    seed = int(kw.pop("seed", 0))
    rng = np.random.default_rng(seed)
    n = len(y)
    idxs = [rng.integers(0, n, size=n) for _ in range(n_boot)]
    out = _run_trials(
        delayed(_trial)(loto_evaluate, [subsets[i] for i in idxs[b]],
                        x[idxs[b]], y[idxs[b]], feature_names, tasks,
                        seed=seed + b, **kw)
        for b in range(n_boot))
    scores = [r["pooled_r"] for r in out if not np.isnan(r["pooled_r"])]
    if not scores:
        return {"lo": float("nan"), "hi": float("nan"), "median": float("nan"), "n": 0}
    return {
        "lo": float(np.percentile(scores, 2.5)),
        "hi": float(np.percentile(scores, 97.5)),
        "median": float(np.median(scores)),
        "n": len(scores),
    }
