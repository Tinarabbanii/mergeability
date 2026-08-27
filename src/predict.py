from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

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

    if np.linalg.norm(w) < 1e-12:
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
    return FitResult(w, r, feature_names)

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
    solver: str = "lasso",) -> dict:
    held_pred = np.full(len(y), np.nan)
    fold_r: list[float] = []
    fold_weights: list[np.ndarray] = []

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

        fit = fit_linear_l1(xtr, y[tr_idx], names, l1_lambda, steps, lr,
                            seed, n_restarts, solver)
        pred = xva @ fit.weights
        held_pred[val_idx] = pred
        fold_weights.append(fit.weights)

        if len(val_idx) >= 3 and np.std(pred) > 1e-12 and np.std(y[val_idx]) > 1e-12:
            fold_r.append(float(np.corrcoef(pred, y[val_idx])[0, 1]))

    mask = ~np.isnan(held_pred)
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
    }

def null_random_features(subsets, y, tasks, n_features: int, n_trials: int = 20, **kw) -> dict:
    seed = int(kw.pop("seed", 0))
    rng = np.random.default_rng(seed)
    scores = []
    names = [f"noise_{i}" for i in range(n_features)]
    for t in range(n_trials):
        x = rng.standard_normal((len(y), n_features))
        res = loto_evaluate(subsets, x, y, names, tasks, seed=seed + t, **kw)
        if not np.isnan(res["pooled_r"]):
            scores.append(res["pooled_r"])
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
    scores = []
    for t in range(n_trials):
        y_shuf = rng.permutation(y)
        res = loto_evaluate(subsets, x, y_shuf, feature_names, tasks, seed=seed + t, **kw)
        if not np.isnan(res["pooled_r"]):
            scores.append(res["pooled_r"])
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
    kw = {k: v for k, v in kw.items() if k != "seed"}
    lo, hi = minmax_fit(x)
    return fit_linear_l1(minmax_apply(x, lo, hi), y, feature_names, **kw).weights

def bootstrap_r(
    subsets, x, y, feature_names, tasks, n_boot: int = 200, **kw
) -> dict:
    seed = int(kw.pop("seed", 0))
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_s = [subsets[i] for i in idx]
        res = loto_evaluate(sub_s, x[idx], y[idx], feature_names, tasks,
                            seed=seed + b, **kw)
        if not np.isnan(res["pooled_r"]):
            scores.append(res["pooled_r"])
    if not scores:
        return {"lo": float("nan"), "hi": float("nan"), "median": float("nan"), "n": 0}
    return {
        "lo": float(np.percentile(scores, 2.5)),
        "hi": float(np.percentile(scores, 97.5)),
        "median": float(np.median(scores)),
        "n": len(scores),
    }
