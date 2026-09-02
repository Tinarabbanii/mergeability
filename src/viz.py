"""Figure generation. One figure per claim.
    fig1  expert accuracy and task vector norm             (from e0)
    fig2  distribution of post-merge accuracy by k, method (from e1)
    fig3  RQ1 HEADLINE: data-free vs full predictor        (from e2)
    fig4  metric importance across merge methods           (from e3)
    fig5  split-half reliability                           (from e3)
    fig6  RQ2: which aggregator predicts k-way             (from e4)
    fig7  observed r against both null baselines           (from e5)
    fig8  RQ2's control: pairwise vs the additive baseline (from e4)
    fig9  item 3: TIES density sweep, quality vs predictability
    fig10 item 4: calibration size, metric stability vs prediction"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from .config import Config  # noqa: E402

# --------------------------------------------------------------------------- #
# palette -- muted, print-safe, colour-blind friendly ordering
# --------------------------------------------------------------------------- #
BLUE   = "#7BA7CC"   # data-free / primary series
CORAL  = "#E8927C"   # full metric set / secondary series
SAGE   = "#8FBC94"   # third series
SAND   = "#E9C46A"   # fourth series
MAUVE  = "#B08EAD"   # fifth series
GREY   = "#BCC5CC"   # nulls, chance levels, reference lines
INK    = "#2F3437"   # text
MUTED  = "#7A8388"   # secondary text
GRID   = "#E6EAED"
FAINT  = "#AFB8BE"

DIVERGING = LinearSegmentedColormap.from_list("mrg", ["#4A7C9B", "#EDF2F5", "#D96C52"])

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 9.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": "#D2D8DC",
    "axes.labelsize": 9,
    "axes.titlesize": 10.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _save(fig, cfg: Config, name: str) -> None:
    path = cfg.figure(name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")

def _read(cfg: Config, name: str) -> pd.DataFrame | None:
    p = cfg.artifact(name)
    return pd.read_csv(p) if p.exists() else None


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", pad=18 if subtitle else 8,
                 fontweight="bold", color=INK)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.2, color=MUTED)


def _figtitle(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.008, 1.075, title, ha="left", va="bottom",
             fontsize=11.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.008, 1.020, subtitle, ha="left", va="bottom",
                 fontsize=8.2, color=MUTED)


def _grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def _room(ax, n: int, rows: float = 0.55) -> None:
    ax.set_ylim(-0.5 - rows, n - 0.5)


def _note(ax, x: float, text: str, color=MUTED) -> None:
    ax.text(x, 0.985, text, transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color=color)


def _pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _short(name: str) -> str:
    return _pretty(name).split()[0]


def _no_yticks(ax) -> None:
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)


def _label_barh(ax, bars, values, fmt="{:+.2f}", pad=0.012) -> None:
    span = max((abs(v) for v in values if not np.isnan(v)), default=1.0) or 1.0
    for bar, v in zip(bars, values):
        if np.isnan(v):
            continue
        off = pad * span * (1 if v >= 0 else -1)
        ax.text(v + off, bar.get_y() + bar.get_height() / 2, fmt.format(v),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8.2, color=INK)


def _label_barv(ax, bars, values, fmt="{:.2f}") -> None:
    span = max((abs(v) for v in values if not np.isnan(v)), default=1.0) or 1.0
    for bar, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015 * span, fmt.format(v),
                ha="center", va="bottom", fontsize=8.2, color=INK)


# =========================================================================== #
def _cleared_methods(cfg: Config, name: str = "e5_nulls_k2.csv") -> set | None:
    """Methods whose observed correlation beat BOTH null baselines, from e5.

    fig3 draws e2's correlations, but only e5 knows which of them are real.
    Returns None when e5 has not been run, so fig3 still renders standalone.
    """
    d = _read(cfg, name)
    if d is None or d.empty or "clears_both_nulls" not in d:
        return None
    return set(d.loc[d.clears_both_nulls.astype(bool), "method"])

def fig1_setup(cfg: Config) -> None:
    """Expert accuracy and task-vector norm per task."""
    df = _read(cfg, "e0_sanity.csv")
    if df is None or df.empty:
        return
    df = df.sort_values("expert_accuracy")
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.46 * len(df) + 2.0))

    b = axes[0].barh(y, df.expert_accuracy, color=BLUE, height=0.60, zorder=3)
    if "pretrained_accuracy" in df:
        axes[0].barh(y, df.pretrained_accuracy, color=GREY, height=0.26, zorder=4,
                     label="pretrained (zero-shot)")
        _room(axes[0], len(df))
        axes[0].legend(loc="lower right")
    axes[0].set_yticks(y); axes[0].set_yticklabels(df.task)
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlim(0, 1.10); axes[0].set_xlabel("accuracy")
    _grid(axes[0], "x")
    _label_barh(axes[0], b, df.expert_accuracy.tolist(), "{:.3f}")
    _title(axes[0], "Expert accuracy", "each model on the task it was tuned for")

    b2 = axes[1].barh(y, df.task_vector_norm, color=CORAL, height=0.60, zorder=3)
    axes[1].set_yticks(y); _no_yticks(axes[1])
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlabel(r"$\|\tau\|_2$")
    axes[1].set_xlim(0, df.task_vector_norm.max() * 1.22)
    _grid(axes[1], "x")
    _label_barh(axes[1], b2, df.task_vector_norm.tolist(), "{:.2f}")
    _title(axes[1], "Task vector magnitude", "how far fine-tuning moved the weights")

    fig.tight_layout(w_pad=2.0)
    _save(fig, cfg, "fig1_setup.png")


def fig2_merge_quality(cfg: Config) -> None:
    """Distribution of post-merge accuracy, by subset size and method."""
    df = _read(cfg, "results.csv")
    if df is None or df.empty:
        return
    ks = sorted(df.k.unique())
    methods = sorted(df.method.unique())
    colours = [BLUE, CORAL, SAGE, SAND, MAUVE][:len(methods)]

    fig, axes = plt.subplots(1, len(ks), figsize=(2.9 * len(ks) + 0.8, 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, k in zip(axes, ks):
        block = df[df.k == k]
        data = [block[block.method == m]["normalized_accuracy"].values for m in methods]
        parts = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                           medianprops=dict(color=INK, lw=1.4),
                           whiskerprops=dict(color="#B4BCC2", lw=1),
                           capprops=dict(color="#B4BCC2", lw=1))
        for patch, c in zip(parts["boxes"], colours):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("none")
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data, start=1):
            ax.scatter(rng.normal(i, 0.055, len(vals)), vals, s=9,
                       color=INK, alpha=0.35, zorder=5, linewidths=0)

        ax.axhline(1.0, ls=(0, (4, 3)), c=GREY, lw=1.2, zorder=1)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([_pretty(m).replace(" ", "\n") for m in methods])
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"k = {k}", loc="left", color=MUTED, fontsize=9.5)
        _grid(ax)

    axes[0].set_ylabel("post-merge normalised accuracy")
    fig.tight_layout()
    _figtitle(fig, "Merge quality by subset size and method",
              "dashed line = no loss from merging;  dots = individual subsets")
    _save(fig, cfg, "fig2_merge_quality.png")


def fig3_datafree(cfg: Config) -> None:
    """RQ1 HEADLINE -- how much predictive power survives without data."""
    df = _read(cfg, "e2_datafree_k2.csv")
    if df is None or df.empty:
        return
    df = df.sort_values("retention")
    y = np.arange(len(df)); h = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 0.80 * len(df) + 2.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # Which methods actually beat their null? e5 knows; e2 does not. Without this,
    # a method whose correlation is indistinguishable from noise gets the same
    # solid bar as one that clears chance, and fig3 contradicts fig7.
    cleared = _cleared_methods(cfg)
    solid = [m in cleared for m in df.method] if cleared is not None else [True] * len(df)

    b1 = axes[0].barh(y + h / 2, df.full_r, h, zorder=3,
                      color=[CORAL if s else FAINT for s in solid],
                      label="full (all five families)")
    b2 = axes[0].barh(y - h / 2, df.data_free_r, h, zorder=3,
                      color=[BLUE if s else FAINT for s in solid],
                      label="data-free (weights only)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([_pretty(m) for m in df.method])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlabel("held-out correlation  r  (leave-one-task-out)")
    lo = min(0.0, df.full_r.min(), df.data_free_r.min()) * 1.30
    axes[0].set_xlim(lo, max(df.full_r.max(), df.data_free_r.max()) * 1.24)
    _room(axes[0], len(df))
    axes[0].legend(loc="lower right")
    _grid(axes[0], "x")
    _label_barh(axes[0], b1, df.full_r.tolist(), "{:.2f}")
    _label_barh(axes[0], b2, df.data_free_r.tolist(), "{:.2f}")
    sub = "higher is better;  1.0 = perfect prediction"
    if cleared is not None and not all(solid):
        sub += ";  grey = does not clear its null"
    _title(axes[0], "Predictive power", sub)

    pct = 100 * df.retention
    # retention is a ratio of two correlations; it is only meaningful when the
    # underlying correlation is itself distinguishable from noise.
    bars = axes[1].barh(y, pct, 0.58, zorder=3,
                        color=[(BLUE if p >= 80 else SAND) if s else FAINT
                               for p, s in zip(pct, solid)])
    axes[1].axvline(100, ls=(0, (4, 3)), c=GREY, lw=1.2, zorder=1)
    axes[1].set_yticks(y); _no_yticks(axes[1])
    axes[1].set_xlabel("% of full-metric power retained")
    axes[1].set_xlim(0, max(pct.max() * 1.2, 118))
    axes[1].set_ylim(axes[0].get_ylim())
    _note(axes[1], 100, "  no loss")
    _grid(axes[1], "x")
    _label_barh(axes[1], bars, pct.tolist(), "{:.0f}%")
    sub2 = "what you keep by dropping the data"
    if cleared is not None and not all(solid):
        sub2 += ";  grey bars rest on a null result"
    _title(axes[1], "Retention", sub2)

    fig.tight_layout(w_pad=2.4)
    _save(fig, cfg, "fig3_datafree.png")


def fig4_importance(cfg: Config) -> None:
    """Which weight-space properties predict merge success, per method."""
    df = _read(cfg, "e3_coefficients_k2.csv")
    if df is None or df.empty:
        return
    piv = df.pivot(index="metric", columns="method", values="coefficient")
    piv = piv.reindex(piv.abs().max(axis=1).sort_values().index)   # weakest at top
    m = piv.to_numpy()
    v = np.abs(m).max() or 1.0

    fig, ax = plt.subplots(figsize=(1.55 * len(piv.columns) + 4.6,
                                    0.36 * len(piv) + 2.1))
    im = ax.imshow(m, cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto")

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m[i, j]
            if np.isnan(val):
                continue
            if val == 0:
                ax.text(j, i, "0", ha="center", va="center", fontsize=7.5, color=FAINT)
            else:
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.8,
                        color="white" if abs(val) > 0.62 * v else INK)

    cleared = _cleared_methods(cfg)
    labels = []
    for j, c in enumerate(piv.columns):
        if cleared is not None and c not in cleared:
            ax.axvspan(j - 0.5, j + 0.5, color="white", alpha=0.62, zorder=3)
            labels.append(f"{_pretty(c)}\n(at chance)")
        else:
            labels.append(_pretty(c))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([i.replace("_", " ") for i in piv.index], fontsize=8.2)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-.5, len(piv.columns)), minor=True)
    ax.set_yticks(np.arange(-.5, len(piv.index)), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.025)
    cb.outline.set_visible(False)
    cb.set_label("coefficient", color=MUTED, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, color=MUTED)

    _title(ax, "Which data-free metrics carry the signal",
           "red = higher metric predicts better merging,  blue = the opposite,  "
           "0 = dropped by the L1 penalty;  faded = method does not clear its null")
    fig.tight_layout()
    _save(fig, cfg, "fig4_importance.png")


def fig5_reliability(cfg: Config) -> None:
    """Is the metric-importance ordering reproducible, or resampling noise?"""
    df = _read(cfg, "e3_reliability_k2.csv")
    if df is None or df.empty:
        return
    y = np.arange(len(df)); h = 0.32

    fig, ax = plt.subplots(figsize=(7.6, 0.80 * len(df) + 2.0))
    b1 = ax.barh(y + h / 2, df.spearman_brown, h, color=BLUE, zorder=3,
                 label="Spearman-Brown (corrected)")
    b2 = ax.barh(y - h / 2, df.split_half_r, h, color=GREY, zorder=3,
                 label="split-half r (raw)")

    ax.set_yticks(y); ax.set_yticklabels([_pretty(m) for m in df.method])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("rank correlation between two halves of the data")
    ax.set_xlim(min(0, df.split_half_r.min() * 1.3), 1.0)
    _room(ax, len(df))

    for xv, lab, c in ((0.4, "moderate", SAND), (0.7, "reliable", SAGE)):
        ax.axvline(xv, ls=(0, (4, 3)), c=c, lw=1.3, zorder=1)
        _note(ax, xv, f" {lab}", c)

    ax.legend(loc="lower right")
    _grid(ax, "x")
    _label_barh(ax, b1, df.spearman_brown.tolist())
    _label_barh(ax, b2, df.split_half_r.tolist())
    _title(ax, "Is the metric ordering reproducible?",
           "split the subsets in half, refit, compare -- low means the ranking is noise")
    fig.tight_layout()
    _save(fig, cfg, "fig5_reliability.png")


def fig6_kway(cfg: Config) -> None:
    """RQ2 -- which aggregation of pairwise information predicts k-way merging."""
    a = _read(cfg, "e4_oracle.csv")
    b = _read(cfg, "e4_transfer.csv")
    if a is None or a.empty:
        return

    panels = [(a, "Test A", "aggregating measured pairwise accuracy")]
    if b is not None and not b.empty:
        panels.append((b, "Test B", "transferring the k = 2 predictor"))

    n = len(a)
    fig, axes = plt.subplots(1, len(panels), sharey=True,
                             figsize=(4.9 * len(panels) + 0.8, 0.52 * n + 2.3))
    axes = np.atleast_1d(axes)
    aggs = [("mean", BLUE), ("min", CORAL), ("max", SAND)]
    h = 0.24

    for ax, (df, tag, sub) in zip(axes, panels):
        y = np.arange(len(df))[::-1]          # first row at the top
        for off, (name, c) in zip((h, 0, -h), aggs):
            col = f"r_{name}"
            if col not in df:
                continue
            bars = ax.barh(y + off, df[col], h, color=c, zorder=3, label=name)
            _label_barh(ax, bars, df[col].tolist(), "{:.2f}")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{_short(m)}  k={k}" for m, k in zip(df.method, df.k)],
                           fontsize=8.4)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("correlation with k-way accuracy")
        _room(ax, len(df), rows=0.85)   # this legend carries a title row
        _grid(ax, "x")
        _title(ax, tag, sub)

    axes[0].legend(title="aggregator", title_fontsize=8.5, loc="lower right", ncol=3)
    fig.tight_layout(w_pad=2.0)
    _figtitle(fig, "Does pairwise mergeability predict groups?",
              "three ways to summarise the pairs inside a group, scored against the "
              "group's real merge quality")
    _save(fig, cfg, "fig6_kway.png")


def fig7_nulls(cfg: Config) -> None:
    """Does the observed correlation clear what chance alone produces?"""
    df = _read(cfg, "e5_nulls_k2.csv")
    if df is None or df.empty:
        return
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.6, 1.0 * len(df) + 2.2))

    for i, row in df.reset_index().iterrows():
        hi = max(row.get("null_random_p95", 0), row.get("null_shuffled_p95", 0))
        ax.barh(i, hi, 0.72, color=GREY, alpha=0.45, zorder=2)

    err = None
    if {"ci_lo", "ci_hi"} <= set(df.columns):
        err = np.clip(np.vstack([df.observed_r - df.ci_lo,
                                 df.ci_hi - df.observed_r]), 0, None)

    clears = df.observed_r > df[["null_random_p95", "null_shuffled_p95"]].max(axis=1) \
        if {"null_random_p95", "null_shuffled_p95"} <= set(df.columns) \
        else pd.Series([True] * len(df))

    ax.errorbar(df.observed_r, y, xerr=err, fmt="o", ms=8, zorder=5,
                color=INK, ecolor="#8C959B", elinewidth=1.4, capsize=4,
                markerfacecolor="white", markeredgewidth=1.8)

    for i, (v, ok) in enumerate(zip(df.observed_r, clears)):
        ax.scatter(v, i, s=26, zorder=6, color=BLUE if ok else CORAL, edgecolors="none")

    # The same fit with metrics collapsed to one feature per family. Fewer free
    # parameters lower the null, which is the whole point -- so the grouped null
    # is drawn as its own shorter band and the grouped estimate as a small square.
    has_grp = {"grouped_r", "grouped_null_p95"} <= set(df.columns)
    if has_grp:
        for i, row in df.reset_index().iterrows():
            ax.barh(i - 0.30, row["grouped_null_p95"], 0.14, color=SAGE,
                    alpha=0.40, zorder=3)
        gclears = (df.grouped_clears.astype(bool) if "grouped_clears" in df
                   else df.grouped_r > df.grouped_null_p95)
        ax.scatter(df.grouped_r, y - 0.30, marker="s", s=34, zorder=6,
                   color=[SAGE if g else CORAL for g in gclears],
                   edgecolors="white", linewidths=1.0)

    ax.set_yticks(y); ax.set_yticklabels([_pretty(m) for m in df.method])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("held-out correlation  r")
    lo = min(0, df.observed_r.min(), df.ci_lo.min() if "ci_lo" in df else 0) - 0.05
    ax.set_xlim(lo, 1.05)
    ax.axvline(0, c="#D2D8DC", lw=1)
    _room(ax, len(df), rows=0.75)
    _grid(ax, "x")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.8,
               ms=8, label="all metrics  (95% CI)"),
        Patch(facecolor=GREY, alpha=0.45, label="its null (p95)"),
    ]
    if has_grp:
        handles += [
            Line2D([], [], marker="s", ls="none", mfc=SAGE, mec="white", mew=1.0,
                   ms=7, label="one feature per family"),
            Patch(facecolor=SAGE, alpha=0.40, label="its null (p95)"),
        ]
    ax.legend(handles=handles, loc="lower right", ncol=2)

    _title(ax, "Does the result clear chance?",
           "a marker inside its own band is not evidence;  fewer features, lower band")
    fig.tight_layout()
    _save(fig, cfg, "fig7_nulls.png")


def fig8_additive(cfg: Config) -> None:
    """RQ2's control -- is k-way merging simply additive in the tasks?"""
    df = _read(cfg, "e4_additive.csv")
    if df is None or df.empty:
        return
    y = np.arange(len(df)); h = 0.32
    labels = [f"{_short(m)}  k={k}" for m, k in zip(df.method, df.k)]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 0.60 * len(df) + 2.2),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    b1 = axes[0].barh(y + h / 2, df.r_pairwise_mean, h, color=BLUE, zorder=3,
                      label="pairwise mean")
    b2 = axes[0].barh(y - h / 2, df.r_additive, h, color=GREY, zorder=3,
                      label="additive task-level baseline")
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8.4)
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlabel("correlation with k-way accuracy")
    axes[0].set_xlim(0, 1.18)
    _room(axes[0], len(df))
    axes[0].legend(loc="lower right")
    _grid(axes[0], "x")
    _label_barh(axes[0], b1, df.r_pairwise_mean.tolist(), "{:.3f}")
    _label_barh(axes[0], b2, df.r_additive.tolist(), "{:.3f}")
    _title(axes[0], "Pairwise vs the additive baseline",
           "nearly equal means the pairs add nothing")

    inc = df.increment_r
    bars = axes[1].barh(y, inc, 0.56, zorder=3,
                        color=[SAGE if v > 0.2 else GREY for v in inc])
    axes[1].axvline(0.2, ls=(0, (4, 3)), c=MUTED, lw=1.1, zorder=1)
    axes[1].set_yticks(y); _no_yticks(axes[1])
    axes[1].set_xlabel("partial r, after removing the additive part")
    axes[1].set_xlim(min(0, inc.min() * 1.25), max(inc.max() * 1.3, 0.35))
    axes[1].set_ylim(axes[0].get_ylim())
    _note(axes[1], 0.2, "  threshold")
    _grid(axes[1], "x")
    _label_barh(axes[1], bars, inc.tolist(), "{:.3f}")
    _title(axes[1], "Genuine higher-order signal",
           "what the pairs explain that the tasks alone do not")

    fig.tight_layout(w_pad=2.2)
    _save(fig, cfg, "fig8_additive.png")


def fig9_density(cfg: Config) -> None:
    sweep = _read(cfg, "density_sweep.csv")
    pred = _read(cfg, "density_predictability.csv")
    if sweep is None or pred is None or sweep.empty or pred.empty:
        return

    g = sweep.groupby("density").normalized_accuracy.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    ax.errorbar(g.density, g["mean"], yerr=g["std"], marker="o", ms=5.5,
                color=BLUE, ecolor=FAINT, elinewidth=1.1, capsize=3, zorder=3)
    ax.axvline(0.2, color=CORAL, lw=1.2, ls="--", zorder=2)
    ax.text(0.21, ax.get_ylim()[0], "  paper's 0.2", color=CORAL, fontsize=8,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("TIES density (fraction of weights kept)")
    ax.set_ylabel("normalised accuracy")
    _grid(ax); _title(ax, "Merge quality rises, then plateaus",
                      "error bars: std across the 21 pairs")

    ax = axes[1]
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.plot(pred.density, pred.data_free_r, marker="o", ms=5.5, color=BLUE,
            label="data-free", zorder=3)
    ax.plot(pred.density, pred.full_r, marker="s", ms=5.0, color=CORAL,
            label="full metric set", zorder=3)
    ax.axvline(1.0, color=SAGE, lw=1.2, ls="--", zorder=2)
    ax.text(0.99, 0.02, "trim OFF  ", color=SAGE, fontsize=8, rotation=90,
            va="bottom", ha="right", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("TIES density (fraction of weights kept)")
    ax.set_ylabel("held-out r (LOTO)")
    ax.legend(loc="lower left", fontsize=8.5)
    _grid(ax); _title(ax, "Predictability does not follow",
                      "at or below zero throughout, including with the trim off")

    _figtitle(fig, "Density sweep (k=2): the trim is not the cause of TIES's unpredictability",
              "quality responds to density; predictability does not. Pairs only -- "
              "TIES behaves differently at k=4, which this sweep did not test")
    _save(fig, cfg, "fig9_density.png")


def fig10_calibration(cfg: Config) -> None:
    stab = _read(cfg, "calibration_stability.csv")
    pred = _read(cfg, "calibration_prediction.csv")
    if stab is None or pred is None or stab.empty or pred.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.40 * len(stab) + 2.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    d = stab.sort_values("corr_10_vs_100")
    y = np.arange(len(d))
    cols = [GREY if v < 0.3 else BLUE for v in d.corr_10_vs_100]
    bars = ax.barh(y, d.corr_10_vs_100, 0.6, color=cols, zorder=3)
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(d.metric, fontsize=8.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.35, 1.12)
    ax.set_xlabel("corr(10 samples, 100 samples)")
    _grid(ax, "x")
    _label_barh(ax, bars, d.corr_10_vs_100.tolist(), "{:+.2f}")
    _title(ax, "Metric stability across calibration size",
           "grey: below 0.3, not a stable quantity at 10 samples")

    ax = axes[1]
    y = np.arange(len(pred)); h = 0.30
    b1 = ax.barh(y + h / 2, pred.full_cal10, h, color=GREY, zorder=3,
                 label="full, 10 samples")
    b2 = ax.barh(y - h / 2, pred.full_cal100, h, color=CORAL, zorder=3,
                 label="full, 100 samples")
    ax.scatter(pred.data_free_cal10, y, marker="D", s=34, color=BLUE, zorder=4,
               label="data-free (calibration-independent)")
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([_pretty(m) for m in pred.method], fontsize=8.4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("held-out r (LOTO)")
    ax.legend(loc="upper right", fontsize=8)
    _grid(ax, "x")
    _label_barh(ax, b1, pred.full_cal10.tolist(), "{:+.2f}")
    _label_barh(ax, b2, pred.full_cal100.tolist(), "{:+.2f}")
    _title(ax, "More calibration data does not help",
           "data-free still dominates at 10x the samples")

    _figtitle(fig, "Calibration size: a real flaw that is not the explanation",
              "two gradient metrics are unstable, yet fixing that does not rescue the full set")
    _save(fig, cfg, "fig10_calibration.png")


ALL = [fig1_setup, fig2_merge_quality, fig3_datafree, fig4_importance,
       fig5_reliability, fig6_kway, fig7_nulls, fig8_additive,
       fig9_density, fig10_calibration]


def make_all(cfg: Config) -> None:
    print(f"generating figures  [backend={cfg.backend}]")
    for fn in ALL:
        try:
            fn(cfg)
        except Exception as exc:
            print(f"  skipped {fn.__name__}: {type(exc).__name__}: {exc}")
