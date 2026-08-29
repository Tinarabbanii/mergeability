"""Figure generation. One figure per claim.

    fig1  expert accuracy and task vector norm             (from e0)
    fig2  distribution of post-merge accuracy by k, method (from e1)
    fig3  RQ1 HEADLINE: data-free vs full predictor        (from e2)
    fig4  metric importance across merge methods           (from e3)
    fig5  split-half reliability                           (from e3)
    fig6  RQ2: which aggregator predicts k-way             (from e4)
    fig7  observed r against both null baselines           (from e5)
    fig8  RQ2's control: pairwise vs the additive baseline (from e4)

Every figure is regenerated from the cached CSVs, so make_figures.py never
re-runs an experiment. A missing CSV skips that figure instead of crashing.

DESIGN RULES, applied to all eight:

  * HORIZONTAL bars whenever the category labels are long. Rotated x-labels are
    the commonest source of an unreadable plot; turning the chart sideways
    removes the problem instead of shrinking the font until it fits.
  * Every bar carries its own value. A reader should never have to measure a
    bar against an axis to know what it says.
  * Colour is SEMANTIC and consistent across the whole set: data-free is always
    blue, the full/comparison series always coral, nulls and baselines always
    grey. The same colour means the same thing in every figure.
  * Gridlines behind the data, one axis only, very light.
  * Left-aligned title with a grey subtitle carrying the interpretation, so a
    figure can be read without the caption.
  * Nothing is ever drawn on top of anything else. Legends get a narrow
    reserved band under the bars (_room); labels on reference lines are pinned
    inside the axes (_note).
"""

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
    """Left-aligned title with an optional grey interpretation line."""
    ax.set_title(title, loc="left", pad=15 if subtitle else 8,
                 fontweight="bold", color=INK)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.2, color=MUTED)


def _figtitle(fig, title: str, subtitle: str = "") -> None:
    """Figure-level title. Two separate texts with explicit vertical anchors --
    suptitle + fig.text at hand-picked y values collide as soon as the figure
    height changes."""
    fig.text(0.008, 1.075, title, ha="left", va="bottom",
             fontsize=11.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.008, 1.020, subtitle, ha="left", va="bottom",
                 fontsize=8.2, color=MUTED)


def _grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def _room(ax, n: int, rows: float = 0.55) -> None:
    """Reserve a narrow empty band below the bars of a horizontal chart.

    Without this, a legend at "lower right" lands on the last bar and its value
    label. Extending the limit is better than moving the legend outside: the
    figure keeps its rectangle and the legend stays next to what it explains.
    0.55 of a row is the smallest band that still clears a two-line legend.
    """
    ax.set_ylim(-0.5 - rows, n - 0.5)


def _note(ax, x: float, text: str, color=MUTED) -> None:
    """Label a vertical reference line, pinned just INSIDE the top of the axes.

    Data coordinates on x, axes fraction on y, so the label cannot drift above
    the axes and collide with the title however the y limits are set.
    """
    ax.text(x, 0.985, text, transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color=color)


def _pretty(name: str) -> str:
    """weight_averaging -> Weight averaging"""
    return name.replace("_", " ").capitalize()


def _short(name: str) -> str:
    """weight_averaging -> Weight"""
    return _pretty(name).split()[0]


def _no_yticks(ax) -> None:
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)


def _label_barh(ax, bars, values, fmt="{:+.2f}", pad=0.012) -> None:
    """Value at the end of each horizontal bar, outside and on the correct side."""
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

        # jittered raw points: the boxes hide how few subsets there are
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

    b1 = axes[0].barh(y + h / 2, df.full_r, h, color=CORAL, zorder=3,
                      label="full (all five families)")
    b2 = axes[0].barh(y - h / 2, df.data_free_r, h, color=BLUE, zorder=3,
                      label="data-free (weights only)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([_pretty(m) for m in df.method])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlabel("held-out correlation  r  (leave-one-task-out)")
    axes[0].set_xlim(0, max(df.full_r.max(), df.data_free_r.max()) * 1.24)
    _room(axes[0], len(df))
    axes[0].legend(loc="lower right")
    _grid(axes[0], "x")
    _label_barh(axes[0], b1, df.full_r.tolist(), "{:.2f}")
    _label_barh(axes[0], b2, df.data_free_r.tolist(), "{:.2f}")
    _title(axes[0], "Predictive power", "higher is better;  1.0 = perfect prediction")

    pct = 100 * df.retention
    bars = axes[1].barh(y, pct, 0.58, color=[BLUE if p >= 80 else SAND for p in pct],
                        zorder=3)
    axes[1].axvline(100, ls=(0, (4, 3)), c=GREY, lw=1.2, zorder=1)
    axes[1].set_yticks(y); _no_yticks(axes[1])
    axes[1].set_xlabel("% of full-metric power retained")
    axes[1].set_xlim(0, max(pct.max() * 1.2, 118))
    axes[1].set_ylim(axes[0].get_ylim())
    _note(axes[1], 100, "  no loss")
    _grid(axes[1], "x")
    _label_barh(axes[1], bars, pct.tolist(), "{:.0f}%")
    _title(axes[1], "Retention", "what you keep by dropping the data")

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

    # every cell is annotated. A blank cell reads as missing data; an explicit
    # faint 0 reads as "the L1 penalty dropped this metric", which is the point.
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

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([_pretty(c) for c in piv.columns])
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
           "0 = dropped by the L1 penalty")
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

    # horizontal, so the six method/k labels get their own row each and the
    # three value labels per row can never collide the way stacked x-ticks did
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

    # nulls as a shaded "chance band" behind each row, not competing bars
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

    ax.set_yticks(y); ax.set_yticklabels([_pretty(m) for m in df.method])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("held-out correlation  r")
    ax.set_xlim(min(0, df.observed_r.min() - 0.15), 1.05)
    ax.axvline(0, c="#D2D8DC", lw=1)
    _room(ax, len(df))
    _grid(ax, "x")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.8,
               ms=8, label="observed  (95% CI)"),
        Patch(facecolor=GREY, alpha=0.45, label="what chance alone reaches (p95)"),
    ], loc="lower right", ncol=2)

    _title(ax, "Does the result clear chance?",
           "the grey band is the null -- a result inside it is not evidence")
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


ALL = [fig1_setup, fig2_merge_quality, fig3_datafree, fig4_importance,
       fig5_reliability, fig6_kway, fig7_nulls, fig8_additive]


def make_all(cfg: Config) -> None:
    print(f"generating figures  [backend={cfg.backend}]")
    for fn in ALL:
        try:
            fn(cfg)
        except Exception as exc:
            print(f"  skipped {fn.__name__}: {type(exc).__name__}: {exc}")
