from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
ap.add_argument("--out", type=Path, default=ROOT / "figures" / "report")
args = ap.parse_args()

A = args.artifacts
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)

BLUE, CORAL, GREY, INK, MUTED = "#4C7FA8", "#D9694A", "#C3CBD1", "#22282C", "#6B767E"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.6,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": 7.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#95A0A8", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 300,
})

fig, axes = plt.subplots(1, 2, figsize=(3.32, 1.32), sharey=True)
for ax, (b, lab) in zip(axes, (("clip", "ViT-B/32"), ("clip16", "ViT-B/16"))):
    ks = [2, 3, 4]
    free, full, null = [], [], []
    for k in ks:
        e2 = pd.read_csv(A / b / f"e2_datafree_k{k}.csv")
        e5 = pd.read_csv(A / b / f"e5_nulls_k{k}.csv")
        free.append(e2.data_free_r.mean())
        full.append(e2.full_r.mean())
        null.append(e5[["null_random_p95", "null_shuffled_p95"]].max(axis=1).mean())
    tr = pd.read_csv(A / b / "e4_transfer.csv")
    trv = [tr[tr.k == k].r_mean.mean() for k in (3, 4)]
    ax.fill_between(ks, 0, null, color=GREY, alpha=0.55, lw=0, zorder=1)
    ax.plot(ks, null, color="#8E9AA2", lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.plot(ks, full, "s--", color=CORAL, ms=3.4, lw=1.2, zorder=3, label="full (18)")
    ax.plot(ks, free, "o-", color=BLUE, ms=4.0, lw=1.5, zorder=4, label="data-free (11)")
    ax.plot([3, 4], trv, "^:", color="#5F8A5B", ms=4.0, lw=1.3, zorder=5,
            label="fit on pairs only")
    ax.set_xticks(ks)
    ax.set_xlabel("group size $k$")
    ax.set_title(lab, pad=3)
    ax.set_ylim(0, 0.86)
    ax.grid(axis="y", color="#E7ECEF", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("held-out $r$")
h, l = axes[1].get_legend_handles_labels()
h.append(plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.55)); l.append("chance")
fig.tight_layout(pad=0.25, w_pad=0.9, rect=(0, 0.10, 1, 1))
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.4,
           handlelength=1.4, columnspacing=1.0, handletextpad=0.4,
           bbox_to_anchor=(0.54, -0.03))
fig.savefig(OUT / "main_result.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print(f"  {OUT}/main_result.pdf")

order = ["act_dot", "act_l2", "act_cosine", "act_magnitude_ratio",
         "grad_l2", "grad_dot", "grad_cosine"]
pretty = {"act_dot": "act dot", "act_l2": "act $L_2$", "act_cosine": "act cos",
          "act_magnitude_ratio": "act mag. ratio", "grad_l2": "grad $L_2$",
          "grad_dot": "grad dot", "grad_cosine": "grad cos"}
s32 = pd.read_csv(A / "clip" / "calibration_stability.csv").set_index("metric")
s16 = pd.read_csv(A / "clip16" / "calibration_stability.csv").set_index("metric")

fig, ax = plt.subplots(figsize=(3.32, 1.38))
y = np.arange(len(order))[::-1]
h = 0.36
v32 = [s32.loc[m, "corr_10_vs_100"] for m in order]
v16 = [s16.loc[m, "corr_10_vs_100"] for m in order]
ax.barh(y + h / 2, v32, h, color=BLUE, zorder=3, label="ViT-B/32")
ax.barh(y - h / 2, v16, h, color=CORAL, zorder=3, label="ViT-B/16")
ax.axvline(0, color=INK, lw=0.9, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([pretty[m] for m in order])
ax.set_xlim(-0.62, 1.08)
ax.set_xlabel(r"corr(10 samples, 100 samples)")
ax.grid(axis="x", color="#E7ECEF", lw=0.7, zorder=0)
ax.set_axisbelow(True)
ax.axhspan(-0.5, 2.5, color="#F3E7E3", alpha=0.75, zorder=0)
ax.text(-0.58, 2.62, "gradient family", fontsize=6.9, color=CORAL, style="italic")
h, l = ax.get_legend_handles_labels()
fig.tight_layout(pad=0.25, rect=(0, 0.075, 1, 1))
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6.8,
           handlelength=1.3, columnspacing=1.6, handletextpad=0.5,
           bbox_to_anchor=(0.56, -0.02))
fig.savefig(OUT / "stability.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print(f"  {OUT}/stability.pdf")

fig, axes = plt.subplots(1, 2, figsize=(3.32, 0.72),
                         gridspec_kw={"width_ratios": [1.0, 1.25]})

ax = axes[0]
ks = [2, 3, 4]
free, full, null = [], [], []
for k in ks:
    e2 = pd.read_csv(A / "clip" / f"e2_datafree_k{k}.csv")
    e5 = pd.read_csv(A / "clip" / f"e5_nulls_k{k}.csv")
    free.append(e2.data_free_r.mean())
    full.append(e2.full_r.mean())
    null.append(e5[["null_random_p95", "null_shuffled_p95"]].max(axis=1).mean())
ax.fill_between(ks, 0, null, color=GREY, alpha=0.55, lw=0, zorder=1)
ax.plot(ks, full, "s--", color=CORAL, ms=2.8, lw=1.0, zorder=3, label="full")
ax.plot(ks, free, "o-", color=BLUE, ms=3.2, lw=1.3, zorder=4, label="data-free")
ax.set_xticks(ks); ax.set_ylim(0, 0.86)
ax.set_xlabel("group size $k$", fontsize=7.2, labelpad=1.5)
ax.set_ylabel("held-out $r$", fontsize=7.2, labelpad=1.5)
ax.tick_params(labelsize=6.4, pad=1.5)
ax.text(2.06, 0.09, "chance", fontsize=6.0, color=MUTED, style="italic")
ax.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.2,
          borderpad=0.0, labelspacing=0.15, borderaxespad=0.1)
ax.grid(axis="y", color="#E7ECEF", lw=0.6, zorder=0); ax.set_axisbelow(True)

ax = axes[1]
short = {"act_dot": "a.dot", "act_l2": "a.$L_2$", "act_cosine": "a.cos",
         "act_magnitude_ratio": "a.mag", "grad_l2": "g.$L_2$",
         "grad_dot": "g.dot", "grad_cosine": "g.cos"}
y = np.arange(len(order))[::-1]
ax.barh(y + 0.19, [s32.loc[m, "corr_10_vs_100"] for m in order], 0.36,
        color=BLUE, zorder=3, label="B/32")
ax.barh(y - 0.19, [s16.loc[m, "corr_10_vs_100"] for m in order], 0.36,
        color=CORAL, zorder=3, label="B/16")
ax.axhspan(-0.5, 2.5, color="#F3E7E3", alpha=0.8, zorder=0)
ax.axvline(0, color=INK, lw=0.8, zorder=4)
ax.set_yticks(y); ax.set_yticklabels([short[m] for m in order], fontsize=6.2)
ax.set_xlim(-0.62, 1.05); ax.set_xticks([-0.5, 0, 0.5, 1.0])
ax.tick_params(axis="x", labelsize=6.4, pad=1.5)
ax.tick_params(axis="y", pad=1.0, length=0)
ax.set_xlabel("corr(10, 100 samples)", fontsize=7.2, labelpad=1.5)
ax.legend(loc="lower right", frameon=False, fontsize=6.2, handlelength=1.0,
          borderpad=0.0, labelspacing=0.15, borderaxespad=0.15)
ax.grid(axis="x", color="#E7ECEF", lw=0.6, zorder=0); ax.set_axisbelow(True)

fig.tight_layout(pad=0.15, w_pad=0.6)
fig.savefig(OUT / "compact.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print(f"  {OUT}/compact.pdf")

fig, axes = plt.subplots(1, 2, figsize=(3.32, 0.98), sharey=True)
for ax, (b, lab) in zip(axes, (("clip", "ViT-B/32"), ("clip16", "ViT-B/16"))):
    ks = [2, 3, 4]; free = []; full = []; null = []
    for k in ks:
        e2 = pd.read_csv(A / b / f"e2_datafree_k{k}.csv")
        e5 = pd.read_csv(A / b / f"e5_nulls_k{k}.csv")
        free.append(e2.data_free_r.mean()); full.append(e2.full_r.mean())
        null.append(e5[["null_random_p95", "null_shuffled_p95"]].max(axis=1).mean())
    tr = pd.read_csv(A / b / "e4_transfer.csv")
    trv = [tr[tr.k == k].r_mean.mean() for k in (3, 4)]
    ax.fill_between(ks, 0, null, color=GREY, alpha=0.6, lw=0, zorder=1)
    ax.plot(ks, full, "s--", color=CORAL, ms=3.0, lw=1.1, zorder=3, label="full (18)")
    ax.plot(ks, free, "o-", color=BLUE, ms=3.4, lw=1.4, zorder=4, label="data-free (11)")
    ax.plot([3, 4], trv, "^:", color="#5F8A5B", ms=3.4, lw=1.2, zorder=5,
            label="fit on pairs only")
    ax.set_xticks(ks); ax.set_ylim(0, 0.86)
    ax.set_xlabel("group size $k$", fontsize=7.4, labelpad=1.5)
    ax.set_title(lab, fontsize=7.6, pad=2)
    ax.tick_params(labelsize=6.6, pad=1.5)
    ax.grid(axis="y", color="#E7ECEF", lw=0.6, zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("held-out $r$", fontsize=7.4, labelpad=1.5)
h, l = axes[1].get_legend_handles_labels()
h.append(plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.6)); l.append("chance")
fig.tight_layout(pad=0.15, w_pad=0.7, rect=(0, 0.11, 1, 1))
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=5.8,
           handlelength=1.2, columnspacing=0.8, handletextpad=0.35,
           bbox_to_anchor=(0.54, -0.04))
fig.savefig(OUT / "body.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print(f"  {OUT}/body.pdf")

fig, ax = plt.subplots(figsize=(3.32, 0.80))
act = ["act_dot", "act_l2", "act_cosine", "act_magnitude_ratio"]
grd = ["grad_l2", "grad_dot", "grad_cosine"]
for lab, tab, yy in (("ViT-B/32", s32, 1.0), ("ViT-B/16", s16, 0.0)):
    ax.plot([tab.loc[m, "corr_10_vs_100"] for m in act], [yy + 0.20] * len(act),
            "o", ms=5.0, color=BLUE, alpha=0.88, mec="white", mew=0.7, zorder=4)
    ax.plot([tab.loc[m, "corr_10_vs_100"] for m in grd], [yy - 0.20] * len(grd),
            "D", ms=4.6, color=CORAL, alpha=0.88, mec="white", mew=0.7, zorder=4)
ax.axvspan(-0.68, 0.0, color="#F6EAE6", zorder=0)
ax.axvline(0, color=INK, lw=0.9, zorder=2)
ax.set_yticks([0.0, 1.0]); ax.set_yticklabels(["B/16", "B/32"], fontsize=7.0)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_ylim(-0.92, 1.42); ax.set_xlim(-0.68, 1.06)
ax.set_xticks([-0.5, 0, 0.5, 1.0])
ax.tick_params(axis="x", labelsize=6.8, pad=1.5)
ax.set_xlabel("agreement of a metric with itself at 10 vs 100 samples",
              fontsize=7.2, labelpad=1.5)
ax.plot([], [], "o", ms=4.4, color=BLUE, mec="white", mew=0.6, label="activation")
ax.plot([], [], "D", ms=4.1, color=CORAL, mec="white", mew=0.6, label="gradient")
h, l = ax.get_legend_handles_labels()
ax.text(-0.645, -0.80, "anti-correlated with itself", fontsize=6.4,
        color=CORAL, style="italic")
ax.grid(axis="x", color="#E7ECEF", lw=0.6, zorder=1); ax.set_axisbelow(True)
for sp in ("left", "right", "top"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.12, rect=(0, 0.0, 1, 0.84))
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.6,
           handlelength=0.9, columnspacing=1.6, handletextpad=0.4,
           bbox_to_anchor=(0.56, 1.03))
fig.savefig(OUT / "stability_strip.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print(f"  {OUT}/stability_strip.pdf")

NICE = {"tv_norm_mean": "mean norm", "tv_l2": "distance", "tv_dot": "inner prod.",
        "tv_norm_ratio": "norm ratio", "tv_cosine": "cosine (angle)",
        "eff_rank_global": "eff. rank (global)", "eff_rank_layerwise": "eff. rank (layer)",
        "sub_sv_overlap": "sing.-value overlap", "sub_left_top": "left top-10",
        "sub_right_top": "right top-10", "sub_right_bot": "right bottom-10"}
FAM = {"tv_norm_mean": 0, "tv_l2": 0, "tv_dot": 0, "tv_norm_ratio": 0, "tv_cosine": 0,
       "eff_rank_global": 1, "eff_rank_layerwise": 1, "sub_sv_overlap": 2,
       "sub_left_top": 2, "sub_right_top": 2, "sub_right_bot": 2}
FCOL = {0: BLUE, 1: "#8FA98C", 2: CORAL}

imp = {}
for b in ("clip", "clip16"):
    d = pd.concat([pd.read_csv(A / b / f"e3_coefficients_k{k}.csv") for k in (2, 3, 4)])
    d["a"] = d.coefficient.abs()
    imp[b] = d.groupby("metric").a.mean()
order = (imp["clip"] + imp["clip16"]).sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(4.6, 2.05))
y = np.arange(len(order))
ax.barh(y + 0.20, [imp["clip"][m] for m in order], 0.38,
        color=[FCOL[FAM[m]] for m in order], zorder=3)
ax.barh(y - 0.20, [imp["clip16"][m] for m in order], 0.38,
        color=[FCOL[FAM[m]] for m in order], alpha=0.45, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([NICE[m] for m in order], fontsize=7.4)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_xlabel("mean $|w|$ over 3 merge rules $\\times$ 3 group sizes", labelpad=2)
ax.set_ylim(-0.75, len(order) - 0.25)
h = [plt.Rectangle((0, 0), 1, 1, color=FCOL[i]) for i in (0, 1, 2)]
ax.grid(axis="x", color="#E7ECEF", lw=0.6, zorder=0); ax.set_axisbelow(True)
fig.tight_layout(pad=0.2, rect=(0, 0.085, 1, 1))
fig.legend(h, ["geometry", "rank", "subspace"], loc="lower center", ncol=3,
           frameon=False, fontsize=7.0, handlelength=1.0, columnspacing=1.8,
           handletextpad=0.5, bbox_to_anchor=(0.56, -0.025))
fig.savefig(OUT / "importance.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"  {OUT}/importance.pdf")

MLAB = {"weight_averaging": "averaging", "task_arithmetic": "task arith.", "ties": "TIES"}
MCOL = {"weight_averaging": BLUE, "task_arithmetic": CORAL, "ties": "#8FA98C"}
fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.55), sharey=True)
for ax, (b, lab) in zip(axes, (("clip", "ViT-B/32"), ("clip16", "ViT-B/16"))):
    d = pd.read_csv(A / b / "increment_null.csv")
    for j, m in enumerate(("weight_averaging", "task_arithmetic", "ties")):
        sub = d[d.method == m].sort_values("k")
        x = np.array([0, 1]) + (j - 1) * 0.27
        ax.bar(x, sub.observed_increment, 0.24, color=MCOL[m], zorder=3)
        for xi, (nu, ob, cl) in enumerate(zip(sub.null_p95, sub.observed_increment,
                                              sub.clears)):
            ax.plot([x[xi] - 0.13, x[xi] + 0.13], [nu, nu], color=INK, lw=1.1,
                    zorder=5)
            if not cl:
                ax.text(x[xi], ob + 0.03, "n.s.", ha="center", fontsize=6.4,
                        color=MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$k=3$", "$k=4$"])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_title(lab, fontsize=8.0, pad=3); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#E7ECEF", lw=0.6, zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("increment over additive", labelpad=2)
h = [plt.Rectangle((0, 0), 1, 1, color=MCOL[m])
     for m in ("weight_averaging", "task_arithmetic", "ties")]
h.append(plt.Line2D([0], [0], color=INK, lw=1.1))
fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0.10, 1, 1))
fig.legend(h, ["averaging", "task arith.", "TIES", "null (95th pct.)"],
           loc="lower center", ncol=4, frameon=False, fontsize=6.8,
           handlelength=1.0, columnspacing=1.5, handletextpad=0.5,
           bbox_to_anchor=(0.54, -0.03))
fig.savefig(OUT / "increment.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"  {OUT}/increment.pdf")

fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.62), sharey=True)
for ax, (b, lab) in zip(axes, (("clip", "ViT-B/32"), ("clip16", "ViT-B/16"))):
    r = pd.read_csv(A / b / "results.csv")
    for j, m in enumerate(("weight_averaging", "task_arithmetic", "ties")):
        col = (BLUE, CORAL, "#8FA98C")[j]
        for i, k in enumerate((2, 3, 4)):
            v = r[(r.method == m) & (r.k == k)].normalized_accuracy.values
            p = ax.violinplot([v], positions=[i + (j - 1) * 0.26], widths=0.24,
                              showextrema=False, showmedians=True)
            for body in p["bodies"]:
                body.set_facecolor(col); body.set_alpha(0.55); body.set_lw(0)
            p["cmedians"].set_color(INK); p["cmedians"].set_lw(0.9)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["2", "3", "4"])
    ax.set_xlabel("group size $k$", labelpad=2)
    ax.set_title(lab, fontsize=8.0, pad=3)
    ax.grid(axis="y", color="#E7ECEF", lw=0.6, zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("retained accuracy $y(S)$", labelpad=2)
h = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55)
     for c in (BLUE, CORAL, "#8FA98C")]
fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0.10, 1, 1))
fig.legend(h, ["averaging", "task arith.", "TIES"], loc="lower center", ncol=3,
           frameon=False, fontsize=7.0, handlelength=1.0, columnspacing=1.8,
           handletextpad=0.5, bbox_to_anchor=(0.54, -0.03))
fig.savefig(OUT / "target.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"  {OUT}/target.pdf")
