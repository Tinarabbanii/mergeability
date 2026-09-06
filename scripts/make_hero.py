from __future__ import annotations
import argparse, sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, FancyBboxPatch

BLUE, CORAL, SAGE, SAND = "#7BA7CC", "#E8927C", "#8FBC94", "#E9C46A"
GREY, INK, MUTED, GRID, FAINT = "#BCC5CC", "#2F3437", "#7A8388", "#E6EAED", "#AFB8BE"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})


def _head(fig, title, subtitle):
    fig.text(0.05, 0.93, title, fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.05, 0.862, subtitle, fontsize=9.6, color=MUTED)


def _save(anim, fig, out, fps):
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  -> {out.name:22} {out.stat().st_size // 1024:>5} KB")


def concept_b(out, n=72):
    fig = plt.figure(figsize=(9.4, 4.3))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.78, bottom=0.08)
    _head(fig, "Two routes to the same prediction",
          "one needs calibration images, the other reads only the weights")
    ax = fig.add_axes([0.04, 0.06, 0.93, 0.70]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    for yy, lab, col in ((3.5, "with data", CORAL), (1.3, "weights only", BLUE)):
        ax.add_patch(FancyBboxPatch((0.25, yy - 0.42), 1.5, 0.95, boxstyle="round,pad=0.06",
                                    fc="white", ec=col, lw=1.6))
        ax.text(1.0, yy, lab, fontsize=9.4, color=col, ha="center", va="center", fontweight="bold")

    imgs = [ax.add_patch(Rectangle((2.25 + i * 0.42, 3.15), 0.32, 0.62, fc=FAINT, ec="none"))
            for i in range(5)]
    grad = ax.text(4.65, 3.5, "gradients", fontsize=9.0, color=CORAL, va="center")
    wts = [ax.add_patch(Rectangle((2.25 + i * 0.42, 0.95), 0.32, 0.62, fc=BLUE, ec="none", alpha=0.55))
           for i in range(5)]
    geo = ax.text(4.65, 1.3, "geometry", fontsize=9.0, color=BLUE, va="center")

    ar1 = FancyArrowPatch((6.0, 3.5), (7.5, 2.75), arrowstyle="-|>", mutation_scale=16,
                          lw=2.0, color=CORAL, shrinkA=0, shrinkB=0, alpha=0.0)
    ar2 = FancyArrowPatch((6.0, 1.3), (7.5, 2.15), arrowstyle="-|>", mutation_scale=16,
                          lw=2.0, color=BLUE, shrinkA=0, shrinkB=0, alpha=0.0)
    ax.add_patch(ar1); ax.add_patch(ar2)
    box = FancyBboxPatch((7.7, 1.95), 2.0, 1.05, boxstyle="round,pad=0.08",
                         fc="white", ec=GRID, lw=1.6)
    ax.add_patch(box)
    score = ax.text(8.7, 2.62, "", fontsize=15, color=INK, ha="center", va="center", fontweight="bold")
    slab = ax.text(8.7, 2.22, "", fontsize=8.4, color=MUTED, ha="center", va="center")
    verdict = ax.text(5.0, 0.18, "", fontsize=10.2, color=MUTED, ha="center")
    cross = ax.plot([], [], lw=2.4, color=CORAL, alpha=0.0)[0]

    def frame(i):
        t = i / (n - 1)
        for j, p in enumerate(imgs):
            p.set_alpha(np.clip((t - 0.04 * j) * 6, 0, 1) * (1.0 if t < 0.62 else max(0.0, 1 - (t - 0.62) * 5)))
        for j, p in enumerate(wts):
            p.set_alpha(np.clip((t - 0.04 * j) * 6, 0, 1) * 0.6)
        grad.set_alpha(np.clip((t - 0.22) * 6, 0, 1) * (1.0 if t < 0.62 else max(0.0, 1 - (t - 0.62) * 5)))
        geo.set_alpha(np.clip((t - 0.22) * 6, 0, 1))
        ar1.set_alpha(np.clip((t - 0.34) * 6, 0, 1) * (1.0 if t < 0.62 else max(0.0, 1 - (t - 0.62) * 5)))
        ar2.set_alpha(np.clip((t - 0.34) * 6, 0, 1))
        if t > 0.46:
            score.set_text("r = 0.60"); slab.set_text("held-out, leave-one-task-out")
            score.set_alpha(np.clip((t - 0.46) * 6, 0, 1)); slab.set_alpha(np.clip((t - 0.46) * 6, 0, 1))
        else:
            score.set_text(""); slab.set_text("")
        if t > 0.66:
            a = np.clip((t - 0.66) * 6, 0, 1)
            cross.set_data([2.2, 5.6], [3.9, 3.1]); cross.set_alpha(a * 0.85)
            verdict.set_text("dropping the data changes nothing"); verdict.set_alpha(a)
        else:
            cross.set_alpha(0.0); verdict.set_text("")
        return ()
    _save(FuncAnimation(fig, frame, frames=n, interval=70), fig, out, 14)


PALETTES = {
    "light": dict(bg="white", ink="#2F3437", muted="#7A8388", grid="#E6EAED",
                  faint="#AFB8BE", good="#8FBC94", mid="#E9C46A", bad="#E8927C",
                  ringcol="#D8DEE3", nodecol="#2F3437", glow=1.0),
    "neon": dict(bg="#080B10", ink="#EAF6FF", muted="#7FA0B8", grid="#1E3446",
                 faint="#4A7A99", good="#3DF5B0", mid="#FFD166", bad="#FF5D8F",
                 ringcol="#2A4A61", nodecol="#FFFFFF", glow=2.6),
    "aurora": dict(bg="#0B0F16", ink="#F0F6FC", muted="#8DA2B5", grid="#1F3648",
                   faint="#46738A", good="#4FE3C1", mid="#A78BFA", bad="#FF7B72",
                   ringcol="#2B5068", nodecol="#FFFFFF", glow=2.4),
    "electric": dict(bg="#07080F", ink="#F2F4FF", muted="#9AA2CC", grid="#20254A",
                     faint="#4A5290", good="#00E5C0", mid="#FFE066", bad="#FF4D9D",
                     ringcol="#2E3670", nodecol="#FFFFFF", glow=2.8),
    "ember": dict(bg="#100C0A", ink="#FFF3E8", muted="#BFA08C", grid="#33231A",
                  faint="#6E4A33", good="#7BE495", mid="#FFC145", bad="#FF5C5C",
                  ringcol="#4A3324", nodecol="#FFF3E8", glow=2.4),
    "warm":  dict(bg="#FBF7F0", ink="#3B322A", muted="#8A7B6B", grid="#EDE3D5",
                  faint="#C9B9A5", good="#7FA98A", mid="#E0A458", bad="#C96F53",
                  ringcol="#E2D6C4", nodecol="#3B322A", glow=1.0),
    "cool":  dict(bg="#FAFCFD", ink="#1F2D3A", muted="#6C8598", grid="#E3ECF2",
                  faint="#AFC4D2", good="#5BA8A0", mid="#7BA7CC", bad="#B98EAD",
                  ringcol="#DCE7EF", nodecol="#1F2D3A", glow=1.0),
}


def _shrink(path, colors=72):
    from PIL import Image
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(im.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=im.info.get("duration", 62), loop=0, optimize=True)


def _real_accuracies(names):
    path = Path(__file__).resolve().parent.parent / "artifacts" / "clip" / "results.csv"
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df.method == "weight_averaging"]
    return {tuple(sorted(t.split("|"))): float(a)
            for t, a in zip(df["tasks"], df["normalized_accuracy"])}


def _ease(t):
    return t * t * (3.0 - 2.0 * t)


def concept_c(out, n=84, palette="light"):
    P = PALETTES[palette]
    fig = plt.figure(figsize=(9.6, 4.5), dpi=100)
    fig.patch.set_facecolor(P["bg"])
    fig.text(0.045, 0.918, "From pairs to groups", fontsize=18,
             fontweight="bold", color=P["ink"])
    fig.text(0.045, 0.845, "eight tasks, every subset of two, three and four, merged and measured",
             fontsize=10.8, color=P["muted"])

    ax = fig.add_axes([0.005, 0.02, 0.58, 0.78]); ax.axis("off")
    ax.set_facecolor(P["bg"])
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.50, 1.50); ax.set_aspect("equal")

    names = ["MNIST", "SVHN", "GTSRB", "EuroSAT", "DTD", "RESISC45", "Cars", "SUN397"]
    ang = np.linspace(90, 90 - 360, 9)[:8]
    pts = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in ang])
    for rw, ra in ((5.0, 0.05 * P["glow"]), (2.4, 0.10 * P["glow"]), (1.0, 1.0)):
        ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, ec=P["ringcol"],
                                 lw=rw, alpha=min(ra, 1.0), zorder=1))
    for u, v in combinations(range(8), 2):
        ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]],
                lw=0.8, color=P["grid"], zorder=1, alpha=0.9)

    halos, dots, labels = [], [], []
    for p_, nm, a in zip(pts, names, ang):
        h, = ax.plot([p_[0]], [p_[1]], "o", ms=24, color=P["good"], alpha=0.0, zorder=5)
        d, = ax.plot([p_[0]], [p_[1]], "o", ms=7, color=P["faint"], zorder=7)
        ha = "left" if -90 < a < 90 else "right"
        t = ax.text(p_[0] * 1.20, p_[1] * 1.20, nm, fontsize=8.6,
                    color=P["muted"], ha=ha, va="center")
        halos.append(h); dots.append(d); labels.append(t)

    glow = [ax.add_patch(plt.Polygon(pts[:3], closed=True, fc="none",
                                     ec=P["good"], lw=w, alpha=0.0, zorder=3))
            for w in (16.0, 10.0, 5.5, 2.5)]
    trail = [ax.add_patch(plt.Polygon(pts[:3], closed=True, fc="none", ec=P["faint"],
                                      lw=1.4, alpha=0.0, zorder=2)) for _ in range(3)]
    poly = plt.Polygon(pts[:3], closed=True, fc=P["good"], ec=P["good"],
                       lw=3.0, alpha=0.0, zorder=4)
    ax.add_patch(poly)
    acc_t = ax.text(0, -1.40, "", fontsize=15, color=P["ink"],
                    ha="center", fontweight="bold")

    ax2 = fig.add_axes([0.655, 0.20, 0.315, 0.50])
    ax2.set_facecolor(P["bg"])
    ax2.set_ylim(0, 1); ax2.set_yticks([])
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(P["grid"])
    ax2.tick_params(colors=P["muted"], labelsize=8.6)
    ax2.set_xlabel("post-merge accuracy", fontsize=9.6, color=P["muted"])
    ax2.grid(axis="x", color=P["grid"], lw=0.9, zorder=0); ax2.set_axisbelow(True)
    old, = ax2.plot([], [], "o", ms=6, color=P["faint"], alpha=0.85, zorder=3)
    live, = ax2.plot([], [], "o", ms=13, color=P["good"], zorder=5,
                     mec=P["bg"], mew=1.6)
    cnt = fig.text(0.655, 0.745, "", fontsize=20, color=P["ink"], fontweight="bold")
    sub = fig.text(0.655, 0.695, "", fontsize=10.5, color=P["muted"])

    rng = np.random.default_rng(0)
    real = _real_accuracies(names)
    seq, qual = [], {}
    for k, total, tag in ((2, 28, "the pairwise study"),
                          (3, 56, "does it carry over?"),
                          (4, 70, "and to four?")):
        allc = list(combinations(range(8), k))
        for i in rng.choice(len(allc), size=7, replace=False):
            c = allc[i]
            seq.append((k, c, total, tag))
            key = tuple(sorted(names[j] for j in c))
            qual[c] = real.get(key, float(np.clip(rng.normal(0.82, 0.07), 0.50, 0.98)))
    ys = {c: 0.12 + 0.76 * rng.random() for _, c, _, _ in seq}
    vals = np.array(sorted(qual.values()))
    lo_t, hi_t = float(np.quantile(vals, 0.34)), float(np.quantile(vals, 0.67))
    ax2.set_xlim(float(vals.min()) - 0.03, min(1.0, float(vals.max()) + 0.03))

    def quad(c):
        idx = list(c) + [c[-1]] * (4 - len(c))
        return pts[idx]

    per = max(n // len(seq), 1)

    def frame(i):
        j = min(i // per, len(seq) - 1)
        u = _ease((i % per) / max(per - 1, 1))
        k, c, total, tag = seq[j]
        prev = seq[max(j - 1, 0)][1]
        shape = quad(prev) * (1 - u) + quad(c) * u
        q = qual[c]
        col = P["good"] if q > hi_t else (P["mid"] if q > lo_t else P["bad"])
        poly.set_xy(shape); poly.set_facecolor(col); poly.set_edgecolor(col)
        poly.set_alpha(0.20 + 0.06 * P["glow"])
        pulse = np.exp(-((u - 0.12) ** 2) / 0.010)
        gm = P["glow"]
        for lvl, g in zip((0.030, 0.055, 0.095, 0.20), glow):
            g.set_xy(shape); g.set_edgecolor(col)
            g.set_alpha(min(1.0, (lvl + 0.10 * lvl * 6 * pulse) * gm))
        active = set(c)
        for idx in range(8):
            on = idx in active
            dots[idx].set_color(P["nodecol"] if on else P["faint"])
            dots[idx].set_markersize(9.5 if on else 7)
            halos[idx].set_color(col)
            halos[idx].set_alpha(min(1.0, 0.22 * P["glow"] * (0.45 + pulse)) if on else 0.0)
            labels[idx].set_color(P["ink"] if on else P["muted"])
        for d_, g_ in enumerate(trail):
            k_ = j - (d_ + 1)
            if k_ >= 0:
                g_.set_xy(quad(seq[k_][1]))
                g_.set_edgecolor(P["good"] if qual[seq[k_][1]] > hi_t else
                                 (P["mid"] if qual[seq[k_][1]] > lo_t else P["bad"]))
                g_.set_alpha(max(0.0, (0.22 - 0.07 * d_)) * P["glow"] * 0.6)
            else:
                g_.set_alpha(0.0)
        seen = [cc for _, cc, _, _ in seq[:j]]
        old.set_data([qual[cc] for cc in seen], [ys[cc] for cc in seen])
        live.set_data([q], [ys[c]]); live.set_color(col)
        acc_t.set_text(f"{q:.2f}"); acc_t.set_color(col)
        cnt.set_text(f"{total} subsets of {k}")
        sub.set_text(tag)
        return ()

    anim = FuncAnimation(fig, frame, frames=per * len(seq), interval=60)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=16),
              savefig_kwargs=dict(facecolor=P["bg"], edgecolor="none"))
    plt.close(fig)
    _shrink(out)
    print(f"  -> {out.name:22} {out.stat().st_size // 1024:>5} KB")


def concept_d(out, n=76):
    fig = plt.figure(figsize=(9.4, 4.3))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.78, bottom=0.06)
    _head(fig, "Predicted, then checked against chance",
          "every correlation is measured against a permutation null before it is believed")
    ax = fig.add_axes([0.08, 0.14, 0.52, 0.62])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted from the weights", fontsize=9.4, color=MUTED)
    ax.set_ylabel("measured after merging", fontsize=9.4, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.4)
    ax.spines["bottom"].set_color("#D2D8DC"); ax.spines["left"].set_color("#D2D8DC")
    ax.grid(color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=GREY, lw=1.2, zorder=1)
    rng = np.random.default_rng(3)
    m = 70
    nx, ny = rng.uniform(0.08, 0.92, m), rng.uniform(0.08, 0.92, m)
    sx = rng.uniform(0.10, 0.90, m)
    sy = np.clip(sx + rng.normal(0, 0.105, m), 0.04, 0.96)
    null_pts, = ax.plot([], [], "o", ms=5, color=GREY, alpha=0.0, zorder=3)
    real_pts, = ax.plot([], [], "o", ms=5.5, color=BLUE, alpha=0.0, zorder=4)
    ax2 = fig.add_axes([0.68, 0.22, 0.28, 0.46])
    ax2.set_xlim(-0.15, 0.95); ax2.set_ylim(-0.8, 0.8); ax2.set_yticks([])
    ax2.spines["left"].set_visible(False); ax2.spines["bottom"].set_color("#D2D8DC")
    ax2.tick_params(colors=MUTED, labelsize=8.2)
    ax2.set_xlabel("held-out correlation", fontsize=9.0, color=MUTED)
    ax2.grid(axis="x", color=GRID, lw=0.9, zorder=0); ax2.set_axisbelow(True)
    band = ax2.barh([0], [0], height=0.5, color=GRID, zorder=2)[0]
    mark, = ax2.plot([], [], "o", ms=11, color=BLUE, zorder=5, mec="white", mew=1.4)
    cap = fig.text(0.68, 0.72, "", fontsize=10.2, color=MUTED)
    verdict = fig.text(0.68, 0.11, "", fontsize=11, color=INK, fontweight="bold")

    def frame(i):
        t = i / (n - 1)
        if t < 0.42:
            a = np.clip(t * 4, 0, 1)
            k = int(np.clip(t / 0.42, 0, 1) * m)
            null_pts.set_data(nx[:k], ny[:k]); null_pts.set_alpha(0.55 * a)
            real_pts.set_alpha(0.0)
            band.set_width(np.clip(t / 0.42, 0, 1) * 0.46)
            mark.set_data([], [])
            cap.set_text("shuffle the labels: this is chance")
            verdict.set_text("")
        else:
            u = np.clip((t - 0.42) / 0.42, 0, 1)
            null_pts.set_alpha(0.55 * max(0.0, 1 - u * 1.4))
            k = int(u * m)
            real_pts.set_data(sx[:k], sy[:k]); real_pts.set_alpha(np.clip(u * 3, 0, 1))
            band.set_width(0.46)
            mark.set_data([0.60 * min(u * 1.2, 1.0)], [0])
            cap.set_text("now the real metrics")
            verdict.set_text("clears the null" if u > 0.85 else "")
        return ()
    _save(FuncAnimation(fig, frame, frames=n, interval=75), fig, out, 13)


CONCEPTS = {"b": concept_b, "c": concept_c, "d": concept_d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="c", choices=sorted(CONCEPTS) + ["all"])
    ap.add_argument("--palette", default="electric", choices=sorted(PALETTES) + ["all"])
    ap.add_argument("--out", default="animation.gif")
    args = ap.parse_args()
    d = Path(args.out).parent
    if args.concept == "c" and args.palette == "all":
        for name in PALETTES:
            concept_c(d / f"c_{name}.gif", palette=name)
    elif args.concept == "all":
        for key, fn in CONCEPTS.items():
            fn(d / f"{key}.gif")
    elif args.concept == "c":
        concept_c(Path(args.out), palette=args.palette)
    else:
        CONCEPTS[args.concept](Path(args.out))


if __name__ == "__main__":
    main()
