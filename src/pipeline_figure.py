"""Full measurement pipeline (horizontal), for the Methods section.

python -m src.pipeline_figure   ->  figures/fig01_pipeline_full.{pdf,png}

Stages, left to right: video -> frozen encoders -> temporal alignment ->
encoder pairing -> four cross-encoder measures -> per-video score ->
statistical comparison. The bottom band lists the controls and confound
analyses that test what the measures depend on. Nothing here plots data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "fig01_pipeline_full"

# Okabe-Ito, colour-blind safe
APP, MOT, GEO = "#0072B2", "#E69F00", "#009E73"
MEAS, MEAS_EC = "#DCEBF7", "#2B6CA3"
CTRL, CTRL_EC = "#EEEEEE", "#6B6B6B"
INK, SOFT = "#222222", "#555555"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.8, r=0.6, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(ax, x, y, s, fs=7, c=INK, w="normal", ha="center", va="center",
         style="normal", z=5, rot=0):
    return ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va,
                   style=style, zorder=z, linespacing=1.25, rotation=rot)


def arrow(ax, x0, y0, x1, y1, c=INK, lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4,
                                 shrinkA=0, shrinkB=0))


def lock(ax, x, y, s=0.55, c=SOFT):
    ax.add_patch(Rectangle((x - s, y - s), 2 * s, 1.4 * s, fc=c, ec=c, zorder=6))
    t = np.linspace(0, np.pi, 20)
    ax.plot(x + 0.62 * s * np.cos(t), y + 0.4 * s + 0.75 * s * np.sin(t),
            color=c, lw=0.7, zorder=6)


def stage(ax, x, w, n, title):
    text(ax, x + w / 2, 47.2, f"{n}", fs=6.5, c="white", w="bold")
    ax.add_patch(plt.Circle((x + w / 2, 47.2), 1.05, color=INK, zorder=4))
    text(ax, x + w / 2, 45.0, title, fs=7, w="bold")


def main():
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    # ---------------- 1. input video ---------------------------------------
    stage(ax, 0.5, 10.5, 1, "Video $V$")
    for k in range(4):                                 # stacked frames
        box(ax, 1.6 + 1.1 * k, 27.5 + 1.1 * k, 7.0, 5.2, fc="#F4F4F4",
            ec=SOFT, lw=0.6, r=0.3, z=2 + k)
    fx, fy = 1.6 + 3.3, 27.5 + 3.3                      # front frame corner
    ax.fill([fx + 0.5, fx + 2.6, fx + 4.2, fx + 6.5, fx + 6.5, fx + 0.5],
            [fy + 0.5, fy + 3.0, fy + 1.6, fy + 3.4, fy + 0.5, fy + 0.5],
            color="#C9C9C9", lw=0, zorder=9)                # hills
    ax.add_patch(plt.Circle((fx + 5.4, fy + 4.2), 0.55, color="#D8D8D8", zorder=9))
    text(ax, 5.75, 23.2, "$T=32$ frames\n$384\\times384$\ndecoded once,\nno lossy\nre-encoding",
         fs=6.2, c=SOFT)

    # ---------------- 2. frozen encoders -------------------------------------
    x2, w2 = 14.0, 19.0
    stage(ax, x2, w2, 2, "Frozen encoders $E_k$")
    groups = [("Appearance", APP, [("CLIP ViT-B/32", "image–text"),
                                   ("DINOv2 ViT-B/14", "self-supervised")]),
              ("Motion", MOT, [("VideoMAE-B", "masked video"),
                               ("RAFT-small", "optical flow")]),
              ("Geometry", GEO, [("MiDaS DPT-Hybrid", "relative depth"),
                                 ("Depth Anything V2", "relative depth"),
                                 ("UniDepth V2", "metric depth")])]
    y, eh, gap = 41.2, 3.05, 0.45
    enc_rows = []
    for gname, col, encs in groups:
        top = y
        for name, sub in encs:
            box(ax, x2 + 1.4, y - eh, w2 - 1.4, eh, fc="white", ec=col, lw=0.8, r=0.4)
            lock(ax, x2 + 2.9, y - eh / 2 - 0.1, c=col)
            text(ax, x2 + 4.2, y - eh / 2 + 0.55, name, fs=6.3, w="bold", ha="left")
            text(ax, x2 + 4.2, y - eh / 2 - 0.8, sub, fs=5.6, c=SOFT, ha="left")
            enc_rows.append(y - eh / 2)
            y -= eh + 0.35
        ax.add_patch(Rectangle((x2, y + 0.35), 0.9, top - y - 0.35, fc=col, ec="none", zorder=3))
        text(ax, x2 - 0.9, (top + y + 0.35) / 2, gname, fs=6.3, c=col, w="bold",
             rot=90)
        y -= gap
    enc_bottom = y + gap + 0.35
    text(ax, x2 + w2 / 2, enc_bottom - 1.4,
         "pretrained weights, eval mode,\nno fine-tuning, no gradients",
         fs=5.8, c=SOFT, style="italic")
    arrow(ax, 11.5, 30.5, x2 - 1.9, 30.5)

    # ---------------- 3. alignment -------------------------------------------
    x3, w3 = 36.5, 11.0
    stage(ax, x3, w3, 3, "Temporal alignment")
    for k, (c, lab, n) in enumerate([(APP, "$E_a(V)$", 32), (GEO, "$E_b(V)$", 32)]):
        yy = 36.5 - k * 7.5
        box(ax, x3 + 0.4, yy - 2.6, 3.8, 5.2, fc=c + "33", ec=c, lw=0.7, r=0.2)
        text(ax, x3 + 2.3, yy, lab, fs=6.2)
        text(ax, x3 + 2.3, yy - 3.5, f"$T'\\times d$", fs=5.6, c=SOFT)
        arrow(ax, x3 + 4.6, yy, x3 + 6.5, yy, lw=0.7)
        box(ax, x3 + 6.8, yy - 1.6, 3.8, 3.2, fc=c + "33", ec=c, lw=0.7, r=0.2)
        text(ax, x3 + 8.7, yy, "$16\\times d$", fs=5.8)
    text(ax, x3 + w3 / 2, 21.2,
         "resample every pair\nto $T_c=16$;\ncentre columns", fs=6.2, c=SOFT)
    arrow(ax, x2 + w2 + 0.4, 33.0, x3 + 0.2, 33.0)

    # ---------------- 4. pairing ---------------------------------------------
    x4, w4 = 50.5, 11.5
    stage(ax, x4, w4, 4, "Encoder pairs")
    box(ax, x4, 34.3, w4, 6.6, fc="white", ec=INK, lw=0.8)
    text(ax, x4 + w4 / 2, 37.6, "21 pairs\n16 different-role\n5 same-role", fs=6.2)
    box(ax, x4, 25.7, w4, 6.6, fc="#FFF4DD", ec=MOT, lw=1.0)
    text(ax, x4 + w4 / 2, 29.0, "Primary\n(pre-specified)\nCLIP–MiDaS", fs=6.2, w="bold")
    arrow(ax, x3 + w3 + 0.2, 33.0, x4 - 0.2, 33.0)

    # ---------------- 5. measures --------------------------------------------
    x5, w5 = 65.0, 15.5
    stage(ax, x5, w5, 5, "Cross-encoder measures")
    meas = [("Global", "debiased CKA  $D_{\\mathrm{CKA}}$"),
            ("Local", "neighbourhood  $D_{\\mathrm{NBR}}$"),
            ("Local", "CKNNA ($k=4$)"),
            ("Predictive", "ridge residual  $D_{\\mathrm{RIDGE}}$")]
    my, mh = 41.2, 4.1
    centers = []
    for lvl, name in meas:
        box(ax, x5, my - mh, w5, mh, fc=MEAS, ec=MEAS_EC, lw=0.8, r=0.4)
        text(ax, x5 + 0.7, my - 1.15, lvl, fs=5.4, c=MEAS_EC, w="bold", ha="left")
        text(ax, x5 + w5 / 2, my - mh / 2 - 0.45, name, fs=6.3)
        centers.append(my - mh / 2)
        my -= mh + 0.7
    ridge_y = centers[-1]
    for cy in centers:
        arrow(ax, x4 + w4 + 0.2, 33.0, x5 - 0.2, cy, lw=0.6,
              rad=0.0)
    # ridge calibration side input
    box(ax, x5 - 0.6, 14.4, w5 + 1.2, 5.5, fc="white", ec=MEAS_EC, lw=0.8, ls="--", r=0.4)
    text(ax, x5 + w5 / 2, 17.15,
         "real calibration videos only\n$\\mathbf{W}^\\star$ in closed form, $\\alpha=1$\nexcluded from scoring",
         fs=5.5, c=MEAS_EC)
    arrow(ax, x5 + w5 / 2, 19.9, x5 + w5 / 2, ridge_y - mh / 2 - 0.15,
          c=MEAS_EC, lw=0.7, ls="--")

    # ---------------- 6. per-video score + comparison -------------------------
    x6, w6 = 83.5, 16.0
    stage(ax, x6, w6, 6, "Real vs. generated")
    box(ax, x6, 35.6, w6, 5.4, fc="white", ec=INK, lw=0.8)
    text(ax, x6 + w6 / 2, 38.3, "one score per video\n$D(V)$, larger = weaker\nagreement",
         fs=6.0)
    for cy in centers:
        arrow(ax, x5 + w5 + 0.2, cy, x6 - 0.2, 38.3, lw=0.6)
    # two-distribution icon
    xs = np.linspace(x6 + 1.0, x6 + w6 - 1.0, 200)
    mid = x6 + w6 / 2
    for mu, col in [(mid - 1.2, APP), (mid + 1.2, MOT)]:
        yv = 27.0 + 5.2 * np.exp(-0.5 * ((xs - mu) / 2.1) ** 2)
        ax.fill_between(xs, 27.0, yv, color=col, alpha=0.35, lw=0, zorder=3)
        ax.plot(xs, yv, color=col, lw=0.8, zorder=4)
    ax.plot([x6 + 1.0, x6 + w6 - 1.0], [27.0, 27.0], color=INK, lw=0.6, zorder=4)
    text(ax, mid - 4.3, 31.6, "real", fs=5.8, c=APP, w="bold")
    text(ax, mid + 4.6, 31.6, "gen.", fs=5.8, c=MOT, w="bold")
    arrow(ax, mid, 35.4, mid, 32.8, lw=0.7)
    text(ax, mid, 22.6,
         "Cohen's $d$, AUROC\nvideo-level bootstrap CI\npermutation test\nrecall at 1% / 5% FPR",
         fs=5.9, c=SOFT)

    # ---------------- bottom band: controls and confounds --------------------
    by, bh = 1.2, 10.4
    box(ax, 14.0, by, 42.0, bh, fc=CTRL, ec=CTRL_EC, lw=0.8, r=0.5)
    text(ax, 15.0, by + bh - 1.4, "Controls: what does $D$ depend on?", fs=6.6,
         w="bold", ha="left", c=INK)
    ctrl = ["same-role pairs (H2)",
            "correspondence shuffle and same-class mismatch (H3)",
            "additive baseline $s(E_1)+s(E_2)$: two encoders, no relation",
            "random projection; within-encoder statistics; STALL, SPLIT (H5)"]
    for i, s in enumerate(ctrl):
        text(ax, 15.3, by + bh - 3.4 - 1.75 * i, "•  " + s, fs=6.0, ha="left")
    box(ax, 58.0, by, 41.5, bh, fc=CTRL, ec=CTRL_EC, lw=0.8, r=0.5)
    text(ax, 59.0, by + bh - 1.4, "Confound and robustness analyses", fs=6.6,
         w="bold", ha="left")
    conf = ["video movement $c(V)$: terciles and regression",
            "temporal resolution $T_c\\in\\{8,12,16,24,32\\}$",
            "provenance: real vs. real, matching, generator vs. generator",
            "H.264 compression (CRF 18–32); patch-level variant (H4)"]
    for i, s in enumerate(conf):
        text(ax, 59.3, by + bh - 3.4 - 1.75 * i, "•  " + s, fs=6.0, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"-> {OUT}.pdf")


if __name__ == "__main__":
    main()
