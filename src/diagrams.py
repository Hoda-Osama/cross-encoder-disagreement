"""Conceptual diagrams: Figure 1 (pipeline) and Figure 13 (taxonomy).

    python -m src.diagrams

These were previously specified in FIGURES.md for hand drawing. Generating
them in code keeps them editable, keeps the typography consistent with the
other figures, and produces vector output, which MDPI prefers for line art.

A NOTE ON WHAT FIGURE 1 IS NOT. It is not a model architecture. Nothing in
this study is trained, so there are no layers, no parameters, and no loss. The
figure is a measurement pipeline, and it has to read as one: every encoder is
drawn frozen, the only operation with any depth is the comparison between two
representations, and the banner at the foot states the constraint outright. A
reader who mistakes it for an architecture diagram has misunderstood the
paper, so the drawing has to prevent that.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

from . import config as C

FIGDIR = Path(C.PROJECT_ROOT) / "figures"
FIGDIR.mkdir(exist_ok=True)

APPEAR, MOTION, GEOM = "#0173B2", "#DE8F05", "#029E73"
PAIR, MEASURE, CTRL = "#CC78BC", "#56B4E9", "#949494"
FROZEN = "#E8E8E8"

plt.rcParams.update({"font.size": 7, "figure.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def box(ax, x, y, w, h, text, fc="white", ec="#333333", fs=6.5, lw=0.8,
        weight="normal", style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            zorder=3, weight=weight, linespacing=1.35)
    return x, y


def frozen_box(ax, x, y, w, h, name, sub, fs=6.0):
    """An encoder box, drawn so it reads as frozen at a glance."""
    box(ax, x, y, w, h, "", fc=FROZEN, ec="#666666")
    ax.text(x, y + h * 0.17, name, ha="center", va="center",
            fontsize=fs, weight="bold", zorder=3)
    ax.text(x, y - h * 0.20, sub, ha="center", va="center",
            fontsize=fs - 0.9, color="#444444", zorder=3)
    # padlock glyph, top-left corner
    lx, ly = x - w / 2 + 0.018, y + h / 2 - 0.016
    ax.add_patch(Rectangle((lx - 0.006, ly - 0.007), 0.012, 0.010,
                           fc="#555555", ec="none", zorder=4))
    ax.plot([lx - 0.003, lx - 0.003, lx + 0.003, lx + 0.003],
            [ly + 0.003, ly + 0.007, ly + 0.007, ly + 0.003],
            color="#555555", lw=0.6, zorder=4)


def arrow(ax, p0, p1, color="#333333", lw=0.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=7, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


# ==========================================================================
# Figure 1 -- the measurement pipeline
# ==========================================================================
def figure1(path="fig01_pipeline"):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- input ----------------------------------------------------------
    box(ax, 0.5, 0.965, 0.15, 0.032, "Video $V$", fs=8, weight="bold")

    box(ax, 0.5, 0.905, 0.40, 0.040,
        "decode once: $T = 32$ frames at $384\\times384$\n"
        "uniform sampling · no lossy re-encoding", fs=6.2)
    arrow(ax, (0.5, 0.949), (0.5, 0.925))

    # ---- encoder columns -------------------------------------------------
    cols = [
        (0.17, APPEAR, "APPEARANCE",
         [("CLIP ViT-B/32", "image–text"), ("DINOv2 ViT-B/14", "self-supervised")]),
        (0.50, MOTION, "MOTION",
         [("VideoMAE-B", "masked recon."), ("RAFT-small", "optical flow")]),
        (0.83, GEOM, "GEOMETRY",
         [("MiDaS DPT-H", "relative depth"), ("Depth Anything V2", "relative depth"),
          ("UniDepth V2", "metric depth")]),
    ]
    bottom = {}
    for cx, colour, role, encs in cols:
        ax.text(cx, 0.862, role, ha="center", fontsize=7.2,
                weight="bold", color=colour)
        arrow(ax, (0.5, 0.885), (cx, 0.836), color=colour, rad=0.0)
        y = 0.808
        for name, sub in encs:
            frozen_box(ax, cx, y, 0.235, 0.048, name, sub)
            y -= 0.062
        bottom[role] = y + 0.062 - 0.024

    ax.text(0.5, 0.622, "all encoders frozen · no fine-tuning · no gradients",
            ha="center", fontsize=6.2, style="italic", color="#555555")

    # ---- pair formation: the step that defines the paper -----------------
    py = 0.560
    box(ax, 0.5, py, 0.46, 0.050,
        "FORM A PAIR   $(E_1(V),\\; E_2(V))$\n"
        "cross-role, or same-role as a control",
        fc=PAIR, fs=7.0, weight="bold", lw=1.1)
    for (cx, colour, role, _) in cols:
        arrow(ax, (cx, bottom[role] - 0.012), (0.5, py + 0.027),
              color=colour, rad=0.10 if cx < 0.5 else -0.10)

    box(ax, 0.5, 0.489, 0.34, 0.032,
        "resample both to $T_c = 16$, every pair alike", fs=6.2)
    arrow(ax, (0.5, py - 0.025), (0.5, 0.505))

    # ---- measures and controls -------------------------------------------
    ax.text(0.29, 0.440, "MEASURES", ha="center", fontsize=6.6,
            weight="bold", color=MEASURE)
    ax.text(0.755, 0.440, "CONTROLS", ha="center", fontsize=6.6,
            weight="bold", color=CTRL)

    meas = [(0.115, "CKA"), (0.235, "CKNNA"), (0.355, "Neigh-\nbourhood"),
            (0.475, "Ridge\n(real-cal.)")]
    for mx, label in meas:
        box(ax, mx, 0.390, 0.108, 0.042, label, fc=MEASURE, fs=6.2)
        arrow(ax, (0.5, 0.473), (mx, 0.413), color=MEASURE, rad=0.12)

    ctrls = [(0.615, "shuffle\ncorrespondence"), (0.755, "additive\n$s(E_1)+s(E_2)$"),
             (0.895, "random\nprojection")]
    for cx, label in ctrls:
        box(ax, cx, 0.390, 0.125, 0.042, label, fc=CTRL, fs=5.8)
        arrow(ax, (0.5, 0.473), (cx, 0.413), color=CTRL, rad=-0.12)

    box(ax, 0.5, 0.322, 0.30, 0.030,
        "one scalar per video, per measure", fs=6.4)
    for mx, _ in meas:
        arrow(ax, (mx, 0.369), (0.5, 0.337), color=MEASURE, rad=0.10)
    for cx, _ in ctrls:
        arrow(ax, (cx, 0.369), (0.5, 0.337), color=CTRL, rad=-0.10)

    # ---- the two distributions, drawn with honest overlap ----------------
    ax.text(0.5, 0.272, "compare the two populations",
            ha="center", fontsize=6.4)
    arrow(ax, (0.5, 0.303), (0.5, 0.284))

    # inset_axes takes AXES coordinates, so it stays aligned with the boxes
    # around it. fig.add_axes would use figure coordinates and land elsewhere,
    # because the axes carries default margins.
    dax = ax.inset_axes([0.30, 0.150, 0.40, 0.100])
    x = np.linspace(-3.2, 4.6, 400)
    g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2)
    dax.fill_between(x, g(0.0, 1.0), color=APPEAR, alpha=0.55, lw=0)
    dax.fill_between(x, g(0.45, 1.05), color=MOTION, alpha=0.55, lw=0)
    dax.axvline(0.0, color=APPEAR, lw=0.9, ls="--")
    dax.axvline(0.45, color=MOTION, lw=0.9, ls="--")
    dax.set_yticks([]); dax.set_xticks([])
    dax.set_xlabel("discrepancy", fontsize=6)
    for sp in ("top", "right", "left"):
        dax.spines[sp].set_visible(False)
    dax.text(-1.9, 0.86, "real", color=APPEAR, fontsize=6.2, weight="bold")
    dax.text(2.0, 0.86, "generated", color=MOTION, fontsize=6.2, weight="bold")

    box(ax, 0.5, 0.092, 0.46, 0.030,
        "Cohen's $d$  ·  AUROC  ·  bootstrap CI  ·  recall at fixed FPR",
        fs=6.2)

    # ---- the banner ------------------------------------------------------
    ax.add_patch(Rectangle((0.14, 0.018), 0.72, 0.034, fc="white",
                           ec="#222222", lw=1.3, zorder=3))
    ax.text(0.5, 0.035,
            "NO CLASSIFIER   ·   NO FINE-TUNING   ·   NO DETECTOR TRAINING",
            ha="center", va="center", fontsize=7.2, weight="bold", zorder=4)

    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{path}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{path}.pdf")


# ==========================================================================
# Figure 13 -- the three-paper taxonomy
# ==========================================================================
def figure13(path="fig13_taxonomy"):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.5, 0.945, 0.13, 0.075, "Video $V$", fs=8, weight="bold")

    panels = [
        (0.17, "STALL", [("$E$", FROZEN)],
         "likelihood under\nreal-video statistics",
         "Is $E(V)$ likely\nunder real video?"),
        (0.50, "SPLIT", [("$E$", FROZEN)],
         "patch coherence\nTTR · LSMI",
         "Is $E(V)$ spatially and\ntemporally coherent?"),
        (0.83, "This work", [("$E_1$", FROZEN), ("$E_2$", FROZEN)],
         "$D\\,(E_1(V),\\, E_2(V))$",
         "Do $E_1(V)$ and\n$E_2(V)$ agree?"),
    ]

    for px, title, encs, op, question in panels:
        ours = title == "This work"
        ax.text(px, 0.862, title, ha="center", fontsize=8,
                weight="bold", color=PAIR if ours else "#333333")
        arrow(ax, (0.5, 0.905), (px, 0.790), rad=0.0)

        if len(encs) == 1:
            frozen_box(ax, px, 0.700, 0.150, 0.090, encs[0][0], "frozen encoder", fs=7)
            arrow(ax, (px, 0.655), (px, 0.585))
        else:
            for dx, (lbl, _) in zip((-0.068, 0.068), encs):
                frozen_box(ax, px + dx, 0.700, 0.100, 0.090, lbl, "frozen", fs=7)
            # the operator sits BETWEEN the two encoders -- the whole point
            arrow(ax, (px - 0.068, 0.655), (px, 0.600), rad=-0.18)
            arrow(ax, (px + 0.068, 0.655), (px, 0.600), rad=0.18)

        box(ax, px, 0.520, 0.240, 0.095, op,
            fc=PAIR if ours else "white", fs=6.6,
            weight="bold" if ours else "normal", lw=1.2 if ours else 0.8)
        arrow(ax, (px, 0.470), (px, 0.400))

        box(ax, px, 0.330, 0.265, 0.115, question, fs=7.2,
            weight="bold" if ours else "normal",
            ec=PAIR if ours else "#333333", lw=1.2 if ours else 0.8)

    ax.plot([0.335, 0.335], [0.22, 0.88], color="#DDDDDD", lw=0.7, zorder=0)
    ax.plot([0.665, 0.665], [0.22, 0.88], color="#DDDDDD", lw=0.7, zorder=0)

    ax.text(0.5, 0.155,
            "The first two panels apply an operator $\\it{to}$ one representation; "
            "the third applies one $\\it{between}$ two.",
            ha="center", fontsize=6.8, color="#444444")
    ax.text(0.5, 0.075,
            "All three use frozen encoders and train no detector. "
            "The difference is the unit of analysis.",
            ha="center", fontsize=6.4, style="italic", color="#666666")

    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{path}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{path}.pdf")


# ==========================================================================
# Figure 1, alternative panel layout
#
# Follows the presentation convention used by recent papers in this area:
# coloured panels left to right, the frozen encoder marked with a snowflake,
# each measure stated as a boxed formula beside its illustration, and the two
# class distributions at the end.
#
# The substantive difference from those figures is the second panel. Where a
# single-encoder method shows one tensor feeding two statistics, this shows
# two tensors meeting at a comparison. That is the contribution, so it gets
# the visual centre.
# ==========================================================================

def _snowflake(ax, x, y, r=0.008, color="#4FA8DC"):
    for a in range(0, 180, 60):
        t = np.deg2rad(a)
        ax.plot([x - r * np.cos(t), x + r * np.cos(t)],
                [y - r * np.sin(t) * 1.6, y + r * np.sin(t) * 1.6],
                color=color, lw=1.0, zorder=6, solid_capstyle="round")


def _panel(ax, x0, y0, x1, y1, title, fc, ec):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.0, zorder=0))
    ax.text((x0 + x1) / 2, y1 - 0.035, title, ha="center", va="center",
            fontsize=8.2, weight="bold", zorder=4, linespacing=1.3)


def _frames(ax, x, y, n=4, w=0.050, h=0.085, dx=0.008, dy=0.011):
    for i in reversed(range(n)):
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h,
                               fc="#9DC3E6" if i else "#6FA8DC",
                               ec="#33587A", lw=0.5, zorder=2 + i))


def _grid(ax, x, y, k=4, cell=0.011, fc="#8FBC6F", ec="#4F7A3A", z=3):
    for i in range(k):
        for j in range(k):
            ax.add_patch(Rectangle((x + j * cell, y + i * cell), cell * 0.9,
                                   cell * 0.9, fc=fc, ec=ec, lw=0.35, zorder=z))


def _encoder(ax, x, y, w, h, label, colour):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.003,rounding_size=0.006",
                                fc=colour, ec="#33587A", lw=0.7, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=5.9,
            rotation=90, zorder=4, linespacing=1.25)
    _snowflake(ax, x, y + h / 2 + 0.030)


def figure1_panels(path="fig01_pipeline_panels"):
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    BIG = dict(arrowstyle="-|>", color="#111111", lw=2.4, mutation_scale=16)

    # ---------------- Panel A: extraction -------------------------------
    _panel(ax, 0.012, 0.045, 0.268, 0.955,
           "Input Video &\nMulti-Encoder Extraction", "#FAF0C8", "#C8A93A")
    _frames(ax, 0.024, 0.455)
    ax.text(0.045, 0.420, "Video", ha="center", fontsize=7.2)
    ax.text(0.045, 0.390, "$T=32$ @ $384^2$", ha="center", fontsize=5.6,
            color="#555555")

    encs = [(0.745, "#9DC3E6", "CLIP · DINOv2", "APPEARANCE", APPEAR),
            (0.500, "#F5C98A", "VideoMAE · RAFT", "MOTION", MOTION),
            (0.255, "#9FD8B8", "MiDaS · DAv2\n· UniDepth", "GEOMETRY", GEOM)]
    for ey, col, lbl, role, rc in encs:
        _encoder(ax, 0.132, ey, 0.028, 0.140, lbl, col)
        arrow(ax, (0.082, 0.490), (0.117, ey), color="#555555", lw=0.9,
              rad=0.14 if ey > 0.5 else -0.14)
        _grid(ax, 0.186, ey - 0.014, k=4, cell=0.010)
        ax.text(0.232, ey + 0.052, role, ha="center", fontsize=5.9,
                weight="bold", color=rc)
        ax.text(0.206, ey - 0.040,
                "$E(V)\\in\\mathbb{R}^{T\\times d}$",
                ha="center", va="center", fontsize=5.8)
        arrow(ax, (0.147, ey), (0.182, ey), color="#555555", lw=0.9)

    ax.text(0.140, 0.088, "frozen · no fine-tuning · no gradients",
            ha="center", fontsize=6.0, style="italic", color="#666666")

    # ---------------- Panel B: the comparison ---------------------------
    _panel(ax, 0.292, 0.600, 0.665, 0.955,
           "Cross-Encoder Comparison  $D\\,(E_1(V),E_2(V))$",
           "#F7E4EC", "#C48AA6")
    ax.add_patch(FancyArrowPatch((0.270, 0.780), (0.290, 0.780), **BIG))

    for gx, lbl, col in ((0.330, "$E_1(V)$", "#9DC3E6"),
                         (0.330, "$E_2(V)$", "#9FD8B8")):
        pass
    _grid(ax, 0.320, 0.820, k=4, fc="#9DC3E6", ec="#33587A")
    ax.text(0.342, 0.808, "$E_1(V)$", ha="center", fontsize=6.2)
    _grid(ax, 0.320, 0.680, k=4, fc="#9FD8B8", ec="#3A7A55")
    ax.text(0.342, 0.668, "$E_2(V)$", ha="center", fontsize=6.2)

    ax.text(0.412, 0.762, "resample both\nto $T_c=16$", ha="center",
            fontsize=6.0, linespacing=1.3)
    arrow(ax, (0.372, 0.838), (0.400, 0.790), color="#555555", lw=0.9)
    arrow(ax, (0.372, 0.698), (0.400, 0.735), color="#555555", lw=0.9)

    # similarity structures
    for sy, col in ((0.822, "#9DC3E6"), (0.700, "#9FD8B8")):
        _grid(ax, 0.462, sy - 0.010, k=5, cell=0.0095, fc=col,
              ec="#444444")
    ax.text(0.452, 0.622, "frame-similarity\nstructures", ha="center",
            fontsize=5.8, linespacing=1.3, color="#555555")
    arrow(ax, (0.444, 0.775), (0.460, 0.828), color="#555555", lw=0.9)
    arrow(ax, (0.444, 0.748), (0.460, 0.712), color="#555555", lw=0.9)

    box(ax, 0.592, 0.822, 0.128, 0.058,
        "$D_{\\mathrm{CKA}} = 1-\\frac{\\|B^{\\top}A\\|_F^2}"
        "{\\|A^{\\top}A\\|_F\\,\\|B^{\\top}B\\|_F}$", fs=6.4)
    box(ax, 0.592, 0.722, 0.128, 0.050,
        "$D_{\\mathrm{NBR}} = 1-\\rho_s\\!\\left(u(S_A),u(S_B)\\right)$\n"
        "CKNNA: mutual $k$-NN masked", fs=6.0)
    box(ax, 0.592, 0.648, 0.128, 0.044,
        "$D_{\\mathrm{RIDGE}}$: residual under a\nmap calibrated on real video",
        fs=6.0)
    for by in (0.800, 0.700, 0.648):
        arrow(ax, (0.512, 0.770), (0.527, by), color="#555555", lw=0.9,
              rad=0.10 if by > 0.72 else -0.10)

    # ---------------- Panel C: controls ---------------------------------
    _panel(ax, 0.292, 0.045, 0.665, 0.485,
           "Controls: what the measurement depends on", "#E4F0E2", "#7FA870")
    ax.add_patch(FancyArrowPatch((0.270, 0.265), (0.290, 0.265), **BIG))

    ctrl = [
        (0.365, "Same-role pair",
         "$E_1{\\leftrightarrow}E_1'$\nCLIP–DINOv2", "any two\nrepresentations?"),
        (0.470, "Shuffle",
         "$(A_i,B_{\\pi(i)})$\nmarginals kept", "marginal\nstatistics?"),
        (0.575, "Additive",
         "$s(E_1)+s(E_2)$\nno relation", "merely two\nencoders?"),
    ]
    for cx, name, expr, asks in ctrl:
        ax.text(cx, 0.400, name, ha="center", fontsize=6.8, weight="bold")
        box(ax, cx, 0.330, 0.092, 0.060, expr, fc="white", fs=6.2)
        ax.text(cx, 0.238, "rules out", ha="center", fontsize=5.6,
                style="italic", color="#666666")
        box(ax, cx, 0.178, 0.092, 0.048, asks, fc="#F2F2F2", fs=6.0)
        arrow(ax, (cx, 0.298), (cx, 0.204), color="#555555", lw=0.9)

    ax.text(0.478, 0.090,
            "plus movement covariate, provenance bound, frame-count sweep",
            ha="center", fontsize=6.0, color="#555555")

    # ---------------- Panel D: statistics -------------------------------
    _panel(ax, 0.690, 0.045, 0.988, 0.955,
           "Statistics & Reporting", "#FBE8D5", "#C98A4B")
    ax.add_patch(FancyArrowPatch((0.668, 0.780), (0.688, 0.780), **BIG))
    ax.add_patch(FancyArrowPatch((0.668, 0.265), (0.688, 0.265), **BIG))

    box(ax, 0.839, 0.845, 0.240, 0.070,
        "one scalar per video, per measure, per pair\n"
        "no threshold · no classifier fitted", fc="white", fs=6.4)

    dax = ax.inset_axes([0.715, 0.470, 0.250, 0.230])
    xg = np.linspace(-3.2, 4.8, 400)
    g = lambda m, s: np.exp(-0.5 * ((xg - m) / s) ** 2)
    dax.fill_between(xg, g(0.0, 1.0), color=APPEAR, alpha=0.55, lw=0)
    dax.fill_between(xg, g(0.55, 1.05), color=MOTION, alpha=0.55, lw=0)
    dax.axvline(0.0, color=APPEAR, lw=0.9, ls="--")
    dax.axvline(0.55, color=MOTION, lw=0.9, ls="--")
    dax.set_yticks([]); dax.set_xticks([])
    dax.set_xlabel("discrepancy", fontsize=5.8)
    dax.set_ylabel("density", fontsize=5.8)
    for sp in ("top", "right"):
        dax.spines[sp].set_visible(False)
    dax.text(-2.6, 0.82, "real", color=APPEAR, fontsize=6.2, weight="bold")
    dax.text(2.3, 0.82, "generated", color=MOTION, fontsize=6.2, weight="bold")

    box(ax, 0.839, 0.300, 0.240, 0.095,
        "Cohen's $d$ (generated $-$ real)\n"
        "AUROC · 2000-sample bootstrap CI\n"
        "recall at FPR 1% and 5%\n"
        "label permutation over videos", fc="white", fs=6.4)
    arrow(ax, (0.839, 0.455), (0.839, 0.352), color="#555555", lw=1.0)

    box(ax, 0.839, 0.150, 0.240, 0.075,
        "Reported per benchmark, per generator,\n"
        "and per manipulation category.\n"
        "Pre-specified primary: CLIP–MiDaS, CKA.",
        fc="#F2F2F2", fs=6.2)
    arrow(ax, (0.839, 0.252), (0.839, 0.190), color="#555555", lw=1.0)

    # ---------------- banner ---------------------------------------------
    ax.add_patch(Rectangle((0.296, 0.512), 0.366, 0.030, fc="white",
                           ec="#222222", lw=1.1, zorder=5))
    ax.text(0.478, 0.527,
            "NO CLASSIFIER  ·  NO FINE-TUNING  ·  NO DETECTOR TRAINING",
            ha="center", va="center", fontsize=6.2, weight="bold", zorder=6)

    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{path}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{path}.pdf")


# ==========================================================================
# Figure 1, grayscale academic style
#
# The panel version uses colour to carry the appearance/motion/geometry
# coding, which is consistent with the other figures but reads as a
# presentation slide rather than a journal page, and fails if printed or
# photocopied in grayscale -- a real constraint for some venues and readers.
#
# This version uses line weight, hatching, and box style instead of colour to
# carry every distinction. It should remain fully legible if converted to
# grayscale, and it matches the plainer convention (thin rules, serif-safe
# labels, no fills except white/light-gray) common in Elsevier and MDPI
# methods figures.
# ==========================================================================

HATCH = {"appearance": "//", "motion": "\\\\", "geometry": "..",
         "measure": "", "control": "xx"}


def _gbox(ax, x, y, w, h, text, fs=6.6, lw=0.9, hatch=None, fc="white",
          style="round,pad=0.010"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                fc=fc, ec="black", lw=lw, hatch=hatch,
                                zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            linespacing=1.3)


def _garrow(ax, p0, p1, lw=0.9, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color="black",
                                 lw=lw, mutation_scale=7, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def figure1_grayscale(path="fig01_pipeline_grayscale"):
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- input ------------------------------------------------------------
    _gbox(ax, 0.5, 0.968, 0.16, 0.032, "Video $V$", fs=8, lw=1.3)
    _gbox(ax, 0.5, 0.910, 0.42, 0.040,
          "decode once: $T=32$ frames at $384\\times384$\n"
          "uniform sampling, no lossy re-encoding", fs=6.2)
    _garrow(ax, (0.5, 0.952), (0.5, 0.930))

    # ---- three encoder columns, distinguished by hatch only --------------
    cols = [
        (0.17, "appearance", "APPEARANCE",
         [("CLIP ViT-B/32", "image--text"), ("DINOv2 ViT-B/14", "self-supervised")]),
        (0.50, "motion", "MOTION",
         [("VideoMAE-B", "masked recon."), ("RAFT-small", "optical flow")]),
        (0.83, "geometry", "GEOMETRY",
         [("MiDaS DPT-H", "relative depth"), ("Depth Anything V2", "relative depth"),
          ("UniDepth V2", "metric depth")]),
    ]
    bottom = {}
    for cx, key, role, encs in cols:
        ax.text(cx, 0.852, role, ha="center", fontsize=7.0, weight="bold")
        _garrow(ax, (0.5, 0.890), (cx, 0.862))
        y = 0.812
        for name, sub in encs:
            _gbox(ax, cx, y, 0.230, 0.046, "", fs=6.2,
                  hatch=HATCH[key], fc="#F7F7F7")
            ax.add_patch(FancyBboxPatch((cx - 0.100, y - 0.020), 0.200, 0.040,
                                        boxstyle="round,pad=0.002", fc="white",
                                        ec="none", zorder=2.5))
            ax.text(cx, y + 0.010, name, ha="center", va="center",
                    fontsize=6.2, weight="bold", zorder=3)
            ax.text(cx, y - 0.011, sub, ha="center", va="center",
                    fontsize=5.4, style="italic", zorder=3)
            y -= 0.060
        bottom[role] = y + 0.060 - 0.023

    ax.text(0.5, 0.628, "frozen -- no fine-tuning, no gradients",
            ha="center", fontsize=6.2, style="italic")

    # ---- pair formation, drawn as the only double-bordered box -----------
    py = 0.562
    _gbox(ax, 0.5, py, 0.46, 0.052,
          "form a pair  $(E_1(V),\\,E_2(V))$\ncross-role, or same-role as a control",
          fs=6.8, lw=1.6, style="round,pad=0.014")
    for cx, key, role, _ in cols:
        _garrow(ax, (cx, bottom[role] - 0.010), (0.5, py + 0.028),
                rad=0.10 if cx < 0.5 else -0.10)

    _gbox(ax, 0.5, 0.492, 0.34, 0.032,
          "resample both to $T_c=16$, every pair alike", fs=6.2)
    _garrow(ax, (0.5, py - 0.027), (0.5, 0.508))

    # ---- measures (plain box) vs controls (hatched) -----------------------
    ax.text(0.29, 0.446, "measures", ha="center", fontsize=6.6,
            style="italic")
    ax.text(0.755, 0.446, "controls", ha="center", fontsize=6.6,
            style="italic")

    meas = [(0.115, "CKA"), (0.235, "CKNNA"), (0.355, "Neigh-\nbourhood"),
            (0.475, "Ridge\n(real-cal.)")]
    for mx, label in meas:
        _gbox(ax, mx, 0.404, 0.106, 0.044, label, fs=6.2, lw=1.1)
        _garrow(ax, (0.5, 0.476), (mx, 0.427), rad=0.12)

    ctrls = [(0.615, "shuffle\ncorrespondence"), (0.755, "additive\n$s(E_1){+}s(E_2)$"),
             (0.895, "random\nprojection")]
    for cxp, label in ctrls:
        _gbox(ax, cxp, 0.404, 0.124, 0.044, "", fs=5.8, lw=0.9,
              hatch=HATCH["control"], fc="#F7F7F7")
        ax.add_patch(FancyBboxPatch((cxp - 0.052, 0.390), 0.104, 0.028,
                                    boxstyle="round,pad=0.003", fc="white",
                                    ec="none", zorder=2.5))
        ax.text(cxp, 0.404, label, ha="center", va="center", fontsize=5.8,
                zorder=3, linespacing=1.2)
        _garrow(ax, (0.5, 0.476), (cxp, 0.427), rad=-0.12)

    _gbox(ax, 0.5, 0.330, 0.30, 0.030,
          "one scalar per video, per measure", fs=6.4)
    for mx, _ in meas:
        _garrow(ax, (mx, 0.382), (0.5, 0.345), rad=0.10)
    for cxp, _ in ctrls:
        _garrow(ax, (cxp, 0.382), (0.5, 0.345), rad=-0.10)

    # ---- distributions, grayscale-safe: hatch instead of colour fill ------
    ax.text(0.5, 0.278, "compare the two populations", ha="center", fontsize=6.4)
    _garrow(ax, (0.5, 0.315), (0.5, 0.292))

    dax = ax.inset_axes([0.30, 0.155, 0.40, 0.100])
    xg = np.linspace(-3.2, 4.8, 400)
    g = lambda m, s: np.exp(-0.5 * ((xg - m) / s) ** 2)
    dax.plot(xg, g(0.55, 1.05), color="black", lw=1.3, zorder=3)
    dax.fill_between(xg, g(0.55, 1.05), fc="white", ec="none", zorder=2)
    dax.fill_between(xg, g(0.0, 1.0), fc="#DDDDDD", ec="black", lw=1.3,
                     hatch="//", zorder=1, alpha=0.9)
    dax.set_yticks([]); dax.set_xticks([])
    dax.set_xlabel("discrepancy", fontsize=6)
    for sp in ("top", "right", "left"):
        dax.spines[sp].set_visible(False)
    dax.text(-2.9, 0.85, "real", fontsize=6.2, weight="bold")
    dax.text(1.9, 0.85, "generated", fontsize=6.2, weight="bold")

    _gbox(ax, 0.5, 0.070, 0.44, 0.032,
          "Cohen's $d$, AUROC, bootstrap CI, recall at fixed FPR", fs=6.2)

    # ---- banner: rule lines rather than colour --------------------------
    ax.plot([0.16, 0.84], [0.012, 0.012], color="black", lw=0.8)
    ax.plot([0.16, 0.84], [0.040, 0.040], color="black", lw=0.8)
    ax.text(0.5, 0.026,
            "no classifier  --  no fine-tuning  --  no detector training",
            ha="center", va="center", fontsize=7.2, weight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{path}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{path}.pdf")


if __name__ == "__main__":
    print("figure 1 (vertical, colour):"); figure1()
    print("figure 1 (panels, colour):");    figure1_panels()
    print("figure 1 (grayscale):");         figure1_grayscale()
    print("figure 13:");                    figure13()
    print(f"\nvector output in {FIGDIR}")
    print("Both are editable in Inkscape or Illustrator if the layout needs")
    print("adjusting for the final template.")
