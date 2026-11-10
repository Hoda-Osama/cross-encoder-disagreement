"""Detailed measurement pipeline figure.

python -m src.pipeline_detailed  ->  figures/fig01_pipeline_detailed.{pdf,png}

(a) input video and uniform sampling
(b) frozen encoders: input each receives, native output T' x d, and the
    aligned output 16 x d
(c) pairing: the 7 x 7 grid of encoder pairs
(d) cross-encoder similarity for one pair: representations -> frame-similarity
    matrices -> four measures -> one score per video
(e) comparison of real and generated videos

All matrices drawn here are ILLUSTRATIVE (smooth random features with the
correct shapes); no data is plotted. The caption must say so.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "fig01_pipeline_detailed"

APP, MOT, GEO = "#0072B2", "#E69F00", "#009E73"
ROLE_C = {"A": APP, "M": MOT, "G": GEO}
MEAS, MEAS_EC = "#DCEBF7", "#2B6CA3"
INK, SOFT, FAINT = "#222222", "#5A5A5A", "#BDBDBD"
MIN_FS = 6.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5,
                     "mathtext.fontset": "dejavusans"})

ENCODERS = [  # name, role, input, native T', d
    ("CLIP ViT-B/32", "A", "frame, 224²", 32, 512),
    ("DINOv2 ViT-B/14", "A", "frame, 224²", 32, 768),
    ("VideoMAE-B", "M", "clip, 2-frame tubelets", 16, 768),
    ("RAFT-small", "M", "consecutive frame pairs", 31, 64),
    ("MiDaS DPT-Hybrid", "G", "frame, 384²", 32, 768),
    ("Depth Anything V2", "G", "frame, 384²", 32, 768),
    ("UniDepth V2", "G", "frame, 384²", 32, 1024),
]
SHORT = ["CLIP", "DINOv2", "VideoMAE", "RAFT", "MiDaS", "DA V2", "UniDepth"]


# --------------------------------------------------------------------------
def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.8, r=0.5, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, fs=6.5, c=INK, w="normal", ha="center", va="center",
        style="normal", rot=0, z=6):
    fs = max(fs, MIN_FS)
    return ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va,
                   style=style, rotation=rot, zorder=z, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, c=INK, lw=0.8, ls="-", rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5,
                                 shrinkA=0, shrinkB=0))


def lock(ax, x, y, s=0.5, c=SOFT):
    ax.add_patch(Rectangle((x - s, y - s), 2 * s, 1.4 * s, fc=c, ec=c, zorder=7))
    tt = np.linspace(0, np.pi, 20)
    ax.plot(x + 0.62 * s * np.cos(tt), y + 0.4 * s + 0.75 * s * np.sin(tt),
            color=c, lw=0.7, zorder=7)


def panel(ax, x, y, w, h, letter, title):
    box(ax, x, y, w, h, fc="#FAFAFA", ec="#9A9A9A", lw=0.7, r=0.8, z=0)
    txt(ax, x + 1.2, y + h - 1.5, f"({letter})", fs=7.2, w="bold", ha="left")
    txt(ax, x + 3.9, y + h - 1.5, title, fs=7.2, w="bold", ha="left")


def feats(T, d, seed, k=6):
    """Smooth, time-correlated illustrative features (T x d)."""
    r = np.random.default_rng(seed)
    z = np.cumsum(r.standard_normal((T, k)), 0)
    return z @ r.standard_normal((k, d)) + 0.6 * r.standard_normal((T, d))


def mat(ax, M, x, y, w, h, cmap, ec=INK, vmin=None, vmax=None):
    ax.imshow(M, extent=(x, x + w, y, y + h), aspect="auto", cmap=cmap,
              vmin=vmin, vmax=vmax, zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.6, zorder=4))


def resample(X, t=16):
    i = np.linspace(0, len(X) - 1, t)
    lo = np.floor(i).astype(int); hi = np.minimum(lo + 1, len(X) - 1)
    w = (i - lo)[:, None]
    return X[lo] * (1 - w) + X[hi] * w


def cos(X):
    X = X - X.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X @ X.T


# --------------------------------------------------------------------------
def panel_b_table(ax):
    # ======================= (b) frozen encoders ===========================
    panel(ax, 15.5, 28.8, 84.0, 26.7, "b", "Frozen encoders: input and output")
    for j, (lab, c) in enumerate([("appearance", APP), ("motion", MOT), ("geometry", GEO)]):
        xl = 66.0 + 11.0 * j
        ax.add_patch(Rectangle((xl, 53.55), 1.1, 1.1, fc=c, ec="none", zorder=3))
        txt(ax, xl + 1.6, 54.1, lab, fs=5.8, c=c, w="bold", ha="left")
    INPUTS = ["$T$ frames", "$T$ frames", "$T$ frames (clips)", "$T-1$ frame pairs",
              "$T$ frames", "$T$ frames", "$T$ frames"]
    TPRIME = ["T", "T", "T/2", "(T-1)", "T", "T", "T"]
    OUTPUTS = ["image embedding / frame", "CLS token / frame",
               "token / tubelet", "pooled flow / frame pair",
               "pooled patch tokens / frame", "pooled patch tokens / frame",
               "pooled patch tokens / frame"]
    xi, wi = 17.2, 11.0
    xe, we = 30.6, 15.8
    xo, wo = 48.8, 33.4
    xa, wa = 84.6, 13.6
    ytop = 51.2
    for x, w, lab in [(xi, wi, "input"), (xe, we, "encoder (frozen)"),
                      (xo, wo, "output $\\mathbf{Z}_k\\in\\mathbb{R}^{T_k^{\\prime}\\times d_k}$"),
                      (xa, wa, "aligned to $T_c$")]:
        txt(ax, x + w / 2, ytop, lab, fs=5.9, c=SOFT, w="bold")
    row_h, y = 2.92, ytop - 1.0
    for i, (name, role, _, T, d) in enumerate(ENCODERS):
        c = ROLE_C[role]
        yc = y - row_h / 2
        h = row_h - 0.55
        box(ax, xi, yc - h / 2, wi, h, fc="white", ec=FAINT, lw=0.5, r=0.25)
        txt(ax, xi + wi / 2, yc, INPUTS[i], fs=5.6, c=SOFT)
        arrow(ax, xi + wi + 0.15, yc, xe - 0.15, yc, lw=0.55, ms=4.5)
        box(ax, xe, yc - h / 2, we, h, fc=c + "22", ec=c, lw=0.8, r=0.3)
        lock(ax, xe + 1.1, yc - 0.05, s=0.38, c=c)
        txt(ax, xe + we / 2 + 0.7, yc, name, fs=5.8, w="bold")
        arrow(ax, xe + we + 0.15, yc, xo - 0.15, yc, lw=0.55, ms=4.5)
        box(ax, xo, yc - h / 2, wo, h, fc="white", ec=c, lw=0.55, r=0.25)
        txt(ax, xo + 1.0, yc,
            f"$\\mathbf{{Z}}_{i+1}\\in\\mathbb{{R}}^{{{TPRIME[i]}\\times d_{i+1}}}$",
            fs=6.3, ha="left")
        txt(ax, xo + wo - 0.8, yc, OUTPUTS[i], fs=5.3, c=SOFT, ha="right")
        arrow(ax, xo + wo + 0.15, yc, xa - 0.15, yc, lw=0.55, ms=4.5)
        box(ax, xa, yc - h / 2, wa, h, fc=c + "22", ec=c, lw=0.55, r=0.25)
        txt(ax, xa + wa / 2, yc, f"$\\mathbb{{R}}^{{T_c\\times d_{i+1}}}$", fs=6.5)
        y -= row_h


def trapezoid(ax, x, y, w, h, c):
    """Encoder glyph: wide input side, narrow output side."""
    ax.add_patch(plt.Polygon([(x, y - h / 2), (x + w, y - h * 0.3),
                              (x + w, y + h * 0.3), (x, y + h / 2)],
                             closed=True, fc=c + "30", ec=c, lw=0.8, zorder=3))


def matglyph(ax, x, y, w, h, c, rows=6):
    """Output glyph: a small T' x d matrix drawn as horizontal bands."""
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc="white", ec=c, lw=0.6, zorder=3))
    for r in range(1, rows):
        yy = y - h / 2 + r * h / rows
        ax.plot([x, x + w], [yy, yy], color=c, lw=0.35, alpha=0.7, zorder=4)


INPUTS_S = ["$T$ frames", "$T$ frames", "$T$ frames (clips)", "$T-1$ frame pairs",
            "$T$ frames", "$T$ frames", "$T$ frames"]
TPRIME_S = ["T", "T", "T/2", "(T-1)", "T", "T", "T"]
DESC_S = ["image embedding", "CLS token", "tubelet token", "pooled flow",
          "pooled patch tokens", "pooled patch tokens", "pooled patch tokens"]
ROLE_NAME = {"A": "Appearance", "M": "Motion", "G": "Geometry"}


def panel_b_tree(ax):
    """Fan-out: input bus -> encoder trapezoids -> output matrices -> resample."""
    panel(ax, 15.5, 28.8, 84.0, 26.7, "b", "Frozen encoders")
    row_h, ytop = 3.25, 51.6
    ys = [ytop - row_h * (i + 0.5) for i in range(7)]
    # role bands
    for role in "AMG":
        idx = [i for i, e in enumerate(ENCODERS) if e[1] == role]
        top, bot = ys[idx[0]] + row_h / 2 - 0.15, ys[idx[-1]] - row_h / 2 + 0.15
        box(ax, 17.0, bot, 64.5, top - bot, fc=ROLE_C[role] + "12",
            ec=ROLE_C[role] + "55", lw=0.5, r=0.4, z=1)
        txt(ax, 17.9, (top + bot) / 2, ROLE_NAME[role], fs=5.6, c=ROLE_C[role],
            w="bold", rot=90)
    # input bus
    ax.plot([20.0, 20.0], [ys[0], ys[-1]], color=INK, lw=0.8, zorder=3)
    yb = (ys[3] + ys[4]) / 2
    arrow(ax, 14.75, yb, 19.9, yb, lw=0.7, ms=5)
    for i, (name, role, *_ ) in enumerate(ENCODERS):
        c, y = ROLE_C[role], ys[i]
        txt(ax, 24.6, y + 0.85, INPUTS_S[i], fs=5.1, c=SOFT)
        arrow(ax, 20.0, y, 29.2, y, lw=0.55, ms=4.5)
        trapezoid(ax, 29.4, y, 17.0, row_h - 0.5, c)
        lock(ax, 31.0, y - 0.05, s=0.36, c=c)
        txt(ax, 38.6, y, name, fs=5.8, w="bold")
        arrow(ax, 46.6, y, 49.4, y, lw=0.55, ms=4.5)
        matglyph(ax, 49.6, y, 3.2, row_h - 0.8, c)
        txt(ax, 53.6, y,
            f"$\\mathbf{{Z}}_{i+1}\\in\\mathbb{{R}}^{{{TPRIME_S[i]}\\times d_{i+1}}}$",
            fs=6.2, ha="left")
        txt(ax, 80.8, y, DESC_S[i], fs=5.2, c=SOFT, ha="right")
        # to resample block
        ax.plot([81.6, 84.0], [y, y], color=INK, lw=0.5, zorder=3)
    ax.plot([84.0, 84.0], [ys[0], ys[-1]], color=INK, lw=0.8, zorder=3)
    arrow(ax, 84.0, (ys[0] + ys[-1]) / 2, 85.8, (ys[0] + ys[-1]) / 2, lw=0.7, ms=5)
    box(ax, 86.0, ys[-1] - 1.2, 12.3, ys[0] - ys[-1] + 2.4, fc="#F2F2F2", ec=INK,
        lw=0.7, r=0.4)
    txt(ax, 92.15, (ys[0] + ys[-1]) / 2,
        "resample\nto $T_c$,\ncenter\n\n$\\tilde{\\mathbf{Z}}_k\\in$\n$\\mathbb{R}^{T_c\\times d_k}$",
        fs=6.0)
    txt(ax, 57.5, 54.1, "pretrained weights, no fine-tuning", fs=5.6, c=SOFT,
        style="italic", ha="left")


def panel_b_cards(ax):
    """Three role cards side by side; one chip per encoder with in -> out."""
    panel(ax, 15.5, 28.8, 84.0, 26.7, "b", "Frozen encoders, grouped by role")
    arrow(ax, 14.75, 40.0, 15.45, 40.0, lw=0.7, ms=5)
    cards = [("A", 17.0, 26.4), ("M", 44.4, 26.4), ("G", 71.8, 26.4)]
    top = 52.4
    for role, x, w in cards:
        c = ROLE_C[role]
        encs = [(i, e) for i, e in enumerate(ENCODERS) if e[1] == role]
        box(ax, x, 33.2, w, top - 33.2, fc=c + "10", ec=c, lw=0.8, r=0.5, z=1)
        desc = {"A": "content", "M": "change", "G": "structure"}[role]
        txt(ax, x + w / 2, top - 1.3, ROLE_NAME[role] + "  ", fs=6.6, c=c, w="bold",
            ha="right")
        txt(ax, x + w / 2, top - 1.3, "· " + desc, fs=6.0, c=c, style="italic",
            ha="left")
        ch = 4.9
        for k, (i, (name, _, _, T, d)) in enumerate(encs):
            y1 = top - 3.0 - k * (ch + 0.45)
            box(ax, x + 0.8, y1 - ch, w - 1.6, ch, fc="white", ec=c, lw=0.6, r=0.35, z=2)
            lock(ax, x + 2.0, y1 - 1.35, s=0.34, c=c)
            txt(ax, x + 2.9, y1 - 1.35, name, fs=5.9, w="bold", ha="left")
            txt(ax, x + 2.0, y1 - 3.55,
                f"{INPUTS_S[i]}  $\\rightarrow$  "
                f"$\\mathbf{{Z}}_{i+1}\\in\\mathbb{{R}}^{{{TPRIME_S[i]}\\times d_{i+1}}}$",
                fs=5.6, ha="left")
    box(ax, 17.0, 29.6, 81.2, 2.8, fc="#F2F2F2", ec=INK, lw=0.6, r=0.3)
    txt(ax, 57.6, 31.0,
        "all encoders frozen (no fine-tuning)  $\\cdot$  every output resampled to "
        "$T_c$ rows and column-centered: $\\tilde{\\mathbf{Z}}_k\\in\\mathbb{R}^{T_c\\times d_k}$",
        fs=5.8)


PANEL_B = {"table": panel_b_table, "tree": panel_b_tree, "cards": panel_b_cards}
VARIANT = "table"


def main():
    fig = plt.figure(figsize=(7.2, 4.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    # ======================= (a) input and sampling =======================
    panel(ax, 0.5, 28.8, 14.2, 26.7, "a", "Input")
    n_native, x0, x1, ys = 72, 1.8, 13.4, 46.2
    fw = (x1 - x0) / n_native
    idx = set(np.linspace(0, n_native - 1, 24).round().astype(int))
    for i in range(n_native):
        ax.add_patch(Rectangle((x0 + i * fw, ys), fw * 0.9, 3.0,
                               fc=INK if i in idx else "#E0E0E0", ec="none", zorder=2))
    txt(ax, (x0 + x1) / 2, ys + 4.3, "video $V$, $N$ frames", fs=5.9, c=SOFT)
    txt(ax, (x0 + x1) / 2, ys - 1.4, "sample $T$ uniformly", fs=5.7, c=SOFT)
    arrow(ax, (x0 + x1) / 2, ys - 2.6, (x0 + x1) / 2, 41.2, lw=0.7, ms=5)
    for k in range(4):
        box(ax, 3.2 + 0.9 * k, 33.4 + 0.9 * k, 6.4, 4.8, fc="#F4F4F4",
            ec=SOFT, lw=0.5, r=0.15, z=2 + k)
    fx, fy = 3.2 + 2.7, 33.4 + 2.7
    ax.fill([fx + 0.4, fx + 2.3, fx + 3.7, fx + 6.0, fx + 6.0, fx + 0.4],
            [fy + 0.4, fy + 2.7, fy + 1.5, fy + 3.2, fy + 0.4, fy + 0.4],
            color="#C9C9C9", lw=0, zorder=9)
    ax.add_patch(plt.Circle((fx + 5.0, fy + 3.9), 0.45, color="#D6D6D6", zorder=9))
    txt(ax, 7.6, 30.7, "$T\\times H\\times W\\times3$", fs=6.2)

    PANEL_B[VARIANT](ax)
    if VARIANT == "table":
        arrow(ax, 14.75, 36.0, 15.45, 36.0, lw=0.7, ms=5)

    # ======================= (c) pairing ===================================
    panel(ax, 0.5, 0.5, 19.0, 27.4, "c", "Encoder pairs")
    role = [en[1] for en in ENCODERS]

    def token(x, y, k, w=2.15, h=2.0):
        c = ROLE_C[role[k]]
        box(ax, x - w / 2, y - h / 2, w, h, fc=c + "30", ec=c, lw=0.8, r=0.25, z=4)
        txt(ax, x, y, f"$\\tilde{{\\mathbf{{Z}}}}_{k + 1}$", fs=6.0, z=7)

    # pool of aligned outputs, fed from (b)
    txt(ax, 10.0, 24.0, "aligned outputs from (b)", fs=5.4, c=SOFT)
    for k in range(7):
        token(2.6 + k * 2.47, 21.3, k)
    arrow(ax, 17.6, 29.5, 17.6, 25.0, lw=0.8, ms=5)          # (b) -> (c)
    arrow(ax, 10.0, 19.8, 10.0, 17.6, lw=0.7, ms=5)
    txt(ax, 10.6, 18.7, "any two, $a<b$", fs=5.4, c=SOFT, ha="left")
    # example pairs
    examples = [(1, 2, "different roles", "compared", INK, "normal"),
                (4, 5, "same role", "control", SOFT, "normal"),
                (0, 4, "primary pair", "CLIP × MiDaS", MOT, "bold")]
    for r_, (a_, b_, lab, sub, col, wt) in enumerate(examples[:2]):
        y = 15.2 - r_ * 4.0
        token(3.0, y, a_); txt(ax, 4.6, y, "×", fs=7, c=SOFT); token(6.2, y, b_)
        txt(ax, 8.2, y + 0.55, lab, fs=5.7, w="bold", ha="left", c=col)
        txt(ax, 8.2, y - 0.75, sub, fs=5.2, c=SOFT, ha="left", style="italic")
    y = 6.6
    box(ax, 1.3, y - 1.8, 17.2, 3.6, fc="#FFF4DD", ec=MOT, lw=0.9, r=0.35, z=2)
    token(3.0, y, 0); txt(ax, 4.6, y, "×", fs=7, c=SOFT); token(6.2, y, 4)
    txt(ax, 8.2, y + 0.55, "primary pair", fs=5.7, w="bold", ha="left")
    txt(ax, 8.2, y - 0.75, "CLIP × MiDaS", fs=5.2, c=SOFT,
        ha="left", style="italic")
    arrow(ax, 18.6, y, 23.4, 14.9, lw=0.9, ms=6, rad=0.35)   # (c) -> (d)

    # ======================= (d) cross-encoder similarity ==================
    panel(ax, 20.3, 0.5, 63.0, 27.4, "d", "Cross-encoder similarity for one pair")
    # inputs: the pair from (c)
    for (lab, sub, c, y) in [("$\\mathbf{A}=\\tilde{\\mathbf{Z}}_a$", "e.g. CLIP", APP, 19.6),
                             ("$\\mathbf{B}=\\tilde{\\mathbf{Z}}_b$", "e.g. MiDaS", GEO, 10.2)]:
        box(ax, 21.2, y - 1.8, 8.6, 3.6, fc=c + "30", ec=c, lw=0.9, r=0.3, z=4)
        txt(ax, 25.5, y + 0.45, lab, fs=6.0, z=7)
        txt(ax, 25.5, y - 1.0, sub, fs=4.9, c=SOFT, style="italic", z=7)
    # frame-similarity matrices
    A = resample(feats(32, 48, seed=10)); B = resample(feats(32, 48, seed=14))
    for (S, y, c, lab) in [(cos(A), 19.6, APP, "$K_{\\mathbf{A}},\\,S_{\\mathbf{A}}$"),
                           (cos(B), 10.2, GEO, "$K_{\\mathbf{B}},\\,S_{\\mathbf{B}}$")]:
        arrow(ax, 30.0, y, 31.3, y, lw=0.6, ms=4.5)
        mat(ax, S, 31.5, y - 2.4, 4.8, 4.8, "viridis", ec=c, vmin=-1, vmax=1)
        txt(ax, 33.9, y + 3.2, lab + "$\\in\\mathbb{R}^{T_c\\times T_c}$", fs=5.6)
    # measure cards
    mx, mw, mh = 41.0, 41.3, 4.95
    cards = [("Global", "Debiased CKA  $D_{\\mathrm{CKA}}$",
              "$1-\\mathrm{CKA}_u(K_{\\mathbf{A}},\\,K_{\\mathbf{B}})$",
              "same overall structure?"),
             ("Local", "Neighborhood  $D_{\\mathrm{NBR}}$",
              "$1-\\rho_s\\left(\\mathbf{u}(S_{\\mathbf{A}}),\\,\\mathbf{u}(S_{\\mathbf{B}})\\right)$",
              "same ranking of frame pairs?"),
             ("Local", "CKNNA",
              "CKA of $K_{\\mathbf{A}},K_{\\mathbf{B}}$ on mutual $k$-NN frames",
              "same closest frames?"),
             ("Predictive", "Ridge residual  $D_{\\mathrm{RIDGE}}$",
              "$\\|\\mathbf{B}\\mathbf{W}^{\\star}-\\mathbf{A}\\|_F^2\\,/\\,\\|\\mathbf{A}\\|_F^2$",
              "$\\mathbf{W}^{\\star}$ fitted on real videos only")]
    tops = [24.3, 18.85, 13.4, 7.2]
    mids = []
    for (lvl, name, eq, q), top in zip(cards, tops):
        box(ax, mx, top - mh, mw, mh, fc="white", ec=MEAS_EC, lw=0.8, r=0.35)
        box(ax, mx + 0.6, top - 1.95, 7.4, 1.45, fc=MEAS, ec="none", r=0.25, z=3)
        txt(ax, mx + 4.3, top - 1.22, lvl, fs=5.1, c=MEAS_EC, w="bold")
        txt(ax, mx + 0.8, top - mh + 1.25, name, fs=6.0, w="bold", ha="left")
        txt(ax, mx + 26.0, top - 1.6, eq, fs=6.0)
        txt(ax, mx + mw - 0.7, top - mh + 1.1, q, fs=5.0, c=SOFT, style="italic",
            ha="right")
        mids.append(top - mh / 2)
    # similarity-based measures fed from S_A, S_B
    ax.plot([36.5, 38.0], [19.6, 19.6], color=INK, lw=0.6, zorder=3)
    ax.plot([36.5, 38.0], [10.2, 10.2], color=INK, lw=0.6, zorder=3)
    ax.plot([38.0, 38.0], [10.2, 19.6], color=INK, lw=0.6, zorder=3)
    for yy in mids[:3]:
        arrow(ax, 38.0, min(max(yy, 10.2), 19.6), mx - 0.15, yy, lw=0.55, ms=4.5)
    # ridge fed directly from A and B
    ax.plot([25.5, 25.5], [10.2 - 1.8, mids[3]], color=MEAS_EC, lw=0.6,
            ls="--", zorder=3)
    arrow(ax, 25.5, mids[3], mx - 0.15, mids[3], c=MEAS_EC, lw=0.6, ls="--", ms=4.5)
    txt(ax, 32.8, mids[3] + 0.8, "uses $\\mathbf{A},\\mathbf{B}$ directly", fs=4.9,
        c=MEAS_EC, style="italic")

    # ======================= (e) output and comparison =====================
    panel(ax, 84.1, 0.5, 15.4, 27.4, "e", "Output")
    box(ax, 85.3, 16.6, 13.0, 6.6, fc="white", ec=INK, lw=0.7)
    txt(ax, 91.8, 19.9, "one score per video\n$D(V)$; larger =\nweaker agreement", fs=5.5)
    for yy in mids:
        arrow(ax, mx + mw + 0.15, yy, 85.2, 19.9, lw=0.45, ms=4.5)
    arrow(ax, 91.8, 16.4, 91.8, 13.6, lw=0.6, ms=4.5)
    xs = np.linspace(86.0, 97.6, 200)
    for mu, c in [(90.8, APP), (92.9, MOT)]:
        yv = 6.8 + 5.4 * np.exp(-0.5 * ((xs - mu) / 1.7) ** 2)
        ax.fill_between(xs, 6.8, yv, color=c, alpha=0.35, lw=0, zorder=3)
        ax.plot(xs, yv, color=c, lw=0.7, zorder=4)
    ax.plot([86.0, 97.6], [6.8, 6.8], color=INK, lw=0.5, zorder=4)
    txt(ax, 87.4, 11.9, "real", fs=5.3, c=APP, w="bold")
    txt(ax, 96.3, 11.9, "gen.", fs=5.3, c=MOT, w="bold")
    txt(ax, 91.8, 3.6, "Cohen's $d$, AUROC,\nbootstrap CI", fs=5.3, c=SOFT)

    out = OUT if VARIANT == "table" else OUT.with_name(OUT.name + "_" + VARIANT)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"-> {out}.pdf")



# ==========================================================================
# ONE-ROW LAYOUT
# ==========================================================================
def main_row():
    fig = plt.figure(figsize=(7.2, 3.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 46.5); ax.axis("off")
    PT, PB = 46.0, 0.5                     # panel top / bottom
    H = PT - PB
    role = [en[1] for en in ENCODERS]
    ENC_SHORT = ["CLIP ViT-B/32", "DINOv2 ViT-B/14", "VideoMAE-B", "RAFT-small",
                 "MiDaS DPT-Hybrid", "Depth Anything V2", "UniDepth V2"]

    def token(x, y, k, w=1.95, h=2.2, fs=6.0):
        c = ROLE_C[role[k]]
        box(ax, x - w / 2, y - h / 2, w, h, fc=c + "30", ec=c, lw=0.8, r=0.25, z=4)
        txt(ax, x, y, f"$\\tilde{{\\mathbf{{Z}}}}_{k + 1}$", fs=fs, z=7)

    # ---------------- (a) input -----------------------------------------
    xa0, wa0 = 0.5, 9.5
    panel(ax, xa0, PB, wa0, H, "a", "Input")
    n_native, x0, x1, ys = 38, 1.5, 9.0, 36.0
    fw = (x1 - x0) / n_native
    idx = set(np.linspace(0, n_native - 1, 14).round().astype(int))
    for i in range(n_native):
        ax.add_patch(Rectangle((x0 + i * fw, ys), fw * 0.85, 3.0,
                               fc=INK if i in idx else "#E0E0E0", ec="none", zorder=2))
    txt(ax, 5.25, ys + 4.3, "video $V$\n$N$ frames", fs=6.0, c=SOFT)
    txt(ax, 5.25, ys - 2.0, "sample $T$\nuniformly", fs=6.0, c=SOFT)
    arrow(ax, 5.25, ys - 4.0, 5.25, 27.4, lw=0.7, ms=5)
    for k in range(4):
        box(ax, 1.6 + 0.8 * k, 17.0 + 0.8 * k, 5.2, 4.2, fc="#F4F4F4",
            ec=SOFT, lw=0.5, r=0.15, z=2 + k)
    fx, fy = 1.6 + 2.4, 17.0 + 2.4
    ax.fill([fx + 0.35, fx + 1.9, fx + 3.0, fx + 4.9, fx + 4.9, fx + 0.35],
            [fy + 0.35, fy + 2.3, fy + 1.3, fy + 2.8, fy + 0.35, fy + 0.35],
            color="#C9C9C9", lw=0, zorder=9)
    ax.add_patch(plt.Circle((fx + 4.1, fy + 3.4), 0.4, color="#D6D6D6", zorder=9))
    txt(ax, 5.25, 13.2, "$T\\times H$\n$\\times W\\times3$", fs=6.2)

    # ---------------- (b) frozen encoders ---------------------------------
    xb, wb = 11.0, 28.5
    panel(ax, xb, PB, wb, H, "b", "Frozen encoders")
    INP = ["$T$ frames", "$T$ frames", "$T$ frames (clips)", "$T-1$ frame pairs",
           "$T$ frames", "$T$ frames", "$T$ frames"]
    TP = ["T", "T", "T/2", "(T-1)", "T", "T", "T"]
    ch, gap, head, pad = 3.35, 0.3, 2.3, 0.45
    y = 42.6
    tok_y = {}
    for rl, desc in [("A", "content"), ("M", "change"), ("G", "structure")]:
        c = ROLE_C[rl]
        ks = [k for k in range(7) if role[k] == rl]
        hh = head + len(ks) * ch + (len(ks) - 1) * gap + pad
        box(ax, xb + 1.0, y - hh, wb - 2.0, hh, fc=c + "10", ec=c, lw=0.8, r=0.4, z=1)
        txt(ax, xb + 2.0, y - 1.15, ROLE_NAME[rl], fs=6.4, c=c, w="bold", ha="left")
        txt(ax, xb + wb - 2.0, y - 1.15, desc, fs=6.0, c=c, style="italic", ha="right")
        yy = y - head
        for k in ks:
            box(ax, xb + 1.6, yy - ch, wb - 3.2, ch, fc="white", ec=c, lw=0.6, r=0.3, z=2)
            lock(ax, xb + 2.7, yy - 1.0, s=0.32, c=c)
            txt(ax, xb + 3.5, yy - 1.0, ENC_SHORT[k], fs=6.0, w="bold", ha="left")
            txt(ax, xb + 2.4, yy - 2.45,
                f"{INP[k]} $\\rightarrow$ $\\mathbf{{Z}}_{k+1}\\in\\mathbb{{R}}^{{{TP[k]}\\times d_{k+1}}}$",
                fs=6.0, ha="left")
            tok_y[k] = yy - ch / 2
            yy -= ch + gap
        y -= hh + 0.55
    box(ax, xb + 1.0, PB + 0.7, wb - 2.0, 3.4, fc="#F2F2F2", ec=INK, lw=0.6, r=0.3)
    txt(ax, xb + wb / 2, PB + 2.4,
        "no fine-tuning; resample to $T_c$ rows:\n"
        "$\\tilde{\\mathbf{Z}}_k\\in\\mathbb{R}^{T_c\\times d_k}$", fs=6.0)
    arrow(ax, xa0 + wa0 + 0.05, 23.0, xb - 0.05, 23.0, lw=0.8, ms=5.5)

    # ---------------- (c) pairs ---------------------------------------------
    xc, wc = 40.5, 18.0
    panel(ax, xc, PB, wc, H, "c", "Encoder pairs")
    txt(ax, xc + wc / 2, 40.3, "aligned outputs", fs=6.0, c=SOFT)
    for k in range(7):
        token(xc + 1.75 + k * 2.42, 37.6, k, w=2.15)
    arrow(ax, xb + wb + 0.05, 37.6, xc - 0.05, 37.6, lw=0.8, ms=5.5)   # (b)->(c)
    arrow(ax, xc + 4.0, 36.1, xc + 4.0, 33.4, lw=0.7, ms=5)
    txt(ax, xc + 4.8, 34.8, "pick any two", fs=6.0, c=SOFT, ha="left")
    yp = 29.9
    box(ax, xc + 0.6, yp - 2.6, wc - 2.3, 5.2, fc="#FFF4DD", ec=MOT, lw=0.9, r=0.35, z=2)
    token(xc + 2.2, yp, 0); txt(ax, xc + 3.75, yp, "×", fs=7, c=SOFT); token(xc + 5.3, yp, 4)
    txt(ax, xc + 7.0, yp + 0.8, "primary pair", fs=6.2, w="bold", ha="left")
    txt(ax, xc + 7.0, yp - 0.9, "CLIP × MiDaS", fs=6.0, c=SOFT, style="italic", ha="left")
    rows_ = [(1, 2, "different roles", "$r(a)\\neq r(b)$"), (4, 5, "same role", "$r(a)=r(b)$")]
    for r_, (a_, b_, lab, sub) in enumerate(rows_):
        yy = 21.8 - r_ * 6.4
        token(xc + 2.2, yy, a_); txt(ax, xc + 3.75, yy, "×", fs=7, c=SOFT)
        token(xc + 5.3, yy, b_)
        txt(ax, xc + 6.8, yy + 0.8, lab, fs=6.0, w="bold", ha="left")
        txt(ax, xc + 6.8, yy - 0.9, sub, fs=6.0, c=SOFT, style="italic", ha="left")
    box(ax, xc + 0.6, 1.3, wc - 1.2, 8.2, fc="white", ec=FAINT, lw=0.6, r=0.3)
    txt(ax, xc + wc / 2, 7.4, "$\\mathcal{P}=\\{(a,b):\\ 1\\leq a<b\\leq K\\}$", fs=6.4)
    txt(ax, xc + wc / 2, 4.9, "$|\\mathcal{P}|=\\binom{K}{2}$,  $K$ encoders", fs=6.0)
    txt(ax, xc + wc / 2, 2.6, "$r(k)$: role of encoder $k$", fs=6.0, c=SOFT)

    # ---------------- (d) cross-encoder similarity -------------------------
    xd, wd = 59.5, 29.0
    panel(ax, xd, PB, wd, H, "d", "Similarity, for each pair")
    A = resample(feats(32, 48, seed=10)); B = resample(feats(32, 48, seed=14))
    tx = {"A": xd + 6.8, "B": xd + 20.8}
    for key, lab, sub, c in [("A", "$\\mathbf{A}=\\tilde{\\mathbf{Z}}_a$", "encoder $a$", APP),
                             ("B", "$\\mathbf{B}=\\tilde{\\mathbf{Z}}_b$", "encoder $b$", GEO)]:
        box(ax, tx[key] - 4.0, 36.2, 8.0, 3.9, fc=c + "30", ec=c, lw=0.9, r=0.3, z=4)
        txt(ax, tx[key], 38.8, lab, fs=6.2, z=7)
        txt(ax, tx[key], 37.2, sub, fs=6.0, c=SOFT, style="italic", z=7)
    # every pair (not only the primary one) goes to (d): bracket + one arrow
    brx = xc + wc - 0.6
    ax.plot([brx - 0.6, brx, brx, brx - 0.6], [yp + 2.6, yp + 2.6, 15.4 - 1.3, 15.4 - 1.3],
            color=INK, lw=0.8, zorder=3)
    ax.plot([brx, xd + 1.4], [yp + 2.6, yp + 2.6], color=INK, lw=0.8, zorder=3)
    arrow(ax, xd + 1.4, yp + 2.6, tx["A"] - 4.1, 37.4, lw=0.8, ms=5.5)
    for key, M, c in [("A", A, APP), ("B", B, GEO)]:
        arrow(ax, tx[key], 36.0, tx[key], 34.3, lw=0.6, ms=4.5)
        mat(ax, cos(M), tx[key] - 2.4, 29.2, 4.8, 4.8, "viridis", ec=c, vmin=-1, vmax=1)
        sub = "\\mathbf{A}" if key == "A" else "\\mathbf{B}"
        txt(ax, tx[key] + (4.8 if key == "A" else -4.8), 31.6,
            f"$K_{{{sub}}},\\,S_{{{sub}}}$", fs=6.0)
    txt(ax, xd + wd / 2, 42.0,
        "$K=\\bar{\\mathbf{A}}\\bar{\\mathbf{A}}^{\\top}$ (centered Gram),  $S$: cosine similarity",
        fs=6.0, c=SOFT)
    # bus: both matrices -> the three structure measures
    ax.plot([tx["A"], tx["A"]], [29.0, 27.9], color=INK, lw=0.6, zorder=3)
    ax.plot([tx["B"], tx["B"]], [29.0, 27.9], color=INK, lw=0.6, zorder=3)
    ax.plot([xd + 2.0, tx["B"]], [27.9, 27.9], color=INK, lw=0.6, zorder=3)
    cards = [("Global", "Debiased CKA", "$1-\\mathrm{CKA}_u(K_{\\mathbf{A}},K_{\\mathbf{B}})$", ""),
             ("Local", "Neighborhood", "$1-\\rho_s(\\mathbf{u}(S_{\\mathbf{A}}),\\mathbf{u}(S_{\\mathbf{B}}))$", ""),
             ("Local", "CKNNA", "CKA of $K$ on mutual $k$-NN", ""),
             ("Predictive", "Ridge residual", "$\\|\\mathbf{BW}^{\\star}-\\mathbf{A}\\|_F^2/\\|\\mathbf{A}\\|_F^2$",
              "$\\mathbf{W}^{\\star}$: real videos")]
    cx0, cw, chh = xd + 3.6, wd - 4.6, 5.25
    tops = [26.6, 20.85, 15.1, 8.1]
    mids = []
    for (lvl, name, eq, note), top in zip(cards, tops):
        box(ax, cx0, top - chh, cw, chh, fc="white", ec=MEAS_EC, lw=0.8, r=0.35)
        box(ax, cx0 + 0.5, top - 1.95, 7.0, 1.5, fc=MEAS, ec="none", r=0.25, z=3)
        txt(ax, cx0 + 4.0, top - 1.2, lvl, fs=6.0, c=MEAS_EC, w="bold")
        txt(ax, cx0 + 8.2, top - 1.2, name, fs=6.1, w="bold", ha="left")
        txt(ax, cx0 + (cw * 0.36 if note else cw / 2), top - 3.75, eq, fs=6.0)
        if note:
            txt(ax, cx0 + cw - 0.5, top - 3.75, note, fs=6.0, c=SOFT, style="italic",
                ha="right")
        mids.append(top - chh / 2)
    ax.plot([xd + 2.0, xd + 2.0], [27.9, mids[2]], color=INK, lw=0.6, zorder=3)
    for yy in mids[:3]:
        arrow(ax, xd + 2.0, yy, cx0 - 0.1, yy, lw=0.6, ms=4.5)
    # ridge: A, B directly (dashed)
    ax.plot([tx["A"] - 2.6, tx["A"] - 2.6], [36.2, 31.0], color=MEAS_EC, lw=0.7, ls="--", zorder=3)
    ax.plot([xd + 1.0, tx["A"] - 2.6], [31.0, 31.0], color=MEAS_EC, lw=0.7, ls="--", zorder=3)
    ax.plot([xd + 1.0, xd + 1.0], [31.0, mids[3]], color=MEAS_EC, lw=0.7, ls="--", zorder=3)
    arrow(ax, xd + 1.0, mids[3], cx0 - 0.1, mids[3], c=MEAS_EC, lw=0.7, ls="--", ms=4.5)
    txt(ax, cx0 + cw / 2, tops[3] + 0.9, "uses $\\mathbf{A},\\mathbf{B}$ directly",
        fs=6.0, c=MEAS_EC, style="italic")

    # ---------------- (e) output -------------------------------------------
    xe, we = 89.5, 10.0
    panel(ax, xe, PB, we, H, "e", "Output")
    box(ax, xe + 0.6, 29.0, we - 1.2, 9.0, fc="white", ec=INK, lw=0.8)
    txt(ax, xe + we / 2, 33.5, "score per\nvideo\n$D(V)$\nlarger =\nweaker\nagreement",
        fs=6.0)
    for yy in mids:
        arrow(ax, cx0 + cw + 0.1, yy, xe + 0.5, 33.5, lw=0.5, ms=4.5)
    arrow(ax, xe + we / 2, 28.8, xe + we / 2, 24.6, lw=0.7, ms=5)
    xs = np.linspace(xe + 0.8, xe + we - 0.8, 200)
    mid = xe + we / 2
    for mu, c in [(mid - 0.9, APP), (mid + 0.9, MOT)]:
        yv = 14.5 + 7.5 * np.exp(-0.5 * ((xs - mu) / 1.35) ** 2)
        ax.fill_between(xs, 14.5, yv, color=c, alpha=0.35, lw=0, zorder=3)
        ax.plot(xs, yv, color=c, lw=0.7, zorder=4)
    ax.plot([xe + 0.8, xe + we - 0.8], [14.5, 14.5], color=INK, lw=0.5, zorder=4)
    txt(ax, mid - 2.6, 23.0, "real", fs=6.0, c=APP, w="bold")
    txt(ax, mid + 2.6, 23.0, "gen.", fs=6.0, c=MOT, w="bold")
    txt(ax, mid, 8.5, "Cohen's $d$,\nAUROC,\nbootstrap CI", fs=6.0, c=SOFT)

    out = OUT.with_name(OUT.name + "_row")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"-> {out}.pdf")


if __name__ == "__main__":
    import sys
    for v in (sys.argv[1:] or ["table"]):
        VARIANT = v
        main_row() if v == "row" else main()
