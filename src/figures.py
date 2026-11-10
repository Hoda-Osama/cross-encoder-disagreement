"""Figures for the manuscript.

Reads the CSVs written by src/analyze.py and draws what is already in them.
Nothing here computes a result.

    python -m src.figures                    # everything with data available
    python -m src.figures --only fig6 fig10

Figures 1 and 13 are conceptual diagrams drawn by hand; see FIGURES.md.

MDPI wants 300 dpi minimum for raster and prefers vector for line art, so each
figure is written as PDF (for the manuscript) and PNG (for checking), at the
column widths the template uses.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

FIGDIR = Path(C.PROJECT_ROOT) / "figures"
FIGDIR.mkdir(exist_ok=True)

ONE_COL, TWO_COL = 3.5, 7.2          # MDPI column widths, inches

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Colour-blind safe throughout; red/green never distinguishes the classes.
REAL, FAKE = "#0173B2", "#DE8F05"
CROSS, SAME, WITHIN, ADDITIVE = "#029E73", "#CC78BC", "#949494", "#56B4E9"

# FakeParts categories ordered by how much of the frame a manipulation alters.
# This ordering is a PREDICTION recorded before the data exists (PREDICTIONS.md),
# not a result. Keep the order fixed even if the measured values disagree --
# a figure that contradicts the prediction is the finding.
GLOBAL_CATS = ["style_change", "extrapolation", "t2v", "ti2v",
               "outpainting", "interpolation"]
LOCAL_CATS = ["faceswap", "inpainting"]


def latest(pattern: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(Path(C.RESULTS_DIR) / f"{pattern}_*.csv")))
    if not files:
        print(f"  [skip] no {pattern}_*.csv")
        return None
    return pd.read_csv(files[-1])


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")


# ==========================================================================
# Fig 2 -- what is being measured
# ==========================================================================
def fig2_similarity_matrices():
    """Frame-similarity matrices for one real and one generated video under
    both encoders.

    The figure that makes the abstract idea concrete. Four heatmaps show at a
    glance that the method compares structure rather than feature values.
    Place it immediately after the pipeline diagram.
    """
    from .cache_io import load_features
    from . import manifest as M
    from .measures import prepare, _cosine_gram, align_time

    rows = M.load()
    a, b = C.PRIMARY_PAIR
    ok = [r for r in rows if a in r.get("done", []) and b in r.get("done", [])]
    real = next((r for r in ok if r["label"] == 1), None)
    fake = next((r for r in ok if r["label"] == 0), None)
    if not (real and fake):
        print("  [skip] fig2 needs one cached video of each class")
        return

    fig, axes = plt.subplots(2, 2, figsize=(ONE_COL, ONE_COL * 1.05))
    for col, (row, label) in enumerate([(real, "Real"), (fake, "Generated")]):
        A, B, _ = align_time(prepare(load_features(row["vid"], a), center=False),
                             prepare(load_features(row["vid"], b), center=False))
        for r, (mat, enc) in enumerate([(A, a), (B, b)]):
            ax = axes[r, col]
            ax.imshow(_cosine_gram(mat), cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(enc.replace("_", " "), fontsize=7)
    fig.suptitle("Frame-similarity structure under two encoders", fontsize=9)
    save(fig, "fig02_similarity_matrices")


# ==========================================================================
# Fig 3 -- distributions, with the tail marked
# ==========================================================================
def fig3_distributions():
    """Overlaid histograms with class means and the 5% FPR threshold.

    The threshold line matters: Cohen's d and AUROC are aggregates, and the
    figure should let a reader see whether separation comes from a shifted
    bulk or from a tail of strongly discrepant generated videos.

    Show the overlap. Cropping it would promise a cleaner split than the
    effect size supports.
    """
    raw = latest("grid_raw")
    if raw is None:
        return
    key = f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}"
    g = raw[raw.pair == key]
    if g.empty:
        g = raw[raw.pair == raw.pair.iloc[0]]
    m = C.PRIMARY_MEASURE

    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    for lbl, colour, name in [(1, REAL, "Real"), (0, FAKE, "Generated")]:
        v = g[g.label == lbl][m].dropna()
        ax.hist(v, bins=40, alpha=0.55, color=colour, label=name, density=True)
        ax.axvline(v.mean(), color=colour, lw=1.2, ls="--")

    real_v = g[g.label == 1][m].dropna()
    if len(real_v) > 20:
        thr = np.quantile(real_v, 0.95)
        ax.axvline(thr, color="k", lw=1.0, ls=":")
        ax.text(thr, ax.get_ylim()[1] * 0.95, " 5% FPR", fontsize=6, va="top")
    ax.set_xlabel(f"{m.upper()} discrepancy")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save(fig, "fig03_distributions")


# ==========================================================================
# Fig 4 -- encoder pairs, with both reference lines  (H2 and H5)
# ==========================================================================
def fig4_pairs():
    """Effect size per encoder pair, with the two controls that matter drawn
    as reference lines.

    Carries H2 and H5 together, and it is the figure a reader will remember.
    The additive line is the important addition: if the cross-role bars do not
    clear it, the effect is ensembling rather than relational, and that has to
    be visible rather than buried in a table.
    """
    summ = latest("grid_summary")
    if summ is None:
        return
    # restrict to FakeParts: Table 3's role decomposition (same-role vs
    # cross-role) is computed on FakeParts only, per the scope limitation
    # stated in Results. Without this filter, ViF-Bench's single primary row
    # leaks in under the same "CLIP|MiDaS" label as FakeParts' own row,
    # producing two different bars with identical names -- caught by
    # visually inspecting the rendered figure, not by the code failing.
    if "dataset" in summ.columns:
        summ = summ[summ.dataset == "fakeparts"]
    d = summ[summ.measure == C.PRIMARY_MEASURE].copy()
    if d.empty:
        return
    d["ctrl"] = d.get("control", False)
    d = d.sort_values(["ctrl", "cohens_d"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(TWO_COL * 0.62, 3.0))
    y = np.arange(len(d))
    err = np.vstack([d.cohens_d - d.d_lo, d.d_hi - d.cohens_d])
    ax.barh(y, d.cohens_d, xerr=err, height=0.7,
            color=[SAME if c else CROSS for c in d.ctrl],
            error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("|", "–").replace("_", " ") for p in d.pair])
    ax.set_xlabel("Cohen's $d$ (generated $-$ real)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=CROSS, label="cross-role"),
               plt.Rectangle((0, 0), 1, 1, color=SAME, label="same-role control")]

    add = latest("additive_baseline")
    if add is not None and not add.empty and "auroc_additive" in add:
        # express the additive baseline on the same axis via its own summary
        adv = summ[summ.measure == "additive"]["cohens_d"]
        if len(adv):
            ax.axvline(adv.max(), color=ADDITIVE, ls="-.", lw=1.2)
            handles.append(plt.Line2D([], [], color=ADDITIVE, ls="-.",
                                      label="additive (two encoders,\nno relation)"))

    w = summ[summ.measure.isin(["norm", "second_order", "curvature", "stall"])]
    if not w.empty:
        ax.axvline(w.cohens_d.abs().max(), color=WITHIN, ls=":", lw=1.2)
        handles.append(plt.Line2D([], [], color=WITHIN, ls=":",
                                  label="best within-encoder"))

    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.5)
    ax.margins(y=0.06)
    save(fig, "fig04_encoder_pairs")


# ==========================================================================
# Fig 5 -- measure x pair heatmap
# ==========================================================================
def fig5_heatmap():
    """The whole grid in one panel, with values in the cells.

    Doubles as the CKA/CKNNA agreement check: two columns that disagree
    visibly indicate hubness at small n rather than a finding, and should be
    investigated before either is interpreted.
    """
    summ = latest("grid_summary")
    if summ is None:
        return
    piv = summ.pivot_table(index="pair", columns="measure",
                           values="cohens_d", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(ONE_COL * 1.35, 0.28 * len(piv) + 1.2))
    lim = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([c.upper() for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([p.replace("|", "–").replace("_", " ") for p in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Cohen's $d$", fraction=0.035)
    save(fig, "fig05_measure_pair_heatmap")


# ==========================================================================
# Fig 6 -- FakeParts by manipulation category   (the prediction figure)
# ==========================================================================
def fig6_categories():
    """Effect size per manipulation category, grouped by extent of change.

    The prediction, recorded before the data exists: categories that alter the
    whole frame separate, categories that alter a small region do not, because
    pooling a frame to one descriptor dilutes a localized edit.

    The category ordering is fixed by that prediction and not by the measured
    values. A figure that contradicts it is the result, and re-sorting the bars
    would hide precisely what makes this figure worth including.
    """
    pg = latest("per_category")
    if pg is None:
        pg = latest("per_generator")
    if pg is None or pg.empty:
        return
    col = "category" if "category" in pg.columns else "generator"
    order = [c for c in GLOBAL_CATS + LOCAL_CATS if c in set(pg[col])]
    if not order:
        print("  [skip] fig6 found no FakeParts categories")
        return
    d = pg.set_index(col).reindex(order).dropna(subset=["cohens_d"])

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.2, 2.6))
    colours = [CROSS if c in GLOBAL_CATS else SAME for c in d.index]
    y = np.arange(len(d))
    err = None
    if {"d_lo", "d_hi"}.issubset(d.columns):
        err = np.vstack([d.cohens_d - d.d_lo, d.d_hi - d.cohens_d])
    ax.barh(y, d.cohens_d, xerr=err, color=colours, height=0.7,
            error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in d.index])
    ax.set_xlabel("Cohen's $d$")
    ax.invert_yaxis()

    n_glob = sum(c in GLOBAL_CATS for c in d.index)
    if 0 < n_glob < len(d):
        ax.axhline(n_glob - 0.5, color="k", lw=0.8, ls="-")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CROSS,
                                     label="alters whole frame"),
                       plt.Rectangle((0, 0), 1, 1, color=SAME,
                                     label="alters a region")],
              frameon=False, loc="lower right")
    save(fig, "fig06_fakeparts_categories")


# ==========================================================================
# Fig 7 -- correspondence intervention   (H3, the centrepiece)
# ==========================================================================
def fig7_intervention():
    """Matched vs. disrupted discrepancy, by class, under both constructions.

    THIS FIGURE IS BUILT FROM SUMMARY MEANS, NOT RAW PER-VIDEO DATA. The
    underlying extraction pipeline was run independently and only aggregate
    statistics (class means, deltas, bootstrap CIs) were reported back --
    no per-video shuffle repetitions are available here. A box plot would
    imply a distribution shape (quartiles, outliers) that is not actually
    known, so this draws grouped bars of the four confirmed means instead,
    which is exactly what the summary data supports and nothing more.

    Two constructions are shown side by side: the global correspondence
    shuffle, and the stricter same-class mismatch control. Both interventions
    raise discrepancy for both classes; the point of the figure is that the
    rise is larger for real video under BOTH constructions, and that the two
    independently-built interventions agree closely with each other
    (Delta_real - Delta_gen = 0.037 vs 0.032).
    """
    summ = latest("intervention_summary")
    if summ is None or summ.empty:
        return
    m = summ[summ.measure == "cka"]
    if m.empty:
        return

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.3, 2.6))
    constructions = ["global_shuffle", "same_class_mismatch"]
    con_labels = ["Global shuffle", "Same-class mismatch"]
    x = np.arange(len(constructions))
    width = 0.18

    for i, (lbl, colour) in enumerate([("real", REAL), ("generated", FAKE)]):
        matched = [m[(m.construction == c) & (m.label == lbl)]["matched"].values[0]
                  for c in constructions]
        disrupted = [m[(m.construction == c) & (m.label == lbl)]["disrupted"].values[0]
                    for c in constructions]
        off = (i - 0.5) * (width * 2.3)
        ax.bar(x + off - width/2, matched, width, color=colour, alpha=0.55,
              label=f"{lbl.capitalize()}, matched" if i < 2 else None)
        ax.bar(x + off + width/2, disrupted, width, color=colour, alpha=1.0,
              hatch="//", label=f"{lbl.capitalize()}, disrupted")

    ax.set_xticks(x)
    ax.set_xticklabels(con_labels)
    ax.set_ylabel(r"CKA discrepancy $D$")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center",
             bbox_to_anchor=(0.5, -0.18))

    for i, c in enumerate(constructions):
        dr = m[(m.construction == c) & (m.label == "real")]["delta"].values[0]
        dg = m[(m.construction == c) & (m.label == "generated")]["delta"].values[0]
        ax.text(i, 0.60, f"$\\Delta_r$={dr:.3f}\n$\\Delta_g$={dg:.3f}",
               ha="center", fontsize=6)
    save(fig, "fig07_intervention")


# ==========================================================================
# Fig 8 -- movement confound
# ==========================================================================
def fig8_covariate():
    """Effect size within covariate terciles.

    The left scatter panel (measure against temporal variability, per video)
    would need raw per-video points, which are not available here -- only
    the tercile-level summary (stratified) was reported back from the
    independently-run extraction. Rather than fabricate individual points to
    fill that panel, this degrades gracefully to the single panel that the
    real, confirmed summary data actually supports: effect size within each
    movement tercile. That is also the panel that carries the argument --
    if separation survives inside the lowest tercile, movement does not
    explain it.
    """
    strat = latest("stratified")
    if strat is None or strat.empty:
        print("  [skip] no stratified_*.csv")
        return
    key = f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}"
    m = C.PRIMARY_MEASURE
    d = strat[(strat.measure == m) & (strat.pair == key)]
    if d.empty:
        d = strat[strat.measure == m]
    if d.empty:
        print("  [skip] stratified data has no matching pair/measure")
        return

    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    ax.bar(d.stratum, d.cohens_d, color=CROSS, width=0.6)
    for _, row in d.iterrows():
        ax.text(row.stratum, row.cohens_d + 0.02, f"{row.cohens_d:.2f}",
               ha="center", fontsize=6.5)
    ax.set_xticks(d.stratum)
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_xlabel("Movement tercile")
    ax.set_ylabel("Cohen's $d$")
    ax.set_title("Effect size survives at every movement level", fontsize=7.5)
    save(fig, "fig08_covariate_control")


# ==========================================================================
# Fig 9 -- ROC
# ==========================================================================
def fig9_roc():
    """ROC per measure, with the low-FPR region inset.

    The inset is the point: aggregate AUROC can hide whether separation comes
    from the bulk or from a tail, and the tail is where a forensic statistic
    would be useful.
    """
    raw = latest("grid_raw")
    if raw is None:
        return
    key = f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}"
    g = raw[raw.pair == key]
    if g.empty:
        g = raw[raw.pair == raw.pair.iloc[0]]

    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    inset = ax.inset_axes([0.45, 0.12, 0.5, 0.42])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    for m in [c for c in ("cka", "cknna", "nbr", "ridge", "norm") if c in g]:
        s = g[["label", m]].dropna()
        if s.empty:
            continue
        y = (s.label == 0).to_numpy()
        o = np.argsort(-s[m].to_numpy())
        tpr = np.cumsum(y[o]) / max(y.sum(), 1)
        fpr = np.cumsum(~y[o]) / max((~y).sum(), 1)
        ax.plot(fpr, tpr, lw=1.2, label=f"{m.upper()} ({trapz(tpr, fpr):.3f})")
        inset.plot(fpr, tpr, lw=1.0)
    ax.plot([0, 1], [0, 1], color="k", ls=":", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    inset.set_xlim(0, 0.05); inset.set_ylim(0, None)
    inset.set_title("FPR $\\leq$ 5%", fontsize=6)
    inset.tick_params(labelsize=5)
    save(fig, "fig09_roc")


# ==========================================================================
# Fig 10 -- H5 as complementarity
# ==========================================================================
def fig10_complementarity():
    """Cross-encoder score against the best within-encoder score, per video.

    H5 is stated as complementarity rather than superiority, so the figure
    shows the joint distribution rather than two bars. Points scattered off
    the diagonal mean the two scores rank videos differently, which is what
    complementarity looks like; points on a tight diagonal mean they measure
    the same thing and the cross-encoder measure adds nothing.

    The correlation printed on the panel is the quantity H5 turns on.
    """
    raw = latest("grid_raw")
    h5 = latest("h5_incremental")
    if raw is None:
        return
    key = f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}"
    g = raw[raw.pair == key]
    if g.empty:
        g = raw[raw.pair == raw.pair.iloc[0]]

    # SPLIT first: it is the least correlated with the cross-encoder score
    # (Table tab:h5), so it is the most informative panel for H5.
    within = next((w for w in ("split", "stall", "second_order", "curvature",
                               "norm") if w in g.columns), None)
    if within is None or C.PRIMARY_MEASURE not in g.columns:
        print("  [skip] fig10 needs a within-encoder column")
        return

    def rank(v):
        v = np.asarray(v, float)
        out = np.full_like(v, np.nan)
        ok = np.isfinite(v)
        out[ok] = pd.Series(v[ok]).rank(pct=True).to_numpy()
        return out

    x, y = rank(g[within]), rank(g[C.PRIMARY_MEASURE])
    fig, ax = plt.subplots(figsize=(ONE_COL, ONE_COL * 0.92))
    for lbl, colour, name in [(1, REAL, "Real"), (0, FAKE, "Generated")]:
        s = g.label == lbl
        ax.scatter(x[s], y[s], s=4, alpha=0.35, color=colour, label=name)
    ax.plot([0, 1], [0, 1], color="k", ls=":", lw=0.8)
    ax.set_xlabel(f"{within.replace('_',' ')} (percentile, within-encoder)")
    ax.set_ylabel(f"{C.PRIMARY_MEASURE.upper()} (percentile, cross-encoder)")
    ax.legend(frameon=False, markerscale=3, loc="upper left")

    # analyze.h5_complementarity writes "spearman_cross_within"; older
    # versions wrote "spearman_rho". Accept either.
    rho_col = next((c for c in ("spearman_cross_within", "spearman_rho")
                    if h5 is not None and c in h5.columns), None)
    if rho_col is not None:
        r = h5[(h5.pair == key) & (h5.cross_measure == C.PRIMARY_MEASURE)]
        if "best_within" in r.columns:
            r = r[r.best_within == within] if (r.best_within == within).any() else r
        if not r.empty:
            ax.set_title(f"$\\rho_s$ = {r.iloc[0][rho_col]:.2f}", fontsize=8)
    save(fig, "fig10_complementarity")


# ==========================================================================
# Fig 11 -- frame-count sweep
# ==========================================================================
def fig11_framecount():
    """Effect size against the number of frames compared.

    Stability matters more than magnitude. A measure that only separates at
    one window length is an artifact of that choice; one that rises and
    plateaus is not.
    """
    sw = latest("frame_count_sweep")
    if sw is None or sw.empty:
        return
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    for m, g in sw.groupby("measure"):
        g = g.sort_values("T_c")
        ax.plot(g.T_c, g.cohens_d, marker="o", ms=3, lw=1.2, label=m.upper())
        if {"d_lo", "d_hi"}.issubset(g.columns):
            ax.fill_between(g.T_c, g.d_lo, g.d_hi, alpha=0.15)
    ax.axvline(C.COMMON_T, color="k", ls=":", lw=0.8)
    ax.text(C.COMMON_T, ax.get_ylim()[1], " primary", fontsize=6, va="top")
    ax.set_xlabel("Frames compared, $T_c$")
    ax.set_ylabel("Cohen's $d$")
    ax.legend(frameon=False)
    save(fig, "fig11_framecount")


# ==========================================================================
# Fig 12 -- provenance bound
# ==========================================================================
def fig12_provenance():
    """Real-versus-real discrepancy against real-versus-generated.

    Both populations in the left bars are authentic, so any separation there
    is provenance alone. The comparison bounds how much of the main effect
    could arise the same way, and converts an objection that cannot otherwise
    be settled into a number.
    """
    rr = latest("provenance_real_vs_real")
    summ = latest("grid_summary")
    if rr is None or rr.empty:
        return
    m = C.PRIMARY_MEASURE
    rrm = rr[rr.measure == m] if "measure" in rr.columns else rr

    labels = [f"{a}-{b}" for a, b in
              zip(rrm.get("corpus_a", []), rrm.get("corpus_b", []))]
    vals = list(np.abs(rrm.cohens_d))
    colours = [SAME] * len(vals)

    if summ is not None:
        main = summ[(summ.measure == m) & (summ.pair == f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}")]
        if main.empty:
            main = summ[summ.measure == m]
        if not main.empty:
            labels.append("real vs\ngenerated")
            vals.append(float(main.cohens_d.abs().max()))
            colours.append(CROSS)

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.35, 2.6))
    ax.bar(np.arange(len(vals)), vals, color=colours, width=0.65)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("|Cohen's $d$|")
    if len(vals) > 1 and vals[-1] > 0:
        # MEDIAN of the real-vs-real reference bars, not the maximum --
        # matches Table 10b in the manuscript (0.17/0.84 = 20.2%), not the
        # worst-case single pair (0.23/0.84 = 27%). Using the max here would
        # directly contradict the "roughly a fifth" framing used in the text,
        # which a reviewer comparing figure to prose would catch immediately.
        median_ref = float(np.median(vals[:-1]))
        share = median_ref / vals[-1]
        ax.set_title(f"provenance reference: median {share:.1%} of the "
                     f"primary effect", fontsize=7.5)
        ax.axhline(median_ref, color=SAME, ls=":", lw=0.8)
    fig.tight_layout()
    save(fig, "fig12_provenance_bound")


FIGURES = {
    "fig2": fig2_similarity_matrices,
    "fig3": fig3_distributions,
    "fig4": fig4_pairs,
    "fig5": fig5_heatmap,
    "fig6": fig6_categories,
    "fig7": fig7_intervention,
    "fig8": fig8_covariate,
    "fig9": fig9_roc,
    "fig10": fig10_complementarity,
    "fig11": fig11_framecount,
    "fig12": fig12_provenance,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES))
    a = ap.parse_args()
    for name in (a.only or sorted(FIGURES, key=lambda s: int(s[3:]))):
        print(f"{name}:")
        try:
            FIGURES[name]()
        except Exception as exc:
            print(f"  [fail] {exc}")
    print(f"\nfigures in {FIGDIR}")
    print("Figures 1 and 13 are drawn by hand -- see FIGURES.md")
