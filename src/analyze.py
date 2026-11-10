"""Analysis. CPU only.

Computes every measure over every encoder pair, then reports effect sizes,
AUROC and bootstrap confidence intervals. Once the feature cache exists this
runs in minutes, so the analysis can be iterated on freely.

Usage
-----
    python -m src.analyze pilot                 # the go/no-go gate
    python -m src.analyze grid                  # full measure x pair grid
    python -m src.analyze grid --dataset gvd --measures cka nbr
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import manifest as M
from . import measures as ms
from .cache_io import load_features


# ==========================================================================
# Statistics
# ==========================================================================
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Standardised mean difference, generated minus real."""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / (s + 1e-12))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC. `pos` should be the generated class.

    Reported so that values below 0.5 are visible rather than folded away:
    a measure that is *lower* for generated video is still informative, and
    silently flipping it would hide a sign disagreement between encoder pairs.
    """
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def bootstrap_ci(fn, x, y, n_boot=2000, alpha=0.05, seed=C.SEED):
    rng = np.random.default_rng(seed)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    stats = [fn(rng.choice(x, len(x), replace=True),
               rng.choice(y, len(y), replace=True)) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ==========================================================================
# Scoring
# ==========================================================================
def available(rows, encoder):
    return [r for r in rows if encoder in r.get("done", [])]


def score_pair(rows, enc_a, enc_b, measures):
    """Compute all requested measures for one encoder pair over all videos."""
    usable = [r for r in rows
              if enc_a in r.get("done", []) and enc_b in r.get("done", [])]
    if len(usable) < 20:
        return pd.DataFrame()

    ridge = ridge_rev = randproj = None
    if "ridge" in measures:
        # FIX 6 -- both directions. b_to_a (complementary predicts
        # appearance) is primary because it mirrors Paper 1's CMID.
        ridge = _fit_ridge(usable, enc_a, enc_b, direction=C.RIDGE_DIRECTION)
        rev = "a_to_b" if C.RIDGE_DIRECTION == "b_to_a" else "b_to_a"
        ridge_rev = _fit_ridge(usable, enc_a, enc_b, direction=rev)
    if "randproj" in measures:
        probe_a = load_features(usable[0]["vid"], enc_a)
        probe_b = load_features(usable[0]["vid"], enc_b)
        randproj = ms.RandomProjectionCMID(probe_a.shape[1], probe_b.shape[1])

    out = []
    for r in usable:
        A, B = load_features(r["vid"], enc_a), load_features(r["vid"], enc_b)
        if A is None or B is None:
            continue
        rec = {"vid": r["vid"], "dataset": r["dataset"], "label": r["label"],
               "generator": r["generator"], "pair": f"{enc_a}|{enc_b}"}
        # FIX 3 -- confound covariate, recorded for every video so effect
        # sizes can be reported stratified by how much the video changes.
        try:
            rec["covariate"] = ms.temporal_variability(A)
        except Exception:
            rec["covariate"] = np.nan
        for m in measures:
            try:
                if m == "cka":
                    rec[m] = ms.cka(A, B)
                elif m == "cknna":
                    rec[m] = ms.cknna(A, B, k=C.CKNNA_K)
                elif m == "nbr":
                    rec[m] = ms.neighbourhood(A, B)
                elif m == "ridge":
                    ok = ridge and not r.get("_in_ridge_fit")
                    rec[m] = ridge.score(A, B) if ok else np.nan
                    rec["ridge_rev"] = (ridge_rev.score(A, B)
                                        if ok and ridge_rev else np.nan)
                elif m == "randproj":
                    rec[m] = randproj.score(A, B)
                elif m == "norm":
                    rec[m] = ms.feature_norm_stat(A)
                elif m == "lstsq_degenerate":
                    rec[m] = ms.linear_residual_per_video(A, B)
            except Exception:
                rec[m] = np.nan
        out.append(rec)
    return pd.DataFrame(out)


def _fit_ridge(rows, enc_a, enc_b, n_fit=150, seed=C.SEED,
               direction=None):
    """Fit the reference map on REAL videos only, held out from scoring.

    Videos used for fitting are tagged so `score_pair` excludes them; scoring
    a video with the map fitted to it would give an optimistically low residual.
    """
    real = [r for r in rows if r["label"] == 1]
    if len(real) < 30:
        return None
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(real), min(n_fit, len(real) // 2), replace=False)
    pairs = []
    for i in pick:
        real[i]["_in_ridge_fit"] = True
        A, B = load_features(real[i]["vid"], enc_a), load_features(real[i]["vid"], enc_b)
        if A is not None and B is not None:
            pairs.append((A, B))
    if not pairs:
        return None
    return ms.RidgeReference(
        direction=direction or C.RIDGE_DIRECTION).fit(pairs)


def summarise(df, measures):
    """Effect size, AUROC and bootstrap CIs per (pair, measure)."""
    out = []
    for (pair, ds), g in df.groupby(["pair", "dataset"]):
        real = g[g.label == 1]
        fake = g[g.label == 0]
        for m in measures:
            if m not in g:
                continue
            x, y = fake[m].to_numpy(float), real[m].to_numpy(float)
            d = cohens_d(x, y)
            a = auroc(x, y)
            d_lo, d_hi = bootstrap_ci(cohens_d, x, y)
            out.append({
                "pair": pair, "dataset": ds, "measure": m,
                "n_real": int(np.isfinite(y).sum()),
                "n_fake": int(np.isfinite(x).sum()),
                "mean_real": float(np.nanmean(y)), "mean_fake": float(np.nanmean(x)),
                "cohens_d": d, "d_lo": d_lo, "d_hi": d_hi,
                "auroc": a, "auroc_dist": abs(a - 0.5) + 0.5 if np.isfinite(a) else np.nan,
                # permutation over videos, not a t-test over frames:
                # the video is the independent unit, the frame is not
                "perm_p": permutation_test(x, y, n_perm=2000)[1],
                # tail behaviour: d and AUROC are aggregates and would hide a
                # subset of extremely discrepant generated videos
                **recall_at_fpr(x, y),
            })
    return pd.DataFrame(out).sort_values("auroc_dist", ascending=False)


def per_generator(df, measure):
    out = []
    real = df[df.label == 1]
    for (pair, ds, gen), g in df[df.label == 0].groupby(["pair", "dataset", "generator"]):
        ref = real[(real.pair == pair) & (real.dataset == ds)][measure].to_numpy(float)
        x = g[measure].to_numpy(float)
        out.append({"pair": pair, "dataset": ds, "generator": gen, "measure": measure,
                    "n": len(x), "cohens_d": cohens_d(x, ref), "auroc": auroc(x, ref)})
    return pd.DataFrame(out).sort_values(["pair", "auroc"])


def per_category(df, measure, dataset="fakeparts"):
    """Table 6 -- FakeParts by manipulation category.

    Exact sibling of per_generator above, with 'category' as the grouping
    key instead of 'generator'. Requires the raw dataframe to carry a
    'category' column for FakeParts rows (T2V, TI2V, Interpolation,
    Extrapolation, Style_change, Outpainting, Inpainting, FaceSwap),
    populated at extraction time the same way 'generator' already is for
    ViF-Bench rows -- see src/manifest.py or wherever the per-video row
    dicts are first assembled. If that column does not exist yet, this
    function will raise a KeyError, which is the correct failure mode: it
    means the extraction manifest needs the field added, not that this
    function needs changing.

    No new extraction is required if FakeParts features are already cached:
    this groups already-computed per-video CKA scores by their existing
    category label, it does not recompute anything.

    The real reference is the FULL FakeParts real pool (12,075 videos, all
    five folders pooled), not a per-category real subset -- FakeParts'
    manipulations are applied to real video, so there is no separate
    "real T2V" class to compare against. Every category row in Table 6
    is compared against the same real distribution.
    """
    out = []
    real = df[(df.label == 1) & (df.dataset == dataset)]
    fake = df[(df.label == 0) & (df.dataset == dataset)]
    if "category" not in fake.columns:
        raise KeyError(
            "'category' column not found on FakeParts rows. Add it to the "
            "per-video row dict at extraction time (T2V, TI2V, Interpolation, "
            "Extrapolation, Style_change, Outpainting, Inpainting, FaceSwap), "
            "then re-run -- no re-extraction of features needed, only the "
            "manifest/label needs updating.")
    for (pair, cat), g in fake.groupby(["pair", "category"]):
        ref = real[real.pair == pair][measure].to_numpy(float)
        x = g[measure].to_numpy(float)
        d = cohens_d(x, ref)
        d_lo, d_hi = bootstrap_ci(cohens_d, x, ref)
        out.append({"pair": pair, "category": cat, "measure": measure,
                    "n": len(x), "cohens_d": d, "d_lo": d_lo, "d_hi": d_hi,
                    "auroc": auroc(x, ref)})
    order = ["T2V", "TI2V", "Interpolation", "Extrapolation", "Style_change",
             "Outpainting", "Inpainting", "FaceSwap"]
    result = pd.DataFrame(out)
    if result.empty:
        return result
    result["category"] = pd.Categorical(result["category"], categories=order, ordered=True)
    return result.sort_values(["pair", "category"])


def stratified(df, measures, k=None):
    """FIX 3 -- effect size within covariate strata.

    If the separation survives inside each stratum, it is not explained by
    the videos simply changing more or less. If it does not, the covariate is
    what the measure is tracking, and that must be reported.
    """
    k = k or C.COVARIATE_STRATA
    out = []
    for pair, g in df.groupby("pair"):
        cov = g["covariate"].to_numpy(float)
        finite = np.isfinite(cov)
        if finite.sum() < 30:
            continue
        edges = np.nanquantile(cov[finite], np.linspace(0, 1, k + 1))
        for i in range(k):
            lo, hi = edges[i], edges[i + 1]
            sel = g[(g.covariate >= lo) & (g.covariate <= hi)]
            for m in measures:
                if m not in sel:
                    continue
                x = sel[sel.label == 0][m].to_numpy(float)
                y = sel[sel.label == 1][m].to_numpy(float)
                out.append({"pair": pair, "measure": m, "stratum": i + 1,
                            "cov_lo": lo, "cov_hi": hi,
                            "n": len(sel), "cohens_d": cohens_d(x, y),
                            "auroc": auroc(x, y)})
    return pd.DataFrame(out)


def covariate_correlation(df, measures):
    """Correlation between each measure and the confound covariate."""
    from scipy.stats import spearmanr
    out = []
    for pair, g in df.groupby("pair"):
        for m in measures:
            if m not in g:
                continue
            v, c = g[m].to_numpy(float), g["covariate"].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(c)
            if ok.sum() < 20:
                continue
            rho, p = spearmanr(v[ok], c[ok])
            out.append({"pair": pair, "measure": m, "n": int(ok.sum()),
                        "spearman_rho": float(rho), "p": float(p)})
    return pd.DataFrame(out)


def report_primary(summ):
    """FIX 4 -- the pre-specified headline, reported before the grid."""
    key = f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}"
    row = summ[(summ.pair == key) & (summ.measure == C.PRIMARY_MEASURE)]
    print("\n" + "=" * 74)
    print(f"PRE-SPECIFIED PRIMARY COMPARISON: {key}, {C.PRIMARY_MEASURE}")
    print("=" * 74)
    if row.empty:
        print("  not available in this run")
    else:
        print(row[["dataset", "n_real", "n_fake", "cohens_d",
                   "d_lo", "d_hi", "auroc"]]
              .to_string(index=False, float_format="%.4f"))
    print("\nEverything below is exploratory. The maximum effect size over")
    print("the full grid is a maximum over ~60 tests and is biased upward;")
    print("report it as such, not as the headline.")
    return row


# ==========================================================================
# ROBUSTNESS AND CONFOUND CONTROLS
# Added in response to external review. Each addresses a specific objection.
# ==========================================================================

def permutation_test(x, y, n_perm=10000, seed=C.SEED):
    """Label-permutation test on the difference of means.

    WHY THIS REPLACES THE t-TEST. The per-video statistic is computed from
    T_c = 16 frames that are strongly autocorrelated, so the effective sample
    size underlying each score is well below 16 and any parametric test that
    assumes independent observations is mis-specified.

    Permuting the LABEL across videos sidesteps this entirely: the resampling
    unit is the video, which is the unit that is actually independent. The
    within-video dependence structure is carried along untouched.

    Report this instead of a t-test, and state the resampling unit explicitly.
    """
    rng = np.random.default_rng(seed)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    n = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= abs(obs):
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))


def frame_count_sweep(rows, enc_a, enc_b, lengths=(8, 12, 16, 24, 32),
                      measures=("cka", "cknna", "nbr")):
    """Effect size as a function of the number of frames compared.

    THE OBJECTION THIS ANSWERS. Treating the frames of one video as the
    observations for a representation-similarity measure is a change of usage,
    and at T_c = 16 the sample is small. A reviewer will reasonably ask
    whether the result is a genuine per-video alignment measure or an unstable
    statistic produced by too few correlated observations.

    If the effect is present at 8, 12, 16, 24 and 32 frames and the ranking of
    encoder pairs is stable across them, that objection is closed empirically
    rather than argued away.

    Cheap: no re-extraction, the features are already cached at T = 32.
    """
    usable = [r for r in rows
              if enc_a in r.get("done", []) and enc_b in r.get("done", [])]
    out = []
    for t in lengths:
        recs = []
        for r in usable:
            A, B = load_features(r["vid"], enc_a), load_features(r["vid"], enc_b)
            if A is None or B is None:
                continue
            rec = {"label": r["label"]}
            for m in measures:
                try:
                    if m == "cka":
                        rec[m] = ms.cka(A, B, t=t)
                    elif m == "cknna":
                        rec[m] = ms.cknna(A, B, k=min(C.CKNNA_K, t // 4), t=t)
                    elif m == "nbr":
                        rec[m] = ms.neighbourhood(A, B, t=t)
                except Exception:
                    rec[m] = np.nan
            recs.append(rec)
        df = pd.DataFrame(recs)
        if df.empty:
            continue
        for m in measures:
            if m not in df:
                continue
            xf = df[df.label == 0][m].to_numpy(float)
            yr = df[df.label == 1][m].to_numpy(float)
            lo, hi = bootstrap_ci(cohens_d, xf, yr)
            out.append({"T_c": t, "measure": m, "pair": f"{enc_a}|{enc_b}",
                        "n": len(df), "cohens_d": cohens_d(xf, yr),
                        "d_lo": lo, "d_hi": hi, "auroc": auroc(xf, yr)})
    return pd.DataFrame(out)


def motion_regression(raw, measure=None):
    """Is the class difference explained by how much the video moves?

    Fits  D(V) = b0 + b1*I[generated] + b2*c(V) + e  by ordinary least
    squares, and reports b1 with a bootstrap interval.

    Stronger than tercile stratification alone: it asks whether the class
    coefficient survives when the movement covariate is in the model, rather
    than only whether the effect persists inside coarse bins.
    """
    measure = measure or C.PRIMARY_MEASURE
    out = []
    for pair, g in raw.groupby("pair"):
        d = g[[measure, "covariate", "label"]].dropna()
        if len(d) < 50:
            continue
        X = np.column_stack([np.ones(len(d)),
                             (d.label == 0).astype(float),     # generated = 1
                             d.covariate.to_numpy(float)])
        y = d[measure].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        rng = np.random.default_rng(C.SEED)
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d))
            b, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
            boot.append(b[1])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"pair": pair, "measure": measure, "n": len(d),
                    "beta_generated": float(beta[1]), "b1_lo": float(lo),
                    "b1_hi": float(hi), "beta_covariate": float(beta[2]),
                    "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(out)


def paired_auroc_test(scores_a, scores_b, labels, n_boot=2000, seed=C.SEED):
    """Paired bootstrap on the AUROC difference between two statistics.

    WHY PAIRED. H5 asks whether between-encoder statistics add information
    beyond within-encoder ones. Observing d_cross > d_within on the same
    videos is not a test: the two are computed on the SAME sample, so their
    difference has far less variance than two independent estimates would,
    and comparing their separate confidence intervals is the wrong
    comparison -- overlapping intervals do not imply no difference.

    Resampling videos and recomputing BOTH AUROCs on each resample keeps the
    pairing and gives an interval on the difference itself.

    Returns (auroc_a, auroc_b, difference, lo, hi).
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(scores_a) & np.isfinite(scores_b) & np.isfinite(labels)
    a, b, y = scores_a[ok], scores_b[ok], labels[ok]
    pos, neg = y == 0, y == 1                      # generated = positive

    def _auc(s):
        return auroc(s[pos], s[neg])

    obs_a, obs_b = _auc(a), _auc(b)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n_boot):
        i = rng.choice(idx, len(idx), replace=True)
        ya = y[i]
        p_, n_ = ya == 0, ya == 1
        if p_.sum() < 2 or n_.sum() < 2:
            continue
        diffs.append(auroc(a[i][p_], a[i][n_]) - auroc(b[i][p_], b[i][n_]))
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    return float(obs_a), float(obs_b), float(obs_a - obs_b), float(lo), float(hi)


def h5_complementarity(raw, cross=None,
                       within=("norm", "second_order", "curvature", "stall", "split")):
    """H5, stated as complementarity rather than superiority.

    REVISED. The earlier version asked whether a cross-encoder measure beats
    the strongest within-encoder statistic. That framing has two problems.
    It is a comparative detection claim, which this paper explicitly does not
    make; and the within-encoder statistics available here are weak against
    current generators -- one reaches under 1% recall at a strict operating
    point on a recent benchmark -- so beating them would establish little.

    The question instead: does the cross-encoder measure carry information the
    within-encoder one does not? Answered by (a) the correlation between the
    two scores on the same videos, and (b) whether their combination separates
    the classes better than either alone.

    This is robust to losing. Decorrelated scores are the interesting result
    whether or not ours is the larger. It also follows the logic of the
    methods it compares against, which select their own components by which
    combination gives the lowest cross-domain correlation.

    Include a contemporary single-encoder score in `within` when available --
    only the per-video score is needed, not a reimplemented detection
    pipeline.
    """
    cross = cross or ["cka", "cknna", "nbr", "ridge"]
    out = []
    for pair, g in raw.groupby("pair"):
        avail = [w for w in within if w in g.columns]
        if not avail:
            continue
        y = g.label.to_numpy(float)
        best_w = max(avail, key=lambda w: abs(
            (auroc(g[w].to_numpy(float)[y == 0], g[w].to_numpy(float)[y == 1]) or 0.5) - 0.5))
        for c in cross:
            if c not in g.columns:
                continue
            xc = g[c].to_numpy(float)
            xw = g[best_w].to_numpy(float)
            ac, aw, diff, lo, hi = paired_auroc_test(xc, xw, y)
            # (a) correlation of the two scores on the same videos
            from scipy.stats import spearmanr
            ok = np.isfinite(xc) & np.isfinite(xw)
            rho = float(spearmanr(xc[ok], xw[ok])[0]) if ok.sum() > 2 else np.nan
            # (b) combination: average of real-referenced percentiles, oriented
            # so that larger = more generated-like for both scores.
            # Labels: 1 = real, 0 = generated (manifest convention).
            def _oriented_perc(v):
                a = auroc(v[ok][y[ok] == 0], v[ok][y[ok] == 1]) or 0.5
                v = v if a >= 0.5 else -v
                ref = np.sort(v[ok][y[ok] == 1])        # label 1 = real
                return np.searchsorted(ref, v) / max(ref.size, 1)
            comb = 0.5 * (_oriented_perc(xc) + _oriented_perc(xw))
            acomb, _, dcomb, clo, chi = paired_auroc_test(comb, xc, y)
            out.append({"pair": pair, "cross_measure": c,
                        "best_within": best_w, "auroc_cross": ac,
                        "auroc_within": aw, "difference": diff,
                        "diff_lo": lo, "diff_hi": hi,
                        "incremental": bool(lo > 0),
                        "spearman_cross_within": rho,
                        "auroc_combined": acomb,
                        "combined_minus_cross": dcomb,
                        "comb_lo": clo, "comb_hi": chi})
    return pd.DataFrame(out)


def leave_one_generator_out(rows, enc_a, enc_b, measure="ridge"):
    """Hold out one generator entirely, including from ridge calibration.

    SCOPE. For CKA, CKNNA and neighbourhood agreement there is nothing to
    hold out -- no quantity is estimated from other videos, so a
    leave-one-generator-out protocol is identical to the per-generator
    breakdown already reported. It is only meaningful for the ridge map,
    which is the one fitted component.

    Even there the calibration uses real video only, so a generator is never
    "seen" in the ordinary sense. What this tests is subtler: whether the
    real-video pool that happens to accompany a generator in a benchmark
    influences that generator's score.
    """
    gens = sorted({r["generator"] for r in rows if r["label"] == 0})
    out = []
    for held in gens:
        fit_pool = [r for r in rows if r["label"] == 1
                    and r.get("paired_generator") != held]
        if len(fit_pool) < 30:
            continue
        pairs = []
        for r in fit_pool[:150]:
            A, B = load_features(r["vid"], enc_a), load_features(r["vid"], enc_b)
            if A is not None and B is not None:
                pairs.append((A, B))
        if not pairs:
            continue
        ridge = ms.RidgeReference(direction=C.RIDGE_DIRECTION).fit(pairs)

        fake = [r for r in rows if r["generator"] == held]
        real = [r for r in rows if r["label"] == 1][:400]
        sf, sr = [], []
        for bucket, store in ((fake, sf), (real, sr)):
            for r in bucket:
                A, B = load_features(r["vid"], enc_a), load_features(r["vid"], enc_b)
                if A is not None and B is not None:
                    try:
                        store.append(ridge.score(A, B))
                    except Exception:
                        pass
        x, y = np.array(sf), np.array(sr)
        if len(x) < 10 or len(y) < 10:
            continue
        lo, hi = bootstrap_ci(cohens_d, x, y)
        out.append({"held_out_generator": held, "measure": measure,
                    "n_generated": len(x), "n_real": len(y),
                    "cohens_d": cohens_d(x, y), "d_lo": lo, "d_hi": hi,
                    "auroc": auroc(x, y)})
    return pd.DataFrame(out)


# ==========================================================================
# PROVENANCE CONTROLS
#
# GVD pairs generated video collected from Discord against real video from
# YouTube-VOS and GOT-10k. Those corpora differ in resolution, codec, frame
# rate, duration, compression history and content. A measured real/generated
# discrepancy could reflect any of that rather than synthesis.
#
# Re-encoding to a common pipeline equalises the ENCODING but cannot undo the
# original compression history, and does nothing about content. So the
# controls below approach the problem from three other directions.
# ==========================================================================

def real_vs_real(rows, enc_a, enc_b, measures=("cka", "cknna", "nbr")):
    """Measure discrepancy between two corpora of REAL video.

    THE DECISIVE CONTROL, and almost free.

    Corpora are identified by folder, not by name. FakeParts' real videos are
    confirmed (by direct download) to sit in five separate folders drawn from
    five named sources, but the folders themselves carry no label matching
    them to those names individually. That does not matter here: the control
    only needs the pools to be DIFFERENT from each other, not identified,
    since any two distinct real corpora give a valid provenance bound
    regardless of which specific dataset each one is.

    With five folders this runs all C(5,2)=10 pairwise comparisons
    automatically (the loop below is already general). Report the spread
    (min/mean/max), not a single pair -- see summarize_real_vs_real below.

    Before trusting this, check the folder sizes are not wildly imbalanced:
    the five source datasets FakeParts draws on (DAVIS 2016/2017, YouTube-VOS
    2019, MOSE, LVD-2M) differ hugely in scale in the wild, and a folder with
    only a few dozen videos will produce a noisy, low-power comparison.

    Both populations are real, so any separation is pure provenance: corpus,
    codec, resolution, capture characteristics, content. It gives a direct
    upper bound on how much of the real/generated effect could be
    attributable to provenance rather than synthesis.

    Read it against the main result:

        d(corpus A vs corpus B)  <<  d(real vs generated)
            -> provenance accounts for a small part of the effect

        d(corpus A vs corpus B)  ~=  d(real vs generated)
            -> the measure is largely tracking corpus identity, and the
               benchmark cannot support the paper's claim

    The second outcome is a finding about the benchmark, not a failed study,
    and it should be reported as one.
    """
    real = [r for r in rows if r["label"] == 1
            and enc_a in r.get("done", []) and enc_b in r.get("done", [])]
    corpora = sorted({r["generator"] for r in real})     # "real:got10k" etc.
    if len(corpora) < 2:
        print("  [skip] real_vs_real needs two or more real corpora")
        return pd.DataFrame()

    scored = []
    for r in real:
        A, B = load_features(r["vid"], enc_a), load_features(r["vid"], enc_b)
        if A is None or B is None:
            continue
        rec = {"corpus": r["generator"], "vid": r["vid"]}
        for m in measures:
            fn = {"cka": ms.cka, "cknna": ms.cknna, "nbr": ms.neighbourhood}[m]
            try:
                rec[m] = fn(A, B)
            except Exception:
                rec[m] = np.nan
        scored.append(rec)
    df = pd.DataFrame(scored)

    out = []
    for i, ca in enumerate(corpora):
        for cb in corpora[i + 1:]:
            for m in measures:
                x = df[df.corpus == ca][m].to_numpy(float)
                y = df[df.corpus == cb][m].to_numpy(float)
                if min(len(x), len(y)) < 20:
                    continue
                lo, hi = bootstrap_ci(cohens_d, x, y)
                out.append({"corpus_a": ca, "corpus_b": cb, "measure": m,
                            "n_a": len(x), "n_b": len(y),
                            "cohens_d": cohens_d(x, y), "d_lo": lo, "d_hi": hi,
                            "auroc": auroc(x, y)})
    return pd.DataFrame(out)


def summarize_real_vs_real(rvr: pd.DataFrame) -> pd.DataFrame:
    """Collapse the pairwise real_vs_real output to min/mean/max per measure.

    This is what belongs in the manuscript table, not the full 10-row grid --
    the spread across folder pairs is the informative quantity, and reporting
    only one pair (as a two-corpus benchmark like GVD would force) hides
    whether that pair happened to be the best or worst case.
    """
    if rvr.empty:
        return pd.DataFrame()
    out = []
    for m, g in rvr.groupby("measure"):
        out.append({"measure": m, "n_pairs": len(g),
                    "d_min": g.cohens_d.min(), "d_mean": g.cohens_d.mean(),
                    "d_max": g.cohens_d.max(),
                    "worst_pair": f"{g.loc[g.cohens_d.idxmax(),'corpus_a']} vs "
                                  f"{g.loc[g.cohens_d.idxmax(),'corpus_b']}"})
    return pd.DataFrame(out)


def generator_vs_generator(raw, measures=("cka", "cknna", "nbr")):
    """Discrepancy between two GENERATORS, provenance held fixed.

    The complement of real_vs_real. Both populations are generated, collected
    the same way, so acquisition is matched by construction. If the measure
    still varies systematically across generators, it responds to something
    about the generation process rather than to how the videos were obtained.

    Neither control alone settles the question. Together they bracket it:
    real_vs_real bounds the provenance contribution from above,
    generator_vs_generator shows the measure is generation-sensitive with
    provenance controlled.
    """
    gen = raw[raw.label == 0]
    gens = sorted(gen.generator.unique())
    out = []
    for i, ga in enumerate(gens):
        for gb in gens[i + 1:]:
            for m in measures:
                if m not in gen.columns:
                    continue
                x = gen[gen.generator == ga][m].to_numpy(float)
                y = gen[gen.generator == gb][m].to_numpy(float)
                if min(len(x), len(y)) < 20:
                    continue
                out.append({"gen_a": ga, "gen_b": gb, "measure": m,
                            "n_a": len(x), "n_b": len(y),
                            "cohens_d": cohens_d(x, y), "auroc": auroc(x, y)})
    return pd.DataFrame(out)


def provenance_regression(raw, audit_csv="results/dataset_audit.csv",
                          measure=None):
    """Does the class effect survive with encoding covariates in the model?

    Extends the motion regression with provenance terms:

        D(V) = b0 + b1*generated + b2*motion + b3*width + b4*fps
               + b5*duration + e

    If b1 collapses once encoding parameters are included, the measured
    separation is largely explained by how the videos were encoded.

    Covariates come from src/dataset_audit.py, which probes each group with
    ffprobe. Group-level values are a coarse proxy for per-video ones; if the
    result is marginal, probe every video rather than a sample.
    """
    path = pathlib.Path(audit_csv)
    if not path.exists():
        print(f"  [skip] {audit_csv} not found -- run src.dataset_audit first")
        return pd.DataFrame()
    measure = measure or C.PRIMARY_MEASURE
    audit = pd.read_csv(path)
    audit["w"] = audit.resolution.str.split("x").str[0]
    audit["w"] = pd.to_numeric(audit["w"], errors="coerce")
    audit["fps_n"] = pd.to_numeric(audit.fps, errors="coerce")
    key = audit.set_index("group")[["w", "fps_n", "mean_dur_s"]].to_dict("index")

    out = []
    for pair, g in raw.groupby("pair"):
        d = g.copy()
        d["w"] = d.generator.map(lambda x: key.get(x, {}).get("w", np.nan))
        d["fps"] = d.generator.map(lambda x: key.get(x, {}).get("fps_n", np.nan))
        d["dur"] = d.generator.map(lambda x: key.get(x, {}).get("mean_dur_s", np.nan))
        cols = [measure, "covariate", "w", "fps", "dur", "label"]
        d = d[[c for c in cols if c in d.columns]].dropna()
        if len(d) < 80:
            continue

        terms = [c for c in ("covariate", "w", "fps", "dur") if c in d.columns]
        X = np.column_stack([np.ones(len(d)), (d.label == 0).astype(float)]
                            + [d[c].to_numpy(float) for c in terms])
        y = d[measure].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        rng = np.random.default_rng(C.SEED)
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d))
            b, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
            boot.append(b[1])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # same model without provenance terms, for comparison
        X0 = np.column_stack([np.ones(len(d)), (d.label == 0).astype(float)])
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)

        out.append({"pair": pair, "measure": measure, "n": len(d),
                    "beta_generated_raw": float(b0[1]),
                    "beta_generated_adjusted": float(beta[1]),
                    "b1_lo": float(lo), "b1_hi": float(hi),
                    "attenuation": float(1 - beta[1] / b0[1]) if b0[1] else np.nan,
                    "survives": bool(lo > 0 or hi < 0),
                    "covariates": ",".join(terms)})
    return pd.DataFrame(out)


def cmd_provenance(args):
    """All provenance controls in one pass."""
    rows = M.load()
    a, b = args.pair

    print("1. real vs real -- upper bound on the provenance contribution")
    rr = real_vs_real(rows, a, b)
    if not rr.empty:
        write(rr, "provenance_real_vs_real")
        print(rr.to_string(index=False, float_format="%.4f"))

    raw = latest("grid_raw")
    if raw is not None:
        print("\n2. generator vs generator -- provenance matched by construction")
        gg = generator_vs_generator(raw)
        if not gg.empty:
            write(gg, "provenance_gen_vs_gen")
            print(gg.head(20).to_string(index=False, float_format="%.4f"))

        print("\n3. regression with encoding covariates")
        pr = provenance_regression(raw)
        if not pr.empty:
            write(pr, "provenance_regression")
            print(pr.to_string(index=False, float_format="%.4f"))
            print("\nattenuation is the fraction of the class effect that")
            print("disappears once encoding parameters are in the model.")

    if not rr.empty and raw is not None:
        main = summarise(raw, [C.PRIMARY_MEASURE])
        if not main.empty:
            dm = main.cohens_d.abs().max()
            dp = rr.cohens_d.abs().max()
            print("\n" + "=" * 62)
            print(f"real vs generated : d = {dm:.3f}")
            print(f"real vs real      : d = {dp:.3f}   (pure provenance)")
            print(f"ratio             : {dp/dm:.2f}")
            if dp / dm > 0.6:
                print("\n-> Provenance accounts for most of the effect on this")
                print("   benchmark. Report it as a finding about the benchmark")
                print("   and lead with the source-aligned one.")
            elif dp / dm > 0.3:
                print("\n-> Provenance accounts for a substantial minority.")
                print("   Report both, and interpret the main result with the")
                print("   bound stated explicitly.")
            else:
                print("\n-> Provenance accounts for a small part. The main")
                print("   result is not reducible to corpus identity.")
            print("=" * 62)


def additive_baseline(raw, enc_a_stat="norm", enc_b_stat="norm",
                      relational=None):
    """Two encoders scored independently and combined, with no relational term.

    THE CONTROL THAT MOST DIRECTLY ISOLATES THE CLAIM.

    STALL (CVPR 2026) extends to two encoders trivially: compute its whitening
    statistics for each encoder separately, score a video under both, average
    the percentiles. That uses two independently pretrained encoders and
    measures no relationship between them at all.

    If that construction performs as well as D(E1, E2), then the contribution
    is HAVING TWO ENCODERS, not measuring their relationship, and the paper
    collapses into an ensembling result. So the comparison is:

        s(E1) + s(E2)        two encoders, no relationship
        D(E1, E2)            the relationship

    This is a stronger control than the random projection, which tests whether
    the shared projection matters rather than whether the relationship does.

    Together with the correspondence shuffle it closes the loop: the shuffle
    shows the statistic depends on PAIRING, this shows pairing beats NOT
    pairing. Neither alone suffices, and no within-encoder method can run
    either.

    Percentile-combine per STALL rather than summing raw scores, since the
    two per-encoder statistics need not share a scale.
    """
    relational = relational or C.PRIMARY_MEASURE
    out = []
    for pair, g in raw.groupby("pair"):
        need = {relational, enc_a_stat}
        if not need.issubset(g.columns):
            continue
        y = g.label.to_numpy(float)

        # per-encoder statistics, percentile-ranked against the real class
        def _perc(v):
            v = np.asarray(v, float)
            ref = np.sort(v[(y == 1) & np.isfinite(v)])
            if ref.size < 10:
                return np.full_like(v, np.nan)
            return np.searchsorted(ref, v) / ref.size

        pa = _perc(g[enc_a_stat].to_numpy(float))
        pb = (_perc(g[enc_b_stat].to_numpy(float))
              if enc_b_stat in g.columns else pa)
        additive = 0.5 * (pa + pb)
        rel = g[relational].to_numpy(float)

        a_add, a_rel, diff, lo, hi = paired_auroc_test(rel, additive, y)
        out.append({"pair": pair, "relational": relational,
                    "auroc_relational": a_add, "auroc_additive": a_rel,
                    "difference": diff, "diff_lo": lo, "diff_hi": hi,
                    "relationship_adds": bool(lo > 0),
                    "n": int(np.isfinite(rel).sum())})
    return pd.DataFrame(out)


def recall_at_fpr(x_fake, y_real, alphas=(0.01, 0.05)):
    """Fake recall at a false-positive rate fixed on the real class.

    WHY, given that this paper is not deployment-oriented. Cohen's d and AUROC
    are aggregates and hide the shape of the distribution. If cross-encoder
    disagreement has a heavy tail -- a subset of generated videos that are
    extremely discrepant while the rest overlap -- that is forensically
    interesting and invisible in both. Recall at a fixed FPR shows it.

    We report 1% and 5% and omit 0.1%. At the sample sizes here the 0.1%
    threshold sits in the extreme tail of the real distribution, where the
    estimate is determined by a handful of videos and is not stable. Methods
    engineered for deployment report it because that is their operating point;
    for a measurement study it would be noise reported to three figures.

    Assumes larger = more likely generated. If the measure runs the other way,
    the caller should negate before passing.
    """
    x = np.asarray(x_fake, float); y = np.asarray(y_real, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    out = {}
    for a in alphas:
        if len(y) < 1 / a:            # fewer real videos than the tail needs
            out[f"recall@fpr{a:.0%}"] = np.nan
            continue
        thr = np.quantile(y, 1 - a)
        out[f"recall@fpr{a:.0%}"] = float((x >= thr).mean())
    return out


def linear_probe_baseline(raw, measure_cols, train_generator=None,
                          seed=C.SEED):
    """Supervised reference: a linear probe on frozen features, cross-generator.

    A supervised row is useful for one purpose -- showing how much of the
    achievable signal a classifier-free measure captures -- and it does not
    require reproducing a large trained detector. A single linear layer on the
    features already extracted, trained on ONE generator and tested on the
    others, is the protocol used in prior work and costs minutes.

    The cross-generator split is what makes the comparison legitimate. Every
    training-free paper that compares against supervised detectors evaluates
    them on generators they did not see: D3 trains baselines on Pika and tests
    on ten other generators; SPLIT trains on GenVideo and tests on FakeParts
    and ViF-Bench. A supervised baseline trained and tested on the same
    generator would be uninformative, and its number should never be reported
    without stating which generators it saw.

    Reads the per-video measure columns rather than raw features, so it is a
    probe on the discrepancy space -- a fair upper reference for what a
    classifier could extract from the same quantities.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gens = sorted(raw[raw.label == 0].generator.unique())
    if not gens:
        return pd.DataFrame()
    train_generator = train_generator or gens[0]

    cols = [c for c in measure_cols if c in raw.columns]
    d = raw[cols + ["label", "generator"]].dropna()
    if len(d) < 100:
        return pd.DataFrame()

    tr = d[(d.generator == train_generator) | (d.label == 1)]
    real_idx = tr[tr.label == 1].sample(frac=0.5, random_state=seed).index
    tr = tr.loc[tr.index.isin(real_idx) | (tr.label == 0)]

    X = StandardScaler().fit(tr[cols])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
        X.transform(tr[cols]), (tr.label == 0).astype(int))

    out = []
    for g in gens:
        if g == train_generator:
            continue
        te = d[(d.generator == g) | ((d.label == 1) & (~d.index.isin(real_idx)))]
        if te.label.nunique() < 2:
            continue
        p = clf.predict_proba(X.transform(te[cols]))[:, 1]
        yv = (te.label == 0).to_numpy()
        out.append({"trained_on": train_generator, "tested_on": g,
                    "n": len(te), "auroc": auroc(p[yv], p[~yv]),
                    "regime": "supervised, cross-generator"})
    return pd.DataFrame(out)


def write(df, name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = C.RESULTS_DIR / f"{name}_{ts}.csv"
    df.to_csv(p, index=False)
    print(f"  -> {p}")
    return p


# ==========================================================================
# Commands
# ==========================================================================
def cmd_pilot(args):
    """The go/no-go gate. CLIP vs MiDaS, cka and nbr, whatever is cached."""
    rows = M.load()
    a, b = args.pair
    raw = score_pair(rows, a, b, ["cka", "cknna", "nbr", "randproj", "norm"])
    if raw.empty:
        raise SystemExit(f"not enough cached videos with both {a} and {b}")

    summ = summarise(raw, ["cka", "cknna", "nbr", "randproj", "norm"])
    write(raw, "pilot_raw")
    write(summ, "pilot_summary")

    print("\n" + "=" * 74)
    print(f"PILOT: {a}  vs  {b}")
    print("=" * 74)
    print(summ[["dataset", "measure", "n_real", "n_fake",
                "mean_real", "mean_fake", "cohens_d", "d_lo", "d_hi",
                "auroc"]].to_string(index=False, float_format="%.4f"))

    best = summ.cohens_d.abs().max()
    print("\n" + "-" * 74)
    print(f"largest |Cohen's d| across measures: {best:.3f}")
    if best > 0.4:
        print("  -> POSITIVE. Build the full grid. Aim at CVIU.")
    elif best > 0.15:
        print("  -> PARTIAL. Still a paper; foreground the encoder-pair")
        print("     comparison, since which pairs work becomes the question.")
    else:
        print("  -> NEGATIVE. Cross-encoder discrepancy is largely a product")
        print("     of discriminative training. Reframe as a null result;")
        print("     consider a shorter venue. Paper 1 is not exposed.")
    print("-" * 74)


def cmd_grid(args):
    rows = M.load()
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]

    have = {e for r in rows for e in r.get("done", [])}
    pairs = [(a, b) for a, b in C.cross_role_pairs() if a in have and b in have]
    ctrl = [(a, b) for a, b in C.same_role_pairs() if a in have and b in have]
    print(f"{len(pairs)} cross-role pairs, {len(ctrl)} same-role control pairs")

    frames = []
    for a, b in pairs + ctrl:
        print(f"  scoring {a} | {b}")
        d = score_pair(rows, a, b, args.measures)
        if not d.empty:
            d["control"] = (a, b) in ctrl
            frames.append(d)
    if not frames:
        raise SystemExit("nothing scored -- check the cache")

    raw = pd.concat(frames, ignore_index=True)
    summ = summarise(raw, args.measures)
    summ["control"] = summ.pair.map(
        {f"{a}|{b}": True for a, b in ctrl}).fillna(False)

    write(raw, "grid_raw")
    write(summ, "grid_summary")

    report_primary(summ)                                    # FIX 4

    add = additive_baseline(raw)                            # the isolating control
    if not add.empty:
        write(add, "additive_baseline")
        print("\nAdditive baseline: two encoders scored separately vs their relationship")
        print(add.to_string(index=False, float_format="%.4f"))
        print("If relationship_adds is False, the effect is ensembling, not")
        print("relational, and the paper's claim has to be reframed.")

    h5 = h5_complementarity(raw)                            # H5: complementarity
    if not h5.empty:
        write(h5, "h5_incremental")
        print("\nH5: complementarity with the best within-encoder statistic")
        print(h5.to_string(index=False, float_format="%.4f"))

    cov = covariate_correlation(raw, args.measures)         # FIX 3
    strat = stratified(raw, args.measures)
    write(cov, "covariate_correlation")
    write(strat, "stratified")
    print("\ncovariate correlation (measure vs temporal variability):")
    print(cov.to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 88)
    print("CROSS-ROLE PAIRS (the result)")
    print("=" * 88)
    print(summ[~summ.control][["pair", "dataset", "measure", "cohens_d",
                               "d_lo", "d_hi", "auroc"]]
          .to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 88)
    print("SAME-ROLE PAIRS (control: should be weaker if priors matter)")
    print("=" * 88)
    print(summ[summ.control][["pair", "dataset", "measure", "cohens_d", "auroc"]]
          .to_string(index=False, float_format="%.4f"))

    best = summ[~summ.control].sort_values("auroc_dist", ascending=False).iloc[0]
    pg = per_generator(raw[raw.pair == best.pair], best.measure)
    write(pg, "per_generator")
    print(f"\nper-generator breakdown for best pair ({best.pair}, {best.measure}):")
    print(pg.to_string(index=False, float_format="%.4f"))

    # Table 6 -- FakeParts per-category breakdown. Uses the SAME primary
    # pair fixed a priori (CLIP--MiDaS, debiased CKA per config.PRIMARY_PAIR
    # and config.PRIMARY_MEASURE), not the best-scoring pair above -- Table 6
    # is a breakdown of the primary comparison, not a second search over the
    # grid. Runs only if FakeParts rows are present and carry a 'category'
    # column; otherwise prints what is missing rather than failing silently.
    fp = raw[raw.dataset == "fakeparts"]
    if fp.empty:
        print("\n[Table 6] no FakeParts rows in this run -- nothing to do.")
    elif "category" not in fp.columns:
        print("\n[Table 6] FakeParts rows present but have no 'category' "
              "column. Add it to the per-video row dict at extraction time "
              "(T2V, TI2V, Interpolation, Extrapolation, Style_change, "
              "Outpainting, Inpainting, FaceSwap) and re-run -- this does "
              "NOT require re-extracting features, only relabelling the "
              "existing cached rows.")
    else:
        primary_key = f"{C.PRIMARY_PAIR[0]}|{C.PRIMARY_PAIR[1]}"
        pc = per_category(raw, C.PRIMARY_MEASURE, dataset="fakeparts")
        pc = pc[pc.pair == primary_key]
        if pc.empty:
            print(f"\n[Table 6] 'category' column present but no rows match "
                  f"the primary pair {primary_key} -- check pair naming.")
        else:
            write(pc, "per_category")
            print(f"\nTable 6 -- FakeParts by category "
                  f"({primary_key}, {C.PRIMARY_MEASURE}):")
            print(pc[["category", "n", "cohens_d", "d_lo", "d_hi", "auroc"]]
                  .to_string(index=False, float_format="%.4f"))
            print("\nPre-registered prediction (write this down BEFORE reading")
            print("the numbers above, not after): T2V/TI2V/Interpolation/")
            print("Extrapolation/Style_change should separate strongly;")
            print("Inpainting/FaceSwap weakly; Outpainting intermediate.")


def cmd_intervention(args):
    """Correspondence intervention, classifier-free.

    Pairs encoder A's features from video i with encoder B's from video j.
    All parameters fixed; every marginal distribution preserved; only the
    pairing changes.

    REVISED after external review. The earlier version drew one shuffle and
    compared means, then concluded the measures track correspondence. That
    inference was too quick. Two changes:

      1. REPEATED shuffles (default 20), so the shuffled condition has a
         distribution rather than a single draw, and the interval reflects
         shuffle variability rather than only sampling variability.

      2. An explicit test of the asymmetry. The interesting claim is not that
         shuffling raises discrepancy -- it is that it raises it MORE for real
         video, because real video has more cross-encoder correspondence to
         destroy. That is

             Delta_real = D_shuffled,real - D_matched,real
             Delta_fake = D_shuffled,fake - D_matched,fake

         and the prediction is Delta_real > Delta_fake. We test the
         difference of differences directly, with a bootstrap interval.
    """
    rows = M.load()
    a, b = args.pair
    usable = [r for r in rows
              if a in r.get("done", []) and b in r.get("done", [])]
    rng = np.random.default_rng(C.SEED)
    measures = ["cka", "cknna", "nbr"]

    recs = []
    for label in (1, 0):
        sub = [r for r in usable if r["label"] == label][:args.n]
        feats = [(load_features(r["vid"], a), load_features(r["vid"], b))
                 for r in sub]
        feats = [(A, B) for A, B in feats if A is not None and B is not None]

        for i, (A, B) in enumerate(feats):
            base = {"label": label, "video": i}
            for m in measures:
                fn = {"cka": ms.cka, "cknna": ms.cknna,
                      "nbr": ms.neighbourhood}[m]
                try:
                    recs.append({**base, "condition": "matched",
                                 "rep": 0, "measure": m, "value": fn(A, B)})
                except Exception:
                    pass
            # repeated shuffles per video
            for rep in range(args.reps):
                j = rng.integers(len(feats))
                while j == i and len(feats) > 1:
                    j = rng.integers(len(feats))
                Bj = feats[j][1]
                for m in measures:
                    fn = {"cka": ms.cka, "cknna": ms.cknna,
                          "nbr": ms.neighbourhood}[m]
                    try:
                        recs.append({**base, "condition": "shuffled",
                                     "rep": rep, "measure": m,
                                     "value": fn(A, Bj)})
                    except Exception:
                        pass

    df = pd.DataFrame(recs)
    write(df, "intervention_raw")

    # Difference of differences, bootstrapped OVER VIDEOS.
    #
    # BUG FIXED HERE. The first version resampled rows of the long-format
    # frame, which contains 20 shuffle repetitions per video. Resampling
    # those as if they were independent observations inflates the effective
    # sample size roughly twentyfold and produces a confidence interval far
    # narrower than the data supports.
    #
    # The 20 shuffles are repeated perturbations of one video, not 20
    # experimental units. The independent unit is the video. So: average the
    # shuffles within each video to obtain one Delta per video, then
    # bootstrap over videos.
    summ = []
    for m in measures:
        d = df[df.measure == m]
        if d.empty:
            continue

        # one Delta per video: mean over its 20 shuffles, minus its matched value
        per_video = (d.pivot_table(index=["label", "video"], columns="condition",
                                   values="value", aggfunc="mean")
                       .dropna())
        if not {"matched", "shuffled"}.issubset(per_video.columns):
            continue
        per_video["delta"] = per_video["shuffled"] - per_video["matched"]

        dreal = per_video.xs(1, level="label")["delta"].to_numpy()
        dgen = per_video.xs(0, level="label")["delta"].to_numpy()
        if len(dreal) < 5 or len(dgen) < 5:
            continue

        rng_b = np.random.default_rng(C.SEED)
        boot = [rng_b.choice(dreal, len(dreal), replace=True).mean()
                - rng_b.choice(dgen, len(dgen), replace=True).mean()
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        summ.append({"measure": m,
                     "n_real_videos": len(dreal), "n_gen_videos": len(dgen),
                     "shuffles_per_video": args.reps,
                     "delta_real": float(dreal.mean()),
                     "delta_generated": float(dgen.mean()),
                     "difference": float(dreal.mean() - dgen.mean()),
                     "diff_lo": float(lo), "diff_hi": float(hi),
                     "asymmetry_supported": bool(lo > 0)})

    out = pd.DataFrame(summ)
    write(out, "intervention_summary")

    print("\nCorrespondence intervention, classifier-free")
    print(df.groupby(["label", "condition", "measure"]).value
          .agg(["mean", "std", "count"]).to_string(float_format="%.4f"))
    print("\nAsymmetry test: is shuffling more damaging to REAL video?")
    print("(prediction: difference > 0, interval excluding zero)")
    print(out.to_string(index=False, float_format="%.4f"))


def cmd_robustness(args):
    """Frame-count sweep and motion regression: the two confound controls."""
    rows = M.load()
    a, b = args.pair

    print("frame-count sensitivity (does the effect survive fewer/more frames?)")
    sweep = frame_count_sweep(rows, a, b)
    if not sweep.empty:
        write(sweep, "frame_count_sweep")
        print(sweep.to_string(index=False, float_format="%.4f"))

    raw = latest("grid_raw")
    if raw is not None and "covariate" in raw.columns:
        print("\nmotion regression: D(V) = b0 + b1*generated + b2*c(V)")
        reg = motion_regression(raw)
        if not reg.empty:
            write(reg, "motion_regression")
            print(reg.to_string(index=False, float_format="%.4f"))
            print("\nb1 is the class effect AFTER controlling for movement.")
            print("If excludes_zero is False, movement explains the difference.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pilot")
    p.add_argument("--pair", nargs=2, default=["clip_b32", "midas_hybrid"])
    p.set_defaults(fn=cmd_pilot)

    g = sub.add_parser("grid")
    g.add_argument("--dataset", default=None)
    g.add_argument("--measures", nargs="+",
                   default=["cka", "cknna", "nbr", "ridge", "randproj", "norm"])
    g.set_defaults(fn=cmd_grid)

    i = sub.add_parser("intervention")
    i.add_argument("--pair", nargs=2, default=["clip_b32", "midas_hybrid"])
    i.add_argument("--n", type=int, default=400)
    i.add_argument("--reps", type=int, default=20,
                   help="shuffles per video; one draw is not enough")
    i.set_defaults(fn=cmd_intervention)

    pv = sub.add_parser("provenance")
    pv.add_argument("--pair", nargs=2, default=["clip_b32", "midas_hybrid"])
    pv.set_defaults(fn=cmd_provenance)

    r = sub.add_parser("robustness")
    r.add_argument("--pair", nargs=2, default=["clip_b32", "midas_hybrid"])
    r.set_defaults(fn=cmd_robustness)

    a = ap.parse_args()
    a.fn(a)
