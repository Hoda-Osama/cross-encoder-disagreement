"""Training-free cross-encoder discrepancy measures.

Nothing here touches a GPU and nothing is trained. Every measure is a
closed-form function of two cached feature matrices.

Features are cached fp16 to save disk; everything here casts to float64
first, because CKA takes Frobenius norms of Gram matrices and the ridge map
is a linear solve, and neither is safe in half precision.

--------------------------------------------------------------------------
DESIGN FIXES APPLIED (see PAPER2_ARCHITECTURE_REVIEW.md)
--------------------------------------------------------------------------
FIX 2  Common sequence length. `align_time` previously resampled the longer
       sequence down to the shorter, which meant appearance-geometry pairs
       were measured at T=32 (496 frame pairs) while appearance-motion pairs
       were measured at T=16 (120). Any comparison between role-pairs was
       then partly a comparison between sample sizes, which is exactly what
       H2 rests on. All pairs are now resampled to config.COMMON_T.

FIX 5  Debiased CKA. Linear CKA is biased at small n, and n here is 16.
       `cka(..., debiased=True)` uses the unbiased HSIC estimator of
       Song et al. (2012), as recommended in Kornblith et al. (2019).
       Both are computed so the paper can show the ordering is unchanged.

FIX 6  Ridge direction. Previously A->B only, with no justification. Now
       bidirectional, with COMPLEMENTARY -> APPEARANCE as the default,
       because that is the training-free analogue of Paper 1's CMID: both
       ask how poorly complementary evidence reconstructs the appearance
       reference. See PAPER_LINKAGE.md.

--------------------------------------------------------------------------
Why `linear_residual_per_video` is not a primary measure
--------------------------------------------------------------------------
Fitting W per video by least squares is degenerate. With T=16 frames and
d1=512, W has 512 x d2 free parameters against 16 equations, so least
squares fits B exactly and the residual is ~1e-30 for every video regardless
of label. Verified on synthetic data; see Appendix A of the manuscript. The
function is retained ONLY so the degeneracy is reproducible. Never report it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from . import config as C

EPS = 1e-10


# ==========================================================================
# Preparation
# ==========================================================================
def prepare(X: np.ndarray, center: bool = True) -> np.ndarray:
    """fp16 -> float64, drop non-finite rows, optionally column-centre."""
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] == 0:
        raise ValueError("no finite rows")
    return X - X.mean(0, keepdims=True) if center else X


def _resample(X: np.ndarray, t: int) -> np.ndarray:
    """Linear resampling along the time axis to exactly t rows."""
    if X.shape[0] == t:
        return X
    idx = np.linspace(0, X.shape[0] - 1, t)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, X.shape[0] - 1)
    w = (idx - lo)[:, None]
    return X[lo] * (1 - w) + X[hi] * w


def align_time(A: np.ndarray, B: np.ndarray,
               t: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Bring both sequences to a COMMON length.

    FIX 2. Every pair is resampled to the same t regardless of the encoders'
    native granularity, so that the number of frame pairs entering
    Equations (cka) and (nbr) is identical across pairs and role-pair
    comparisons are not confounded with sample size.

    t defaults to config.COMMON_T (16), the binding constraint from
    VideoMAE's tubelet granularity. Sequences are never resampled upward
    beyond their native length, so nothing is interpolated into existence
    that was not measured: t <= min(len(A), len(B)) is enforced.
    """
    t = t or C.COMMON_T
    t = min(t, A.shape[0], B.shape[0])
    return _resample(A, t), _resample(B, t), t


# ==========================================================================
# PRIMARY 1 -- Centered Kernel Alignment
# ==========================================================================
def _gram_centered(X: np.ndarray) -> np.ndarray:
    K = X @ X.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., 2012).

    Diagonals are zeroed, which removes the O(1/n) bias that inflates linear
    CKA at small n. At n = 16 this matters.
    """
    n = K.shape[0]
    if n < 4:
        return np.nan
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    ones = np.ones(n)
    term1 = float(np.sum(K * L))
    term2 = float((ones @ K @ ones) * (ones @ L @ ones)) / ((n - 1) * (n - 2))
    term3 = 2.0 * float(ones @ K @ L @ ones) / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def cka(A: np.ndarray, B: np.ndarray, debiased: bool = True,
        t: int | None = None) -> float:
    """1 - linear CKA between two representations of the same video.

    Invariant to isotropic scaling and to orthogonal transforms of either
    representation, so it measures whether the encoders agree about the
    STRUCTURE of the video rather than whether their coordinate systems
    align.

    debiased=True uses the unbiased HSIC estimator (FIX 5). The biased form
    is retained so the paper can report both and show the ordering is
    unchanged.

    Returns a discrepancy; 0 means perfect structural agreement. The debiased
    estimator can return values slightly outside [0, 1] at small n, which is
    expected and should not be clipped.
    """
    A, B, n = align_time(prepare(A), prepare(B), t)
    if n < 4:
        return np.nan

    if not debiased:
        num = np.linalg.norm(B.T @ A, "fro") ** 2
        den = (np.linalg.norm(A.T @ A, "fro")
               * np.linalg.norm(B.T @ B, "fro"))
        return float(1.0 - num / (den + EPS))

    K, L = _gram_centered(A), _gram_centered(B)
    hkl = _hsic_unbiased(K, L)
    hkk = _hsic_unbiased(K, K)
    hll = _hsic_unbiased(L, L)
    if not np.isfinite(hkl) or hkk <= 0 or hll <= 0:
        return np.nan
    return float(1.0 - hkl / np.sqrt(hkk * hll))


# ==========================================================================
# PRIMARY 2 -- Neighbourhood agreement
# ==========================================================================
def _cosine_gram(X: np.ndarray) -> np.ndarray:
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    return X @ X.T


def neighbourhood(A: np.ndarray, B: np.ndarray, t: int | None = None) -> float:
    """1 - Spearman correlation between the two frame-similarity structures.

    Do the two encoders agree about which frames of this video resemble which
    others? Purely relational, so invariant to any monotone
    reparameterisation of either representation, and it does not require
    d1 == d2.

    Uses the STRICT upper triangle. Masking with ~eye would take both
    triangles of a symmetric matrix: rho is unchanged, but the effective
    sample size doubles and any significance computed from it would be wrong.
    """
    A, B, n = align_time(prepare(A, center=False),
                         prepare(B, center=False), t)
    if n < 4:
        return np.nan
    iu = np.triu_indices(n, k=1)
    ga, gb = _cosine_gram(A)[iu], _cosine_gram(B)[iu]
    if np.std(ga) < EPS or np.std(gb) < EPS:
        return np.nan                      # degenerate: a near-static video
    rho, _ = spearmanr(ga, gb)
    return float(1.0 - rho) if np.isfinite(rho) else np.nan


# ==========================================================================
# PRIMARY 3 -- CKNNA (mutual k-nearest-neighbour alignment)
# ==========================================================================
def cknna(A: np.ndarray, B: np.ndarray, k: int = 4,
          t: int | None = None) -> float:
    """1 - Centered Kernel Nearest-Neighbour Alignment.

    WHY THIS WAS ADDED. CKNNA is the measure Huh et al. (2024) introduce to
    test the Platonic Representation Hypothesis: a CKA restricted to pairs
    that are mutual k-nearest neighbours under BOTH representations. Since
    the paper anchors its hypothesis to PRH, using PRH's own measure removes
    the obvious question of why an ad-hoc neighbourhood statistic was chosen
    instead. It is also widely used and citable.

    THE TRANSFER THIS PAPER MAKES. In every published use, the n samples are
    a DATASET and the output is one alignment number per model pair. Here the
    n samples are the T frames of ONE VIDEO, and the output is one number per
    video. That is what turns a model-comparison tool into a per-sample
    statistic, and it is the methodological step the paper contributes.

    CAUTION. Neighbourhood-based alignment metrics are known to be sensitive
    to hubness: a single outlier can disrupt the neighbourhood structure
    globally. At T_c = 16 frames this is a live risk, so report CKNNA
    alongside CKA rather than instead of it, and check that they agree.

    k defaults to 4, roughly T_c/4, following the observation that smaller k
    gives sharper alignment estimates.
    """
    A, B, n = align_time(prepare(A), prepare(B), t)
    if n < 2 * k + 2:
        return np.nan

    K, L = _gram_centered(A), _gram_centered(B)

    def _knn_mask(G: np.ndarray, kk: int) -> np.ndarray:
        M = np.zeros_like(G, dtype=bool)
        Gm = G.copy()
        np.fill_diagonal(Gm, -np.inf)
        idx = np.argsort(-Gm, axis=1)[:, :kk]
        rows = np.repeat(np.arange(G.shape[0]), kk)
        M[rows, idx.ravel()] = True
        return M

    mask = _knn_mask(K, k) & _knn_mask(L, k)      # mutual neighbours only
    np.fill_diagonal(mask, False)
    if mask.sum() < 3:
        return np.nan

    Km, Lm = K * mask, L * mask
    num = float((Km * Lm).sum())
    den = float(np.sqrt((Km * Km).sum() * (Lm * Lm).sum()))
    if den <= 0:
        return np.nan
    return float(1.0 - num / den)


# ==========================================================================
# DIRECTIONAL -- ridge map calibrated on real video
# ==========================================================================
class RidgeReference:
    """One linear map, calibrated on real video, used to score every video.

    The well-posed replacement for a degenerate per-video fit. It trains no
    detector: one closed-form ridge solve on real frames only, no generated
    video seen, no label beyond "real".

    DIRECTION (FIX 6). Default is `b_to_a`: the complementary stream predicts
    the appearance reference. This is the training-free analogue of Paper 1's
    CMID, which asks how far the appearance representation is displaced when
    reconstructed from motion and depth evidence. Attention supplies a
    data-dependent nonlinear reconstruction there; ridge supplies a fixed
    linear one here. Run `a_to_b` as well and report both.

    Calibration videos are excluded from scoring: a video scored under a map
    fitted to it would receive an optimistically low residual.
    """

    def __init__(self, alpha: float = C.RIDGE_ALPHA,
                 direction: str = "b_to_a"):
        if direction not in ("a_to_b", "b_to_a"):
            raise ValueError(direction)
        self.alpha = alpha
        self.direction = direction
        self.W = self.mu_src = self.mu_dst = None
        self.n_fit = 0

    def _orient(self, A, B):
        return (A, B) if self.direction == "a_to_b" else (B, A)

    def fit(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> "RidgeReference":
        src_parts, dst_parts = [], []
        for a, b in pairs:
            a, b, _ = align_time(prepare(a, center=False),
                                 prepare(b, center=False))
            s, d = self._orient(a, b)
            src_parts.append(s)
            dst_parts.append(d)
        S = np.concatenate(src_parts, 0)
        D = np.concatenate(dst_parts, 0)

        self.mu_src, self.mu_dst = S.mean(0), D.mean(0)
        S, D = S - self.mu_src, D - self.mu_dst

        d = S.shape[1]
        G = S.T @ S + self.alpha * np.eye(d) * len(S) / d
        self.W = np.linalg.solve(G, S.T @ D)
        self.n_fit = len(pairs)
        return self

    def score(self, A: np.ndarray, B: np.ndarray) -> float:
        """Normalised residual ||S W - D||^2 / ||D||^2 for one video."""
        if self.W is None:
            raise RuntimeError("fit() first")
        A, B, _ = align_time(prepare(A, center=False),
                             prepare(B, center=False))
        S, D = self._orient(A, B)
        S, D = S - self.mu_src, D - self.mu_dst
        r = S @ self.W - D
        return float((r ** 2).sum() / ((D ** 2).sum() + EPS))


# ==========================================================================
# CONTROL -- random projection (projection control, NOT pretraining control)
# ==========================================================================
class RandomProjectionCMID:
    """Paper 1's CMID formula with random, untrained projections and attention.

    SPECIFICITY CONTROL, not a primary measure.

    On synthetic data where two representations share a latent trajectory,
    this does not discriminate that case from two independent
    representations: the ratio between conditions stays at 1.00 across
    temperatures from 1 to 300. With random projections the attention logits
    carry no information about which frame of B corresponds to which frame of
    A, so the softmax is near-uniform, H_cross collapses toward the mean of
    Hb, and the normalised displacement goes to 1 regardless of input.

    That makes it a useful control rather than a useless function. It asks
    whether Paper 1's mechanism requires LEARNED projections and attention.
    It does NOT ask whether pretraining is required: random projections of a
    pretrained representation still carry everything that representation
    encodes. The control for pretraining would be a randomly initialised
    encoder.
    """

    def __init__(self, d_a: int, d_b: int, d: int = C.RANDPROJ_DIM,
                 seed: int = C.SEED):
        rng = np.random.default_rng(seed)
        self.Pa = rng.standard_normal((d_a, d)) / np.sqrt(d_a)
        self.Pb = rng.standard_normal((d_b, d)) / np.sqrt(d_b)
        self.d = d

    @staticmethod
    def _standardise(X: np.ndarray) -> np.ndarray:
        """Centre, then scale to unit mean row norm.

        Without this the measure is dominated by the relative magnitude of
        the two encoders' features rather than by their disagreement:
        rescaling B by 10x moved the raw score from 1.9 to 102 while the
        agreement structure was unchanged. This is the same magnitude-
        inflation failure that makes the unnormalised CMID in Paper 1 unsafe
        to optimise directly.
        """
        X = X - X.mean(0, keepdims=True)
        return X / (np.linalg.norm(X, axis=1).mean() + EPS)

    def score(self, A: np.ndarray, B: np.ndarray, normalize: bool = True) -> float:
        A, B, _ = align_time(prepare(A, center=False),
                             prepare(B, center=False))
        Ha = self._standardise(A) @ self.Pa
        Hb = self._standardise(B) @ self.Pb

        logits = Ha @ Hb.T / np.sqrt(self.d)
        logits -= logits.max(1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(1, keepdims=True) + EPS
        H_cross = w @ Hb

        disp = ((H_cross - Ha) ** 2).sum(1).mean()
        if not normalize:
            return float(disp)
        return float(disp / ((Ha ** 2).sum(1).mean() + EPS))


# ==========================================================================
# CONTROL -- trivial within-encoder statistic
# ==========================================================================
def feature_norm_stat(A: np.ndarray) -> float:
    """Coefficient of variation of per-frame feature norms, one encoder.

    Answers "could you have got this from a single encoder with no comparison
    at all?" If it separates the classes as well as the cross-encoder
    measures, the paper's premise is in trouble. Report it in the headline
    table, not a footnote.
    """
    A = prepare(A, center=False)
    n = np.linalg.norm(A, axis=1)
    return float(n.std() / (n.mean() + EPS))


# ==========================================================================
# COVARIATE -- temporal variability (FIX 3)
# ==========================================================================
def temporal_variability(A: np.ndarray) -> float:
    """Mean normalised step size along the representation trajectory.

    A proxy for how much the video changes, used as a CONFOUND COVARIATE.

    Both primary measures read the T x T frame-similarity structure. A
    near-static video has a nearly uniform similarity matrix, so the rank
    correlation is estimated over a degenerate spread and becomes unstable.
    If generated videos are systematically more static than the real videos
    in a benchmark -- plausible, since generation is prompt-conditioned and
    short -- part of any measured separation is a motion difference rather
    than an agreement difference.

    This is the first objection a referee will raise against a positive
    result. Report the correlation between this covariate and each measure,
    and report effect sizes stratified by covariate tercile. If the effect
    survives within strata, the objection is closed.

    Computed from the appearance encoder so it is available for every pair
    without requiring RAFT.
    """
    A = prepare(A, center=False)
    if A.shape[0] < 2:
        return np.nan
    steps = np.linalg.norm(A[1:] - A[:-1], axis=1)
    return float(steps.mean() / (np.linalg.norm(A, axis=1).mean() + EPS))


def flow_magnitude(raft_features: np.ndarray) -> float:
    """Mean flow magnitude from cached RAFT descriptors.

    A more direct motion covariate than `temporal_variability` when RAFT
    features are available. The first 16 dimensions of the RAFT descriptor
    are the 4x4 pooled magnitude grid.
    """
    A = prepare(raft_features, center=False)
    return float(A[:, :16].mean())


# ==========================================================================
# DEGENERATE -- retained only to reproduce the failure in the paper
# ==========================================================================
def linear_residual_per_video(A: np.ndarray, B: np.ndarray) -> float:
    """Per-video least-squares residual. Returns ~0 whenever T < d1.

    Kept so Appendix A of the manuscript is reproducible rather than
    asserted. DO NOT REPORT AS A RESULT.
    """
    A, B, _ = align_time(prepare(A, center=False), prepare(B, center=False))
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    return float(((A @ W - B) ** 2).sum() / ((B ** 2).sum() + EPS))


# ==========================================================================
# PATCH-LEVEL VARIANTS
#
# SPLIT (ECCV 2026) argues that collapsing a frame into a single embedding
# discards localized evidence, and supports it with a large result: on
# FakeParts at 0.1% FPR, patch-level scoring reaches 74.5% Fake Recall where
# clip-pooled D3 reaches 2.3%, using the same frozen backbone.
#
# Our pooled per-frame descriptors are exactly the operation they criticise.
# These variants answer it directly rather than conceding it, and they fix a
# second problem at the same time: CKA over T_c = 16 frames is our largest
# methodological vulnerability, while a 14x14 patch grid gives 196
# observations per frame.
#
# The question changes from "do the encoders agree about which FRAMES resemble
# which others" to "do they agree about which REGIONS of this frame resemble
# which others". Report both; if patch-level is stronger that is a result, and
# if it is not, the objection has been answered with data.
# ==========================================================================

def _resample_grid(P: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a patch grid (H, W, D) to hw.

    Encoders patchify differently -- CLIP ViT-B/32 at 224 gives 7x7, DINOv2
    ViT-B/14 gives 16x16 -- so grids must be brought to a common resolution
    before they can be compared position by position.
    """
    H, W, D = P.shape
    th, tw = hw
    if (H, W) == (th, tw):
        return P
    yi = np.linspace(0, H - 1, th)
    xi = np.linspace(0, W - 1, tw)
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    wy = (yi - y0)[:, None, None]
    wx = (xi - x0)[None, :, None]
    top = P[y0][:, x0] * (1 - wx) + P[y0][:, x1] * wx
    bot = P[y1][:, x0] * (1 - wx) + P[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def cka_patch(Pa: np.ndarray, Pb: np.ndarray, debiased: bool = True,
              grid: tuple[int, int] | None = None) -> float:
    """1 - CKA computed over PATCHES within each frame, averaged over frames.

    Pa, Pb: (T, H, W, D) patch grids from two encoders.

    Each frame contributes one CKA value computed over H*W patch tokens, so
    the sample size per measurement is the number of patches rather than the
    number of frames. Frames are then averaged.

    Grids are resampled to the coarser of the two by default, since
    upsampling would invent structure that was never measured.
    """
    Pa = np.asarray(Pa, dtype=np.float64)
    Pb = np.asarray(Pb, dtype=np.float64)
    if Pa.ndim != 4 or Pb.ndim != 4:
        raise ValueError("expected (T, H, W, D) patch grids")
    T = min(Pa.shape[0], Pb.shape[0])
    grid = grid or (min(Pa.shape[1], Pb.shape[1]),
                    min(Pa.shape[2], Pb.shape[2]))

    vals = []
    for t in range(T):
        A = _resample_grid(Pa[t], grid).reshape(-1, Pa.shape[-1])
        B = _resample_grid(Pb[t], grid).reshape(-1, Pb.shape[-1])
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
        if A.shape[0] < 8:
            continue
        if debiased:
            K, L = _gram_centered(A), _gram_centered(B)
            hkl, hkk, hll = (_hsic_unbiased(K, L), _hsic_unbiased(K, K),
                             _hsic_unbiased(L, L))
            if np.isfinite(hkl) and hkk > 0 and hll > 0:
                vals.append(1.0 - hkl / np.sqrt(hkk * hll))
        else:
            num = np.linalg.norm(B.T @ A, "fro") ** 2
            den = (np.linalg.norm(A.T @ A, "fro")
                   * np.linalg.norm(B.T @ B, "fro"))
            vals.append(1.0 - num / (den + EPS))
    return float(np.mean(vals)) if vals else np.nan


def neighbourhood_patch(Pa: np.ndarray, Pb: np.ndarray,
                        grid: tuple[int, int] | None = None) -> float:
    """1 - Spearman between the two patch-similarity structures, per frame.

    Do the two encoders agree about which REGIONS of this frame resemble which
    others? The spatial analogue of the frame-level measure, and the variant
    that remains sensitive to a localized edit.
    """
    Pa = np.asarray(Pa, dtype=np.float64)
    Pb = np.asarray(Pb, dtype=np.float64)
    T = min(Pa.shape[0], Pb.shape[0])
    grid = grid or (min(Pa.shape[1], Pb.shape[1]),
                    min(Pa.shape[2], Pb.shape[2]))

    vals = []
    for t in range(T):
        A = _resample_grid(Pa[t], grid).reshape(-1, Pa.shape[-1])
        B = _resample_grid(Pb[t], grid).reshape(-1, Pb.shape[-1])
        n = A.shape[0]
        if n < 8:
            continue
        iu = np.triu_indices(n, k=1)
        ga, gb = _cosine_gram(A)[iu], _cosine_gram(B)[iu]
        if np.std(ga) < EPS or np.std(gb) < EPS:
            continue
        rho, _ = spearmanr(ga, gb)
        if np.isfinite(rho):
            vals.append(1.0 - rho)
    return float(np.mean(vals)) if vals else np.nan


def patch_disagreement_map(Pa: np.ndarray, Pb: np.ndarray,
                           grid: tuple[int, int] | None = None) -> np.ndarray:
    """Per-patch disagreement, averaged over time. Returns (H, W).

    Not a detection score -- a localisation output. Useful for the
    qualitative figure and for the partial-edit extension, where the question
    is not whether a video is generated but which region was edited.
    """
    Pa = np.asarray(Pa, dtype=np.float64)
    Pb = np.asarray(Pb, dtype=np.float64)
    T = min(Pa.shape[0], Pb.shape[0])
    grid = grid or (min(Pa.shape[1], Pb.shape[1]),
                    min(Pa.shape[2], Pb.shape[2]))
    acc = np.zeros(grid)
    for t in range(T):
        A = _resample_grid(Pa[t], grid).reshape(-1, Pa.shape[-1])
        B = _resample_grid(Pb[t], grid).reshape(-1, Pb.shape[-1])
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + EPS)
        # local disagreement: how differently each patch relates to its own
        # frame under the two encoders
        ra, rb = A @ A.T, B @ B.T
        acc += np.abs(ra - rb).mean(1).reshape(grid)
    return acc / max(T, 1)


MEASURE_FN = {
    "cka": cka,
    "cknna": cknna,
    "cka_patch": cka_patch,
    "nbr_patch": neighbourhood_patch,
    "nbr": neighbourhood,
    "lstsq_degenerate": linear_residual_per_video,
}

COVARIATE_FN = {
    "temporal_variability": temporal_variability,
}
