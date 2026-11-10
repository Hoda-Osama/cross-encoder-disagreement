"""Within-encoder baselines.

These are the fair same-regime comparison for Paper 2. D3 and ReStraV are
both published, both training-free, and both compute a statistic from frozen
representations. The only structural difference from our measures is the unit
of analysis: they operate *within* one encoder, we operate *between* two.

That contrast is the paper's organising question, so these rows are not
optional garnish. They are what RQ5 is asking about.

D3 and ReStraV are left unimplemented on purpose. Check whether the authors
released code before reimplementing, and state plainly in the paper which
route you took. A reimplementation presented as the original is the kind of
thing that surfaces badly in review.
"""
from __future__ import annotations

import numpy as np

from .measures import prepare, EPS


# ==========================================================================
# Implemented
# ==========================================================================
def feature_norm_variation(A: np.ndarray) -> float:
    """Coefficient of variation of per-frame feature norms.

    The trivial within-encoder statistic. If this separates the classes as
    well as the between-encoder measures, the paper's premise is in trouble,
    so it belongs in the headline table rather than a footnote.
    """
    A = prepare(A, center=False)
    n = np.linalg.norm(A, axis=1)
    return float(n.std() / (n.mean() + EPS))


def temporal_second_order(A: np.ndarray) -> float:
    """Mean magnitude of the second temporal difference, normalised.

    NOT D3. This is a simple second-order temporal statistic in the same
    family, included as an honest stand-in so the within-encoder column is
    not empty while D3 is pending. Label it as such in the paper; do not
    present it as D3.
    """
    A = prepare(A, center=False)
    if A.shape[0] < 3:
        return np.nan
    d2 = A[2:] - 2 * A[1:-1] + A[:-2]
    return float(np.linalg.norm(d2, axis=1).mean()
                 / (np.linalg.norm(A, axis=1).mean() + EPS))


def trajectory_curvature(A: np.ndarray) -> float:
    """Mean angle between consecutive representation-space steps.

    NOT ReStraV. A simple curvature statistic in the same family, same
    caveat as above. ReStraV's actual formulation involves more than this.
    """
    A = prepare(A, center=False)
    if A.shape[0] < 3:
        return np.nan
    v = A[1:] - A[:-1]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)
    cos = np.clip((v[1:] * v[:-1]).sum(1), -1, 1)
    return float(np.arccos(cos).mean())


# ==========================================================================
# Pending
# ==========================================================================
def d3(A: np.ndarray) -> float:
    raise NotImplementedError(
        "D3 (Zheng et al., ICCV 2025). Check for released code first. "
        "If reimplementing, say so explicitly in the paper.")


def restrav(A: np.ndarray) -> float:
    raise NotImplementedError(
        "ReStraV (Interno et al., NeurIPS 2025). Check for released code "
        "first. If reimplementing, say so explicitly in the paper.")


WITHIN_ENCODER = {
    "norm": feature_norm_variation,
    "second_order": temporal_second_order,
    "curvature": trajectory_curvature,
}
