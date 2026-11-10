"""Central configuration.

Everything path- or hyperparameter-shaped lives here so that no experiment
depends on a value buried in a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths.  Edit these two, everything else is derived.
# --------------------------------------------------------------------------
GVD_ROOT = Path("/data/GVD")
GVF_ROOT = Path("/data/GVF")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
FEATURE_DIR = CACHE_DIR / "features"
FRAME_DIR = CACHE_DIR / "frames"          # only used with --cache-frames
MANIFEST_PATH = CACHE_DIR / "manifest.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (CACHE_DIR, FEATURE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
# T = 32 rather than 16.  CKA and neighbourhood agreement operate on the
# T x T frame-similarity structure; at T = 16 there are only 120 unique
# off-diagonal pairs to correlate over, which makes the per-video measure
# noisy.  At T = 32 there are 496.  See DESIGN.md section 2.
NUM_FRAMES = 32

# Decode resolution.  Depth models want 384; appearance models want 224.
# Decode once at the larger size and let each encoder downsample.  Never
# the other way around.
DECODE_SIZE = 384

# VideoMAE consumes 16-frame clips, so it sees NUM_FRAMES // 16 clips.
VIDEOMAE_CLIP_LEN = 16

# FIX 2 -- common sequence length for ALL pairs.
#
# Encoders have different native granularity: per-frame encoders emit 32
# vectors, VideoMAE emits 16 tubelets, RAFT emits 31 flow descriptors.
# Resampling each pair down to its own shorter side meant that
# appearance-geometry pairs were measured over 496 frame pairs and
# appearance-motion pairs over 120, so any comparison BETWEEN role-pairs was
# partly a comparison between sample sizes -- which is exactly what H2 rests
# on. Every pair is now resampled to COMMON_T.
#
# 16 is the binding constraint, from VideoMAE.
COMMON_T = 16

# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE = "float16"          # weights and cached features
FRAME_CHUNK = 4            # frames per forward pass for depth models (8 GB safe)
SEED = 42                  # fixed for random projections; never varied

# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------
# role -> list of encoder keys.  Each role deliberately contains one encoder
# that Paper 1 also uses and at least one that it does not, so the paper can
# ask whether a result depends on the specific encoder.
ENCODER_ROLES: dict[str, list[str]] = {
    "appearance": ["clip_b32", "dinov2_b14"],
    "motion": ["videomae_b", "raft_small"],
    "geometry": ["midas_hybrid", "dav2_base", "unidepth_l"],
}

# Encoders Paper 1 uses.  Tagged so the analysis can report the Paper 1
# pairing separately from the alternatives.
PAPER1_ENCODERS = {"clip_b32", "videomae_b", "midas_hybrid"}

ALL_ENCODERS = [e for v in ENCODER_ROLES.values() for e in v]


def role_of(encoder: str) -> str:
    for role, members in ENCODER_ROLES.items():
        if encoder in members:
            return role
    raise KeyError(encoder)


def cross_role_pairs(encoders: list[str] | None = None) -> list[tuple[str, str]]:
    """All encoder pairs drawn from *different* roles.

    Same-role pairs (CLIP vs DINOv2) are excluded from the main grid because
    they measure something else -- agreement between two encoders of the same
    kind -- though they make a useful control.  Use `same_role_pairs` for that.
    """
    encoders = encoders or ALL_ENCODERS
    out = []
    for i, a in enumerate(encoders):
        for b in encoders[i + 1:]:
            if role_of(a) != role_of(b):
                out.append((a, b))
    return out


def same_role_pairs(encoders: list[str] | None = None) -> list[tuple[str, str]]:
    """Control pairs: two encoders playing the same role.

    If discrepancy separates real from generated video just as well for
    same-role pairs, the effect is not about complementary priors and the
    paper's framing has to change.  Report these.
    """
    encoders = encoders or ALL_ENCODERS
    out = []
    for i, a in enumerate(encoders):
        for b in encoders[i + 1:]:
            if role_of(a) == role_of(b):
                out.append((a, b))
    return out


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------
MEASURES = ["cka", "cknna", "nbr", "ridge", "randproj", "norm"]

# k for the mutual-nearest-neighbour mask in CKNNA. ~T_c/4.
CKNNA_K = 4

RIDGE_ALPHA = 1.0          # regularisation for the reference map
RANDPROJ_DIM = 128         # shared dimension for the random-projection control

# FIX 6 -- ridge direction.
# "b_to_a" = complementary stream predicts the appearance reference. This is
# the training-free analogue of Paper 1's CMID, which asks how far the
# appearance representation is displaced when reconstructed from motion and
# depth evidence. Both directions are computed; this one is primary.
RIDGE_DIRECTION = "b_to_a"

# FIX 5 -- debiased CKA. Linear CKA is biased at small n and n here is 16.
# Both estimators are computed so the paper can show the ordering is
# unchanged; the debiased one is primary.
CKA_DEBIASED = True

# FIX 1 -- depth and flow representations.
#   "backbone"  : ViT patch-token features from the depth model's encoder.
#   "descriptor": hand-pooled 4x4 grid of the PREDICTED depth map.
#
# "backbone" is the default and the better choice. With "descriptor" the
# study compares 512-d ViT features against an 80-d hand-designed summary,
# so if appearance-geometry pairs behave differently from appearance-motion
# pairs the cause could be complementary priors OR the fact that one
# comparison is ViT-vs-ViT and the other is ViT-vs-statistic. Nothing in the
# design distinguishes those. Backbone features make every role like-for-like
# and remove an unpublished, arbitrary descriptor from the method.
DEPTH_FEATURE_MODE = "backbone"

# FIX 4 -- pre-specified primary comparison.
#
# The grid contains ~12 pairs x 5 measures. Reporting the maximum effect size
# over ~60 tests is biased upward and its bootstrap CI does not account for
# the selection. Fix the headline comparison BEFORE looking at the full grid;
# everything else is reported as exploratory.
#
# CLIP-MiDaS with debiased CKA is the natural choice: both encoders are
# inherited from Paper 1, and CKA is the measure with no free parameters.
PRIMARY_PAIR = ("clip_b32", "midas_hybrid")
PRIMARY_MEASURE = "cka"

# FIX 3 -- confound covariate.
# Both primary measures read the frame-similarity structure, which depends on
# how much the video changes. Report the correlation with this covariate and
# stratified effect sizes.
COVARIATE = "temporal_variability"
COVARIATE_STRATA = 3


@dataclass
class ExtractConfig:
    encoders: list[str] = field(default_factory=lambda: list(ALL_ENCODERS))
    limit: int | None = None
    cache_frames: bool = False
    overwrite: bool = False
    datasets: list[str] = field(default_factory=lambda: ["fakeparts", "vif_bench"])
