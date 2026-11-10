"""Feature cache I/O. Pure numpy, no torch.

Kept separate from src/extract.py on purpose. The analysis path
(measures, analyze, figures) reads cached features and must run on a machine
with no GPU stack installed; importing the extraction driver would pull in
torch and every encoder library for no reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C


def feature_path(vid: str, encoder: str) -> Path:
    return C.FEATURE_DIR / encoder / f"{vid.replace('/', '__')}.npy"


def save_features(vid: str, encoder: str, feats: np.ndarray) -> None:
    p = feature_path(vid, encoder)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, feats)
    tmp.replace(p)              # atomic; a kill mid-write cannot corrupt


def load_features(vid: str, encoder: str) -> np.ndarray | None:
    p = feature_path(vid, encoder)
    return np.load(p) if p.exists() else None
