"""Video decoding.

One job: turn a video file into a (T, H, W, 3) uint8 array of T uniformly
spaced frames at DECODE_SIZE resolution.

Two things matter here and both are forensics-specific.

Frames are never re-encoded lossily.  If you cache them (--cache-frames)
they go to disk as raw .npy, not JPEG.  Re-compressing frames in a study
about compression-sensitive artifacts would contaminate the measurement.

Decoding happens at DECODE_SIZE = 384 because the depth models want that
much.  Appearance models downsample from it.  Decoding at 224 and upsampling
for depth would degrade exactly the stream whose contribution is in question.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C

try:
    import decord
    decord.bridge.set_bridge("native")
    _BACKEND = "decord"
except ImportError:                              # pragma: no cover
    import cv2
    _BACKEND = "opencv"


class DecodeError(RuntimeError):
    pass


def _uniform_indices(n_available: int, n_want: int) -> np.ndarray:
    """Uniformly spaced frame indices, with cyclic repetition if too short.

    Repetition rather than padding keeps motion continuous, which matters
    for VideoMAE and RAFT.  Paper 1 uses the same policy.
    """
    if n_available <= 0:
        raise DecodeError("zero frames")
    if n_available >= n_want:
        return np.linspace(0, n_available - 1, n_want).round().astype(int)
    return np.arange(n_want) % n_available


def _resize(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize (T,H,W,3) uint8 to (T,size,size,3) with area interpolation."""
    import cv2
    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    return out


def _decode_decord(path: str, n: int) -> np.ndarray:
    vr = decord.VideoReader(path, num_threads=2)
    idx = _uniform_indices(len(vr), n)
    return vr.get_batch(idx).asnumpy()           # (T,H,W,3) RGB uint8


def _decode_opencv(path: str, n: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise DecodeError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0:
        idx = set(_uniform_indices(total, n).tolist())
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in idx:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            i += 1
    else:
        # Some containers report no frame count.  Read everything, subsample.
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if frames:
            frames = [frames[j] for j in _uniform_indices(len(frames), n)]
    cap.release()

    if not frames:
        raise DecodeError(f"no frames decoded from {path}")
    if len(frames) < n:
        frames = [frames[j] for j in _uniform_indices(len(frames), n)]
    return np.stack(frames)


def decode(path: str, n_frames: int = C.NUM_FRAMES,
           size: int = C.DECODE_SIZE) -> np.ndarray:
    """Decode a video to (n_frames, size, size, 3) uint8 RGB."""
    try:
        frames = (_decode_decord(path, n_frames) if _BACKEND == "decord"
                  else _decode_opencv(path, n_frames))
    except DecodeError:
        raise
    except Exception as exc:
        raise DecodeError(f"{path}: {exc}") from exc

    if frames.shape[0] != n_frames:
        frames = frames[_uniform_indices(frames.shape[0], n_frames)]
    if frames.shape[1] != size or frames.shape[2] != size:
        frames = _resize(frames, size)
    return np.ascontiguousarray(frames, dtype=np.uint8)


def frame_cache_path(vid: str) -> Path:
    return C.FRAME_DIR / f"{vid.replace('/', '__')}.npy"


def load_or_decode(vid: str, path: str, use_cache: bool) -> np.ndarray:
    if use_cache:
        p = frame_cache_path(vid)
        if p.exists():
            return np.load(p)
        frames = decode(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, frames)                       # lossless, deliberately
        return frames
    return decode(path)
