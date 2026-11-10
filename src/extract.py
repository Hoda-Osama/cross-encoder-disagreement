"""Feature extraction.

Single pass over the manifest. All requested encoders are held resident and
each video is decoded exactly once, which avoids both the 200 GB frame cache
and the seven-fold re-decode.

Resumable: the manifest records which encoders have finished for each video,
and finished work is skipped on restart.

Usage
-----
    # pilot: two encoders, 400 videos, enough for the go/no-go gate
    python -m src.extract --encoders clip_b32 midas_hybrid --limit 400

    # full run
    python -m src.extract

    # add an encoder later (re-decodes, writes only the new features)
    python -m src.extract --encoders unidepth_l
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .cache_io import feature_path, load_features, save_features  # noqa: F401
from . import decode as D
from . import manifest as M
from .encoders import build as build_encoder


def run(cfg: C.ExtractConfig) -> None:
    rows = M.load()
    if not rows:
        raise SystemExit("empty manifest -- run `python -m src.manifest build` first")

    rows = [r for r in rows if r["dataset"] in cfg.datasets]

    todo = [r for r in rows
            if cfg.overwrite or any(e not in r.get("done", []) for e in cfg.encoders)]
    if cfg.limit:
        # Stratify the pilot subset so it is not all one generator.
        by_gen: dict[str, list] = {}
        for r in todo:
            by_gen.setdefault(r["generator"], []).append(r)
        per = max(1, cfg.limit // max(1, len(by_gen)))
        todo = [r for g in by_gen.values() for r in g[:per]][:cfg.limit]

    print(f"{len(todo)} videos to process, encoders: {cfg.encoders}")
    if not todo:
        return

    print("loading encoders...")
    encs = {}
    for name in cfg.encoders:
        try:
            encs[name] = build_encoder(name)
            mem = torch.cuda.memory_allocated() / 2**30
            print(f"  {name:<16s} ok   ({mem:.2f} GiB resident)")
        except Exception as exc:
            print(f"  {name:<16s} FAILED: {exc}")
    if not encs:
        raise SystemExit("no encoders loaded")

    index = {r["vid"]: r for r in M.load()}
    t0, n_ok, n_fail = time.time(), 0, 0

    for i, row in enumerate(todo, 1):
        vid = row["vid"]
        done = set(index[vid].get("done", []))
        need = [e for e in encs if cfg.overwrite or e not in done]
        if not need:
            continue

        try:
            frames = D.load_or_decode(vid, row["path"], cfg.cache_frames)
        except Exception as exc:
            print(f"  [decode fail] {vid}: {exc}")
            n_fail += 1
            continue

        for name in need:
            try:
                save_features(vid, name, encs[name].encode(frames))
                done.add(name)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [OOM] {vid} / {name} -- lower FRAME_CHUNK in config.py")
            except Exception:
                print(f"  [encode fail] {vid} / {name}")
                traceback.print_exc(limit=1)

        index[vid]["done"] = sorted(done)
        n_ok += 1

        if i % 50 == 0 or i == len(todo):
            M.save(list(index.values()))          # checkpoint the manifest
            el = time.time() - t0
            rate = i / el
            eta = (len(todo) - i) / rate / 3600
            print(f"  {i}/{len(todo)}  {rate:.2f} vid/s  "
                  f"eta {eta:.1f}h  ok={n_ok} fail={n_fail}")

    M.save(list(index.values()))
    print(f"\ndone: {n_ok} processed, {n_fail} failed, "
          f"{(time.time() - t0) / 3600:.2f}h elapsed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=C.ALL_ENCODERS)
    ap.add_argument("--datasets", nargs="+", default=["gvd", "gvf"])
    ap.add_argument("--limit", type=int, default=None,
                    help="stratified subset, for the pilot")
    ap.add_argument("--cache-frames", action="store_true",
                    help="write decoded frames as lossless .npy (needs disk)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    run(C.ExtractConfig(encoders=a.encoders, datasets=a.datasets, limit=a.limit,
                        cache_frames=a.cache_frames, overwrite=a.overwrite))
