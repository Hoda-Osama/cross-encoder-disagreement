"""Dataset indexing and extraction bookkeeping.

The manifest is a JSONL file, one row per video:

    {"vid": "gvd/sora/00017", "path": "...", "dataset": "gvd",
     "label": 0, "generator": "sora", "split": "test",
     "done": ["clip_b32", "midas_hybrid"]}

label: 1 = real, 0 = generated.  This matches the convention in Paper 1
(p = P(real)); keep it consistent across both papers or the sign of every
effect size flips.

Extraction reads `done` and skips work already finished, so an interrupted
two-day run resumes cleanly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".gif"}


# --------------------------------------------------------------------------
# Dataset scanners
# --------------------------------------------------------------------------
# These two functions are the only place that knows about directory layout.
# Adjust them to match how GVD and GVF are actually laid out on your disk,
# then never touch dataset structure anywhere else in the codebase.

def scan_gvd(root: Path) -> list[dict]:
    """GVD: 11,618 videos, 11 generators, T2V and I2V, plus real controls.

    Expected layout (adjust to yours):
        root/fake/<generator>/*.mp4
        root/real/<source>/*.mp4
    """
    rows = []
    for sub, label in (("fake", 0), ("real", 1)):
        base = root / sub
        if not base.exists():
            print(f"  [warn] missing {base}")
            continue
        for gen_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(gen_dir.rglob("*")):
                if f.suffix.lower() not in VIDEO_EXTS:
                    continue
                rows.append({
                    "vid": f"gvd/{gen_dir.name}/{f.stem}",
                    "path": str(f),
                    "dataset": "gvd",
                    "label": label,
                    # Real corpora are tagged separately, not lumped as
                    # "real". GVD draws real video from YouTube-VOS and
                    # GOT-10k, and the real-vs-real provenance control in
                    # src/analyze.py needs to tell them apart -- it measures
                    # discrepancy BETWEEN real corpora to bound how much of
                    # the real/generated effect is corpus identity.
                    "generator": gen_dir.name if label == 0 else f"real:{gen_dir.name}",
                    "split": "test",     # training-free: everything is measurement data
                    "done": [],
                })
    return rows


def scan_gvf(root: Path) -> list[dict]:
    """GVF: built on MSVD/MSR-VTT, four training generators, nine test generators.

    Expected layout (adjust to yours):
        root/<split>/fake/<generator>/*.mp4
        root/<split>/real/*.mp4
    """
    rows = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sub, label in (("fake", 0), ("real", 1)):
            base = split_dir / sub
            if not base.exists():
                continue
            gen_dirs = [p for p in base.iterdir() if p.is_dir()] or [base]
            for gen_dir in sorted(gen_dirs):
                for f in sorted(gen_dir.rglob("*")):
                    if f.suffix.lower() not in VIDEO_EXTS:
                        continue
                    rows.append({
                        "vid": f"gvf/{split_dir.name}/{gen_dir.name}/{f.stem}",
                        "path": str(f),
                        "dataset": "gvf",
                        "label": label,
                        "generator": gen_dir.name if label == 0 else "real:msrvtt",
                        "split": split_dir.name,
                        "done": [],
                    })
    return rows


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load() -> list[dict]:
    if not C.MANIFEST_PATH.exists():
        return []
    with open(C.MANIFEST_PATH) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save(rows: list[dict]) -> None:
    tmp = C.MANIFEST_PATH.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(C.MANIFEST_PATH)     # atomic; survives a kill mid-write


def build(datasets: list[str], merge: bool = True) -> list[dict]:
    rows: list[dict] = []
    if "gvd" in datasets:
        print(f"scanning GVD at {C.GVD_ROOT}")
        rows += scan_gvd(C.GVD_ROOT)
    if "gvf" in datasets:
        print(f"scanning GVF at {C.GVF_ROOT}")
        rows += scan_gvf(C.GVF_ROOT)

    if merge:
        # Preserve `done` for videos already indexed, so rebuilding the
        # manifest after adding a dataset does not discard finished work.
        prev = {r["vid"]: r.get("done", []) for r in load()}
        for r in rows:
            r["done"] = prev.get(r["vid"], [])

    save(rows)
    return rows


def summary(rows: list[dict]) -> None:
    from collections import Counter
    print(f"\n{len(rows)} videos total")
    for ds in sorted({r['dataset'] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        n_real = sum(r["label"] == 1 for r in sub)
        print(f"\n  {ds}: {len(sub)}  ({n_real} real / {len(sub) - n_real} generated)")
        for gen, n in sorted(Counter(r["generator"] for r in sub).items()):
            print(f"      {gen:<28s} {n:>6d}")

    done = Counter(e for r in rows for e in r.get("done", []))
    if done:
        print("\n  extracted:")
        for enc, n in sorted(done.items()):
            print(f"      {enc:<28s} {n:>6d} / {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "status"])
    ap.add_argument("--datasets", nargs="+", default=["gvd", "gvf"])
    a = ap.parse_args()

    rows = build(a.datasets) if a.cmd == "build" else load()
    summary(rows)
