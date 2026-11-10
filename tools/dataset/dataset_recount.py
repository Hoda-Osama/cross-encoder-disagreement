#!/usr/bin/env python3
"""
dataset_recount.py: recount a video benchmark from the files on disk and
check it for duplicate clips.

Answers Reviewer 1, points 2 and 3:
  * exact video counts per folder and sub-folder (categories, generators,
    real groups), taken from the files themselves rather than from any paper;
  * frame rate, frame count, duration and resolution per group, plus how many
    videos have fewer than 32 frames and are therefore padded by repetition;
  * duplicate clips:
      - exact duplicates (identical file content, MD5),
      - files with the same name in different folders,
      - near-duplicates (same content re-encoded, resized or trimmed),
        found by comparing perceptual hashes of three frames.

Usage (run once per benchmark):

    pip install opencv-python numpy
    python dataset_recount.py /path/to/FakeParts  --out fakeparts_audit
    python dataset_recount.py /path/to/ViF-Bench  --out vifbench_audit

Optional:
    --workers 8        parallel processes (default: number of CPUs)
    --min-frames 32    frame threshold for the padding count (default 32)
    --hamming 6        near-duplicate tolerance in bits, out of 192 (default 6)
    --no-md5           skip exact-duplicate hashing (faster on slow disks)

Output folder:
    per_video.csv        one row per video, with all measured properties
    group_summary.csv    counts and statistics per folder / sub-folder
    duplicates_exact.csv identical files
    duplicates_name.csv  same file name in more than one folder
    duplicates_near.csv  visually near-identical videos (check these by eye)
    REPORT.txt           readable summary of everything above

The script only reads files. It never moves, renames or deletes anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".gif", ".m4v", ".mpg", ".mpeg"}
HASH_POS = (0.1, 0.5, 0.9)          # where in the clip the three frames are taken


# ---------------------------------------------------------------- per video
def dhash(frame: np.ndarray) -> int:
    """64-bit difference hash; robust to re-encoding, resizing and colour shifts."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe(args) -> dict:
    path, root, do_md5 = args
    rel = path.relative_to(root)
    parts = rel.parts
    row = {
        "relpath": str(rel),
        "group1": parts[0] if len(parts) > 1 else ".",
        "group2": "/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else "."),
        "name": path.stem,
        "bytes": path.stat().st_size,
        "md5": "", "fps": "", "frames_header": "", "frames_read": "",
        "duration_s": "", "width": "", "height": "", "hash": "", "error": "",
    }
    try:
        if do_md5:
            row["md5"] = md5(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("cannot open")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_hdr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # Count frames by decoding: header counts are often wrong for webm/gif
        # and for variable-frame-rate files, and the padding question depends
        # on the true number of decodable frames.
        frames, n = [], 0
        want = None
        if n_hdr > 0:
            want = {max(0, min(n_hdr - 1, int(p * n_hdr))): i for i, p in enumerate(HASH_POS)}
        keep = {}
        while True:
            ok = cap.grab()
            if not ok:
                break
            if want is not None and n in want:
                ok2, fr = cap.retrieve()
                if ok2:
                    keep[want[n]] = fr
            elif want is None:
                ok2, fr = cap.retrieve()
                if ok2:
                    frames.append(fr)
            n += 1
        cap.release()
        if want is None and frames:                   # header had no count
            keep = {i: frames[max(0, min(len(frames) - 1, int(p * len(frames))))]
                    for i, p in enumerate(HASH_POS)}
        if n == 0:
            raise RuntimeError("no decodable frames")
        if len(keep) < len(HASH_POS):                 # header overstated length
            cap = cv2.VideoCapture(str(path))
            idx = {max(0, min(n - 1, int(p * n))): i for i, p in enumerate(HASH_POS)}
            j = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if j in idx:
                    ok2, fr = cap.retrieve()
                    if ok2:
                        keep[idx[j]] = fr
                j += 1
            cap.release()
        hashes = [dhash(keep[i]) for i in range(len(HASH_POS)) if i in keep]
        row.update({
            "fps": round(fps, 3) if fps else "",
            "frames_header": n_hdr,
            "frames_read": n,
            "duration_s": round(n / fps, 3) if fps else "",
            "width": w, "height": h,
            "hash": "-".join(f"{x:016x}" for x in hashes),
        })
    except Exception as e:  # keep going; report the file
        row["error"] = str(e)[:200]
    return row


# ---------------------------------------------------------------- duplicates
def near_duplicates(rows, max_bits: int):
    """Pairs whose concatenated 192-bit hashes differ in <= max_bits bits.

    Band LSH: the 192 bits are cut into (max_bits + 1) bands, so any pair within
    max_bits must agree exactly on at least one band (pigeonhole). Only pairs
    sharing a band are compared in full.
    """
    items = []
    for i, r in enumerate(rows):
        hs = r["hash"].split("-") if r["hash"] else []
        if len(hs) == len(HASH_POS):
            items.append((i, int("".join(hs), 16)))
    nbits = 64 * len(HASH_POS)
    nb = max_bits + 1
    edges = [int(x) for x in np.linspace(0, nbits, nb + 1)]   # plain ints: numpy ints overflow on 192-bit shifts
    buckets = defaultdict(list)
    for i, v in items:
        for b in range(nb):
            lo, hi = edges[b], edges[b + 1]
            key = (b, (v >> lo) & ((1 << (hi - lo)) - 1))
            buckets[key].append((i, v))
    seen, pairs = set(), []
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:   # skip degenerate buckets (e.g. black frames)
            continue
        for a in range(len(members)):
            ia, va = members[a]
            for b in range(a + 1, len(members)):
                ib, vb = members[b]
                key = (min(ia, ib), max(ia, ib))
                if key in seen:
                    continue
                seen.add(key)
                d = bin(va ^ vb).count("1")
                if d <= max_bits:
                    pairs.append((key[0], key[1], d))
    return pairs


# ---------------------------------------------------------------- reporting
def num(xs):
    return [float(x) for x in xs if x not in ("", None)]


def describe(xs):
    xs = num(xs)
    if not xs:
        return "", "", "", ""
    return (round(min(xs), 2), round(st.median(xs), 2), round(max(xs), 2),
            round(st.mean(xs), 2))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default="dataset_audit")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--min-frames", type=int, default=32)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--no-md5", action="store_true")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        sys.exit(f"No video files found under {root}")
    print(f"{len(files)} video files under {root}; probing with {a.workers} workers ...")

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(probe, (p, root, not a.no_md5)) for p in files]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if k % 1000 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: r["relpath"])

    fields = ["relpath", "group1", "group2", "name", "bytes", "md5", "fps", "frames_header",
              "frames_read", "duration_s", "width", "height", "hash", "error"]
    write_csv(out / "per_video.csv", rows, fields)

    # ---- group summary (both folder levels)
    summ = []
    for level in ("group1", "group2"):
        by = defaultdict(list)
        for r in rows:
            by[r[level]].append(r)
        for g, rs in sorted(by.items()):
            ok = [r for r in rs if not r["error"]]
            fps = describe([r["fps"] for r in ok])
            fr = describe([r["frames_read"] for r in ok])
            du = describe([r["duration_s"] for r in ok])
            res = Counter(f'{r["width"]}x{r["height"]}' for r in ok).most_common(3)
            short = sum(1 for r in ok if int(r["frames_read"]) < a.min_frames)
            summ.append({
                "level": level, "group": g, "videos": len(rs), "unreadable": len(rs) - len(ok),
                "fps_min": fps[0], "fps_median": fps[1], "fps_max": fps[2],
                "frames_min": fr[0], "frames_median": fr[1], "frames_max": fr[2],
                f"videos_under_{a.min_frames}_frames": short,
                f"pct_under_{a.min_frames}_frames": round(100 * short / len(ok), 1) if ok else "",
                "duration_min_s": du[0], "duration_median_s": du[1], "duration_max_s": du[2],
                "top_resolutions": "; ".join(f"{k} ({v})" for k, v in res),
            })
    write_csv(out / "group_summary.csv", summ, list(summ[0].keys()))

    # ---- exact duplicates
    exact = []
    if not a.no_md5:
        by = defaultdict(list)
        for r in rows:
            if r["md5"]:
                by[r["md5"]].append(r)
        gid = 0
        for h, rs in by.items():
            if len(rs) > 1:
                gid += 1
                for r in rs:
                    exact.append({"dup_group": gid, "md5": h, "relpath": r["relpath"], "folder": r["group2"]})
    write_csv(out / "duplicates_exact.csv", exact, ["dup_group", "md5", "relpath", "folder"])

    # ---- same file name in different top-level folders
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    names = []
    for n, rs in by.items():
        if len({r["group2"] for r in rs}) > 1:
            for r in rs:
                names.append({"name": n, "relpath": r["relpath"], "folder": r["group2"]})
    write_csv(out / "duplicates_name.csv", names, ["name", "relpath", "folder"])

    # ---- near duplicates
    pairs = near_duplicates(rows, a.hamming)
    near = [{"hamming_bits": d,
             "video_a": rows[i]["relpath"], "folder_a": rows[i]["group2"],
             "frames_a": rows[i]["frames_read"], "res_a": f'{rows[i]["width"]}x{rows[i]["height"]}',
             "video_b": rows[j]["relpath"], "folder_b": rows[j]["group2"],
             "frames_b": rows[j]["frames_read"], "res_b": f'{rows[j]["width"]}x{rows[j]["height"]}',
             "same_folder": rows[i]["group2"] == rows[j]["group2"]}
            for i, j, d in sorted(pairs, key=lambda t: t[2])]
    write_csv(out / "duplicates_near.csv", near,
              ["hamming_bits", "video_a", "folder_a", "frames_a", "res_a",
               "video_b", "folder_b", "frames_b", "res_b", "same_folder"])

    # ---- readable report
    L = []
    L.append(f"Dataset audit of {root}")
    L.append(f"Video files found: {len(rows)}   unreadable: {sum(1 for r in rows if r['error'])}")
    L.append("")
    L.append("COUNTS AND PROPERTIES PER FOLDER")
    hdr = f"{'folder':40s} {'videos':>7s} {'fps(med)':>9s} {'frames(med)':>12s} {'<'+str(a.min_frames)+'fr':>7s} {'dur s(med)':>11s}"
    for level in ("group1", "group2"):
        L.append("")
        L.append("Top level:" if level == "group1" else "Second level:")
        L.append(hdr)
        for s in summ:
            if s["level"] != level:
                continue
            L.append(f"{s['group'][:40]:40s} {s['videos']:>7d} {str(s['fps_median']):>9s} "
                     f"{str(s['frames_median']):>12s} {s[f'videos_under_{a.min_frames}_frames']:>7d} "
                     f"{str(s['duration_median_s']):>11s}")
    L.append("")
    L.append("DUPLICATES")
    xg = {e["dup_group"] for e in exact}
    xcross = {g for g in xg if len({e["folder"] for e in exact if e["dup_group"] == g}) > 1}
    L.append(f"Exact duplicate files (identical MD5): {len(xg)} groups, {len(xcross)} spanning different sub-folders"
             + ("" if not a.no_md5 else " (skipped: --no-md5)"))
    L.append(f"Same file name in more than one sub-folder: {len({n['name'] for n in names})} names")
    cross_near = [p for p in near if not p["same_folder"]]
    L.append(f"Near-duplicate pairs (<= {a.hamming} of 192 bits): {len(near)} "
             f"({len(cross_near)} between different sub-folders)")
    pc = Counter(tuple(sorted((p["folder_a"], p["folder_b"]))) for p in near)
    for (ga, gb), c in pc.most_common(20):
        L.append(f"   {ga}  <->  {gb}: {c} pairs")
    L.append("")
    L.append("Near-duplicate pairs are candidates: open a few from duplicates_near.csv and")
    L.append("look at them before reporting. Static or near-black clips can match by chance.")
    (out / "REPORT.txt").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nAll files written to {out.resolve()}")


if __name__ == "__main__":
    main()
