# =============================================================================
# dataset_recount_kaggle.py
# Recount a benchmark subset by subset on Kaggle, then check duplicates across
# all subsets, even after the earlier subsets have been deleted.
#
# HOW TO USE (each "# %%" block is one Kaggle cell):
#   Cell 1  settings          -> edit DATASET, SUBSET_NAME, SUBSET_DIR
#   Cell 2  functions         -> run once per session
#   Cell 3  (optional) download one subset from Hugging Face
#   Cell 4  audit this subset -> saves results to /kaggle/working/audit/...
#   Cell 5  (optional) delete the subset to free disk
#   ...repeat cells 1, 3, 4, 5 for the next subset...
#   Cell 6  final report      -> run once, after all subsets are done
#
# Only the hashes and properties are kept, never the videos, so duplicates are
# still found between a subset you deleted and the one you have now.
#
# IMPORTANT: /kaggle/working is erased when the session ends unless you save a
# version ("Save Version" -> "Save & Run All" or "Quick Save"). If you work over
# several sessions, either save a version after each subset, or download
# /kaggle/working/audit/<DATASET>/subsets/*.csv and upload them later as a
# Kaggle dataset, then point PREVIOUS_RESULTS_DIR to it.
# =============================================================================


# %% ------------------------------------------------------------ CELL 1: SETTINGS
DATASET      = "FakeParts"            # "FakeParts" or "ViF-Bench" (keeps results apart)
SUBSET_NAME  = "real_folder1"         # a short name for the subset you are auditing now
SUBSET_DIR   = "/kaggle/working/data/real_folder1"   # folder that holds this subset's videos

WORKERS      = 4                      # Kaggle CPU sessions have 4 cores
MIN_FRAMES   = 32                     # videos below this are padded by repetition in the paper
HAMMING_BITS = 6                      # near-duplicate tolerance, out of 192 bits
DO_MD5       = True                   # exact-duplicate hashing; set False if disk is slow

OUT_ROOT     = "/kaggle/working/audit"
PREVIOUS_RESULTS_DIR = None           # e.g. "/kaggle/input/my-audit-results" if results
                                      # from earlier sessions were uploaded as a dataset

# Optional Hugging Face download (cell 3). Leave HF_PATTERN = None to skip.
HF_REPO      = "hi-paris/FakeParts"   # ViF-Bench: "JoeLeelyf/ViF-Bench"
HF_PATTERN   = None                   # e.g. "real/folder1/*" : only files matching this
HF_LOCAL_DIR = SUBSET_DIR


# %% ------------------------------------------------------------ CELL 2: FUNCTIONS
import csv, hashlib, os, shutil, statistics as st, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed   # threads: work in any notebook; OpenCV decoding releases the GIL
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".gif", ".m4v", ".mpg", ".mpeg"}
HASH_POS = (0.1, 0.5, 0.9)
FIELDS = ["subset", "relpath", "folder", "name", "bytes", "md5", "fps", "frames_header",
          "frames_read", "duration_s", "width", "height", "hash", "error"]


def _dhash(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_frames(path, n_hint):
    """Decode every frame once; keep the three frames used for hashing."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    want = {max(0, min(n_hint - 1, int(p * n_hint))): i for i, p in enumerate(HASH_POS)} if n_hint > 0 else {}
    keep, all_frames, n = {}, [], 0
    while cap.grab():
        if n_hint > 0:
            if n in want:
                ok, fr = cap.retrieve()
                if ok:
                    keep[want[n]] = fr
        else:
            ok, fr = cap.retrieve()
            if ok:
                all_frames.append(fr)
        n += 1
    cap.release()
    if n_hint <= 0 and all_frames:
        keep = {i: all_frames[max(0, min(len(all_frames) - 1, int(p * len(all_frames))))]
                for i, p in enumerate(HASH_POS)}
    return fps, w, h, n, keep


def _probe(args):
    path, root, subset, do_md5 = args
    path, root = Path(path), Path(root)
    rel = path.relative_to(root)
    folder = "/".join([subset] + list(rel.parts[:-1]))   # subset + its sub-folders
    row = dict.fromkeys(FIELDS, "")
    row.update(subset=subset, relpath=f"{subset}/{rel}", folder=folder,
               name=path.stem, bytes=path.stat().st_size)
    try:
        if do_md5:
            row["md5"] = _md5(path)
        cap = cv2.VideoCapture(str(path))
        n_hdr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        fps, w, h, n, keep = _read_frames(path, n_hdr)
        if n == 0:
            raise RuntimeError("no decodable frames")
        if len(keep) < len(HASH_POS):            # header overstated the length
            fps, w, h, n, keep = _read_frames(path, n)
        hashes = [_dhash(keep[i]) for i in range(len(HASH_POS)) if i in keep]
        row.update(fps=round(fps, 3) if fps else "", frames_header=n_hdr, frames_read=n,
                   duration_s=round(n / fps, 3) if fps else "", width=w, height=h,
                   hash="-".join(f"{x:016x}" for x in hashes))
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


def _write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def audit_subset():
    root = Path(SUBSET_DIR)
    if not root.exists():
        raise FileNotFoundError(f"SUBSET_DIR not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise RuntimeError(f"No video files under {root}")
    print(f"[{DATASET} / {SUBSET_NAME}] {len(files)} videos, {WORKERS} workers")
    t0, rows = time.time(), []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_probe, (str(p), str(root), SUBSET_NAME, DO_MD5)) for p in files]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if k % 500 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}  ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["relpath"])
    out = Path(OUT_ROOT) / DATASET / "subsets" / f"{SUBSET_NAME}.csv"
    _write_csv(out, rows, FIELDS)          # rerunning a subset overwrites its own file only
    bad = sum(1 for r in rows if r["error"])
    short = sum(1 for r in rows if not r["error"] and int(r["frames_read"]) < MIN_FRAMES)
    fps = [float(r["fps"]) for r in rows if r["fps"] not in ("", None)]
    print(f"Saved {out}")
    print(f"  videos {len(rows)} | unreadable {bad} | under {MIN_FRAMES} frames {short}"
          + (f" | fps median {st.median(fps):.2f}" if fps else ""))
    return rows


def _load_all():
    dirs = [Path(OUT_ROOT) / DATASET / "subsets"]
    if PREVIOUS_RESULTS_DIR:
        dirs.append(Path(PREVIOUS_RESULTS_DIR))
    by_subset = {}
    for d in dirs:
        for p in sorted(d.rglob("*.csv")) if d.exists() else []:
            rows = _read_csv(p)
            if rows and "subset" in rows[0]:
                by_subset[rows[0]["subset"]] = rows      # newest copy of each subset wins
    return [r for s in sorted(by_subset) for r in by_subset[s]], sorted(by_subset)


def _near_pairs(rows, max_bits):
    items = []
    for i, r in enumerate(rows):
        hs = r["hash"].split("-") if r["hash"] else []
        if len(hs) == len(HASH_POS):
            items.append((i, int("".join(hs), 16)))
    nbits, nb = 64 * len(HASH_POS), max_bits + 1
    edges = [int(x) for x in np.linspace(0, nbits, nb + 1)]
    buckets = defaultdict(list)
    for i, v in items:
        for b in range(nb):
            lo, hi = edges[b], edges[b + 1]
            buckets[(b, (v >> lo) & ((1 << (hi - lo)) - 1))].append((i, v))
    seen, pairs = set(), []
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:    # skip degenerate buckets (black frames)
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


def _describe(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    if not xs:
        return "", "", ""
    return round(min(xs), 2), round(st.median(xs), 2), round(max(xs), 2)


def final_report():
    rows, subsets = _load_all()
    if not rows:
        raise RuntimeError("No subset results found. Run cell 4 first.")
    out = Path(OUT_ROOT) / DATASET
    _write_csv(out / "per_video_all.csv", rows, FIELDS)

    # summary per subset and per folder
    summ = []
    for level in ("subset", "folder"):
        by = defaultdict(list)
        for r in rows:
            by[r[level]].append(r)
        for g, rs in sorted(by.items()):
            ok = [r for r in rs if not r["error"]]
            fps, fr, du = (_describe([r[k] for r in ok]) for k in ("fps", "frames_read", "duration_s"))
            short = sum(1 for r in ok if int(r["frames_read"]) < MIN_FRAMES)
            res = Counter(f'{r["width"]}x{r["height"]}' for r in ok).most_common(3)
            summ.append({"level": level, "group": g, "videos": len(rs), "unreadable": len(rs) - len(ok),
                         "fps_min": fps[0], "fps_median": fps[1], "fps_max": fps[2],
                         "frames_min": fr[0], "frames_median": fr[1], "frames_max": fr[2],
                         f"videos_under_{MIN_FRAMES}_frames": short,
                         f"pct_under_{MIN_FRAMES}_frames": round(100 * short / len(ok), 1) if ok else "",
                         "duration_min_s": du[0], "duration_median_s": du[1], "duration_max_s": du[2],
                         "top_resolutions": "; ".join(f"{k} ({v})" for k, v in res)})
    _write_csv(out / "group_summary.csv", summ, list(summ[0].keys()))

    # exact duplicates
    exact, by = [], defaultdict(list)
    for r in rows:
        if r["md5"]:
            by[r["md5"]].append(r)
    gid = 0
    for h, rs in by.items():
        if len(rs) > 1:
            gid += 1
            exact += [{"dup_group": gid, "md5": h, "relpath": r["relpath"], "folder": r["folder"]} for r in rs]
    _write_csv(out / "duplicates_exact.csv", exact, ["dup_group", "md5", "relpath", "folder"])

    # same file name in different folders
    names, by = [], defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    for n, rs in by.items():
        if len({r["folder"] for r in rs}) > 1:
            names += [{"name": n, "relpath": r["relpath"], "folder": r["folder"]} for r in rs]
    _write_csv(out / "duplicates_name.csv", names, ["name", "relpath", "folder"])

    # near duplicates
    near = [{"hamming_bits": d,
             "video_a": rows[i]["relpath"], "folder_a": rows[i]["folder"],
             "frames_a": rows[i]["frames_read"], "res_a": f'{rows[i]["width"]}x{rows[i]["height"]}',
             "video_b": rows[j]["relpath"], "folder_b": rows[j]["folder"],
             "frames_b": rows[j]["frames_read"], "res_b": f'{rows[j]["width"]}x{rows[j]["height"]}',
             "same_folder": rows[i]["folder"] == rows[j]["folder"]}
            for i, j, d in sorted(_near_pairs(rows, HAMMING_BITS), key=lambda t: t[2])]
    _write_csv(out / "duplicates_near.csv", near,
               ["hamming_bits", "video_a", "folder_a", "frames_a", "res_a",
                "video_b", "folder_b", "frames_b", "res_b", "same_folder"])

    # readable report
    L = [f"Dataset audit: {DATASET}", f"Subsets included: {', '.join(subsets)}",
         f"Videos: {len(rows)}   unreadable: {sum(1 for r in rows if r['error'])}", ""]
    hdr = f"{'group':44s} {'videos':>7s} {'fps(med)':>9s} {'frames(med)':>12s} {'<' + str(MIN_FRAMES) + 'fr':>7s} {'dur s(med)':>11s}"
    for level, title in (("subset", "PER SUBSET"), ("folder", "PER FOLDER")):
        L += [title, hdr]
        for s in summ:
            if s["level"] == level:
                L.append(f"{s['group'][:44]:44s} {s['videos']:>7d} {str(s['fps_median']):>9s} "
                         f"{str(s['frames_median']):>12s} {s[f'videos_under_{MIN_FRAMES}_frames']:>7d} "
                         f"{str(s['duration_median_s']):>11s}")
        L.append("")
    xg = {e["dup_group"] for e in exact}
    xcross = {g for g in xg if len({e["folder"] for e in exact if e["dup_group"] == g}) > 1}
    L += ["DUPLICATES",
          f"Exact duplicate files (identical MD5): {len(xg)} groups, {len(xcross)} spanning different folders"
          + ("" if DO_MD5 else " (MD5 was switched off)"),
          f"Same file name in more than one folder: {len({n['name'] for n in names})} names",
          f"Near-duplicate pairs (<= {HAMMING_BITS} of 192 bits): {len(near)} "
          f"({sum(1 for p in near if not p['same_folder'])} between different folders)"]
    for (ga, gb), c in Counter(tuple(sorted((p["folder_a"], p["folder_b"]))) for p in near).most_common(25):
        L.append(f"   {ga}  <->  {gb}: {c} pairs")
    L += ["", "Near-duplicate pairs are candidates: open a few before reporting them.",
          "Static or near-black clips can match by chance."]
    (out / "REPORT.txt").write_text("\n".join(L))
    print("\n".join(L))

    # one zip to download
    zip_path = shutil.make_archive(str(Path(OUT_ROOT) / f"{DATASET}_audit"), "zip", root_dir=out)
    print(f"\nEverything saved in {out}\nDownload this one file: {zip_path}")


print("Functions loaded.")


# %% ------------------------------------------------ CELL 3 (optional): DOWNLOAD SUBSET
# Downloads only the files matching HF_PATTERN into HF_LOCAL_DIR.
if HF_PATTERN:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                      allow_patterns=[HF_PATTERN], local_dir=HF_LOCAL_DIR)
    print("Downloaded to", HF_LOCAL_DIR)


# %% ------------------------------------------------------- CELL 4: AUDIT THIS SUBSET
subset_rows = audit_subset()


# %% ------------------------------------------- CELL 5 (optional): DELETE THIS SUBSET
# Only run after cell 4 has printed "Saved ...". The results stay in OUT_ROOT.
# shutil.rmtree(SUBSET_DIR); print("Deleted", SUBSET_DIR)


# %% ------------------------------------------- CELL 6: FINAL REPORT (after all subsets)
# final_report()
