"""Dataset audit: exact counts plus the metadata the provenance control needs.

Run this before writing anything into the manuscript. Every number in the
dataset table should come from your own copy, not from a paper's description
of it -- papers report the subset they used, which is rarely the release.

    python -m src.dataset_audit
    python -m src.dataset_audit --roots /data/GVD /data/GVF --sample 50

The resolution, codec, frame-rate and duration columns are not decoration.
If they differ systematically between the real and generated groups, then a
cross-encoder discrepancy could reflect acquisition rather than synthesis,
and the matched-provenance control becomes mandatory rather than optional.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".gif", ".m4v"}


def detect_duplicate_frames(path: pathlib.Path, n: int = 32) -> float:
    """Fraction of consecutive frames that are byte-identical.

    WHY THIS EXISTS. STALL (CVPR 2026, Appendix A.4) documents that the
    MSR-VTT real videos distributed with GenVideo are stored at 3 FPS and
    were upsampled to 8 FPS by DUPLICATING FRAMES -- applied only to the real
    videos, since generated ones were already at 8 FPS or higher. That single
    artifact moved D3's mean AP from 0.74 to 0.97.

    It is a direct hazard here. Every measure in this pipeline reads the
    frame-similarity structure, and duplicated frames produce identical
    embeddings. Real videos would show artificially high internal similarity
    and the measures would register frame duplication as cross-encoder
    agreement.

    A non-zero value on real videos and zero on generated ones is the
    signature. If you see it, obtain high-frame-rate sources and DOWNSAMPLE;
    never upsample by duplication.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        prev, dup, total = None, 0, 0
        while total < n:
            ok, fr = cap.read()
            if not ok:
                break
            if prev is not None:
                total += 1
                if fr.shape == prev.shape and (fr == prev).all():
                    dup += 1
            prev = fr
        cap.release()
        return round(dup / total, 3) if total else 0.0
    except Exception:
        return float("nan")


def probe(path: pathlib.Path) -> dict:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=20)
        data = json.loads(r.stdout)
        st = next(s for s in data["streams"] if s["codec_type"] == "video")
        num, den = (st.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
        return {"w": st.get("width"), "h": st.get("height"),
                "codec": st.get("codec_name"),
                "fps": round(int(num) / max(int(den), 1), 2),
                "dur": round(float(data.get("format", {}).get("duration", 0)), 2),
                "nb_frames": int(st.get("nb_frames", 0) or 0),
                "dup_frac": detect_duplicate_frames(path)}
    except Exception:
        return {}


def audit(root: pathlib.Path, sample: int = 25) -> list[dict]:
    groups: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for f in root.rglob("*"):
        if f.suffix.lower() in VIDEO_EXTS:
            groups[f.parent.relative_to(root).as_posix() or "."].append(f)

    rows = []
    for g in sorted(groups):
        files = groups[g]
        meta = [m for m in (probe(f) for f in files[:sample]) if m]
        res = collections.Counter(f"{m['w']}x{m['h']}" for m in meta if m.get("w"))
        cod = collections.Counter(m["codec"] for m in meta if m.get("codec"))
        fps = collections.Counter(m["fps"] for m in meta if m.get("fps"))
        dur = [m["dur"] for m in meta if m.get("dur")]
        nfr = [m["nb_frames"] for m in meta if m.get("nb_frames")]
        dup = [m["dup_frac"] for m in meta if m.get("dup_frac") == m.get("dup_frac")]
        rows.append({
            "group": g, "n": len(files), "sampled": len(meta),
            "resolution": res.most_common(1)[0][0] if res else "?",
            "res_variants": len(res),
            "codec": cod.most_common(1)[0][0] if cod else "?",
            "fps": fps.most_common(1)[0][0] if fps else "?",
            "mean_dur_s": round(sum(dur) / len(dur), 2) if dur else 0.0,
            "min_frames": min(nfr) if nfr else 0,
            "short_clips": sum(1 for n in nfr if n < 32),
            "dup_frac": round(sum(dup) / len(dup), 3) if dup else 0.0,
        })
    return rows


def report(rows, name):
    print(f"\n{'='*100}\n{name}\n{'='*100}")
    hdr = (f"{'group':<34s} {'n':>6s} {'resolution':>12s} {'var':>4s} "
           f"{'codec':>7s} {'fps':>7s} {'dur(s)':>7s} {'<32fr':>6s} {'dup':>6s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['group']:<34s} {r['n']:>6d} {r['resolution']:>12s} "
              f"{r['res_variants']:>4d} {r['codec']:>7s} {str(r['fps']):>7s} "
              f"{r['mean_dur_s']:>7.1f} {r['short_clips']:>6d} "
              f"{r['dup_frac']:>6.3f}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<34s} {sum(r['n'] for r in rows):>6d}")

    # the provenance warning, computed rather than assumed
    res = {r["resolution"] for r in rows if r["resolution"] != "?"}
    cod = {r["codec"] for r in rows if r["codec"] != "?"}
    fps = {str(r["fps"]) for r in rows if r["fps"] != "?"}
    print()
    if len(res) > 1 or len(cod) > 1 or len(fps) > 1:
        print("PROVENANCE WARNING")
        print(f"  {len(res)} distinct dominant resolutions: {sorted(res)}")
        print(f"  {len(cod)} distinct codecs: {sorted(cod)}")
        print(f"  {len(fps)} distinct frame rates: {sorted(fps)}")
        print("  If these differ systematically between real and generated")
        print("  groups, a measured discrepancy could reflect acquisition")
        print("  rather than synthesis. The matched-provenance control is")
        print("  then mandatory, and belongs in the Method, not the")
        print("  Limitations.")
    else:
        print("Encoding parameters look uniform across groups. Still run the")
        print("matched-provenance control, but the risk is lower.")

    worst = max((r["dup_frac"] for r in rows), default=0.0)
    if worst > 0.05:
        print("\nDUPLICATE-FRAME WARNING")
        for r in rows:
            if r["dup_frac"] > 0.05:
                print(f"  {r['group']:<34s} {r['dup_frac']:.1%} of consecutive "
                      f"frames identical")
        print("  Frames appear to have been upsampled by duplication. STALL")
        print("  (CVPR 2026, App. A.4) shows this artifact moved D3's mean AP")
        print("  from 0.74 to 0.97 on GenVideo, because it applied only to the")
        print("  real videos. Every measure here reads frame-similarity")
        print("  structure, so duplication would be registered as agreement.")
        print("  Obtain high-frame-rate sources and DOWNSAMPLE. Do not proceed")
        print("  with extraction until this is resolved.")

    short = sum(r["short_clips"] for r in rows)
    if short:
        print(f"\n{short} sampled clips have fewer than 32 frames. These are")
        print("extended by cyclic repetition, never dropped -- clip length")
        print("plausibly correlates with generator, so filtering on it would")
        print("condition the sample on the label. Report the count.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pp = sub.add_parser("purity", help="check folder purity before "
                        "trusting a real-vs-real comparison")
    pp.add_argument("roots", nargs="+")

    ap.add_argument("--roots", nargs="+", default=["/data/FakeParts"])
    ap.add_argument("--sample", type=int, default=25,
                    help="videos per group to probe with ffprobe")
    ap.add_argument("--csv", default="results/dataset_audit.csv")
    a = ap.parse_args()
    if a.cmd == "purity":
        cmd_folder_purity(a.roots)
        raise SystemExit(0)

    allrows = []
    for r in a.roots:
        root = pathlib.Path(r)
        if not root.exists():
            print(f"{root}: not found"); continue
        rows = audit(root, a.sample)
        report(rows, root.name)
        for row in rows:
            row["benchmark"] = root.name
        allrows += rows

    if allrows:
        import csv
        out = pathlib.Path(a.csv); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
            w.writeheader(); w.writerows(allrows)
        print(f"\n-> {out}")
        print("These counts go straight into Table 2. Do not use numbers")
        print("quoted from any paper's description of the dataset.")


# ==========================================================================
# FOLDER PURITY CHECK
#
# The real-vs-real provenance control assumes each folder is a single,
# distinct source. That assumption has not been verified for FakeParts' five
# real-video folders, and it matters: if a folder secretly mixes sources, a
# "folder A vs folder B" comparison becomes "mixture vs mixture", which
# dilutes the real provenance signal and makes the bound look smaller than
# it actually is -- the dangerous direction of error, since it would let a
# real confound through unflagged.
#
# Two cheap, independent checks, neither requiring a single video to be
# opened beyond ffprobe:
#
#   1. FILENAME PATTERN. Source datasets have distinct ID conventions --
#      DAVIS uses short descriptive names ("bear", "camel"), YouTube-VOS
#      uses hex video IDs, etc. A folder with one consistent naming pattern
#      is evidence of a single source; a folder mixing patterns is evidence
#      of contamination.
#
#   2. ENCODING SIGNATURE. Real-world source datasets differ in resolution,
#      frame rate, and typical duration, often quite sharply, because they
#      were captured or released under different pipelines. A folder that
#      clusters tightly on these axes is probably pure; a folder with two or
#      more distinct clusters is probably mixed.
#
# Neither check PROVES purity -- a source could coincidentally share another
# source's naming convention and encoding signature. But if both checks agree
# a folder looks homogeneous, that is meaningfully more trustworthy than
# assuming purity from folder structure alone.
# ==========================================================================

def _filename_signature(name: str) -> str:
    """Classify a filename into a coarse pattern bucket."""
    stem = pathlib.Path(name).stem
    if re.fullmatch(r"[0-9a-f]{8,}", stem):
        return "hex-id"                       # YouTube-VOS style
    if re.fullmatch(r"[0-9]+", stem):
        return "numeric-id"
    if re.fullmatch(r"[a-z]+(-[0-9]+)?", stem):
        return "short-word"                   # DAVIS style ("bear", "camel-2")
    if "_" in stem and any(c.isdigit() for c in stem):
        return "underscore-mixed"
    return "other"


def check_folder_purity(root: pathlib.Path, sample: int = 200) -> dict:
    """Run both purity checks on one folder. Returns a dict, not a verdict --
    read the printed report and decide; this does not auto-exclude anything.
    """
    files = sorted(root.glob("*"))
    files = [f for f in files if f.suffix.lower() in VIDEO_EXTS]
    if not files:
        return {"folder": root.name, "n": 0}

    # 1. filename pattern distribution
    sig_counts = collections.Counter(_filename_signature(f.name) for f in files)
    dominant_frac = sig_counts.most_common(1)[0][1] / len(files)

    # 2. encoding signature clustering (resolution x fps, coarse)
    probed = [probe(f) for f in files[:sample]]
    probed = [p for p in probed if p]
    combos = collections.Counter(
        (p.get("w"), p.get("h"), p.get("fps")) for p in probed)
    top_combo_frac = (combos.most_common(1)[0][1] / len(probed)) if probed else 0.0

    return {
        "folder": root.name, "n": len(files),
        "filename_patterns": dict(sig_counts),
        "filename_dominant_fraction": round(dominant_frac, 3),
        "n_distinct_encoding_signatures": len(combos),
        "encoding_dominant_fraction": round(top_combo_frac, 3),
        "verdict": (
            "likely pure" if dominant_frac > 0.9 and top_combo_frac > 0.7 else
            "likely mixed" if dominant_frac < 0.6 or len(combos) > 4 else
            "ambiguous -- inspect manually"
        ),
    }


def cmd_folder_purity(roots: list[str]):
    """python -m src.dataset_audit purity /data/FakeParts/real/folder1 ..."""
    print(f"\n{'='*100}\nFOLDER PURITY CHECK\n{'='*100}")
    print("Neither check proves purity. Read the breakdown, not just the verdict.\n")
    for r in roots:
        res = check_folder_purity(pathlib.Path(r))
        print(f"\n{res['folder']}  (n={res.get('n', 0)})")
        if res.get("n", 0) == 0:
            print("  [no video files found]")
            continue
        print(f"  filename patterns : {res['filename_patterns']}")
        print(f"  dominant pattern  : {res['filename_dominant_fraction']:.1%} of files")
        print(f"  encoding clusters : {res['n_distinct_encoding_signatures']} distinct "
              f"(w,h,fps) combinations, top one covers "
              f"{res['encoding_dominant_fraction']:.1%}")
        print(f"  VERDICT: {res['verdict']}")
    print(f"\n{'='*100}")
    print("If any folder comes back 'likely mixed' or 'ambiguous', do not use that")
    print("folder in a real-vs-real pairwise comparison until resolved. Two folders")
    print("both classified 'likely pure' but with the SAME dominant filename pattern")
    print("and SAME dominant encoding signature may be the same underlying source")
    print("split in two -- that pair tells you nothing about provenance either.")
