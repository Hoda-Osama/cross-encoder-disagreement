# =============================================================================
# fakeparts_manifest_summary.py  (one Kaggle cell, CPU is enough, ~70 MB download)
#
# Reads the official FakeParts manifests on Hugging Face (one JSON line per
# video, with fps, duration, frame count, source dataset and source pairing)
# WITHOUT downloading any video, and writes:
#   fakeparts_manifest_counts.csv    videos, fps, duration, frames per task/method
#   fakeparts_manifest_sources.csv   source_dataset and pairing status per task
#   fakeparts_manifest_all.csv       one row per video (to match your own files)
#   REPORT_manifests.txt             readable summary
#
# The repository is still being updated, so the script records the exact
# commit it read. Put that commit hash in the paper as the dataset version.
#
# Optional: set MY_FILES_CSV to the per_video_all.csv from the recount notebook
# to see which of YOUR downloaded videos appear in the current release.
# =============================================================================
REPO          = "hi-paris/FakeParts"
REVISION      = None            # None = latest; or a commit hash to pin a version
OUT_DIR       = "/kaggle/working/fakeparts_manifests"
MY_FILES_CSV  = None            # e.g. "/kaggle/working/audit/FakeParts/per_video_all.csv"
MIN_FRAMES    = 32

import json, os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

api = HfApi()
info = api.dataset_info(REPO, revision=REVISION)
commit = info.sha
local = snapshot_download(REPO, repo_type="dataset", revision=commit,
                          allow_patterns=["manifests/*.jsonl"], local_dir=OUT_DIR)

rows = []
for p in sorted(Path(local, "manifests").glob("*.jsonl")):
    with open(p) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["manifest"] = p.stem
                rows.append(r)
df = pd.DataFrame(rows)
for c in ("fps", "duration_seconds", "frame_count"):
    df[c] = pd.to_numeric(df.get(c), errors="coerce")
df["stem"] = df["file_name"].astype(str).str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
df.drop(columns=[c for c in ("evidence", "remote_candidates", "uncertainties") if c in df],
        errors="ignore").to_csv(f"{OUT_DIR}/fakeparts_manifest_all.csv", index=False)

def q(s, p): return round(s.quantile(p), 2) if s.notna().any() else None
g = df.groupby(["label", "task", "method"])
counts = g.agg(videos=("sample_id", "size"),
               fps_min=("fps", "min"), fps_median=("fps", "median"), fps_max=("fps", "max"),
               dur_median=("duration_seconds", "median"),
               dur_min=("duration_seconds", "min"), dur_max=("duration_seconds", "max"),
               frames_median=("frame_count", "median"),
               under_1s=("duration_seconds", lambda s: int((s < 1).sum())),
               fps_below_8=("fps", lambda s: int((s < 8).sum())),
               frames_under_32=("frame_count", lambda s: int((s < MIN_FRAMES).sum()))
               ).reset_index()
counts.to_csv(f"{OUT_DIR}/fakeparts_manifest_counts.csv", index=False)

src = (df.assign(source_dataset=df.get("source_dataset").fillna("(not given)"),
                 pairing_status=df.get("pairing_status").fillna("(not given)"),
                 has_source_video=df.get("source_video").notna())
         .groupby(["label", "task", "method", "source_dataset", "pairing_status", "has_source_video"])
         .size().reset_index(name="videos"))
src.to_csv(f"{OUT_DIR}/fakeparts_manifest_sources.csv", index=False)

dupe_ids = df[df.duplicated("sample_id", keep=False)]
L = [f"FakeParts manifests, repository {REPO}, commit {commit}",
     f"(last modified {info.last_modified})", "",
     f"Total videos: {len(df)}   real: {(df.label == 'real').sum()}   fake: {(df.label == 'fake').sum()}", "",
     "PER TASK", df.groupby(["label", "task"]).size().to_string(), "",
     "PER TASK / METHOD (fps, duration, padding risk)", counts.to_string(index=False), "",
     "SOURCES AND PAIRING", src.to_string(index=False), "",
     f"Duplicate sample_id rows: {len(dupe_ids)}",
     "Methods that may be the same generator under two names (check overlap):"]
for a, b in [("allegro", "allegroai"), ("cogvideox", "cogvidx"), ("hunyuanvid", "hunyuanvideo"),
             ("mochi", "mochi1")]:
    sa = set(df[(df.task == "t2v") & (df.method == a)].stem)
    sb = set(df[(df.task == "t2v") & (df.method == b)].stem)
    if sa or sb:
        L.append(f"   t2v {a} ({len(sa)}) vs {b} ({len(sb)}): {len(sa & sb)} shared file names")

if MY_FILES_CSV and os.path.exists(MY_FILES_CSV):
    mine = pd.read_csv(MY_FILES_CSV)
    mine["stem"] = mine["name"].astype(str)
    m = mine.merge(df[["stem", "label", "task", "method", "fps", "duration_seconds", "source_dataset"]],
                   on="stem", how="left", indicator=True)
    L += ["", "YOUR DOWNLOADED FILES vs CURRENT RELEASE (matched by file name)",
          m.groupby(["folder", "_merge"]).size().unstack(fill_value=0).to_string()]
    m.to_csv(f"{OUT_DIR}/my_files_matched.csv", index=False)

Path(OUT_DIR, "REPORT_manifests.txt").write_text("\n".join(L))
print("\n".join(L))
