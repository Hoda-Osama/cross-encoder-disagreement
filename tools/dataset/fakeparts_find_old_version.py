# =============================================================================
# fakeparts_find_old_version.py  (Kaggle, CPU only, downloads no videos)
#
# Hugging Face keeps the full history of a dataset. This script:
#   Cell A  lists every commit of hi-paris/FakeParts with its date
#   Cell B  counts the video files of ONE chosen commit, per folder, without
#           downloading them, so you can find the version that matches your
#           numbers (12,075 real in 5 folders: 4,209 / 2,156 / 2,156 / 2,149 /
#           1,405, and 64,253 generated)
#   Cell C  (optional) checks your own video ids against that commit
# =============================================================================

# %% Cell A: list commits ----------------------------------------------------
from huggingface_hub import HfApi
import pandas as pd

REPO = "hi-paris/FakeParts"
api = HfApi()
commits = api.list_repo_commits(REPO, repo_type="dataset")
log = pd.DataFrame([{"date": c.created_at, "commit": c.commit_id, "title": c.title} for c in commits])
log = log.sort_values("date").reset_index(drop=True)
pd.set_option("display.max_rows", 500, "display.max_colwidth", 80)
log.to_csv("/kaggle/working/fakeparts_commits.csv", index=False)
print(log.to_string())
# Look for the dates around when you downloaded the data.


# %% Cell B: count files in one commit ----------------------------------------
REVISION = "PUT_COMMIT_HASH_HERE"     # from Cell A, e.g. the last commit before your download
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif")

from collections import Counter
files = [f.path for f in api.list_repo_tree(REPO, repo_type="dataset", revision=REVISION,
                                             recursive=True)
         if f.path.lower().endswith(VIDEO_EXTS)]
by_top = Counter(p.split("/")[0] for p in files)
by_two = Counter("/".join(p.split("/")[:2]) for p in files)
by_three = Counter("/".join(p.split("/")[:3]) for p in files if p.count("/") >= 2)
print(f"Commit {REVISION}: {len(files)} video files")
print("\nTop level:", dict(by_top))
print("\nSecond level:"); [print(f"  {k:60s} {v:>7d}") for k, v in sorted(by_two.items())]
print("\nThird level:");  [print(f"  {k:60s} {v:>7d}") for k, v in sorted(by_three.items())]
pd.DataFrame({"path": files}).to_csv(f"/kaggle/working/fakeparts_files_{REVISION[:8]}.csv", index=False)


# %% Cell C (optional): match your own videos --------------------------------
# Any CSV with one row per video you used; set the column holding the file
# name or path (e.g. the recount's per_video_all.csv, column "relpath", or the
# video list saved with your extracted features).
MY_CSV, MY_COL = None, "relpath"
if MY_CSV:
    import os
    mine = pd.read_csv(MY_CSV)[MY_COL].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0])
    theirs = pd.Series(files).map(lambda s: os.path.splitext(os.path.basename(s))[0])
    hit = mine.isin(set(theirs))
    print(f"Your videos found in commit {REVISION[:8]}: {hit.sum()} of {len(mine)} ({100 * hit.mean():.1f}%)")
