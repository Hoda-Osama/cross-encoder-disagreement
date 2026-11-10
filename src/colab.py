"""Colab helpers: keep the feature cache alive across session restarts.

THE PROBLEM THIS SOLVES
-----------------------
Extraction takes many hours and Colab disconnects long before it finishes.
The pipeline is already resumable at video granularity, so restarts are
survivable -- but only if the cache and manifest outlive the session.

The obvious fix, writing the cache directly to Google Drive, does not work
well. The cache is roughly 15,000 videos x 7 encoders = 100,000 small .npy
files, and Drive's FUSE mount is extremely slow for many small files. A run
that takes hours on local disk can take days through Drive.

So: write to Colab's LOCAL disk, which is fast, and periodically snapshot the
whole cache to Drive as a single tar. One large file is exactly what Drive
handles well. On restart, restore once and continue.

    from src.colab import restore, snapshot, autosnapshot

    restore()                 # at the start of every session
    autosnapshot(minutes=20)  # background snapshots while extraction runs
    snapshot()                # once more before the session ends
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from . import config as C

DRIVE = Path("/content/drive/MyDrive/paper2")
ARCHIVE = DRIVE / "cache.tar"
RESULTS_MIRROR = DRIVE / "results"


def _drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").exists()


def mount() -> bool:
    """Mount Drive if we are on Colab and it is not mounted yet."""
    if _drive_mounted():
        return True
    try:
        from google.colab import drive          # type: ignore
        drive.mount("/content/drive")
        DRIVE.mkdir(parents=True, exist_ok=True)
        return True
    except ImportError:
        print("not running on Colab; snapshots disabled")
        return False


def snapshot(quiet: bool = False) -> None:
    """Tar the cache and manifest to Drive as one file.

    Written to a temporary name and moved into place, so an interrupted
    snapshot cannot destroy the previous good one.
    """
    if not mount():
        return
    DRIVE.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE.with_suffix(".tmp.tar")
    t0 = time.time()
    with tarfile.open(tmp, "w") as tar:
        if C.FEATURE_DIR.exists():
            tar.add(C.FEATURE_DIR, arcname="features")
        if C.MANIFEST_PATH.exists():
            tar.add(C.MANIFEST_PATH, arcname="manifest.jsonl")
    shutil.move(str(tmp), str(ARCHIVE))

    # results are small; mirror them uncompressed so they are readable in Drive
    if Path(C.RESULTS_DIR).exists():
        RESULTS_MIRROR.mkdir(parents=True, exist_ok=True)
        for f in Path(C.RESULTS_DIR).glob("*.csv"):
            shutil.copy2(f, RESULTS_MIRROR / f.name)

    if not quiet:
        mb = ARCHIVE.stat().st_size / 2**20
        print(f"snapshot -> {ARCHIVE}  ({mb:.0f} MB, {time.time()-t0:.0f}s)")


def restore() -> bool:
    """Restore the cache and manifest from the Drive snapshot."""
    if not mount() or not ARCHIVE.exists():
        print("no snapshot found; starting fresh")
        return False
    t0 = time.time()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r") as tar:
        tar.extractall(C.CACHE_DIR)
    src = C.CACHE_DIR / "manifest.jsonl"
    if src.exists() and src != C.MANIFEST_PATH:
        shutil.move(str(src), str(C.MANIFEST_PATH))

    n = sum(1 for _ in C.FEATURE_DIR.rglob("*.npy")) if C.FEATURE_DIR.exists() else 0
    print(f"restored {n} cached features in {time.time()-t0:.0f}s")
    return True


def autosnapshot(minutes: int = 20) -> threading.Thread:
    """Snapshot on a timer in the background while extraction runs.

    Daemon thread, so it dies with the session rather than blocking exit.
    Twenty minutes is a reasonable default: frequent enough that a
    disconnection costs little, rare enough that the tar overhead is
    negligible.
    """
    def loop():
        while True:
            time.sleep(minutes * 60)
            try:
                snapshot(quiet=True)
                print(f"[autosnapshot {time.strftime('%H:%M')}]")
            except Exception as exc:
                print(f"[autosnapshot failed: {exc}]")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"autosnapshot every {minutes} min")
    return t


def stage_dataset(drive_archive: str, dest: str = "/content/data") -> Path:
    """Copy a dataset archive from Drive to local disk and extract it.

    Decoding video directly from the Drive mount is slow enough to dominate
    extraction time. Copying the archive once per session and working from
    local disk is much faster, even counting the copy.
    """
    src = Path(drive_archive)
    if not src.exists():
        raise FileNotFoundError(src)
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    print(f"staging {src.name} ({src.stat().st_size / 2**30:.1f} GB) -> {d}")
    t0 = time.time()
    if src.suffix in (".zip",):
        subprocess.run(["unzip", "-q", "-o", str(src), "-d", str(d)], check=True)
    else:
        subprocess.run(["tar", "-xf", str(src), "-C", str(d)], check=True)
    print(f"staged in {time.time()-t0:.0f}s")
    return d


def gpu_info() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print("no GPU -- set Runtime > Change runtime type > GPU")
            return
        p = torch.cuda.get_device_properties(0)
        print(f"{p.name}, {p.total_memory / 2**30:.1f} GiB")
        if p.total_memory / 2**30 < 12:
            print("  under 12 GiB: consider FRAME_CHUNK=2 in config.py")
    except ImportError:
        print("torch not installed")
