"""GVF assembly: match generated videos to their real counterparts.

GVF ships prompts and generated videos but not real videos. The real ones are
pulled from MSVD and MSR-VTT and matched by `video_id`, which is what makes
the benchmark prompt-aligned.

    # 1. generated videos from the project's Drive link -> /data/GVF/generated
    # 2. real videos from Hugging Face:
    #      hf download friedrichor/MSVD    --repo-type dataset --local-dir /data/MSVD
    #      hf download friedrichor/MSR-VTT --repo-type dataset --local-dir /data/MSRVTT
    # 3. then:
    python -m src.build_gvf --prompts /data/GVF/datas/prompts.jsonl \\
                            --generated /data/GVF/generated \\
                            --msvd /data/MSVD --msrvtt /data/MSRVTT \\
                            --out /data/GVF_assembled

Two things this does beyond copying files.

It keeps the prompt metadata on every row. GVF labels each prompt with its
content categories, the kind of temporal change it involves, and its
complexity. That metadata is free stratification: the `temporal` field
distinguishes actions from kinetic motions, fluid motions and light change,
which is a dataset-provided proxy for how much the video moves and therefore
a direct check on the movement confound.

It tags MSVD and MSR-VTT separately. They are two distinct real corpora, so
the real-versus-real provenance control can run on GVF alone -- no second
benchmark needed to bound how much of the effect is corpus identity.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from collections import Counter, defaultdict

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}


def index_videos(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Map bare filename stem -> path, for every video under root."""
    out = {}
    for f in root.rglob("*"):
        if f.suffix.lower() in VIDEO_EXTS:
            out.setdefault(f.stem, f)
    return out


def load_prompts(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(args):
    prompts = load_prompts(pathlib.Path(args.prompts))
    print(f"{len(prompts)} prompts")

    real_idx = {}
    for name, root in (("MSVD", args.msvd), ("MSRVTT", args.msrvtt)):
        if root:
            idx = index_videos(pathlib.Path(root))
            print(f"  {name:8s} {len(idx):>6d} videos indexed")
            for k, v in idx.items():
                real_idx[k] = (name, v)

    gen_idx = index_videos(pathlib.Path(args.generated))
    print(f"  generated {len(gen_idx):>6d} videos indexed")

    # Generated files are usually <generator>/<video_id>.mp4. Group by the
    # parent directory, which is the generator name.
    by_generator = defaultdict(dict)
    for f in pathlib.Path(args.generated).rglob("*"):
        if f.suffix.lower() in VIDEO_EXTS:
            by_generator[f.parent.name][f.stem] = f
    print(f"  {len(by_generator)} generators: {sorted(by_generator)}")

    out = pathlib.Path(args.out)
    manifest, missing_real, missing_gen = [], Counter(), Counter()

    for p in prompts:
        vid = p["video_id"]
        meta = {
            "prompt": p.get("prompt"),
            "prompt_source": p.get("source"),
            "complexity": p.get("prompt complexity"),
            # dataset-provided movement proxy -- see module docstring
            "temporal": (p.get("major content") or {}).get("temporal"),
            "spatial": (p.get("major content") or {}).get("spatial"),
            "attr_spatial": (p.get("attribute control") or {}).get("spatial"),
            "attr_temporal": (p.get("attribute control") or {}).get("temporal"),
        }

        # real counterpart
        if vid in real_idx:
            corpus, src = real_idx[vid]
            dst = out / "real" / corpus / f"{vid}{src.suffix}"
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    (shutil.copy2 if args.copy else _link)(src, dst)
            manifest.append({**meta, "video_id": vid, "label": 1,
                             "generator": f"real:{corpus}", "path": str(dst)})
        else:
            missing_real[p.get("source", "?")] += 1

        # generated counterparts, one per generator
        found = False
        for gen, idx in by_generator.items():
            if vid in idx:
                found = True
                src = idx[vid]
                dst = out / "generated" / gen / f"{vid}{src.suffix}"
                if not args.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        (shutil.copy2 if args.copy else _link)(src, dst)
                manifest.append({**meta, "video_id": vid, "label": 0,
                                 "generator": gen, "path": str(dst)})
        if not found:
            missing_gen[p.get("source", "?")] += 1

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "gvf_manifest.jsonl", "w") as fh:
            for r in manifest:
                fh.write(json.dumps(r) + "\n")

    # ------------------------------------------------------------------
    print(f"\nmatched rows: {len(manifest)}")
    n_real = sum(r["label"] == 1 for r in manifest)
    print(f"  real      {n_real}")
    print(f"  generated {len(manifest) - n_real}")

    print("\nper generator:")
    for g, n in sorted(Counter(r["generator"] for r in manifest).items()):
        print(f"  {g:<24s} {n:>6d}")

    if missing_real:
        print(f"\nprompts with no real video: {sum(missing_real.values())}")
        for k, v in missing_real.items():
            print(f"  from {k}: {v}")
        print("  Real videos are YouTube-sourced and some are no longer")
        print("  retrievable. Report the count and the resulting real/generated")
        print("  imbalance rather than leaving it implicit.")
    if missing_gen:
        print(f"\nprompts with no generated video: {sum(missing_gen.values())}")

    # the free stratification
    print("\ntemporal categories (dataset-provided movement proxy):")
    cats = Counter(t for r in manifest for t in (r.get("temporal") or []))
    for c, n in cats.most_common():
        print(f"  {c:<20s} {n:>6d}")
    print("\nUse these to stratify effect sizes. If the separation holds for")
    print("'actions' and 'light change' alike, movement does not explain it,")
    print("and the check costs nothing because the labels already exist.")


def _link(src: pathlib.Path, dst: pathlib.Path):
    """Hard-link rather than copy. Same bytes, no extra disk."""
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="GVF datas/*.jsonl")
    ap.add_argument("--generated", required=True, help="unpacked Drive archive")
    ap.add_argument("--msvd", default=None)
    ap.add_argument("--msrvtt", default=None)
    ap.add_argument("--out", default="/data/GVF_assembled")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hard-linking (needs 2x disk)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report matches without writing anything")
    build(ap.parse_args())
