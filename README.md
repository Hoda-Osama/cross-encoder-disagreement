# Cross-Encoder Representation Disagreement for AI-Generated Video

Code for the paper *"Characterizing AI-Generated Video through Cross-Encoder
Representation Disagreement"* (H. O. Elkhodary, S. M. Youssef, M. Elshenawy,
D. Sobhy; AASTMT).

The method measures, for each video, how much two independently pretrained and
frozen visual encoders agree when they describe the same frames. No
real-versus-generated classifier is trained: three of the four measures have no
fitted parameters, and the ridge residual is calibrated on real videos only.

## Repository layout

```
configs/                YAML settings: datasets, encoders, measures, defaults
src/
  config.py             paths and run settings
  manifest.py           dataset indexing and resumable status tracking
  decode.py             frame sampling (T = 32 evenly spaced frames)
  encoders/             the seven frozen encoders behind one interface
                        (CLIP, DINOv2, VideoMAE, RAFT, MiDaS,
                         Depth Anything V2, UniDepth V2)
  extract.py            feature extraction driver (resumable)
  measures.py           debiased CKA, neighbourhood alignment, CKNNA,
                        ridge residual (CPU only)
  baselines.py          within-encoder statistics
  analyze.py            effect sizes, AUROC, recall at fixed FPR, bootstrap
                        intervals, role permutation test, interventions,
                        sensitivity analyses
  dataset_audit.py      dataset checks
  figures.py, diagrams.py, pipeline_*.py   figures
tools/
  baselines/            Kaggle notebooks running SPLIT, STALL and D3 with
                        their official code and default protocols
  dataset/              dataset recount and duplicate check, FakeParts
                        manifest summary, FakeParts version lookup
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Set the dataset locations in `src/config.py`, then:

```bash
python -m src.manifest build          # index the videos
python -m src.extract                 # extract features (GPU, resumable)
python -m src.analyze grid            # all encoder pairs and measures
python -m src.analyze intervention    # correspondence-disruption controls
```

## Datasets

- **FakeParts** — https://huggingface.co/datasets/hi-paris/FakeParts
  (the release is versioned; the revision used is given in the paper)
- **ViF-Bench** — https://huggingface.co/datasets/JoeLeelyf/ViF-Bench

Videos are not redistributed here; follow each dataset's licence.

## Baselines

`tools/baselines/` runs each baseline from its official repository at a
pinned commit, with default settings:

| Method | Official code |
|---|---|
| SPLIT | https://github.com/mldljyh/SPLIT |
| STALL | https://github.com/OmerBenHayun/STALL |
| D3 | https://github.com/Zig-HS/D3 |

Videos that a method's protocol cannot process are recorded as excluded, not
scored under modified settings.

## Citation

Citation details will be added on publication.
