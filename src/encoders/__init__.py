"""Frozen encoders.

Every encoder implements the same contract:

    encode(frames: uint8 (T, H, W, 3)) -> float16 (T', d)

T' is usually T.  VideoMAE returns one vector per tubelet, so T' = T // 2.
RAFT returns one per consecutive frame pair, so T' = T - 1.  The measures
in measures.py resample to a common length when pairing encoders of
different temporal granularity, and record which side was resampled.

Nothing here is trained.  Nothing here is fine-tuned.  Every model is loaded
with pretrained weights, put in eval mode, and has requires_grad disabled.
That is the whole point of Paper 2.

All models are held resident simultaneously.  fp16 weights total roughly
1.7 GB across the seven encoders, which leaves about 6 GB of an 8 GB card
for activations.  Depth models run at 384 and are chunked over frames.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)


def to_tensor(frames: np.ndarray, size: int, mean, std, device) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,size,size) normalised float on device."""
    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - mean.to(device)) / std.to(device)


class Encoder(ABC):
    name: str
    out_dim: int
    input_size: int = 224

    def __init__(self, device: str = C.DEVICE):
        self.device = device
        self.model = None

    @abstractmethod
    def _build(self): ...

    def load(self):
        if self.model is None:
            self.model = self._build().to(self.device).eval().half()
            for p in self.model.parameters():
                p.requires_grad_(False)
        return self

    def unload(self):
        self.model = None
        torch.cuda.empty_cache()

    @abstractmethod
    def _forward(self, frames: np.ndarray) -> torch.Tensor: ...

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        out = self._forward(frames)
        return out.detach().float().cpu().numpy().astype(np.float16)

    def _chunked(self, x: torch.Tensor, fn, chunk: int = C.FRAME_CHUNK):
        """Apply fn over frame chunks. Keeps peak memory bounded on 8 GB."""
        return torch.cat([fn(x[i:i + chunk]) for i in range(0, len(x), chunk)], 0)


# ==========================================================================
# Appearance
# ==========================================================================
class CLIPB32(Encoder):
    """CLIP ViT-B/32 vision tower. Also used in Paper 1."""
    name, out_dim, input_size = "clip_b32", 512, 224

    def _build(self):
        from transformers import CLIPVisionModelWithProjection
        return CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32")

    def _forward(self, frames):
        x = to_tensor(frames, self.input_size, CLIP_MEAN, CLIP_STD, self.device).half()
        return self._chunked(x, lambda c: self.model(pixel_values=c).image_embeds, 16)


class DINOv2B14(Encoder):
    """DINOv2 ViT-B/14. Self-supervised, no language supervision -- the
    contrast with CLIP is the point."""
    name, out_dim, input_size = "dinov2_b14", 768, 224

    def _build(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained("facebook/dinov2-base")

    def _forward(self, frames):
        x = to_tensor(frames, self.input_size, IMAGENET_MEAN, IMAGENET_STD,
                      self.device).half()
        return self._chunked(
            x, lambda c: self.model(pixel_values=c).last_hidden_state[:, 0], 16)


# ==========================================================================
# Motion
# ==========================================================================
class VideoMAEB(Encoder):
    """VideoMAE-Base. Returns one vector per tubelet, so T' = T // 2."""
    name, out_dim, input_size = "videomae_b", 768, 224

    def _build(self):
        from transformers import VideoMAEModel
        return VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    def _forward(self, frames):
        x = to_tensor(frames, self.input_size, IMAGENET_MEAN, IMAGENET_STD,
                      self.device).half()
        L = C.VIDEOMAE_CLIP_LEN
        outs = []
        for s in range(0, len(x) - L + 1, L):
            clip = x[s:s + L].unsqueeze(0)                       # (1,L,3,H,W)
            h = self.model(pixel_values=clip).last_hidden_state  # (1,N,768)
            # N = (L/2) * (H/16) * (W/16); pool spatial tokens per tubelet
            n_t = L // 2
            h = h.squeeze(0).reshape(n_t, -1, self.out_dim).mean(1)
            outs.append(h)
        return torch.cat(outs, 0)


class RAFTSmall(Encoder):
    """RAFT optical flow, pooled to a per-pair descriptor. T' = T - 1.

    The odd one out on purpose: 5M parameters, no transformer, no semantic
    pretraining objective at all. If discrepancy separates the classes for
    CLIP-RAFT as well as for CLIP-VideoMAE, the generality claim is much
    harder to dismiss.
    """
    name, out_dim, input_size = "raft_small", 64, 224

    def _build(self):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        return raft_small(weights=Raft_Small_Weights.DEFAULT)

    def _forward(self, frames):
        # RAFT expects [-1, 1] and dimensions divisible by 8.
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float()
        x = F.interpolate(x / 255.0, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        x = (x * 2 - 1).half()

        feats = []
        for i in range(0, len(x) - 1, C.FRAME_CHUNK):
            j = min(i + C.FRAME_CHUNK, len(x) - 1)
            flow = self.model(x[i:j], x[i + 1:j + 1])[-1]        # (B,2,H,W)
            feats.append(self._pool_flow(flow))
        return torch.cat(feats, 0)

    def _pool_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """Spatial pyramid over flow magnitude and angle -> 64-d descriptor.

        A raw flow field is far too high-dimensional to compare against a
        512-d semantic embedding, and mean flow alone throws away all spatial
        structure. A coarse 4x4 grid of (mean magnitude, mean cos, mean sin,
        magnitude std) keeps layout while staying compact.
        """
        B = flow.shape[0]
        mag = flow.norm(dim=1, keepdim=True)                     # (B,1,H,W)
        ang = torch.atan2(flow[:, 1:2], flow[:, 0:1])
        stack = torch.cat([mag, torch.cos(ang), torch.sin(ang)], 1)
        pooled = F.adaptive_avg_pool2d(stack.float(), 4)         # (B,3,4,4)
        std = F.adaptive_avg_pool2d((mag.float() - mag.float().mean((2, 3),
                                     keepdim=True)) ** 2, 4).sqrt()
        return torch.cat([pooled.reshape(B, -1), std.reshape(B, -1)], 1).half()


# ==========================================================================
# Geometry
# ==========================================================================
class _DepthEncoder(Encoder):
    """Depth models, in one of two feature modes.

    FIX 1 -- feature mode.

    "backbone" (default): pooled ViT patch-token features from the depth
        model's ENCODER, not its predicted depth map. This is the better
        choice and it is what the manuscript describes.

        With the descriptor mode below, the study compares 512-d CLIP ViT
        features against an 80-d hand-designed summary of a predicted depth
        map. If appearance-geometry pairs then behave differently from
        appearance-motion pairs, the cause could be complementary pretraining
        priors, or it could simply be that one comparison is ViT-vs-ViT and
        the other is ViT-vs-statistic. Nothing in the design separates those,
        and H2 is precisely the hypothesis that comparison is meant to test.

        Backbone features make all three roles like-for-like, and they remove
        an unpublished, arbitrary descriptor from the method -- a reviewer
        would reasonably ask why a 4x4 grid and why those four statistics,
        and there is no good answer.

    "descriptor": the original hand-pooled 4x4 grid of the predicted depth
        map. Retained so the two modes can be compared and the choice
        justified empirically rather than by assertion.
    """
    input_size = 384
    mode = C.DEPTH_FEATURE_MODE
    descriptor_dim = 80   # 4x4 grid x (depth, dx, dy, curvature) + 16 var

    @property
    def out_dim(self):
        return self.backbone_dim if self.mode == "backbone" else self.descriptor_dim

    backbone_dim = 768    # overridden per model

    @staticmethod
    def _pool_tokens(h: torch.Tensor) -> torch.Tensor:
        """(B, N, D) patch tokens -> (B, D), mean over spatial positions.

        Same pooling VideoMAE receives, so the representations are
        comparable by construction.
        """
        return h.mean(1)

    def _descriptor(self, depth: torch.Tensor) -> torch.Tensor:
        """(B,H,W) -> (B,80) float16."""
        d = depth.float().unsqueeze(1)
        d = (d - d.amin((2, 3), keepdim=True)) / (
            d.amax((2, 3), keepdim=True) - d.amin((2, 3), keepdim=True) + 1e-6)
        gx = d[:, :, :, 1:] - d[:, :, :, :-1]
        gy = d[:, :, 1:, :] - d[:, :, :-1, :]
        lap = F.avg_pool2d(d, 3, 1, 1) - d
        parts = [
            F.adaptive_avg_pool2d(d, 4),
            F.adaptive_avg_pool2d(gx.abs(), 4),
            F.adaptive_avg_pool2d(gy.abs(), 4),
            F.adaptive_avg_pool2d(lap.abs(), 4),
        ]
        var = F.adaptive_avg_pool2d((d - d.mean((2, 3), keepdim=True)) ** 2, 4)
        B = d.shape[0]
        return torch.cat([p.reshape(B, -1) for p in parts]
                         + [var.reshape(B, -1)], 1).half()


class MiDaSHybrid(_DepthEncoder):
    """MiDaS DPT-Hybrid, relative depth. Also used in Paper 1."""
    name = "midas_hybrid"
    backbone_dim = 768

    def _build(self):
        return torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

    def _forward(self, frames):
        x = to_tensor(frames, self.input_size, IMAGENET_MEAN, IMAGENET_STD,
                      self.device).half()
        if self.mode == "descriptor":
            return self._chunked(x, lambda c: self._descriptor(self.model(c)))

        # Backbone features. DPT-Hybrid's encoder is `pretrained.model`
        # (a timm ViT). Take patch tokens before the depth head.
        # [VERIFY] attribute path depends on the torch.hub revision; print
        # the module tree once and adjust if it differs.
        vit = self.model.pretrained.model

        def run(c):
            h = vit.forward_features(c)          # (B, N, D)
            if h.ndim == 2:                      # some timm versions pool
                return h.half()
            return self._pool_tokens(h[:, 1:]).half()   # drop CLS

        return self._chunked(x, run)


class DepthAnythingV2Base(_DepthEncoder):
    """Depth Anything V2 Base, relative depth, four years newer than MiDaS."""
    name = "dav2_base"
    backbone_dim = 768

    def _build(self):
        from transformers import AutoModelForDepthEstimation
        return AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Base-hf")

    def _forward(self, frames):
        x = to_tensor(frames, self.input_size, IMAGENET_MEAN, IMAGENET_STD,
                      self.device).half()
        if self.mode == "descriptor":
            return self._chunked(
                x, lambda c: self._descriptor(
                    self.model(pixel_values=c).predicted_depth))

        def run(c):
            out = self.model(pixel_values=c, output_hidden_states=True)
            h = out.hidden_states[-1]            # (B, N, D) from the ViT
            return self._pool_tokens(h[:, 1:]).half()

        return self._chunked(x, run)


class UniDepthL(_DepthEncoder):
    """UniDepth V2 ViT-L, *metric* depth.

    Chang et al. argue metric depth gives better cross-video consistency than
    relative depth. Including it answers, with evidence, the question a
    Paper 1 reviewer will ask about the MiDaS choice. Heaviest model in the
    grid (~700 MB fp16); drop it first if memory is tight.
    """
    name = "unidepth_l"
    backbone_dim = 1024

    def _build(self):
        from unidepth.models import UniDepthV2
        return UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")

    def _forward(self, frames):
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2)
        x = F.interpolate(x.float() / 255.0,
                          size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False).half()

        def run(chunk):
            out = self.model.infer(chunk)
            return self._descriptor(out["depth"].squeeze(1))

        return self._chunked(x, run, chunk=2)      # heavier; smaller chunk


# ==========================================================================
REGISTRY: dict[str, type[Encoder]] = {
    "clip_b32": CLIPB32,
    "dinov2_b14": DINOv2B14,
    "videomae_b": VideoMAEB,
    "raft_small": RAFTSmall,
    "midas_hybrid": MiDaSHybrid,
    "dav2_base": DepthAnythingV2Base,
    "unidepth_l": UniDepthL,
}


def build(name: str, device: str = C.DEVICE) -> Encoder:
    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](device).load()
