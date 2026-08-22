"""
MTL training script for three heads: deepfake, anti-spoof, temporal consistency.
Supports GradNorm, PCGrad, TSM, AMP, OOM fallback, power monitoring, and full metrics.
"""

import contextlib
import os
import gc
import csv
import json
import glob
import time
import logging
import random
import warnings
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

from src.config import Config, get_config
from src.utils.flow_utils import (
    accumulate_flow,
    flow_path_for_clip,
    load_clip_flow,
    zero_flow,
)
from src.utils.logger_utils import log_banner, setup_logger
from src.utils.power_utils import GPUSample, PowerMonitor, power_monitor_from_config
from src.utils.repro_utils import set_seed, worker_init_fn

warnings.filterwarnings("ignore")

# Same name setup_logger() configures, so Dataset code (which has no Trainer to
# borrow self.logger from) writes into the run's training.log too.
logger = logging.getLogger("MTL")


# ──────────────────────────────────────────────────────────────────────────────
# Run ID & Checkpoint Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_next_run_id(checkpoint_root: str) -> str:
    """Return next run folder name like run01, run02, ..."""
    existing = sorted(glob.glob(os.path.join(checkpoint_root, "run*")))
    if not existing:
        return "run01"
    last = os.path.basename(existing[-1])
    idx = int(last.replace("run", "")) + 1
    return f"run{idx:02d}"


def get_last_run_id(checkpoint_root: str) -> Optional[str]:
    """Return most recent run folder or None."""
    existing = sorted(glob.glob(os.path.join(checkpoint_root, "run*")))
    return os.path.basename(existing[-1]) if existing else None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

# Numeric task ids. Emitted per sample so the loss/metric masks can index a
# tensor; the human-readable "task" string is kept alongside for logging.
TASK_DEEPFAKE = 0
TASK_SPOOF = 1


def build_transforms(cfg: Config, is_train: bool) -> A.Compose:
    """Build albumentations pipeline."""
    ac = cfg.aug
    if is_train:
        return A.Compose([
            A.Resize(ac.image_size, ac.image_size),
            A.HorizontalFlip(p=0.5 if ac.random_flip else 0.0),
            A.Rotate(limit=ac.random_rotate, p=0.5),
            A.ColorJitter(
                brightness=ac.brightness_jitter,
                contrast=ac.contrast_jitter,
                saturation=ac.saturation_jitter,
                hue=ac.hue_jitter, p=0.5
            ),
            A.GaussianBlur(p=ac.gaussian_blur_p),
            # albumentations >= 2.0 replaced quality_lower/quality_upper with
            # quality_range. The old kwargs are silently ignored (UserWarning
            # only) and the transform falls back to quality_range=(99, 100),
            # which makes JPEG augmentation a no-op — it has to be the tuple.
            A.ImageCompression(
                quality_range=(ac.jpeg_quality_min, ac.jpeg_quality_max),
                p=ac.jpeg_compression_p
            ),
            A.CoarseDropout(p=ac.coarse_dropout_p),
            A.ToGray(p=ac.random_grayscale_p),
            A.Normalize(mean=ac.normalize_mean, std=ac.normalize_std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(ac.image_size, ac.image_size),
            A.Normalize(mean=ac.normalize_mean, std=ac.normalize_std),
            ToTensorV2(),
        ])


class MTLDataset(Dataset):
    """
    Multi-task dataset that loads clips of T frames.
    Each sample contains frames from one clip for temporal modeling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        is_train: bool = True
    ):
        """Initialize with dataframe containing all splits of one dataset."""
        self.cfg = cfg
        self.is_train = is_train
        self.transform = build_transforms(cfg, is_train)
        self.T = cfg.train.num_frames

        # Group frames by clip
        self.clips = self._build_clips(df)

        # Optical flow is *read* here, never computed — see utils/flow_utils.py
        # for why (Farneback is ~0.4 s/clip; two workers cannot hide that).
        self.use_flow = bool(cfg.train.use_optical_flow)
        self.flow_resize = 0
        self._flow_missing = 0
        if self.use_flow:
            self.use_flow, self.flow_resize = self._probe_flow()

    def _probe_flow(self) -> Tuple[bool, int]:
        """Confirm precomputed flow is readable and learn its resolution.

        The resolution is taken from the file rather than from
        PreprocessConfig.flow_resize so a stale set of files is never silently
        reinterpreted at the wrong scale, and so changing the config does not
        require the reader to be updated in lockstep.
        """
        if not self.clips:
            return False, 0
        probe = flow_path_for_clip(
            self.clips[0]["frame_paths"][0], self.clips[0]["clip_index"]
        )
        if probe.exists():
            try:
                return True, int(load_clip_flow(probe).shape[-1])
            except Exception as exc:                   # noqa: BLE001
                logger.warning("Flow file unreadable (%s): %s: %s",
                               probe, type(exc).__name__, exc)
        else:
            logger.warning(
                "use_optical_flow=True but %s is missing — training continues "
                "WITHOUT flow (flow_encoder stays untrained). Re-run "
                "preprocessing with precompute_optical_flow=True.", probe,
            )
        return False, 0

    def _build_clips(self, df: pd.DataFrame) -> List[dict]:
        """Group rows by (video_index, clip_index) and build clip records."""
        clips = []
        for _, grp in df.groupby(["video_path", "clip_index"], sort=False):
            grp = grp.sort_values("frame_num")
            task = grp["task"].iloc[0]
            label_raw = grp["label"].iloc[0]
            binary_label = 0 if label_raw == "real" else 1
            spoof_type = grp["spoof_type"].iloc[0]
            dataset = grp["dataset"].iloc[0]
            video_path = grp["video_path"].iloc[0]
            frame_paths = grp["frame_path"].tolist()

            clips.append({
                "frame_paths": frame_paths,
                "task": task,
                "label": binary_label,
                "spoof_type": spoof_type,
                "dataset": dataset,
                "video_path": video_path,
                # Needed to locate this clip's precomputed flow file.
                "clip_index": int(grp["clip_index"].iloc[0]),
            })
        return clips

    def _sample_frames(self, n_available: int) -> List[int]:
        """Positions of the T frames to load, as indices into the clip.

        Returns indices rather than paths because the precomputed flow is
        indexed by position: stored pair k is the flow from saved frame k to
        k+1, so accumulating it across a jitter gap of 2-4 requires knowing
        which positions were picked.
        """
        tc = self.cfg.train
        N = n_available
        if N <= self.T:
            # Repeat to fill. Wraps back to 0, so the flow accumulator sees a
            # non-increasing step there and emits zeros for that pair.
            return (list(range(N)) * ((self.T // max(N, 1)) + 1))[:self.T]

        if self.is_train and tc.temporal_jitter:
            gap = random.randint(tc.min_frame_gap, tc.max_frame_gap)
            max_start = N - gap * (self.T - 1) - 1
            if max_start < 0:
                return [int(i) for i in np.linspace(0, N - 1, self.T, dtype=int)]
            start = random.randint(0, max_start)
            return [min(start + i * gap, N - 1) for i in range(self.T)]

        return [int(i) for i in np.linspace(0, N - 1, self.T, dtype=int)]

    def _load_flow(self, clip: dict, indices: List[int]) -> torch.Tensor:
        """Precomputed flow for the T frames this sample chose → (T-1, 2, R, R)."""
        path = flow_path_for_clip(clip["frame_paths"][0], clip["clip_index"])
        try:
            pair_flow = load_clip_flow(path)
        except Exception:                              # noqa: BLE001
            # Per-clip fallback: zeros keep the batch collatable. Warn once per
            # worker rather than once per sample.
            self._flow_missing += 1
            if self._flow_missing == 1:
                logger.warning("Flow missing/unreadable for %s — using zeros "
                               "for this clip (further cases silent)", path)
            return torch.from_numpy(zero_flow(self.T - 1, self.flow_resize))
        return torch.from_numpy(accumulate_flow(pair_flow, indices))

    def _load_frame(self, path: str) -> np.ndarray:
        """Load and augment a single frame."""
        img = np.array(Image.open(path).convert("RGB"))
        return img

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        indices = self._sample_frames(len(clip["frame_paths"]))
        frame_paths = [clip["frame_paths"][i] for i in indices]

        # One augmentation drawn per *clip*, not per frame. Albumentations'
        # `images` target applies identical parameters to every frame in the
        # list, so the flip/rotation/jitter is shared across the whole clip.
        # Augmenting each frame independently would inject artificial
        # frame-to-frame inconsistency — exactly the signal the temporal head is
        # supposed to measure — and would spatially misalign the frames that TSM
        # shifts channels between.
        imgs = [self._load_frame(fp) for fp in frame_paths]
        frames = self.transform(images=imgs)["images"]

        frames_tensor = torch.stack(list(frames), dim=0)

        # ── Task-specific labels ──────────────────────────────────
        task = clip["task"]
        binary_label = clip["label"]  # 0=real/live, 1=fake/spoof

        # The off-task label is a placeholder that carries no information: a
        # FaceForensics++ clip says nothing about presentation attacks and a
        # SiW-Mv2 clip says nothing about face swapping. `task_id` is what lets
        # the training and validation loops mask it out — without the mask these
        # zeros teach the spoof head "a deepfake face is bona-fide" and the
        # deepfake head "a presentation attack is real".
        if task == "deepfake":
            deepfake_label = binary_label
            spoof_label = 0                      # placeholder — masked by task_id
        else:  # anti-spoof
            deepfake_label = 0                   # placeholder — masked by task_id
            spoof_label = binary_label

        # Supervised on the whole batch by design: "inauthentic" means the same
        # thing in both datasets (a swapped face and a replayed screen are both
        # non-genuine capture), so this label is real ground truth either way.
        temporal_label = binary_label

        sample = {
            "frames": frames_tensor,
            "deepfake_label": torch.tensor(deepfake_label, dtype=torch.long),
            "spoof_label": torch.tensor(spoof_label, dtype=torch.long),
            "temporal_label": torch.tensor(temporal_label, dtype=torch.long),
            # int, not the `task` string: default_collate turns strings into a
            # python list, which cannot index a tensor. TASK_DEEPFAKE/TASK_SPOOF.
            "task_id": torch.tensor(
                TASK_DEEPFAKE if task == "deepfake" else TASK_SPOOF,
                dtype=torch.long,
            ),
            "task": task,
            "dataset": clip["dataset"],
            # Per-attack-type recall needs this at eval time. Stays a string —
            # it is only ever used to group numpy arrays, never to index a tensor.
            "spoof_type": clip["spoof_type"],
            "video_id": clip["video_path"],  # video_path
        }
        # The key is present for every sample or for none — a mixed batch would
        # not collate. `use_flow` is decided once in __init__, so it cannot vary
        # between samples of the same dataset.
        if self.use_flow:
            sample["flow"] = self._load_flow(clip, indices)
        return sample



class InterleavedBatchSampler(Sampler[List[int]]):
    """Fixed per-batch quota of FF++ and SiW-Mv2 indices into a ConcatDataset.

    Replaces the WeightedRandomSampler this loader used to build. That sampler
    took ``ff_sample_ratio`` as a *weight* — ``siw_weight = (1 - ratio) / n_siw``
    — so ``ratio = 1.0`` gave every SiW-Mv2 clip a weight of exactly 0.0.
    ``torch.multinomial`` never draws a zero-weight index, so run01 trained for
    14 epochs without a single anti-spoof sample: ``loss_sp`` hit exactly 0.0
    from epoch 2 and the spoof head's output ceiling ended at 0.4978.

    A quota is stronger than fixing the weight to 0.5. With plain weighted
    sampling, P(no SiW in a batch of 4) is 6.25%, and an empty task mask makes
    the masked loss a constant: no gradient for PCGrad to project (zero-norm
    task vector) and a degenerate L(0) seed for GradNorm. Here every batch
    contains at least one sample of each task by construction, so the masks in
    `_train_epoch` are always non-empty and neither algorithm needs a special
    case.

    The two datasets are shuffled and recycled independently: an epoch is as
    long as the dataset that needs the most batches to be seen once, and the
    smaller one wraps around (reshuffled each time) rather than truncating the
    larger.
    """

    def __init__(
        self,
        n_ff: int,
        n_siw: int,
        batch_size: int,
        ff_per_batch: int,
        seed: int = 42,
    ):
        if n_ff == 0 or n_siw == 0:
            raise ValueError(
                f"InterleavedBatchSampler needs both datasets non-empty "
                f"(got n_ff={n_ff}, n_siw={n_siw})"
            )
        self.n_ff = n_ff
        self.n_siw = n_siw
        self.ff_per_batch = ff_per_batch
        self.siw_per_batch = batch_size - ff_per_batch
        self.seed = seed
        self.epoch = 0
        # ConcatDataset([ff_ds, siw_ds]) lays SiW out after FF++.
        self.siw_offset = n_ff
        # Ceiling, not floor: with 1085 SiW clips at 2 per batch, floor gives 542
        # batches = 1084 samples and leaves one clip unseen every epoch. Rounding
        # up costs at most one extra batch and the pool-wrap below fills the last
        # slot, so "an epoch shows every clip of the larger dataset once" is
        # actually true rather than nearly true.
        self.num_batches = max(
            -(-self.n_ff // self.ff_per_batch),
            -(-self.n_siw // self.siw_per_batch),
        )

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle differently each epoch (the loader is rebuilt per fit, not per epoch)."""
        self.epoch = epoch

    def _shuffled(self, n: int, g: torch.Generator) -> List[int]:
        return torch.randperm(n, generator=g).tolist()

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        ff_pool = self._shuffled(self.n_ff, g)
        siw_pool = self._shuffled(self.n_siw, g)
        ff_i = siw_i = 0

        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.ff_per_batch):
                if ff_i >= self.n_ff:                  # exhausted → reshuffle
                    ff_pool = self._shuffled(self.n_ff, g)
                    ff_i = 0
                batch.append(ff_pool[ff_i])
                ff_i += 1
            for _ in range(self.siw_per_batch):
                if siw_i >= self.n_siw:
                    siw_pool = self._shuffled(self.n_siw, g)
                    siw_i = 0
                batch.append(self.siw_offset + siw_pool[siw_i])
                siw_i += 1
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def build_interleaved_loader(
    ff_df: pd.DataFrame,
    siwmv2_df: pd.DataFrame,
    cfg: Config,
    is_train: bool
) -> DataLoader:
    """
    Build a DataLoader over both datasets. In training, an
    `InterleavedBatchSampler` guarantees a fixed quota of each task per batch
    (see that class for why a quota, not a sampling weight).
    """
    ff_ds = MTLDataset(ff_df, cfg, is_train)
    siw_ds = MTLDataset(siwmv2_df, cfg, is_train)

    # A batch mixes both datasets, and default_collate cannot stack samples where
    # only some carry a "flow" key. Each dataset probes its own files, so they can
    # legitimately disagree (e.g. preprocessing re-run for FF++ only) — settle it
    # here, once, rather than crashing in the first collate.
    if ff_ds.use_flow != siw_ds.use_flow:
        logger.warning(
            "Precomputed flow found for only one dataset (ff=%s, siw=%s) — "
            "disabling flow for both so batches stay collatable. Re-run "
            "preprocessing over both datasets to enable it.",
            ff_ds.use_flow, siw_ds.use_flow,
        )
        ff_ds.use_flow = siw_ds.use_flow = False
    elif ff_ds.use_flow and ff_ds.flow_resize != siw_ds.flow_resize:
        # Same problem one level down: different R means different tensor shapes.
        logger.warning(
            "Flow resolution differs between datasets (ff=%d, siw=%d) — "
            "disabling flow. Re-run preprocessing with one flow_resize.",
            ff_ds.flow_resize, siw_ds.flow_resize,
        )
        ff_ds.use_flow = siw_ds.use_flow = False

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([ff_ds, siw_ds])

    common = dict(
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )

    if not is_train:
        return DataLoader(
            combined,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        )

    # `ff_sample_ratio` is the FF++ *share of each batch*. Clamped to leave at
    # least one slot for each task, so no value of it can starve a head — 1.0
    # used to mean "100% FF++", which is exactly what broke run01.
    bs = cfg.train.batch_size
    if bs < 2:
        raise ValueError(
            f"batch_size must be >= 2 to fit both tasks in a batch (got {bs})"
        )
    ff_per_batch = int(round(cfg.train.ff_sample_ratio * bs))
    ff_per_batch = max(1, min(bs - 1, ff_per_batch))

    batch_sampler = InterleavedBatchSampler(
        n_ff=len(ff_ds),
        n_siw=len(siw_ds),
        batch_size=bs,
        ff_per_batch=ff_per_batch,
        seed=cfg.train.seed,
    )
    logger.info(
        "Train batches: %d/batch = %d FF++ + %d SiW-Mv2 | %d batches/epoch "
        "(ff=%d clips, siw=%d clips, ff_sample_ratio=%.2f)",
        bs, ff_per_batch, bs - ff_per_batch, len(batch_sampler),
        len(ff_ds), len(siw_ds), cfg.train.ff_sample_ratio,
    )

    # With batch_sampler, batch_size/shuffle/drop_last must not be set — the
    # sampler yields complete index lists and every batch is full by construction.
    return DataLoader(combined, batch_sampler=batch_sampler, **common)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Shift Module (TSM)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalShift(nn.Module):
    """
    TSM: shifts a fraction of channels along the temporal dimension
    to enable temporal modeling without extra parameters.
    """

    def __init__(self, shift_ratio: float = 0.125):
        """Initialize with channel shift ratio."""
        super().__init__()
        self.shift_ratio = shift_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W) — needs B and T to reconstruct
        # Called from backbone hook; reshape done externally
        return x


def apply_tsm(x: torch.Tensor, shift_ratio: float, T: int) -> torch.Tensor:
    """
    Apply temporal shift to a (B*T, C, H, W) tensor.
    Shifts shift_ratio of channels backward and forward in time.
    """
    BT, C, H, W = x.shape
    B = BT // T
    x = x.view(B, T, C, H, W)

    fold = max(1, int(C * shift_ratio))
    out = x.clone()
    # Shift forward (past -> present)
    out[:, 1:, :fold] = x[:, :-1, :fold]
    out[:, 0, :fold] = 0
    # Shift backward (future -> present)
    out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
    out[:, -1, fold:2*fold] = 0

    return out.view(BT, C, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# MTL Model
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeHead(nn.Module):
    """Binary classification head for deepfake detection."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class AntiSpoofHead(nn.Module):
    """Binary classification head for anti-spoofing."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class TemporalHead(nn.Module):
    """
    Temporal consistency head with two independently supervised branches.

    `proj` is trained by the adjacent-frame cosine term and `clf` by BCE on the
    temporal label — `validate()` scores `clf`, so both matter. `supervision`
    only decides whether the third branch, `flow_encoder`, is built:

      - 'cosine_sim':   proj + clf only; flow_encoder is None
      - 'optical_flow': adds flow_encoder, fed the precomputed .npz flow
      - 'combined':     same as optical_flow — the 'pseudo_label' branch below
                        stays unused on purpose, since temporal_label is real
                        ground truth and a deepfake-derived pseudo-label would
                        be circular self-distillation

    Args:
        feature_dim:    backbone output dimension (e.g. 1408 for EfficientNet-B2)
        proj_dim:       temporal projection dimension
        hidden_dim:     classifier hidden dimension
        dropout:        dropout probability
        supervision:    'cosine_sim' | 'optical_flow' | 'combined'
        flow_channels:  optical flow channels (2 = dx, dy)
    """

    def __init__(
        self,
        feature_dim: int = 1408,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        supervision: str = "combined",
        flow_channels: int = 2,
    ) -> None:
        super().__init__()
        self.supervision = supervision

        # ── Per-frame temporal projection ────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        # ── Temporal classifier (for pseudo-label supervision) ──────────────
        self.clf = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # ── Optical flow encoder (optional) ──────────────────────────────────
        if supervision in ("optical_flow", "combined"):
            # Lightweight CNN to encode (T-1, flow_channels, H, W) → scalar
            self.flow_encoder = nn.Sequential(
                nn.Conv2d(flow_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),   # flow consistency score per pair
            )
        else:
            self.flow_encoder = None

    def forward(
        self,
        feats: torch.Tensor,
        df_logits: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            feats:     (B, T, D)  per-frame backbone features
            df_logits: (B, T)     per-frame deepfake logits (for pseudo-label)
            flow:      (B, T-1, 2, H, W) optical flow between consecutive frames

        Returns:
            dict with:
              - 'temp_proj':         (B, T, proj_dim) per-frame projections
              - 'temp_logit':        (B,)             temporal binary logit
              - 'pseudo_label':      (B,)             soft pseudo-label (detached)
              - 'flow_consistency':  (B,)             flow score (if flow given)
        """
        B, T, D = feats.shape

        # ── 1. Project per-frame features ─────────────────────────────────────
        proj = self.proj(feats)        # (B, T, proj_dim)
        pooled = proj.mean(dim=1)      # (B, proj_dim)
        temp_logit = self.clf(pooled).squeeze(-1)  # (B,)

        output = {
            "temp_proj": proj,
            "temp_logit": temp_logit,
            "pseudo_label": None,
            "flow_consistency": None,
        }

        # ── 2. Pseudo-label from adjacent deepfake predictions ─────────────────
        if df_logits is not None and self.supervision in ("pseudo_label", "combined"):
            # df_logits: (B, T) — average over adjacent frames (±1 window)
            df_probs = torch.sigmoid(df_logits.detach())  # (B, T), no grad
            # Smooth label: mean over all frames per sample
            pseudo = df_probs.mean(dim=1)  # (B,) — soft target in [0, 1]
            output["pseudo_label"] = pseudo

        # ── 3. Optical flow consistency ────────────────────────────────────────
        if flow is not None and self.flow_encoder is not None:
            # flow: (B, T-1, 2, H, W) → process each pair, average score
            B, Tm1, C, H, W = flow.shape
            flow_flat = flow.view(B * Tm1, C, H, W)          # (B*(T-1), 2, H, W)
            scores = self.flow_encoder(flow_flat)              # (B*(T-1), 1)
            scores = scores.view(B, Tm1).mean(dim=1)           # (B,)
            output["flow_consistency"] = scores

        return output



class MTLModel(nn.Module):
    """
    Multi-task learning model with shared EfficientNet backbone
    and three task-specific heads.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        mc = cfg.model
        self.cfg = cfg
        self.T = cfg.train.num_frames
        self.use_tsm = mc.use_tsm
        self.tsm_shift_ratio = mc.tsm_shift_ratio
        self.tsm_blocks = tuple(getattr(mc, "tsm_block_indices", (1, 3, 5)))
        # Recompute block activations during backward instead of storing them.
        # PCGrad backwards the graph once per task and GradNorm adds a
        # double-backward pass, so stored activations are the dominant VRAM cost
        # on a small card. Trades ~30% step time for a large memory saving.
        self.grad_checkpointing = getattr(mc, "grad_checkpointing", False)

        # Shared backbone
        self.backbone = timm.create_model(
            mc.backbone,
            pretrained=mc.pretrained,
            num_classes=0,            # remove classifier
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features

        # Task heads
        self.deepfake_head = DeepfakeHead(feat_dim, mc.deepfake_hidden, mc.dropout)
        self.spoof_head = AntiSpoofHead(feat_dim, mc.spoof_hidden, mc.dropout)
        self.temporal_head = TemporalHead(
            feature_dim=feat_dim,
            proj_dim=mc.temporal_proj_dim,
            hidden_dim=mc.temporal_hidden,
            dropout=mc.dropout,
            supervision=mc.temporal_supervision,
            flow_channels=mc.optical_flow_in_channels,
        )


        # GradNorm learnable log-weights
        self.log_weights = nn.Parameter(torch.zeros(3))  # [df, spoof, temp]

    @property
    def task_weights(self) -> torch.Tensor:
        """Softmax-normalized task weights from GradNorm parameters."""
        return F.softmax(self.log_weights, dim=0) * 3   # scale to sum~3

    @task_weights.setter
    def task_weights(self, weights: torch.Tensor) -> None:
        """Set log_weights from desired positive task weights."""
        with torch.no_grad():
            self.log_weights.copy_(torch.log(weights / weights.mean() + 1e-8))


    def forward_backbone(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Forward through EfficientNet backbone with intermediate TSM on feature maps.

        TSM is applied to the *feature maps* entering blocks 1/3/5 (16/48/120
        channels on EfficientNet-B2), never to the raw 3-channel RGB input —
        shifting raw pixel channels would swap colour planes between adjacent
        frames rather than mixing temporal context.

        `act1`/`act2` are absent in timm >= 1.0, where the activation is fused
        into the BatchNormAct2d layers, so both are guarded by hasattr.
        """
        if not hasattr(self.backbone, "blocks"):
            return self.backbone(x)

        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        if hasattr(self.backbone, "act1"):
            x = self.backbone.act1(x)

        for i, block in enumerate(self.backbone.blocks):
            if self.use_tsm and i in self.tsm_blocks:
                x = apply_tsm(x, self.tsm_shift_ratio, T)
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        if hasattr(self.backbone, "act2"):
            x = self.backbone.act2(x)
        x = self.backbone.global_pool(x)
        return x

    def forward(
        self,
        frames: torch.Tensor,                    # (B, T, C, H, W)
        flow: Optional[torch.Tensor] = None,     # (B, T-1, 2, R, R) or None
    ) -> dict:
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        feats = self.forward_backbone(x, T)       # (B*T, D)
        feats = feats.view(B, T, -1)  # (B, T, D)

        frame_feat = feats.mean(dim=1)  # (B, D) — temporal pooled

        df_logit = self.deepfake_head(frame_feat)
        sp_logit = self.spoof_head(frame_feat)
        # `flow` is read from disk by MTLDataset (precomputed in preprocessing);
        # None keeps the flow_encoder out of the graph entirely.
        temp_out   = self.temporal_head(feats, flow=flow)
        temp_proj  = temp_out["temp_proj"]
        temp_logit = temp_out["temp_logit"]

        return {
            "deepfake_logit":   df_logit,
            "spoof_logit":      sp_logit,
            "temp_logit":       temp_logit,
            "temp_proj":        temp_proj,
            "pseudo_label":     temp_out.get("pseudo_label"),
            "flow_consistency": temp_out.get("flow_consistency"),
            "feats":            feats,
        }
# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced anti-spoofing."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def temporal_consistency_loss(
    proj: torch.Tensor,    # (B, T, H)
    labels: torch.Tensor,  # (B,) binary
    logit: Optional[torch.Tensor] = None,   # (B,) temporal head classifier logit
    logit_weight: float = 1.0,
    flow_score: Optional[torch.Tensor] = None,  # (B,) flow_encoder consistency
    flow_weight: float = 0.0,
) -> torch.Tensor:
    """
    Self-supervised temporal loss:
    real clips -> minimize cosine distance between adjacent frames,
    fake clips -> maximize it.

    The cosine term only reaches `TemporalHead.proj`. `TemporalHead.clf` sits on
    a separate branch, so passing `logit` is what gives it a gradient — and
    validate() reports temporal accuracy/AUC on exactly that logit, so without
    the BCE term those metrics score a randomly-initialised head forever.

    `flow_encoder` is a *third* branch with the same problem: it is reached by
    neither term above. Its BCE — judge real vs. fake from motion alone — is the
    only thing that trains it, which is why `use_optical_flow = False` leaves its
    8 parameters dead rather than merely unhelpful.
    """
    # Adjacent frame cosine similarity: (B, T-1)
    p1 = proj[:, :-1]                         # (B, T-1, H)
    p2 = proj[:, 1:]
    cos_sim = F.cosine_similarity(p1, p2, dim=-1)   # (B, T-1)
    mean_sim = cos_sim.mean(dim=1)                   # (B,)

    # real -> high sim (sim->1), fake -> low sim (sim->-1)
    real_mask = (labels == 0).float()
    fake_mask = (labels == 1).float()

    loss = (real_mask * (1 - mean_sim) + fake_mask * (1 + mean_sim)).mean()

    if logit is not None and logit_weight > 0.0:
        loss = loss + logit_weight * F.binary_cross_entropy_with_logits(
            logit.flatten(), labels.float().flatten()
        )

    if flow_score is not None and flow_weight > 0.0:
        loss = loss + flow_weight * F.binary_cross_entropy_with_logits(
            flow_score.flatten(), labels.float().flatten()
        )
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# PCGrad
# ──────────────────────────────────────────────────────────────────────────────




def get_flat_grads(
    loss: torch.Tensor,
    params: List[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Compute flat gradient vector for a loss w.r.t. params."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    flat = []
    for g in grads:
        if g is not None:
            flat.append(g.view(-1))
        else:
            flat.append(torch.zeros(1, device=loss.device))
    return torch.cat(flat) if flat else None


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Threshold metrics + ranking metrics + collapse diagnostics for one head.

    Shared by all three heads so the same definitions are used everywhere. The
    threshold-dependent block is the part that was missing from run01: the spoof
    head had `recall = 0.0` and `f1 = 0.0` from epoch 2, while the ACER that was
    actually being logged sat at exactly 0.5 — the value a coin flip produces —
    so nothing in the logs distinguished a dead head from a mediocre one.

    The `pred_pos_rate` / `score_*` block exists for the same reason. A head
    whose global maximum output is 0.4978 can never cross a 0.5 threshold, and
    that fact is invisible in any rate-based metric.

    Returns {} when the split is single-class, matching the other metric
    functions: no metrics rather than misleading ones.
    """
    results: dict = {}
    if len(np.unique(labels)) < 2:
        return results

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    results.update({
        # ── threshold metrics @ `threshold` ──
        "acc": float((preds == labels).mean()),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),                              # = TPR = 1 - APCER
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # ── threshold-free ──
        "auc_roc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        # ── support ──
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        # ── collapse diagnostics ──
        "pred_pos_rate": float(preds.mean()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean_pos": float(scores[labels == 1].mean()),
        "score_mean_neg": float(scores[labels == 0].mean()),
    })
    return results


def collapse_warnings(
    name: str,
    metrics: dict,
    threshold: float = 0.5,
) -> List[str]:
    """Human-readable reasons to distrust `metrics`, or [] if the head looks alive.

    Called from validate() so a dead head is reported the epoch it dies. Each
    condition here was true of run01's spoof head from epoch 1 and produced no
    log line at all.
    """
    if not metrics:
        return []
    msgs = []
    rate = metrics.get("pred_pos_rate")
    if rate == 0.0:
        msgs.append(
            f"{name}: predicts class 0 for every sample "
            f"(recall={metrics.get('recall', float('nan')):.4f}, "
            f"f1={metrics.get('f1', float('nan')):.4f}) — head has collapsed"
        )
    elif rate == 1.0:
        msgs.append(f"{name}: predicts class 1 for every sample — head has collapsed")
    if metrics.get("score_max", 1.0) < threshold:
        msgs.append(
            f"{name}: max score {metrics['score_max']:.4f} < threshold {threshold} "
            f"— no input can ever be classified positive"
        )
    if metrics.get("auc_roc", 1.0) < 0.5:
        msgs.append(
            f"{name}: AUC {metrics['auc_roc']:.4f} is below chance — scores are "
            f"anti-correlated with labels (mean_pos={metrics['score_mean_pos']:.4f} "
            f"< mean_neg={metrics['score_mean_neg']:.4f})"
        )
    return msgs


def compute_deepfake_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    video_paths: Optional[List[str]] = None,
    video_agg: str = "mean",
    threshold: float = 0.5,
) -> dict:
    """
    Full binary metric set (see binary_classification_metrics) plus the
    deepfake-specific extras: EER, accuracy at the Youden-J threshold, and
    video-level AUC.
    """
    results = binary_classification_metrics(labels, scores, threshold=threshold)
    if not results:
        return results

    fpr, tpr, thresholds = roc_curve(labels, scores)
    # EER
    eer_fn = interp1d(fpr, tpr)
    try:
        eer = brentq(lambda x: 1 - x - eer_fn(x), 0, 1)
    except Exception:
        eer = float("nan")
    results["eer"] = eer

    # Best-threshold accuracy
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    preds = (scores >= best_thresh).astype(int)
    results["acc_best_thresh"] = (preds == labels).mean()
    results["best_threshold"] = best_thresh

    # Video-level AUC. `max` pools a video to its most-suspicious clip, `mean` to
    # its average — a real difference on long videos with a few tampered clips,
    # so it comes from EvalConfig rather than being hardcoded.
    if video_paths is not None:
        vdf = pd.DataFrame({
            "video": video_paths, "label": labels, "score": scores
        })
        agg = video_agg if video_agg in ("mean", "max") else "mean"
        vid_agg = vdf.groupby("video").agg({"label": "first", "score": agg})
        if vid_agg["label"].nunique() > 1:
            results["video_auc"] = roc_auc_score(
                vid_agg["label"], vid_agg["score"]
            )

    return results


def compute_deepfake_metrics_by_compression(
    labels: np.ndarray,
    scores: np.ndarray,
    datasets: List[str],    # or compression label column
    tags: Sequence[str] = ("c23", "c40"),
) -> dict:
    """Compute AUC per compression type (c23, c40) based on dataset column."""
    results = {}
    for tag in tags:
        mask = np.array([tag.lower() in d.lower() for d in datasets])
        if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
            results[f"auc_{tag}"] = roc_auc_score(labels[mask], scores[mask])
    return results


def compute_spoof_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    fpr_threshold: float = 0.01,
    spoof_types: Optional[Sequence[str]] = None,
) -> dict:
    """
    Full binary metric set (see binary_classification_metrics) plus the
    anti-spoofing conventions: APCER, BPCER, ACER, HTER, TPR@FPR=1%, and
    per-attack-type recall when `spoof_types` is given.

    labels: 0=real (bona-fide), 1=attack.

    Note APCER == 1 - recall and BPCER == 1 - specificity; both are kept because
    the anti-spoofing literature reports them under these names. ACER is *not*
    a substitute for recall: it is exactly 0.5 for a head that predicts one class
    for everything, which is how run01's collapse stayed invisible for 14 epochs.
    """
    results = binary_classification_metrics(labels, scores, threshold=threshold)
    preds = (scores >= threshold).astype(int)

    # APCER: Attack Presentation Classification Error Rate
    # = FNR for attacks (predicted real when attack)
    attack_mask = labels == 1
    real_mask = labels == 0

    if attack_mask.sum() > 0:
        apcer = (preds[attack_mask] == 0).mean()
    else:
        apcer = float("nan")

    # BPCER: Bona-fide Presentation Classification Error Rate
    # = FPR for real (predicted attack when real)
    if real_mask.sum() > 0:
        bpcer = (preds[real_mask] == 1).mean()
    else:
        bpcer = float("nan")

    acer = (apcer + bpcer) / 2
    hter = acer  # same formulation

    results.update({
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "hter": hter,
    })

    # TPR @ FPR=fpr_threshold (1% by default). The interpolation is only valid
    # inside the ROC's observed FPR range — a tiny bona-fide set can have a
    # minimum FPR above the threshold, which would extrapolate silently.
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        in_range = fpr.min() <= fpr_threshold <= fpr.max()
        results[f"tpr_at_fpr{int(round(fpr_threshold * 100))}"] = (
            float(interp1d(fpr, tpr)(fpr_threshold)) if in_range else float("nan")
        )
        results["auc"] = roc_auc_score(labels, scores)

    # Per-attack-type recall. The attack types are 11.7x imbalanced (10916 frames
    # for Partial_FunnyeyeGlasses vs 934 for Silicone), so an aggregate recall can
    # look healthy while entire attack families are never caught. `n_` is emitted
    # next to each rate because a recall over 2 test clips is not a measurement.
    if spoof_types is not None:
        types = np.asarray(spoof_types)
        for st in sorted({t for t, lb in zip(types, labels) if lb == 1}):
            m = (types == st) & attack_mask
            if m.sum() == 0:
                continue
            results[f"recall_{st}"] = float((preds[m] == 1).mean())
            results[f"n_{st}"] = int(m.sum())

    return results


def compute_temporal_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Full binary metric set for the temporal head's classifier logit.

    `bin_acc` is kept as an alias of `acc` so existing history files and the
    thesis tables stay readable.
    """
    results = binary_classification_metrics(labels, scores, threshold=threshold)
    if not results:
        # Single-class split: accuracy alone is still well defined, and the
        # temporal head is scored on both val loaders where one may be skewed.
        preds = (scores >= threshold).astype(int)
        return {"bin_acc": float((preds == labels).mean())}
    results["bin_acc"] = results["acc"]
    results["auc"] = results["auc_roc"]
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Result CSV Logger
# ──────────────────────────────────────────────────────────────────────────────

class ResultLogger:
    """Appends per-epoch metrics to a CSV file, tolerating a growing key set.

    The header cannot be fixed from the first row: metric functions return {}
    on a single-class split, and the per-attack-type keys (`recall_<type>`)
    only exist for the types present in that epoch's validation batch. The
    previous version wrote the header once from row 0 but built each
    `DictWriter` from *that row's* keys, so a later row with different keys was
    written in a different column order under the old header — silently.

    Fix: keep the union of all keys seen, and rewrite the whole file whenever a
    new one appears. At ~25 rows the cost is irrelevant.
    """

    def __init__(self, csv_path: str):
        """Initialize with output CSV path."""
        self.path = csv_path
        self.fieldnames: List[str] = []
        self.rows: List[dict] = []
        if os.path.exists(csv_path):
            # Resuming: adopt the existing rows so a rewrite does not lose them.
            with open(csv_path, newline="") as f:
                self.rows = list(csv.DictReader(f))
            if self.rows:
                self.fieldnames = list(self.rows[0].keys())

    def log(self, row: dict) -> None:
        """Append one row, rewriting the file if it introduces new columns."""
        self.rows.append(row)
        new_keys = [k for k in row if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            self._rewrite()
        else:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(
                    f, fieldnames=self.fieldnames, restval=""
                ).writerow(row)

    def _rewrite(self) -> None:
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self.rows)


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Monitors a validation metric and signals when training should stop.

    Supports both 'max' (e.g. AUC) and 'min' (e.g. ACER) modes.

    Two guards were added after run01, where the monitored composite plateaued
    within a range of 0.0139 over epochs 3-13 and the run stopped on noise:

    - `min_epochs`: never stop before this epoch. With `warmup_epochs = 3` the
      backbone learning rate is 3e-9 in epoch 0, so patience counted there
      measures the schedule rather than the model.
    - `smooth_window`: the stop decision runs on a rolling mean of the last N
      values. Best-model selection stays on the raw value (handled by the
      caller), so smoothing cannot cost you the best checkpoint — it only stops
      a single lucky epoch from resetting the patience counter.

    Args:
        patience:      epochs without improvement before stopping
        min_delta:     minimum change to qualify as an improvement
        mode:          'max' to maximize metric, 'min' to minimize
        metric_key:    name of the monitored metric (for logging)
        min_epochs:    earliest epoch at which stopping may trigger
        smooth_window: rolling-mean window for the stop decision (1 = off)
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 2e-3,
        mode: str = "max",
        metric_key: str = "composite",
        min_epochs: int = 0,
        smooth_window: int = 1,
    ) -> None:
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_key = metric_key
        self.min_epochs = min_epochs
        self.smooth_window = max(1, smooth_window)

        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False
        self.history: List[float] = []
        self.epochs_seen: int = 0

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        return current < self.best_value - self.min_delta

    def _smoothed(self) -> float:
        window = self.history[-self.smooth_window:]
        return float(sum(window) / len(window))

    def step(self, current_value: float) -> bool:
        """
        Update state with the latest metric value.

        Args:
            current_value: metric value for the current epoch

        Returns:
            True if training should stop, False otherwise.
        """
        self.history.append(float(current_value))
        self.epochs_seen += 1
        smoothed = self._smoothed()

        if self._is_improvement(smoothed):
            self.best_value = smoothed
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if (
            self.epochs_without_improvement >= self.patience
            and self.epochs_seen >= self.min_epochs
        ):
            self.should_stop = True

        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "should_stop": self.should_stop,
            "history": self.history,
            "epochs_seen": self.epochs_seen,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_value = state["best_value"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.should_stop = state["should_stop"]
        # Older checkpoints predate smoothing; an empty history just means the
        # first resumed epoch is unsmoothed, which is harmless.
        self.history = list(state.get("history", []))
        self.epochs_seen = state.get("epochs_seen", len(self.history))


# ──────────────────────────────────────────────────────────────────────────────
# GradNorm
# ──────────────────────────────────────────────────────────────────────────────
class GradNormManager:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.

    Reference: Chen et al., "GradNorm: Gradient Normalization for
    Adaptive Loss Balancing in Deep Multitask Networks", ICML 2018.

    Maintains learnable task weights and updates them so that each task's
    gradient norm matches a target proportional to the task's training speed.

    Args:
        num_tasks:    number of tasks (3: deepfake, spoof, temporal)
        alpha:        asymmetry hyperparameter (higher → more aggressive rebalancing)
        lr:           learning rate for task-weight optimizer
        device:       torch device
        init_weights: initial task weights [w_deepfake, w_spoof, w_temporal]
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        init_weights: Optional[List[float]] = None,
    ) -> None:
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device

        # Log-scale weights to ensure positivity
        if init_weights is None:
            init_weights = [1.0] * num_tasks
        log_init = [float(np.log(max(w, 1e-6))) for w in init_weights]

        self.log_weights = nn.Parameter(
            torch.tensor(log_init, dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.optimizer = torch.optim.Adam([self.log_weights], lr=lr)

        # Initial losses — set on first update call
        self.initial_losses: Optional[torch.Tensor] = None
        self._update_count = 0

    @property
    def weights(self) -> torch.Tensor:
        """Current task weights (positive, unnormalized)."""
        w = torch.exp(self.log_weights)
        # Normalize around mean=1.0 for numerical stability
        return w * self.num_tasks / w.sum()

    def update(
        self,
        losses: torch.Tensor,
        shared_params: List[nn.Parameter],
    ) -> torch.Tensor:
        """
        Compute GradNorm loss and update task weights.

        Args:
            losses:         (num_tasks,) tensor of per-task losses
            shared_params:  list of backbone parameters with requires_grad=True

        Returns:
            normalized_weights: (num_tasks,) summing to num_tasks (for logging)
        """
        self._update_count += 1

        # 1. Store initial losses on first call
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            # Return current weights without update
            return self.weights.detach()

        # 2. Filter only trainable shared params
        trainable_shared = [p for p in shared_params if p.requires_grad]
        if not trainable_shared:
            return self.weights.detach()

        # 3. Get current task weights
        weights = self.weights

        # 4. Per-task gradient norms w.r.t. the shared backbone.
        #    ‖∂(w_i·L_i)/∂θ‖ = w_i·‖∂L_i/∂θ‖ — the task weight is a scalar
        #    factor, so its gradient path is analytic and does NOT need to run
        #    through the backbone. Differentiating the norm through θ instead
        #    (create_graph=True) built a full double-backward graph per task on
        #    top of the still-retained PCGrad graph, and with gradient
        #    checkpointing each one also re-ran every block's forward: 3
        #    backbone-sized graphs alive at once, which OOMs a 4 GB card at the
        #    first GradNorm step that gets past the L(0) early return.
        #    Detaching ‖∂L_i/∂θ‖ and multiplying by the live `weights` gives
        #    the identical gradient w.r.t. log_weights for a fraction of the
        #    memory.
        grad_norms = []
        for i in range(self.num_tasks):
            grads = torch.autograd.grad(
                outputs=losses[i],
                inputs=trainable_shared,
                # The last task no longer needs the graph — freeing it here
                # releases the buffers PCGrad retained for us.
                retain_graph=(i < self.num_tasks - 1),
                create_graph=False,
                allow_unused=True,
            )
            sq = torch.zeros((), device=self.device, dtype=torch.float32)
            for g in grads:
                if g is not None:
                    sq = sq + g.detach().float().pow(2).sum()
            grad_norms.append(sq.sqrt())
            del grads

        # weights is live (grad flows to log_weights); the norms are constants.
        norms = weights * torch.stack(grad_norms).detach()  # (num_tasks,)

        # 5. Compute loss ratios: L_i(t) / L_i(0)
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)

        # 6. Compute target gradient norms
        #    target_i = mean_norm × r_i^alpha
        mean_norm = norms.mean().detach()
        target_norms = mean_norm * (loss_ratios ** self.alpha)

        # 7. GradNorm loss: L1 distance between actual and target norms
        gn_loss = F.l1_loss(norms, target_norms.detach())

        # 8. Update log_weights.
        #    Taking the grad explicitly w.r.t. log_weights — rather than
        #    gn_loss.backward() — keeps GradNorm's update confined to the task
        #    weights. log_weights is not in the main optimizer's param groups,
        #    so a stray .backward() here would leave gradient in tensors the
        #    main optimizer *does* own and get applied at the next accumulation
        #    boundary. check_train_step.py asserts the resulting backbone .grad
        #    drift across this call is exactly 0.
        gn_grad, = torch.autograd.grad(gn_loss, [self.log_weights])
        self.optimizer.zero_grad(set_to_none=True)
        self.log_weights.grad = gn_grad
        self.optimizer.step()

        # 9. Re-normalize weights around mean=1.0 (cosmetic, for stability)
        with torch.no_grad():
            w = torch.exp(self.log_weights)
            self.log_weights.data = torch.log(w * self.num_tasks / w.sum())

        return self.weights.detach()

    def state_dict(self) -> dict:
        """Everything needed to resume task balancing where it left off.

        `initial_losses` matters as much as the weights: GradNorm's target is
        each task's loss *relative to its own L(0)*. Re-seeding L(0) from
        mid-training losses on resume resets every training-rate ratio to ~1,
        so balancing restarts from a baseline that no longer corresponds to
        the start of training.
        """
        return {
            "log_weights": self.log_weights.detach().cpu(),
            "initial_losses": (None if self.initial_losses is None
                               else self.initial_losses.detach().cpu()),
            "update_count": self._update_count,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            self.log_weights.copy_(state["log_weights"].to(self.device))
        init = state.get("initial_losses")
        self.initial_losses = None if init is None else init.to(self.device)
        self._update_count = state.get("update_count", 0)
        self.optimizer.load_state_dict(state["optimizer"])


# ──────────────────────────────────────────────────────────────────────────────
# PCGrad
# ──────────────────────────────────────────────────────────────────────────────

def compute_pcgrad_grads(
    losses: List[torch.Tensor],
    params: List[nn.Parameter],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accum_factor: float = 1.0,
    retain_graph_last: bool = False,
) -> None:
    """
    PCGrad: Project Conflicting Gradients.
    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

    Computes one gradient per task, projects each onto the normal plane of any
    task it conflicts with, and *accumulates* the conflict-free sum into
    ``param.grad`` so it composes with gradient accumulation.

    AMP: the gradients are left multiplied by the scaler's current scale, so the
    caller must go through ``scaler.unscale_(opt)`` → clip → ``scaler.step(opt)``
    → ``scaler.update()``. That keeps the scaler's inf/nan detection and dynamic
    backoff working. Unscaling here instead would silently bypass both, and with
    float16 the fixed initial scale (65536) overflows often enough that the
    resulting inf gradients would poison the weights on the first bad step.
    Projection is unaffected: scaling every gradient by s scales ``dot`` and
    ``norm()**2`` by s² alike, so the projection coefficient is unchanged.

    Args:
        losses:            K task-specific scalar losses (already task-weighted)
        params:            model parameters to compute gradients for
        scaler:            active GradScaler, or None when AMP is off
        accum_factor:      1 / grad_accum_steps
        retain_graph_last: keep the graph alive after the final task's backward
                           (needed when GradNorm runs on this step)
    """
    num_tasks = len(losses)
    trainable_params = [p for p in params if p.requires_grad]
    if not trainable_params:
        return

    use_scaler = scaler is not None and scaler.is_enabled()

    # 1. Per-task gradients, flattened into one vector each.
    task_grads = []
    for k, loss in enumerate(losses):
        # scaler.scale() both multiplies by the current scale and lazily creates
        # the scaler's internal scale tensor — unscale_()/step()/update() raise
        # if that never happened, which is why get_scale() alone is not enough.
        scaled_loss = scaler.scale(loss) if use_scaler else loss
        is_last = k == num_tasks - 1
        grads = torch.autograd.grad(
            scaled_loss,
            trainable_params,
            retain_graph=(not is_last) or retain_graph_last,
            allow_unused=True,
        )
        task_grads.append(torch.cat([
            g.reshape(-1) if g is not None
            else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            for g, p in zip(grads, trainable_params)
        ]))

    # 2. Project conflicting gradients onto each other's normal plane.
    pc_grads = [g.clone() for g in task_grads]
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                continue
            dot = torch.dot(pc_grads[i], task_grads[j])
            if dot < 0:
                coeff = dot / (task_grads[j].norm() ** 2 + 1e-8)
                # add_(alpha=) instead of `- coeff * g` avoids materialising a
                # second full-size gradient vector per projection.
                pc_grads[i].add_(task_grads[j], alpha=-coeff.item())

    # 3. Sum across tasks, reusing pc_grads[0]'s buffer rather than stacking —
    #    a stack would allocate another num_tasks x num_params vector, which is
    #    ~120 MB of VRAM on this model for no benefit.
    combined_grad = pc_grads[0]
    for g in pc_grads[1:]:
        combined_grad.add_(g)
    combined_grad.mul_(accum_factor)

    # 4. Accumulate into p.grad.
    idx = 0
    for p in trainable_params:
        num_elem = p.numel()
        g_slice = combined_grad[idx : idx + num_elem].view_as(p)
        if p.grad is None:
            p.grad = g_slice.clone()
        else:
            p.grad.add_(g_slice)
        idx += num_elem


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
class Trainer:
    """Main training loop for MTL deepfake/anti-spoof model."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg    = cfg
        self.logger = logger
        tc = cfg.train
        mc = cfg.model

        # ── Device ────────────────────────────────────────────────────
        self.device = self._resolve_device(tc)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # ── Run ID & directories ──────────────────────────────────────
        ckpt_root   = cfg.paths.checkpoint_dir
        self.run_id = self._get_run_id(tc.resume, ckpt_root)
        self.run_dir = os.path.join(ckpt_root, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.logger.info(f"Run directory: {self.run_dir}")

        self.result_logger = ResultLogger(
            os.path.join(self.run_dir, cfg.paths.result_csv)
        )

        # ── Model ─────────────────────────────────────────────────────
        self.model = self._build_model()

        # ── Multi-GPU ─────────────────────────────────────────────────
        self.use_multi_gpu = False
        self.gpu_ids = (
            tc.gpu_ids if tc.gpu_ids
            else list(range(torch.cuda.device_count()))
        )
        if tc.use_data_parallel and self.device.type == "cuda" and len(self.gpu_ids) > 1:
            self.use_multi_gpu = True
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.logger.info(f"DataParallel on GPUs: {self.gpu_ids}")

        # ── Loss functions ────────────────────────────────────────────
        self.criterion_df = nn.BCEWithLogitsLoss()
        self.criterion_sp = FocalLoss(gamma=tc.focal_gamma, alpha=tc.focal_alpha)
        # Supervises TemporalHead.clf, the branch validate() scores. The cosine
        # term alone leaves it untrained (verified: 4 tensors with grad=None).
        self.temporal_logit_weight = getattr(
            cfg.model, "temporal_logit_loss_weight", 1.0
        )
        # Supervises TemporalHead.flow_encoder. Zero unless flow is actually
        # loaded, so the encoder never receives a loss term it has no input for.
        self.flow_loss_weight = (
            float(tc.flow_loss_weight) if tc.use_optical_flow else 0.0
        )
        if tc.use_optical_flow and mc.temporal_supervision == "cosine_sim":
            # flow_encoder is None under this supervision mode, so TemporalHead
            # would drop the loaded flow on the floor — the .npz reads would cost
            # I/O for nothing. Warn rather than raise: the run is still valid.
            self.logger.warning(
                "use_optical_flow=True but temporal_supervision='cosine_sim' — "
                "no flow encoder is built, so the loaded flow is ignored. Set "
                "temporal_supervision to 'optical_flow' or 'combined'."
            )

        # ── Optimizer (differential LR for backbone vs heads) ─────────
        base_model   = self.model.module if self.use_multi_gpu else self.model
        head_params  = (
            list(base_model.deepfake_head.parameters()) +
            list(base_model.spoof_head.parameters())    +
            list(base_model.temporal_head.parameters())
        )
        backbone_params = list(base_model.backbone.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": head_params,     "lr": tc.lr},
                {"params": backbone_params, "lr": tc.lr * mc.backbone_lr_scale},
            ],
            betas=tuple(tc.betas),
            weight_decay=tc.weight_decay,
        )
        # Exactly the parameters the optimizer steps. Gradient clipping and
        # PCGrad must use this list, not model.parameters(): log_weights is a
        # mirror of GradNorm's own weights and is deliberately absent from the
        # optimizer, so scaler.unscale_() never divides its gradient by the AMP
        # scale. Clipping over model.parameters() therefore measured a norm
        # inflated by the scale factor (~44649 instead of ~0.67) and scaled every
        # real gradient down by ~1e-4, throttling training to a standstill.
        self.optim_params = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]
        # GradNormManager owns the task weights; this copy is written with
        # copy_() under no_grad and must never accumulate a gradient itself.
        base_model.log_weights.requires_grad_(False)

        # ── AMP ───────────────────────────────────────────────────────
        self.use_amp = (self.device.type == "cuda") and (tc.amp_dtype in ("float16", "bfloat16"))
        if self.use_amp:
            _dtype         = torch.float16 if tc.amp_dtype == "float16" else torch.bfloat16
            self.scaler    = torch.cuda.amp.GradScaler(enabled=True)
            self.autocast_ctx = torch.cuda.amp.autocast(dtype=_dtype)
            self.logger.info(f"AMP enabled  dtype={tc.amp_dtype}")
        else:
            self.scaler       = torch.cuda.amp.GradScaler(enabled=False)
            self.autocast_ctx = contextlib.nullcontext()

        # ── GradNorm ──────────────────────────────────────────────────
        self.use_gradnorm = tc.use_gradnorm
        if self.use_gradnorm:
            init_w = [tc.w_deepfake, tc.w_spoof, tc.w_temporal]
            self.gradnorm_manager = GradNormManager(
                num_tasks=3,
                alpha=tc.gradnorm_alpha,
                lr=1e-3,
                device=self.device,
                init_weights=init_w,
            )
            base_model.task_weights = self.gradnorm_manager.weights
            self.logger.info("GradNorm enabled.")
        else:
            self.gradnorm_manager = None

        # ── PCGrad ────────────────────────────────────────────────────
        self.use_pcgrad = tc.use_pcgrad
        if self.use_pcgrad:
            self.logger.info("PCGrad enabled.")

        # ── Scheduler (with optional linear warmup) ───────────────────
        self.scheduler = self._build_scheduler()

        # ── Early stopping ────────────────────────────────────────────
        self.early_stopping: Optional[EarlyStopping] = None
        # "acer" is the one monitored metric where lower is better. Catching the
        # mismatch here is cheaper than discovering after 25 epochs that
        # best.pth tracked the worst ACER seen.
        if tc.early_stopping_metric == "acer" and tc.early_stopping_mode != "min":
            raise ValueError(
                "early_stopping_metric='acer' needs early_stopping_mode='min' "
                f"(got '{tc.early_stopping_mode}') — ACER is an error rate."
            )
        if tc.early_stopping_metric != "acer" and tc.early_stopping_mode != "max":
            raise ValueError(
                f"early_stopping_metric='{tc.early_stopping_metric}' is a score, "
                f"so early_stopping_mode must be 'max' (got '{tc.early_stopping_mode}')"
            )
        self.monitor_mode = tc.early_stopping_mode
        if tc.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                mode=tc.early_stopping_mode,
                metric_key=tc.early_stopping_metric,
                min_epochs=tc.early_stopping_min_epochs,
                smooth_window=tc.early_stopping_smooth_window,
            )
            self.logger.info(
                f"Early stopping: metric={tc.early_stopping_metric}  "
                f"mode={tc.early_stopping_mode}  "
                f"patience={tc.early_stopping_patience}  "
                f"min_delta={tc.early_stopping_min_delta}  "
                f"min_epochs={tc.early_stopping_min_epochs}  "
                f"smooth_window={tc.early_stopping_smooth_window}"
            )
        if tc.early_stopping_metric == "composite":
            self.logger.info(
                f"Composite metric form: {tc.composite_metric} "
                f"({'0.5*ff_df_auc_roc + 0.5*siw_sp_auc_roc' if tc.composite_metric == 'auc' else '0.5*ff_df_auc_roc + 0.5*(1 - siw_sp_acer)'})"
            )

        # ── Power monitor (shared with preprocessing.py) ──────────────
        pc = cfg.power
        self.power_monitor = power_monitor_from_config(
            cfg,
            csv_path=os.path.join(self.run_dir, "power_train.csv"),
        )
        # Restrict to the GPUs this run actually trains on, unless the config
        # names an explicit set.
        if pc.monitor_all_gpus and not pc.gpu_ids and self.gpu_ids:
            self.power_monitor = PowerMonitor(
                poll_interval=pc.poll_interval_sec,
                rapl_path=pc.rapl_path,
                ram_coeff=pc.ram_coeff,
                ssd_coeff=pc.ssd_coeff,
                other_coeff=pc.other_coeff,
                gpu_ids=self.gpu_ids,
                per_gpu_log=pc.per_gpu_log,
                co2_kg_per_kwh=pc.co2_kg_per_kwh,
                overhead_multiplier=pc.overhead_multiplier,
                csv_path=os.path.join(self.run_dir, "power_train.csv"),
            )

        # ── Mutable state ─────────────────────────────────────────────
        self.start_epoch  = 0
        # Sign matches monitor_mode so the first epoch is always an improvement
        # in both directions (an ACER-monitored run starts from +inf).
        self.best_metric  = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch   = 0
        self.history: List[Dict] = []          # kept in-memory; persisted via _save_history
        self.current_batch_size = tc.batch_size
        self._warned_no_compression = False

        self._load_checkpoint()

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self, tc) -> torch.device:
        """
        Resolve the training device from cfg.train.device.
        Falls back to CPU if CUDA is unavailable and fallback_to_cpu=True.
        """
        requested = tc.device  # e.g. "cuda", "cuda:0", "cuda:0,1", "cpu"

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                # Use only the primary device for .to(); DataParallel handles the rest
                primary = requested.split(",")[0]
                device  = torch.device(primary)
                self.logger.info(
                    f"Device: {device}  |  GPUs available: {torch.cuda.device_count()}"
                )
                return device
            if tc.fallback_to_cpu:
                self.logger.warning(
                    "CUDA requested but unavailable — falling back to CPU."
                )
                return torch.device("cpu")
            raise RuntimeError(
                "CUDA requested but not available, and fallback_to_cpu=False."
            )

        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        model = MTLModel(self.cfg).to(self.device)
        self.logger.info(f"Backbone features: {model.backbone.num_features}")
        self._log_model_info(model)
        return model


    def _log_model_info(self, model: Optional[nn.Module] = None) -> None:
        """Log total and trainable parameter counts."""
        m = model if model is not None else (
            self.model.module if self.use_multi_gpu else self.model
        )
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.logger.info(
            f"Parameters — total: {total:,}  trainable: {trainable:,}"
        )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(self):
        """
        Build the LR scheduler.
        If cfg.train.warmup_epochs > 0, wraps the main scheduler with
        a linear warmup using SequentialLR.
        """
        tc = self.cfg.train

        # ── Main scheduler ────────────────────────────────────────────
        if tc.scheduler == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=tc.num_epochs - tc.warmup_epochs,
                eta_min=tc.min_lr,
            )
        elif tc.scheduler == "step":
            main_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=tc.step_size,
                gamma=tc.gamma,
            )
        else:
            main_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=5,
                factor=0.5,
                min_lr=tc.min_lr,
            )
            # ReduceLROnPlateau doesn't support SequentialLR; apply warmup manually
            if tc.warmup_epochs > 0:
                self.logger.warning(
                    "warmup_epochs is set but ReduceLROnPlateau is selected — "
                    "warmup will be applied manually in fit()."
                )
            return main_sched

        # ── Optional linear warmup ────────────────────────────────────
        if tc.warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=tc.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[tc.warmup_epochs],
            )
            self.logger.info(
                f"Scheduler: {tc.warmup_epochs}-epoch linear warmup → {tc.scheduler}"
            )
        else:
            scheduler = main_sched
            self.logger.info(f"Scheduler: {tc.scheduler}")

        return scheduler

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
    ) -> None:
        """Save epoch checkpoint, always overwrite last.pth, optionally best.pth."""
        state = {
            "epoch":        epoch,
            "model":        self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "scheduler":    self.scheduler.state_dict(),
            "scaler":       self.scaler.state_dict(),
            "best_metric":  self.best_metric,
            "best_epoch":   self.best_epoch,
            "metrics":      metrics,
            "history":      self.history,
        }
        # model.state_dict() carries MTLModel.log_weights, but that is only a
        # mirror written by copy_(). The manager owns the live weights, its Adam
        # momentum and the L(0) baseline — without them a resumed run restarts
        # task balancing and immediately overwrites the restored mirror.
        if self.gradnorm_manager is not None:
            state["gradnorm"] = self.gradnorm_manager.state_dict()

        # Per-epoch snapshot (optional, controlled by cfg)
        if self.cfg.train.save_every_epoch:
            epoch_path = os.path.join(self.run_dir, f"epoch_{epoch:03d}.pth")
            torch.save(state, epoch_path)

        # Always keep the latest checkpoint
        torch.save(state, os.path.join(self.run_dir, "last.pth"))

        if is_best:
            torch.save(state, os.path.join(self.run_dir, "best.pth"))
            self.logger.info(f"  ★ New best checkpoint saved (epoch {epoch})")

    def _load_checkpoint(self) -> None:
        """Restore training state from last.pth if it exists in run_dir."""
        ckpt_path = os.path.join(self.run_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            return

        self.logger.info(f"Resuming from: {ckpt_path}")
        # weights_only defaults to True from torch 2.6 on, and these checkpoints
        # carry numpy scalars inside "metrics"/"history", so the default raises
        # UnpicklingError and resume=True could never load anything. The file is
        # written by _save_checkpoint in this same run directory — our own data,
        # not an untrusted download.
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # scheduler might not have state_dict for ReduceLROnPlateau in older saves
        try:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            self.logger.warning("Could not restore scheduler state.")

        self.scaler.load_state_dict(ckpt["scaler"])
        # Restore task balancing. Checkpoints written before gradnorm state was
        # persisted simply have no "gradnorm" key — keep the fresh manager then.
        if self.gradnorm_manager is not None and "gradnorm" in ckpt:
            self.gradnorm_manager.load_state_dict(ckpt["gradnorm"])
            base_model = self.model.module if self.use_multi_gpu else self.model
            with torch.no_grad():
                base_model.log_weights.copy_(
                    self.gradnorm_manager.log_weights.detach()
                )
            w = self.gradnorm_manager.weights.detach().cpu().numpy().round(4)
            self.logger.info(f"Restored GradNorm task weights: {w}")
        elif self.gradnorm_manager is not None:
            self.logger.warning(
                "Checkpoint has no GradNorm state — task weights and the L(0) "
                "baseline restart from their initial values."
            )

        self.start_epoch = ckpt["epoch"] + 1
        worst = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.best_metric = ckpt.get("best_metric", worst)
        self.best_epoch  = ckpt.get("best_epoch", 0)
        self.history     = ckpt.get("history", [])
        self.logger.info(
            f"Resumed at epoch {self.start_epoch}  |  best metric so far: {self.best_metric:.4f}"
        )

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        """
        Persist in-memory history list to a JSON file alongside the CSV.
        Called at the end of every epoch so progress survives crashes.
        """
        import json
        hist_path = os.path.join(self.run_dir, "history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Run-ID helpers
    # ------------------------------------------------------------------

    def _get_run_id(self, resume: bool, ckpt_root: str) -> str:
        os.makedirs(ckpt_root, exist_ok=True)
        run_dirs = sorted(
            d for d in os.listdir(ckpt_root)
            if d.startswith("run") and os.path.isdir(os.path.join(ckpt_root, d))
        )

        if resume and run_dirs:
            # Find the most recent run that has last.pth
            for run_id in reversed(run_dirs):
                if os.path.exists(os.path.join(ckpt_root, run_id, "last.pth")):
                    self.logger.info(f"Resuming run: {run_id}")
                    return run_id

        # New run
        max_num = 0
        for d in run_dirs:
            try:
                max_num = max(max_num, int(d.replace("run", "")))
            except ValueError:
                pass
        new_id = f"run{max_num + 1:02d}"
        self.logger.info(f"Starting new run: {new_id}")
        return new_id

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader) -> Dict:
        """
        Run full validation pass.
        Returns a flat dict of all task metrics ready for logging.

        Each head is scored only on the samples whose task it owns. The
        off-task label is a placeholder zero (see MTLDataset.__getitem__), so
        accumulating every head over every sample produced metrics for
        questions the data cannot answer — that is where run01's `ff_sp_acer`
        column of `nan` came from, and on a mixed loader it would have been
        worse than nan: a real number computed against fabricated labels.

        The temporal head is scored on everything by design; its label is
        genuine in both datasets.
        """
        self.model.eval()

        # Accumulators
        df_labels,  df_scores,  df_videos  = [], [], []
        sp_labels,  sp_scores,  sp_types   = [], [], []
        tmp_labels, tmp_scores              = [], []
        ds_labels,  ds_datasets             = [], []   # for per-compression breakdown
        running_loss = {"df": 0.0, "sp": 0.0, "temp": 0.0}
        loss_batches = {"df": 0, "sp": 0, "temp": 0}

        for batch in tqdm(loader, desc="Validate", leave=False):
            frames  = batch["frames"].to(self.device, non_blocking=True)
            df_lbl  = batch["deepfake_label"].to(self.device, non_blocking=True).float()
            sp_lbl  = batch["spoof_label"].to(self.device, non_blocking=True).float()
            tmp_lbl = batch["temporal_label"].to(self.device, non_blocking=True).float()
            task_id = batch["task_id"].to(self.device, non_blocking=True)
            videos  = batch.get("video_id", [""] * frames.size(0))
            datasets= batch.get("dataset",  [""] * frames.size(0))
            types   = batch.get("spoof_type", ["unknown"] * frames.size(0))
            # Present only when use_optical_flow is on and the .npz files exist.
            flow    = batch.get("flow")
            if flow is not None:
                flow = flow.to(self.device, non_blocking=True)

            m_df = task_id == TASK_DEEPFAKE
            m_sp = task_id == TASK_SPOOF

            with self.autocast_ctx:
                out = self.model(frames, flow=flow)

                df_logit  = out["deepfake_logit"].flatten()
                sp_logit  = out["spoof_logit"].flatten()
                tmp_logit = out["temp_logit"].flatten()

                if m_df.any():
                    l_df = self.criterion_df(df_logit[m_df], df_lbl[m_df])
                    running_loss["df"] += l_df.item()
                    loss_batches["df"] += 1
                if m_sp.any():
                    l_sp = self.criterion_sp(sp_logit[m_sp], sp_lbl[m_sp])
                    running_loss["sp"] += l_sp.item()
                    loss_batches["sp"] += 1
                l_temp = temporal_consistency_loss(
                    out["temp_proj"], tmp_lbl,
                    logit=tmp_logit,
                    logit_weight=self.temporal_logit_weight,
                    flow_score=out.get("flow_consistency"),
                    flow_weight=self.flow_loss_weight,
                )
            running_loss["temp"] += l_temp.item()
            loss_batches["temp"] += 1

            # Collect predictions — each head only over the samples it owns.
            m_df_np = m_df.cpu().numpy()
            m_sp_np = m_sp.cpu().numpy()

            df_scores.append(torch.sigmoid(df_logit).cpu().numpy()[m_df_np])
            df_labels.append(df_lbl.cpu().numpy()[m_df_np])
            df_videos.extend([v for v, keep in zip(videos, m_df_np) if keep])

            sp_scores.append(torch.sigmoid(sp_logit).cpu().numpy()[m_sp_np])
            sp_labels.append(sp_lbl.cpu().numpy()[m_sp_np])
            sp_types.extend([t for t, keep in zip(types, m_sp_np) if keep])

            tmp_scores.append(torch.sigmoid(tmp_logit).cpu().numpy())
            tmp_labels.append(tmp_lbl.cpu().numpy())

            ds_labels.extend(df_lbl.cpu().numpy()[m_df_np].tolist())
            ds_datasets.extend([d for d, keep in zip(datasets, m_df_np) if keep])


        # Concatenate
        df_scores  = np.concatenate(df_scores)
        df_labels  = np.concatenate(df_labels)
        sp_scores  = np.concatenate(sp_scores)
        sp_labels  = np.concatenate(sp_labels)
        tmp_scores = np.concatenate(tmp_scores)
        tmp_labels = np.concatenate(tmp_labels)

        # Compute metrics. EvalConfig drives the thresholds/labels so an
        # ablation can change them in one place instead of editing three
        # hardcoded literals inside the metric functions.
        ec = self.cfg.eval
        df_metrics = sp_metrics = {}
        if len(df_labels):
            df_metrics = compute_deepfake_metrics(
                df_labels, df_scores, df_videos,
                video_agg=ec.video_agg, threshold=ec.decision_threshold,
            )
        if len(sp_labels):
            sp_metrics = compute_spoof_metrics(
                sp_labels, sp_scores,
                threshold=ec.decision_threshold,
                fpr_threshold=ec.fpr_threshold,
                spoof_types=sp_types,
            )
        tmp_metrics = compute_temporal_metrics(
            tmp_labels, tmp_scores, threshold=ec.decision_threshold
        )
        comp_metrics = compute_deepfake_metrics_by_compression(
            np.array(ds_labels), df_scores, ds_datasets,
            tags=(ec.c23_label, ec.c40_label),
        )
        # Structurally unavailable with this raw layout rather than merely empty:
        # the FaceForensics++ DFD actor subset filenames carry no c23/c40 token,
        # so no row can ever match. Said once so the thesis can state why the
        # compression analysis is absent instead of it silently reading as "not run".
        if not comp_metrics and not self._warned_no_compression:
            self._warned_no_compression = True
            self.logger.info(
                "Compression breakdown unavailable: no '%s'/'%s' token in any "
                "video path. The raw videos here are the DFD actor subset "
                "(e.g. 01_02__hugging_happy__HASH.mp4), which is not "
                "compression-tagged.", ec.c23_label, ec.c40_label,
            )

        # A dead head must announce itself the epoch it dies, not 12 epochs
        # later when early stopping fires on a plateau.
        for name, m in (("deepfake head", df_metrics),
                        ("spoof head", sp_metrics),
                        ("temporal head", tmp_metrics)):
            for msg in collapse_warnings(name, m, threshold=ec.decision_threshold):
                self.logger.warning("COLLAPSE  %s", msg)

        val_losses = {
            k: running_loss[k] / max(loss_batches[k], 1) for k in running_loss
        }

        metrics = {
            "val_loss_df":   val_losses["df"],
            "val_loss_sp":   val_losses["sp"],
            "val_loss_temp": val_losses["temp"],
            **{f"df_{k}":  v for k, v in df_metrics.items()},
            **{f"sp_{k}":  v for k, v in sp_metrics.items()},
            **{f"tmp_{k}": v for k, v in tmp_metrics.items()},
            **{f"comp_{k}":v for k, v in comp_metrics.items()},
        }
        return metrics

    # ------------------------------------------------------------------
    # Epoch logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        elapsed: float,
        power_w: float,
        lr: float,
    ) -> None:
        """
        1. Append to in-memory history.
        2. Persist history.json.
        3. Write one row to result CSV via ResultLogger.
        4. Print a concise summary to the logger.
        """
        tc = self.cfg.train

        # Assemble row
        base_model = self.model.module if self.use_multi_gpu else self.model
        w = base_model.task_weights.detach().cpu().tolist() if hasattr(base_model, "task_weights") else [
            tc.w_deepfake, tc.w_spoof, tc.w_temporal
        ]

        row = {
            "epoch":      epoch,
            "run_id":     self.run_id,
            "lr":         round(lr, 8),
            "elapsed_s":  round(elapsed, 1),
            "power_w":    round(power_w, 2),
            # Train losses
            "loss_total": round(train_metrics.get("loss_total", 0.0), 5),
            "loss_df":    round(train_metrics.get("loss_df",    0.0), 5),
            "loss_sp":    round(train_metrics.get("loss_sp",    0.0), 5),
            "loss_temp":  round(train_metrics.get("loss_temp",  0.0), 5),
            # Task weights
            "w_df":  round(w[0], 4),
            "w_sp":  round(w[1], 4),
            "w_temp":round(w[2], 4),
            **val_metrics,
        }

        self.history.append(row)
        self._save_history()
        self.result_logger.log(row)

        # ── Console summary ───────────────────────────────────────────
        # fit() double-prefixes: validate() emits "df_auc_roc"/"sp_acer"/
        # "tmp_bin_acc", then the per-loader merge prepends "ff_"/"siw_". The
        # old "df_auc"/"tmp_acc" lookups matched neither layer and printed nan
        # on every epoch line of every run. Deepfake AUC comes from the FF++
        # loader and the spoof metrics from the SiW-Mv2 loader — the other
        # pairing is meaningless (no attacks in FF++, no fakes in SiW-Mv2).
        #
        # sp_f1 and sp_recall are on this line rather than only in the CSV
        # because they are the two numbers that distinguish a working
        # anti-spoof head from a dead one. ACER stayed at exactly 0.5 through
        # run01's entire collapse while both of these were 0.0.
        nan = float("nan")
        df_auc  = val_metrics.get("ff_df_auc_roc",  nan)
        sp_auc  = val_metrics.get("siw_sp_auc_roc", nan)
        sp_rec  = val_metrics.get("siw_sp_recall",  nan)
        sp_f1   = val_metrics.get("siw_sp_f1",      nan)
        tmp_acc = val_metrics.get("ff_tmp_bin_acc", nan)

        self.logger.info(
            f"[{epoch:03d}/{tc.num_epochs}]  "
            f"loss={row['loss_total']:.4f}  "
            f"df_auc={df_auc:.4f}  "
            f"sp_auc={sp_auc:.4f}  "
            f"sp_rec={sp_rec:.4f}  "
            f"sp_f1={sp_f1:.4f}  "
            f"tmp_acc={tmp_acc:.4f}  "
            f"comp={val_metrics.get('composite', nan):.4f}  "
            f"lr={lr:.2e}  "
            f"t={elapsed:.0f}s  "
            f"P={power_w:.1f}W"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_ff_loader, val_siw_loader) -> None:
        tc = self.cfg.train
        log_banner(self.logger, f"Training  run={self.run_id}")

        self.power_monitor.start()

        for epoch in range(self.start_epoch, tc.num_epochs):
            t0 = time.time()

            # ── Train one epoch ───────────────────────────────────────
            train_metrics = self._train_epoch(epoch, train_loader)

            # ── Validate (FF++ → deepfake metrics, SiW → spoof metrics) ──
            ff_metrics  = self.validate(val_ff_loader)
            siw_metrics = self.validate(val_siw_loader)

            # prefix keys to avoid collision, then merge
            val_metrics = (
                {f"ff_{k}":  v for k, v in ff_metrics.items()}  |
                {f"siw_{k}": v for k, v in siw_metrics.items()}
            )

            # ── Composite metric ──────────────────────────────────────
            # Keys must match validate()'s prefixed output:
            # compute_deepfake_metrics returns "auc_roc" -> "df_auc_roc".
            # Reading "df_auc" hit the 0.0 default on every single epoch, so
            # composite collapsed to 0.5 * (1 - sp_acer): best.pth and early
            # stopping were driven by anti-spoofing alone and the deepfake
            # task had no influence on model selection at all.
            composite, comp_terms = self._composite_metric(ff_metrics, siw_metrics)
            val_metrics.update(comp_terms)

            # ── Scheduler step ────────────────────────────────────────
            # Feed the plateau scheduler the same composite that drives
            # best-model tracking and early stopping, so LR reduction cannot
            # disagree with model selection. It previously received
            # ff_metrics["df_auc"] — a key that does not exist — so it saw a
            # constant 0.0, never registered an improvement under mode="max",
            # and walked the LR down to min_lr on a fixed schedule regardless
            # of how the model was actually doing.
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(composite)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t0
            power_w    = float(np.mean(self.power_monitor.readings)) if self.power_monitor.readings else 0.0

            # ── Best-model tracking ───────────────────────────────────
            # On the raw monitored value, not the smoothed one early stopping
            # uses: smoothing exists to stop a lucky epoch from resetting
            # patience, not to reject a genuinely best checkpoint.
            is_best = (
                composite > self.best_metric if self.monitor_mode == "max"
                else composite < self.best_metric
            )
            if is_best:
                self.best_metric = composite
                self.best_epoch  = epoch

            # ── Checkpoint ────────────────────────────────────────────
            self._save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best=is_best)

            # ── Logging ───────────────────────────────────────────────
            self._log_epoch(epoch, train_metrics, val_metrics, elapsed, power_w, current_lr)

            # ── Early stopping ────────────────────────────────────────
            if self.early_stopping and self.early_stopping.step(composite):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}  "
                    f"(best epoch: {self.best_epoch})"
                )
                break

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # ── End of training ───────────────────────────────────────────
        avg_power = self.power_monitor.stop()
        self.power_monitor.log_summary(self.logger)
        log_banner(
            self.logger,
            f"Done  best={self.best_metric:.4f} @ epoch {self.best_epoch}  "
            f"avg_power={avg_power:.1f}W"
        )


    # ------------------------------------------------------------------
    # Single train epoch
    # ------------------------------------------------------------------

    def _composite_metric(
        self, ff_metrics: Dict, siw_metrics: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """The number that drives best.pth, the plateau scheduler and early stopping.

        AUC comes from the FF++ loader and the anti-spoofing terms from the
        SiW-Mv2 loader; the other pairing is meaningless (no attacks in FF++, no
        face swaps in SiW-Mv2).

        Both composite forms are always returned for logging, because they
        disagree in exactly the case that matters. The legacy "acer" form is
        `0.5 * ff_df_auc + 0.5 * (1 - siw_sp_acer)`, and ACER is exactly 0.5 for
        a head that predicts one class for everything — so in run01 the second
        term was frozen at 0.25 for all 14 epochs while the same collapsed head
        had an AUC of 0.41, i.e. visibly below chance. The "auc" form is the
        default for that reason.

        Returns (value, extra metric keys to log).
        """
        tc = self.cfg.train
        df_auc  = ff_metrics.get("df_auc_roc")
        sp_auc  = siw_metrics.get("sp_auc_roc")
        sp_acer = siw_metrics.get("sp_acer")

        missing = [
            name for name, v in
            (("ff_df_auc_roc", df_auc), ("siw_sp_auc_roc", sp_auc),
             ("siw_sp_acer", sp_acer))
            if v is None or not np.isfinite(v)
        ]
        if missing:
            self.logger.warning(
                "composite: %s missing or non-finite — metrics guarded on a "
                "single-class split return nothing; missing terms fall back to "
                "their worst value. Check the split's class balance.",
                ", ".join(missing),
            )

        def ok(v, worst):
            return v if v is not None and np.isfinite(v) else worst

        comp_auc  = 0.5 * ok(df_auc, 0.0) + 0.5 * ok(sp_auc, 0.0)
        comp_acer = 0.5 * ok(df_auc, 0.0) + 0.5 * (1.0 - ok(sp_acer, 1.0))

        terms = {
            "composite":            comp_auc if tc.composite_metric == "auc" else comp_acer,
            "composite_auc_form":   comp_auc,
            "composite_acer_form":  comp_acer,
        }

        # early_stopping_metric selects what is monitored. It used to be a dead
        # field: fit() always passed the composite and EarlyStopping.metric_key
        # was only ever used in log messages.
        key = tc.early_stopping_metric
        if key == "df_auc":
            value = ok(df_auc, 0.0)
        elif key == "sp_auc":
            value = ok(sp_auc, 0.0)
        elif key == "acer":
            value = ok(sp_acer, 1.0)
        else:                                   # "composite"
            value = terms["composite"]
        terms["monitored"] = value
        return value, terms

    def _train_epoch(self, epoch: int, loader) -> Dict:
        """Run one full training epoch, return averaged loss dict."""
        self.model.train()
        tc         = self.cfg.train
        base_model = self.model.module if self.use_multi_gpu else self.model
    
        accum = {"total": 0.0, "df": 0.0, "sp": 0.0, "temp": 0.0}
        n     = 0
    
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Train {epoch:03d}", leave=False)
        for step, batch in enumerate(pbar):
            frames  = batch["frames"].to(self.device, non_blocking=True)
            df_lbl  = batch["deepfake_label"].to(self.device, non_blocking=True).float()
            sp_lbl  = batch["spoof_label"].to(self.device, non_blocking=True).float()
            tmp_lbl = batch["temporal_label"].to(self.device, non_blocking=True).float()
            task_id = batch["task_id"].to(self.device, non_blocking=True)
            # Read from disk by the dataset — nothing here computes flow.
            flow    = batch.get("flow")
            if flow is not None:
                flow = flow.to(self.device, non_blocking=True)

            # Each supervised head sees only the samples that carry a real label
            # for it. The off-task label is a placeholder zero, and training on
            # it taught the spoof head "a deepfake face is bona-fide" and the
            # deepfake head "a presentation attack is real" — measured on run01's
            # best.pth, the deepfake head fired above 0.5 on 12.2% of SiW-Mv2
            # clips that are all label 0 for it.
            #
            # InterleavedBatchSampler guarantees both masks are non-empty in
            # training, so the empty-mask branch below is only reached by an
            # oddly-configured loader — it returns a real graph node scaled to
            # zero rather than nan, which keeps PCGrad and GradNorm well defined.
            m_df = task_id == TASK_DEEPFAKE
            m_sp = task_id == TASK_SPOOF

            with self.autocast_ctx:
                out    = self.model(frames, flow=flow)

                l_df = (
                    self.criterion_df(out["deepfake_logit"].flatten()[m_df], df_lbl[m_df])
                    if m_df.any() else out["deepfake_logit"].sum() * 0.0
                )
                l_sp = (
                    self.criterion_sp(out["spoof_logit"].flatten()[m_sp], sp_lbl[m_sp])
                    if m_sp.any() else out["spoof_logit"].sum() * 0.0
                )
                # Temporal stays over the whole batch: "inauthentic" is genuine
                # ground truth in both datasets, so there is nothing to mask.
                l_temp = temporal_consistency_loss(
                    out["temp_proj"], tmp_lbl,
                    logit=out["temp_logit"],
                    logit_weight=self.temporal_logit_weight,
                    flow_score=out.get("flow_consistency"),
                    flow_weight=self.flow_loss_weight,
                )

                # Task weights
                w    = base_model.task_weights                # (3,) normalized
                loss = (w[0] * l_df + w[1] * l_sp + w[2] * l_temp) / tc.grad_accum_steps
    
            # ── Backward ──────────────────────────────
            run_gradnorm_now = self.use_gradnorm and (step % tc.gradnorm_interval == 0)

            if self.use_pcgrad:
                compute_pcgrad_grads(
                    [w[0] * l_df, w[1] * l_sp, w[2] * l_temp],
                    self.optim_params,
                    scaler=self.scaler if self.use_amp else None,
                    accum_factor=1.0 / tc.grad_accum_steps,
                    retain_graph_last=run_gradnorm_now,
                )
            else:
                self.scaler.scale(loss).backward(
                    retain_graph=run_gradnorm_now
                )

            if run_gradnorm_now:
                losses_tensor = torch.stack([l_df, l_sp, l_temp])
                new_w = self.gradnorm_manager.update(
                    losses_tensor,
                    list(base_model.backbone.parameters()),
                )
                with torch.no_grad():
                    base_model.log_weights.copy_(
                        torch.log(new_w.clamp(min=1e-8))
                    )

            is_accum_step = (step + 1) % tc.grad_accum_steps == 0
            is_last_step  = (step + 1) == len(loader)

            if is_accum_step or is_last_step:
                # Both branches leave p.grad scaled by the AMP scale, so the
                # unscale → clip → step → update sequence is shared. Going
                # through the scaler is what makes an overflowed step get
                # skipped instead of writing inf/nan into the weights.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.optim_params, tc.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)
    
            # ── Accumulate metrics ────────────────────────────────
            accum["total"] += loss.item() * tc.grad_accum_steps   # مقدار واقعی loss
            accum["df"]    += l_df.item()
            accum["sp"]    += l_sp.item()
            accum["temp"]  += l_temp.item()
            n += 1
    
            pbar.set_postfix(loss=f"{loss.item() * tc.grad_accum_steps:.4f}")
    
        denom = max(n, 1)
        return {
            "loss_total": accum["total"] / denom,
            "loss_df":    accum["df"]    / denom,
            "loss_sp":    accum["sp"]    / denom,
            "loss_temp":  accum["temp"]  / denom,
        }
    
# ──────────────────────────────────────────────────────────────────────────────
# Data Loading Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a dataset CSV and verify required columns."""
    required = ["frame_path", "video_path", "dataset", "label",
                "task", "split", "video_index", "clip_index", "frame_num"]
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def build_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build all train/val loaders."""
    pc = cfg.paths

    ff_train = load_csv(pc.ff_train_csv)
    ff_val = load_csv(pc.ff_val_csv)
    siw_train = load_csv(pc.siwmv2_train_csv)
    siw_val = load_csv(pc.siwmv2_val_csv)

    train_loader = build_interleaved_loader(ff_train, siw_train, cfg, is_train=True)

    # Val loaders — separate per dataset for clean metrics
    val_ff_ds = MTLDataset(ff_val, cfg, is_train=False)
    val_siw_ds = MTLDataset(siw_val, cfg, is_train=False)

    val_ff_loader = DataLoader(
        val_ff_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available()
    )
    val_siw_loader = DataLoader(
        val_siw_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available()
    )

    return train_loader, val_ff_loader, val_siw_loader


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Main entry point: setup, build data, train."""
    cfg = get_config()

    # Setup output dir
    os.makedirs(cfg.paths.output_root, exist_ok=True)
    log_path = os.path.join(cfg.paths.checkpoint_dir, cfg.paths.log_file)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    logger = setup_logger(log_path, cfg=cfg)
    set_seed(cfg.train.seed)

    log_banner(logger, "MTL Training: Deepfake | Anti-Spoof | Temporal")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    logger.info("Loading datasets...")
    train_loader, val_ff_loader, val_siw_loader = build_loaders(cfg)
    logger.info(
        f"Train batches: {len(train_loader)} | "
        f"Val FF++: {len(val_ff_loader)} | Val SiW: {len(val_siw_loader)}"
    )

    # Train
    trainer = Trainer(cfg, logger)
    trainer.fit(train_loader, val_ff_loader, val_siw_loader)


if __name__ == "__main__":
    main()
