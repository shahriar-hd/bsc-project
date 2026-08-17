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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

from src.config import Config, get_config
from src.utils.logger_utils import log_banner, setup_logger
from src.utils.power_utils import GPUSample, PowerMonitor, power_monitor_from_config
from src.utils.repro_utils import set_seed, worker_init_fn

warnings.filterwarnings("ignore")


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
            A.ImageCompression(
                quality_lower=ac.jpeg_quality_min,
                quality_upper=ac.jpeg_quality_max,
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

    def _build_clips(self, df: pd.DataFrame) -> List[dict]:
        """Group rows by (video_index, clip_index) and build clip records."""
        clips = []
        group_cols = ["video_index", "clip_index", "video_path"]
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
            })
        return clips

    def _sample_frames(self, paths: List[str]) -> List[str]:
        """Sample T frames with optional random jitter."""
        tc = self.cfg.train
        N = len(paths)
        if N <= self.T:
            # Repeat to fill
            chosen = (paths * ((self.T // N) + 1))[:self.T]
        else:
            if self.is_train and tc.temporal_jitter:
                gap = random.randint(tc.min_frame_gap, tc.max_frame_gap)
                max_start = N - gap * (self.T - 1) - 1
                if max_start < 0:
                    indices = np.linspace(0, N - 1, self.T, dtype=int)
                else:
                    start = random.randint(0, max_start)
                    indices = [min(start + i * gap, N - 1) for i in range(self.T)]
            else:
                indices = np.linspace(0, N - 1, self.T, dtype=int)
            chosen = [paths[i] for i in indices]
        return chosen

    def _load_frame(self, path: str) -> np.ndarray:
        """Load and augment a single frame."""
        img = np.array(Image.open(path).convert("RGB"))
        return img

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        frame_paths = self._sample_frames(clip["frame_paths"])

        frames = []
        for fp in frame_paths:
            img = self._load_frame(fp)
            aug = self.transform(image=img)["image"]
            frames.append(aug)

        frames_tensor = torch.stack(frames, dim=0)

        # ── Task-specific labels ──────────────────────────────────
        task = clip["task"]
        binary_label = clip["label"]  # 0=real/live, 1=fake/spoof

        if task == "deepfake":
            deepfake_label = binary_label
            spoof_label = 0  # placeholder
        else:  # anti-spoof
            deepfake_label = 0  # placeholder
            spoof_label = binary_label

        temporal_label = binary_label  # temporal

        return {
            "frames": frames_tensor,
            "deepfake_label": torch.tensor(deepfake_label, dtype=torch.long),
            "spoof_label": torch.tensor(spoof_label, dtype=torch.long),
            "temporal_label": torch.tensor(temporal_label, dtype=torch.long),
            "task": task,
            "dataset": clip["dataset"],
            "video_id": clip["video_path"],  # video_path 
        }



def build_interleaved_loader(
    ff_df: pd.DataFrame,
    siwmv2_df: pd.DataFrame,
    cfg: Config,
    is_train: bool
) -> DataLoader:
    """
    Build a DataLoader that interleaves FF++ and SiW-Mv2 samples
    via WeightedRandomSampler so each batch has both datasets.
    """
    ff_ds = MTLDataset(ff_df, cfg, is_train)
    siw_ds = MTLDataset(siwmv2_df, cfg, is_train)

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([ff_ds, siw_ds])

    ratio = cfg.train.ff_sample_ratio
    # Equal weight within each dataset, ratio controls dataset balance
    ff_weight = ratio / max(len(ff_ds), 1)
    siw_weight = (1 - ratio) / max(len(siw_ds), 1)
    weights = (
        [ff_weight] * len(ff_ds) +
        [siw_weight] * len(siw_ds)
    )
    sampler = WeightedRandomSampler(
        weights, num_samples=len(combined), replacement=True
    ) if is_train else None

    return DataLoader(
        combined,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        shuffle=(not is_train and sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        drop_last=is_train,
    )


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
    Redesigned temporal consistency head with explicit supervision.

    Supports three modes:
      - 'pseudo_label':  binary label from mean of adjacent deepfake predictions
      - 'optical_flow':  consistency score from precomputed optical flow
      - 'combined':      both pseudo-label + flow auxiliary loss (recommended)

    Args:
        feature_dim:    backbone output dimension (e.g. 1408 for EfficientNet-B2)
        proj_dim:       temporal projection dimension
        hidden_dim:     classifier hidden dimension
        dropout:        dropout probability
        supervision:    'pseudo_label' | 'optical_flow' | 'combined'
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
            proj_dim=mc.temporal_hidden,
            dropout=mc.dropout
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


    def forward(
        self,
        frames: torch.Tensor,          # (B, T, C, H, W)
    ) -> dict:
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        if self.use_tsm:
            x = apply_tsm(x, self.tsm_shift_ratio, T)

        feats = self.backbone(x)       # (B*T, D)
        feats = feats.view(B, T, -1)  # (B, T, D)

        frame_feat = feats.mean(dim=1)  # (B, D) — temporal pooled

        df_logit = self.deepfake_head(frame_feat)
        sp_logit = self.spoof_head(frame_feat)
        temp_out   = self.temporal_head(feats)
        temp_proj  = temp_out["temp_proj"]
        temp_logit = temp_out["temp_logit"]

        return {
   		 "deepfake_logit":         df_logit,
   		 "spoof_logit":         sp_logit,
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
) -> torch.Tensor:
    """
    Self-supervised temporal loss:
    real clips -> minimize cosine distance between adjacent frames,
    fake clips -> maximize it.
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
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# PCGrad
# ──────────────────────────────────────────────────────────────────────────────

def pcgrad_step(
    grads: List[Optional[torch.Tensor]]
) -> List[Optional[torch.Tensor]]:
    """
    PCGrad: project conflicting gradients.
    grads: list of gradient tensors (one per task), each is a flat vector.
    Returns list of adjusted gradient vectors.
    """
    n = len(grads)
    valid = [g for g in grads if g is not None]
    if len(valid) < 2:
        return grads

    adjusted = [g.clone() if g is not None else None for g in grads]

    for i in range(n):
        if grads[i] is None:
            continue
        for j in range(n):
            if i == j or grads[j] is None:
                continue
            gi, gj = adjusted[i], grads[j]
            dot = (gi * gj).sum()
            if dot < 0:
                # Project out conflicting component
                adjusted[i] = gi - (dot / (gj.norm() ** 2 + 1e-8)) * gj

    return adjusted


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

def compute_deepfake_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    video_paths: Optional[List[str]] = None,
) -> dict:
    """
    Compute AUC-ROC, EER, AP, accuracy@best-threshold,
    and video-level AUC if video_paths provided.
    """
    results = {}
    if len(np.unique(labels)) < 2:
        return results

    results["auc_roc"] = roc_auc_score(labels, scores)
    results["ap"] = average_precision_score(labels, scores)

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

    # Video-level AUC
    if video_paths is not None:
        vdf = pd.DataFrame({
            "video": video_paths, "label": labels, "score": scores
        })
        vid_agg = vdf.groupby("video").agg({"label": "first", "score": "mean"})
        if vid_agg["label"].nunique() > 1:
            results["video_auc"] = roc_auc_score(
                vid_agg["label"], vid_agg["score"]
            )

    return results


def compute_deepfake_metrics_by_compression(
    labels: np.ndarray,
    scores: np.ndarray,
    datasets: List[str],    # or compression label column
) -> dict:
    """Compute AUC per compression type (c23, c40) based on dataset column."""
    results = {}
    ds_arr = np.array(datasets)
    for tag in ["c23", "c40"]:
        mask = np.array([tag.lower() in d.lower() for d in datasets])
        if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
            results[f"auc_{tag}"] = roc_auc_score(labels[mask], scores[mask])
    return results


def compute_spoof_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute APCER, BPCER, ACER, HTER, TPR@FPR=1%.
    labels: 0=real (bona-fide), 1=attack.
    """
    results = {}
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

    # TPR @ FPR=1%
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        tpr_at_1fpr = float(interp1d(fpr, tpr)(0.01)) if 0.01 <= fpr.max() else float("nan")
        results["tpr_at_fpr1"] = tpr_at_1fpr
        results["auc"] = roc_auc_score(labels, scores)

    return results


def compute_temporal_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Compute binary accuracy, AUC, and deepfake-AUC for temporal head."""
    results = {}
    preds = (scores >= 0.5).astype(int)
    results["bin_acc"] = (preds == labels).mean()
    if len(np.unique(labels)) > 1:
        results["auc"] = roc_auc_score(labels, scores)
        fake_mask = labels == 1
        real_mask = labels == 0
        if fake_mask.sum() > 0 and real_mask.sum() > 0:
            results["df_auc"] = roc_auc_score(labels, scores)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Result CSV Logger
# ──────────────────────────────────────────────────────────────────────────────

class ResultLogger:
    """Appends per-epoch metrics to a CSV file."""

    def __init__(self, csv_path: str):
        """Initialize with output CSV path."""
        self.path = csv_path
        self._header_written = os.path.exists(csv_path)

    def log(self, row: dict) -> None:
        """Append one row to the CSV."""
        mode = "a" if self._header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Monitors a validation metric and signals when training should stop.

    Supports both 'max' (e.g. AUC) and 'min' (e.g. ACER) modes.

    Args:
        patience:   number of epochs to wait after last improvement
        min_delta:  minimum change to qualify as an improvement
        mode:       'max' to maximize metric, 'min' to minimize
        metric_key: name of the metric in the val_metrics dict (for logging)
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
        metric_key: str = "primary",
    ) -> None:
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_key = metric_key

        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        return current < self.best_value - self.min_delta

    def step(self, current_value: float) -> bool:
        """
        Update state with the latest metric value.

        Args:
            current_value: metric value for the current epoch

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True

        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_value = state["best_value"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.should_stop = state["should_stop"]


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

        # 4. Compute per-task gradient norms w.r.t. shared backbone
        #    This is expensive: one backward pass per task
        norms = []
        for i in range(self.num_tasks):
            # Gradient of weighted task loss w.r.t. backbone
            grads = torch.autograd.grad(
                outputs=losses[i] * weights[i],
                inputs=trainable_shared,
                retain_graph=True,
                create_graph=True,  # needed to backprop through norm
                allow_unused=True,
            )
            # Compute L2 norm of gradients
            grad_norms = [g.norm(2) for g in grads if g is not None]
            if grad_norms:
                task_norm = torch.stack(grad_norms).norm(2)
            else:
                task_norm = torch.tensor(0.0, device=self.device)
            norms.append(task_norm)

        norms = torch.stack(norms)  # (num_tasks,)

        # 5. Compute loss ratios: L_i(t) / L_i(0)
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)

        # 6. Compute target gradient norms
        #    target_i = mean_norm × r_i^alpha
        mean_norm = norms.mean().detach()
        target_norms = mean_norm * (loss_ratios ** self.alpha)

        # 7. GradNorm loss: L1 distance between actual and target norms
        gn_loss = F.l1_loss(norms, target_norms.detach())

        # 8. Update log_weights
        self.optimizer.zero_grad()
        gn_loss.backward()
        self.optimizer.step()

        # 9. Re-normalize weights around mean=1.0 (cosmetic, for stability)
        with torch.no_grad():
            w = torch.exp(self.log_weights)
            self.log_weights.data = torch.log(w * self.num_tasks / w.sum())

        return self.weights.detach()

    def state_dict(self) -> dict:
        return {
            "log_weights": self.log_weights.data.cpu(),
            "initial_losses": self.initial_losses.cpu() if self.initial_losses is not None else None,
            "update_count": self._update_count,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.log_weights.data = state["log_weights"].to(self.device)
        if state["initial_losses"] is not None:
            self.initial_losses = state["initial_losses"].to(self.device)
        self._update_count = state.get("update_count", 0)
        self.optimizer.load_state_dict(state["optimizer"])


# ─── REPLACE pcgrad_step function در train.py:571-598 ───────────────────────

def pcgrad_step(
    losses: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    retain_graph: bool = False,
) -> None:
    """
    PCGrad: Project Conflicting Gradients.

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

    Computes gradient for each task, projects conflicting gradients onto
    the normal plane, and applies the conflict-free combined gradient.

    Args:
        losses:       list of K task-specific scalar losses
        optimizer:    the model optimizer (e.g. AdamW)
        retain_graph: whether to keep computation graph after backward
    """
    num_tasks = len(losses)

    # 1. Collect parameters from optimizer
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                params.append(p)

    if not params:
        return

    # 2. Compute per-task gradients
    task_grads = []
    for i, loss in enumerate(losses):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Flatten gradients into a single vector
        grad_vec = []
        for p in params:
            if p.grad is not None:
                grad_vec.append(p.grad.data.flatten().clone())
            else:
                grad_vec.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        task_grads.append(torch.cat(grad_vec))

    # 3. Project conflicting gradients
    #    For each pair (i, j): if dot(g_i, g_j) < 0, project g_i
    pc_grads = [g.clone() for g in task_grads]

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                continue
            dot_product = torch.dot(pc_grads[i], task_grads[j])
            if dot_product < 0:
                # Project g_i onto plane normal to g_j
                proj_component = dot_product / (task_grads[j].norm() ** 2 + 1e-8)
                pc_grads[i] = pc_grads[i] - proj_component * task_grads[j]

    # 4. Average projected gradients
    final_grad = torch.stack(pc_grads).mean(dim=0)

    # 5. Assign back to parameters
    optimizer.zero_grad()
    idx = 0
    for p in params:
        num_elem = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.data = final_grad[idx : idx + num_elem].view_as(p)
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
            weight_decay=tc.weight_decay,
        )

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
        if tc.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                mode=tc.early_stopping_mode,
                metric_key=tc.early_stopping_metric,
            )
            self.logger.info(
                f"Early stopping: metric={tc.early_stopping_metric}  "
                f"mode={tc.early_stopping_mode}  "
                f"patience={tc.early_stopping_patience}"
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
        self.best_metric  = -float("inf")
        self.best_epoch   = 0
        self.history: List[Dict] = []          # kept in-memory; persisted via _save_history
        self.current_batch_size = tc.batch_size

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
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # scheduler might not have state_dict for ReduceLROnPlateau in older saves
        try:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            self.logger.warning("Could not restore scheduler state.")

        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric = ckpt.get("best_metric", -float("inf"))
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
        """
        self.model.eval()
        base_model = self.model.module if self.use_multi_gpu else self.model

        # Accumulators
        df_labels,  df_scores,  df_videos  = [], [], []
        sp_labels,  sp_scores               = [], []
        tmp_labels, tmp_scores              = [], []
        ds_labels,  ds_datasets             = [], []   # for per-compression breakdown
        running_loss = {"df": 0.0, "sp": 0.0, "temp": 0.0}
        n_batches = 0

        for batch in tqdm(loader, desc="Validate", leave=False):
            frames  = batch["frames"].to(self.device, non_blocking=True)
            df_lbl  = batch["deepfake_label"].to(self.device).float()
            sp_lbl  = batch["spoof_label"].to(self.device).float()
            tmp_lbl = batch["temporal_label"].to(self.device).float()
            videos  = batch.get("video_id", [""] * frames.size(0))
            datasets= batch.get("dataset",  [""] * frames.size(0))

            with self.autocast_ctx:
                out = self.model(frames)

                df_logit  = out["deepfake_logit"].flatten()
                sp_logit  = out["spoof_logit"].flatten()
                tmp_logit = out["temp_logit"].flatten()

                l_df   = self.criterion_df(df_logit, df_lbl)
                l_sp   = self.criterion_sp(sp_logit, sp_lbl)
                l_temp = temporal_consistency_loss(out["temp_proj"], tmp_logit)

            running_loss["df"]   += l_df.item()
            running_loss["sp"]   += l_sp.item()
            running_loss["temp"] += l_temp.item()
            n_batches += 1

            # Collect predictions
            df_scores.append(torch.sigmoid(df_logit).cpu().numpy())
            df_labels.append(df_lbl.cpu().numpy())
            df_videos.extend(videos)

            sp_scores.append(torch.sigmoid(sp_logit).cpu().numpy())
            sp_labels.append(sp_lbl.cpu().numpy())

            tmp_scores.append(torch.sigmoid(tmp_logit).cpu().numpy())
            tmp_labels.append(tmp_lbl.cpu().numpy())

            ds_labels.extend(df_lbl.cpu().tolist())
            ds_datasets.extend(datasets)


        # Concatenate
        df_scores  = np.concatenate(df_scores)
        df_labels  = np.concatenate(df_labels)
        sp_scores  = np.concatenate(sp_scores)
        sp_labels  = np.concatenate(sp_labels)
        tmp_scores = np.concatenate(tmp_scores)
        tmp_labels = np.concatenate(tmp_labels)

        # Compute metrics
        df_metrics  = compute_deepfake_metrics(df_labels, df_scores, df_videos)
        sp_metrics  = compute_spoof_metrics(sp_labels, sp_scores)
        tmp_metrics = compute_temporal_metrics(tmp_labels, tmp_scores)
        comp_metrics= compute_deepfake_metrics_by_compression(
            np.array(ds_labels), df_scores, ds_datasets
        )

        val_losses = {k: v / max(n_batches, 1) for k, v in running_loss.items()}

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
        df_auc  = val_metrics.get("df_auc",  float("nan"))
        sp_acer = val_metrics.get("sp_acer", float("nan"))
        tmp_acc = val_metrics.get("tmp_acc", float("nan"))

        self.logger.info(
            f"[{epoch:03d}/{tc.num_epochs}]  "
            f"loss={row['loss_total']:.4f}  "
            f"df_auc={df_auc:.4f}  "
            f"sp_acer={sp_acer:.4f}  "
            f"tmp_acc={tmp_acc:.4f}  "
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

            # ── Scheduler step ────────────────────────────────────────
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(ff_metrics.get("df_auc", 0.0))
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - t0
            power_w    = float(np.mean(self.power_monitor.readings)) if self.power_monitor.readings else 0.0

            # ── Best-model tracking ───────────────────────────────────
            composite = (
                ff_metrics.get("df_auc",  0.0) * 0.5 +
                (1.0 - siw_metrics.get("sp_acer", 1.0)) * 0.5
            )
            is_best = composite > self.best_metric
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
            df_lbl  = batch["deepfake_label"].to(self.device).float()
            sp_lbl  = batch["spoof_label"].to(self.device).float()
            tmp_lbl = batch["temporal_label"].to(self.device).float()
    
            with self.autocast_ctx:
                out    = self.model(frames)
    
                l_df   = self.criterion_df(out["deepfake_logit"], df_lbl)
                l_sp   = self.criterion_sp(out["spoof_logit"],   sp_lbl)
                l_temp = temporal_consistency_loss(out["temp_proj"], tmp_lbl)
    
                # Task weights
                w    = base_model.task_weights                # (3,) normalized
                loss = (w[0] * l_df + w[1] * l_sp + w[2] * l_temp) / tc.grad_accum_steps
    
            # ── Backward ──────────────────────────────
            if self.use_pcgrad:
                pcgrad_step(
                    [w[0] * l_df / tc.grad_accum_steps,
                     w[1] * l_sp / tc.grad_accum_steps,
                     w[2] * l_temp / tc.grad_accum_steps],
                    self.optimizer,
                    retain_graph=self.use_gradnorm,
                )
            else:
                self.scaler.scale(loss).backward(
                    retain_graph=self.use_gradnorm
                )
    
            if self.use_gradnorm:
                losses_tensor = torch.stack([l_df, l_sp, l_temp])  # fix: list→Tensor
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
                if not self.use_pcgrad:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), tc.max_grad_norm  # fix: grad_clip→max_grad_norm
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


def build_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader,
                                         DataLoader, DataLoader]:
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
