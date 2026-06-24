"""
MTL training script for three heads: deepfake, anti-spoof, temporal consistency.
Supports GradNorm, PCGrad, TSM, AMP, OOM fallback, power monitoring, and full metrics.
"""

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

from src.config import Config, get_config

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    """Configure pretty console + file logger."""
    logger = logging.getLogger("MTL")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def log_banner(logger: logging.Logger, text: str) -> None:
    """Print a section banner to logger."""
    sep = "─" * 60
    logger.info(sep)
    logger.info(f"  {text}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# Power Monitor
# ──────────────────────────────────────────────────────────────────────────────

class PowerMonitor:
    """Background thread that polls GPU (nvidia-smi) and CPU (intel-rapl) power."""

    def __init__(self, cfg: Config):
        """Initialize monitor with config."""
        self.cfg = cfg
        self.readings: List[float] = []        # total watts per sample
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        """Start background polling."""
        self._thread.start()

    def stop(self):
        """Stop polling and return mean watt consumption."""
        self._stop.set()
        self._thread.join()
        return float(np.mean(self.readings)) if self.readings else 0.0

    def _read_gpu_watts(self) -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=3
            )
            return sum(float(x) for x in out.decode().strip().split("\n"))
        except Exception:
            return 0.0

    def _read_cpu_watts(self) -> float:
        rapl = Path(self.cfg.power.rapl_path)
        total = 0.0
        try:
            for pkg in rapl.glob("intel-rapl:*"):
                energy_file = pkg / "energy_uj"
                if energy_file.exists():
                    e1 = int(energy_file.read_text())
                    time.sleep(0.1)
                    e2 = int(energy_file.read_text())
                    total += max(0, (e2 - e1)) / 1e6 / 0.1  # W
        except Exception:
            pass
        return total

    def _read_ram_watts(self) -> float:
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            return used_gb * self.cfg.power.ram_coeff
        except Exception:
            return 0.0

    def _poll(self):
        pc = self.cfg.power
        while not self._stop.is_set():
            gpu = self._read_gpu_watts()
            cpu = self._read_cpu_watts()
            ram = self._read_ram_watts()
            other = pc.ssd_coeff + pc.other_coeff
            total = gpu + cpu + ram + other
            self.readings.append(total)
            time.sleep(pc.poll_interval_sec)


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
            aug = self.transform(image=img)["image"]   # C,H,W tensor
            frames.append(aug)

        # Stack: (T, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)

        return {
            "frames": frames_tensor,                   # (T,C,H,W)
            "label": torch.tensor(clip["label"], dtype=torch.long),
            "task": clip["task"],                      # "deepfake" | "anti-spoof"
            "dataset": clip["dataset"],
            "video_path": clip["video_path"],
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
    Temporal consistency head.
    Projects per-frame features then measures inter-frame consistency.
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.clf = nn.Linear(hidden, 1)  # for binary supervision

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feats: (B, T, D)
        proj = self.proj(feats)                        # (B, T, H)
        # Mean-pooled representation for classification
        pooled = proj.mean(dim=1)                      # (B, H)
        logit = self.clf(pooled).squeeze(-1)           # (B,)
        return proj, logit


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
        self.temporal_head = TemporalHead(feat_dim, mc.temporal_hidden, mc.dropout)

        # GradNorm learnable log-weights
        self.log_weights = nn.Parameter(torch.zeros(3))  # [df, spoof, temp]

    @property
    def task_weights(self) -> torch.Tensor:
        """Softmax-normalized task weights from GradNorm parameters."""
        return F.softmax(self.log_weights, dim=0) * 3   # scale to sum~3

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
        temp_proj, temp_logit = self.temporal_head(feats)

        return {
            "df_logit": df_logit,
            "sp_logit": sp_logit,
            "temp_logit": temp_logit,
            "temp_proj": temp_proj,    # (B, T, H) for temporal loss
            "feats": feats,
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
# GradNorm
# ──────────────────────────────────────────────────────────────────────────────

class GradNormManager:
    """
    GradNorm: adjusts task loss weights so all tasks train at similar speed.
    Reference: Chen et al., 2018.
    """

    def __init__(self, model: MTLModel, alpha: float, lr: float):
        """Initialize with model, alpha (asymmetry), and weight LR."""
        self.model = model
        self.alpha = alpha
        self.initial_losses: Optional[torch.Tensor] = None
        self.optimizer = torch.optim.Adam([model.log_weights], lr=lr)

    def update(
        self,
        losses: torch.Tensor,          # (3,) current task losses
        shared_params: List[torch.Tensor],
    ) -> None:
        """One GradNorm update step."""
        if self.initial_losses is None:
            self.initial_losses = losses.detach()

        weights = self.model.task_weights                  # (3,)
        weighted_losses = weights * losses

        # Gradient norms
        G_norms = []
        for wl in weighted_losses:
            grads = torch.autograd.grad(
                wl, shared_params, retain_graph=True, allow_unused=True
            )
            g_flat = torch.cat([
                g.view(-1) for g in grads if g is not None
            ])
            G_norms.append(g_flat.norm())

        G_norms = torch.stack(G_norms)                    # (3,)
        G_mean = G_norms.mean().detach()

        # Relative inverse training rates
        loss_ratio = losses.detach() / (self.initial_losses + 1e-8)
        loss_ratio_mean = loss_ratio.mean()
        ri = loss_ratio / (loss_ratio_mean + 1e-8)

        # GradNorm targets
        G_targets = (G_mean * ri ** self.alpha).detach()

        # GradNorm loss
        gn_loss = (G_norms - G_targets).abs().sum()

        self.optimizer.zero_grad()
        gn_loss.backward(retain_graph=True)
        self.optimizer.step()

        # Renormalize weights
        with torch.no_grad():
            self.model.log_weights.data = (
                self.model.log_weights - self.model.log_weights.mean()
            )


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Main training loop for MTL model."""

    def __init__(self, cfg: Config, logger: logging.Logger):
        """Initialize all training components."""
        self.cfg = cfg
        self.logger = logger
        # self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.current_batch_size = cfg.train.batch_size

        # Paths
        ckpt_root = cfg.paths.checkpoint_dir
        os.makedirs(ckpt_root, exist_ok=True)

        if cfg.resume:
            last = get_last_run_id(ckpt_root)
            if last and os.path.exists(os.path.join(ckpt_root, last, "last.pth")):
                self.run_id = last
                logger.info(f"Resuming from {last}")
            else:
                self.run_id = get_next_run_id(ckpt_root)
                logger.info(f"Starting new run: {self.run_id}")
        else:
            self.run_id = get_next_run_id(ckpt_root)

        self.run_dir = os.path.join(ckpt_root, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Model
        self.model = MTLModel(cfg).to(self.device)
        self._log_model_info()

        # Losses
        self.criterion_df = nn.BCEWithLogitsLoss()
        self.criterion_spoof = FocalLoss(cfg.train.focal_gamma, cfg.train.focal_alpha)

        # Optimizer (two param groups: heads + backbone)
        backbone_params = list(self.model.backbone.parameters())
        head_params = (
            list(self.model.deepfake_head.parameters()) +
            list(self.model.spoof_head.parameters()) +
            list(self.model.temporal_head.parameters())
        )
        self.optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": cfg.train.lr},
            {"params": backbone_params, "lr": cfg.train.lr * cfg.model.backbone_lr_scale},
        ], weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)

        # Scheduler
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=cfg.train.use_amp and self.device.type == "cuda")

        # GradNorm
        self.gradnorm = None
        if cfg.train.use_gradnorm:
            self.gradnorm = GradNormManager(
                self.model, cfg.train.gradnorm_alpha, lr=1e-3
            )

        # Result CSV
        result_path = os.path.join(self.run_dir, cfg.paths.result_csv)
        self.result_logger = ResultLogger(result_path)

        # State
        self.start_epoch = 0
        self.best_metric = -float("inf")
        self.best_epoch = 0

        # Load checkpoint if resuming
        if cfg.resume:
            self._load_checkpoint()

    def _log_model_info(self) -> None:
        """Log model parameter count."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(
            f"Model: {self.cfg.model.backbone} | "
            f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M"
        )

    def _build_scheduler(self):
        """Build LR scheduler based on config."""
        tc = self.cfg.train
        if tc.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tc.num_epochs, eta_min=tc.min_lr
            )
        elif tc.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=tc.step_size, gamma=tc.gamma
            )
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )

    def _freeze_backbone(self, freeze: bool) -> None:
        """Freeze or unfreeze backbone parameters."""
        for p in self.model.backbone.parameters():
            p.requires_grad = not freeze
        state = "frozen" if freeze else "unfrozen"
        self.logger.info(f"Backbone {state}.")

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        """Save epoch checkpoint and optionally best model."""
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
        }
        epoch_path = os.path.join(self.run_dir, f"epoch_{epoch:03d}.pth")
        torch.save(state, epoch_path)

        last_path = os.path.join(self.run_dir, "last.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.run_dir, "best.pth")
            torch.save(state, best_path)
            self.logger.info(f"  [*] Best model saved at epoch {epoch}")

    def _load_checkpoint(self) -> None:
        """Load from last.pth if it exists."""
        ckpt_path = os.path.join(self.run_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric = ckpt.get("best_metric", -float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.logger.info(
            f"Resumed from epoch {ckpt['epoch']} | best metric: {self.best_metric:.4f}"
        )

    def _compute_task_losses(
        self,
        outputs: dict,
        labels: torch.Tensor,
        task_flags: torch.Tensor,   # 0=deepfake, 1=spoof
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-task losses with task masking."""
        df_mask = (task_flags == 0)
        sp_mask = (task_flags == 1)

        loss_df = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_sp = torch.tensor(0.0, device=self.device, requires_grad=True)

        if df_mask.sum() > 0:
            loss_df = self.criterion_df(
                outputs["df_logit"][df_mask],
                labels[df_mask].float()
            )
        if sp_mask.sum() > 0:
            loss_sp = self.criterion_spoof(
                outputs["sp_logit"][sp_mask],
                labels[sp_mask]
            )

        loss_temp = temporal_consistency_loss(
            outputs["temp_proj"], labels
        )
        return loss_df, loss_sp, loss_temp

    def _train_step(
        self,
        batch: dict,
        accum_step: int,
    ) -> dict:
        """One gradient accumulation step. Returns loss dict."""
        tc = self.cfg.train
        frames = batch["frames"].to(self.device)
        labels = batch["label"].to(self.device)
        tasks = batch["task"]
        task_flags = torch.tensor(
            [0 if t == "deepfake" else 1 for t in tasks],
            device=self.device
        )

        with autocast(enabled=tc.use_amp and self.device.type == "cuda"):
            outputs = self.model(frames)
            loss_df, loss_sp, loss_temp = self._compute_task_losses(
                outputs, labels, task_flags
            )
            losses = torch.stack([loss_df, loss_sp, loss_temp])

            if tc.use_gradnorm:
                weights = self.model.task_weights
            else:
                weights = torch.tensor(
                    [tc.w_deepfake, tc.w_spoof, tc.w_temporal],
                    device=self.device
                )
            total_loss = (weights * losses).sum() / tc.grad_accum_steps

        self.scaler.scale(total_loss).backward()

        return {
            "loss_total": total_loss.item() * tc.grad_accum_steps,
            "loss_df": loss_df.item(),
            "loss_sp": loss_sp.item(),
            "loss_temp": loss_temp.item(),
            "w_df": weights[0].item(),
            "w_sp": weights[1].item(),
            "w_temp": weights[2].item(),
        }

    def _optimizer_step(self, losses_tensor: Optional[torch.Tensor] = None) -> None:
        tc = self.cfg.train

        # GradNorm update
        if tc.use_gradnorm and self.gradnorm is not None and losses_tensor is not None:
            shared_params = [
                p for p in self.model.backbone.parameters() if p.requires_grad
            ]
            if shared_params:
                self.gradnorm.update(losses_tensor, shared_params)

        # Unscale once — required before both PCGrad and grad clipping
        self.scaler.unscale_(self.optimizer)

        # PCGrad (gradients are already unscaled at this point)
        # Note: full PCGrad per-layer projection would go here if implemented

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), tc.max_grad_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()


    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        """Run one training epoch with OOM handling."""
        self.model.train()
        tc = self.cfg.train
        accum_losses = []
        batch_metrics: Dict[str, List] = {
            k: [] for k in ["loss_total", "loss_df", "loss_sp", "loss_temp",
                             "w_df", "w_sp", "w_temp"]
        }

        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            try:
                step_out = self._train_step(batch, step % tc.grad_accum_steps)
                for k, v in step_out.items():
                    batch_metrics[k].append(v)

                if (step + 1) % tc.grad_accum_steps == 0:
                    self._optimizer_step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._handle_oom(e)
                    continue
                raise e

        # Final step if leftover
        if (step + 1) % tc.grad_accum_steps != 0:
            self._optimizer_step()

        return {k: float(np.mean(v)) for k, v in batch_metrics.items() if v}

    def _handle_oom(self, error: Exception) -> None:
        """Handle CUDA OOM by clearing cache or falling back to CPU."""
        self.logger.warning(f"OOM: {error}")
        torch.cuda.empty_cache()
        gc.collect()

        if self.cfg.train.reduce_batch_on_oom:
            new_bs = max(
                self.cfg.train.min_batch_size,
                self.current_batch_size // 2
            )
            if new_bs < self.current_batch_size:
                self.logger.warning(
                    f"Reducing batch size: {self.current_batch_size} -> {new_bs}"
                )
                self.current_batch_size = new_bs

        if self.cfg.train.oom_fallback_cpu and self.device.type == "cuda":
            self.logger.warning("Falling back to CPU for this step.")
            self.model = self.model.cpu()
            self.device = torch.device("cpu")

    @torch.no_grad()
    def evaluate(
        self,
        ff_loader: DataLoader,
        siw_loader: DataLoader,
        epoch: int,
    ) -> dict:
        """Evaluate all three heads and return full metrics dict."""
        self.model.eval()
        metrics = {}

        # ── Deepfake head (FF++) ──────────────────────────────────────────
        df_labels, df_scores, df_videos, df_datasets = [], [], [], []
        for batch in ff_loader:
            frames = batch["frames"].to(self.device)
            labels = batch["label"].numpy()
            out = self.model(frames)
            scores = torch.sigmoid(out["df_logit"]).cpu().numpy()
            df_labels.append(labels)
            df_scores.append(scores)
            df_videos.extend(batch["video_path"])
            df_datasets.extend(batch["dataset"])

        df_labels = np.concatenate(df_labels)
        df_scores = np.concatenate(df_scores)
        df_metrics = compute_deepfake_metrics(df_labels, df_scores, df_videos)
        df_metrics.update(
            compute_deepfake_metrics_by_compression(df_labels, df_scores, df_datasets)
        )
        metrics.update({f"df_{k}": v for k, v in df_metrics.items()})

        # ── Anti-spoof head (SiW-Mv2) ─────────────────────────────────────
        sp_labels, sp_scores = [], []
        for batch in siw_loader:
            frames = batch["frames"].to(self.device)
            labels = batch["label"].numpy()
            out = self.model(frames)
            scores = torch.sigmoid(out["sp_logit"]).cpu().numpy()
            sp_labels.append(labels)
            sp_scores.append(scores)

        sp_labels = np.concatenate(sp_labels)
        sp_scores = np.concatenate(sp_scores)
        sp_metrics = compute_spoof_metrics(sp_labels, sp_scores)
        metrics.update({f"sp_{k}": v for k, v in sp_metrics.items()})

        # ── Temporal head (both datasets) ─────────────────────────────────
        temp_labels, temp_scores = [], []
        for loader in [ff_loader, siw_loader]:
            for batch in loader:
                frames = batch["frames"].to(self.device)
                labels = batch["label"].numpy()
                out = self.model(frames)
                scores = torch.sigmoid(out["temp_logit"]).cpu().numpy()
                temp_labels.append(labels)
                temp_scores.append(scores)

        temp_labels = np.concatenate(temp_labels)
        temp_scores = np.concatenate(temp_scores)
        temp_metrics = compute_temporal_metrics(temp_labels, temp_scores)
        metrics.update({f"temp_{k}": v for k, v in temp_metrics.items()})

        return metrics

    def _log_epoch(self, epoch: int, train_m: dict, val_m: dict, lr: float,
                   elapsed: float, power_w: float) -> None:
        """Print a clean per-epoch summary to logger."""
        log_banner(self.logger, f"Epoch {epoch:03d} Summary")

        self.logger.info(
            f"  LR: {lr:.2e} | Time: {elapsed:.1f}s | Avg Power: {power_w:.1f}W"
        )
        self.logger.info(
            f"  Train | total={train_m['loss_total']:.4f} "
            f"df={train_m['loss_df']:.4f} "
            f"sp={train_m['loss_sp']:.4f} "
            f"temp={train_m['loss_temp']:.4f}"
        )
        self.logger.info(
            f"  Weights | df={train_m['w_df']:.3f} "
            f"sp={train_m['w_sp']:.3f} "
            f"temp={train_m['w_temp']:.3f}"
        )
        # Deepfake metrics
        self.logger.info(
            f"  Deepfake | AUC={val_m.get('df_auc_roc', 0):.4f} "
            f"EER={val_m.get('df_eer', 0):.4f} "
            f"AP={val_m.get('df_ap', 0):.4f} "
            f"Acc@T={val_m.get('df_acc_best_thresh', 0):.4f} "
            f"VideoAUC={val_m.get('df_video_auc', 0):.4f} "
            f"C23={val_m.get('df_auc_c23', 0):.4f} "
            f"C40={val_m.get('df_auc_c40', 0):.4f}"
        )
        # Anti-spoof metrics
        self.logger.info(
            f"  AntiSpoof | HTER={val_m.get('sp_hter', 0):.4f} "
            f"ACER={val_m.get('sp_acer', 0):.4f} "
            f"APCER={val_m.get('sp_apcer', 0):.4f} "
            f"BPCER={val_m.get('sp_bpcer', 0):.4f} "
            f"TPR@FPR1%={val_m.get('sp_tpr_at_fpr1', 0):.4f} "
            f"AUC={val_m.get('sp_auc', 0):.4f}"
        )
        # Temporal metrics
        self.logger.info(
            f"  Temporal | BinAcc={val_m.get('temp_bin_acc', 0):.4f} "
            f"AUC={val_m.get('temp_auc', 0):.4f} "
            f"DF-AUC={val_m.get('temp_df_auc', 0):.4f}"
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_ff_loader: DataLoader,
        val_siw_loader: DataLoader,
    ) -> None:
        """Main training loop."""
        cfg = self.cfg
        tc = cfg.train
        log_banner(self.logger, f"Training: {self.run_id} | Device: {self.device}")

        power_monitor = None
        if cfg.power.enable:
            power_monitor = PowerMonitor(cfg)
            power_monitor.start()

        for epoch in range(self.start_epoch, tc.num_epochs):
            self.logger.info(f"Runing Epoch {epoch + 1}")
            t0 = time.time()

            # Phase 1: freeze backbone
            if epoch < cfg.model.freeze_backbone_epochs:
                self._freeze_backbone(True)
            elif epoch == cfg.model.freeze_backbone_epochs:
                self._freeze_backbone(False)

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluation
            val_metrics = {}
            if (epoch + 1) % cfg.eval.eval_every == 0:
                val_metrics = self.evaluate(val_ff_loader, val_siw_loader, epoch)

            # LR step
            if tc.scheduler == "plateau":
                self.scheduler.step(val_metrics.get("df_auc_roc", 0))
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Power reading
            power_w = 0.0
            if power_monitor and power_monitor.readings:
                power_w = float(np.mean(power_monitor.readings[-5:]))

            # Log
            self._log_epoch(epoch + 1, train_metrics, val_metrics,
                            current_lr, elapsed, power_w)

            # Best model tracking (primary: deepfake AUC + spoof AUC avg)
            primary = (
                val_metrics.get("df_auc_roc", 0) +
                val_metrics.get("sp_auc", 0)
            ) / 2
            is_best = primary > self.best_metric
            if is_best:
                self.best_metric = primary
                self.best_epoch = epoch + 1

            # Save checkpoint
            combined_metrics = {**train_metrics, **val_metrics,
                                 "epoch": epoch + 1, "lr": current_lr,
                                 "power_w": power_w, "elapsed_s": elapsed}
            self._save_checkpoint(epoch + 1, combined_metrics, is_best)

            # CSV log
            self.result_logger.log({
                "epoch": epoch + 1,
                "run_id": self.run_id,
                "lr": current_lr,
                "elapsed_s": round(elapsed, 2),
                "power_w": round(power_w, 2),
                **{k: round(v, 6) if isinstance(v, float) else v
                   for k, v in combined_metrics.items()},
            })

        # Final power average
        total_power = 0.0
        if power_monitor:
            total_power = power_monitor.stop()
            self.logger.info(f"Average power consumption: {total_power:.1f} W")

        log_banner(
            self.logger,
            f"Training done | Best epoch: {self.best_epoch} | "
            f"Best metric: {self.best_metric:.4f}"
        )


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

    logger = setup_logger(log_path)
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
