"""
train3.py
Multi-task learning trainer — EfficientNet-B2 / MobileNetV3 backbone,
TSM temporal modeling, GradNorm loss balancing.
Tasks: Deepfake (FF++), Physical Spoof (SiW-Mv2), Stress (CASME2).

PEP8 compliant. Modular. OOM-safe. Full evaluation metrics.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import csv
import os
import gc
import time
import random
import logging
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as grad_checkpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
)
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    average_precision_score, precision_recall_curve,
    matthews_corrcoef, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.custom_model.train_config import Config

warnings.filterwarnings("ignore")


# ==============================================================================
# SETUP
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "train3.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ==============================================================================
# AUGMENTATION
# ==============================================================================

def get_augmentation(dataset_name: str, split: str) -> A.Compose:
    """
    Dataset-aware albumentations pipeline.

    CASME2 : minimal augmentation — preserve micro-expression signals.
    FF++   : no strong geometric transforms — preserve boundary artifacts.
    SiW    : moderate augmentation — cover lighting variation.
    """
    common_end = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    if split != "train":
        return A.Compose([
            A.Resize(Config.FRAME_SIZE, Config.FRAME_SIZE),
            *common_end,
        ])

    resize = A.Resize(Config.FRAME_SIZE, Config.FRAME_SIZE)

    if dataset_name == "casme2":
        aug = [
            resize,
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
        ]
    elif dataset_name == "faceforensics":
        aug = [
            resize,
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.ColorJitter(
                brightness=0.1, contrast=0.1,
                saturation=0.1, hue=0.05, p=0.3
            ),
        ]
    else:  # siwmv2
        aug = [
            resize,
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.15, hue=0.05, p=0.4
            ),
        ]

    return A.Compose(aug + common_end)


# ==============================================================================
# DATASET
# ==============================================================================

class MultiTaskDataset(Dataset):
    """
    Loads T-frame clips from master.csv.
    Labels are -1 when the task is not applicable for a given dataset row.
    """

    def __init__(self, df: pd.DataFrame, split: str) -> None:
        self.df = df.reset_index(drop=True)
        self.split = split
        self.sequences = self._build_sequences()
        self._aug_cache: dict[str, A.Compose] = {}

    def _build_sequences(self) -> list[dict]:
        groups = self.df.groupby(["dataset", "subject_id", "sequence_id"])
        sequences = []
        for (dataset, subject, seq_id), grp in groups:
            grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
            sequences.append({
                "dataset":              dataset,
                "subject_id":           subject,
                "sequence_id":          seq_id,
                "frames":               grp_sorted["image_path"].tolist(),
                "frame_indices":        grp_sorted["frame_idx"].tolist(),
                "deepfake_label":       int(grp_sorted["deepfake_label"].iloc[0]),
                "physical_spoof_label": int(grp_sorted["physical_spoof_label"].iloc[0]),
                "stress_label":         int(grp_sorted["stress_label"].iloc[0]),
                "apex_frames":          grp_sorted[
                    grp_sorted["is_apex"] == 1
                ]["frame_idx"].tolist(),
            })
        return sequences

    def _get_aug(self, dataset_name: str) -> A.Compose:
        if dataset_name not in self._aug_cache:
            self._aug_cache[dataset_name] = get_augmentation(
                dataset_name, self.split
            )
        return self._aug_cache[dataset_name]

    def _sample_clip_indices(self, seq: dict) -> list[int]:
        T = Config.CLIP_LENGTH
        n = len(seq["frames"])

        if n <= T:
            return list(range(n)) + [n - 1] * (T - n)

        if seq["dataset"] == "casme2" and seq["apex_frames"]:
            apex_fidx = seq["apex_frames"][0]
            try:
                center = seq["frame_indices"].index(apex_fidx)
            except ValueError:
                center = n // 2
            start = max(0, min(center - T // 2, n - T))
        elif self.split == "train":
            start = random.randint(0, n - T)
        else:
            start = (n - T) // 2

        return list(range(start, start + T))

    def _load_frame(self, path: str, aug: A.Compose) -> torch.Tensor:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros(
                (Config.FRAME_SIZE, Config.FRAME_SIZE, 3), dtype=np.uint8
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return aug(image=img)["image"]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        aug = self._get_aug(seq["dataset"])
        clip_indices = self._sample_clip_indices(seq)

        frames = [self._load_frame(seq["frames"][i], aug) for i in clip_indices]
        clip = torch.stack(frames, dim=0)  # [T, C, H, W]

        return {
            "clip":                 clip,
            "dataset":              seq["dataset"],
            "deepfake_label":       torch.tensor(
                seq["deepfake_label"],       dtype=torch.long
            ),
            "physical_spoof_label": torch.tensor(
                seq["physical_spoof_label"], dtype=torch.long
            ),
            "stress_label":         torch.tensor(
                seq["stress_label"],         dtype=torch.long
            ),
        }


# ==============================================================================
# TASK-BALANCED SAMPLER
# ==============================================================================
class TaskBalancedSampler(Sampler):
    """
    Interleaves dataset indices according to Config.SAMPLE_RATIO.
    CASME2 indices are cycled (repeated) to compensate for its smaller size.
    """

    def __init__(self, dataset: MultiTaskDataset) -> None:
        super().__init__()  # fixed: no argument
        self.dataset = dataset
        self.ratio = Config.SAMPLE_RATIO
        self._indices_by_ds: dict[str, list[int]] = defaultdict(list)
        for i, seq in enumerate(dataset.sequences):
            self._indices_by_ds[seq["dataset"]].append(i)

    def __iter__(self):
        # Shuffle each dataset's indices independently
        shuffled: dict[str, list[int]] = {
            ds: random.sample(idxs, len(idxs))
            for ds, idxs in self._indices_by_ds.items()
        }

        # Find the dominant dataset (largest after ratio scaling)
        max_scaled = max(
            len(idxs) // self.ratio.get(ds, 1)
            for ds, idxs in shuffled.items()
        )

        # Repeat smaller datasets to match dominant length (cycling)
        repeated: dict[str, list[int]] = {}
        for ds, idxs in shuffled.items():
            r = self.ratio.get(ds, 1)
            target = max_scaled * r
            # tile then trim to exact length
            tiled = (idxs * (target // len(idxs) + 1))[:target]
            repeated[ds] = tiled

        # Build interleaved slot order and fill result
        slot_order: list[str] = []
        for ds, r in self.ratio.items():
            if ds in repeated:
                slot_order.extend([ds] * r)

        # Interleave: one full cycle of slot_order = one "round"
        result: list[int] = []
        iters = {ds: iter(idxs) for ds, idxs in repeated.items()}

        rounds = max_scaled
        for _ in range(rounds):
            for ds in slot_order:
                if ds not in iters:
                    continue
                try:
                    result.append(next(iters[ds]))
                except StopIteration:
                    pass  # shouldn't happen after cycling

        return iter(result)

    def __len__(self) -> int:
        if not self._indices_by_ds:
            return 0
        max_scaled = max(
            len(idxs) // self.ratio.get(ds, 1)
            for ds, idxs in self._indices_by_ds.items()
        )
        return max_scaled * sum(
            self.ratio.get(ds, 1)
            for ds in self._indices_by_ds
        )



# ==============================================================================
# TSM — TEMPORAL SHIFT MODULE
# ==============================================================================

class TSMWrapper(nn.Module):
    """
    Applies Temporal Shift before the wrapped conv block.
    Lin et al., TSM (ICCV 2019).
    """

    def __init__(self, block: nn.Module, T: int, fold_div: int = 8) -> None:
        super().__init__()
        self.block = block
        self.T = T
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        B = BT // self.T
        fold = C // self.fold_div

        x = x.view(B, self.T, C, H, W)
        out = x.clone()
        # shift backward (past → present)
        out[:, 1:,   :fold]         = x[:, :-1, :fold]
        out[:, 0,    :fold]         = 0
        # shift forward (future → present)
        out[:, :-1, fold:2 * fold]  = x[:, 1:,  fold:2 * fold]
        out[:, -1,  fold:2 * fold]  = 0

        return self.block(out.view(BT, C, H, W))


# ==============================================================================
# BACKBONE FACTORY
# ==============================================================================

def _build_mobilenet(T: int, freeze_stages: int, use_tsm: bool):
    """Returns (features, avgpool, out_dim)."""
    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    features = base.features
    avgpool  = base.avgpool

    # Freeze first `freeze_stages` blocks
    for i in range(min(freeze_stages, len(features))):
        for param in features[i].parameters():
            param.requires_grad = False

    if use_tsm:
        for i in range(freeze_stages, len(features)):
            features[i] = TSMWrapper(
                features[i], T=T, fold_div=Config.TSM_FOLD_DIVISOR
            )

    return features, avgpool, 576


def _inject_tsm_efficientnet(features: nn.Sequential, T: int, freeze_stages: int) -> None:
    """
    Injects TSMWrapper into EfficientNet MBConv blocks beyond freeze_stages.
    EfficientNet features[0] = stem conv, features[1..N] = MBConv stages.
    """
    for i in range(freeze_stages + 1, len(features)):
        features[i] = TSMWrapper(
            features[i], T=T, fold_div=Config.TSM_FOLD_DIVISOR
        )


def _build_efficientnet(
    variant: str, T: int, freeze_stages: int, use_tsm: bool
):
    """Returns (features, avgpool, out_dim)."""
    if variant == "efficientnet_b0":
        base    = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        out_dim = 1280
    else:  # efficientnet_b2
        base    = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        out_dim = 1408

    features = base.features
    avgpool  = base.avgpool

    # Freeze stem + first `freeze_stages` MBConv stages
    for i in range(min(freeze_stages + 1, len(features))):
        for param in features[i].parameters():
            param.requires_grad = False

    if use_tsm:
        _inject_tsm_efficientnet(features, T, freeze_stages)

    return features, avgpool, out_dim


def build_backbone(
    backbone_type: str, T: int, freeze_stages: int, use_tsm: bool
) -> tuple[nn.Sequential, nn.Module, int]:
    """
    Factory: returns (features, avgpool, out_dim) for the configured backbone.
    """
    if backbone_type == "mobilenet":
        return _build_mobilenet(T, freeze_stages, use_tsm)
    return _build_efficientnet(backbone_type, T, freeze_stages, use_tsm)


# ==============================================================================
# TASK ADAPTER
# ==============================================================================

class TaskAdapter(nn.Module):
    """Projection → BatchNorm → ReLU → Dropout"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# MULTI-TASK MODEL
# ==============================================================================

class MultiTaskModel(nn.Module):
    """
    Shared backbone (MobileNetV3-Small | EfficientNet-B0/B2) with optional TSM,
    three task-specific adapters, and three classification heads.

    Tasks:
        0 — Deepfake detection   (FF++)
        1 — Physical spoof       (SiW-Mv2)
        2 — Stress/micro-expr    (CASME2)
    """

    def __init__(self) -> None:
        super().__init__()
        T             = Config.CLIP_LENGTH
        backbone_type = Config.BACKBONE_TYPE
        freeze_stages = Config.FREEZE_BACKBONE_STAGES
        use_tsm       = Config.USE_TSM

        self.T = T
        self.use_grad_checkpoint = Config.USE_GRAD_CHECKPOINT

        self.features, self.avgpool, feat_dim = build_backbone(
            backbone_type, T, freeze_stages, use_tsm
        )

        dropout = Config.ADAPTER_DROPOUT
        self.adapter_df = TaskAdapter(feat_dim, Config.ADAPTER_DIM_DF, dropout)
        self.adapter_sp = TaskAdapter(feat_dim, Config.ADAPTER_DIM_SP, dropout)
        self.adapter_st = TaskAdapter(feat_dim, Config.ADAPTER_DIM_ST, dropout)

        self.head_deepfake = nn.Linear(Config.ADAPTER_DIM_DF, 2)
        self.head_spoof    = nn.Linear(Config.ADAPTER_DIM_SP, 2)
        self.head_stress   = nn.Linear(Config.ADAPTER_DIM_ST, 2)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grad_checkpoint and self.training:
            # Gradient checkpointing — trades compute for VRAM
            return grad_checkpoint(self.features, x, use_reentrant=False)
        return self.features(x)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            clip: [B, T, C, H, W]
        Returns:
            (logits_df, logits_sp, logits_st) — each [B, 2]
        """
        B, T, C, H, W = clip.shape
        x = clip.view(B * T, C, H, W)

        x = self._forward_backbone(x)    # [B*T, D, h, w]
        x = self.avgpool(x)              # [B*T, D, 1, 1]
        x = x.flatten(1)                 # [B*T, D]
        x = x.view(B, T, -1).mean(dim=1) # [B, D]

        logits_df = self.head_deepfake(self.adapter_df(x))
        logits_sp = self.head_spoof(self.adapter_sp(x))
        logits_st = self.head_stress(self.adapter_st(x))

        return logits_df, logits_sp, logits_st

    def get_parameter_groups(self) -> list[dict]:
        """Differential learning rates per layer group."""
        backbone_trainable = [
            p for p in self.features.parameters() if p.requires_grad
        ]
        adapter_params = (
            list(self.adapter_df.parameters())
            + list(self.adapter_sp.parameters())
            + list(self.adapter_st.parameters())
        )
        head_params = (
            list(self.head_deepfake.parameters())
            + list(self.head_spoof.parameters())
            + list(self.head_stress.parameters())
        )
        return [
            {"params": backbone_trainable, "lr": Config.BACKBONE_LR},
            {"params": adapter_params,     "lr": Config.ADAPTER_LR},
            {"params": head_params,        "lr": Config.HEAD_LR},
        ]


# ==============================================================================
# GRADNORM
# ==============================================================================

class GradNormLossWeights(nn.Module):
    """Learnable per-task loss weights. Chen et al., GradNorm (ICML 2018)."""

    def __init__(self, num_tasks: int = 3) -> None:
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(num_tasks))

    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_weights)


def compute_gradnorm_loss(
    model: nn.Module,
    task_losses: list[torch.Tensor],
    initial_losses: torch.Tensor,
    alpha: float = 1.5,
) -> torch.Tensor:
    """
    GradNorm auxiliary loss:
      L = Σ_i | ‖G_i‖ − Ḡ · r_i^α |₁
    where r_i = L_i(t) / L_i(0).
    """
    # Last shared parameter — last layer of backbone features
    last_shared: Optional[torch.Tensor] = None
    for p in reversed(list(model.features.parameters())):
        if p.requires_grad:
            last_shared = p
            break

    if last_shared is None:
        return torch.tensor(0.0, requires_grad=True)

    grad_norms = []
    for loss in task_losses:
        grads = torch.autograd.grad(
            loss, last_shared, retain_graph=True, create_graph=True
        )
        grad_norms.append(grads[0].norm())

    grad_norms_t = torch.stack(grad_norms)
    mean_norm    = grad_norms_t.mean().detach()

    current_losses = torch.stack([l.detach() for l in task_losses])
    loss_ratio     = current_losses / (initial_losses + 1e-8)
    r              = loss_ratio / (loss_ratio.mean() + 1e-8)

    target_norms  = (mean_norm * r ** alpha).detach()
    gradnorm_loss = (grad_norms_t - target_norms).abs().sum()
    return gradnorm_loss


# ==============================================================================
# MASKED CROSS-ENTROPY
# ==============================================================================

def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy ignoring label == -1."""
    mask = labels != -1
    if mask.sum() == 0:
        return logits.sum() * 0.0  # differentiable zero
    return F.cross_entropy(logits[mask], labels[mask])


# ==============================================================================
# METRICS
# ==============================================================================

def _eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Equal Error Rate from ROC curve arrays."""
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _tpr_at_fpr(
    fpr: np.ndarray, tpr: np.ndarray, target_fpr: float
) -> float:
    """TPR at a given FPR threshold."""
    idx = np.searchsorted(fpr, target_fpr)
    idx = min(idx, len(tpr) - 1)
    return float(tpr[idx])


def compute_metrics_deepfake(
    labels: np.ndarray,
    probs: np.ndarray,
    video_labels: Optional[list] = None,
    video_probs: Optional[list]  = None,
) -> dict:
    """
    FaceForensics++ metrics:
    AUC-ROC, Accuracy, EER, AP, AUC-PR, Video-level AUC.
    """
    preds   = probs.argmax(axis=1)
    scores  = probs[:, 1]
    metrics: dict = {}

    metrics["acc"] = float((preds == labels).mean())

    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        metrics["eer"] = _eer_from_roc(fpr, tpr)
        metrics["ap"]  = float(average_precision_score(labels, scores))

        prec, rec, _ = precision_recall_curve(labels, scores)
        # AUC-PR via trapezoidal rule (same as AP but explicit)
        metrics["auc_pr"] = float(np.trapezoid(prec[::-1], rec[::-1]))

        if video_labels is not None and video_probs is not None:
            v_labels = np.array(video_labels)
            v_scores = np.array(video_probs)
            if len(np.unique(v_labels)) > 1:
                metrics["video_auc"] = float(Train
                    roc_auc_score(v_labels, v_scores)
                )

    return metrics


def compute_metrics_spoof(
    labels: np.ndarray, probs: np.ndarray
) -> dict:
    """
    SiW-Mv2 metrics:
    ACER, APCER, BPCER, AUC-ROC, TPR@FPR(1%, 0.5%), EER, HTER.
    """
    preds  = probs.argmax(axis=1)
    scores = probs[:, 1]
    metrics: dict = {}

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        apcer = float(fp / (tn + fp + 1e-8))   # Attack Presentation Classification Error Rate
        bpcer = float(fn / (fn + tp + 1e-8))   # Bonafide Presentation Classification Error Rate
        metrics["apcer"] = apcer
        metrics["bpcer"] = bpcer
        metrics["acer"]  = (apcer + bpcer) / 2.0
        metrics["hter"]  = (apcer + bpcer) / 2.0   # Half Total Error Rate = ACER for binary

    metrics["acc"] = float((preds == labels).mean())

    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        metrics["eer"]          = _eer_from_roc(fpr, tpr)
        metrics["tpr_at_fpr1"]  = _tpr_at_fpr(fpr, tpr, 0.01)
        metrics["tpr_at_fpr05"] = _tpr_at_fpr(fpr, tpr, 0.005)

    return metrics


def compute_metrics_stress(
    labels: np.ndarray, probs: np.ndarray
) -> dict:
    """
    CASME2 metrics:
    UAR, UF1, Accuracy, F1-macro, MCC, Confusion Matrix summary,
    AUPRC, FAR/FRR, ROC-AUC (macro OvR).
    """
    preds  = probs.argmax(axis=1)
    scores = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
    metrics: dict = {}

    metrics["acc"]      = float((preds == labels).mean())
    metrics["f1_macro"] = float(
        f1_score(labels, preds, average="macro", zero_division=0)
    )
    metrics["mcc"] = float(matthews_corrcoef(labels, preds))

    cm = confusion_matrix(labels, preds)
    per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    metrics["uar"] = float(per_class_recall.mean())   # Unweighted Average Recall

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    metrics["uf1"] = float(per_class_f1.mean())       # Unweighted Average F1

    n_classes = len(np.unique(labels))
    if n_classes == 2 and len(np.unique(labels)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(labels, scores))
        metrics["ap"]      = float(average_precision_score(labels, scores))

        prec, rec, _ = precision_recall_curve(labels, scores)
        metrics["auprc"] = float(np.trapezoid(prec[::-1], rec[::-1]))

        fpr, tpr, _ = roc_curve(labels, scores)
        # FAR = FPR at operating threshold (argmax of Youden's J)
        j_scores = tpr - fpr
        opt_idx  = np.argmax(j_scores)
        metrics["far"] = float(fpr[opt_idx])
        metrics["frr"] = float(1.0 - tpr[opt_idx])

    return metrics


def compute_metrics(
    all_labels: list, all_probs: list, task: str
) -> dict:
    """Dispatch to per-task metric function."""
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    if task == "deepfake":
        return compute_metrics_deepfake(labels, probs)
    if task == "spoof":
        return compute_metrics_spoof(labels, probs)
    return compute_metrics_stress(labels, probs)


# ==============================================================================
# OOM-SAFE FORWARD PASS
# ==============================================================================

def safe_forward(
    model: nn.Module,
    clip: torch.Tensor,
    device: torch.device,
    logger: logging.Logger,
) -> Optional[tuple[torch.Tensor, ...]]:
    """
    Attempts forward pass. On CUDA OOM, clears cache and returns None
    so the training loop can skip the batch gracefully.
    """
    try:
        return model(clip)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — skipping batch, clearing cache.")
        torch.cuda.empty_cache()
        gc.collect()
        return None


# ==============================================================================
# EPOCH RUNNER
# ==============================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_weights: GradNormLossWeights,
    gn_optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    initial_losses: torch.Tensor,
    is_train: bool,
    logger: logging.Logger,
    epoch: int,
) -> dict:
    """
    Runs one full epoch (train or val).
    Returns a dict of averaged losses and per-task metrics.
    """
    model.train(is_train)
    phase = "Train" if is_train else "Val"

    total_loss   = 0.0
    n_batches    = 0
    task_preds   = {"deepfake": [], "spoof": [], "stress": []}
    task_labels  = {"deepfake": [], "spoof": [], "stress": []}
    t_start      = time.time()

    if is_train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        clip   = batch["clip"].to(device, non_blocking=True)
        lbl_df = batch["deepfake_label"].to(device, non_blocking=True)
        lbl_sp = batch["physical_spoof_label"].to(device, non_blocking=True)
        lbl_st = batch["stress_label"].to(device, non_blocking=True)

        # ── Forward ──────────────────────────────────────────────────────────
        ctx = autocast(enabled=Config.USE_AMP) if is_train else torch.no_grad()
        with ctx:
            out = safe_forward(model, clip, device, logger)
            if out is None:
                optimizer.zero_grad()
                continue

            logits_df, logits_sp, logits_st = out

            loss_df = masked_cross_entropy(logits_df, lbl_df)
            loss_sp = masked_cross_entropy(logits_sp, lbl_sp)
            loss_st = masked_cross_entropy(logits_st, lbl_st)

            w = loss_weights.weights
            weighted_loss = (
                w[0] * loss_df + w[1] * loss_sp + w[2] * loss_st
            ) / Config.GRAD_ACCUM_STEPS

        # ── Backward ─────────────────────────────────────────────────────────
        if is_train:
            scaler.scale(weighted_loss).backward(retain_graph=True)

            do_step = (
                (step + 1) % Config.GRAD_ACCUM_STEPS == 0
                or (step + 1) == len(loader)
            )
            if do_step:
                # GradNorm weight update
                gn_loss = compute_gradnorm_loss(
                    model,
                    [loss_df, loss_sp, loss_st],
                    initial_losses,
                    alpha=Config.GRADNORM_ALPHA,
                )
                gn_optimizer.zero_grad()
                gn_loss.backward()
                gn_optimizer.step()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Re-normalize: keep weights summing to num_tasks
                with torch.no_grad():
                    loss_weights.log_weights.data -= (
                        loss_weights.log_weights.mean()
                    )

        total_loss += weighted_loss.item() * Config.GRAD_ACCUM_STEPS
        n_batches  += 1

        # ── Collect predictions ───────────────────────────────────────────────
        for logits, labels, key in (
            (logits_df, lbl_df, "deepfake"),
            (logits_sp, lbl_sp, "spoof"),
            (logits_st, lbl_st, "stress"),
        ):
            mask = labels != -1
            if mask.sum() > 0:
                probs = F.softmax(logits[mask], dim=1).detach().cpu().numpy()
                task_preds[key].extend(probs.tolist())
                task_labels[key].extend(labels[mask].cpu().numpy().tolist())

        # ── Step log ─────────────────────────────────────────────────────────
        if step % 20 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"[{phase}] Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {weighted_loss.item() * Config.GRAD_ACCUM_STEPS:.4f} | "
                f"W: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    results: dict = {"loss": total_loss / max(n_batches, 1)}
    task_name_map = {"deepfake": "deepfake", "spoof": "spoof", "stress": "stress"}

    for key, task in task_name_map.items():
        if task_labels[key]:
            m = compute_metrics(task_labels[key], task_preds[key], task)
            for mk, mv in m.items():
                results[f"{key}_{mk}"] = mv

    return results


# ==============================================================================
# CHECKPOINT HELPERS
# ==============================================================================

def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0)


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_metrics(all_metrics: list[dict], log_dir: str) -> str:
    epochs    = [r["epoch"] for r in all_metrics]
    plot_keys = [k for k in all_metrics[0] if k != "epoch"]
    n    = len(plot_keys)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.array(axes).flatten()

    for i, key in enumerate(plot_keys):
        axes[i].plot(
            epochs, [r.get(key, None) for r in all_metrics],
            marker="o", linewidth=1.5
        )
        axes[i].set_title(key, fontsize=9)
        axes[i].set_xlabel("Epoch")
        axes[i].grid(True, alpha=0.3)

    for j in range(len(plot_keys), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(log_dir, "metrics_plot.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    set_seed(Config.SEED)
    logger = setup_logging(Config.TRAIN_LOG_DIR)

    # ── Device ────────────────────────────────────────────────────────────────
    if Config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        logger.warning("Running on CPU — training will be slow.")
    logger.info(f"Device: {device}")
    logger.info(f"Backbone: {Config.BACKBONE_TYPE} | TSM: {Config.USE_TSM} | "
                f"GradCheckpoint: {Config.USE_GRAD_CHECKPOINT}")

    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading master CSV...")
    master_df = pd.read_csv(Config.MASTER_CSV_PATH)
    train_df  = master_df[master_df["split"] == "train"].reset_index(drop=True)
    val_df    = master_df[master_df["split"] == "val"].reset_index(drop=True)
    logger.info(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    train_dataset = MultiTaskDataset(train_df, split="train")
    val_dataset   = MultiTaskDataset(val_df,   split="val")
    train_sampler = TaskBalancedSampler(train_dataset)

    _prefetch = 2 if Config.NUM_WORKERS > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        prefetch_factor=_prefetch,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=_prefetch,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model       = MultiTaskModel().to(device)
    loss_weights = GradNormLossWeights(num_tasks=3).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_p:,} | Trainable: {trainable_p:,}")

    # ── Optimizers & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.get_parameter_groups(),
        weight_decay=Config.WEIGHT_DECAY,
    )
    gn_optimizer = torch.optim.Adam(
        loss_weights.parameters(),
        lr=Config.GRADNORM_LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=1e-6,
    )
    scaler = GradScaler(enabled=Config.USE_AMP)

    # ── Estimate Initial Losses ───────────────────────────────────────────────
    logger.info("Estimating initial losses for GradNorm...")
    model.eval()
    accum = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= Config.N_INIT_BATCHES:
                break
            clip_s   = batch["clip"].to(device)
            lbl_df_s = batch["deepfake_label"].to(device)
            lbl_sp_s = batch["physical_spoof_label"].to(device)
            lbl_st_s = batch["stress_label"].to(device)
            with autocast(enabled=Config.USE_AMP):
                lg_df, lg_sp, lg_st = model(clip_s)
            accum[0] += masked_cross_entropy(lg_df, lbl_df_s).item()
            accum[1] += masked_cross_entropy(lg_sp, lbl_sp_s).item()
            accum[2] += masked_cross_entropy(lg_st, lbl_st_s).item()

    initial_losses = torch.tensor(
        [max(v / Config.N_INIT_BATCHES, 1e-4) for v in accum],
        dtype=torch.float32, device=device,
    )
    logger.info(
        f"Initial losses — df: {initial_losses[0]:.4f} | "
        f"sp: {initial_losses[1]:.4f} | st: {initial_losses[2]:.4f}"
    )
    model.train()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = 0.0
    ckpt_last   = Path(Config.CHECKPOINT_DIR) / "last.pt"
    if ckpt_last.exists():
        logger.info(f"Resuming from {ckpt_last}")
        start_epoch, best_metric = load_checkpoint(
            str(ckpt_last), model, optimizer, device=device
        )
        logger.info(f"Resumed at epoch {start_epoch} | Best: {best_metric:.4f}")

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    csv_path     = os.path.join(Config.TRAIN_LOG_DIR, "metrics.csv")
    all_metrics: list[dict] = []
    csv_file     = open(csv_path, "w", newline="")
    csv_writer   = None

    # ── Training Loop ─────────────────────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch + 1, Config.NUM_EPOCHS + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{Config.NUM_EPOCHS}")

        train_metrics = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            loss_weights=loss_weights, gn_optimizer=gn_optimizer,
            scaler=scaler, device=device, initial_losses=initial_losses,
            is_train=True, logger=logger, epoch=epoch,
        )

        val_metrics = run_epoch(
            model=model, loader=val_loader, optimizer=optimizer,
            loss_weights=loss_weights, gn_optimizer=gn_optimizer,
            scaler=scaler, device=device, initial_losses=initial_losses,
            is_train=False, logger=logger, epoch=epoch,
        )

        scheduler.step()

        # ── Log ───────────────────────────────────────────────────────────────
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        all_metrics.append(row)

        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        logger.info(
            f"[Train] Loss: {train_metrics['loss']:.4f} | "
            f"[Val] Loss: {val_metrics['loss']:.4f}"
        )
        for task in ("deepfake", "spoof", "stress"):
            task_keys = [k for k in val_metrics if k.startswith(task + "_")]
            if task_keys:
                parts = " | ".join(
                    f"{k[len(task)+1:].upper()}: {val_metrics[k]:.4f}"
                    for k in task_keys
                )
                logger.info(f"[Val] {task.capitalize()}: {parts}")

        # Primary metric: mean of available AUCs / F1
        primary_candidates = [
            val_metrics[k]
            for k in ("deepfake_auc_roc", "spoof_auc_roc", "stress_f1_macro")
            if k in val_metrics
        ]
        primary_metric = float(np.mean(primary_candidates)) if primary_candidates else 0.0
        logger.info(f"[Val] Primary (mean AUC/F1): {primary_metric:.4f}")

        # ── Checkpoints ───────────────────────────────────────────────────────
        ckpt_state = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "loss_weights":  loss_weights.state_dict(),
            "best_metric":   best_metric,
            "val_metrics":   val_metrics,
            "train_metrics": train_metrics,
            "config": {
                "backbone":   Config.BACKBONE_TYPE,
                "clip_length": Config.CLIP_LENGTH,
                "frame_size":  Config.FRAME_SIZE,
            },
        }
        save_checkpoint(ckpt_state, str(ckpt_last))

        if primary_metric > best_metric:
            best_metric      = primary_metric
            patience_counter = 0
            best_path        = Path(Config.CHECKPOINT_DIR) / "best.pt"
            save_checkpoint(ckpt_state, str(best_path))
            logger.info(
                f"New best model → {best_path} (metric: {best_metric:.4f})"
            )
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: "
                f"{patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early stopping after {epoch} epochs."
            )
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_file.close()

    # ── Plot & Summary ────────────────────────────────────────────────────────
    plot_path = plot_metrics(all_metrics, Config.TRAIN_LOG_DIR)
    logger.info(f"Metrics CSV  → {csv_path}")
    logger.info(f"Metrics plot → {plot_path}")
    logger.info(f"Training complete. Best metric: {best_metric:.4f}")
    logger.info(f"Best checkpoint: {Path(Config.CHECKPOINT_DIR) / 'best.pt'}")


if __name__ == "__main__":
    main()
