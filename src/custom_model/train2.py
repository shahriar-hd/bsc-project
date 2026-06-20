# train.py
# Multi-task learning trainer for Deepfake, Physical Spoof, and Stress detection.
# Architecture: TSM-MobileNetV3-Small with task-specific adapters and GradNorm loss balancing.

import csv
import os
import time
import random
import logging
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from custom_model.train_config import Config

warnings.filterwarnings("ignore")


# ==================================================
# SETUP
# ==================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ==================================================
# AUGMENTATION
# ==================================================

def get_augmentation(dataset_name: str, split: str) -> A.Compose:
    """
    Dataset-aware albumentations pipeline.
    CASME2: minimal augmentation to preserve micro-expression signals.
    FF++: no strong geometric transforms to preserve boundary artifacts.
    SiW-Mv2: moderate augmentation for lighting variation.
    """
    if split != "train":
        return A.Compose([
            A.Resize(Config.FRAME_SIZE, Config.FRAME_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    if dataset_name == "casme2":
        transforms = [
            A.Resize(Config.FRAME_SIZE, Config.FRAME_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
    elif dataset_name == "faceforensics":
        transforms = [
            A.Resize(Config.FRAME_SIZE, Config.FRAME_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        ]
    else:  # siwmv2
        transforms = [
            A.Resize(Config.FRAME_SIZE, Config.FRAME_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.4),
        ]

    transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


# ==================================================
# DATASET
# ==================================================

class MultiTaskDataset(Dataset):
    """
    Loads clips of T frames from master.csv.
    Labels are -1 when the task is not applicable for that dataset.
    """

    def __init__(self, df: pd.DataFrame, split: str):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.sequences = self._build_sequences()
        self.aug_cache = {}

    def _build_sequences(self) -> list:
        groups = self.df.groupby(["dataset", "subject_id", "sequence_id"])
        sequences = []
        for (dataset, subject, seq_id), grp in groups:
            grp_sorted = grp.sort_values("frame_idx").reset_index(drop=True)
            sequences.append({
                "dataset":dataset,
                "subject_id":           subject,
                "sequence_id":          seq_id,
                "frames":               grp_sorted["image_path"].tolist(),
                "frame_indices":        grp_sorted["frame_idx"].tolist(),
                "deepfake_label":       int(grp_sorted["deepfake_label"].iloc[0]),
                "physical_spoof_label": int(grp_sorted["physical_spoof_label"].iloc[0]),
                "stress_label":         int(grp_sorted["stress_label"].iloc[0]),
                "apex_frames":          grp_sorted[grp_sorted["is_apex"] == 1]["frame_idx"].tolist(),
            })
        return sequences

    def _get_aug(self, dataset_name: str) -> A.Compose:
        if dataset_name not in self.aug_cache:
            self.aug_cache[dataset_name] = get_augmentation(dataset_name, self.split)
        return self.aug_cache[dataset_name]

    def _sample_clip_indices(self, seq: dict) -> list:
        T = Config.CLIP_LENGTH
        frames = seq["frames"]
        n = len(frames)

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
            img = np.zeros((Config.FRAME_SIZE, Config.FRAME_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return aug(image=img)["image"]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        aug = self._get_aug(seq["dataset"])
        clip_indices = self._sample_clip_indices(seq)

        frames = [self._load_frame(seq["frames"][i], aug) for i in clip_indices]
        clip = torch.stack(frames, dim=0)

        return {
            "clip":                 clip,
            "dataset":              seq["dataset"],
            "deepfake_label":       torch.tensor(seq["deepfake_label"],       dtype=torch.long),
            "physical_spoof_label": torch.tensor(seq["physical_spoof_label"], dtype=torch.long),
            "stress_label":         torch.tensor(seq["stress_label"],         dtype=torch.long),
        }


# ==================================================
# TASK-BALANCED SAMPLER
# ==================================================

class TaskBalancedSampler(Sampler):
    """
    Yields indices such that each batch contains samples from all datasets
    according to Config.SAMPLE_RATIO. CASME2 is oversampled to compensate
    for its smaller size.
    """

    def __init__(self, dataset: MultiTaskDataset):
        self.dataset = dataset
        self.ratio = Config.SAMPLE_RATIO
        self.indices_by_dataset = defaultdict(list)
        for i, seq in enumerate(dataset.sequences):
            self.indices_by_dataset[seq["dataset"]].append(i)

    def __iter__(self):
        shuffled = {
            ds: random.sample(idxs, len(idxs))
            for ds, idxs in self.indices_by_dataset.items()
        }
        iterators = {ds: iter(idxs) for ds, idxs in shuffled.items()}
        order = []
        for ds, ratio in self.ratio.items():
            if ds in iterators:
                order.extend([ds] * ratio)

        result = []
        exhausted = set()
        while len(exhausted) < len(iterators):
            random.shuffle(order)
            for ds in order:
                if ds in exhausted:
                    continue
                try:
                    result.append(next(iterators[ds]))
                except StopIteration:
                    exhausted.add(ds)
        return iter(result)

    def __len__(self) -> int:
        return sum(
            len(v) * self.ratio.get(k, 1)
            for k, v in self.indices_by_dataset.items()
        )


# ==================================================
# TSM (Temporal Shift Module)
# ==================================================

class TSMWrapper(nn.Module):
    """
    Wraps a conv block to apply Temporal Shift before the convolution.
    Reference: Lin et al., TSM: Temporal Shift Module for Efficient Video Understanding.
    """

    def __init__(self, block: nn.Module, T: int, fold_div: int = 8):
        super().__init__()
        self.block = block
        self.T = T
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        B = BT // self.T
        T = self.T

        x = x.view(B, T, C, H, W)
        fold = C // self.fold_div
        out = x.clone()
        out[:, 1:,  :fold]       = x[:, :-1, :fold]
        out[:, 0,   :fold]       = 0
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]
        out[:, -1,  fold:2*fold] = 0

        return self.block(out.view(BT, C, H, W))


# ==================================================
# MODEL
# ==================================================

class TaskAdapter(nn.Module):
    """Task-specific adapter: Linear → BatchNorm → ReLU → Dropout"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with TSM-MobileNetV3-Small backbone and three task heads.
    Tasks: Deepfake detection, Physical spoof detection, Stress/micro-expression.
    MobileNetV3-Small: ~2.5M params, 576-dim features — ~4x lighter than EfficientNet-B2.
    """

    def __init__(self, T: int = 8):
        super().__init__()
        self.T = T

        # ── Backbone ──────────────────────────────────────────
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # features: Sequential of InvertedResidual blocks + final conv
        # classifier outputs 576-dim after avgpool
        self.features = base.features   # Sequential
        self.avgpool  = base.avgpool    # AdaptiveAvgPool2d → [B, 576, 1, 1]

        # Freeze early blocks (0-2)
        for i in range(3):
            for param in self.features[i].parameters():
                param.requires_grad = False

        # Inject TSM into blocks 3 onward
        for i in range(3, len(self.features)):
            self.features[i] = TSMWrapper(
                self.features[i], T=T, fold_div=Config.TSM_FOLD_DIVISOR
            )

        # ── Task Adapters ──────────────────────────────────────
        dim = Config.BACKBONE_DIM   # must be 576 for MobileNetV3-Small
        self.adapter_df = TaskAdapter(dim, Config.ADAPTER_DIM_DF)
        self.adapter_sp = TaskAdapter(dim, Config.ADAPTER_DIM_SP)
        self.adapter_st = TaskAdapter(dim, Config.ADAPTER_DIM_ST)

        # ── Task Heads ─────────────────────────────────────────
        self.head_deepfake = nn.Linear(Config.ADAPTER_DIM_DF, 2)
        self.head_spoof    = nn.Linear(Config.ADAPTER_DIM_SP, 2)
        self.head_stress   = nn.Linear(Config.ADAPTER_DIM_ST, 2)

    def forward(self, clip: torch.Tensor) -> tuple:
        """
        Args:
            clip: [B, T, C, H, W]
        Returns:
            (logits_df, logits_sp, logits_st) each [B, 2]
        """
        B, T, C, H, W = clip.shape
        x = clip.view(B * T, C, H, W)

        x = self.features(x)   # [B*T, 576, H', W']
        x = self.avgpool(x)    # [B*T, 576, 1, 1]
        x = x.flatten(1)       # [B*T, 576]

        # Temporal aggregation: mean over T frames → [B, 576]
        x = x.view(B, T, -1).mean(dim=1)

        logits_df = self.head_deepfake(self.adapter_df(x))
        logits_sp = self.head_spoof(self.adapter_sp(x))
        logits_st = self.head_stress(self.adapter_st(x))

        return logits_df, logits_sp, logits_st

    def get_parameter_groups(self) -> list:
        backbone_params = []
        for i in range(3, len(self.features)):
            backbone_params += list(self.features[i].parameters())

        adapter_params = (
            list(self.adapter_df.parameters()) +
            list(self.adapter_sp.parameters()) +
            list(self.adapter_st.parameters())
        )
        head_params = (
            list(self.head_deepfake.parameters()) +
            list(self.head_spoof.parameters()) +
            list(self.head_stress.parameters())
        )
        return [
            {"params": backbone_params, "lr": Config.BACKBONE_LEARNING_RATE},
            {"params": adapter_params,  "lr": Config.ADAPTER_LEARNING_RATE},
            {"params": head_params,     "lr": Config.HEAD_LEARNING_RATE},
        ]


# ==================================================
# GRADNORM
# ==================================================

class GradNormLossWeights(nn.Module):
    """
    Learnable per-task loss weights for GradNorm.
    Reference: Chen et al., GradNorm (ICML 2018).
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(num_tasks))

    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_weights)


def compute_gradnorm_loss(
    model: nn.Module,
    task_losses: list,
    loss_weights: GradNormLossWeights,
    initial_losses: torch.Tensor,
    alpha: float = 1.5,
) -> torch.Tensor:
    """
    GradNorm auxiliary loss: L = sum_i |‖G_i‖ - G_bar * r_i^alpha|_1
    where r_i = L_i(t) / L_i(0)
    """
    last_shared = list(model.features[-1].parameters())[-1]

    grad_norms = []
    for loss in task_losses:
        grads = torch.autograd.grad(loss, last_shared, retain_graph=True, create_graph=True)
        grad_norms.append(grads[0].norm())

    grad_norms  = torch.stack(grad_norms)
    mean_norm   = grad_norms.mean().detach()

    current_losses = torch.stack([l.detach() for l in task_losses])
    loss_ratio = current_losses / (initial_losses + 1e-8)
    r = loss_ratio / loss_ratio.mean()

    target_norms  = (mean_norm * r ** alpha).detach()
    gradnorm_loss = (grad_norms - target_norms).abs().sum()
    return gradnorm_loss


# ==================================================
# MASKED LOSS
# ==================================================

def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy ignoring label == -1.
    Returns a zero-valued differentiable tensor when no valid samples exist,
    avoiding division-by-zero in GradNorm's loss ratio.
    """
    mask = labels != -1
    if mask.sum() == 0:
        # Differentiable zero — keeps gradient graph intact for GradNorm
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


# ==================================================
# METRICS
# ==================================================

def compute_metrics(all_labels: list, all_probs: list, task: str) -> dict:
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = probs.argmax(axis=1)
    metrics = {}

    if task in ("deepfake", "spoof"):
        if len(np.unique(labels)) > 1:
            metrics["auc"] = roc_auc_score(labels, probs[:, 1])
        metrics["acc"] = (preds == labels).mean()
        if task == "spoof":
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            apcer = fp / (tn + fp + 1e-8)
            bpcer = fn / (fn + tp + 1e-8)
            metrics["apcer"] = apcer
            metrics["bpcer"] = bpcer
            metrics["acer"]  = (apcer + bpcer) / 2

    elif task == "stress":
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["acc"]      = (preds == labels).mean()
        cm = confusion_matrix(labels, preds)
        per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        metrics["uar"] = per_class_recall.mean()

    return metrics


# ==================================================
# TRAINING / VALIDATION LOOPS
# ==================================================

def run_epoch(
    model:nn.Module,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    loss_weights:   GradNormLossWeights,
    gn_optimizer:   torch.optim.Optimizer,
    scaler:         GradScaler,
    device:         torch.device,
    initial_losses: torch.Tensor,
    is_train:       bool,
    logger:         logging.Logger,
    epoch:          int,
) -> dict:
    """Run one full epoch (train or val). Returns dict of averaged metrics."""
    model.train(is_train)
    phase = "Train" if is_train else "Val"

    total_loss  = 0.0
    task_preds  = {"deepfake": [], "spoof": [], "stress": []}
    task_labels = {"deepfake": [], "spoof": [], "stress": []}
    start = time.time()

    if is_train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        clip   = batch["clip"].to(device)
        lbl_df = batch["deepfake_label"].to(device)
        lbl_sp = batch["physical_spoof_label"].to(device)
        lbl_st = batch["stress_label"].to(device)

        ctx = autocast(enabled=Config.USE_AMP) if is_train else torch.no_grad()
        with ctx:
            logits_df, logits_sp, logits_st = model(clip)

            loss_df = masked_cross_entropy(logits_df, lbl_df)
            loss_sp = masked_cross_entropy(logits_sp, lbl_sp)
            loss_st = masked_cross_entropy(logits_st, lbl_st)

            w = loss_weights.weights
            weighted_loss = (w[0] * loss_df + w[1] * loss_sp + w[2] * loss_st) / Config.GRAD_ACCUM_STEPS

        if is_train:
            scaler.scale(weighted_loss).backward(retain_graph=True)

            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
                # GradNorm update — before scaler.step so gradients are still available
                gn_loss = compute_gradnorm_loss(
                    model,
                    [loss_df, loss_sp, loss_st],
                    loss_weights,
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

                # Re-normalize weights so they sum to num_tasks
                with torch.no_grad():
                    loss_weights.log_weights.data -= loss_weights.log_weights.mean()

        total_loss += weighted_loss.item() * Config.GRAD_ACCUM_STEPS

        for logits, labels, key in [
            (logits_df, lbl_df, "deepfake"),
            (logits_sp, lbl_sp, "spoof"),
            (logits_st, lbl_st, "stress"),
        ]:
            mask = labels != -1
            if mask.sum() > 0:
                probs = F.softmax(logits[mask], dim=1).detach().cpu().numpy()
                task_preds[key].extend(probs.tolist())
                task_labels[key].extend(labels[mask].cpu().numpy().tolist())

        if step % 20 == 0:
            elapsed = time.time() - start
            logger.info(
                f"[{phase}] Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {weighted_loss.item() * Config.GRAD_ACCUM_STEPS:.4f} | "
                f"W: [{w[0].item():.2f}, {w[1].item():.2f}, {w[2].item():.2f}] | "
                f"Elapsed: {elapsed:.1f}s"
            )

    results = {"loss": total_loss / len(loader)}
    for task in ("deepfake", "spoof", "stress"):
        if task_labels[task]:
            m = compute_metrics(task_labels[task], task_preds[task], task)
            for k, v in m.items():
                results[f"{task}_{k}"] = v

    return results


# def run_epoch(
#     model:nn.Module,
#     loader:         DataLoader,
#     optimizer:      torch.optim.Optimizer,
#     loss_weights:   GradNormLossWeights,
#     gn_optimizer:   torch.optim.Optimizer,
#     scaler:         GradScaler,
#     device:         torch.device,
#     initial_losses: torch.Tensor,
#     is_train:       bool,
#     logger:         logging.Logger,
#     epoch:          int,
# ) -> dict:
#     """Run one full epoch (train or val). Returns dict of averaged metrics."""
#     model.train(is_train)
#     phase = "Train" if is_train else "Val"

#     total_loss  = 0.0
#     task_preds  = {"deepfake": [], "spoof": [], "stress": []}
#     task_labels = {"deepfake": [], "spoof": [], "stress": []}
#     start = time.time()

#     if is_train:
#         optimizer.zero_grad()

#     for step, batch in enumerate(loader):
#         clip   = batch["clip"].to(device)
#         lbl_df = batch["deepfake_label"].to(device)
#         lbl_sp = batch["physical_spoof_label"].to(device)
#         lbl_st = batch["stress_label"].to(device)

#         with autocast(enabled=Config.USE_AMP):
#             logits_df, logits_sp, logits_st = model(clip)

#             loss_df = masked_cross_entropy(logits_df, lbl_df)
#             loss_sp = masked_cross_entropy(logits_sp, lbl_sp)
#             loss_st = masked_cross_entropy(logits_st, lbl_st)

#             w = loss_weights.weights
#             # Normalize by grad_accum_steps so effective loss magnitude is unchanged
#             weighted_loss = (w[0] * loss_df + w[1] * loss_sp + w[2] * loss_st) / Config.GRAD_ACCUM_STEPS
#             if is_train:
#                 scaler.scale(weighted_loss).backward(retain_graph=True) 
                
#                 if is_train and (step + 1) % Config.GRAD_ACCUM_STEPS == 0: 
#                     # GradNorm update (uses unscaled losses, so compute before scaler.step)
#                     gn_loss = compute_gradnorm_loss(
#                         model,
#                         [loss_df, loss_sp, loss_st],
#                         loss_weights,
#                         initial_losses,
#                         alpha=Config.GRADNORM_ALPHA,
#                     )
#                     gn_optimizer.zero_grad()
#                     gn_loss.backward()
#                     gn_optimizer.step()

#                     scaler.unscale_(optimizer)
#                     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                     scaler.step(optimizer)
#                     scaler.update()
#                     optimizer.zero_grad()

#                     # Re-normalize weights so they sum to num_tasks
#                     with torch.no_grad():
#                         loss_weights.log_weights.data -= loss_weights.log_weights.mean()

#         total_loss += weighted_loss.item() * Config.GRAD_ACCUM_STEPS  # restore unscaled for logging

#         for logits, labels, key in [
#             (logits_df, lbl_df, "deepfake"),
#             (logits_sp, lbl_sp, "spoof"),
#             (logits_st, lbl_st, "stress"),
#         ]:
#             mask = labels != -1
#             if mask.sum() > 0:
#                 probs = F.softmax(logits[mask], dim=1).detach().cpu().numpy()
#                 task_preds[key].extend(probs.tolist())
#                 task_labels[key].extend(labels[mask].cpu().numpy().tolist())

#         if step % 20 == 0:
#             elapsed = time.time() - start
#             logger.info(
#                 f"[{phase}] Epoch {epoch} | Step {step}/{len(loader)} | "
#                 f"Loss: {weighted_loss.item() * Config.GRAD_ACCUM_STEPS:.4f} | "
#                 f"W: [{w[0].item():.2f}, {w[1].item():.2f}, {w[2].item():.2f}] | "
#                 f"Elapsed: {elapsed:.1f}s"
#             )

#     results = {"loss": total_loss / len(loader)}
#     for task in ("deepfake", "spoof", "stress"):
#         if task_labels[task]:
#             m = compute_metrics(task_labels[task], task_preds[task], task)
#             for k, v in m.items():
#                 results[f"{task}_{k}"] = v

#     return results


# ==================================================
# CHECKPOINT
# ==================================================

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0)


# ==================================================
# MAIN
# ==================================================

def main():
    set_seed(Config.TRAINING_RANDOM_SEED)
    logger = setup_logging(Config.TRAIN_LOG_DIR)

    csv_path = os.path.join(Config.TRAIN_LOG_DIR, "metrics.csv")
    all_metrics: list[dict] = []
    csv_file = open(csv_path, "w", newline="")
    csv_writer = None  # initialized on first epoch

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # Force CPU for debugging; change to above line for GPU training
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        logger.warning(
            "Running on CPU — training will be very slow. "
            "Install CUDA-enabled PyTorch: "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )

    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────────
    logger.info("Loading master CSV...")
    master_df = pd.read_csv(Config.MASTER_CSV_PATH)
    train_df  = master_df[master_df["split"] == "train"].reset_index(drop=True)
    val_df    = master_df[master_df["split"] == "val"].reset_index(drop=True)
    logger.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # ── Datasets & Loaders ────────────────────────────────────
    train_dataset = MultiTaskDataset(train_df, split="train")
    val_dataset   = MultiTaskDataset(val_df,   split="val")
    train_sampler = TaskBalancedSampler(train_dataset)

    _prefetch = 2 if Config.NUM_WORKERS > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=_prefetch,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    model = MultiTaskModel(T=Config.CLIP_LENGTH).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Loss Weights & Optimizers ─────────────────────────────
    loss_weights = GradNormLossWeights(num_tasks=3).to(device)

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

    # ── Estimate Initial Losses for GradNorm ─────────────────
    # Average over N_INIT_BATCHES batches for a stable L_i(0) estimate.
    # Clamp to 1e-4 to prevent division-by-zero when a task has no valid
    # samples in the initial batches (e.g. spoof task absent from first batches).
    logger.info("Estimating initial losses for GradNorm...")
    model.eval()
    N_INIT_BATCHES = 5
    accum = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= N_INIT_BATCHES:
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
        [max(v / N_INIT_BATCHES, 1e-4) for v in accum],
        dtype=torch.float32,
        device=device,
    )
    logger.info(f"Initial losses: df={initial_losses[0]:.4f}, sp={initial_losses[1]:.4f}, st={initial_losses[2]:.4f}")
    model.train()

    # ── Resume from Checkpoint ────────────────────────────────
    start_epoch  = 0
    best_metric  = 0.0
    ckpt_path    = Path(Config.CHECKPOINT_DIR) / "last.pt"
    if ckpt_path.exists():
        logger.info(f"Resuming from {ckpt_path}")
        start_epoch, best_metric = load_checkpoint(
            str(ckpt_path), model, optimizer, device=device
        )
        logger.info(f"Resumed at epoch {start_epoch} | Best metric so far: {best_metric:.4f}")

    # ── Training Loop ─────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch + 1, Config.NUM_EPOCHS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        logger.info(f"LR: backbone={scheduler.get_last_lr()}")

        # Train
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_weights=loss_weights,
            gn_optimizer=gn_optimizer,
            scaler=scaler,
            device=device,
            initial_losses=initial_losses,
            is_train=True,
            logger=logger,
            epoch=epoch,
        )

        # Validate
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                loss_weights=loss_weights,
                gn_optimizer=gn_optimizer,
                scaler=scaler,
                device=device,
                initial_losses=initial_losses,
                is_train=False,
                logger=logger,
                epoch=epoch,
            )

        scheduler.step()

        # ── Log Metrics ───────────────────────────────────────

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        all_metrics.append(row)

        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        logger.info(f"[Train] Loss: {train_metrics['loss']:.4f}")
        logger.info(f"[Val]   Loss: {val_metrics['loss']:.4f}")
        for task in ("deepfake", "spoof", "stress"):
            task_keys = [k for k in val_metrics if k.startswith(task)]
            if task_keys:
                metric_str = " | ".join(
                    f"{k.replace(task+'_', '').upper()}: {val_metrics[k]:.4f}"
                    for k in task_keys
                )
                logger.info(f"[Val]   {task.capitalize()}: {metric_str}")

        primary_candidates = [val_metrics[k] for k in ("deepfake_auc", "spoof_auc", "stress_f1_macro") if k in val_metrics]
        primary_metric = np.mean(primary_candidates) if primary_candidates else 0.0
        logger.info(f"[Val]   Primary metric (mean AUC/F1): {primary_metric:.4f}")

        # ── Save Last Checkpoint ──────────────────────────────
        save_checkpoint(
            {
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "loss_weights": loss_weights.state_dict(),
                "best_metric":  best_metric,
                "val_metrics":  val_metrics,
                "train_metrics": train_metrics,
            },
            str(ckpt_path),
        )

        # ── Save Best Checkpoint ──────────────────────────────
        if primary_metric > best_metric:
            best_metric      = primary_metric
            patience_counter = 0
            best_path        = Path(Config.CHECKPOINT_DIR) / "best.pt"
            save_checkpoint(
                {
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "loss_weights": loss_weights.state_dict(),
                    "best_metric":  best_metric,
                    "val_metrics":  val_metrics,
                },
                str(best_path),
            )
            logger.info(f"New best model saved → {best_path} (metric: {best_metric:.4f})")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # ── Early Stopping ────────────────────────────────────
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {Config.EARLY_STOPPING_PATIENCE} epochs)."
            )
            break

        # ── VRAM Cleanup ──────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_file.close()

    # Plot
    epochs = [r["epoch"] for r in all_metrics]
    plot_keys = [k for k in all_metrics[0] if k != "epoch"]
    n = len(plot_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = axes.flatten()
    for i, key in enumerate(plot_keys):
        axes[i].plot(epochs, [r[key] for r in all_metrics], marker="o", linewidth=1.5)
        axes[i].set_title(key, fontsize=9)
        axes[i].set_xlabel("Epoch")
        axes[i].grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(Config.TRAIN_LOG_DIR, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Metrics saved → {csv_path}")
    logger.info(f"Plot saved    → {plot_path}")

    logger.info(f"\nTraining complete. Best metric: {best_metric:.4f}")
    logger.info(f"Best checkpoint: {Path(Config.CHECKPOINT_DIR) / 'best.pt'}")


if __name__ == "__main__":
    main()
