# train.py
# Multi-task learning trainer for Deepfake, Physical Spoof, and Stress detection.
# Architecture: TSM-EfficientNet-B2 with task-specific adapters and GradNorm loss balancing.

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
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from app.config import Config

warnings.filterwarnings("ignore")


# ==================================================
# SETUP
# ==================================================

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure file + console logging."""
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
    Return dataset-aware albumentations pipeline.
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
    Each sample is a dict with a clip tensor and three task labels.
    Labels are -1 when the task is not applicable for that dataset.
    """

    def __init__(self, df: pd.DataFrame, split: str):
        self.df = df.reset_index(drop=True)
        self.split = split
        # Group frames by sequence for clip building
        self.sequences = self._build_sequences()
        # Per-dataset augmentation (keyed by dataset name)
        self.aug_cache = {}

    def _build_sequences(self) -> list:
        """Group rows into sequences; each sequence becomes one or more clips."""
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
                "apex_frames":          grp_sorted[grp_sorted["is_apex"] == 1]["frame_idx"].tolist(),
            })
        return sequences

    def _get_aug(self, dataset_name: str) -> A.Compose:
        """Cache augmentation pipelines per dataset."""
        if dataset_name not in self.aug_cache:
            self.aug_cache[dataset_name] = get_augmentation(dataset_name, self.split)
        return self.aug_cache[dataset_name]

    def _sample_clip_indices(self, seq: dict) -> list:
        """
        Sample T frame indices from a sequence.
        For CASME2: center clip on apex frame if available.
        For others: random window during training, center window during eval.
        """
        T = Config.CLIP_LENGTH
        frames = seq["frames"]
        n = len(frames)

        if n <= T:
            # Repeat last frame to fill clip
            indices = list(range(n)) + [n - 1] * (T - n)
            return indices

        if seq["dataset"] == "casme2" and seq["apex_frames"]:
            # Find position of apex in sorted frame list
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
        """Load a single frame, apply augmentation, return CHW tensor."""
        img = cv2.imread(path)
        if img is None:
            # Return black frame on read failure
            img = np.zeros((Config.FRAME_SIZE, Config.FRAME_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = aug(image=img)
        return result["image"]  # CHW float tensor

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        aug = self._get_aug(seq["dataset"])
        clip_indices = self._sample_clip_indices(seq)

        frames = []
        for i in clip_indices:
            frame_path = seq["frames"][i]
            frames.append(self._load_frame(frame_path, aug))

        # Stack to [T, C, H, W]
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
    according to Config.'sample_ratio']. CASME2 is oversampled to compensate
    for its smaller size.
    """

    def __init__(self, dataset: MultiTaskDataset):
        self.dataset = dataset
        self.ratio = Config.SAMPLE_RATIO
        # Build per-dataset index lists
        self.indices_by_dataset = defaultdict(list)
        for i, seq in enumerate(dataset.sequences):
            self.indices_by_dataset[seq["dataset"]].append(i)

    def __iter__(self):
        # Shuffle each dataset's indices
        shuffled = {
            ds: random.sample(idxs, len(idxs))
            for ds, idxs in self.indices_by_dataset.items()
        }
        # Build interleaved sequence based on ratio
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
        total = sum(
            len(v) * self.ratio.get(k, 1)
            for k, v in self.indices_by_dataset.items()
        )
        return total


# ==================================================
# TSM (Temporal Shift Module)
# ==================================================

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM).
    Shifts 1/fold_div channels forward and 1/fold_div channels backward
    along the temporal dimension. Zero parameter overhead.
    Reference: Lin et al., TSM: Temporal Shift Module for Efficient Video Understanding.
    """

    def __init__(self, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]  — reshaped inside forward pass
        # We need T to be passed; handled by TSMWrapper
        return x  # actual shift done in TSMWrapper


class TSMWrapper(nn.Module):
    """
    Wraps a conv block to apply TSM before the convolution.
    Expects input shape [B, T, C, H, W] reshaped to [B*T, C, H, W].
    """

    def __init__(self, block: nn.Module, T: int, fold_div: int = 8):
        super().__init__()
        self.block = block
        self.T = T
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal shift then forward through wrapped block."""
        BT, C, H, W = x.shape
        B = BT // self.T
        T = self.T

        # Reshape to [B, T, C, H, W]
        x = x.view(B, T, C, H, W)

        fold = C // self.fold_div
        out = x.clone()
        # Shift forward (past → present)
        out[:, 1:,    :fold]      = x[:, :-1, :fold]
        out[:, 0,     :fold]      = 0
        # Shift backward (future → present)
        out[:, :-1,   fold:2*fold] = x[:, 1:, fold:2*fold]
        out[:, -1,    fold:2*fold] = 0

        # Reshape back to [B*T, C, H, W]
        out = out.view(BT, C, H, W)
        return self.block(out)


# ==================================================
# MODEL
# ==================================================

class TaskAdapter(nn.Module):
    """
    Task-specific adapter: projects shared features to task feature space.
    Linear → BatchNorm → ReLU → Dropout
    """

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
    Multi-task model with TSM-EfficientNet-B2 backbone and three task heads.
    Tasks: Deepfake detection, Physical spoof detection, Stress/micro-expression.
    """

    def __init__(self, T: int = 8):
        super().__init__()
        self.T = T

        # ── Backbone ──────────────────────────────────────────
        base = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

        # Extract feature blocks (drop classifier)
        self.features = base.features  # Sequential of MBConv blocks
        self.avgpool  = base.avgpool   # AdaptiveAvgPool2d → [B, 1408, 1, 1]

        # Freeze early blocks (0-2)
        for i in range(3):
            for param in self.features[i].parameters():
                param.requires_grad = False

        # Inject TSM into blocks 3-7
        for i in range(3, 8):
            self.features[i] = TSMWrapper(self.features[i], T=T, fold_div=Config.TSM_FOLD_DIVISOR)

        # ── Task Adapters ──────────────────────────────────────
        dim = Config.BACKBONE_DIM
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

        # Merge batch and time dims → [B*T, C, H, W]
        x = clip.view(B * T, C, H, W)

        # Backbone forward (TSM applied inside TSMWrapper blocks)
        x = self.features(x)          # [B*T, 1408, 7, 7]
        x = self.avgpool(x)           # [B*T, 1408, 1, 1]
        x = x.flatten(1)              # [B*T, 1408]

        # Temporal aggregation: mean over T frames → [B, 1408]
        x = x.view(B, T, -1).mean(dim=1)

        # Task-specific paths
        logits_df = self.head_deepfake(self.adapter_df(x))
        logits_sp = self.head_spoof(self.adapter_sp(x))
        logits_st = self.head_stress(self.adapter_st(x))

        return logits_df, logits_sp, logits_st

    def get_parameter_groups(self) -> list:
        """Return parameter groups with differential learning rates."""
        # Frozen blocks (0-2) are excluded automatically (requires_grad=False)
        backbone_params = []
        for i in range(3, 8):
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
    Reference: Chen et al., GradNorm: Gradient Normalization for Adaptive Loss
               Balancing in Deep Multitask Networks (ICML 2018).
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Initialize weights to 1.0 (log-space for positivity)
        self.log_weights = nn.Parameter(torch.zeros(num_tasks))

    @property
    def weights(self) -> torch.Tensor:
        """Return positive task weights."""
        return torch.exp(self.log_weights)


def compute_gradnorm_loss(
    model: nn.Module,
    task_losses: list,
    loss_weights: GradNormLossWeights,
    initial_losses: torch.Tensor,
    alpha: float = 1.5,
) -> torch.Tensor:
    """
    Compute GradNorm auxiliary loss to balance task gradients.

    L_gradnorm = sum_i | ||G_i|| - G_bar * r_i^alpha |_1
    where r_i = L_i(t) / L_i(0)  (relative inverse training rate)
    """
    # Get gradient norms for each task w.r.t. last shared layer
    last_shared = list(model.features[-1].parameters())[-1]

    grad_norms = []
    for loss in task_losses:
        grads = torch.autograd.grad(loss, last_shared, retain_graph=True, create_graph=True)
        grad_norms.append(grads[0].norm())

    grad_norms = torch.stack(grad_norms)
    mean_norm   = grad_norms.mean().detach()

    # Relative inverse training rate
    current_losses = torch.stack([l.detach() for l in task_losses])
    loss_ratio = current_losses / (initial_losses + 1e-8)
    r = loss_ratio / loss_ratio.mean()

    target_norms = (mean_norm * r ** alpha).detach()
    gradnorm_loss = (grad_norms - target_norms).abs().sum()
    return gradnorm_loss


# ==================================================
# MASKED LOSS
# ==================================================

def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss ignoring samples where label == -1.
    Normalizes by the number of valid samples to avoid undertraining
    when a task has few samples in the batch.
    """
    mask = labels != -1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(logits[mask], labels[mask])


# ==================================================
# METRICS
# ==================================================

def compute_metrics(all_labels: list, all_probs: list, task: str) -> dict:
    """
    Compute task-specific evaluation metrics.
    Returns a dict of metric_name → value.
    """
    labels = np.array(all_labels)
    probs  = np.array(all_probs)   # shape [N, 2]
    preds  = probs.argmax(axis=1)

    metrics = {}

    if task in ("deepfake", "spoof"):
        # AUC-ROC (positive class probability)
        if len(np.unique(labels)) > 1:
            metrics["auc"] = roc_auc_score(labels, probs[:, 1])
        metrics["acc"] = (preds == labels).mean()

        if task == "spoof":
            # APCER / BPCER / ACER  (ISO/IEC 30107-3)
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            apcer = fp / (tn + fp + 1e-8)   # attack misclassified as bona fide
            bpcer = fn / (fn + tp + 1e-8)   # bona fide misclassified as attack
            metrics["apcer"] = apcer
            metrics["bpcer"] = bpcer
            metrics["acer"]  = (apcer + bpcer) / 2

    elif task == "stress":
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["acc"]      = (preds == labels).mean()
        # UAR (Unweighted Average Recall)
        cm = confusion_matrix(labels, preds)
        per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        metrics["uar"] = per_class_recall.mean()

    return metrics


# ==================================================
# TRAINING / VALIDATION LOOPS
# ==================================================

def run_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    loss_weights: GradNormLossWeights,
    gn_optimizer: torch.optim.Optimizer,
    scaler:       GradScaler,
    device:       torch.device,
    initial_losses: torch.Tensor,
    is_train:     bool,
    logger:       logging.Logger,
    epoch:        int,
) -> dict:
    """Run one full epoch (train or val). Returns dict of averaged metrics."""
    model.train(is_train)
    phase = "Train" if is_train else "Val"

    total_loss = 0.0
    task_preds = {"deepfake": [], "spoof": [], "stress": []}
    task_labels = {"deepfake": [], "spoof": [], "stress": []}

    start = time.time()

    for step, batch in enumerate(loader):
        clip   = batch["clip"].to(device)                          # [B, T, C, H, W]
        lbl_df = batch["deepfake_label"].to(device)
        lbl_sp = batch["physical_spoof_label"].to(device)
        lbl_st = batch["stress_label"].to(device)

        with autocast(enabled=Config.USE_AMP):
            logits_df, logits_sp, logits_st = model(clip)

            loss_df = masked_cross_entropy(logits_df, lbl_df)
            loss_sp = masked_cross_entropy(logits_sp, lbl_sp)
            loss_st = masked_cross_entropy(logits_st, lbl_st)

            w = loss_weights.weights
            weighted_loss = w[0] * loss_df + w[1] * loss_sp + w[2] * loss_st

        if is_train:
            optimizer.zero_grad()
            gn_optimizer.zero_grad()

            scaler.scale(weighted_loss).backward(retain_graph=True)

            # GradNorm update
            gn_loss = compute_gradnorm_loss(
                model,
                [loss_df, loss_sp, loss_st],
                loss_weights,
                initial_losses,
                alpha=Config.GRADNORM_ALPHA,
            )
            gn_loss.backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            gn_optimizer.step()

            # Re-normalize weights so they sum to num_tasks
            with torch.no_grad():
                loss_weights.log_weights.data = (
                    loss_weights.log_weights - loss_weights.log_weights.mean()
                )

        total_loss += weighted_loss.item()

        # Collect predictions for metrics (only valid labels)
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
                f"Loss: {weighted_loss.item():.4f} | "
                f"W: [{w[0].item():.2f}, {w[1].item():.2f}, {w[2].item():.2f}] | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # Aggregate metrics
    results = {"loss": total_loss / len(loader)}
    for task in ("deepfake", "spoof", "stress"):
        if task_labels[task]:
            m = compute_metrics(task_labels[task], task_preds[task], task)
            for k, v in m.items():
                results[f"{task}_{k}"] = v

    return results


# ==================================================
# CHECKPOINT
# ==================================================

def save_checkpoint(state: dict, path: str):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device=None):
    """Load checkpoint and return epoch + best metric."""
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
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
    # Run one forward pass on a small batch to get L_i(0)
    logger.info("Estimating initial losses for GradNorm...")
    model.train()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        clip_s   = sample_batch["clip"].to(device)
        lbl_df_s = sample_batch["deepfake_label"].to(device)
        lbl_sp_s = sample_batch["physical_spoof_label"].to(device)
        lbl_st_s = sample_batch["stress_label"].to(device)
        with autocast(enabled=Config.USE_AMP):
            lg_df, lg_sp, lg_st = model(clip_s)
            l0_df = masked_cross_entropy(lg_df, lbl_df_s).detach()
            l0_sp = masked_cross_entropy(lg_sp, lbl_sp_s).detach()
            l0_st = masked_cross_entropy(lg_st, lbl_st_s).detach()
    initial_losses = torch.stack([l0_df, l0_sp, l0_st])
    # Clamp to avoid division by zero if a task has no valid samples in first batch
    initial_losses = initial_losses.clamp(min=1e-4)
    logger.info(
        f"Initial losses — Deepfake: {l0_df.item():.4f} | "
        f"Spoof: {l0_sp.item():.4f} | Stress: {l0_st.item():.4f}"
    )

    # ── Resume from Checkpoint (optional) ────────────────────
    start_epoch  = 0
    best_metric  = 0.0
    ckpt_path    = Path(Config.CHECKPOINT_DIR) / "last.pt"
    best_path    = Path(Config.CHECKPOINT_DIR) / "best.pt"

    if ckpt_path.exists():
        logger.info(f"Resuming from {ckpt_path}")
        start_epoch, best_metric = load_checkpoint(
            str(ckpt_path), model, optimizer, device
        )
        logger.info(f"Resumed at epoch {start_epoch} | Best metric so far: {best_metric:.4f}")

    # ── Training Loop ─────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch + 1, Config.NUM_EPOCHS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{Config.NUM_EPOCHS}")
        logger.info(f"{'='*60}")

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

        # ── Log Epoch Summary ──────────────────────────────────
        logger.info(f"\n--- Epoch {epoch} Summary ---")
        logger.info(f"[Train] Loss: {train_metrics['loss']:.4f}")
        logger.info(f"[Val]   Loss: {val_metrics['loss']:.4f}")

        for task, short in [("deepfake", "DF"), ("spoof", "SP"), ("stress", "ST")]:
            parts = []
            for k, v in val_metrics.items():
                if k.startswith(task):
                    metric_name = k.replace(f"{task}_", "").upper()
                    parts.append(f"{metric_name}: {v:.4f}")
            if parts:
                logger.info(f"[Val]   {short} | " + " | ".join(parts))

        # ── Composite Metric for Model Selection ──────────────
        # Average of available primary metrics: AUC (DF), ACER-inv (SP), F1 (ST)
        primary = []
        if "deepfake_auc" in val_metrics:
            primary.append(val_metrics["deepfake_auc"])
        if "spoof_acer" in val_metrics:
            primary.append(1.0 - val_metrics["spoof_acer"])   # lower ACER = better
        if "stress_f1_macro" in val_metrics:
            primary.append(val_metrics["stress_f1_macro"])

        composite = float(np.mean(primary)) if primary else 0.0
        logger.info(f"[Val]   Composite metric: {composite:.4f} (best: {best_metric:.4f})")

        # ── Save Checkpoints ───────────────────────────────────
        state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "loss_weights": loss_weights.state_dict(),
            "best_metric":  best_metric,
            "cfg":          Config.training_config(),
        }
        save_checkpoint(state, str(ckpt_path))

        if composite > best_metric:
            best_metric      = composite
            patience_counter = 0
            save_checkpoint(state, str(best_path))
            logger.info(f"New best model saved → {best_path}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # ── Early Stopping ─────────────────────────────────────
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info("\nTraining complete.")
    logger.info(f"Best composite metric: {best_metric:.4f}")
    logger.info(f"Best checkpoint: {best_path}")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
