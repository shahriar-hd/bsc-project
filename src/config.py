"""
Central configuration for MTL training (deepfake, anti-spoof, temporal heads).
All tunable hyperparameters and paths are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
@dataclass
class PathConfig:
    data_root: str = "/home/shahriar/Documents/bank_did_auth/data/datasets/processed/"
    csv_root: str = data_root + "csv/"
    master_csv: str = csv_root + "master.csv"
    siwmv2_train_csv: str = csv_root + "SiW-Mv2_train.csv"
    siwmv2_val_csv: str = csv_root + "SiW-Mv2_val.csv"
    siwmv2_test_csv: str = csv_root + "SiW-Mv2_test.csv"
    ff_train_csv: str = csv_root + "FaceForensics++_train.csv"
    ff_val_csv: str = csv_root + "FaceForensics++_val.csv"
    ff_test_csv: str = csv_root + "FaceForensics++_test.csv"

    output_root: str = "./runs"
    checkpoint_dir: str = "./checkpoints"
    log_file: str = "training.log"
    result_csv: str = "results.csv"


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b2"          # or "vit_small_patch16_224"
    pretrained: bool = True
    pretrained_source: str = "imagenet"        # "imagenet" | "vggface2" | path
    freeze_backbone_epochs: int = 2            # phase-1: heads only
    backbone_lr_scale: float = 0.1             # lr_backbone = scale * lr_heads
    feature_dim: int = 1408                    # EfficientNet-B2 output dim
    dropout: float = 0.3

    # Head dims
    deepfake_hidden: int = 256
    spoof_hidden: int = 256
    temporal_hidden: int = 256
    num_spoof_classes: int = 2                 # binary spoof (real/attack)

    # TSM (Temporal Shift Module)
    use_tsm: bool = True
    tsm_shift_ratio: float = 0.125


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    seed: int = 42
    num_epochs: int = 25
    batch_size: int = 16                       # per GPU
    num_workers: int = 4
    pin_memory: bool = True
    grad_accum_steps: int = 2                  # effective batch = batch_size * accum

    # Temporal sampling
    num_frames: int = 8                        # T frames per clip
    temporal_jitter: bool = True
    min_frame_gap: int = 1
    max_frame_gap: int = 4

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"                 # "cosine" | "step" | "plateau"
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    step_size: int = 10                        # for StepLR
    gamma: float = 0.5                         # for StepLR

    # Mixed precision
    use_amp: bool = True

    # GradNorm
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5               # task difficulty balancing

    # PCGrad (conflict resolution)
    use_pcgrad: bool = True

    # Initial task loss weights
    w_deepfake: float = 1.0
    w_spoof: float = 1.0
    w_temporal: float = 0.5

    # Focal loss (spoof head)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # OOM handling
    oom_fallback_cpu: bool = True
    reduce_batch_on_oom: bool = True
    min_batch_size: int = 4

    # Gradient clipping
    max_grad_norm: float = 5.0

    # Interleaved sampling ratio (FF++ : SiW-Mv2)
    ff_sample_ratio: float = 0.5


# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────
@dataclass
class AugConfig:
    image_size: int = 224
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)

    # Train augmentations
    random_flip: bool = True
    random_rotate: float = 10.0
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    saturation_jitter: float = 0.1
    hue_jitter: float = 0.05
    gaussian_blur_p: float = 0.2
    jpeg_compression_p: float = 0.3
    jpeg_quality_min: int = 50
    jpeg_quality_max: int = 95
    coarse_dropout_p: float = 0.1
    random_grayscale_p: float = 0.05


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
@dataclass
class EvalConfig:
    eval_every: int = 1                        # evaluate every N epochs
    fpr_threshold: float = 0.01               # TPR @ FPR=1%
    # Video-level aggregation
    video_agg: str = "mean"                   # "mean" | "max"
    c23_label: str = "c23"                    # compression label in dataset col
    c40_label: str = "c40"


# ──────────────────────────────────────────────
# Power Monitoring
# ──────────────────────────────────────────────
@dataclass
class PowerConfig:
    enable: bool = True
    poll_interval_sec: float = 5.0
    # Coefficients for non-GPU/CPU components
    ram_coeff: float = 0.375                  # W per GB used (DDR4 ~3W/8GB)
    ssd_coeff: float = 2.0                    # W fixed estimate for NVMe SSD
    other_coeff: float = 5.0                  # W for mobo, fans, etc.
    # RAPL paths (Linux)
    rapl_path: str = "/sys/class/powercap/intel-rapl"


# ──────────────────────────────────────────────
# Demo Config
# ──────────────────────────────────────────────


@dataclass
class DemoPathConfig:
    """File/directory paths used by the demo app."""
    input_dir: str = "data/input"
    model_path: str = "models/best.pth"
    instance_file: str = "data/instance.json"
    buffalo_model: str = "buffalo_l"          # insightface model name


@dataclass
class DemoCameraConfig:
    """Camera capture settings."""
    device_index: int = 0
    target_frames: int = 64
    fps: float = 20.0


@dataclass
class DemoModelConfig:
    """MTL inference and identity matching thresholds."""
    deepfake_threshold: float = 0.6
    spoof_threshold: float = 0.6
    temporal_threshold: float = 0.8
    identity_threshold: float = 0.75          # cosine similarity
    enroll_frame_indices: list = None         # filled in __post_init__

    def __post_init__(self):
        if self.enroll_frame_indices is None:
            self.enroll_frame_indices = [20, 25, 30, 35, 40]


@dataclass
class DemoConfig:
    """Top-level demo configuration."""
    paths: DemoPathConfig = None
    camera: DemoCameraConfig = None
    model: DemoModelConfig = None

    def __post_init__(self):
        if self.paths is None:
            self.paths = DemoPathConfig()
        if self.camera is None:
            self.camera = DemoCameraConfig()
        if self.model is None:
            self.model = DemoModelConfig()


# ──────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)


    # Runtime (set automatically)
    device: str = "cuda"
    run_id: str = ""
    resume: bool = True                        # auto-resume from last checkpoint


def get_config() -> Config:
    """Return default config instance."""
    return Config()