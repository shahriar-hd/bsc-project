"""
config.py
Central configuration for the entire project:
  - preprocessing pipeline  (PreprocessConfig)
  - model architecture      (ModelConfig)
  - training                (TrainConfig)
  - augmentation            (AugConfig)
  - evaluation              (EvalConfig)
  - power monitoring        (PowerConfig)
  - demo app                (DemoConfig)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    raw_data_root: Path = Path(
        "/home/shahriar/Documents/bank_did_auth/data/datasets/raw"
    )
    processed_root: Path = Path(
        "/home/shahriar/Documents/bsc-project/data/datasets/processed"
    )
    master_csv_path: Path = Path(
        "/home/shahriar/Documents/bsc-project/data/datasets/processed/csv/master.csv"
    )

    # ── Dataset names (must match folder names under raw_data_root) ───────────
    ff_dataset_name: str = "FaceForensics++"
    siw_dataset_name: str = "SiW-Mv2"

    # FaceForensics++ sub-folders
    ff_real_dir: str = "real"
    ff_fake_dir: str = "fake"

    # SiW-Mv2 sub-folders
    siw_live_dir: str = "live"
    siw_spoof_dir: str = "spoof"
    siw_spoof_types: List[str] = field(default_factory=lambda: [
        "Makeup_Cosmetic",
        "Makeup_Impersonation",
        "Makeup_Obfuscation",
        "Mannequin",
        "Mask_HalfMask",
        "Mask_PaperMask",
        "Mask_TransparentMask",
        "Paper",
        "Partial_Eye",
        "Partial_FunnyeyeGlasses",
        "Partial_Mouth",
        "Partial_PaperGlasses",
        "Replay",
        "Silicone",
    ])

    # ── Video extensions to scan ──────────────────────────────────────────────
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    # ── InsightFace / Buffalo_L ───────────────────────────────────────────────
    insightface_model_name: str = "buffalo_l"
    insightface_ctx_id: int = 0           # GPU id; -1 for CPU
    insightface_det_size: Tuple[int, int] = (640, 640)

    # ── Face crop & output image ──────────────────────────────────────────────
    output_face_size: int = 224           # final square crop (px)
    crop_scale: float = 1.1               # padding around aligned face

    # ── Bounding-box smoothing ────────────────────────────────────────────────
    # Exponential moving average alpha for bbox smoothing.
    # Lower = more smoothing, higher = faster response to real motion.
    bbox_ema_alpha: float = 0.35

    # Max allowed per-frame displacement relative to face size before a
    # detection is treated as jitter (not real motion).
    # Expressed as fraction of the face's shorter side.
    bbox_jitter_threshold: float = 0.40

    # Number of consecutive "jitter" frames allowed before we accept the
    # new position as genuine subject motion.
    bbox_jitter_tolerance: int = 3

    # ── Frame sampling ────────────────────────────────────────────────────────
    # Target clip length (number of frames per clip stored in CSV).
    t_clip: int = 64

    # Skip every N source frames before sampling a clip frame.
    # At 30 fps: skip=2 → effective 10 fps input to clip.
    # At 60 fps: skip=4 → same effective rate.
    frame_skip: int = 2

    # Minimum frames that must be extracted from a video for it to be
    # included in the dataset.
    min_frames_per_video: int = 8

    # ── Train / Val / Test split ratios (must sum to 1.0) ─────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed for reproducible splits
    split_seed: int = 42

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_every_n_videos: int = 20          # progress log interval
    jpeg_quality: int = 92                # 90-95: sharp enough, ~3× smaller than PNG
    frames_per_clip: int = 64
    min_valid_frames: int = 48
    min_face_score: float = 0.65

    # clip sampling
    frames_per_clip: int = 64
    clip_stride_real:  int = 120
    clip_stride_fake:  int = 50
    clip_stride_spoof: int = 20
    max_clips_siw: int = 1

    # quality gate
    min_face_score: float = 0.65
    min_valid_frames: int = 48
    margin: int = 25
    disp_ratio: float = 0.30

# ══════════════════════════════════════════════════════════════════════════════
# Paths  (training artefacts)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    data_root: str = ( # TODO: Change path
        "/home/shahriar/Documents/bsc-project/data/datasets/processed/"
    )
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


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Temporal head redesign ───────────────────────────────────────────────
    temporal_supervision: str = "combined"
    # "cosine_sim"   → original (kept for ablation)
    # "pseudo_label" → mean of adjacent deepfake predictions (recommended)
    # "optical_flow" → uses precomputed optical flow signal
    # "combined"     → pseudo_label + optical flow (best)
    temporal_pseudo_threshold: float = 0.5   # min confidence to use a pseudo-label
    temporal_proj_dim: int = 128             # projection dim inside temporal head
    optical_flow_in_channels: int = 2        # must match TrainConfig.optical_flow_channels


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    seed: int = 42
    num_epochs: int = 25
    batch_size: int = 16                     # per GPU TODO: change batch size
    num_workers: int = 4
    pin_memory: bool = True
    grad_accum_steps: int = 2                  # effective batch = batch_size * accum
    gradnorm_interval = 10
    resume: bool = True                        # TODO: True 

    # Temporal sampling
    num_frames: int = 16                       # T frames per clip
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

    # Initial task loss weights
    w_deepfake: float = 1.0
    w_spoof: float = 1.0
    w_temporal: float = 1.0

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
    ff_sample_ratio: float = 1.0

    # ── Device & precision ──────────────────────────────────────────────────
    device: str = "cuda"               # "cuda", "cpu", or "cuda:0,1,..."
    amp_dtype: str = "bfloat16"         # "float16" | "bfloat16" | "float32" (float32 = AMP disabled)
    fallback_to_cpu: bool = True       # if CUDA unavailable, fall back to CPU silently
    # TODO:
    # ── Multi-GPU ────────────────────────────────────────────────────────────
    use_ddp: bool = True               # DistributedDataParallel (multi-node)
    use_data_parallel: bool = True     # DataParallel (single-node multi-GPU, simpler)
    gpu_ids: list = field(default_factory=lambda: [])  # e.g. [0,1]; empty = all visible

    # ── Early stopping ───────────────────────────────────────────────────────
    use_early_stopping: bool = True
    early_stopping_patience: int = 7   # epochs without improvement before stopping
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "primary"  # "primary" | "df_auc" | "sp_auc" | "acer"
    early_stopping_mode: str = "max"        # "max" for AUC, "min" for ACER

    # ── Warmup ─────────────────────────────
    warmup_epochs: int = 3             # linear warmup before cosine schedule

    # ── Optical flow for temporal branch ────────────────────────────────────
    use_optical_flow: bool = True
    optical_flow_method: str = "farneback"  # "farneback" | "raft" (raft needs extra install)
    optical_flow_channels: int = 2          # dx, dy → 2 channels appended to RGB
    optical_flow_cache: bool = True         # cache precomputed flows to disk

    # PCGrad
    use_pcgrad: bool = True

    # Temporal supervision
    flow_resize: int = 112
    pseudo_label_weight: float = 0.3
    flow_loss_weight: float = 0.2
    temporal_supervision: str = "combined"  # "pseudo_label" | "optical_flow" | "combined"



# ══════════════════════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    eval_every: int = 1                        # evaluate every N epochs
    fpr_threshold: float = 0.01               # TPR @ FPR=1%
    # Video-level aggregation
    video_agg: str = "mean"                   # "mean" | "max"
    c23_label: str = "c23"                    # compression label in dataset col
    c40_label: str = "c40"


# ══════════════════════════════════════════════════════════════════════════════
# Power Monitoring
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PowerConfig:
    enable: bool = True
    poll_interval_sec: float = 5.0
    # Coefficients for non-GPU/CPU components
    ram_coeff: float = 0.375                  # W per GB used (DDR4 ~3W/8GB)
    ssd_coeff: float = 2.0                    # W fixed estimate for NVMe SSD
    other_coeff: float = 5.0                  # W for mobo, fans, etc.
    # RAPL paths (Linux) TODO: run <sudo chmod -R a+r /sys/class/powercap/intel-rapl> in cli first
    rapl_path: str = "/sys/class/powercap/intel-rapl"

    # ── Multi-GPU power monitoring ───────────────────────────────────────────
    monitor_all_gpus: bool = True      # track every visible CUDA device
    gpu_ids: list = field(default_factory=lambda: [])  # empty = all visible GPUs
    per_gpu_log: bool = True           # include per-GPU breakdown in output


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

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
    enroll_frame_indices: List[int] = None    # filled in __post_init__

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


# ══════════════════════════════════════════════════════════════════════════════
# Master Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
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
