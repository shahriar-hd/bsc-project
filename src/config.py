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
        "/home/shahriar/Documents/bsc-project/data/datasets/raw"
    )
    processed_root: Path = Path(
        "/home/shahriar/Documents/bsc-project/data/datasets/processed"
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

    # ── CSV output directory (must match PathConfig.csv_root) ─────────────────
    csv_dir: Path = Path(
        "/home/shahriar/Documents/bsc-project/data/datasets/processed/csv"
    )

    # ── Video extensions to scan ──────────────────────────────────────────────
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    # ── InsightFace ───────────────────────────────────────────────────────────
    # buffalo_l  → detection + 512-d recognition embedding (demo app: identity)
    # buffalo_sc → detection only, ~3x faster (preprocessing: quality gate)
    insightface_model_name: str = "buffalo_l"
    insightface_det_model: str = "buffalo_sc"
    insightface_ctx_id: int = 0           # GPU id; -1 for CPU
    insightface_det_size: Tuple[int, int] = (640, 640)
    insightface_modules: List[str] = field(default_factory=lambda: ["detection"])

    # ── Face crop & output image ──────────────────────────────────────────────
    output_face_size: int = 224           # final square crop (px)
    crop_scale: float = 1.1               # padding around aligned face

    # ── Train / Val split ratios (test gets the remainder) ────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Random seed for reproducible splits
    split_seed: int = 42

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_every_n_videos: int = 20          # progress log interval
    jpeg_quality: int = 92                # 90-95: sharp enough, ~3× smaller than PNG

    # ── Clip sampling ─────────────────────────────────────────────────────────
    # A clip is `frames_per_clip` frames taken every `frame_skip` source frames,
    # so it spans  (frames_per_clip - 1) * frame_skip + 1  source frames.
    frames_per_clip: int = 64

    # Temporal subsampling *inside* a clip. 1 = every source frame.
    # Keep at 1: SiW-Mv2 videos are short (median 150 frames), and skip=2 would
    # make one clip span 127 source frames, dropping most SiW videos entirely.
    frame_skip: int = 1

    # Distance between the START of consecutive clips in the same video, in
    # source frames (sliding-window stride):
    #   stride >  clip span → clips are separated by (stride - span) frames
    #   stride == clip span → clips are back-to-back, no shared frames
    #   stride <  clip span → clips overlap and share (span - stride) frames
    #
    # Tuned on the measured frame-count distribution of the two datasets so
    # that each task's own classes are balanced (that is what the deepfake BCE
    # and the spoof focal loss actually see) and both datasets contribute a
    # comparable number of clips (ff_sample_ratio=1.0 assumes this):
    #
    #   FF++  real  200 videos, median 840 frames → ~1328 clips
    #   FF++  fake  200 videos, median 703 frames → ~1297 clips   (ratio 1.02)
    #   SiW   live  785 videos, median 179 frames → ~1260 clips
    #   SiW   spoof 915 videos, median 150 frames → ~1263 clips   (ratio 1.00)
    #
    # FF++ real and SiW live are both labelled "real" but need different
    # strides (840 vs 179 median frames), hence two separate values.
    clip_stride_real:  int = 120          # FF++ real  (no overlap, 56f gap)
    clip_stride_fake:  int = 100          # FF++ fake  (no overlap, 36f gap)
    clip_stride_live:  int = 120          # SiW-Mv2 live (no overlap)
    clip_stride_spoof: int = 45           # SiW-Mv2 spoof (19f overlap: short videos)

    # Disallow overlapping frames between clips extracted from the same video.
    # If False, strides < span (e.g. spoof=45) will produce overlapping clips.
    # If True, effective stride is clamped to at least clip_span (64), guaranteeing
    # that every 64-frame clip extracted from a video contains unique non-overlapping frames.
    allow_clip_overlap: bool = False

    # Minimum clips to extract per video if total_frames >= frames_per_clip.
    # Ensures every video in the dataset contributes at least 1 clip if long enough.
    min_clips_per_video: int = 1

    # Hard cap on clips per video. 0 = no cap, keep every clip the stride finds.
    # Set to 1 to extract at least/most 1 clip per video, or > 1 for multiple clips.
    max_clips_per_video: int = 1

    # Per-dataset and per-label clip caps for perfect dataset balancing across MTL heads:
    # - FaceForensics++ (200 vids/class, ~800 frames each): 4 clips -> ~750-800 balanced clips
    # - SiW-Mv2 (800-900 vids/class, ~160 frames each): 1 clip -> ~750-800 balanced clips
    # This guarantees 1:1 balance for both Deepfake (FF) and Anti-Spoof (SiW) heads with 0 frame overlap.
    max_clips_ff_real: Optional[int] = 4
    max_clips_ff_fake: Optional[int] = 4
    max_clips_siw_live: Optional[int] = 1
    max_clips_siw_spoof: Optional[int] = 1

    # ── Quality gate ──────────────────────────────────────────────────────────
    min_face_score: float = 0.65          # InsightFace det_score threshold
    min_valid_frames: int = 48            # min surviving frames for a clip to count
    disp_ratio: float = 0.30              # min IoU(face, clip window) to keep a frame
    margin: int = 25                      # frames skipped at video head/tail

    # Shrink `margin` on short videos instead of rejecting them: a fixed 25-frame
    # margin at both ends costs 50 of a 150-frame SiW video, which dropped 168
    # SiW videos (125 spoof + 43 live) that are otherwise perfectly usable.
    # Effective margin = min(margin, (total_frames - clip_span) // 4).
    adaptive_margin: bool = True

    # ── Optical flow precompute ───────────────────────────────────────────────
    # Flow is computed here, once, and written next to each clip's crops as
    # c<clip>_flow.npz. train.py only ever *reads* it — Farneback over a
    # 64-frame clip costs ~0.4 s, which 2 dataloader workers cannot absorb at
    # training speed, so computing it in __getitem__ would starve the GPU.
    # Consumed only when TrainConfig.use_optical_flow is True.
    precompute_optical_flow: bool = True
    optical_flow_method: str = "farneback"   # only "farneback" is implemented

    # Stored flow resolution. The encoder is Conv2d(2→32, 3×3) followed by
    # AdaptiveAvgPool2d((4, 4)), so spatial detail beyond ~56 px is pooled away
    # before it reaches a linear layer. Disk is the real constraint: 2848 clips
    # × 63 pairs is 16.8 GB at 112 in float32, against 29 GB free. At 56 with
    # int8 + a per-clip scale it measures ~126 KB/clip → ~350 MB total.
    flow_resize: int = 56

    # ── Parallelism ───────────────────────────────────────────────────────────
    # Worker processes for video extraction. Each worker runs its own
    # InsightFace/ONNX session, so both GPU memory and RAM scale with it.
    #   0 = auto-size from free GPU memory, available RAM and core count
    #   1 = sequential, everything in the parent process (easiest to debug)
    num_workers: int = 0
    # Measured on a 4 GB RTX 3050 Ti (12 videos, 1326 frames): 1 worker 23.2s,
    # 3 workers 14.8s, 4 workers 14.7s, 6 workers 15.3s. Detection on one GPU
    # is the bottleneck, so past 4 workers contention costs more than it gains.
    max_workers: int = 4

    # Per-worker ONNX CUDA arena limit. det_500m at 640x640 needs very little;
    # the cost is mostly the ~300 MB CUDA context each process creates.
    worker_gpu_mem_mb: int = 256

    # Threads *inside* each worker. Left at 1-2 on purpose: N workers each
    # spawning 16 OpenCV threads oversubscribes the CPU and runs slower than
    # the sequential version. `worker_omp_threads` is exported as
    # OMP_NUM_THREADS before the ONNX session is built.
    worker_cv_threads: int = 2
    worker_omp_threads: int = 2

    # RAM budget per worker (MB). A worker buffers `frames_per_clip` decoded
    # frames; 64 x 1080p BGR is ~400 MB, so this is what actually limits the
    # worker count on a 16 GB machine.
    worker_ram_mb: int = 900

    # ── Power monitoring ──────────────────────────────────────────────────────
    monitor_power: bool = True
    power_log_csv: str = "power_preprocess.csv"

# ══════════════════════════════════════════════════════════════════════════════
# Paths  (training artefacts)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    data_root: str = (
        "/home/shahriar/Documents/bsc-project/data/datasets/processed/"
    )
    csv_root: str = data_root + "csv/"
    master_csv: str = csv_root + "master.csv"
    siwmv2_train_csv: str = csv_root + "siw_train.csv"
    siwmv2_val_csv: str = csv_root + "siw_val.csv"
    siwmv2_test_csv: str = csv_root + "siw_test.csv"
    ff_train_csv: str = csv_root + "ff_train.csv"
    ff_val_csv: str = csv_root + "ff_val.csv"
    ff_test_csv: str = csv_root + "ff_test.csv"

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
    backbone_lr_scale: float = 0.1             # lr_backbone = scale * lr_heads
    feature_dim: int = 1408                    # EfficientNet-B2 output dim
    dropout: float = 0.3

    # Head dims
    deepfake_hidden: int = 256
    spoof_hidden: int = 256
    temporal_hidden: int = 256

    # TSM (Temporal Shift Module)
    use_tsm: bool = True
    tsm_shift_ratio: float = 0.125
    # TSM is applied to the feature maps entering these backbone blocks. On
    # EfficientNet-B2 they carry 16 / 48 / 120 channels, all divisible by the
    # 1/8 shift ratio. Never shift the raw 3-channel input — that swaps colour
    # planes between frames instead of mixing temporal context.
    tsm_block_indices: tuple = (1, 3, 5)

    # Recompute backbone block activations in the backward pass instead of
    # storing them. Measured on the 4 GB RTX 3050 Ti at batch_size=4,
    # num_frames=8 with PCGrad + GradNorm: 2426 MiB → OOM without it,
    # 1133 MiB peak with it. Costs ~30% step time because PCGrad backwards the
    # graph once per task and each backward re-runs the blocks.
    grad_checkpointing: bool = True

    # ── Temporal head ────────────────────────────────────────────────────────
    # Which branches TemporalHead builds and supervises:
    #   "cosine_sim"   → projection + classifier only; no flow encoder built
    #   "optical_flow" → adds the flow encoder
    #   "combined"     → same as optical_flow here, since the pseudo-label
    #                    branch is deliberately not wired (temporal_label is
    #                    real ground truth, so a deepfake-derived pseudo-label
    #                    would be circular self-distillation)
    temporal_supervision: str = "combined"
    temporal_proj_dim: int = 128             # projection dim inside temporal head
    optical_flow_in_channels: int = 2        # dx, dy — see utils.flow_utils.FLOW_CHANNELS
    # Weight of the BCE term on the temporal head's classifier logit. The
    # cosine-similarity term only trains `proj`; `clf` — whose output is what
    # validate() reports temporal accuracy/AUC on — receives no gradient
    # without this. Set to 0.0 to reproduce the projection-only ablation.
    temporal_logit_loss_weight: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    seed: int = 42
    num_epochs: int = 25
    batch_size: int = 4                      # per GPU (optimized for 4GB VRAM)
    num_workers: int = 2                     # 2 workers to keep RAM footprint low
    pin_memory: bool = True
    grad_accum_steps: int = 4                # effective batch = batch_size * accum = 16
    gradnorm_interval: int = 10              # run GradNorm update every 10 steps
    resume: bool = True
    # _save_checkpoint() reads this every epoch; without it the first
    # checkpoint save raised AttributeError and killed the run *after* a full
    # epoch of training had already completed. False keeps only last.pth and
    # best.pth — a per-epoch snapshot of this model is ~107 MB, so 25 epochs
    # of them would write ~2.7 GB.
    save_every_epoch: bool = False

    # Temporal sampling
    num_frames: int = 8                      # 8 frames per clip (optimal memory & temporal representation)
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

    # Gradient clipping
    max_grad_norm: float = 5.0

    # Interleaved sampling ratio (FF++ : SiW-Mv2)
    ff_sample_ratio: float = 1.0

    # ── Device & precision ──────────────────────────────────────────────────
    device: str = "cuda"               # "cuda", "cpu", or "cuda:0,1,..."
    amp_dtype: str = "float16"         # "float16" | "bfloat16" | "float32" (float32 = AMP disabled)
    fallback_to_cpu: bool = True       # if CUDA unavailable, fall back to CPU silently

    # ── Multi-GPU ────────────────────────────────────────────────────────────
    use_data_parallel: bool = False    # DataParallel (single-node multi-GPU)
    gpu_ids: list = field(default_factory=lambda: [])  # e.g. [0,1]; empty = all visible

    # ── Early stopping ───────────────────────────────────────────────────────
    use_early_stopping: bool = True
    early_stopping_patience: int = 7   # epochs without improvement before stopping
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "primary"  # "primary" | "df_auc" | "sp_auc" | "acer"
    early_stopping_mode: str = "max"        # "max" for AUC, "min" for ACER

    # ── Warmup ─────────────────────────────
    warmup_epochs: int = 3             # linear warmup before cosine schedule

    # ── Optical flow (read from disk; never computed here) ───────────────────
    # Off by default: turning it on only makes the dataloader *load* the .npz
    # files PreprocessConfig.precompute_optical_flow wrote, so nothing about the
    # training step computes flow either way. With it off, TemporalHead.flow_encoder
    # receives no gradient — which is the current, verified baseline.
    #
    # Requires: preprocessing run with precompute_optical_flow = True, and
    # ModelConfig.temporal_supervision in ("optical_flow", "combined") so the
    # encoder is actually built. Missing files downgrade to a warning, not a crash.
    use_optical_flow: bool = False
    # Weight of the BCE term on the flow encoder's consistency score. This is
    # what gives flow_encoder a gradient: judge real vs. fake from motion alone.
    flow_loss_weight: float = 0.2

    # PCGrad
    use_pcgrad: bool = True

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

    # ── Emission factor ──────────────────────────────────────────────────────
    # kg CO₂ per kWh. 0.494 = Iran grid average (also close to EU average).
    co2_kg_per_kwh: float = 0.494
    # Accounts for PSU losses and components not covered by RAPL/nvidia-smi.
    overhead_multiplier: float = 1.15

    # ── Multi-GPU power monitoring ───────────────────────────────────────────
    monitor_all_gpus: bool = True      # track every visible CUDA device
    gpu_ids: list = field(default_factory=lambda: [])  # empty = all visible GPUs
    per_gpu_log: bool = True           # include per-GPU breakdown in output


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogConfig:
    """Shared logger settings (train / preprocessing / demo)."""
    console_level: str = "INFO"        # level for stdout handler
    file_level: str = "DEBUG"          # level for file handler
    fmt: str = "%(asctime)s | %(levelname)-7s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


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
    log: LogConfig = field(default_factory=LogConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    # Runtime (set automatically)
    device: str = "cuda"
    run_id: str = ""
    resume: bool = True                        # auto-resume from last checkpoint


def get_config() -> Config:
    """Return default config instance."""
    return Config()
