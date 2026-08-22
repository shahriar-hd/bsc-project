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

    # ── What keeps a FaceForensics++ video out of two splits ──────────────────
    # Measured on the DFD actor subset shipped here (400 videos), by
    # `scripts/audit_ff_split.py`:
    #
    #   "scene"     16 scripted scenarios, 9-35 videos each. Every one of the 200
    #               fakes shares its scenario with a real video, so this is the
    #               100 %-prevalence leak: the fake reuses that scenario's room,
    #               lighting, framing and clothing and differs only in the face
    #               region. Splitting here is clean AND label-balanced —
    #               real 142/31/27, fake 138/31/31 (~50 % fake in every split).
    #               Identities still span all splits; the residual is reported.
    #   "identity"  26 actor tokens, but a fake names two actors and ties them
    #               together — the union-find graph collapses to 3 components with
    #               353/400 videos (88.2 %) in one, so an identity-disjoint split
    #               does not exist. Assigning each fake to its first actor's split
    #               gets identity leakage to 0 only at real 167/22/11, fake
    #               186/11/3 — a test set resting on 3 fake videos. Scenario
    #               overlap stays at 100 % either way.
    #   "video"     one subject per video: what run01 used. Both leaks active,
    #               which is why its 0.9899 test AUC is not a generalisation
    #               estimate.
    #
    # "scene" is the default because it removes the leak that affects every fake
    # and is the only option that yields usable, balanced val/test sets.
    ff_split_key: str = "scene"           # "scene" | "identity" | "video"

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_every_n_videos: int = 20          # progress log interval
    jpeg_quality: int = 95                # 90-95: sharp enough, ~3× smaller than PNG

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
    # All four are set to the clip span (64), i.e. back-to-back with zero shared
    # frames, and the per-label caps below decide how many of those clips are
    # kept. Measured clip yields (`starts_for_stride` over the real frame counts):
    #
    #   FF++  real  200 videos, median 840 frames, cap 7 → 1350 clips
    #   FF++  fake  200 videos, median 696 frames, cap 7 → 1250 clips  (ratio 1.08)
    #   SiW   live  785 videos, median 179 frames, cap 2 → 1311 clips
    #   SiW   spoof 915 videos, per-type plan below     → 1364 clips  (ratio 1.04)
    #
    # So ~2600 FF++ clips against ~2675 SiW clips: each task is class-balanced
    # against itself, and the two tasks are balanced against each other, which
    # matters because InterleavedBatchSampler recycles whichever side is smaller
    # (run01 had 1287 FF vs 1533 SiW clips and saw every FF clip ~1.2x per epoch).
    #
    # A larger stride is what previously limited the yield, not the video length:
    # at stride 120 with cap 4 the same videos gave only 773 real clips out of the
    # ~7 non-overlapping windows an 840-frame video contains.
    clip_stride_real:  int = 64            # FF++ real    (back-to-back)
    clip_stride_fake:  int = 64            # FF++ fake    (back-to-back)
    clip_stride_live:  int = 64            # SiW-Mv2 live (back-to-back)
    clip_stride_spoof: int = 64            # SiW-Mv2 spoof fallback; the per-type
                                           # plan below overrides it when enabled

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

    # Per-dataset and per-label clip caps. With every stride at the clip span,
    # the stride finds all non-overlapping windows a video contains and the cap
    # decides how many are kept — so these are the knobs that set dataset size
    # and class balance. Yields are listed with the strides above.
    #
    # run01 used 4/4/1/1, which produced 1287 FF++ and 1533 SiW clips: a val split
    # of only ~197 clips, where a single clip moves AUC by ~0.005 and the epoch
    # curve looked like noise. These values roughly double both sides.
    max_clips_ff_real: Optional[int] = 7
    max_clips_ff_fake: Optional[int] = 7
    max_clips_siw_live: Optional[int] = 2
    max_clips_siw_spoof: Optional[int] = 1   # superseded per type by the plan below

    # ── Per-attack-type clip balancing (SiW-Mv2 spoof only) ───────────────────
    # The caps above balance the two *tasks* against each other, but say nothing
    # about the 14 attack types inside SiW-Mv2 spoof, which are very unevenly
    # represented. Measured video counts:
    #
    #   Partial_FunnyeyeGlasses 179   Mask_TransparentMask  60   Makeup_Obfuscation  22
    #   Paper                   135   Partial_Eye           57   Mask_PaperMask      17
    #   Replay                   98   Makeup_Cosmetic       52   Silicone            17
    #   Partial_PaperGlasses     76   Mannequin             40
    #   Mask_HalfMask            72   Partial_Mouth         29
    #
    # A 10.5x spread in videos, and run01 turned it into an 11.7x spread in
    # frames. Rare types cannot be lifted to parity: Silicone has 17 videos of
    # 210 frames, so reaching the 179-video types' clip count would need ~10
    # near-identical clips from each, sharing 55+ of their 64 frames. The honest
    # goal is therefore to *narrow* the gap, not close it — take more clips from
    # the rare types' videos while a real frame gap still exists, take one from
    # each video of the common types, and never drop a video (that would trade
    # imbalance for lost identity/scene diversity). What remains is closed at
    # training time, where a per-type sampling weight is free, and is reported
    # per type by the `sp_recall_<type>` metrics.
    #
    # None disables this entirely and falls back to the flat clip_stride_spoof.
    target_clips_per_spoof_type: Optional[int] = 90
    max_clips_per_video_spoof: int = 4    # ceiling on clips from one spoof video
    min_spoof_stride: int = 25            # floor on stride: caps overlap at 39/64 frames

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
    #
    # alpha = 0.5 is neutral. The RetinaNet convention (0.25) assumes the
    # positive class is *rare* — it puts weight 0.25 on class 1 and 0.75 on
    # class 0. Here class 1 is "presentation attack", which is ~52% of SiW-Mv2,
    # so 0.25 down-weighted the class the head exists to detect by 3x. gamma
    # still does the hard-example mining that focal loss is actually for.
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5

    # Gradient clipping
    max_grad_norm: float = 5.0

    # FF++ share of every training batch, enforced as a per-batch quota by
    # InterleavedBatchSampler: 0.5 at batch_size=4 means 2 FF++ + 2 SiW-Mv2 in
    # every batch. Clamped to [1, batch_size-1] slots, so no value can starve a
    # task.
    #
    # This used to be a *sampling weight* (siw_weight = (1 - ratio) / n_siw), and
    # the default of 1.0 therefore gave every SiW-Mv2 clip weight exactly 0.0.
    # torch.multinomial never draws a zero-weight index, so run01 trained for 14
    # epochs on FF++ alone: loss_sp was exactly 0.0 from epoch 2 and the spoof
    # head never produced an output above 0.4978.
    ff_sample_ratio: float = 0.5

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

    # 1e-4 is below the resolution of the validation sets. One clip out of ~197
    # moves AUC by ~0.005, so at 1e-4 pure jitter counts as "improvement" and
    # the plateau in run01 (range 0.0139 over epochs 3-13) read as 11 epochs of
    # progress followed by a patience countdown on noise.
    early_stopping_min_delta: float = 2e-3

    # Never stop before this epoch. warmup_epochs=3 puts the backbone at 3e-9 in
    # epoch 0, so patience counted during warmup measures the schedule, not the
    # model.
    early_stopping_min_epochs: int = 8

    # The *stop* decision uses a rolling mean over this many epochs; best.pth is
    # still selected on the raw value. 1 disables smoothing.
    early_stopping_smooth_window: int = 3

    # Which validation metric drives best-model selection, the plateau scheduler
    # and early stopping. "composite" uses composite_metric below; the others
    # are single-task, for ablations.
    early_stopping_metric: str = "composite"   # "composite" | "df_auc" | "sp_auc" | "acer"
    early_stopping_mode: str = "max"           # "max" for AUC, "min" for ACER

    # Composite formula:
    #   "auc"  → 0.5 * ff_df_auc_roc + 0.5 * siw_sp_auc
    #   "acer" → 0.5 * ff_df_auc_roc + 0.5 * (1 - siw_sp_acer)   (legacy)
    #
    # "acer" hid the run01 collapse. ACER = (APCER + BPCER) / 2 is exactly 0.5
    # for a head that predicts one class for everything — identical to a coin
    # flip — so the second term froze at 0.25 for all 14 epochs and the
    # composite tracked deepfake AUC alone. siw_sp_auc for the same collapsed
    # head was 0.41, i.e. visibly below chance. Both forms are always logged.
    composite_metric: str = "auc"              # "auc" | "acer"

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
    jpeg_quality_min: int = 75
    jpeg_quality_max: int = 95
    coarse_dropout_p: float = 0.1
    random_grayscale_p: float = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    fpr_threshold: float = 0.01               # TPR @ FPR=1%

    # Operating point for every threshold-dependent metric (accuracy, precision,
    # recall, F1, MCC, APCER/BPCER/ACER). Kept here rather than hardcoded in the
    # three metric functions so an ablation moves one number. `df_best_threshold`
    # is still reported separately (Youden-J on the deepfake ROC) — this is the
    # threshold the *reported* metrics and the demo actually use.
    decision_threshold: float = 0.5

    # Video-level aggregation
    video_agg: str = "mean"                   # "mean" | "max"

    # Compression breakdown. Requires a c23/c40 token in the source video paths;
    # the raw FaceForensics++ layout here (DFD actor subset, e.g.
    # "01_02__hugging_happy__YVGY8LOK.mp4") carries no such token, so this
    # breakdown is unavailable and logged as such once per run rather than
    # silently returning nothing.
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
