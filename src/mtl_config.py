"""
Central configuration for the MTL face analysis preprocessing pipeline.
All tunable parameters are defined here and imported project-wide.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    raw_data_root: Path = Path("/home/shahriar/Documents/bank_did_auth/data/datasets/raw")
    processed_root: Path = Path("/home/shahriar/Documents/bank_did_auth/data/datasets/processed")
    master_csv_path: Path = Path("/home/shahriar/Documents/bank_did_auth/data/datasets/master.csv")

    # ------------------------------------------------------------------ #
    # Dataset names (must match folder names under raw_data_root)
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Video extensions to scan
    # ------------------------------------------------------------------ #
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    # ------------------------------------------------------------------ #
    # InsightFace / Buffalo_L
    # ------------------------------------------------------------------ #
    insightface_model_name: str = "buffalo_l"
    insightface_ctx_id: int = 0          # GPU id; -1 for CPU
    insightface_det_size: Tuple[int, int] = (640, 640)

    # ------------------------------------------------------------------ #
    # Face crop & output image
    # ------------------------------------------------------------------ #
    output_face_size: int = 224          # final square crop (px)
    crop_scale: float = 1.1              # padding around aligned face

    # ------------------------------------------------------------------ #
    # Bounding-box smoothing
    # ------------------------------------------------------------------ #
    # Exponential moving average alpha for bbox smoothing.
    # Lower = more smoothing, higher = faster response to real motion.
    bbox_ema_alpha: float = 0.35

    # Max allowed per-frame displacement relative to face size
    # before a detection is treated as a jitter (not real motion).
    # Expressed as fraction of the face's shorter side.
    bbox_jitter_threshold: float = 0.40

    # Number of consecutive "jitter" frames allowed before we
    # accept the new position as genuine subject motion.
    bbox_jitter_tolerance: int = 3

    # ------------------------------------------------------------------ #
    # Frame sampling
    # ------------------------------------------------------------------ #
    # Target clip length (number of frames per clip stored in CSV).
    t_clip: int = 64

    # Skip every N source frames before sampling a clip frame.
    # At 30 fps: skip=2 → effective 10 fps input to clip.
    # At 60 fps: skip=4 → same effective rate.
    frame_skip: int = 2

    # Minimum frames that must be extracted from a video for it
    # to be included in the dataset.
    min_frames_per_video: int = 8

    # ------------------------------------------------------------------ #
    # Train / Val / Test split ratios (must sum to 1.0)
    # ------------------------------------------------------------------ #
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed for reproducible splits
    split_seed: int = 42

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    log_every_n_videos: int = 20         # progress log interval

    jpeg_quality: int = 92   # 90-95: sharp enough, ~3x smaller than PNG
    frames_per_clip: int = 64
    min_valid_frames: int = 48
    min_face_score: float = 0.65

