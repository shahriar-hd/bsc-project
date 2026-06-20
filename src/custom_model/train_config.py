"""
train_config.py
Unified configuration for train3.py.
All values are overridable via .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: str = "False") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# Backbone output feature dimensions
BACKBONE_DIMS: dict[str, int] = {
    "mobilenet":       576,
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
}

_VALID_BACKBONES = set(BACKBONE_DIMS.keys())

_backbone_raw = os.getenv("BACKBONE_TYPE", "mobilenet").strip().lower()
if _backbone_raw not in _VALID_BACKBONES:
    raise ValueError(
        f"BACKBONE_TYPE='{_backbone_raw}' is invalid. "
        f"Choose from: {sorted(_VALID_BACKBONES)}"
    )


class Config:
    # ── Dataset Paths ──────────────────────────────────────────────────────────
    DATASETS_PATH              = os.getenv("DATASETS_PATH",              "data/datasets/")
    SIW_DATASET_PATH           = os.getenv("SIW_DATASET_PATH",           "data/datasets/raw/SiW-Mv2/")
    FACEFORENSICS_DATASET_PATH = os.getenv("FACEFORENSICS_DATASET_PATH", "data/datasets/raw/FaceForensics++/")
    CASME2_DATASET_PATH        = os.getenv("CASME2_DATASET_PATH",        "data/datasets/raw/CASME2/raw/")
    CASME2_XLSX_PATH           = os.getenv("CASME2_XLSX_PATH",           "data/datasets/raw/CASME2/casme2_metadata.xlsx")
    PROCESSED_DATASET_PATH     = os.getenv("PROCESSED_DATASET_PATH",     "data/datasets/processed/")
    ANNOTATIONS_PATH           = os.getenv("ANNOTATIONS_PATH",           "data/datasets/processed/annotations/")
    CHECKPOINTS_PATH           = os.getenv("CHECKPOINTS_PATH",           "data/datasets/processed/checkpoints/")

    # ── Training I/O ───────────────────────────────────────────────────────────
    MASTER_CSV_PATH = os.getenv("MASTER_CSV_PATH", "data/datasets/processed/annotations/master.csv")
    CHECKPOINT_DIR  = os.getenv("CHECKPOINT_DIR",  "models/checkpoints/")
    TRAIN_LOG_DIR   = os.getenv("TRAIN_LOG_DIR",   "logs/")          # unified: was LOG_DIR in some places

    # ── Data Processing ────────────────────────────────────────────────────────
    FACE_SIZE           = _int("FACE_SIZE",        224)
    FACE_MARGIN         = _float("FACE_MARGIN",    0.10)
    SIW_CROP_SIZE       = _int("SIW_CROP_SIZE",    450)
    FF_SAMPLE_EVERY     = _int("FF_SAMPLE_EVERY",  2)
    SIW_SAMPLE_EVERY    = _int("SIW_SAMPLE_EVERY", 5)
    CASME_NEUTRAL_PAD   = _int("CASME_NEUTRAL_PAD", 5)
    TRAIN_RATIO         = _float("TRAIN_RATIO",    0.70)
    VAL_RATIO           = _float("VAL_RATIO",      0.15)
    TEST_RATIO          = _float("TEST_RATIO",      0.15)
    DATASET_RANDOM_SEED = _int("DATASET_RANDOM_SEED", 64)
    FACE_ALIGN             = _bool("FACE_ALIGN",             "False")
    CASME2_USE_INSIGHTFACE = _bool("CASME2_USE_INSIGHTFACE", "False")
    VIDEO_EXTS             = os.getenv("VIDEO_EXTS", ".mp4,.mov,.avi,.mkv").split(",")
    USE_GPU                = _bool("USE_GPU", "True")
    DET_SIZE               = _int("DET_SIZE", 640)
    INSIGHTFACE_MODEL      = os.getenv("INSIGHTFACE_MODEL", "buffalo_sc")

    # ── Input / Clip ───────────────────────────────────────────────────────────
    FRAME_SIZE   = _int("FRAME_SIZE",   112)   # spatial resize target
    CLIP_LENGTH  = _int("CLIP_LENGTH",  16)    # unified: was CLIP_LENGTH / CLIP_T / CLIP_LEN_TRAINING

    # ── Training Loop ──────────────────────────────────────────────────────────
    NUM_EPOCHS              = _int("NUM_EPOCHS",              30)
    BATCH_SIZE              = _int("BATCH_SIZE",              8)
    GRAD_ACCUM_STEPS        = _int("GRAD_ACCUM_STEPS",        4)
    N_INIT_BATCHES          = _int("N_INIT_BATCHES",          5)
    NUM_WORKERS             = _int("NUM_WORKERS",             4)
    SEED                    = _int("TRAINING_RANDOM_SEED",    42)
    EARLY_STOPPING_PATIENCE = _int("EARLY_STOPPING_PATIENCE", 10)

    # ── Backbone ───────────────────────────────────────────────────────────────
    BACKBONE_TYPE          = _backbone_raw
    BACKBONE_DIM           = _int("BACKBONE_DIM", BACKBONE_DIMS[_backbone_raw])
    USE_TSM                = _bool("USE_TSM", "True")
    USE_GRAD_CHECKPOINT    = _bool("USE_GRAD_CHECKPOINT", "True")
    FREEZE_BACKBONE_STAGES = _int("FREEZE_BACKBONE_STAGES", 2)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    BACKBONE_LR  = _float("BACKBONE_LR",  1e-4)
    ADAPTER_LR   = _float("ADAPTER_LR",   5e-4)
    HEAD_LR      = _float("HEAD_LR",      1e-3)
    WEIGHT_DECAY = _float("WEIGHT_DECAY", 1e-4)

    # ── GradNorm ───────────────────────────────────────────────────────────────
    GRADNORM_ALPHA = _float("GRADNORM_ALPHA", 1.5)
    GRADNORM_LR    = _float("GRADNORM_LR",    1e-3)

    # ── Dataset Sampling Ratios ────────────────────────────────────────────────
    SAMPLE_RATIO: dict[str, int] = {
        "faceforensics": _int("FF_SAMPLE_RATIO",     2),
        "siwmv2":        _int("SIW_SAMPLE_RATIO",    2),
        "casme2":        _int("CASME2_SAMPLE_RATIO", 1),
    }

    # ── Model Architecture ─────────────────────────────────────────────────────
    TSM_FOLD_DIVISOR    = _int("TSM_FOLD_DIVISOR",    8)
    ADAPTER_DIM_DF      = _int("ADAPTER_DIM_DF",      256)
    ADAPTER_DIM_SP      = _int("ADAPTER_DIM_SP",      256)
    ADAPTER_DIM_ST      = _int("ADAPTER_DIM_ST",      256)
    ADAPTER_DROPOUT     = _float("ADAPTER_DROPOUT",   0.3)

    # ── AMP ────────────────────────────────────────────────────────────────────
    USE_AMP = _bool("USE_AMP", "True")

    # ── OOM Recovery ───────────────────────────────────────────────────────────
    OOM_RETRY_BATCH_SCALE = _float("OOM_RETRY_BATCH_SCALE", 0.5)   # scale batch on OOM
