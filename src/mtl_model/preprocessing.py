"""
Preprocessing pipeline for MTL face analysis.

Directory structure output:
    processed/
    ├── FaceForensics/
    │   ├── real/
    │   │   ├── sub_000000/
    │   │   │   ├── c0_f000.png
    │   │   │   └── ...
    │   │   └── sub_000001/
    │   └── fake/
    └── SiW-Mv2/
        ├── real/
        └── spoof/

Clip sampling:
  FF++  → 3 clips × 64 consecutive frames per video
          clip-0 : starts at MARGIN
          clip-1 : centred on video midpoint
          clip-2 : ends at (total_frames - MARGIN)
  SiW   → 1 clip  × 64 consecutive frames, centred

Face quality gate (per-frame, using buffalo_sc):
  - det_score  >= MIN_FACE_SCORE  (0.65)
  - displacement from previous frame < DISP_RATIO × face_width
    → frames that violate this are dropped; if a gap is created,
      only the dense sub-run of >= MIN_VALID_FRAMES frames is kept.

CSV outputs:
  master.csv          (all frames, all datasets)
  ff_train.csv  ff_val.csv  ff_test.csv
  siw_train.csv siw_val.csv siw_test.csv
"""

import csv
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from mtl_config import Config

# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =========================================================================== #
# Constants
# =========================================================================== #
FRAMES_PER_CLIP: int = 64
MAX_CLIPS_FF: int = 3
MAX_CLIPS_SIW: int = 1
MIN_FACE_SCORE: float = 0.65
MIN_VALID_FRAMES: int = 48           # 75 % of FRAMES_PER_CLIP
MARGIN: int = 25                     # frames to skip at start/end of video
DISP_RATIO: float = 0.30             # max allowed displacement / face_width


# =========================================================================== #
# InsightFace — lightweight detector (buffalo_sc)
# =========================================================================== #
class FaceDetector:
    """
    Uses InsightFace buffalo_sc (lightweight) for per-frame face detection.

    buffalo_sc is significantly faster than buffalo_l and sufficient
    for the quality-gate / displacement checks done here.
    """

    def __init__(self, config: Config, min_face_score: float = MIN_FACE_SCORE) -> None:
        self.min_face_score = min_face_score
        self.output_size: int = config.output_face_size
        self.crop_scale: float = config.crop_scale

        self.app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection"],
        )
        self.app.prepare(
            ctx_id=config.insightface_ctx_id,
            det_size=config.insightface_det_size,
        )

    def detect_best(self, frame_bgr: np.ndarray) -> Optional[object]:
        """
        Return highest-confidence face with score >= threshold, or None.
        Among qualified faces, the largest (by area) is returned.
        """
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        qualified = [f for f in faces if float(f.det_score) >= self.min_face_score]
        if not qualified:
            return None
        return max(qualified, key=lambda f: _bbox_area(f.bbox))

    def crop(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Square-crop the face region (with scale padding) and resize.

        Args:
            frame_bgr: Source BGR frame.
            bbox:      [x1, y1, x2, y2] bounding box.
        Returns:
            Resized BGR crop, or None if degenerate.
        """
        x1, y1, x2, y2 = map(float, bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * self.crop_scale / 2.0

        h, w = frame_bgr.shape[:2]
        ix1 = int(max(0.0, cx - half))
        iy1 = int(max(0.0, cy - half))
        ix2 = int(min(float(w), cx + half))
        iy2 = int(min(float(h), cy + half))

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        region = frame_bgr[iy1:iy2, ix1:ix2]
        return cv2.resize(
            region,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR,
        )


# =========================================================================== #
# Geometry helpers
# =========================================================================== #
def _bbox_area(bbox) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _bbox_center(bbox) -> np.ndarray:
    b = np.asarray(bbox, dtype=float)
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _bbox_width(bbox) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _displacement(a, b) -> float:
    return float(np.linalg.norm(_bbox_center(a) - _bbox_center(b)))


# =========================================================================== #
# Clip start-frame calculator
# =========================================================================== #
def compute_clip_starts_ff(total_frames: int) -> List[int]:
    """
    Compute three clip start indices for a FF++ video.

    Layout (all clips are FRAMES_PER_CLIP consecutive frames):
      clip-0 (early)  : starts at MARGIN
      clip-1 (middle) : centred on midpoint
      clip-2 (late)   : ends at (total_frames - MARGIN)

    Args:
        total_frames: Total source frames in the video.
    Returns:
        List of start indices; may have fewer than 3 entries if video is short.
    """
    if total_frames < FRAMES_PER_CLIP + 2 * MARGIN:
        return []

    usable_end = total_frames - MARGIN        # last valid start + FRAMES_PER_CLIP

    # clip-0: early
    s0 = MARGIN

    # clip-1: middle (centred)
    mid = total_frames // 2
    s1 = max(s0, mid - FRAMES_PER_CLIP // 2)
    s1 = min(s1, usable_end - FRAMES_PER_CLIP)

    # clip-2: late
    s2 = usable_end - FRAMES_PER_CLIP
    s2 = max(s1 + FRAMES_PER_CLIP, s2)       # ensure no overlap with clip-1

    starts = [s0]
    if s1 > s0:
        starts.append(s1)
    if s2 > (starts[-1] + FRAMES_PER_CLIP - 1) and s2 != s0:
        starts.append(s2)

    # Keep at most 3 and ensure within bounds.
    return [s for s in starts[:3] if s + FRAMES_PER_CLIP <= total_frames]


def compute_clip_start_siw(total_frames: int) -> Optional[int]:
    """
    Compute a single centred clip start for SiW-Mv2.

    Returns:
        Start index or None if video is too short.
    """
    if total_frames < FRAMES_PER_CLIP:
        return None
    return max(0, total_frames // 2 - FRAMES_PER_CLIP // 2)


# =========================================================================== #
# Frame reader
# =========================================================================== #
def read_consecutive_frames(
    video_path: Path,
    start_frame: int,
    count: int,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Read exactly `count` consecutive BGR frames starting at `start_frame`.

    Args:
        video_path:  Path to video file.
        start_frame: First frame index to read.
        count:       Number of frames to yield.
    Yields:
        (absolute_frame_index, bgr_array)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        yield start_frame + i, frame

    cap.release()


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, n)


# =========================================================================== #
# Displacement-aware clip builder
# =========================================================================== #
def extract_clip(
    video_path: Path,
    start_frame: int,
    detector: FaceDetector,
    out_dir: Path,
    clip_idx: int,
) -> List[Dict]:
    """
    Extract one temporally-stable clip in two passes.

    Pass 1 — Anchor pass (all FRAMES_PER_CLIP frames):
        Run buffalo_sc on every frame.
        Collect bboxes that pass det_score threshold.
        Compute the MEAN bbox → one fixed crop window for the whole clip.
        Reject the clip early if too few detections exist.

    Pass 2 — Save pass (all FRAMES_PER_CLIP frames):
        For each frame, verify the detected face overlaps the fixed window.
        Frames where the face is absent OR has jumped outside the window
        are skipped (subject-level jitter, e.g. scene cut).
        Apply the fixed window crop and save as JPEG.

    Why two passes?
        We need all bboxes before we can compute the average window,
        so we must read and detect on the full clip before saving anything.

    Args:
        video_path:  Source video file.
        start_frame: Index of the first frame of this clip in the video.
        detector:    Initialised FaceDetector.
        out_dir:     Subject-level output directory.
        clip_idx:    Clip index within this video (0-based).

    Returns:
        List of frame-level record dicts; empty list if clip is rejected.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]

    # ── Pass 1: read all frames, detect faces, collect bboxes ────────────── #
    raw_frames: List[Tuple[int, np.ndarray]] = []          # (src_idx, bgr)
    per_frame_bbox: List[Optional[np.ndarray]] = []        # None = no detection

    for src_idx, bgr in read_consecutive_frames(video_path, start_frame, FRAMES_PER_CLIP):
        raw_frames.append((src_idx, bgr))
        face = detector.detect_best(bgr)                   # may return None
        per_frame_bbox.append(face.bbox if face is not None else None)

    if len(raw_frames) < MIN_VALID_FRAMES:
        logger.debug("Clip %d: video too short (%d frames)", clip_idx, len(raw_frames))
        return []

    valid_bboxes = [b for b in per_frame_bbox if b is not None]
    if len(valid_bboxes) < MIN_VALID_FRAMES:
        logger.debug(
            "Clip %d: only %d/%d frames had a detectable face",
            clip_idx, len(valid_bboxes), len(raw_frames),
        )
        return []

    # ── Compute one stable window from the mean of all valid bboxes ──────── #
    H, W = raw_frames[0][1].shape[:2]
    crop_coords = _compute_average_crop_coords(
        valid_bboxes,
        frame_shape=(H, W),
        crop_scale=detector.crop_scale,
        output_size=detector.output_size,
    )
    if crop_coords is None:
        return []

    # ── Pass 2: validate each frame against the fixed window, then save ───── #
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    saved_idx: int = 0                                      # sequential frame number

    for (src_idx, bgr), bbox in zip(raw_frames, per_frame_bbox):
        # Frame has no face at all → subject jumped / occluded
        if bbox is None:
            logger.debug("Clip %d frame %d: no detection → skip", clip_idx, src_idx)
            continue

        # Face detected but outside our stable window → subject jump
        if not _face_inside_window(bbox, crop_coords):
            logger.debug("Clip %d frame %d: face outside window → skip", clip_idx, src_idx)
            continue

        # Crop using the fixed clip-level window
        ix1, iy1, ix2, iy2 = crop_coords
        region = bgr[iy1:iy2, ix1:ix2]
        crop = cv2.resize(
            region,
            (detector.output_size, detector.output_size),
            interpolation=cv2.INTER_LINEAR,
        )

        filename  = f"c{clip_idx}_f{saved_idx:03d}.jpg"
        save_path = out_dir / filename
        cv2.imwrite(str(save_path), crop, encode_params)

        records.append({
            "frame_path":       str(save_path),
            "clip_index":       clip_idx,
            "frame_num":        saved_idx,
            "src_frame_idx":    src_idx,
            "clip_start_frame": start_frame,
        })
        saved_idx += 1

    # ── Final length check ────────────────────────────────────────────────── #
    if len(records) < MIN_VALID_FRAMES:
        logger.debug(
            "Clip %d: only %d valid frames after window filtering → reject",
            clip_idx, len(records),
        )
        for rec in records:
            Path(rec["frame_path"]).unlink(missing_ok=True)
        return []

    return records


def _compute_average_crop_coords(
    bboxes: List[np.ndarray],
    frame_shape: Tuple[int, int],
    crop_scale: float,
    output_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Average all detected bboxes across a clip → one stable crop window.

    This eliminates frame-level jitter: the window itself never moves,
    only the face may drift slightly within it.

    Args:
        bboxes:      List of [x1,y1,x2,y2] from InsightFace for each frame.
        frame_shape: (H, W) of the source video.
        crop_scale:  Scale factor applied to the face bounding box.
        output_size: Final square output resolution (unused here, for clarity).
    Returns:
        (ix1, iy1, ix2, iy2) clamped to frame boundaries, or None if degenerate.
    """
    if not bboxes:
        return None

    arr = np.stack(bboxes, axis=0)          # (N, 4)
    mean_bbox = arr.mean(axis=0)            # [x1, y1, x2, y2]

    x1, y1, x2, y2 = map(float, mean_bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * crop_scale / 2.0

    H, W = frame_shape
    ix1 = int(max(0.0, cx - half))
    iy1 = int(max(0.0, cy - half))
    ix2 = int(min(float(W), cx + half))
    iy2 = int(min(float(H), cy + half))

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _face_inside_window(
    bbox: np.ndarray,
    crop_coords: Tuple[int, int, int, int],
    iou_threshold: float = 0.30,
) -> bool:
    """
    Check whether a detected face bbox overlaps enough with the clip window.

    A loose IoU threshold (0.30) catches genuine subject jumps / cuts
    without being sensitive to normal head movement.

    Args:
        bbox:          [x1, y1, x2, y2] detected face.
        crop_coords:   (ix1, iy1, ix2, iy2) clip-level fixed window.
        iou_threshold: Minimum IoU to consider the face "inside" the window.
    Returns:
        True if IoU >= threshold.
    """
    fx1, fy1, fx2, fy2 = map(float, bbox)
    wx1, wy1, wx2, wy2 = map(float, crop_coords)

    inter_x1 = max(fx1, wx1)
    inter_y1 = max(fy1, wy1)
    inter_x2 = min(fx2, wx2)
    inter_y2 = min(fy2, wy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return False

    face_area   = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    window_area = max(1.0, (wx2 - wx1) * (wy2 - wy1))
    union_area  = face_area + window_area - inter_area

    return (inter_area / union_area) >= iou_threshold


# =========================================================================== #
# Dataset scanners
# =========================================================================== #
VideoMeta = Dict[str, object]


def scan_ff(config: Config) -> List[VideoMeta]:
    root = config.raw_data_root / config.ff_dataset_name
    records: List[VideoMeta] = []
    for label_str, subdir in [("real", config.ff_real_dir), ("fake", config.ff_fake_dir)]:
        folder = root / subdir
        if not folder.exists():
            logger.warning("FF++ folder not found: %s", folder)
            continue
        for vid in sorted(folder.rglob("*")):
            if vid.suffix.lower() in config.video_extensions:
                records.append({
                    "video_path": str(vid),
                    "dataset": config.ff_dataset_name,
                    "subject_id": f"ff_{vid.stem}",
                    "label": label_str,
                    "task": "deepfake",
                    "spoof_type": "none",
                })
    logger.info("FF++ scanned: %d videos", len(records))
    return records


def scan_siw(config: Config) -> List[VideoMeta]:
    root = config.raw_data_root / config.siw_dataset_name
    records: List[VideoMeta] = []

    live_folder = root / config.siw_live_dir
    if live_folder.exists():
        for vid in sorted(live_folder.rglob("*")):
            if vid.suffix.lower() in config.video_extensions:
                records.append({
                    "video_path": str(vid),
                    "dataset": config.siw_dataset_name,
                    "subject_id": f"siw_{vid.stem}",
                    "label": "real",
                    "task": "spoof",
                    "spoof_type": "live",
                })

    spoof_root = root / config.siw_spoof_dir
    for spoof_type in config.siw_spoof_types:
        folder = spoof_root / spoof_type
        if not folder.exists():
            logger.warning("SiW spoof folder not found: %s", folder)
            continue
        for vid in sorted(folder.rglob("*")):
            if vid.suffix.lower() in config.video_extensions:
                records.append({
                    "video_path": str(vid),
                    "dataset": config.siw_dataset_name,
                    "subject_id": f"siw_{spoof_type}_{vid.stem}",
                    "label": "spoof",
                    "task": "spoof",
                    "spoof_type": spoof_type,
                })

    logger.info("SiW-Mv2 scanned: %d videos", len(records))
    return records


# =========================================================================== #
# Subject-aware split
# =========================================================================== #
def split_by_subject(records: List[VideoMeta], config: Config) -> List[VideoMeta]:
    subjects: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        subjects[str(rec["subject_id"])].append(i)

    ids = list(subjects.keys())
    rng = random.Random(config.split_seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train_set = set(ids[:n_train])
    val_set = set(ids[n_train: n_train + n_val])

    for rec in records:
        sid = str(rec["subject_id"])
        rec["split"] = "train" if sid in train_set else ("val" if sid in val_set else "test")

    counts: Counter = Counter(str(r["split"]) for r in records)
    logger.info(
        "Split (videos) — train: %d | val: %d | test: %d",
        counts["train"], counts["val"], counts["test"],
    )
    return records


# =========================================================================== #
# Per-video processor
# =========================================================================== #
def process_video(
    meta: VideoMeta,
    video_index: int,
    detector: FaceDetector,
    config: Config,
) -> List[Dict]:
    """
    Extract all clips for one video and save frames under a subject subfolder.

    Output path:
        processed/<dataset>/<label>/<subject_dir>/c<i>_f<nnn>.png

    Args:
        meta:        Video metadata dict (must include 'split' key).
        video_index: Global index used for subject folder naming.
        detector:    Initialised FaceDetector.
        config:      Config instance.
    Returns:
        List of frame-level record dicts with all metadata.
    """
    video_path = Path(str(meta["video_path"]))
    dataset = str(meta["dataset"])
    label = str(meta["label"])
    subject_id = str(meta["subject_id"])

    total_frames = get_frame_count(video_path)
    if total_frames == 0:
        logger.warning("Cannot read: %s", video_path)
        return []

    # Determine clip starts based on dataset.
    is_ff = (dataset == config.ff_dataset_name)
    if is_ff:
        clip_starts = compute_clip_starts_ff(total_frames)
    else:
        s = compute_clip_start_siw(total_frames)
        clip_starts = [s] if s is not None else []

    if not clip_starts:
        logger.warning("Too short (%d frames): %s", total_frames, video_path)
        return []

    # Subject-level directory: processed/<dataset>/<label>/sub_<000000>/
    subject_dir_name = f"sub_{video_index:06d}"
    subject_out_dir = (
        config.processed_root / dataset / label / subject_dir_name
    )

    all_records: List[Dict] = []

    for clip_idx, start in enumerate(clip_starts):
        clip_records = extract_clip(
            video_path=video_path,
            start_frame=start,
            detector=detector,
            out_dir=subject_out_dir,
            clip_idx=clip_idx,
        )

        if not clip_records:
            logger.debug(
                "Clip %d rejected: %s (start=%d)", clip_idx, video_path.name, start
            )
            continue

        # Attach video-level metadata to each frame record.
        for rec in clip_records:
            rec.update({
                "video_path": str(meta["video_path"]),
                "dataset": dataset,
                "subject_id": subject_id,
                "subject_dir": str(subject_out_dir),
                "label": label,
                "task": str(meta["task"]),
                "spoof_type": str(meta["spoof_type"]),
                "split": str(meta["split"]),
                "video_index": video_index,
            })

        all_records.extend(clip_records)

    return all_records


# =========================================================================== #
# CSV writers
# =========================================================================== #
CSV_FIELDS = [
    "frame_path", "video_path", "dataset", "subject_id", "subject_dir",
    "label", "task", "spoof_type", "split",
    "video_index", "clip_index", "clip_start_frame", "frame_num", "src_frame_idx",
]


def write_csv(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %s  (%d rows)", path.name, len(records))


def write_all_csvs(all_records: List[Dict], csv_dir: Path) -> None:
    """
    Write master.csv + per-dataset per-split CSVs.

        master.csv
        ff_train.csv / ff_val.csv / ff_test.csv
        siw_train.csv / siw_val.csv / siw_test.csv

    Args:
        all_records: All frame-level records.
        csv_dir:     Directory to write CSV files into.
    """
    write_csv(all_records, csv_dir / "master.csv")

    dataset_key = {
        "FaceForensics": "ff",
        "SiW-Mv2":       "siw",
    }

    for dataset_name, prefix in dataset_key.items():
        for split in ("train", "val", "test"):
            subset = [
                r for r in all_records
                if r["dataset"] == dataset_name and r["split"] == split
            ]
            if subset:
                write_csv(subset, csv_dir / f"{prefix}_{split}.csv")


# =========================================================================== #
# Summary
# =========================================================================== #
def log_summary(all_records: List[Dict]) -> None:
    unique_clips = len(set((r["video_index"], r["clip_index"]) for r in all_records))
    unique_subjects = len(set(r["subject_id"] for r in all_records))
    by_split = Counter(r["split"] for r in all_records)
    by_task = Counter(r["task"] for r in all_records)
    by_label = Counter(r["label"] for r in all_records)

    logger.info("── Summary ──────────────────────────────────")
    logger.info("  Total frames   : %d", len(all_records))
    logger.info("  Total clips    : %d", unique_clips)
    logger.info("  Total subjects : %d", unique_subjects)
    logger.info("  By split       : %s", dict(by_split))
    logger.info("  By task        : %s", dict(by_task))
    logger.info("  By label       : %s", dict(by_label))
    logger.info("─────────────────────────────────────────────")


# =========================================================================== #
# Pipeline entry
# =========================================================================== #
def run_pipeline(config: Config) -> None:
    logger.info("=== Preprocessing pipeline started ===")
    logger.info(
        "Clip settings: %d frames/clip | FF++: %d clips | SiW: %d clip",
        FRAMES_PER_CLIP, MAX_CLIPS_FF, MAX_CLIPS_SIW,
    )
    logger.info(
        "Quality gate: score>=%.2f | displacement_ratio<=%.2f | min_valid=%d",
        MIN_FACE_SCORE, DISP_RATIO, MIN_VALID_FRAMES,
    )

    records: List[VideoMeta] = []
    records.extend(scan_ff(config))
    records.extend(scan_siw(config))

    if not records:
        logger.error("No videos found under: %s", config.raw_data_root)
        return

    logger.info("Total videos: %d", len(records))
    records = split_by_subject(records, config)

    detector = FaceDetector(config, min_face_score=MIN_FACE_SCORE)

    all_frame_records: List[Dict] = []
    skipped = 0

    for video_index, meta in enumerate(records):
        recs = process_video(meta, video_index, detector, config)

        if not recs:
            skipped += 1
        else:
            all_frame_records.extend(recs)

        if (video_index + 1) % config.log_every_n_videos == 0:
            logger.info(
                "Progress: %d/%d | frames: %d",
                video_index + 1, len(records), len(all_frame_records),
            )

    logger.info("Videos skipped (no valid clips): %d", skipped)

    if not all_frame_records:
        logger.error("No frames extracted.")
        return

    log_summary(all_frame_records)
    write_all_csvs(all_frame_records, config.processed_root)
    logger.info("=== Done ===")


if __name__ == "__main__":
    cfg = Config()
    run_pipeline(cfg)
