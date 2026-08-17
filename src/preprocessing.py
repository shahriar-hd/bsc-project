"""
Preprocessing pipeline for MTL face analysis.

Generic helpers (logging, face detection, video I/O, power monitoring) live in
src/utils/ and are shared with train.py and app_demo.py. This module keeps only
the dataset-specific logic: scanning, clip sampling, quality gating, splitting
and CSV writing.

Directory structure output:
    processed/
    ├── FaceForensics++/
    │   ├── real/
    │   │   ├── sub_000000/
    │   │   │   ├── c0_f000.jpg
    │   │   │   └── ...
    │   │   └── sub_000001/
    │   └── fake/
    └── SiW-Mv2/
        ├── real/
        └── spoof/

Clip sampling (dense, stride depends on label):
  FF++  real  → clip_stride_real    (sparse)
        fake  → clip_stride_fake
  SiW   live  → clip_stride_real
        spoof → clip_stride_spoof   (dense)

Face quality gate (per-frame):
  - det_score >= min_face_score (0.65)
  - frames whose bbox does not overlap the clip-level mean window by
    disp_ratio IoU are dropped; a clip with fewer than min_valid_frames
    survivors is rejected.

CSV outputs (under processed/csv/):
  master.csv
  ff_train.csv  ff_val.csv  ff_test.csv
  siw_train.csv siw_val.csv siw_test.csv
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import Config, PreprocessConfig, get_config
from src.utils.face_utils import FaceDetector
from src.utils.logger_utils import log_banner, setup_logger
from src.utils.power_utils import power_monitor_from_config
from src.utils.repro_utils import set_seed
from src.utils.video_utils import get_frame_count, read_consecutive_frames

logger = setup_logger(name="MTL")

VideoMeta = Dict[str, object]


# =========================================================================== #
# Clip start-frame calculators
# =========================================================================== #
def clip_span(config: PreprocessConfig) -> int:
    """
    Source frames one clip covers.

    `frames_per_clip` frames taken every `frame_skip` source frames span
    (frames_per_clip - 1) * frame_skip + 1 source frames — not
    `frames_per_clip`, which is what the clip-start maths needs.
    """
    return (config.frames_per_clip - 1) * max(1, config.frame_skip) + 1


def effective_margin(total_frames: int, config: PreprocessConfig) -> int:
    """
    Head/tail margin for this video.

    A fixed margin is too expensive on short videos: 25 frames at both ends of
    a 150-frame SiW clip removes a third of it, which rejected 168 otherwise
    usable SiW videos. When `adaptive_margin` is on, the margin shrinks to a
    quarter of the slack the video actually has.
    """
    if not config.adaptive_margin:
        return config.margin
    slack = max(0, total_frames - clip_span(config))
    return min(config.margin, slack // 4)


def compute_clip_starts(
    total_frames: int,
    stride: int,
    config: PreprocessConfig,
) -> List[int]:
    """
    Sliding-window clip sampling: a clip starts every `stride` source frames
    inside the usable region [margin, total_frames - margin].

    `stride` is measured start-to-start, so relative to the clip span it means:
        stride >  span → clips separated by (stride - span) frames
        stride == span → back-to-back clips, no shared frames
        stride <  span → clips overlap, sharing (span - stride) frames

    Both datasets share this rule; only the stride differs by dataset and
    label, so a single implementation replaces the former per-dataset
    duplicates.
    """
    if stride <= 0:
        return []

    span = clip_span(config)
    margin = effective_margin(total_frames, config)

    usable_start = margin
    usable_end = total_frames - margin
    if usable_end - usable_start < span:
        return []

    starts: List[int] = []
    s = usable_start
    while s + span <= usable_end:
        starts.append(s)
        s += stride

    if config.max_clips_per_video > 0:
        starts = starts[: config.max_clips_per_video]
    return starts


def stride_for_label(dataset: str, label: str, config: PreprocessConfig) -> int:
    """
    Map (dataset, label) to its clip stride.

    FF++ real and SiW-Mv2 live are both labelled "real" but are very different
    videos — 840 vs 179 median frames — so they cannot share a stride if the
    two tasks are to stay class-balanced.
    """
    if dataset == config.siw_dataset_name:
        return (
            config.clip_stride_live if label == "real"
            else config.clip_stride_spoof
        )
    return (
        config.clip_stride_fake if label == "fake"
        else config.clip_stride_real
    )


# =========================================================================== #
# Clip window geometry
# =========================================================================== #
def compute_average_crop_coords(
    bboxes: List[np.ndarray],
    frame_shape: Tuple[int, int],
    crop_scale: float,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Average all detected bboxes across a clip → one stable crop window.

    A single window per clip (instead of per-frame boxes) removes crop jitter,
    so the temporal head sees real motion rather than detector noise.
    """
    if not bboxes:
        return None

    mean_bbox = np.stack(bboxes, axis=0).mean(axis=0)
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


def face_inside_window(
    bbox: np.ndarray,
    window: Tuple[int, int, int, int],
    iou_threshold: float = 0.30,
) -> bool:
    """IoU(face bbox, clip window) >= threshold."""
    fx1, fy1, fx2, fy2 = map(float, bbox)
    wx1, wy1, wx2, wy2 = map(float, window)

    inter_w = max(0.0, min(fx2, wx2) - max(fx1, wx1))
    inter_h = max(0.0, min(fy2, wy2) - max(fy1, wy1))
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return False

    face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    window_area = max(1.0, (wx2 - wx1) * (wy2 - wy1))
    union_area = face_area + window_area - inter_area
    return (inter_area / union_area) >= iou_threshold


# =========================================================================== #
# Clip extractor
# =========================================================================== #
def extract_clip(
    video_path: Path,
    start_frame: int,
    detector: FaceDetector,
    out_dir: Path,
    clip_idx: int,
    config: PreprocessConfig,
) -> List[Dict]:
    """
    Detect faces across one clip, derive a stable crop window, and write the
    surviving frames as JPEGs. Returns one record per saved frame ([] if the
    clip fails the quality gate).
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]

    raw_frames: List[Tuple[int, np.ndarray]] = []
    per_frame_bbox: List[Optional[np.ndarray]] = []

    for src_idx, bgr in read_consecutive_frames(
        video_path, start_frame, config.frames_per_clip, step=config.frame_skip
    ):
        raw_frames.append((src_idx, bgr))
        face = detector.detect_best(bgr)
        per_frame_bbox.append(face.bbox if face is not None else None)

    if len(raw_frames) < config.min_valid_frames:
        return []

    valid_bboxes = [b for b in per_frame_bbox if b is not None]
    if len(valid_bboxes) < config.min_valid_frames:
        return []

    H, W = raw_frames[0][1].shape[:2]
    window = compute_average_crop_coords(
        valid_bboxes, frame_shape=(H, W), crop_scale=detector.crop_scale
    )
    if window is None:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    saved_idx = 0

    for (src_idx, bgr), bbox in zip(raw_frames, per_frame_bbox):
        if bbox is None:
            continue
        if not face_inside_window(bbox, window, iou_threshold=config.disp_ratio):
            continue

        crop = detector.crop_stable(bgr, window)
        if crop.size == 0:
            continue

        save_path = out_dir / f"c{clip_idx}_f{saved_idx:03d}.jpg"
        cv2.imwrite(str(save_path), crop, encode_params)

        records.append({
            "frame_path": str(save_path),
            "clip_index": clip_idx,
            "frame_num": saved_idx,
            "src_frame_idx": src_idx,
            "clip_start_frame": start_frame,
        })
        saved_idx += 1

    if len(records) < config.min_valid_frames:
        logger.debug(
            "Clip %d: only %d valid frames after window filtering → reject",
            clip_idx, len(records),
        )
        for rec in records:
            Path(rec["frame_path"]).unlink(missing_ok=True)
        return []

    return records
# =========================================================================== #
# Dataset scanners
# =========================================================================== #
def scan_ff(config: PreprocessConfig) -> List[VideoMeta]:
    root = config.raw_data_root / config.ff_dataset_name
    records: List[VideoMeta] = []
    for label_str, subdir in [
        ("real", config.ff_real_dir),
        ("fake", config.ff_fake_dir),
    ]:
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


def scan_siw(config: PreprocessConfig) -> List[VideoMeta]:
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
    else:
        logger.warning("SiW live folder not found: %s", live_folder)

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
def split_by_subject(
    records: List[VideoMeta],
    config: PreprocessConfig,
) -> List[VideoMeta]:
    """
    Assign train/val/test per *subject*, so no subject appears in two splits.

    Splitting is done per dataset: with one global shuffle a small dataset can
    land almost entirely in one split, which would silently break validation
    for that task.
    """
    by_dataset: Dict[str, List[VideoMeta]] = defaultdict(list)
    for rec in records:
        by_dataset[str(rec["dataset"])].append(rec)

    for dataset_name, ds_records in by_dataset.items():
        subjects: Dict[str, List[VideoMeta]] = defaultdict(list)
        for rec in ds_records:
            subjects[str(rec["subject_id"])].append(rec)

        ids = sorted(subjects.keys())          # sorted → deterministic
        rng = random.Random(config.split_seed)
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        train_set = set(ids[:n_train])
        val_set = set(ids[n_train: n_train + n_val])

        for sid, recs in subjects.items():
            split = (
                "train" if sid in train_set else
                "val" if sid in val_set else
                "test"
            )
            for rec in recs:
                rec["split"] = split

        counts = Counter(str(r["split"]) for r in ds_records)
        logger.info(
            "%-16s split (videos) — train: %d | val: %d | test: %d | subjects: %d",
            dataset_name, counts["train"], counts["val"], counts["test"], n,
        )

    return records


# =========================================================================== #
# Per-video processor
# =========================================================================== #
def process_video(
    meta: VideoMeta,
    video_index: int,
    detector: FaceDetector,
    config: PreprocessConfig,
) -> List[Dict]:
    video_path = Path(str(meta["video_path"]))
    dataset = str(meta["dataset"])
    label = str(meta["label"])
    subject_id = str(meta["subject_id"])

    total_frames = get_frame_count(video_path)
    if total_frames == 0:
        logger.warning("Cannot read: %s", video_path)
        return []

    stride = stride_for_label(dataset, label, config)
    clip_starts = compute_clip_starts(total_frames, stride, config)

    if not clip_starts:
        logger.warning("Too short (%d frames): %s", total_frames, video_path)
        return []

    subject_out_dir = (
        config.processed_root / dataset / label / f"sub_{video_index:06d}"
    )

    all_records: List[Dict] = []
    for clip_idx, start in enumerate(clip_starts):
        clip_records = extract_clip(
            video_path=video_path,
            start_frame=start,
            detector=detector,
            out_dir=subject_out_dir,
            clip_idx=clip_idx,
            config=config,
        )
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
            all_records.append(rec)

    return all_records


# =========================================================================== #
# Parallel extraction driver
# =========================================================================== #
# Per-process state. Each worker builds its own InsightFace session once, in
# the Pool initializer, instead of once per video.
_WORKER: Dict[str, object] = {}


def free_gpu_mb() -> Optional[float]:
    """Free memory on the configured GPU, or None if it cannot be determined."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.free / (1024 * 1024)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


def auto_worker_count(config: PreprocessConfig) -> int:
    """
    Pick a worker count that the machine can actually sustain.

    Three independent ceilings, whichever is lowest wins:
      * GPU  — each worker creates its own CUDA context (~350 MB) plus an
               ONNX arena of `worker_gpu_mem_mb`.
      * RAM  — a worker holds `frames_per_clip` decoded frames at once
               (~400 MB for 64 x 1080p), budgeted as `worker_ram_mb`.
      * CPU  — half the cores, leaving room for decode threads.
    """
    if config.num_workers > 0:
        return config.num_workers

    limits: List[int] = [max(1, (os.cpu_count() or 2) // 2)]
    reasons = [f"cpu={limits[0]}"]

    gpu_free = free_gpu_mb()
    if gpu_free is not None:
        per_worker = config.worker_gpu_mem_mb + 350
        gpu_limit = max(1, int(gpu_free // per_worker))
        limits.append(gpu_limit)
        reasons.append(f"gpu={gpu_limit} ({gpu_free:.0f} MB free)")

    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        ram_limit = max(1, int(avail_mb // max(1, config.worker_ram_mb)))
        limits.append(ram_limit)
        reasons.append(f"ram={ram_limit} ({avail_mb:.0f} MB avail)")
    except ImportError:
        pass

    workers = max(1, min(config.max_workers, min(limits)))
    logger.info("Worker auto-size → %d  [%s | max=%d]",
                workers, ", ".join(reasons), config.max_workers)
    return workers


def _init_worker(config: PreprocessConfig, log_path: Optional[str]) -> None:
    """
    Pool initializer: one detector per process, threads kept low.

    N workers each spawning as many OpenCV/OpenMP threads as there are cores
    oversubscribes the CPU badly enough to be slower than running sequentially,
    so both thread pools are pinned before any session is created.
    """
    import signal

    # Only the parent handles Ctrl-C; workers ignoring it lets the parent
    # terminate the pool cleanly instead of deadlocking on a half-dead worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.environ["OMP_NUM_THREADS"] = str(max(1, config.worker_omp_threads))
    cv2.setNumThreads(max(1, config.worker_cv_threads))

    setup_logger(log_path=log_path, name="MTL")

    detector = FaceDetector(
        config, need_embeddings=False, gpu_mem_mb=config.worker_gpu_mem_mb
    )
    _WORKER["config"] = config
    _WORKER["detector"] = detector

    logger.info("Worker %d ready — providers: %s",
                os.getpid(), detector.active_providers())


def _process_video_task(task: Tuple[int, VideoMeta]) -> Tuple[int, List[Dict]]:
    """Worker entry point: never raise, a bad video must not kill the pool."""
    video_index, meta = task
    config = _WORKER["config"]        # type: ignore[assignment]
    detector = _WORKER["detector"]    # type: ignore[assignment]
    try:
        return video_index, process_video(meta, video_index, detector, config)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("Failed %s: %s: %s",
                       meta.get("video_path"), type(exc).__name__, exc)
        return video_index, []


def _log_progress(done: int, total: int, frames: int, skipped: int) -> None:
    logger.info("Progress: %d/%d | frames: %d | skipped: %d",
                done, total, frames, skipped)


def extract_sequential(
    records: List[VideoMeta],
    config: PreprocessConfig,
) -> Tuple[List[Dict], int]:
    """Single-process extraction (num_workers=1, and the debugging path)."""
    detector = FaceDetector(
        config, need_embeddings=False, gpu_mem_mb=config.worker_gpu_mem_mb
    )
    all_records: List[Dict] = []
    skipped = 0
    try:
        logger.info("Detector providers: %s", detector.active_providers())
        for video_index, meta in enumerate(records):
            recs = process_video(meta, video_index, detector, config)
            if recs:
                all_records.extend(recs)
            else:
                skipped += 1
            if (video_index + 1) % config.log_every_n_videos == 0:
                _log_progress(video_index + 1, len(records),
                              len(all_records), skipped)
    except KeyboardInterrupt:
        logger.warning("Interrupted — keeping %d frames extracted so far",
                       len(all_records))
    finally:
        detector.close()
    return all_records, skipped


def extract_parallel(
    records: List[VideoMeta],
    config: PreprocessConfig,
    workers: int,
    log_path: Optional[str],
) -> Tuple[List[Dict], int]:
    """
    Extraction spread over `workers` processes, one video per task.

    Uses the "spawn" start method: forking a process that already holds CUDA
    and ONNX Runtime state gives a child with unusable GPU handles.

    `imap` (ordered) rather than `imap_unordered` so the CSV row order is
    identical to a sequential run — reproducibility is worth the small amount
    of head-of-line blocking, since tasks are one video each.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    all_records: List[Dict] = []
    skipped = 0

    pool = ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(config, log_path),
    )
    try:
        tasks = list(enumerate(records))
        for done, (_, recs) in enumerate(
            pool.imap(_process_video_task, tasks, chunksize=1), start=1
        ):
            if recs:
                all_records.extend(recs)
            else:
                skipped += 1
            if done % config.log_every_n_videos == 0:
                _log_progress(done, len(records), len(all_records), skipped)
    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating %d workers, keeping %d frames",
                       workers, len(all_records))
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()

    return all_records, skipped


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


def write_all_csvs(
    all_records: List[Dict],
    csv_dir: Path,
    config: PreprocessConfig,
) -> None:
    """Write master.csv plus per-dataset, per-split CSVs."""
    write_csv(all_records, csv_dir / "master.csv")

    dataset_key = {
        config.ff_dataset_name: "ff",
        config.siw_dataset_name: "siw",
    }

    for dataset_name, prefix in dataset_key.items():
        for split in ("train", "val", "test"):
            subset = [
                r for r in all_records
                if r["dataset"] == dataset_name and r["split"] == split
            ]
            if subset:
                write_csv(subset, csv_dir / f"{prefix}_{split}.csv")
            else:
                logger.warning(
                    "No rows for %s/%s — training will fail if this split is used",
                    dataset_name, split,
                )


# =========================================================================== #
# Summary
# =========================================================================== #
def log_summary(all_records: List[Dict]) -> None:
    unique_clips = len({(r["video_index"], r["clip_index"]) for r in all_records})
    unique_subjects = len({r["subject_id"] for r in all_records})

    logger.info("── Summary ──────────────────────────────────")
    logger.info("  Total frames   : %d", len(all_records))
    logger.info("  Total clips    : %d", unique_clips)
    logger.info("  Total subjects : %d", unique_subjects)
    logger.info("  By split       : %s", dict(Counter(r["split"] for r in all_records)))
    logger.info("  By task        : %s", dict(Counter(r["task"] for r in all_records)))
    logger.info("  By label       : %s", dict(Counter(r["label"] for r in all_records)))
    logger.info("  By spoof type  : %s", dict(Counter(
        r["spoof_type"] for r in all_records if r["spoof_type"] not in ("none",)
    )))
    logger.info("─────────────────────────────────────────────")


# =========================================================================== #
# Pipeline entry
# =========================================================================== #
def run_pipeline(cfg: Config) -> None:
    config = cfg.preprocess
    set_seed(config.split_seed)

    span = clip_span(config)
    log_banner(logger, "Preprocessing pipeline")
    logger.info(
        "Clips: %d frames | frame_skip=%d → span %d source frames | margin=%d%s",
        config.frames_per_clip, config.frame_skip, span, config.margin,
        " (adaptive)" if config.adaptive_margin else "",
    )
    logger.info(
        "Stride (start-to-start): ff_real=%d ff_fake=%d siw_live=%d siw_spoof=%d"
        " | max_clips/video=%s",
        config.clip_stride_real, config.clip_stride_fake,
        config.clip_stride_live, config.clip_stride_spoof,
        config.max_clips_per_video or "unlimited",
    )
    # Overlap is the part of the stride story that silently changes the data:
    # a stride below the clip span produces clips that share frames.
    overlaps = {
        "ff_real": span - config.clip_stride_real,
        "ff_fake": span - config.clip_stride_fake,
        "siw_live": span - config.clip_stride_live,
        "siw_spoof": span - config.clip_stride_spoof,
    }
    logger.info(
        "Clip overlap (shared frames): %s",
        " ".join(f"{k}={max(0, v)}" for k, v in overlaps.items()),
    )
    logger.info(
        "Quality gate: score>=%.2f | window IoU>=%.2f | min_valid=%d",
        config.min_face_score, config.disp_ratio, config.min_valid_frames,
    )

    records: List[VideoMeta] = []
    records.extend(scan_ff(config))
    records.extend(scan_siw(config))

    if not records:
        logger.error("No videos found under: %s", config.raw_data_root)
        return

    logger.info("Total videos: %d", len(records))
    records = split_by_subject(records, config)

    # Power monitoring — same PowerMonitor the trainer uses, so the thesis can
    # report preprocessing and training energy on a comparable basis.
    monitor = None
    if config.monitor_power and cfg.power.enable:
        csv_path = str(config.processed_root / config.power_log_csv)
        monitor = power_monitor_from_config(cfg, csv_path=csv_path).start()
        logger.info("Power monitoring on → %s", csv_path)

    workers = auto_worker_count(config)
    log_path = str(config.processed_root / "preprocessing.log")

    all_frame_records: List[Dict] = []
    skipped = 0
    try:
        if workers > 1:
            logger.info("Extracting with %d worker processes (spawn)", workers)
            all_frame_records, skipped = extract_parallel(
                records, config, workers, log_path
            )
        else:
            logger.info("Extracting sequentially (single process)")
            all_frame_records, skipped = extract_sequential(records, config)
    finally:
        if monitor is not None:
            monitor.stop()
            monitor.log_summary(logger)

    logger.info("Videos skipped (no valid clips): %d", skipped)

    if not all_frame_records:
        logger.error("No frames extracted.")
        return

    log_summary(all_frame_records)
    write_all_csvs(all_frame_records, config.csv_dir, config)
    log_banner(logger, "Done")


def main() -> None:
    cfg = get_config()
    cfg.preprocess.processed_root.mkdir(parents=True, exist_ok=True)
    setup_logger(
        log_path=str(cfg.preprocess.processed_root / "preprocessing.log"),
        cfg=cfg,
        name="MTL",
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
