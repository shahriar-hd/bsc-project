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
    │   │   │   ├── ...
    │   │   │   └── c0_flow.npz     ← precomputed optical flow for clip 0
    │   │   └── sub_000001/
    │   └── fake/
    └── SiW-Mv2/
        ├── real/
        └── spoof/

Optical flow is produced here and nowhere else (see src/utils/flow_utils.py).
train.py only reads these files, and only when TrainConfig.use_optical_flow is
on — Farneback is far too slow to run inside a dataloader worker.

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
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.config import Config, PreprocessConfig, get_config
from src.utils.face_utils import FaceDetector
from src.utils.flow_utils import (
    compute_clip_flow,
    flow_path_for_clip,
    save_clip_flow,
)
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
    max_clips: int = 0,
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
    if total_frames < span:
        return []

    margin = effective_margin(total_frames, config)

    usable_start = margin
    usable_end = total_frames - margin
    if usable_end - usable_start < span:
        if config.min_clips_per_video > 0:
            return [max(0, (total_frames - span) // 2)]
        return []

    starts: List[int] = []
    s = usable_start
    while s + span <= usable_end:
        starts.append(s)
        s += stride

    if not starts and config.min_clips_per_video > 0:
        starts.append(max(0, (total_frames - span) // 2))

    cap = max_clips if max_clips > 0 else config.max_clips_per_video
    if cap > 0:
        starts = starts[:cap]
    return starts


def stride_for_label(dataset: str, label: str, config: PreprocessConfig) -> int:
    """
    Map (dataset, label) to its clip stride.

    FF++ real and SiW-Mv2 live are both labelled "real" but are very different
    videos — 840 vs 179 median frames — so they cannot share a stride if the
    two tasks are to stay class-balanced.
    If allow_clip_overlap is False, stride is clamped to at least clip_span.
    """
    if dataset == config.siw_dataset_name:
        stride = (
            config.clip_stride_live if label == "real"
            else config.clip_stride_spoof
        )
    else:
        stride = (
            config.clip_stride_fake if label == "fake"
            else config.clip_stride_real
        )
    if not config.allow_clip_overlap:
        stride = max(stride, clip_span(config))
    return stride


def max_clips_for_label(dataset: str, label: str, config: PreprocessConfig) -> int:
    """
    Get max clips allowed per video for a specific dataset and label.
    Falls back to config.max_clips_per_video if specific setting is None.
    """
    if dataset == config.siw_dataset_name:
        specific = (
            config.max_clips_siw_live if label == "real"
            else config.max_clips_siw_spoof
        )
    else:
        specific = (
            config.max_clips_ff_fake if label == "fake"
            else config.max_clips_ff_real
        )
    return specific if specific is not None else config.max_clips_per_video


def starts_for_stride(total_frames: int, stride: int, config: PreprocessConfig) -> int:
    """How many clip starts `compute_clip_starts` would find, ignoring caps."""
    if stride <= 0:
        return 0
    span = clip_span(config)
    if total_frames < span:
        return 0
    margin = effective_margin(total_frames, config)
    window = (total_frames - margin) - margin - span
    if window < 0:
        return 1 if config.min_clips_per_video > 0 else 0
    return window // stride + 1


def stride_for_clip_count(
    total_frames: int, wanted: int, config: PreprocessConfig
) -> int:
    """
    Smallest stride ≥ `min_spoof_stride` that yields `wanted` clips from a video
    of `total_frames`, spread as widely as the usable region allows.

    Returns `clip_span` when one clip is enough, so the common case keeps a
    non-overlapping stride rather than an arbitrarily small one.
    """
    span = clip_span(config)
    if wanted <= 1:
        return span
    margin = effective_margin(total_frames, config)
    window = (total_frames - margin) - margin - span
    if window <= 0:
        return span
    # `wanted` starts spread over `window` frames need this gap between them;
    # floor keeps the count at or above `wanted`.
    stride = window // (wanted - 1)
    return max(config.min_spoof_stride, min(stride, span))


def plan_spoof_sampling(
    records: List[VideoMeta],
    config: PreprocessConfig,
) -> Dict[str, Dict[str, int]]:
    """
    Choose a stride and a per-video clip cap for each SiW-Mv2 attack type, and
    write them onto the records so the workers need no extra state.

    Rare attack types get more clips from each of their videos, up to
    ``max_clips_per_video_spoof`` and never below ``min_spoof_stride``; common
    types get one clip per video. No video is ever dropped — see
    ``PreprocessConfig.target_clips_per_spoof_type`` for why balance is narrowed
    rather than forced.

    Reads frame counts for the spoof videos (~2 s for 915 files, metadata only).
    Returns a per-type report for logging.
    """
    target = config.target_clips_per_spoof_type
    spoof_records = [
        r for r in records
        if str(r["dataset"]) == config.siw_dataset_name and str(r["label"]) == "spoof"
    ]
    if target is None or target <= 0 or not spoof_records:
        return {}

    by_type: Dict[str, List[VideoMeta]] = defaultdict(list)
    for rec in spoof_records:
        by_type[str(rec["spoof_type"])].append(rec)

    report: Dict[str, Dict[str, int]] = {}
    for spoof_type, recs in sorted(by_type.items()):
        lengths = {}
        for rec in recs:
            n = get_frame_count(Path(str(rec["video_path"])))
            if n > 0:
                lengths[str(rec["video_path"])] = n
        if not lengths:
            logger.warning("No readable videos for spoof type %s", spoof_type)
            continue

        n_videos = len(lengths)
        # Clips per video needed to reach the target, capped so overlap stays sane.
        wanted = max(1, min(config.max_clips_per_video_spoof,
                            -(-target // n_videos)))          # ceil division
        median_len = sorted(lengths.values())[n_videos // 2]
        stride = stride_for_clip_count(median_len, wanted, config)

        produced = 0
        for rec in recs:
            n = lengths.get(str(rec["video_path"]), 0)
            rec["clip_stride"] = stride
            rec["max_clips"] = wanted
            produced += min(wanted, starts_for_stride(n, stride, config)) if n else 0

        report[spoof_type] = {
            "videos": n_videos,
            "median_frames": median_len,
            "stride": stride,
            "clips_per_video": wanted,
            "clips": produced,
        }

    if report:
        counts = [r["clips"] for r in report.values()]
        span = clip_span(config)
        logger.info(
            "Per-attack-type sampling (target %d clips/type, cap %d clips/video, "
            "min stride %d):", target, config.max_clips_per_video_spoof,
            config.min_spoof_stride,
        )
        for spoof_type, r in sorted(report.items(), key=lambda kv: -kv[1]["clips"]):
            overlap = max(0, span - r["stride"])
            logger.info(
                "  %-26s %3d videos | median %3df | stride %3d (%2df overlap) | "
                "%d clip/video -> %4d clips",
                spoof_type, r["videos"], r["median_frames"], r["stride"], overlap,
                r["clips_per_video"], r["clips"],
            )
        logger.info(
            "  spread %d..%d clips = %.1fx (was 10.5x by video count); the "
            "remainder is left to per-type sampling weights at training time",
            min(counts), max(counts), max(counts) / max(min(counts), 1),
        )
    return report


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

    valid_crops: List[Tuple[int, int, np.ndarray]] = []
    saved_idx = 0

    for (src_idx, bgr), bbox in zip(raw_frames, per_frame_bbox):
        if bbox is None:
            continue
        if not face_inside_window(bbox, window, iou_threshold=config.disp_ratio):
            continue

        crop = detector.crop_stable(bgr, window)
        if crop.size == 0:
            continue

        valid_crops.append((src_idx, saved_idx, crop))
        saved_idx += 1

    if len(valid_crops) < config.min_valid_frames:
        logger.debug(
            "Clip %d: only %d valid frames after window filtering → reject",
            clip_idx, len(valid_crops),
        )
        return []

    # Quality gate passed: create output directory and write files to disk
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []

    for src_idx, s_idx, crop in valid_crops:
        save_path = out_dir / f"c{clip_idx}_f{s_idx:03d}.jpg"
        cv2.imwrite(str(save_path), crop, encode_params)

        records.append({
            "frame_path": str(save_path),
            "clip_index": clip_idx,
            "frame_num": s_idx,
            "src_frame_idx": src_idx,
            "clip_start_frame": start_frame,
        })

    # Optical flow, computed here and only here. Pair k is the flow from saved
    # frame k to k+1, so it is indexed by `frame_num` and stays aligned however
    # the training loader later samples or jitters the clip.
    if config.precompute_optical_flow:
        try:
            flow = compute_clip_flow(
                [c for _, _, c in valid_crops],
                resize=config.flow_resize,
                method=config.optical_flow_method,
            )
            save_clip_flow(
                flow_path_for_clip(records[0]["frame_path"], clip_idx), flow
            )
        except Exception as exc:                       # noqa: BLE001
            # A missing flow file degrades to a warning at training time, so
            # losing one clip's flow must not cost us its frames.
            logger.warning("Flow failed for clip %d of %s: %s: %s",
                           clip_idx, video_path.name, type(exc).__name__, exc)

    return records
# =========================================================================== #
# Dataset scanners
# =========================================================================== #
def ff_identity_tokens(stem: str) -> List[str]:
    """
    Identity tokens encoded in a FaceForensics++ DFD filename.

    real  ``NN__scene``              -> ["NN"]        one actor
    fake  ``NN_MM__scene__HASH``     -> ["NN", "MM"]  actor NN's face on MM's video

    The identity block is everything before the first ``__``; within it actors
    are ``_``-separated. Anything that does not parse yields ``[]``, and the
    caller falls back to a per-video subject so an unrecognised name is never
    silently merged into someone else's identity.
    """
    head = stem.split("__", 1)[0]
    tokens = [t for t in head.split("_") if t]
    return tokens if all(t.isdigit() for t in tokens) and tokens else []


def build_ff_identity_map(
    video_paths: Iterable[Path],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Group FF++ videos into identity components via union-find.

    A fake video names *two* actors, which ties those identities together: if
    they landed in different splits, that fake's face would appear in one split
    and its driving video in another. Treating each identity token as a node and
    each fake as an edge, a connected component is the smallest unit that can be
    assigned to a split without leaking.

    Returns ``(path_str -> subject_id, stats)``. ``stats`` carries
    ``n_videos``/``n_identities``/``n_components``/``largest_component_videos``
    /``n_unparsed`` so the caller can report how well this actually separated —
    if the actor pairing is dense enough, a single component can swallow most of
    the corpus and no identity-disjoint split exists.
    """
    paths = list(video_paths)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tokens_per_path: Dict[str, List[str]] = {}
    for vid in paths:
        tokens = ff_identity_tokens(vid.stem)
        tokens_per_path[str(vid)] = tokens
        for tok in tokens:
            find(tok)
        for tok in tokens[1:]:
            union(tokens[0], tok)          # a fake ties its actors together

    # Number components by their smallest identity token so ids are stable
    # across runs and independent of filesystem ordering.
    members: Dict[str, List[str]] = defaultdict(list)
    for tok in parent:
        members[find(tok)].append(tok)
    roots = sorted(members, key=lambda r: min(members[r]))
    comp_of_root = {root: i for i, root in enumerate(roots)}

    subject_of_path: Dict[str, str] = {}
    videos_per_component: Counter = Counter()
    n_unparsed = 0
    for path_str, tokens in tokens_per_path.items():
        if not tokens:
            n_unparsed += 1
            subject_of_path[path_str] = f"ff_vid_{Path(path_str).stem}"
            continue
        comp = comp_of_root[find(tokens[0])]
        subject_of_path[path_str] = f"ff_id{comp:03d}"
        videos_per_component[comp] += 1

    stats = {
        "n_videos": len(paths),
        "n_identities": len(parent),
        "n_components": len(roots),
        "largest_component_videos": max(videos_per_component.values(),
                                        default=0),
        "n_unparsed": n_unparsed,
    }
    return subject_of_path, stats


def ff_scene_token(stem: str) -> str:
    """
    The scripted-scenario token of a FaceForensics++ DFD filename.

    real  ``NN__scene``           -> "scene"
    fake  ``NN_MM__scene__HASH``  -> "scene"

    Returns ``""`` when the name has no ``__`` separator, so the caller can fall
    back to a per-video subject rather than lumping unparsed names together.
    """
    parts = stem.split("__")
    return parts[1] if len(parts) > 1 else ""


def build_ff_subject_map(
    video_paths: Iterable[Path],
    split_key: str,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """
    Choose the unit that must not straddle two splits, and report what still does.

    ``split_key`` is ``PreprocessConfig.ff_split_key`` — see that field for the
    measured trade-off between "scene", "identity" and "video". Whichever is
    chosen, ``stats`` carries the *residual* overlap of the other unit so the
    summary can state it instead of implying the split is leak-free.
    """
    paths = list(video_paths)
    if split_key not in ("scene", "identity", "video"):
        raise ValueError(
            f"ff_split_key must be 'scene', 'identity' or 'video', got {split_key!r}"
        )

    ident_map, ident_stats = build_ff_identity_map(paths)

    if split_key == "identity":
        subject_of_path = ident_map
    else:
        subject_of_path = {}
        for vid in paths:
            if split_key == "video":
                subject_of_path[str(vid)] = f"ff_vid_{vid.stem}"
                continue
            scene = ff_scene_token(vid.stem)
            subject_of_path[str(vid)] = (
                f"ff_scene_{scene}" if scene else f"ff_vid_{vid.stem}"
            )

    scenes = {ff_scene_token(v.stem) for v in paths} - {""}
    subjects = set(subject_of_path.values())
    sizes = Counter(subject_of_path.values())
    stats: Dict[str, object] = {
        "split_key": split_key,
        "n_videos": len(paths),
        "n_subjects": len(subjects),
        "largest_subject_videos": max(sizes.values(), default=0),
        "n_scenes": len(scenes),
        "n_identities": ident_stats["n_identities"],
        "n_identity_components": ident_stats["n_components"],
        "largest_identity_component": ident_stats["largest_component_videos"],
        "n_unparsed": ident_stats["n_unparsed"],
    }
    return subject_of_path, stats


def scan_ff(config: PreprocessConfig) -> List[VideoMeta]:
    root = config.raw_data_root / config.ff_dataset_name

    # Collect first, then resolve subjects: a fake's identity component depends on
    # the other videos, so it cannot be decided one file at a time.
    found: List[Tuple[Path, str]] = []
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
                found.append((vid, label_str))

    subject_of_path, stats = build_ff_subject_map(
        (v for v, _ in found), config.ff_split_key
    )
    logger.info(
        "FF++ split unit '%s': %d videos -> %d subjects "
        "(largest holds %d videos, %.1f%%)",
        stats["split_key"], stats["n_videos"], stats["n_subjects"],
        stats["largest_subject_videos"],
        100.0 * int(stats["largest_subject_videos"]) / max(int(stats["n_videos"]), 1),
    )
    logger.info(
        "FF++ structure: %d scenarios | %d actor tokens | %d identity components",
        stats["n_scenes"], stats["n_identities"], stats["n_identity_components"],
    )
    if config.ff_split_key == "scene":
        logger.info(
            "FF++ residual: actors are NOT split-disjoint under 'scene' — actor "
            "overlap between splits is expected and is the accepted trade-off "
            "(an identity-disjoint split does not exist: %d components, largest "
            "holds %d of %d videos)",
            stats["n_identity_components"], stats["largest_identity_component"],
            stats["n_videos"],
        )
    elif config.ff_split_key == "identity":
        logger.warning(
            "FF++ residual: all %d fakes reuse a scenario that also appears as a "
            "real video, and 'identity' does not separate scenarios — source-"
            "content leakage stays at 100%%",
            stats["n_videos"] // 2,
        )
    else:
        logger.warning(
            "FF++ ff_split_key='video': neither actors nor scenarios are split-"
            "disjoint. Test metrics will be optimistic — this is the run01 setting"
        )
    if stats["n_unparsed"]:
        logger.warning(
            "%d FF++ filenames did not match the DFD pattern and fall back to "
            "per-video subjects (leakage possible for those)",
            stats["n_unparsed"],
        )

    records: List[VideoMeta] = [
        {
            "video_path": str(vid),
            "dataset": config.ff_dataset_name,
            "subject_id": subject_of_path[str(vid)],
            "label": label_str,
            "task": "deepfake",
            "spoof_type": "none",
        }
        for vid, label_str in found
    ]
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

    What a "subject" is differs by dataset, and only one of them is genuinely
    identity-aware:

    * **FaceForensics++** — ``PreprocessConfig.ff_split_key`` (default "scene").
      The DFD actor subset cannot be split identity-disjointly at all; see that
      field for the measured numbers.
    * **SiW-Mv2** — one subject **per video**. The filenames carry no token that
      links the same person across attack types, and no official protocol file
      ships with the copy here, so there is nothing to group on. This split is
      per-video and is *not* subject-aware; calling it so would be false.

    Assignment is greedy rather than a slice of a shuffled list. Slicing at
    ``int(n * ratio)`` balances *subject counts*, but subjects hold different
    numbers of videos and different real/fake mixes — run01's ff_val came out
    58/42 against a 49/51 corpus. Here each subject goes to whichever split is
    currently least full **relative to its quota, per label**, so both the size
    ratios and the label ratios are tracked. Subjects are visited largest-first
    (a shuffled order within equal size keeps ``split_seed`` meaningful), since
    placing the big ones while all splits are still empty is what keeps the tail
    able to correct the balance.
    """
    by_dataset: Dict[str, List[VideoMeta]] = defaultdict(list)
    for rec in records:
        by_dataset[str(rec["dataset"])].append(rec)

    test_ratio = max(0.0, 1.0 - config.train_ratio - config.val_ratio)
    targets = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": test_ratio,
    }
    targets = {k: v for k, v in targets.items() if v > 0.0}

    for dataset_name, ds_records in by_dataset.items():
        subjects: Dict[str, List[VideoMeta]] = defaultdict(list)
        for rec in ds_records:
            subjects[str(rec["subject_id"])].append(rec)

        # Per-subject video count per label, and the corpus totals to scale against.
        per_subject_labels: Dict[str, Counter] = {
            sid: Counter(str(r["label"]) for r in recs)
            for sid, recs in subjects.items()
        }
        totals = Counter()
        for counts in per_subject_labels.values():
            totals.update(counts)

        ids = sorted(subjects.keys())
        rng = random.Random(config.split_seed)
        rng.shuffle(ids)
        # Largest first; the shuffle above breaks ties deterministically.
        ids.sort(key=lambda sid: -len(subjects[sid]))

        filled: Dict[str, Counter] = {s: Counter() for s in targets}
        split_of: Dict[str, str] = {}
        for sid in ids:
            counts = per_subject_labels[sid]
            best_split, best_cost = None, None
            for split, ratio in targets.items():
                # Fullness after adding, per label: >1 means past quota. The worst
                # label drives the choice, so a split cannot be balanced overall
                # while being lopsided on one class.
                cost = max(
                    (filled[split][lab] + counts[lab]) / max(ratio * totals[lab], 1e-9)
                    for lab in totals
                )
                if best_cost is None or cost < best_cost:
                    best_split, best_cost = split, cost
            split_of[sid] = str(best_split)
            filled[str(best_split)].update(counts)

        for sid, recs in subjects.items():
            for rec in recs:
                rec["split"] = split_of[sid]

        counts = Counter(str(r["split"]) for r in ds_records)
        n_subjects = len(subjects)
        logger.info(
            "%-16s split (videos) — train: %d | val: %d | test: %d | subjects: %d",
            dataset_name, counts["train"], counts["val"], counts["test"], n_subjects,
        )
        for split in targets:
            lab = filled[split]
            total = sum(lab.values())
            if total == 0:
                logger.warning(
                    "%s %s split is EMPTY — every metric guarded on class count "
                    "will return nothing for it", dataset_name, split,
                )
                continue
            mix = "  ".join(f"{k}={v} ({100.0 * v / total:.0f}%)"
                            for k, v in sorted(lab.items()))
            logger.info("%-16s   %-5s labels: %s", dataset_name, split, mix)
            if len(totals) > 1 and min(lab.get(k, 0) for k in totals) == 0:
                logger.warning(
                    "%s %s split is single-class — metrics for it will be empty",
                    dataset_name, split,
                )

        # FF++ under an identity/scene key must collapse many videos into few
        # subjects; equality means the grouping silently did nothing.
        if dataset_name == config.ff_dataset_name:
            n_videos = len(ds_records)
            if config.ff_split_key != "video" and n_subjects >= n_videos:
                logger.warning(
                    "FF++ has %d subjects for %d videos under ff_split_key=%r — "
                    "the grouping did not take effect and both actors and "
                    "scenarios leak across splits",
                    n_subjects, n_videos, config.ff_split_key,
                )
            else:
                logger.info(
                    "FF++ grouping check: %d videos -> %d subjects (%.1f videos "
                    "per subject)", n_videos, n_subjects, n_videos / max(n_subjects, 1),
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

    # A per-attack-type plan, when one was made, overrides the flat per-label
    # stride. Carried on the record itself so workers stay stateless.
    stride = int(meta.get("clip_stride") or stride_for_label(dataset, label, config))
    max_clips = int(meta.get("max_clips") or max_clips_for_label(dataset, label, config))
    clip_starts = compute_clip_starts(total_frames, stride, config, max_clips=max_clips)

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
def log_summary(all_records: List[Dict], config: PreprocessConfig) -> None:
    """
    Report the corpus, then check the two properties that silently failed before.

    Counting frames is not enough. `subject_id` used to be `ff_<stem>`, one
    subject per video, which made every FF++ split subject-disjoint by definition
    and held nothing back — `Unique subjects == Unique videos == 1938` was the
    only visible symptom and it read like a coincidence. And the split cut a
    shuffled list at `int(n * ratio)`, which put ff_val at 58/42 real/fake against
    ff_train's 49/51. Both are now printed per dataset and warned about.
    """
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

    # ── grouping: did subject_id actually collapse videos together? ──────────
    logger.info("  Subject grouping (videos per subject — 1.0 means no grouping):")
    for dataset in sorted({str(r["dataset"]) for r in all_records}):
        rows = [r for r in all_records if str(r["dataset"]) == dataset]
        n_vid = len({r["video_index"] for r in rows})
        n_sub = len({r["subject_id"] for r in rows})
        logger.info(
            "    %-18s %4d videos / %4d subjects = %.2f videos per subject",
            dataset, n_vid, n_sub, n_vid / max(n_sub, 1),
        )
        if dataset == config.ff_dataset_name and config.ff_split_key != "video":
            if n_sub >= n_vid:
                logger.warning(
                    "  FF++ has one subject per video (ff_split_key=%r) — the "
                    "splits share content and every metric is optimistic. "
                    "Run scripts/audit_ff_split.py.",
                    config.ff_split_key,
                )
            else:
                logger.info(
                    "    FF++ grouped %d videos into %d subjects by %r",
                    n_vid, n_sub, config.ff_split_key,
                )

    # ── split balance: each split's label mix against the corpus mix ─────────
    logger.info("  Label balance per split (clip counts):")
    for dataset in sorted({str(r["dataset"]) for r in all_records}):
        rows = [r for r in all_records if str(r["dataset"]) == dataset]
        clips_by = defaultdict(set)
        for r in rows:
            clips_by[(str(r["split"]), str(r["label"]))].add(
                (r["video_index"], r["clip_index"])
            )
        labels = sorted({lab for _, lab in clips_by})
        overall = {
            lab: sum(len(v) for (_, l), v in clips_by.items() if l == lab)
            for lab in labels
        }
        total = max(sum(overall.values()), 1)
        logger.info(
            "    %-18s overall %s",
            dataset,
            "  ".join(f"{lab}={overall[lab]} ({100*overall[lab]/total:.0f}%)"
                      for lab in labels),
        )
        for split in ("train", "val", "test"):
            counts = {lab: len(clips_by.get((split, lab), ())) for lab in labels}
            n = sum(counts.values())
            if n == 0:
                continue
            drift = max(
                abs(counts[lab] / n - overall[lab] / total) for lab in labels
            )
            line = "  ".join(f"{lab}={counts[lab]} ({100*counts[lab]/n:.0f}%)"
                             for lab in labels)
            logger.info("      %-5s %5d clips  %s", split, n, line)
            if min(counts.values()) == 0:
                logger.warning(
                    "  %s/%s is single-class — every guarded metric will return "
                    "nothing for it", dataset, split,
                )
            elif drift > 0.05:
                logger.warning(
                    "  %s/%s label mix is %.0f pp off the corpus mix — check "
                    "split_by_subject", dataset, split, 100 * drift,
                )
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
        "Sampling: allow_overlap=%s | min_clips/vid=%d | max_clips/vid=%s"
        " (ff_real=%s, ff_fake=%s, siw_live=%s, siw_spoof=%s)",
        config.allow_clip_overlap,
        config.min_clips_per_video,
        config.max_clips_per_video or "unlimited",
        config.max_clips_ff_real if config.max_clips_ff_real is not None else "default",
        config.max_clips_ff_fake if config.max_clips_ff_fake is not None else "default",
        config.max_clips_siw_live if config.max_clips_siw_live is not None else "default",
        config.max_clips_siw_spoof if config.max_clips_siw_spoof is not None else "default",
    )
    strides = {
        "ff_real": stride_for_label(config.ff_dataset_name, "real", config),
        "ff_fake": stride_for_label(config.ff_dataset_name, "fake", config),
        "siw_live": stride_for_label(config.siw_dataset_name, "real", config),
        "siw_spoof": stride_for_label(config.siw_dataset_name, "spoof", config),
    }
    logger.info(
        "Effective strides: %s",
        " ".join(f"{k}={v}" for k, v in strides.items()),
    )
    overlaps = {k: max(0, span - v) for k, v in strides.items()}
    logger.info(
        "Clip overlap (shared frames): %s",
        " ".join(f"{k}={v}" for k, v in overlaps.items()),
    )
    logger.info(
        "Quality gate: score>=%.2f | window IoU>=%.2f | min_valid=%d",
        config.min_face_score, config.disp_ratio, config.min_valid_frames,
    )
    if config.precompute_optical_flow:
        # The array is (N-1, 2, R, R) int8, but it is written with
        # savez_compressed — measured ~3.1x on the previous full run (385.9 KB
        # raw -> ~126 KB on disk). Report the compressed figure, since that is
        # the one a disk budget needs; the raw size would over-state it 3x.
        raw_kb = (config.frames_per_clip - 1) * 2 * config.flow_resize ** 2 / 1024
        logger.info(
            "Optical flow: %s at %dpx → c<clip>_flow.npz beside the crops "
            "(int8 %.0f KB raw, ~%.0f KB on disk after zip). Training reads "
            "these; it never computes flow.",
            config.optical_flow_method, config.flow_resize, raw_kb, raw_kb / 3.1,
        )
    else:
        logger.info("Optical flow precompute: OFF "
                    "(TrainConfig.use_optical_flow will have nothing to read)")

    records: List[VideoMeta] = []
    records.extend(scan_ff(config))
    records.extend(scan_siw(config))

    if not records:
        logger.error("No videos found under: %s", config.raw_data_root)
        return

    logger.info("Total videos: %d", len(records))
    plan_spoof_sampling(records, config)
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

    log_summary(all_frame_records, config)
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
