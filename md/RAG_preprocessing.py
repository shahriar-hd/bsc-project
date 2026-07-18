# =========================================================================== #
# Constants
# =========================================================================== #
FRAMES_PER_CLIP: int = 64
MAX_CLIPS_FF: int = 3
MAX_CLIPS_SIW: int = 1
MIN_FACE_SCORE: float = 0.65
MIN_VALID_FRAMES: int = 48           # 75 % of FRAMES_PER_CLIP
MARGIN: int = 25                     # frames to skip at start/end of video
DISP_RATIO: float = 0.30             # kept for log message only



# =========================================================================== #
# InsightFace — lightweight detector (buffalo_sc)
# =========================================================================== #
class FaceDetector:
    """
    Uses InsightFace buffalo_sc (lightweight) for per-frame face detection.

    buffalo_sc is significantly faster than buffalo_l and sufficient
    for the quality-gate / displacement checks done here.
    """

    def __init__(
        self,
        config: PreprocessConfig,
        min_face_score: float = MIN_FACE_SCORE,
    ) -> None:
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

    usable_end = total_frames - MARGIN

    # clip-0: early
    s0 = MARGIN

    # clip-1: middle (centred)
    mid = total_frames // 2
    s1 = max(s0, mid - FRAMES_PER_CLIP // 2)
    s1 = min(s1, usable_end - FRAMES_PER_CLIP)

    # clip-2: late
    s2 = usable_end - FRAMES_PER_CLIP
    s2 = max(s1 + FRAMES_PER_CLIP, s2)

    starts = [s0]
    if s1 > s0:
        starts.append(s1)
    if s2 > (starts[-1] + FRAMES_PER_CLIP - 1) and s2 != s0:
        starts.append(s2)

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
# Displacement-aware clip builder
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

    Args:
        video_path:  Source video file.
        start_frame: Index of the first frame of this clip in the video.
        detector:    Initialised FaceDetector.
        out_dir:     Subject-level output directory.
        clip_idx:    Clip index within this video (0-based).
        config:      PreprocessConfig instance (for jpeg_quality etc.).

    Returns:
        List of frame-level record dicts; empty list if clip is rejected.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]

    # ── Pass 1: read all frames, detect faces, collect bboxes ────────────── #
    raw_frames: List[Tuple[int, np.ndarray]] = []
    per_frame_bbox: List[Optional[np.ndarray]] = []

    for src_idx, bgr in read_consecutive_frames(video_path, start_frame, FRAMES_PER_CLIP):
        raw_frames.append((src_idx, bgr))
        face = detector.detect_best(bgr)
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
    saved_idx: int = 0

    for (src_idx, bgr), bbox in zip(raw_frames, per_frame_bbox):
        if bbox is None:
            logger.debug("Clip %d frame %d: no detection → skip", clip_idx, src_idx)
            continue

        if not _face_inside_window(bbox, crop_coords):
            logger.debug("Clip %d frame %d: face outside window → skip", clip_idx, src_idx)
            continue

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



# =========================================================================== #
# Subject-aware split
# =========================================================================== #
def split_by_subject(
    records: List[VideoMeta],
    config: PreprocessConfig,
) -> List[VideoMeta]:
    subjects: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        subjects[str(rec["subject_id"])].append(i)

    ids = list(subjects.keys())
    rng = random.Random(config.split_seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * config.train_ratio)
    n_val   = int(n * config.val_ratio)

    train_set = set(ids[:n_train])
    val_set   = set(ids[n_train: n_train + n_val])

    for rec in records:
        sid = str(rec["subject_id"])
        rec["split"] = (
            "train" if sid in train_set else
            "val"   if sid in val_set   else
            "test"
        )

    counts: Counter = Counter(str(r["split"]) for r in records)
    logger.info(
        "Split (videos) — train: %d | val: %d | test: %d",
        counts["train"], counts["val"], counts["test"],
    )
    return records


# =========================================================================== #
# Pipeline entry
# =========================================================================== #
def run_pipeline(config: PreprocessConfig) -> None:
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
