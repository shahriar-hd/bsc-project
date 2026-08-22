"""
app_demo.py — CLI face authentication demo.
Uses InsightFace buffalo for embedding/face alignment and a trained MTL model
for deepfake / anti-spoof / temporal detection.

Frame processing pipeline (anti-jitter):
  Pass 1 – buffalo detects face bounding boxes frame-by-frame.
  Jitter check – if center-of-box moves more than JITTER_THR × face_size,
                 the session is rejected.
  Aggregate box – union of all per-frame boxes (min x1/y1, max x2/y2)
                  plus MARGIN padding, clamped to image boundaries.
  Pass 2 – every frame is cropped with that single stable box and saved.
  Embeddings are taken from Pass 1 (no second detection on crops).

Menu option `b` is an offline **evaluation** of the trained checkpoint over the
held-out test split, driven by `data/datasets/processed/csv/master.csv` rather
than by a directory of loose files. It walks the raw videos the CSV points at,
pushes every `split == test` clip through this same detect → stable-crop →
MTL path, and reports the full metric set for all three heads (plus latency and
the end-to-end verdict the app itself would return). See `test_batch`.
"""

from __future__ import annotations

import csv as csvmod
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torchvision import transforms

from insightface.app import FaceAnalysis

from src.config import get_config
# The metric implementations are imported, never re-written: a demo that computed
# AUC/ACER its own way could disagree with results.csv and there would be no way
# to tell which of the two was wrong.
from src.train import (
    MTLModel,
    binary_classification_metrics,
    collapse_warnings,
    compute_deepfake_metrics,
    compute_spoof_metrics,
    compute_temporal_metrics,
)
# Same window geometry and same frame reader preprocessing used, so an evaluation
# clip is the clip the model was trained on rather than a lookalike.
from src.preprocessing import compute_average_crop_coords, face_inside_window
from src.utils.face_utils import FaceDetector, bbox_area, union_bbox
from src.utils.flow_utils import accumulate_flow, compute_clip_flow
from src.utils.power_utils import power_monitor_from_config
from src.utils.video_utils import read_consecutive_frames

# ── globals ───────────────────────────────────────────────────────────────────

_full_cfg = get_config()
cfg = _full_cfg.demo

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Pipeline constants
MARGIN     = 0.15   # extra padding around aggregate box (15 %)
JITTER_THR = 0.30   # max allowed center-shift relative to face size per consecutive frame
MIN_FACES  = 8      # minimum detected faces to proceed

# ── storage helpers ───────────────────────────────────────────────────────────

def load_instance() -> dict:
    p = Path(cfg.paths.instance_file)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_instance(db: dict) -> None:
    p = Path(cfg.paths.instance_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(db, indent=2))


def save_run_metadata(run_dir: Path, data: dict) -> None:
    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(data, indent=2))
    print(f"  [debug] Metadata saved to {meta_path}")


# ── InsightFace ───────────────────────────────────────────────────────────────

def get_face_detector() -> FaceDetector:
    """
    Shared detector with the recognition module enabled — the demo needs
    `normed_embedding` for identity matching, unlike preprocessing.
    """
    return FaceDetector(
        _full_cfg.preprocess,
        need_embeddings=True,
        det_name=cfg.paths.buffalo_model,
    )


def get_face_app(detector: FaceDetector | None = None) -> FaceAnalysis:
    """Underlying FaceAnalysis session (kept: call sites use `face_app.get`).

    Takes an existing detector when there is one — an InsightFace session holds
    GPU memory, and the batch evaluator needs the `FaceDetector` wrapper (crop
    geometry, score gate) alongside the raw session the interactive flows use.
    """
    detector = detector or get_face_detector()
    detector._ensure_app()
    return detector.app


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ── shared crop geometry ──────────────────────────────────────────────────────
# Factored out of _process_raw_frames so the offline evaluator can reproduce the
# live camera path's geometry exactly instead of a second implementation of it.

def _jitter_shifts(boxes_arr: np.ndarray) -> tuple[float, float, int]:
    """Largest consecutive centre shift, normalised by the mean face size.

    Returns (max_cx_shift, max_cy_shift, index_of_worst_pair). Normalising by the
    mean box size is what makes JITTER_THR scale-invariant: the same head movement
    at half the distance covers half the pixels.
    """
    if len(boxes_arr) < 2:
        return 0.0, 0.0, -1

    mean_face_w = float((boxes_arr[:, 2] - boxes_arr[:, 0]).mean()) or 1.0
    mean_face_h = float((boxes_arr[:, 3] - boxes_arr[:, 1]).mean()) or 1.0

    cx = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
    cy = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
    dx = np.abs(np.diff(cx)) / mean_face_w
    dy = np.abs(np.diff(cy)) / mean_face_h

    worst = int(np.argmax(np.maximum(dx, dy)))
    return float(dx[worst]), float(dy[worst]), worst


def _demo_aggregate_window(
    boxes_arr: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Union of every per-frame box, padded by MARGIN and clamped to the frame.

    Example: boxes [200,160,600,500], [205,155,605,505] and [190,162,590,498]
    → union [190,155,605,505], then ±15 %.
    """
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = (int(v) for v in union_bbox(list(boxes_arr)))

    pad_x = int((x2 - x1) * MARGIN)
    pad_y = int((y2 - y1) * MARGIN)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(W, x2 + pad_x),
        min(H, y2 + pad_y),
    )


# ── core frame-processing logic ───────────────────────────────────────────────

def _process_raw_frames(
    raw_frames: list[np.ndarray],
    save_dir: Path,
    face_app: FaceAnalysis,
) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    """
    Two-pass pipeline that converts raw BGR frames into stable, jitter-free crops.

    Pass 1  – run buffalo on every frame; collect per-frame box + embedding.
    Check   – reject session if consecutive face-centre shift exceeds JITTER_THR.
    Aggregate box – union of all valid boxes, expanded by MARGIN.
    Pass 2  – crop every valid frame with the single aggregate box; save to disk.

    Returns
    -------
    saved_paths   : list of saved crop file paths
    cropped_faces : list of RGB np.ndarray (224×224) for MTL input
    embeddings    : list of normed_embedding from Pass 1 (no second detection)
    """
    if not raw_frames:
        return [], [], []

    H, W = raw_frames[0].shape[:2]

    # ── Pass 1: detect ────────────────────────────────────────────────────────
    # Each entry is either None (no face) or a dict with box/embedding/frame-idx.
    detections: list[dict | None] = []

    for idx, frame_bgr in enumerate(raw_frames):
        faces = face_app.get(frame_bgr)
        if not faces:
            detections.append(None)
            continue

        # Largest face wins
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        detections.append({
            "frame_idx": idx,
            "box":       [x1, y1, x2, y2],
            "embedding": face.normed_embedding,   # (512,) float32, already normed
        })

    valid = [d for d in detections if d is not None]
    print(f"  [buffalo] {len(valid)}/{len(raw_frames)} frames with a detected face.")

    if len(valid) < MIN_FACES:
        print(f"  [ERROR] Need at least {MIN_FACES} faces, got {len(valid)}. Aborting.")
        return [], [], []

    # ── Jitter / motion check ─────────────────────────────────────────────────
    # Normalise shift by the mean face size so the threshold is scale-invariant.
    boxes_arr = np.array([d["box"] for d in valid], dtype=float)   # (N, 4)

    cx_shift, cy_shift, worst = _jitter_shifts(boxes_arr)
    if cx_shift > JITTER_THR or cy_shift > JITTER_THR:
        print(
            f"  [WARNING] Excessive head movement between frames "
            f"{valid[worst]['frame_idx']} and {valid[worst + 1]['frame_idx']} "
            f"(cx_shift={cx_shift:.2f}, cy_shift={cy_shift:.2f}, "
            f"threshold={JITTER_THR}). Please keep your face still."
        )
        print("  [REJECTED] Session rejected due to excessive movement. Please try again.")
        return [], [], []

    # ── Aggregate box ─────────────────────────────────────────────────────────
    agg_x1, agg_y1, agg_x2, agg_y2 = _demo_aggregate_window(boxes_arr, (H, W))

    print(
        f"  [buffalo] Stable aggregate box (with {int(MARGIN*100)}% margin): "
        f"({agg_x1}, {agg_y1}, {agg_x2}, {agg_y2})"
    )

    # ── Pass 2: crop with aggregate box and save ──────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths:   list[Path]       = []
    cropped_faces: list[np.ndarray] = []
    embeddings:    list[np.ndarray] = []

    crop_idx = 0
    for det in valid:
        frame_bgr = raw_frames[det["frame_idx"]]

        # Single stable crop region – no per-frame jitter
        crop_bgr = frame_bgr[agg_y1:agg_y2, agg_x1:agg_x2]
        crop_rgb = cv2.cvtColor(
            cv2.resize(crop_bgr, (224, 224)),
            cv2.COLOR_BGR2RGB,
        )

        path = save_dir / f"frame_{crop_idx:04d}.jpg"
        cv2.imwrite(str(path), cv2.resize(crop_bgr, (224, 224)))
        saved_paths.append(path)

        cropped_faces.append(crop_rgb)
        # Embedding from Pass 1 – no re-detection on crop
        embeddings.append(det["embedding"])

        crop_idx += 1

    print(f"  [INFO] Saved {len(saved_paths)} stable crops to {save_dir}")
    return saved_paths, cropped_faces, embeddings


# ── public capture functions ──────────────────────────────────────────────────

def capture_from_camera(
    save_dir: Path,
    face_app: FaceAnalysis,
) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    """
    Open webcam; press C to start recording, Q to cancel.
    Raw frames are collected first, then processed by _process_raw_frames.
    """
    cap = cv2.VideoCapture(cfg.camera.device_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    raw_frames: list[np.ndarray] = []
    capturing    = False
    capture_start = 0.0
    last_t        = 0.0
    interval      = 1.0 / cfg.camera.fps
    total_dur     = cfg.camera.target_frames / cfg.camera.fps
    WIN           = "Authentication"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if not capturing:
            print("\r[camera] Press C to capture | Q to quit", end="", flush=True)
        else:
            elapsed   = time.time() - capture_start
            remaining = max(0.0, total_dur - elapsed)
            pct       = int(100 * len(raw_frames) / cfg.camera.target_frames)
            print(
                f"\r[camera] Recording… {remaining:.1f}s | "
                f"Frames: {len(raw_frames)}/{cfg.camera.target_frames} ({pct}%)",
                end="", flush=True,
            )

            now = time.time()
            if len(raw_frames) < cfg.camera.target_frames and (now - last_t) >= interval:
                raw_frames.append(frame.copy())
                last_t = now

            if len(raw_frames) >= cfg.camera.target_frames:
                print("\n[camera] Done recording.")
                cv2.imshow(WIN, display)
                cv2.waitKey(600)
                break

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Cancelled by user.")
        if key == ord("c") and not capturing:
            print("\n[camera] Capture started…")
            capturing     = True
            capture_start = time.time()
            last_t        = capture_start

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[camera] Collected {len(raw_frames)} raw frames.")

    result = _process_raw_frames(raw_frames, save_dir, face_app)
    if not result[0]:
        raise RuntimeError("Session rejected. Please try again.")
    return result


def capture_from_file(
    video_path: str,
    save_dir: Path,
    face_app: FaceAnalysis,
) -> tuple[list[Path], list[np.ndarray], list[np.ndarray]]:
    """Read up to target_frames from a video file, then run _process_raw_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_frames: list[np.ndarray] = []
    while len(raw_frames) < cfg.camera.target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    print(f"  [file] Read {len(raw_frames)} raw frames.")

    result = _process_raw_frames(raw_frames, save_dir, face_app)
    if not result[0]:
        raise RuntimeError("Session rejected. Please try again.")
    return result


# ── run directory helper ──────────────────────────────────────────────────────

def next_run_dir(subject_id: str) -> Path:
    base    = Path(cfg.paths.input_dir) / subject_id
    run_num = 1
    while (base / f"run{run_num:02d}").exists():
        run_num += 1
    return base / f"run{run_num:02d}"


# ── MTL model ─────────────────────────────────────────────────────────────────

def load_mtl_model(device: torch.device):
    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np.dtype,
    ])
    checkpoint = torch.load(
        cfg.paths.model_path, map_location=device, weights_only=False
    )
    model = MTLModel(get_config())
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


# Head name → the key `MTLModel.forward` actually returns (see train.py). Kept as
# an explicit map, and checked below, because the previous hard-coded "df_logit" /
# "sp_logit" did not match the model's "deepfake_logit" / "spoof_logit": every
# capture path raised KeyError before printing a single score, which is what
# "the camera produced no result" looked like from the outside.
_HEAD_KEYS = {
    "deepfake": "deepfake_logit",
    "spoof":    "spoof_logit",
    "temporal": "temp_logit",
}

# Per-head score range observed this session, so a head that never approaches its
# threshold can be reported as unreachable instead of quietly always reading "OK".
_score_seen: dict[str, list[float]] = {}


def run_mtl_on_frames(
    model,
    cropped_rgb_list: list[np.ndarray],
    device: torch.device,
    flow: torch.Tensor | None = None,
    record: bool = True,
) -> dict:
    """Score one clip. `flow` is (1, T-1, 2, R, R) when the checkpoint was trained
    with optical flow, else None — the flow_encoder stays out of the graph.

    `record=False` keeps a score out of the session extremes tracked for the
    [DIAG] line (used by the startup self-check, whose input is filler).
    """
    tensors      = [TRANSFORM(face_rgb) for face_rgb in cropped_rgb_list]
    frames_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # (1,T,C,H,W)

    with torch.no_grad():
        out = model(frames_tensor, flow=flow)

    missing = [k for k in _HEAD_KEYS.values() if k not in out]
    if missing:
        raise KeyError(
            f"MTLModel.forward returned {sorted(out)} — missing {missing}. "
            f"Update _HEAD_KEYS in app_demo.py to match train.py."
        )

    scores: dict = {}
    for name, key in _HEAD_KEYS.items():
        logit = out[key].flatten().float()
        score = torch.sigmoid(logit).item()
        scores[name]            = score
        scores[f"{name}_logit"] = logit.item()
        if record:
            _score_seen.setdefault(name, []).append(score)
    return scores


def check_mtl_results(scores: dict) -> bool:
    t       = cfg.model
    flagged = False
    heads   = [
        ("Deepfake Detection",   "deepfake", t.deepfake_threshold),
        ("Anti-Spoofing",        "spoof",    t.spoof_threshold),
        ("Temporal Consistency", "temporal", t.temporal_threshold),
    ]

    print("\n  ── MTL Detection Report ──────────────────────────")
    for label, key, threshold in heads:
        score  = scores[key]
        status = "WARNING" if score > threshold else "OK"
        print(f"  [{status}] {label}: score={score:.4f}  "
              f"logit={scores[f'{key}_logit']:+.4f}  threshold={threshold:.2f}")
        if score > threshold:
            flagged = True
    print("  ──────────────────────────────────────────────────")

    # A head whose observed maximum sits below its threshold can only ever return
    # "OK", so an all-OK verdict says nothing about that head. Report it, rather
    # than letting a collapsed head read as a clean pass.
    for label, key, threshold in heads:
        seen = _score_seen.get(key, [])
        if not seen:
            continue
        hi = max(seen)
        if hi < threshold:
            print(f"  [DIAG] {label}: max score this session {hi:.4f} < "
                  f"threshold {threshold:.2f} over {len(seen)} clip(s) — this head "
                  f"cannot flag anything. If the gap is large the head is likely "
                  f"untrained; check siw_sp_auc/siw_sp_recall in results.csv.")
    return flagged


# ── shared pipeline ───────────────────────────────────────────────────────────

def acquire_frames(
    subject_id: str,
    face_app: FaceAnalysis,
) -> tuple[list[Path], Path, list[np.ndarray], list[np.ndarray]]:
    src     = input("  Input source  [c=camera / f=file]: ").strip().lower()
    run_dir = next_run_dir(subject_id)

    if src == "c":
        paths, faces, embeds = capture_from_camera(run_dir, face_app)
    elif src == "f":
        video_path = input("  Video file path: ").strip()
        paths, faces, embeds = capture_from_file(video_path, run_dir, face_app)
    else:
        raise ValueError(f"Unknown source: {src!r}")

    return paths, run_dir, faces, embeds


def process_frames(
    frame_paths:   list[Path],
    run_dir:       Path,
    cropped_faces: list[np.ndarray],
    embeddings:    list[np.ndarray],
    mtl_model,
    device:        torch.device,
) -> bool:
    """Run MTL checks. Returns True if all pass."""
    print("  [MTL] Running inference…")
    scores      = run_mtl_on_frames(mtl_model, cropped_faces, device)
    mtl_flagged = check_mtl_results(scores)

    save_run_metadata(run_dir, {
        "mtl_scores":  scores,
        "mtl_flagged": mtl_flagged,
        "embeddings":  [e.tolist() for e in embeddings],
        "frame_count": len(embeddings),
    })

    if mtl_flagged:
        print("  [BLOCKED] MTL model flagged this session.")
        return False

    return True


# ── register / login ──────────────────────────────────────────────────────────

def register_flow(face_app: FaceAnalysis, mtl_model, device: torch.device) -> None:
    name = input("  Full name  : ").strip()
    sid  = input("  Student ID : ").strip()

    db = load_instance()
    if sid in db:
        print(f"  [INFO] User '{sid}' already registered. Re-registering…")

    print(f"\n  Registering [{name}] | ID: {sid}")
    frame_paths, run_dir, cropped, embeddings = acquire_frames(sid, face_app)
    passed = process_frames(frame_paths, run_dir, cropped, embeddings, mtl_model, device)

    if not passed:
        print("  [REGISTER FAILED] Could not complete registration due to security flags.")
        return

    indices      = [i for i in cfg.model.enroll_frame_indices if i < len(embeddings)]
    if not indices:
        indices  = [len(embeddings) // 2]

    enroll_vecs  = [embeddings[i].tolist() for i in indices]
    db[sid]      = {"name": name, "student_id": sid, "embeddings": enroll_vecs}
    save_instance(db)
    print(f"\n  [OK] Registered '{name}' ({sid}) with {len(enroll_vecs)} embedding(s).")


def login_flow(face_app: FaceAnalysis, mtl_model, device: torch.device) -> None:
    sid = input("  Student ID: ").strip()

    db = load_instance()
    if sid not in db:
        print(f"  [ERROR] User '{sid}' not found. Please register first.")
        return

    user   = db[sid]
    stored = [np.array(e, dtype=np.float32) for e in user["embeddings"]]

    print(f"\n  Authenticating [{user['name']}] | ID: {sid}")
    frame_paths, run_dir, cropped, embeddings = acquire_frames(sid, face_app)
    passed = process_frames(frame_paths, run_dir, cropped, embeddings, mtl_model, device)

    if not passed:
        print("  [LOGIN FAILED] Security check did not pass.")
        return

    sims     = [max(cosine_sim(p, r) for r in stored) for p in embeddings]
    mean_sim = float(np.mean(sims))

    (run_dir / "identity_verification.json").write_text(json.dumps({
        "mean_similarity":        mean_sim,
        "threshold":              cfg.model.identity_threshold,
        "per_frame_similarities": sims,
        "verified":               mean_sim >= cfg.model.identity_threshold,
    }, indent=2))

    print(f"\n  ── Identity Verification ─────────────────────────")
    print(f"  Mean similarity : {mean_sim:.4f}")
    print(f"  Threshold       : {cfg.model.identity_threshold:.2f}")

    if mean_sim >= cfg.model.identity_threshold:
        print(f"  [SUCCESS] Welcome, {user['name']}!")
    else:
        print(f"  [FAILED]  Identity not verified (score={mean_sim:.4f}).")
    print("  ──────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# Offline evaluation over the test split  (menu option `b`)
# ══════════════════════════════════════════════════════════════════════════════
# The metric *definitions* come from train.py so this report cannot disagree with
# results.csv. Everything below adds what training does not measure: uncertainty
# on the AUC, calibration, operating points other than 0.5, video-level pooling
# both ways, per-attack-type ranking, the end-to-end verdict the app would give,
# and latency — the one number a live demo is judged on that never appears in a
# training log.

_MIN_EVAL_FRAMES = 2      # 2 crops = one adjacent pair, the minimum the temporal
                          # head and the flow encoder can be given at all
_N_BOOTSTRAP = 1000
_CALIB_BINS = 10
_SWEEP = np.round(np.arange(0.05, 1.00, 0.05), 2)

# APCER operating points from ISO/IEC 30107-3, and the TPR operating points the
# deepfake literature reports. Kept as constants so the report is comparable
# between runs even when the score distribution moves.
#
# 0.1 % FPR is deliberately absent: the smallest measurable non-zero FPR is
# 1/n_bonafide, which is 1/191 = 0.0052 on the SiW test split, so a 0.1 % figure
# would be interpolation between the origin and the first step — a number the
# data cannot support. `tpr_at_fpr0` covers the strict end honestly instead.
_APCER_TARGETS = (0.01, 0.05, 0.10)
_FPR_TARGETS = (0.01, 0.05, 0.10)
_TPR_TARGETS = (0.90, 0.95, 0.99)


def _ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    return input(f"  {prompt}{shown}: ").strip() or default


def _next_eval_dir() -> Path:
    base = Path(_full_cfg.paths.output_root) / "demo_eval"
    n = 1
    while (base / f"eval{n:02d}").exists():
        n += 1
    return base / f"eval{n:02d}"


# ── manifest ──────────────────────────────────────────────────────────────────

def load_test_clips(csv_path: str, split: str = "test") -> list[dict]:
    """One record per (video, clip) of `split` in master.csv.

    Iteration is per **clip**, not per video, and that is the whole point: a clip
    is the unit the model was trained on, and pooling clips back up to their
    source video is what makes `video_auc` a measurement rather than an identity.
    Scoring one clip per video would leave nothing to pool.

    The frame rows are collapsed here — the crops on disk are not read at all.
    This walks the *raw* video the CSV points at, so the report measures the
    decode → detect → crop → infer path the demo actually runs, not a
    pre-extracted shortcut through it.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    df = pd.read_csv(path)
    required = {"video_path", "clip_index", "clip_start_frame", "dataset",
                "task", "label", "split", "subject_id", "spoof_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} is missing {sorted(missing)} — re-run preprocessing")

    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(
            f"no rows with split == {split!r} in {path.name} "
            f"(present: {sorted(df['split'].unique())})"
        )

    clips: list[dict] = []
    for (video_path, clip_index), grp in sub.groupby(
        ["video_path", "clip_index"], sort=True
    ):
        first = grp.iloc[0]
        clips.append({
            "video_path": str(video_path),
            "clip_index": int(clip_index),
            "clip_start_frame": int(first["clip_start_frame"]),
            # How many frames survived preprocessing's quality gate. The gate is
            # re-applied here, so a large shortfall against this number means the
            # two paths disagree and the crops are not comparable.
            "n_frames_csv": int(len(grp)),
            "dataset": str(first["dataset"]),
            "task": str(first["task"]),
            "subject_id": str(first["subject_id"]),
            "spoof_type": str(first["spoof_type"]),
            "label_raw": str(first["label"]),
            # 'real' is bona-fide in both datasets; 'fake' and 'spoof' are both
            # positives, which is also why the temporal head has a real label on
            # every row of both.
            "label": 0 if str(first["label"]).lower() == "real" else 1,
        })
    return clips


def _stratified_subset(clips: list[dict], limit: int, seed: int) -> list[dict]:
    """At most `limit` clips, proportional over (dataset, label, spoof_type).

    A head-of-list slice would be single-class — master.csv is grouped by video
    and the videos are label-sorted — and every metric guards on that, so a quick
    smoke run would produce an empty report instead of a small one.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(clips):
        return clips

    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[dict]] = {}
    for c in clips:
        strata.setdefault((c["dataset"], c["label"], c["spoof_type"]), []).append(c)

    out: list[dict] = []
    # Largest stratum first, so the rounding remainder lands where it costs the
    # least — and every stratum keeps at least one clip.
    order = sorted(strata.items(), key=lambda kv: -len(kv[1]))
    for i, (_, members) in enumerate(order):
        share = limit * len(members) / len(clips)
        take = max(1, int(round(share)))
        take = min(take, len(members), max(1, limit - len(out) - (len(order) - i - 1)))
        idx = rng.choice(len(members), size=take, replace=False)
        out.extend(members[int(j)] for j in sorted(idx))
    return out[:limit]


# ── one clip → one scored record ──────────────────────────────────────────────

def _sample_positions(n_available: int, T: int) -> list[int]:
    """Which of the clip's crops go to the model — the eval branch of
    `MTLDataset._sample_frames`, deliberately duplicated in only that one form.

    No jitter: jitter is a training augmentation, and a random gap would make two
    evaluation runs of the same checkpoint disagree.
    """
    if n_available <= T:
        return (list(range(n_available)) * ((T // max(n_available, 1)) + 1))[:T]
    return [int(i) for i in np.linspace(0, n_available - 1, T, dtype=int)]


def _clip_flow_tensor(
    crops_bgr: list[np.ndarray],
    positions: list[int],
    resize: int,
) -> torch.Tensor:
    """Flow for the sampled positions → (1, T-1, 2, R, R).

    Computed here rather than read from the `.npz` beside the crops, because this
    path re-derives its own crops from the raw video: reading a file produced from
    *different* crops would silently pair one geometry's flow with another's
    frames. The functions are the ones preprocessing calls, including the `/R`
    normalisation `load_clip_flow` applies, so the encoder sees its trained scale.
    """
    pair = compute_clip_flow(crops_bgr, resize=resize,
                             method=_full_cfg.preprocess.optical_flow_method)
    pair = pair / max(resize, 1)
    flow = accumulate_flow(pair, positions)
    return torch.from_numpy(flow).unsqueeze(0)


def _score_clip(
    clip: dict,
    detector: FaceDetector,
    face_app: FaceAnalysis,
    model,
    device: torch.device,
    *,
    crop_mode: str,
    frames_mode: str,
    use_flow: bool,
) -> dict:
    """Decode → detect → stable-crop → infer one clip, timing every stage.

    `crop_mode` exists because the two paths in this repo do not agree on the
    crop window: preprocessing centres a square on the **mean** box of the clip
    and drops frames whose face leaves it (IoU < `disp_ratio`), while the live
    demo takes the **union** of all boxes plus a 15 % margin and keeps every
    frame. The union window is strictly wider, so a demo-cropped face is smaller
    within the 224 px frame than anything the model was trained on. `preproc` is
    the default — it measures the checkpoint; `demo` measures the app.
    """
    pc = _full_cfg.preprocess
    rec: dict = {
        k: clip[k] for k in
        ("video_path", "clip_index", "clip_start_frame", "n_frames_csv",
         "dataset", "task", "subject_id", "spoof_type", "label_raw", "label")
    }
    rec["video"] = Path(clip["video_path"]).name
    rec["crop_mode"] = crop_mode
    rec["skip_reason"] = ""

    # ── decode ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    frames = [
        bgr for _, bgr in read_consecutive_frames(
            Path(clip["video_path"]), clip["clip_start_frame"],
            pc.frames_per_clip, step=pc.frame_skip,
        )
    ]
    t_decode = time.perf_counter()
    rec["n_read"] = len(frames)
    if not frames:
        rec["decode_ms"] = (t_decode - t0) * 1e3
        rec["skip_reason"] = "no frames decoded (missing or unreadable video)"
        return rec

    H, W = frames[0].shape[:2]
    rec["frame_h"], rec["frame_w"] = int(H), int(W)

    # ── detect: one pass, both modes ──────────────────────────────────────────
    # `detect_best` and the demo's largest-face pick differ only in the score
    # gate, so running the session once and applying the right rule afterwards
    # keeps the two modes comparable on identical detector output — and makes the
    # detect timing independent of which mode is being measured.
    boxes: list[np.ndarray | None] = []
    scores_det: list[float] = []
    for bgr in frames:
        faces = face_app.get(bgr) or []
        if crop_mode == "preproc":
            faces = [f for f in faces if float(f.det_score) >= detector.min_face_score]
        if not faces:
            boxes.append(None)
            continue
        face = max(faces, key=lambda f: bbox_area(f.bbox))
        boxes.append(np.asarray(face.bbox, dtype=float))
        scores_det.append(float(face.det_score))
    t_detect = time.perf_counter()

    valid_boxes = [b for b in boxes if b is not None]
    rec["n_detected"] = len(valid_boxes)
    rec["det_score_mean"] = float(np.mean(scores_det)) if scores_det else float("nan")
    rec["det_score_min"] = float(np.min(scores_det)) if scores_det else float("nan")

    # Jitter is *reported*, never used to reject: the live path rejects a shaky
    # session and asks the user to hold still, which is not an option for a fixed
    # test set. Recording it keeps the two paths' behaviour on the same clip
    # traceable — a high-jitter clip is one the demo would have refused outright.
    cx_shift, cy_shift, _ = _jitter_shifts(np.asarray(valid_boxes, dtype=float)) \
        if len(valid_boxes) >= 2 else (0.0, 0.0, -1)
    rec["jitter_cx"], rec["jitter_cy"] = cx_shift, cy_shift
    rec["would_reject_jitter"] = bool(
        cx_shift > JITTER_THR or cy_shift > JITTER_THR
    )
    rec["would_reject_faces"] = bool(len(valid_boxes) < MIN_FACES)

    if len(valid_boxes) < _MIN_EVAL_FRAMES:
        rec.update(decode_ms=(t_decode - t0) * 1e3,
                   detect_ms=(t_detect - t_decode) * 1e3)
        rec["skip_reason"] = f"only {len(valid_boxes)} frame(s) with a face"
        return rec

    # ── crop window + crops ───────────────────────────────────────────────────
    if crop_mode == "preproc":
        window = compute_average_crop_coords(
            valid_boxes, frame_shape=(H, W), crop_scale=detector.crop_scale
        )
        if window is None:
            rec.update(decode_ms=(t_decode - t0) * 1e3,
                       detect_ms=(t_detect - t_decode) * 1e3)
            rec["skip_reason"] = "degenerate crop window"
            return rec
        keep = [
            (bgr, b) for bgr, b in zip(frames, boxes)
            if b is not None and face_inside_window(b, window, pc.disp_ratio)
        ]
    else:
        window = _demo_aggregate_window(np.asarray(valid_boxes, dtype=float), (H, W))
        keep = [(bgr, b) for bgr, b in zip(frames, boxes) if b is not None]

    crops_bgr = [detector.crop_stable(bgr, window) for bgr, _ in keep]
    crops_bgr = [c for c in crops_bgr if c.size]
    t_crop = time.perf_counter()

    rec["n_valid"] = len(crops_bgr)
    rec["window"] = "%d,%d,%d,%d" % tuple(int(v) for v in window)
    # The gate preprocessing applied, reported rather than enforced: a clip that
    # now falls short is still scored, and `met_quality_gate` says which rows to
    # exclude if the thesis wants the strict subset.
    rec["met_quality_gate"] = bool(len(crops_bgr) >= pc.min_valid_frames)

    if len(crops_bgr) < _MIN_EVAL_FRAMES:
        rec.update(decode_ms=(t_decode - t0) * 1e3,
                   detect_ms=(t_detect - t_decode) * 1e3,
                   crop_ms=(t_crop - t_detect) * 1e3)
        rec["skip_reason"] = f"only {len(crops_bgr)} frame(s) inside the window"
        return rec

    # ── frames to the model ───────────────────────────────────────────────────
    if frames_mode == "all":
        positions = list(range(len(crops_bgr)))
    else:
        positions = _sample_positions(len(crops_bgr), _full_cfg.train.num_frames)
    rec["n_frames_model"] = len(positions)

    flow = None
    t_flow = t_crop
    if use_flow:
        flow = _clip_flow_tensor(
            crops_bgr, positions, _full_cfg.preprocess.flow_resize
        ).to(device)
        t_flow = time.perf_counter()

    # ── infer ─────────────────────────────────────────────────────────────────
    crops_rgb = [
        cv2.cvtColor(crops_bgr[p], cv2.COLOR_BGR2RGB) for p in positions
    ]
    scores = run_mtl_on_frames(model, crops_rgb, device, flow=flow)
    if device.type == "cuda":
        torch.cuda.synchronize()      # otherwise infer_ms times the launch, not the work
    t_infer = time.perf_counter()

    rec.update(scores)
    rec["decode_ms"] = (t_decode - t0) * 1e3
    rec["detect_ms"] = (t_detect - t_decode) * 1e3
    rec["crop_ms"] = (t_crop - t_detect) * 1e3
    rec["flow_ms"] = (t_flow - t_crop) * 1e3
    rec["infer_ms"] = (t_infer - t_flow) * 1e3
    rec["total_ms"] = (t_infer - t0) * 1e3
    # What the live path must fit into the acquisition window. Decode has no
    # analogue there — camera frames arrive on their own schedule.
    rec["online_ms"] = rec["detect_ms"] + rec["crop_ms"] + rec["flow_ms"] + rec["infer_ms"]

    dm = cfg.model
    rec["flag_deepfake"] = bool(scores["deepfake"] > dm.deepfake_threshold)
    rec["flag_spoof"] = bool(scores["spoof"] > dm.spoof_threshold)
    rec["flag_temporal"] = bool(scores["temporal"] > dm.temporal_threshold)
    rec["flagged"] = bool(
        rec["flag_deepfake"] or rec["flag_spoof"] or rec["flag_temporal"]
    )
    # The score a deployment that knows which task it is running would use. The
    # OR-gate above is what the app does with an unknown input; this is the
    # ceiling that gate is working against.
    rec["routed_score"] = (
        scores["deepfake"] if clip["task"] == "deepfake" else scores["spoof"]
    )
    return rec


# ── metric helpers (everything training does not already compute) ─────────────

def _dedup_curve(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last point of each run of equal x.

    `interp1d` requires strictly increasing x and an ROC has ties wherever
    several clips share a score — with 714 clips and a saturated head that is
    most of the curve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return x, y
    keep = np.ones(len(x), dtype=bool)
    keep[:-1] = x[1:] != x[:-1]
    return x[keep], y[keep]


def _interp_at(x, y, at: float) -> float:
    """y at x == `at`, or NaN when `at` is outside the observed range.

    Refusing to extrapolate matters at the strict operating points: with 191
    bona-fide clips the smallest non-zero FPR is 1/191 = 0.0052, so
    `tpr_at_fpr0.1%` is genuinely unmeasurable here and must read NaN rather
    than a fabricated number.
    """
    x, y = _dedup_curve(x, y)
    if len(x) < 2 or not (float(x.min()) <= at <= float(x.max())):
        return float("nan")
    return float(interp1d(x, y)(at))


def _finite_threshold(t, scores) -> float:
    """`roc_curve` prepends `thresholds[0] = inf` for the reject-everything point.

    Reported verbatim it becomes an `inf` in the JSON and a threshold nothing can
    cross; the equivalent finite value is just above the largest score.
    """
    t = float(t)
    return t if np.isfinite(t) else float(np.max(scores)) + 1e-6


def _eer(labels, scores) -> tuple[float, float]:
    """Equal error rate and the threshold that achieves it."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fx, fy = _dedup_curve(fpr, tpr)
    try:
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fx, fy)(x), 0.0, 1.0))
        i = int(np.argmin(np.abs(fpr - eer)))
    except Exception:                                              # noqa: BLE001
        # brentq needs a sign change; a step-like ROC from few clips may not have
        # one. Fall back to the observed point where FPR and FNR are closest.
        i = int(np.argmin(np.abs((1.0 - tpr) - fpr)))
        eer = float((fpr[i] + (1.0 - tpr[i])) / 2.0)
    return eer, _finite_threshold(thr[i], scores)


def _calibration(labels, scores, bins: int = _CALIB_BINS) -> tuple[dict, list[dict]]:
    """Brier / log-loss / ECE / MCE and the reliability table.

    Ranking metrics are invariant to any monotone squash of the scores, so a head
    can have AUC 0.95 and still put every clip at 0.49. The demo compares scores
    to a *fixed* threshold, which lives in score space — this block is what says
    whether `spoof_threshold = 0.6` is a sane place to stand.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    p = np.clip(scores, 1e-12, 1.0 - 1e-12)

    out = {
        "brier": float(np.mean((scores - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))),
        "score_std_pos": float(scores[labels == 1].std(ddof=1)) if (labels == 1).sum() > 1 else float("nan"),
        "score_std_neg": float(scores[labels == 0].std(ddof=1)) if (labels == 0).sum() > 1 else float("nan"),
    }

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    mce = 0.0
    table: list[dict] = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(scores[m].mean())
        frac = float(labels[m].mean())
        w = float(m.sum()) / len(scores)
        ece += w * abs(conf - frac)
        mce = max(mce, abs(conf - frac))
        table.append({
            "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": int(m.sum()), "mean_score": conf, "frac_pos": frac,
            "gap": conf - frac,
        })
    out["ece"] = float(ece)
    out["mce"] = float(mce)
    return out, table


def _dprime(labels, scores) -> float:
    """Distance between the two score means in pooled standard deviations.

    Reported alongside AUC because it is the same separation measured in the
    units a threshold uses. Two heads can share an AUC while one leaves a wide
    margin around the decision boundary and the other none.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = float(np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0))
    if pooled <= 0.0:
        return float("nan")
    return float((pos.mean() - neg.mean()) / pooled)


def _bootstrap_auc_ci(labels, scores, n: int = _N_BOOTSTRAP) -> dict:
    """Percentile 95 % CI on the AUC, resampling each class separately.

    The test split is 316 deepfake and 398 spoof clips. At that size two
    checkpoints differing by 0.02 AUC are not distinguishable, and an ablation
    table quoting point estimates alone cannot say so. Stratified resampling
    keeps every replicate two-class, which unstratified resampling does not
    guarantee on the smaller per-type subsets.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if len(pos) < 2 or len(neg) < 2:
        return {}

    rng = np.random.default_rng(_full_cfg.preprocess.split_seed)
    vals: list[float] = []
    for _ in range(n):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    a = np.asarray(vals, dtype=float)
    return {
        "auc_ci_lo": float(np.percentile(a, 2.5)),
        "auc_ci_hi": float(np.percentile(a, 97.5)),
        "auc_boot_std": float(a.std(ddof=1)),
        "auc_boot_n": int(len(a)),
    }


def _operating_points(labels, scores) -> dict:
    """The thresholds a deployment would actually choose, and their cost.

    Training reports metrics at a single fixed 0.5 (`EvalConfig.decision_threshold`)
    plus Youden-J for the deepfake head. Min-ACER is the anti-spoofing
    convention and best-F1 the detection one; quoting all three makes the gap
    between "the head can separate these" and "the shipped threshold does"
    explicit instead of leaving it to the reader.
    """
    fpr, tpr, thr = roc_curve(labels, scores)
    acer = ((1.0 - tpr) + fpr) / 2.0
    i = int(np.argmin(acer))
    j = int(np.argmax(tpr - fpr))

    prec, rec, pthr = precision_recall_curve(labels, scores)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    # precision_recall_curve returns one more point than thresholds (the
    # recall=0 endpoint), which has no threshold to report.
    k = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0

    return {
        "min_acer": float(acer[i]),
        "min_acer_threshold": _finite_threshold(thr[i], scores),
        "youden_j": float(float(tpr[j]) - float(fpr[j])),
        "youden_threshold": _finite_threshold(thr[j], scores),
        "best_f1": float(f1[k]) if len(f1) > 1 else float("nan"),
        "best_f1_threshold": float(pthr[k]) if len(pthr) else float("nan"),
    }


def _bpcer_at_apcer(labels, scores, targets=_APCER_TARGETS) -> dict:
    """BPCER at fixed APCER — the ISO/IEC 30107-3 reporting convention.

    With labels 1 = attack: APCER = 1 - TPR and BPCER = FPR, so both live on the
    ROC and no extra sweep is needed. This is the pair of numbers a PAD system is
    normally specified by, and a single ACER at 0.5 cannot substitute — ACER
    averages the two errors at *one* arbitrary point.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        f"bpcer_at_apcer{int(round(t * 100))}": _interp_at(tpr, fpr, 1.0 - t)
        for t in targets
    }


def _extra_metrics(labels, scores) -> tuple[dict, list[dict]]:
    """The shared "beyond training" block, appended to every head."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {}, []

    out: dict = {}
    eer, eer_thr = _eer(labels, scores)
    out["eer"] = eer
    out["eer_threshold"] = eer_thr
    out.update(_operating_points(labels, scores))
    out.update(_bootstrap_auc_ci(labels, scores))

    fpr, tpr, _ = roc_curve(labels, scores)
    # TPR at zero false positives — the strictest operating point, and the only
    # one at the strict end that needs no interpolation at all.
    out["tpr_at_fpr0"] = float(tpr[fpr == 0.0].max()) if (fpr == 0.0).any() else 0.0
    for f in _FPR_TARGETS:
        out[f"tpr_at_fpr{f * 100:g}"] = _interp_at(fpr, tpr, f)
    for t in _TPR_TARGETS:
        out[f"fpr_at_tpr{t * 100:g}"] = _interp_at(tpr, fpr, t)

    # Standardised partial AUC (McClish): the AUC restricted to the low-FPR
    # region, rescaled so 0.5 is still chance. The full AUC averages over
    # operating points no authentication system would ever run at.
    out["pauc_fpr10"] = float(roc_auc_score(labels, scores, max_fpr=0.10))
    out["pauc_fpr1"] = float(roc_auc_score(labels, scores, max_fpr=0.01))
    out["dprime"] = _dprime(labels, scores)

    preds = (scores >= _full_cfg.eval.decision_threshold).astype(int)
    out["cohen_kappa"] = float(cohen_kappa_score(labels, preds))

    cal, table = _calibration(labels, scores)
    out.update(cal)
    return out, table


def _threshold_sweep(labels, scores, grid=_SWEEP) -> list[dict]:
    """Every threshold-dependent metric across the grid, one row per threshold.

    `results.csv` fixes one operating point. This table is what shows whether a
    weak number is the head or the threshold — run01's spoof head topped out at
    0.4978, so every row above 0.50 would have been identical and empty.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    rows: list[dict] = []
    for t in grid:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append({
            "threshold": float(t),
            "acc": float((preds == labels).mean()),
            "balanced_acc": float(balanced_accuracy_score(labels, preds))
            if len(np.unique(labels)) > 1 else float("nan"),
            "precision": float(prec),
            "recall": float(tpr),                 # = TPR = 1 - APCER
            "specificity": float(tnr),            # = TNR = 1 - BPCER
            "f1": float(2 * prec * tpr / (prec + tpr)) if (prec + tpr) else 0.0,
            "mcc": float(matthews_corrcoef(labels, preds))
            if len(np.unique(preds)) > 1 else 0.0,
            "apcer": float(1.0 - tpr) if np.isfinite(tpr) else float("nan"),
            "bpcer": float(1.0 - tnr) if np.isfinite(tnr) else float("nan"),
            "acer": float((2.0 - tpr - tnr) / 2.0),
            "pred_pos_rate": float(preds.mean()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    return rows


def _video_level(labels, scores, videos, threshold: float) -> dict:
    """Pool clips to their source video, both ways.

    `compute_deepfake_metrics` already reports `video_auc` at
    `EvalConfig.video_agg`. Reporting *both* poolings costs nothing and settles
    an empirical question rather than asserting it: `max` wins when only a few
    clips of a video carry the artefact, `mean` when it is everywhere and
    per-clip noise dominates.
    """
    df = pd.DataFrame({"video": videos, "label": labels, "score": scores})
    out: dict = {"n_videos": int(df["video"].nunique()),
                 "n_clips": int(len(df))}
    for agg in ("mean", "max"):
        g = df.groupby("video").agg(label=("label", "first"), score=("score", agg))
        if g["label"].nunique() < 2:
            continue
        y = g["label"].to_numpy()
        s = g["score"].to_numpy()
        m = binary_classification_metrics(y, s, threshold=threshold)
        m["eer"] = _eer(y, s)[0]
        out[agg] = m
    return out


def _per_type_table(labels, scores, types, threshold: float) -> list[dict]:
    """Per attack type: support, recall at `threshold`, and AUC vs. bona-fide.

    `compute_spoof_metrics` gives `recall_<type>`, which conflates two different
    failures: an attack the head cannot rank above bona-fide at all, and one it
    ranks correctly but scores below the shipped threshold. The per-type AUC
    separates them — the first needs more data for that attack, the second needs
    a different threshold. Sorted worst-AUC-first, since that is the row the
    thesis has to explain.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    types = np.asarray(types)
    bona = labels == 0

    rows: list[dict] = []
    for st in sorted({t for t, lb in zip(types, labels) if lb == 1}):
        m = (types == st) & (labels == 1)
        if not m.any():
            continue
        row = {
            "spoof_type": str(st),
            "n": int(m.sum()),
            "recall": float((scores[m] >= threshold).mean()),
            "score_mean": float(scores[m].mean()),
            "score_min": float(scores[m].min()),
            "score_max": float(scores[m].max()),
        }
        if bona.any():
            y = np.concatenate([np.zeros(int(bona.sum())), np.ones(int(m.sum()))])
            s = np.concatenate([scores[bona], scores[m]])
            row["n_bonafide"] = int(bona.sum())
            row["auc_vs_bonafide"] = float(roc_auc_score(y, s))
            row["eer_vs_bonafide"] = _eer(y, s)[0]
        rows.append(row)
    rows.sort(key=lambda r: r.get("auc_vs_bonafide", 0.0))
    return rows


def _curves(labels, scores) -> dict:
    """ROC and PR points, for the report's CSV dumps and the thesis plots."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {}
    fpr, tpr, thr = roc_curve(labels, scores)
    prec, rec, pthr = precision_recall_curve(labels, scores)
    return {
        "roc": [
            {"fpr": float(a), "tpr": float(b),
             "threshold": _finite_threshold(c, scores)}
            for a, b, c in zip(fpr, tpr, thr)
        ],
        "pr": [
            {"recall": float(a), "precision": float(b),
             "threshold": float(c) if i < len(pthr) else float("nan")}
            for i, (a, b, c) in enumerate(zip(
                rec, prec, np.append(pthr, float("nan"))
            ))
        ],
    }


def _pct(values: list[float]) -> dict:
    """mean / sd / min / median / p90 / p95 / p99 / max for one timing series."""
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def _timing_report(records: list[dict], elapsed_s: float) -> dict:
    """Per-stage wall clock, and what it implies for the live path.

    The distinction the table turns on: **decode** reads a file and has no
    analogue on a camera, where frames arrive on the sensor's schedule. So
    `total_ms` is what a batch run costs per clip, and `online_ms`
    (detect + crop + flow + infer) is what the live path must fit inside the
    acquisition window — `camera.target_frames / camera.fps`, a fixed 3.2 s
    here. `realtime_factor < 1` means a verdict lands before the next window
    would close; `max_sustainable_fps` is the frame rate the pipeline could keep
    up with continuously.
    """
    cam = cfg.camera
    window_s = cam.target_frames / max(cam.fps, 1e-9)

    stages = ("decode_ms", "detect_ms", "crop_ms", "flow_ms", "infer_ms",
              "total_ms", "online_ms")
    out: dict = {
        "n_clips": len(records),
        "wall_clock_s": float(elapsed_s),
        "clips_per_s": float(len(records) / elapsed_s) if elapsed_s > 0 else float("nan"),
        "acquisition_window_s": float(window_s),
        "stages_ms": {s: _pct([r[s] for r in records if s in r]) for s in stages},
    }

    online = [r["online_ms"] for r in records if "online_ms" in r]
    if online:
        a = np.asarray(online, dtype=float)
        out["realtime"] = {
            # Verdict latency == online cost: the time from the last frame of the
            # window to a printed decision.
            "verdict_latency_ms_mean": float(a.mean()),
            "verdict_latency_ms_p95": float(np.percentile(a, 95)),
            "realtime_factor_mean": float(a.mean() / (window_s * 1e3)),
            "realtime_factor_p95": float(np.percentile(a, 95) / (window_s * 1e3)),
            "realtime_ok_frac": float((a <= window_s * 1e3).mean()),
        }
        per_frame = [
            r["online_ms"] / max(r.get("n_valid", 1), 1) for r in records
            if "online_ms" in r
        ]
        if per_frame:
            pf = float(np.median(per_frame))
            out["realtime"]["online_ms_per_frame_median"] = pf
            out["realtime"]["max_sustainable_fps"] = float(1e3 / pf) if pf > 0 else float("inf")

    # Where the time actually goes, as a share — the number that says which stage
    # to optimise. Detection dominates by construction: one InsightFace pass per
    # frame against one backbone pass per clip.
    totals = {s: sum(r.get(s, 0.0) for r in records)
              for s in ("decode_ms", "detect_ms", "crop_ms", "flow_ms", "infer_ms")}
    grand = sum(totals.values()) or 1.0
    out["share_pct"] = {k: float(100.0 * v / grand) for k, v in totals.items()}
    return out


# ── report assembly ───────────────────────────────────────────────────────────

def _head_section(
    labels, scores, videos, name: str, threshold: float, *,
    base: dict,
) -> dict:
    """Wrap one head's metrics with the shared extras, curves and diagnostics."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    extra, reliability = _extra_metrics(labels, scores)

    metrics = dict(base)
    # Training's own numbers win on any shared key, so this report cannot drift
    # from results.csv; the extras only fill in what training never computed.
    #
    # One key does collide in practice: `compute_spoof_metrics` reads TPR@FPR=1%
    # off the ROC with `interp1d`, whose result is undefined where the curve has
    # duplicate FPRs — and with 32 bona-fide clips most of it does. The extras
    # dedupe first (see `_dedup_curve`). The gap is ~0.003 and it is a tie-density
    # artefact, not a disagreement about the metric, so keep both under distinct
    # names rather than silently picking one.
    for k, v in extra.items():
        old = metrics.get(k)
        if (isinstance(old, float) and isinstance(v, float)
                and np.isfinite(old) and np.isfinite(v)
                and abs(old - v) > 1e-9):
            metrics[f"{k}_recomputed"] = v
        metrics.setdefault(k, v)

    return {
        "n": int(len(labels)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "threshold": float(threshold),
        "metrics": metrics,
        "video_level": _video_level(labels, scores, videos, threshold) if videos else {},
        "sweep": _threshold_sweep(labels, scores),
        "reliability": reliability,
        "curves": _curves(labels, scores),
        "warnings": collapse_warnings(name, metrics, threshold),
    }


def build_report(
    records: list[dict],
    skipped: list[dict],
    clips: list[dict],
    *,
    crop_mode: str,
    frames_mode: str,
    use_flow: bool,
    csv_path: str,
    device: torch.device,
    elapsed_s: float,
    power: dict | None,
    interrupted: bool,
) -> dict:
    """Everything measurable from `records`, grouped by head.

    Thresholds are deliberately mixed: each head's *metrics* use
    `EvalConfig.decision_threshold` so they line up with `results.csv`, while the
    verdict section uses the three `DemoModelConfig` thresholds the app ships
    with. Those are different questions — "can the head separate these classes"
    and "does the deployed gate fire" — and run01 is the case where the answers
    disagreed for 14 epochs.
    """
    ec, dm = _full_cfg.eval, cfg.model
    thr = ec.decision_threshold

    df = pd.DataFrame(records)
    labels = df["label"].to_numpy().astype(int)
    videos = df["video_path"].tolist()

    report: dict = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(cfg.paths.model_path),
            "manifest": str(csv_path),
            "split": "test",
            "device": str(device),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "crop_mode": crop_mode,
            "frames_mode": frames_mode,
            "num_frames": int(_full_cfg.train.num_frames),
            "use_optical_flow": bool(use_flow),
            # The two paths do not share a detector: preprocessing ran the
            # detection-only `buffalo_sc`, the demo needs `buffalo_l` for its
            # recognition embedding. Slightly different boxes give a slightly
            # different crop window, which is why `frames_delta_vs_csv` is not
            # always 0 and why a few clips fail the IoU gate here that passed
            # there. Recorded so the divergence is attributable.
            "det_model_eval": str(cfg.paths.buffalo_model),
            "det_model_preprocess": str(_full_cfg.preprocess.insightface_det_model),
            "decision_threshold": float(thr),
            "demo_thresholds": {
                "deepfake": dm.deepfake_threshold,
                "spoof": dm.spoof_threshold,
                "temporal": dm.temporal_threshold,
            },
            "interrupted": bool(interrupted),
        },
        "manifest": {
            "clips_requested": len(clips),
            "clips_scored": len(records),
            "clips_skipped": len(skipped),
            "videos_scored": int(df["video_path"].nunique()),
            "subjects": int(df["subject_id"].nunique()),
            "by_dataset": {
                str(k): int(v) for k, v in df["dataset"].value_counts().items()
            },
            "by_label": {
                str(k): int(v) for k, v in df["label_raw"].value_counts().items()
            },
            "skip_reasons": {
                str(k): int(v) for k, v in
                pd.Series([s.get("skip_reason", "?") for s in skipped])
                .value_counts().items()
            } if skipped else {},
        },
        # How faithfully this path reproduced preprocessing's clips. A large
        # negative `frames_delta_vs_csv` means this run found *fewer* usable
        # frames than preprocessing kept, so the crops differ from the trained
        # ones and every metric below is measuring a slightly different input.
        # Small non-zero values in either direction are the two detectors
        # disagreeing (see meta.det_model_*), not a fault.
        "acquisition": {
            "frames_read": _pct([r["n_read"] for r in records if "n_read" in r]),
            "frames_detected": _pct([r["n_detected"] for r in records if "n_detected" in r]),
            "frames_valid": _pct([r["n_valid"] for r in records if "n_valid" in r]),
            "frames_to_model": _pct([r["n_frames_model"] for r in records if "n_frames_model" in r]),
            "frames_delta_vs_csv": _pct([
                r["n_valid"] - r["n_frames_csv"]
                for r in records if "n_valid" in r
            ]),
            "det_score": _pct([
                r["det_score_mean"] for r in records
                if np.isfinite(r.get("det_score_mean", float("nan")))
            ]),
            "met_quality_gate_frac": float(
                np.mean([bool(r.get("met_quality_gate")) for r in records])
            ),
            # Clips the live app would have refused before scoring them at all.
            "would_reject_jitter_frac": float(
                np.mean([bool(r.get("would_reject_jitter")) for r in records])
            ),
            "would_reject_faces_frac": float(
                np.mean([bool(r.get("would_reject_faces")) for r in records])
            ),
        },
        "timing": _timing_report(records, elapsed_s),
    }

    warnings: list[str] = []

    # ── deepfake: FF++ rows only ──────────────────────────────────────────────
    d = df[df["task"] == "deepfake"]
    if len(d) and d["label"].nunique() > 1:
        y, s = d["label"].to_numpy().astype(int), d["deepfake"].to_numpy()
        base = compute_deepfake_metrics(
            y, s, video_paths=d["video_path"].tolist(),
            video_agg=ec.video_agg, threshold=thr,
        )
        base["best_threshold"] = _finite_threshold(base.get("best_threshold", np.nan), s)
        sec = _head_section(y, s, d["video_path"].tolist(), "deepfake", thr, base=base)
        # Structural, not a bug: the raw FF++ files here are the DFD actor subset
        # and carry no c23/c40 token, so there is nothing to break down by.
        sec["compression_breakdown"] = (
            "unavailable — DFD actor filenames carry no compression token"
        )
        report["deepfake"] = sec
        warnings += sec["warnings"]
    else:
        report["deepfake"] = {"n": int(len(d)),
                              "note": "single-class or empty — metrics withheld"}

    # ── spoof: SiW-Mv2 rows only ──────────────────────────────────────────────
    p = df[df["task"] == "spoof"]
    if len(p) and p["label"].nunique() > 1:
        y, s = p["label"].to_numpy().astype(int), p["spoof"].to_numpy()
        base = compute_spoof_metrics(
            y, s, threshold=thr, fpr_threshold=ec.fpr_threshold,
            spoof_types=p["spoof_type"].tolist(),
        )
        base.update(_bpcer_at_apcer(y, s))
        sec = _head_section(y, s, p["video_path"].tolist(), "spoof", thr, base=base)
        sec["per_type"] = _per_type_table(y, s, p["spoof_type"].to_numpy(), thr)
        report["spoof"] = sec
        warnings += sec["warnings"]
    else:
        report["spoof"] = {"n": int(len(p)),
                           "note": "single-class or empty — metrics withheld"}

    # ── temporal: every row, in both datasets ─────────────────────────────────
    # The temporal label is genuine ground truth on both sources — a deepfake and
    # a presentation attack are both temporally inconsistent — so unlike the two
    # classification heads this one is not masked to a task.
    if df["label"].nunique() > 1:
        s = df["temporal"].to_numpy()
        base = compute_temporal_metrics(labels, s, threshold=thr)
        sec = _head_section(labels, s, videos, "temporal", thr, base=base)
        # Per-source breakdown: an aggregate can hide a head that works on one
        # dataset's artefacts and not the other's, which is precisely the claim
        # a shared temporal head makes.
        sec["by_dataset"] = {}
        for name, grp in df.groupby("dataset"):
            if grp["label"].nunique() < 2:
                continue
            gy = grp["label"].to_numpy().astype(int)
            gs = grp["temporal"].to_numpy()
            m = compute_temporal_metrics(gy, gs, threshold=thr)
            m["eer"] = _eer(gy, gs)[0]
            sec["by_dataset"][str(name)] = m
        report["temporal"] = sec
        warnings += sec["warnings"]
    else:
        report["temporal"] = {"n": int(len(df)),
                              "note": "single-class — metrics withheld"}

    # ── the app's own verdict ─────────────────────────────────────────────────
    report["system"] = _system_report(df, labels)
    report["verdict"] = report["system"].get("or_gate", {})

    if power:
        report["power"] = power
    report["collapse_warnings"] = warnings
    return report


def _system_report(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """What the demo would actually have decided, and its ceiling.

    Two rows, and the gap between them is the point:

    * **or_gate** — the shipped rule: flag if *any* head crosses its own
      `DemoModelConfig` threshold. This is the only number that describes the
      product, and it is threshold-bound, so a head whose ceiling sits below its
      threshold contributes exactly nothing to it.
    * **routed** — the deepfake score on FF++ rows and the spoof score on SiW
      rows, i.e. what the same checkpoint achieves when the task is known. It
      bounds what any gate over these three heads could reach.
    """
    dm = cfg.model
    out: dict = {}

    flagged = df["flagged"].to_numpy().astype(int)
    tp = int(((flagged == 1) & (labels == 1)).sum())
    fp = int(((flagged == 1) & (labels == 0)).sum())
    fn = int(((flagged == 0) & (labels == 1)).sum())
    tn = int(((flagged == 0) & (labels == 0)).sum())
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, flagged, average="binary", zero_division=0
    )
    out["or_gate"] = {
        "acc": float((flagged == labels).mean()),
        "balanced_acc": float(balanced_accuracy_score(labels, flagged))
        if len(np.unique(labels)) > 1 else float("nan"),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, flagged))
        if len(np.unique(flagged)) > 1 else 0.0,
        # FAR/FRR in the authentication sense: a bona-fide user wrongly blocked
        # is an FRR event, an attack wrongly admitted an FAR event.
        "far": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
        "frr": float(fp / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "flag_rate": float(flagged.mean()),
        "per_head_fire_rate": {
            "deepfake": float(df["flag_deepfake"].mean()),
            "spoof": float(df["flag_spoof"].mean()),
            "temporal": float(df["flag_temporal"].mean()),
        },
        # A head that never fires cannot influence the gate. This is the machine-
        # readable form of the demo's [DIAG] line.
        "dead_heads": [
            h for h, thr_ in (("deepfake", dm.deepfake_threshold),
                              ("spoof", dm.spoof_threshold),
                              ("temporal", dm.temporal_threshold))
            if float(df[h].max()) < thr_
        ],
        "head_score_max": {
            h: float(df[h].max()) for h in ("deepfake", "spoof", "temporal")
        },
    }

    if len(np.unique(labels)) > 1:
        routed = df["routed_score"].to_numpy()
        base = binary_classification_metrics(
            labels, routed, threshold=_full_cfg.eval.decision_threshold
        )
        extra, _ = _extra_metrics(labels, routed)
        for k, v in extra.items():
            base.setdefault(k, v)
        out["routed"] = base
        out["routed_curves"] = _curves(labels, routed)
    return out


# ── printing ──────────────────────────────────────────────────────────────────

def _fmt(v, width: int = 9) -> str:
    if isinstance(v, bool):
        return f"{str(v):>{width}}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):>{width}d}"
    if isinstance(v, (float, np.floating)):
        return "      n/a" if not np.isfinite(v) else f"{float(v):>{width}.4f}"
    return f"{str(v):>{width}}"


def print_report(report: dict) -> str:
    """Print the human-readable summary and return the same text.

    Returning it keeps `summary.txt` and the console byte-identical — two
    formatters would eventually disagree, and the file is what ends up in the
    thesis appendix.
    """
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    def block(title: str, d: dict, keys=None, indent: str = "    ") -> None:
        if not d:
            return
        out(f"  {title}")
        items = [(k, d[k]) for k in (keys or d.keys()) if k in d]
        for k, v in items:
            if isinstance(v, (dict, list)):
                continue
            out(f"{indent}{k:<26}{_fmt(v)}")

    m = report["meta"]
    out("\n" + "═" * 78)
    out("  TEST-SPLIT EVALUATION" + ("  [INTERRUPTED]" if m["interrupted"] else ""))
    out("═" * 78)
    out(f"  checkpoint      {m['checkpoint']}")
    out(f"  manifest        {m['manifest']}  (split={m['split']})")
    out(f"  crop={m['crop_mode']}  frames={m['frames_mode']}"
        f"({m['num_frames']})  flow={m['use_optical_flow']}  device={m['device']}")
    out(f"  metric threshold {m['decision_threshold']}   demo thresholds "
        f"df={m['demo_thresholds']['deepfake']} sp={m['demo_thresholds']['spoof']} "
        f"tmp={m['demo_thresholds']['temporal']}")

    mf = report["manifest"]
    out(f"\n  clips {mf['clips_scored']}/{mf['clips_requested']} scored"
        f"  ({mf['clips_skipped']} skipped)"
        f"   videos {mf['videos_scored']}   subjects {mf['subjects']}")
    out(f"  by dataset      {mf['by_dataset']}")
    out(f"  by label        {mf['by_label']}")
    for reason, n in mf["skip_reasons"].items():
        out(f"    skipped {n:>4}  {reason}")

    aq = report["acquisition"]
    out("\n  ── acquisition ─────────────────────────────────────────────────────")
    for k in ("frames_read", "frames_detected", "frames_valid",
              "frames_to_model", "frames_delta_vs_csv", "det_score"):
        s = aq.get(k, {})
        if s:
            out(f"    {k:<20} mean {s['mean']:7.2f}  min {s['min']:7.2f}  "
                f"p50 {s['p50']:7.2f}  max {s['max']:7.2f}")
    out(f"    met_quality_gate     {aq['met_quality_gate_frac'] * 100:6.2f} % of clips")
    out(f"    demo would reject    jitter {aq['would_reject_jitter_frac'] * 100:5.2f} % "
        f"| too-few-faces {aq['would_reject_faces_frac'] * 100:5.2f} %")

    _print_head(out, "DEEPFAKE  (FaceForensics++)", report.get("deepfake", {}),
                extra_keys=("acc_best_thresh", "best_threshold", "video_auc"))
    _print_head(out, "ANTI-SPOOF  (SiW-Mv2)", report.get("spoof", {}),
                extra_keys=("apcer", "bpcer", "acer", "hter",
                            "tpr_at_fpr1", "bpcer_at_apcer1",
                            "bpcer_at_apcer5", "bpcer_at_apcer10"))
    _print_head(out, "TEMPORAL CONSISTENCY  (both datasets)",
                report.get("temporal", {}), extra_keys=("bin_acc",))

    sp = report.get("spoof", {})
    if sp.get("per_type"):
        out("\n  ── per attack type (worst AUC first) ────────────────────────────────")
        out(f"    {'type':<30}{'n':>5}{'recall':>9}{'AUC':>9}{'EER':>9}{'mean':>9}")
        for r in sp["per_type"]:
            out(f"    {r['spoof_type'][:29]:<30}{r['n']:>5}"
                f"{_fmt(r['recall'])}{_fmt(r.get('auc_vs_bonafide', float('nan')))}"
                f"{_fmt(r.get('eer_vs_bonafide', float('nan')))}{_fmt(r['score_mean'])}")

    tmp = report.get("temporal", {})
    if tmp.get("by_dataset"):
        out("\n  ── temporal head per source ────────────────────────────────────────")
        out(f"    {'dataset':<22}{'acc':>9}{'AUC':>9}{'EER':>9}{'AP':>9}")
        for name, mm in tmp["by_dataset"].items():
            out(f"    {name[:21]:<22}{_fmt(mm.get('acc', float('nan')))}"
                f"{_fmt(mm.get('auc_roc', float('nan')))}"
                f"{_fmt(mm.get('eer', float('nan')))}"
                f"{_fmt(mm.get('ap', float('nan')))}")

    sysr = report.get("system", {})
    if sysr.get("or_gate"):
        g = sysr["or_gate"]
        out("\n  ── end-to-end verdict (the rule the app ships) ─────────────────────")
        block("OR over the three demo thresholds:", g,
              keys=("acc", "balanced_acc", "precision", "recall", "f1", "mcc",
                    "far", "frr", "tp", "fp", "fn", "tn", "flag_rate"))
        out(f"    fire rate per head        {g['per_head_fire_rate']}")
        out(f"    max score per head        "
            + "  ".join(f"{k}={v:.4f}" for k, v in g["head_score_max"].items()))
        if g["dead_heads"]:
            out(f"    [DIAG] never fires: {', '.join(g['dead_heads'])} — their "
                f"ceiling is below their threshold, so an all-OK verdict says "
                f"nothing about them.")
    if sysr.get("routed"):
        block("task-routed score (upper bound for any gate):", sysr["routed"],
              keys=("acc", "balanced_acc", "recall", "specificity", "f1", "mcc",
                    "auc_roc", "ap", "eer", "min_acer", "min_acer_threshold"))

    t = report["timing"]
    out("\n  ── timing ──────────────────────────────────────────────────────────")
    out(f"    {'stage':<14}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}{'share':>9}")
    for stage, s in t["stages_ms"].items():
        if not s:
            continue
        share = t["share_pct"].get(stage)
        out(f"    {stage:<14}{s['mean']:>9.1f}{s['p50']:>9.1f}{s['p95']:>9.1f}"
            f"{s['max']:>9.1f}" + (f"{share:>8.1f}%" if share is not None else " " * 9))
    rt = t.get("realtime", {})
    if rt:
        out(f"\n    acquisition window        {t['acquisition_window_s']:.2f} s "
            f"({cfg.camera.target_frames} frames @ {cfg.camera.fps:g} fps)")
        out(f"    verdict latency           {rt['verdict_latency_ms_mean']:.1f} ms mean, "
            f"{rt['verdict_latency_ms_p95']:.1f} ms p95")
        out(f"    realtime factor           {rt['realtime_factor_mean']:.3f} mean, "
            f"{rt['realtime_factor_p95']:.3f} p95  "
            f"({'OK' if rt['realtime_factor_p95'] < 1 else 'TOO SLOW'})")
        out(f"    within window             {rt['realtime_ok_frac'] * 100:.1f} % of clips")
        if "max_sustainable_fps" in rt:
            out(f"    max sustainable rate      {rt['max_sustainable_fps']:.1f} fps "
                f"({rt['online_ms_per_frame_median']:.1f} ms/frame median)")
        out(f"    throughput                {t['clips_per_s']:.2f} clips/s over "
            f"{t['wall_clock_s'] / 60:.1f} min")

    if report.get("power"):
        out("\n  ── energy ──────────────────────────────────────────────────────────")
        for k, v in report["power"].items():
            if not isinstance(v, (dict, list)):
                out(f"    {k:<26}{_fmt(v)}")

    if report.get("collapse_warnings"):
        out("\n  ── COLLAPSE WARNINGS ───────────────────────────────────────────────")
        for w in report["collapse_warnings"]:
            out(f"    [!] {w}")
    out("═" * 78 + "\n")
    return "\n".join(lines)


def _print_head(out, title: str, sec: dict, extra_keys=()) -> None:
    """One head's block: the core 20 keys, its extras, then the new ones."""
    out(f"\n  ── {title} ".ljust(76, "─"))
    if "note" in sec:
        out(f"    {sec['note']}  (n={sec.get('n', 0)})")
        return
    m = sec["metrics"]
    out(f"    n={sec['n']}  pos={sec['n_pos']}  neg={sec['n_neg']}  "
        f"@threshold={sec['threshold']}")

    groups = [
        ("threshold metrics", ("acc", "balanced_acc", "precision", "recall",
                               "specificity", "f1", "mcc", "cohen_kappa",
                               "pred_pos_rate")),
        ("ranking / threshold-free", ("auc_roc", "auc_ci_lo", "auc_ci_hi",
                                      "auc_boot_std", "ap", "eer",
                                      "eer_threshold", "pauc_fpr10",
                                      "pauc_fpr1", "dprime")),
        ("operating points", ("min_acer", "min_acer_threshold", "youden_j",
                              "youden_threshold", "best_f1", "best_f1_threshold",
                              "tpr_at_fpr0", "tpr_at_fpr1", "tpr_at_fpr1_recomputed",
                              "tpr_at_fpr5", "tpr_at_fpr10", "fpr_at_tpr90",
                              "fpr_at_tpr95", "fpr_at_tpr99")),
        ("task-specific", tuple(extra_keys)),
        ("calibration", ("brier", "log_loss", "ece", "mce",
                         "score_std_pos", "score_std_neg")),
        ("confusion / collapse", ("tp", "fp", "fn", "tn", "score_min",
                                  "score_max", "score_mean_pos", "score_mean_neg")),
    ]
    for gname, keys in groups:
        present = [(k, m[k]) for k in keys if k in m]
        if not present:
            continue
        out(f"    {gname}:")
        for i in range(0, len(present), 3):
            row = present[i:i + 3]
            out("      " + "  ".join(
                f"{k:<20}{_fmt(v, 8)}" for k, v in row
            ))

    vl = sec.get("video_level", {})
    if vl.get("mean") or vl.get("max"):
        out(f"    video level ({vl['n_videos']} videos from {vl['n_clips']} clips):")
        for agg in ("mean", "max"):
            if agg not in vl:
                continue
            v = vl[agg]
            out(f"      pool={agg:<6} auc={_fmt(v.get('auc_roc', float('nan')), 7)}"
                f"  eer={_fmt(v.get('eer', float('nan')), 7)}"
                f"  acc={_fmt(v.get('acc', float('nan')), 7)}"
                f"  ap={_fmt(v.get('ap', float('nan')), 7)}")
    if isinstance(sec.get("compression_breakdown"), str):
        out(f"    compression breakdown: {sec['compression_breakdown']}")
    for w in sec.get("warnings", []):
        out(f"    [!] {w}")


# ── writing ───────────────────────────────────────────────────────────────────

def _jsonable(obj):
    """numpy scalars → python, non-finite floats → None.

    `json.dump` accepts NaN and Infinity by default but emits bare `NaN`, which
    is not valid JSON and breaks every strict parser downstream — including
    `pandas.read_json` and `jq`.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _flatten(obj, prefix: str = "") -> dict:
    """Nested report → dotted scalar keys, skipping the list-valued tables.

    The tables (curves, sweep, per-type, reliability) get their own CSVs; folding
    them in here would produce thousands of columns for one row.
    """
    flat: dict = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            continue
        else:
            flat[key] = v
    return flat


def _write_rows(path: Path, rows: list[dict]) -> None:
    """CSV from a list of dicts, over the union of their keys.

    Union, not row 0's keys: the per-type rows drop `auc_vs_bonafide` when there
    are no bona-fide clips, and a fixed header would shift later values into the
    wrong columns — the same bug `ResultLogger` was fixed for.
    """
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_report(out_dir: Path, report: dict, records: list[dict],
                 skipped: list[dict], text: str) -> None:
    """report.json + summary.txt + one CSV per table.

    JSON for reprocessing, txt for the appendix, CSVs because the curves and the
    sweep are what the thesis plots — and `clip_predictions.csv` so any number
    above can be recomputed from the raw per-clip scores without re-running the
    2 h pass. `skipped_clips.csv` names the clips that produced no score, since
    report.json only counts them by reason and "which ones" is the question you
    ask next.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
    (out_dir / "summary.txt").write_text(text)

    _write_rows(out_dir / "clip_predictions.csv", records)
    if skipped:
        _write_rows(out_dir / "skipped_clips.csv", skipped)
    _write_rows(out_dir / "metrics_flat.csv",
                [{"metric": k, "value": v} for k, v in
                 sorted(_flatten(report).items())])

    for head in ("deepfake", "spoof", "temporal"):
        sec = report.get(head, {})
        if not isinstance(sec, dict) or "metrics" not in sec:
            continue
        curves = sec.get("curves", {})
        if curves.get("roc"):
            _write_rows(out_dir / f"roc_{head}.csv", curves["roc"])
        if curves.get("pr"):
            _write_rows(out_dir / f"pr_{head}.csv", curves["pr"])
        if sec.get("sweep"):
            _write_rows(out_dir / f"threshold_sweep_{head}.csv", sec["sweep"])
        if sec.get("reliability"):
            _write_rows(out_dir / f"reliability_{head}.csv", sec["reliability"])

    if report.get("spoof", {}).get("per_type"):
        _write_rows(out_dir / "spoof_per_type.csv", report["spoof"]["per_type"])
    if report.get("temporal", {}).get("by_dataset"):
        _write_rows(out_dir / "temporal_by_dataset.csv", [
            {"dataset": k, **v} for k, v in report["temporal"]["by_dataset"].items()
        ])
    if report.get("system", {}).get("routed_curves", {}).get("roc"):
        _write_rows(out_dir / "roc_system_routed.csv",
                    report["system"]["routed_curves"]["roc"])

    written = sorted(p.name for p in out_dir.glob("*"))
    print(f"  Wrote {len(written)} file(s): {', '.join(written)}")


def test_batch(
    detector: FaceDetector,
    mtl_model,
    device: torch.device,
) -> None:
    """Evaluate the checkpoint over the held-out test split of master.csv.

    Replaces the previous "point me at a directory of videos" batch mode. The
    manifest is the ground truth the model was trained against, so every clip
    arrives with a label, a task, an attack type and a subject — which is what
    turns a batch run into a measurement instead of a list of verdicts.
    """
    ec, pc, dm = _full_cfg.eval, _full_cfg.preprocess, cfg.model

    csv_path = _ask("Manifest CSV", _full_cfg.paths.master_csv)
    try:
        clips = load_test_clips(csv_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  [ERROR] {exc}")
        return

    print(f"  {len(clips)} test clip(s) over "
          f"{len({c['video_path'] for c in clips})} video(s).")

    raw_limit = _ask("Max clips (blank = all)", "")
    if raw_limit:
        try:
            clips = _stratified_subset(clips, int(raw_limit), pc.split_seed)
            print(f"  → stratified subset of {len(clips)} clip(s).")
        except ValueError:
            print(f"  [WARN] {raw_limit!r} is not a number — evaluating all clips.")

    crop_mode = "demo" if _ask(
        "Crop window  [p=preprocessing-equivalent / d=demo aggregate box]", "p"
    ).lower().startswith("d") else "preproc"

    frames_mode = "all" if _ask(
        f"Frames to model  [s=sampled {_full_cfg.train.num_frames} / a=all]", "s"
    ).lower().startswith("a") else "sampled"

    out_dir = Path(_ask("Output directory", str(_next_eval_dir())))
    out_dir.mkdir(parents=True, exist_ok=True)

    use_flow = bool(
        _full_cfg.train.use_optical_flow
        and getattr(mtl_model.temporal_head, "flow_encoder", None) is not None
    )

    print(f"\n  crop={crop_mode}  frames={frames_mode}  flow={use_flow}  "
          f"→ {out_dir}\n")

    face_app = get_face_app(detector)
    monitor = None
    if _full_cfg.power.enable:
        monitor = power_monitor_from_config(
            _full_cfg, csv_path=str(out_dir / "power_eval.csv")
        ).start()

    # ── the actual pass ───────────────────────────────────────────────────────
    records: list[dict] = []
    skipped: list[dict] = []
    started = time.time()
    interrupted = False

    for i, clip in enumerate(clips, 1):
        try:
            rec = _score_clip(
                clip, detector, face_app, mtl_model, device,
                crop_mode=crop_mode, frames_mode=frames_mode, use_flow=use_flow,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\n  [INTERRUPTED] Reporting on the clips scored so far.\n")
            break
        except Exception as exc:                                   # noqa: BLE001
            skipped.append({**clip, "skip_reason": f"{type(exc).__name__}: {exc}"})
            continue

        (skipped if rec.get("skip_reason") else records).append(rec)

        done = i
        rate = (time.time() - started) / max(done, 1)
        print(
            f"\r  [{done}/{len(clips)}] {Path(clip['video_path']).name[:38]:38} "
            f"scored={len(records)} skipped={len(skipped)} "
            f"eta={(len(clips) - done) * rate / 60:.1f}m",
            end="", flush=True,
        )

    print()
    if monitor is not None:
        monitor.stop()

    if not records:
        print("  [ERROR] No clip could be scored — nothing to report.")
        return

    report = build_report(
        records, skipped, clips,
        crop_mode=crop_mode, frames_mode=frames_mode, use_flow=use_flow,
        csv_path=csv_path, device=device, elapsed_s=time.time() - started,
        power=monitor.summary() if monitor is not None else None,
        interrupted=interrupted,
    )
    text = print_report(report)
    write_report(out_dir, report, records, skipped, text)
    print(f"  Report → {out_dir}\n")


def sanity_check_model(model, device: torch.device) -> None:
    """
    Push one synthetic clip through the model at startup and print every head's
    raw output. This runs before any capture so a broken model path (missing
    output key, wrong checkpoint, dead head) surfaces immediately instead of
    after a 64-frame acquisition.

    Scores here are meaningless — the input is grey noise-free filler. Only the
    shapes, the keys, and whether a head is stuck are informative.
    """
    t    = cfg.model
    clip = [np.full((224, 224, 3), 128, dtype=np.uint8)
            for _ in range(_full_cfg.demo.camera.target_frames)]
    scores = run_mtl_on_frames(model, clip, device)
    _score_seen.clear()          # filler input must not pollute session extremes

    print("  Model self-check (synthetic input — scores are not predictions):")
    for name, thr in (("deepfake", t.deepfake_threshold),
                      ("spoof",    t.spoof_threshold),
                      ("temporal", t.temporal_threshold)):
        print(f"    {name:9} logit={scores[f'{name}_logit']:+8.4f}  "
              f"score={scores[name]:.4f}  threshold={thr:.2f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════╗")
    print(  "║   Face Authentication Demo  (MTL)    ║")
    print(  "╚══════════════════════════════════════╝\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    print("  Loading models…")
    # One detector, one InsightFace session: the batch evaluator needs the
    # FaceDetector wrapper (crop geometry, score gate) while the interactive
    # flows use the raw session, and a second session would hold GPU memory the
    # 4 GB card does not have to spare.
    detector  = get_face_detector()
    face_app  = get_face_app(detector)
    mtl_model = load_mtl_model(device)
    sanity_check_model(mtl_model, device)
    print("  Models loaded.\n")

    while True:
        choice = input("  [r=register | l=login | q=quit | b=batch] > ").strip().lower()
        if choice == "q":
            print("  Bye.")
            break
        elif choice == "r":
            register_flow(face_app, mtl_model, device)
        elif choice == "l":
            login_flow(face_app, mtl_model, device)
        elif choice == "b":
            test_batch(detector, mtl_model, device)
        else:
            print("  Unknown option. Use r / l / q / b.")


if __name__ == "__main__":
    main()