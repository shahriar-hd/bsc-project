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
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from insightface.app import FaceAnalysis

from src.config import get_config

# ── globals ───────────────────────────────────────────────────────────────────

cfg = get_config().demo

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

def get_face_app() -> FaceAnalysis:
    app = FaceAnalysis(
        name=cfg.paths.buffalo_model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


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

    mean_face_w = float((boxes_arr[:, 2] - boxes_arr[:, 0]).mean())
    mean_face_h = float((boxes_arr[:, 3] - boxes_arr[:, 1]).mean())

    too_much_motion = False
    for i in range(1, len(valid)):
        b_prev = boxes_arr[i - 1]
        b_curr = boxes_arr[i]

        cx_prev = (b_prev[0] + b_prev[2]) / 2
        cx_curr = (b_curr[0] + b_curr[2]) / 2
        cy_prev = (b_prev[1] + b_prev[3]) / 2
        cy_curr = (b_curr[1] + b_curr[3]) / 2

        cx_shift = abs(cx_curr - cx_prev) / mean_face_w
        cy_shift = abs(cy_curr - cy_prev) / mean_face_h

        if cx_shift > JITTER_THR or cy_shift > JITTER_THR:
            print(
                f"  [WARNING] Excessive head movement between frames "
                f"{valid[i-1]['frame_idx']} and {valid[i]['frame_idx']} "
                f"(cx_shift={cx_shift:.2f}, cy_shift={cy_shift:.2f}, "
                f"threshold={JITTER_THR}). Please keep your face still."
            )
            too_much_motion = True
            break

    if too_much_motion:
        print("  [REJECTED] Session rejected due to excessive movement. Please try again.")
        return [], [], []

    # ── Aggregate box ─────────────────────────────────────────────────────────
    # Union of all per-frame boxes so the crop region covers every detected
    # position. Example: boxes [200,160,600,500] and [205,155,605,505] and
    # [190,162,590,498] → union = [190,155,605,505].
    agg_x1 = int(boxes_arr[:, 0].min())
    agg_y1 = int(boxes_arr[:, 1].min())
    agg_x2 = int(boxes_arr[:, 2].max())
    agg_y2 = int(boxes_arr[:, 3].max())

    # Add MARGIN so forehead, ears and chin are included (same context as training).
    agg_w = agg_x2 - agg_x1
    agg_h = agg_y2 - agg_y1
    pad_x = int(agg_w * MARGIN)
    pad_y = int(agg_h * MARGIN)

    agg_x1 = max(0, agg_x1 - pad_x)
    agg_y1 = max(0, agg_y1 - pad_y)
    agg_x2 = min(W, agg_x2 + pad_x)
    agg_y2 = min(H, agg_y2 + pad_y)

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
    import numpy as np
    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np.dtype,
    ])
    checkpoint = torch.load(
        cfg.paths.model_path, map_location=device, weights_only=False
    )
    from src.train import MTLModel
    model = MTLModel(get_config())
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


def run_mtl_on_frames(
    model,
    cropped_rgb_list: list[np.ndarray],
    device: torch.device,
) -> dict:
    tensors      = [TRANSFORM(face_rgb) for face_rgb in cropped_rgb_list]
    frames_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # (1,T,C,H,W)

    with torch.no_grad():
        out = model(frames_tensor)

    return {
        "deepfake": torch.sigmoid(out["df_logit"]).item(),
        "spoof":    torch.sigmoid(out["sp_logit"]).item(),
        "temporal": torch.sigmoid(out["temp_logit"]).item(),
    }


def check_mtl_results(scores: dict) -> bool:
    t       = cfg.model
    flagged = False
    print("\n  ── MTL Detection Report ──────────────────────────")
    for label, key, threshold in [
        ("Deepfake Detection",  "deepfake", t.deepfake_threshold),
        ("Anti-Spoofing",       "spoof",    t.spoof_threshold),
        ("Temporal Consistency","temporal", t.temporal_threshold),
    ]:
        score  = scores[key]
        status = "WARNING" if score > threshold else "OK"
        print(f"  [{status}] {label}: score={score:.4f}  threshold={threshold:.2f}")
        if score > threshold:
            flagged = True
    print("  ──────────────────────────────────────────────────")
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


def test_batch(
    face_app: FaceAnalysis,
    mtl_model,
    device: torch.device,
) -> None:
    """Batch-process all video files in a given directory."""

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    input_path = input("  Batch video directory: ").strip()
    subject_id = input("  Subject ID: ").strip()

    video_dir = Path(input_path)
    if not video_dir.is_dir():
        print(f"  [ERROR] Directory not found: {video_dir}")
        return

    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in video_exts)
    if not videos:
        print("  [ERROR] No video files found in the directory.")
        return

    print(f"\n  Found {len(videos)} video(s). Starting batch..\n")

    results = []
    batch_start = time.time()

    for idx, video_path in enumerate(videos, 1):
        print(f"  [{idx}/{len(videos)}] Processing: {video_path.name}")
        t0 = time.time()

        try:
            run_dir = next_run_dir(subject_id)
            frame_paths, cropped_faces, embeddings = capture_from_file(
                str(video_path),
                run_dir,
                face_app,
            )
            passed = process_frames(
                frame_paths,
                run_dir,
                cropped_faces,
                embeddings,
                mtl_model,
                device,
            )
            elapsed = round(time.time() - t0, 3)
            status = "passed" if passed else "blocked"
            print(f"    → {status} ({elapsed}s)\n")
            results.append({
                "video":    video_path.name,
                "run_dir":  str(run_dir),
                "status":   status,
                "elapsed_s": elapsed,
                "error":    None,
            })

        except RuntimeError as exc:
            elapsed = round(time.time() - t0, 3)
            print(f"    → ERROR: {exc} ({elapsed}s)\n")
            results.append({
                "video":    video_path.name,
                "run_dir":  None,
                "status":   "error",
                "elapsed_s": elapsed,
                "error":    str(exc),
            })

    total_elapsed = round(time.time() - batch_start, 3)
    passed_count  = sum(1 for r in results if r["status"] == "passed")
    blocked_count = sum(1 for r in results if r["status"] == "blocked")
    error_count   = sum(1 for r in results if r["status"] == "error")

    summary = {
        "subject_id":   subject_id,
        "total_videos": len(videos),
        "passed":       passed_count,
        "blocked":      blocked_count,
        "errors":       error_count,
        "total_elapsed_s": total_elapsed,
        "results":      results,
    }

    out_file = video_dir / "batch_results.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("  ── Batch Summary ────────────")
    print(f"  Total   : {len(videos)}")
    print(f"  Passed  : {passed_count}")
    print(f"  Blocked : {blocked_count}")
    print(f"  Errors  : {error_count}")
    print(f"  Time    : {total_elapsed}s")
    print(f"  Report  → {out_file}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════╗")
    print(  "║   Face Authentication Demo  (MTL)    ║")
    print(  "╚══════════════════════════════════════╝\n")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"  Device : {device}")

    print("  Loading models…")
    face_app  = get_face_app()
    mtl_model = load_mtl_model(device)
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
            test_batch(face_app, mtl_model, device)
        else:
            print("  Unknown option. Use r / l / q.")


if __name__ == "__main__":
    main()