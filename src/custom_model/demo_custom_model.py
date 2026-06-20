"""
demo_custom_model.py
Usage:
  python demo_custom_model.py --source camera
  python demo_custom_model.py --source folder --input data/uploads
  python demo_custom_model.py --source folder --input data/uploads --model checkpoints/best.pt
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# ── config ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "models/checkpoints/best.pt"
FACE_SIZE     = (112, 112)   # adjust to whatever your model expects
TASKS         = ["deepfake", "spoof", "stress"]  # adjust per your Config
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ──────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # support both raw state_dict and wrapped checkpoint
    model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(model, dict):
        raise ValueError("Checkpoint is a state_dict; provide the model class to load into.")
    model.eval().to(DEVICE)
    return model


def build_face_detector() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    return app


def preprocess_face(face_img: np.ndarray) -> torch.Tensor:
    """BGR HWC → normalised CHW tensor."""
    img = cv2.resize(face_img, FACE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


def predict(model, face_tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        out = model(face_tensor)

    # ── handle multi-task output (dict) ──
    if isinstance(out, dict):
        results = {}
        for task, logits in out.items():
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            results[task] = {"label": int(probs.argmax()), "conf": float(probs.max())}
        return results

    # ── single-task output (tensor) ──
    probs = torch.softmax(out, dim=-1).cpu().numpy()[0]
    return {"output": {"label": int(probs.argmax()), "conf": float(probs.max())}}


def draw_results(frame: np.ndarray, bbox, kps, results: dict) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label_parts = []
    for task, res in results.items():
        label_parts.append(f"{task}:{res['label']}({res['conf']:.2f})")
    text = "  ".join(label_parts)

    cv2.putText(frame, text, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if kps is not None:
        for pt in kps.astype(int):
            cv2.circle(frame, tuple(pt), 2, (0, 0, 255), -1)

    return frame


def process_frame(frame: np.ndarray, detector: FaceAnalysis, model) -> np.ndarray:
    faces = detector.get(frame)
    if not faces:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        # clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        tensor  = preprocess_face(crop)
        results = predict(model, tensor)
        frame   = draw_results(frame, face.bbox, face.kps, results)

    return frame


# ── sources ───────────────────────────────────
def run_camera(detector: FaceAnalysis, model):
    cap = cv2.VideoCapture(0)
    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, detector, model)
        cv2.imshow("demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_folder(folder: str, detector: FaceAnalysis, model, save_dir: str = "data/results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for vid_path in sorted(Path(folder).iterdir()):
        if vid_path.suffix.lower() not in video_exts:
            continue

        print(f"Processing: {vid_path.name}")
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = Path(save_dir) / vid_path.name
        writer   = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, detector, model)
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  frame {frame_idx}", end="\r")

        cap.release()
        writer.release()
        print(f"  saved → {out_path}  ({frame_idx} frames)")


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["camera", "folder"], default="folder")
    p.add_argument("--input",  default="data/uploads",
                   help="folder path (used when --source=folder)")
    p.add_argument("--model",  default=DEFAULT_MODEL)
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    print(f"Device : {DEVICE}")
    print(f"Model  : {args.model}")

    detector = build_face_detector()
    model    = load_model(args.model)

    if args.source == "camera":
        run_camera(detector, model)
    else:
        run_folder(args.input, detector, model)
