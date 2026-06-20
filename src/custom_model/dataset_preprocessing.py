"""
Preprocessing pipeline for CASME2, FaceForensics++, and SiW-Mv2 datasets.
Custom face detection (InsightFace) for FF++ and SiW, center-crop for CASME2.
Generates aligned face crops and a master CSV with annotations for all datasets.

@project: Bank DID Authentication System
@author: Shahriar-hd
@date: 2026-06-01
"""

import random
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from insightface.app import FaceAnalysis

from app.config import Config

random.seed(Config.DATASET_RANDOM_SEED)
np.random.seed(Config.DATASET_RANDOM_SEED)

# ──────────────────────────────────────────────
# FACE DETECTOR (InsightFace — used only for FF++ and SiW)
# ──────────────────────────────────────────────

_FACE_APP = None

def get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        app = FaceAnalysis(
            name=Config.INSIGHTFACE_MODEL,
            allowed_modules=["detection"],
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if Config.USE_GPU else ["CPUExecutionProvider"]
            ),)
        app.prepare(
            ctx_id=0 if Config.USE_GPU else -1,
            det_size=(Config.DET_SIZE, Config.DET_SIZE),
        )
        _FACE_APP = app
        print(f"[INFO] InsightFace ready ({'GPU' if Config.USE_GPU else 'CPU'})")
    return _FACE_APP


def detect_largest_face(img_bgr):
    faces = get_face_app().get(img_bgr)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return tuple(int(v) for v in largest.bbox)


def get_eye_landmarks(img_bgr):
    faces = get_face_app().get(img_bgr)
    if not faces:
        return None, None
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    if largest.kps is None:
        return None, None
    return tuple(int(v) for v in largest.kps[0]), tuple(int(v) for v in largest.kps[1])


def crop_face_insightface(img_bgr, box):
    """Crop with margin + resize using InsightFace bbox."""
    x1, y1, x2, y2 = box
    h, w = img_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * Config.FACE_MARGIN), int(bh * Config.FACE_MARGIN)
    x1 = max(0, x1 - mx);  y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx);  y2 = min(h, y2 + my)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (Config.FACE_SIZE, Config.FACE_SIZE), interpolation=cv2.INTER_AREA)


def process_frame_insightface(img_bgr, align=True):
    """InsightFace-based detection + optional alignment + crop."""
    box = detect_largest_face(img_bgr)
    if box is None:
        return None
    if align:
        le, re = get_eye_landmarks(img_bgr)
        if le and re:
            angle = math.degrees(math.atan2(re[1]-le[1], re[0]-le[0]))
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]))
            box = detect_largest_face(img_bgr) or box
    return crop_face_insightface(img_bgr, box)


# ──────────────────────────────────────────────
# CASME2: Center-crop (no InsightFace)
# ──────────────────────────────────────────────

def center_crop_and_resize(img_bgr):
    """
    Center-crop a square of SIW_CROP_SIZE from the image,
    then resize to FACE_SIZE x FACE_SIZE.
    Works for controlled datasets where the subject is centered.
    """
    h, w = img_bgr.shape[:2]
    crop_size = Config.SIW_CROP_SIZE  # reused for CASME2 as well (480x640 → 450x450)
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    y1 = max(0, cy - half);  y2 = y1 + crop_size
    x1 = max(0, cx - half);  x2 = x1 + crop_size
    # clamp if image is smaller than crop_size
    y2 = min(h, y2);  x2 = min(w, x2)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (Config.FACE_SIZE, Config.FACE_SIZE), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def save_face(face_img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), face_img)


def extract_clip_from_video(video_path, out_dir, sample_every, clip_len, align=False):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total == 0:
        return []

    sampled = list(range(0, total, sample_every))
    if len(sampled) >= clip_len:
        start = random.randint(0, len(sampled) - clip_len)
        sampled = sampled[start: start + clip_len]

    frame_set = set(sampled)
    cap = cv2.VideoCapture(str(video_path))
    saved, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_set:
            face = process_frame_insightface(frame, align=align)
            if face is not None:
                out_path = out_dir / f"frame_{idx:05d}.png"
                save_face(face, out_path)
                saved.append(str(out_path))
        idx += 1
    cap.release()
    return saved


# ──────────────────────────────────────────────
# SPLIT HELPERS
# ──────────────────────────────────────────────

def subject_independent_split(subjects):
    s = sorted(subjects)
    n = len(s)
    n_val = round(n * Config.VAL_RATIO)
    test  = s[::7]
    val   = [x for x in s if x not in set(test)][::6][:n_val]
    train = [x for x in s if x not in set(test) and x not in set(val)]
    return train, val, test


def video_level_split(video_paths):
    paths = list(video_paths)
    random.shuffle(paths)
    n = len(paths)
    n_train = round(n * Config.TRAIN_RATIO)
    n_val   = round(n * Config.VAL_RATIO)
    return paths[:n_train], paths[n_train:n_train+n_val], paths[n_train+n_val:]


# ──────────────────────────────────────────────
# DATASET PROCESSORS
# ──────────────────────────────────────────────

def process_casme2(raw_dir, out_base, xlsx_path):
    """
    CASME2: images are 640x480, subjects centered.
    Uses center_crop_and_resize() — no InsightFace.
    """
    print("\n[CASME2] Starting preprocessing...")
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names
    col_map = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "")
        if "subject" in lc:    col_map[c] = "Subject"
        elif "filename" in lc: col_map[c] = "Filename"
        elif "onset" in lc:    col_map[c] = "OnsetFrame"
        elif "apex" in lc:     col_map[c] = "ApexFrame"
        elif "offset" in lc:   col_map[c] = "OffsetFrame"
    df.rename(columns=col_map, inplace=True)

    subjects = sorted(df["Subject"].astype(str).str.zfill(2).unique().tolist())
    train_subs, val_subs, test_subs = subject_independent_split(subjects)
    split_map = {s: "train" for s in train_subs}
    split_map.update({s: "val" for s in val_subs})
    split_map.update({s: "test" for s in test_subs})
    print(f"[CASME2] Subjects -> train:{len(train_subs)} val:{len(val_subs)} test:{len(test_subs)}")

    records = []
    for _, row in df.iterrows():
        subj   = str(row["Subject"]).zfill(2)
        fname  = str(row["Filename"]).strip()
        onset  = int(row["OnsetFrame"])
        apex   = int(row["ApexFrame"])
        offset = int(row["OffsetFrame"])
        split  = split_map.get(subj, "train")

        seq_dir = Path(raw_dir) / f"sub{subj}" / fname
        if not seq_dir.exists():
            print(f"  [WARN] Missing: {seq_dir}")
            continue

        img_files = sorted(seq_dir.glob("img*.jpg"))
        if not img_files:
            continue

        start_f = max(1, onset - Config.CASME_NEUTRAL_PAD)
        end_f   = min(len(img_files), offset + Config.CASME_NEUTRAL_PAD)
        out_seq_dir = out_base / "casme2" / split / f"sub{subj}" / fname
        saved_count = 0

        for img_path in img_files:
            try:
                fidx = int(''.join(filter(str.isdigit, img_path.stem)))
            except ValueError:
                continue
            if fidx < start_f or fidx > end_f:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # ← Center-crop instead of InsightFace
            face = center_crop_and_resize(img)
            if face is None:
                continue

            out_path = out_seq_dir / f"frame_{fidx:04d}.png"
            save_face(face, out_path)
            records.append({
                "image_path":          str(out_path),
                "subject_id":          f"sub{subj}",
                "dataset":             "casme2",
                "split":               split,
                "deepfake_label":      -1,
                "physical_spoof_label":-1,
                "stress_label":        1 if onset <= fidx <= offset else 0,
                "sequence_id":         fname,
                "frame_idx":           fidx,
                "is_apex":             int(fidx == apex),
            })
            saved_count += 1
            print(f"  [CASME2] sub{subj}/{fname}: {saved_count} frames (split={split})")

    print(f"[CASME2] Done. Total records: {len(records)}")
    return records


def process_faceforensics(ff_dir, out_base):
    print("\n[FaceForensics++] Starting preprocessing...")
    records = []
    for label_name, df_label in [("real", 0), ("fake", 1)]:
        label_dir = Path(ff_dir) / label_name
        if not label_dir.exists():
            print(f"  [WARN] Missing: {label_dir}")
            continue
        videos = list(label_dir.glob("**/*.mp4")) + list(label_dir.glob("**/*.avi"))
        train_v, val_v, test_v = video_level_split(videos)
        split_map = {str(v): "train" for v in train_v}
        split_map.update({str(v): "val" for v in val_v})
        split_map.update({str(v): "test" for v in test_v})
        for vpath in videos:
            split    = split_map[str(vpath)]
            vid_name = vpath.stem
            out_dir  = out_base / "faceforensics" / split / label_name / vid_name
            saved    = extract_clip_from_video(
                vpath, out_dir, Config.FF_SAMPLE_EVERY, Config.CLIP_LENGTH, align=False
            )
            for p in saved:
                records.append({
                    "image_path": p, "subject_id": vid_name,
                    "dataset": "faceforensics", "split": split,
                    "deepfake_label": df_label, "physical_spoof_label": -1,
                    "stress_label": -1, "sequence_id": vid_name,
                    "frame_idx": int(Path(p).stem.split("_")[1]), "is_apex": -1,
                })
            print(f"  [FF++] {label_name}/{vid_name}: {len(saved)} frames")
    print(f"[FaceForensics++] Done. Total records: {len(records)}")
    return records


def process_siwmv2(siw_dir, out_base):
    print("\n[SiW-Mv2] Starting preprocessing...")
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    records = []

    live_dir    = Path(siw_dir) / "live"
    live_videos = [p for p in live_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS] if live_dir.exists() else []

    spoof_dir, spoof_videos, spoof_type_map = Path(siw_dir) / "spoof", [], {}
    if spoof_dir.exists():
        for attack_dir in spoof_dir.iterdir():
            if attack_dir.is_dir():
                for vp in attack_dir.rglob("*"):
                    if vp.suffix.lower() in VIDEO_EXTS:
                        spoof_videos.append(vp)
                        spoof_type_map[str(vp)] = attack_dir.name

    all_videos = [(v, 0) for v in live_videos] + [(v, 1) for v in spoof_videos]
    random.shuffle(all_videos)
    n       = len(all_videos)
    n_train = round(n * Config.TRAIN_RATIO)
    n_val   = round(n * Config.VAL_RATIO)
    split_map = {
        str(v): ("train" if i < n_train else "val" if i < n_train+n_val else "test")
        for i, (v, _) in enumerate(all_videos)
    }

    for vpath, spoof_label in all_videos:
        split    = split_map[str(vpath)]
        vid_name = vpath.stem
        attack   = spoof_type_map.get(str(vpath), "live")
        out_dir  = out_base / "siwmv2" / split / attack / vid_name
        saved    = extract_clip_from_video(
            vpath, out_dir, Config.SIW_SAMPLE_EVERY, Config.CLIP_LENGTH, align=False
        )
        for p in saved:
            records.append({
                "image_path": p, "subject_id": vid_name,
                "dataset": "siwmv2", "split": split,
                "deepfake_label": -1, "physical_spoof_label": spoof_label,
                "stress_label": -1, "sequence_id": vid_name,
                "frame_idx": int(Path(p).stem.split("_")[1]), "is_apex": -1,
            })
        print(f"  [SiW] {attack}/{vid_name}: {len(saved)} frames (split={split})")
    print(f"[SiW-Mv2] Done. Total records: {len(records)}")
    return records


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    out_base = Path(Config.PROCESSED_DATASET_PATH)
    ann_dir  = Path(Config.ANNOTATIONS_PATH)
    ann_dir.mkdir(parents=True, exist_ok=True)

    all_records  = process_casme2(Config.CASME2_DATASET_PATH, out_base, Config.CASME2_XLSX_PATH)
    all_records += process_faceforensics(Config.FACEFORENSICS_DATASET_PATH, out_base)
    all_records += process_siwmv2(Config.SIW_DATASET_PATH, out_base)

    if not all_records:
        print("[ERROR] No records generated. Check your input paths.")
        return

    master_df  = pd.DataFrame(all_records)
    master_csv = ann_dir / "master.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\n[DONE] Master CSV: {master_csv}  ({len(master_df)} rows)")

    for dataset in master_df["dataset"].unique():
        for split in ["train", "val", "test"]:
            sub = master_df[(master_df["dataset"] == dataset) & (master_df["split"] == split)]
            if len(sub):
                sub.to_csv(ann_dir / f"{dataset}_{split}.csv", index=False)

    print("\n── Summary ──")
    for dataset in master_df["dataset"].unique():
        for split in ["train", "val", "test"]:
            n = len(master_df[(master_df["dataset"] == dataset) & (master_df["split"] == split)])
            print(f"  {dataset:20s} {split:6s}: {n}")


if __name__ == "__main__":
    main()
