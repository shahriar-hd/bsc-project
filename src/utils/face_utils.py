"""
Shared InsightFace wrapper for preprocessing and the demo app.

preprocessing.py and app_demo.py each had their own FaceAnalysis setup and
crop logic; this module is the single place that talks to InsightFace so the
detector configuration stays consistent across the pipeline.

Two modes, driven by config:
  * detection only ("buffalo_sc", modules=["detection"]) — preprocessing
    quality gate; fastest.
  * detection + recognition ("buffalo_l") — demo identity matching, which
    needs `normed_embedding` per face.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.config import PreprocessConfig

Face = object  # insightface.Face — duck-typed to avoid a hard dependency here


class FaceDetector:
    """
    Wraps a lazy-loaded InsightFace FaceAnalysis session.

    Args:
        config: PreprocessConfig (used for model name, ctx, det size and
            the face-crop parameters).
        need_embeddings: If True, enable the recognition module so faces
            carry a `normed_embedding` (demo identity matching).
        det_name: Model name override — "buffalo_sc" for preprocessing,
            "buffalo_l" (default, from config) for the demo.
    """

    def __init__(
        self,
        config: PreprocessConfig,
        need_embeddings: bool = False,
        det_name: Optional[str] = None,
        providers: Optional[List[str]] = None,
        gpu_mem_mb: Optional[int] = None,
    ) -> None:
        self.min_face_score = config.min_face_score
        self.output_size: int = config.output_face_size
        self.crop_scale: float = config.crop_scale
        self.need_embeddings = need_embeddings

        self.app = None          # lazily created → cheap to instantiate
        self._app_name = det_name or (
            config.insightface_model_name if need_embeddings
            else config.insightface_det_model
        )
        self._providers = providers or [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        self._gpu_mem_mb = gpu_mem_mb
        self._config = config
        self.providers: List[str] = []   # filled in by _ensure_app

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _resolve_providers(self) -> List[str]:
        """
        Intersect the requested ONNX providers with what is actually
        installed, preserving preference order.

        Without this, requesting CUDAExecutionProvider on a CPU-only
        onnxruntime build emits a warning on every session creation.
        """
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
        except ImportError:
            return self._providers

        usable = [p for p in self._providers if p in available]
        return usable or ["CPUExecutionProvider"]

    def _provider_options(self, providers: List[str]) -> List[dict]:
        """
        One options dict per provider (ONNX Runtime requires the two lists to
        be the same length).

        The CUDA entry bounds the arena so several preprocessing workers can
        share one GPU: without a limit each session grows its arena greedily
        and the second or third worker fails to allocate.
        """
        options: List[dict] = []
        for provider in providers:
            if provider != "CUDAExecutionProvider":
                options.append({})
                continue
            cuda_opts: dict = {
                "device_id": max(0, self._config.insightface_ctx_id),
                # Only grow the arena by what is actually requested — the
                # default doubles it, which wastes scarce GPU memory.
                "arena_extend_strategy": "kSameAsRequested",
                # EXHAUSTIVE benchmarks every conv algorithm at startup and
                # allocates large scratch buffers; HEURISTIC is enough for a
                # fixed 640x640 detector input.
                "cudnn_conv_algo_search": "HEURISTIC",
            }
            if self._gpu_mem_mb:
                cuda_opts["gpu_mem_limit"] = int(self._gpu_mem_mb) * 1024 * 1024
            options.append(cuda_opts)
        return options

    def _ensure_app(self):
        if self.app is not None:
            return
        from insightface.app import FaceAnalysis

        modules = (
            ["detection", "recognition"] if self.need_embeddings
            else list(self._config.insightface_modules)
        )
        providers = self._resolve_providers()
        self.app = FaceAnalysis(
            name=self._app_name,
            allowed_modules=modules,
            providers=providers,
            provider_options=self._provider_options(providers),
        )
        # ctx_id must match the provider: InsightFace treats >=0 as GPU.
        ctx_id = self._config.insightface_ctx_id
        if not any("CUDA" in p or "Tensorrt" in p for p in providers):
            ctx_id = -1
        self.app.prepare(
            ctx_id=ctx_id,
            det_size=self._config.insightface_det_size,
        )
        self.providers = providers

    def active_providers(self) -> List[str]:
        """
        Providers the detection session actually ended up with (creates the
        session if needed) — the only reliable way to confirm GPU use, since
        a CPU-only onnxruntime silently falls back.
        """
        self._ensure_app()
        try:
            return list(self.app.det_model.session.get_providers())
        except AttributeError:
            return list(self.providers)

    def on_gpu(self) -> bool:
        return any(
            p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
            for p in self.active_providers()
        )

    def close(self) -> None:
        """Release the InsightFace session (it holds GPU memory)."""
        self.app = None

    # ── detection ────────────────────────────────────────────────────────────

    def detect_best(self, frame_bgr: np.ndarray) -> Optional[Face]:
        """
        Return the highest-confidence face with det_score >= threshold,
        or None. Among qualified faces the largest (by area) is returned.
        """
        self._ensure_app()
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        qualified = [f for f in faces if float(f.det_score) >= self.min_face_score]
        if not qualified:
            return None
        return max(qualified, key=lambda f: bbox_area(f.bbox))

    def detect_all(self, frame_bgr: np.ndarray) -> List[Face]:
        """All faces with score >= threshold (demo uses the largest anyway)."""
        self._ensure_app()
        faces = self.app.get(frame_bgr) or []
        return [f for f in faces if float(f.det_score) >= self.min_face_score]

    # ── cropping ─────────────────────────────────────────────────────────────

    def crop(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Square-crop the face region (with scale padding) and resize.

        Args:
            frame_bgr: Source BGR frame.
            bbox:      [x1, y1, x2, y2] bounding box.
        Returns:
            Resized BGR crop, or None if degenerate.
        """
        coords = crop_coords(bbox, frame_bgr.shape, self.crop_scale)
        if coords is None:
            return None
        x1, y1, x2, y2 = coords
        return cv2.resize(
            frame_bgr[y1:y2, x1:x2],
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def crop_stable(
        self,
        frame_bgr: np.ndarray,
        coords: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop with a pre-computed stable window (demo aggregate box)."""
        x1, y1, x2, y2 = coords
        return cv2.resize(
            frame_bgr[y1:y2, x1:x2],
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR,
        )


# ── geometry helpers (module-level so other modules can reuse them) ──────────

def bbox_area(bbox) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def bbox_center(bbox) -> np.ndarray:
    b = np.asarray(bbox, dtype=float)
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def bbox_width(bbox) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def displacement(a, b) -> float:
    return float(np.linalg.norm(bbox_center(a) - bbox_center(b)))


def crop_coords(
    bbox: np.ndarray,
    frame_shape: tuple[int, int],
    crop_scale: float = 1.1,
) -> Optional[tuple[int, int, int, int]]:
    """
    Square crop window centred on the bbox, padded by `crop_scale`,
    clamped to the frame.
    """
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * crop_scale / 2.0

    h, w = frame_shape[:2]
    ix1 = int(max(0.0, cx - half))
    iy1 = int(max(0.0, cy - half))
    ix2 = int(min(float(w), cx + half))
    iy2 = int(min(float(h), cy + half))

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def union_bbox(bboxes: List[np.ndarray]) -> np.ndarray:
    """Union of all boxes: [min_x1, min_y1, max_x2, max_y2]."""
    arr = np.asarray(bboxes, dtype=float)
    if arr.size == 0:
        return np.zeros(4)
    return np.array([
        arr[:, 0].min(), arr[:, 1].min(),
        arr[:, 2].max(), arr[:, 3].max(),
    ])
