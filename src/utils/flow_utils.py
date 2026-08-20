"""
Optical flow for the temporal head — produced offline, never during training.

Flow is a **preprocessing artefact**. `preprocessing.py` writes one file per clip
next to that clip's JPEG crops; `train.py` only ever reads them. This split is
not a stylistic preference: Farneback over a 64-frame clip costs ~0.4 s, and
with `num_workers = 2` a dataloader cannot absorb that at training speed — the
GPU would idle waiting on the CPU for most of every step. Computing flow inside
`__getitem__` is therefore the one thing this module exists to prevent.

Files live at  <subject_dir>/c<clip_index>_flow.npz  and hold:

    q      int8   (N-1, 2, R, R)   quantised flow, N = saved frames in the clip
    scale  f32    ()               q * scale == flow in pixels at resolution R
    resize i16    ()               R, so the reader needs no config to agree

Pair *k* is the flow from saved frame *k* to *k+1* — one entry fewer than there
are frames, which is exactly what `TemporalHead` expects as `(B, T-1, 2, H, W)`.

Why int8: the full dataset is 2848 clips × 63 pairs. At 112×112 float32 that is
16.8 GB, and this machine has 29 GB free. A per-clip scale factor brings 56×56
down to ~126 KB/clip (~350 MB total, measured) while keeping ~1/127 of the clip's
peak displacement as the quantisation step — far below the precision a 3×3 conv
followed by `AdaptiveAvgPool2d((4, 4))` can exploit.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

# dx, dy. Kept here so ModelConfig.optical_flow_in_channels has one thing to
# match rather than a magic number repeated across three files.
FLOW_CHANNELS = 2

# Farneback parameters. Defaults from the OpenCV docs; face crops are small and
# already stabilised by the clip's shared crop window, so the pyramid does not
# need to be deep.
_FARNEBACK = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)


def flow_path_for_clip(frame_path: str | Path, clip_index: int) -> Path:
    """Where this clip's flow file lives, given any one of its frame paths.

    The naming convention lives here alone: preprocessing writes crops as
    `c{clip}_f{frame:03d}.jpg` into the subject directory, so the flow file sits
    beside them and the CSV needs no extra column to locate it.
    """
    return Path(frame_path).parent / f"c{int(clip_index)}_flow.npz"


def compute_clip_flow(
    crops: Sequence[np.ndarray],
    resize: int = 56,
    method: str = "farneback",
) -> np.ndarray:
    """Dense flow between consecutive crops → `(N-1, 2, resize, resize)` float32.

    Crops are downscaled to `resize` *before* the flow is computed, not after:
    Farneback cost is quadratic in resolution, and the encoder pools to 4×4
    after a single convolution anyway.

    Values are pixels at `resize` resolution — `load_clip_flow` converts them to
    a resolution-independent fraction.
    """
    if method != "farneback":
        raise ValueError(
            f"unsupported optical_flow_method {method!r}; only 'farneback' is "
            f"implemented (RAFT needs torchvision weights and a GPU pass)"
        )
    if len(crops) < 2:
        return np.zeros((0, FLOW_CHANNELS, resize, resize), dtype=np.float32)

    grays: List[np.ndarray] = [
        cv2.resize(
            cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (resize, resize),
            interpolation=cv2.INTER_AREA,
        )
        for c in crops
    ]

    out = np.empty((len(grays) - 1, FLOW_CHANNELS, resize, resize), dtype=np.float32)
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None, **_FARNEBACK
        )                                    # (R, R, 2)
        out[i] = flow.transpose(2, 0, 1)     # → (2, R, R)
    return out


def save_clip_flow(path: str | Path, flow: np.ndarray) -> int:
    """Quantise to int8 with one scale per clip and write. Returns bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    peak = float(np.abs(flow).max()) if flow.size else 0.0
    # A clip with no motion at all would give scale 0 and a division by zero;
    # any positive scale reproduces the all-zero array exactly.
    scale = peak / 127.0 if peak > 0.0 else 1.0
    q = np.clip(np.rint(flow / scale), -127, 127).astype(np.int8)

    resize = int(flow.shape[-1]) if flow.size else 0
    np.savez_compressed(
        path, q=q, scale=np.float32(scale), resize=np.int16(resize)
    )
    return path.stat().st_size


def load_clip_flow(path: str | Path) -> np.ndarray:
    """Read a flow file → `(N-1, 2, R, R)` float32, normalised by resolution.

    Dividing by R turns pixel displacement into a fraction of the crop's width,
    so a file written at 56 and one written at 112 present the same magnitudes
    to the encoder and `flow_resize` can change without retraining from scratch.
    """
    with np.load(path) as z:
        flow = z["q"].astype(np.float32) * float(z["scale"])
        resize = int(z["resize"]) if "resize" in z.files else 0
    if resize <= 0:
        resize = flow.shape[-1] if flow.ndim == 4 else 1
    return flow / max(resize, 1)


def accumulate_flow(pair_flow: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Re-index adjacent-pair flow onto the frames a sample actually chose.

    `MTLDataset` samples T frames with a jitter gap of 1-4, so the pair the model
    sees may span several stored pairs. Summing the intermediate flows
    approximates their composition — exact only for translation, but the error
    is second-order in displacement and a face crop moves a few pixels per frame
    at most.

    A non-increasing step (the wrap-around produced when a clip has fewer frames
    than T and positions repeat) has no meaningful flow, so it stays zero.
    """
    n_pairs, C, H, W = pair_flow.shape
    out = np.zeros((max(len(indices) - 1, 0), C, H, W), dtype=np.float32)
    for j in range(len(indices) - 1):
        a, b = int(indices[j]), int(indices[j + 1])
        # pair k spans frame k → k+1, so frames a → b need pairs a … b-1
        b = min(b, n_pairs)
        if b > a >= 0:
            out[j] = pair_flow[a:b].sum(axis=0)
    return out


def zero_flow(num_pairs: int, resize: int) -> np.ndarray:
    """Placeholder for a clip whose flow file is missing, so a batch still collates."""
    return np.zeros((max(num_pairs, 0), FLOW_CHANNELS, resize, resize), dtype=np.float32)
