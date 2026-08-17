"""
Video reading helpers shared by preprocessing and the demo app.

Thin wrappers over cv2.VideoCapture that always release the handle and
never raise on unreadable files (a corrupt video should skip, not crash a
multi-hour preprocessing run).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Tuple

import cv2
import numpy as np

logger = logging.getLogger("MTL")


@contextmanager
def open_video(video_path: Path) -> Iterator[cv2.VideoCapture | None]:
    """Yield an opened VideoCapture (or None) and always release it."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        yield cap if cap.isOpened() else None
    finally:
        cap.release()


def get_frame_count(video_path: Path) -> int:
    """Total frames, or 0 if the video cannot be opened."""
    with open_video(video_path) as cap:
        if cap is None:
            return 0
        return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


def get_video_fps(video_path: Path) -> float:
    """Frames per second, or 0.0 if unavailable."""
    with open_video(video_path) as cap:
        if cap is None:
            return 0.0
        return max(0.0, float(cap.get(cv2.CAP_PROP_FPS)))


def read_consecutive_frames(
    video_path: Path,
    start_frame: int,
    count: int,
    step: int = 1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Read `count` frames starting at `start_frame`, keeping every `step`-th one.

    The capture is advanced sequentially and skipped frames are decoded but
    discarded — seeking per frame is slower and unreliable on B-frame heavy
    encodes, which is exactly what both datasets use.

    Args:
        video_path:  Video to read.
        start_frame: Absolute index of the first frame to yield.
        count:       How many frames to yield (not how many to decode).
        step:        Source-frame spacing between yielded frames (1 = every frame).

    Yields:
        (absolute_frame_index, bgr_array)
    """
    step = max(1, int(step))

    with open_video(video_path) as cap:
        if cap is None:
            logger.warning("Cannot open: %s", video_path)
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        yielded = 0
        offset = 0
        while yielded < count:
            ret, frame = cap.read()
            if not ret:
                break
            if offset % step == 0:
                yield start_frame + offset, frame
                yielded += 1
            offset += 1
