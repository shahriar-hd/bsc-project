"""
Shared utility modules used by train.py, preprocessing.py and app_demo.py.

  logger_utils  → setup_logger, log_banner
  repro_utils   → set_seed, worker_init_fn
  power_utils   → PowerMonitor, power_monitor_from_config
  face_utils    → FaceDetector + bbox geometry
  video_utils   → frame reading / counting
"""

from src.utils.face_utils import (
    FaceDetector,
    bbox_area,
    bbox_center,
    bbox_width,
    crop_coords,
    displacement,
    union_bbox,
)
from src.utils.logger_utils import log_banner, setup_logger
from src.utils.power_utils import GPUSample, PowerMonitor, power_monitor_from_config
from src.utils.repro_utils import set_seed, worker_init_fn
from src.utils.video_utils import (
    get_frame_count,
    get_video_fps,
    open_video,
    read_consecutive_frames,
)

__all__ = [
    # logging
    "setup_logger", "log_banner",
    # reproducibility
    "set_seed", "worker_init_fn",
    # power
    "PowerMonitor", "power_monitor_from_config", "GPUSample",
    # face
    "FaceDetector", "bbox_area", "bbox_center", "bbox_width",
    "crop_coords", "displacement", "union_bbox",
    # video
    "open_video", "get_frame_count", "get_video_fps", "read_consecutive_frames",
]
