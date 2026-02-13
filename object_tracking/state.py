from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


BBox = Tuple[float, float, float, float]

TRACKING_STATE_INACTIVE = "inactive"
TRACKING_STATE_TRACKING = "tracking"
TRACKING_STATE_LOST = "lost"


@dataclass
class TrackState:
    track_id: int
    last_bbox: BBox
    prev_bbox: Optional[BBox] = None
    missed_count: int = 0
    iou_hit_streak: int = 0
    no_contact_hit_streak: int = 0
    stationary_hit_streak: int = 0
