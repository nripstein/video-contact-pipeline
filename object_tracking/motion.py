from __future__ import annotations

from typing import Optional

from object_tracking.geometry import bbox_center, bbox_size
from object_tracking.state import BBox


def predict_next_bbox_linear(prev_bbox: Optional[BBox], last_bbox: BBox) -> BBox:
    if prev_bbox is None:
        return last_bbox

    prev_cx, prev_cy = bbox_center(prev_bbox)
    last_cx, last_cy = bbox_center(last_bbox)
    vx = last_cx - prev_cx
    vy = last_cy - prev_cy

    pred_cx = last_cx + vx
    pred_cy = last_cy + vy
    width, height = bbox_size(last_bbox)
    half_w = width / 2.0
    half_h = height / 2.0

    return (
        pred_cx - half_w,
        pred_cy - half_h,
        pred_cx + half_w,
        pred_cy + half_h,
    )
