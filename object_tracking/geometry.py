from __future__ import annotations

import math
from typing import Tuple

from object_tracking.state import BBox


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def center_distance(box_a: BBox, box_b: BBox) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
