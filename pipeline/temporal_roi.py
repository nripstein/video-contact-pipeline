from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass
class TemporalROIState:
    last_known_stimulus_bbox: Optional[BBox] = None
    missed_count: int = 0


def is_valid_bbox(bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def clip_bbox_to_image(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    return x1, y1, x2, y2


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
    return float(inter / denom)


def select_stimulus_bbox(obj_dets: Optional[np.ndarray], prev_bbox: Optional[BBox]) -> Optional[BBox]:
    if obj_dets is None or len(obj_dets) == 0:
        return None

    candidates = []
    for row in obj_dets:
        bbox = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        if not is_valid_bbox(bbox):
            continue
        score = float(row[4]) if row.shape[0] > 4 else 0.0
        iou = bbox_iou(bbox, prev_bbox) if prev_bbox is not None else 0.0
        candidates.append((bbox, score, iou))

    if not candidates:
        return None

    if prev_bbox is None:
        candidates.sort(key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2], item[0][3]))
        return candidates[0][0]

    candidates.sort(key=lambda item: (-item[2], -item[1], item[0][0], item[0][1], item[0][2], item[0][3]))
    return candidates[0][0]


class TemporalROIPropagator:
    def __init__(
        self,
        detect_once_fn: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, Optional[np.ndarray]]],
        blue_guard_fn: Callable[[np.ndarray, BBox], bool],
        max_missed_frames: int = 8,
    ):
        self.detect_once_fn = detect_once_fn
        self.blue_guard_fn = blue_guard_fn
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.state = TemporalROIState()

    def reset(self) -> None:
        self.state = TemporalROIState()

    def _clear_state(self) -> None:
        self.state.last_known_stimulus_bbox = None
        self.state.missed_count = 0

    def _on_miss(self) -> None:
        self.state.missed_count += 1
        if self.state.missed_count >= self.max_missed_frames:
            self._clear_state()

    def _can_propagate(self) -> bool:
        return (
            self.max_missed_frames > 0
            and self.state.last_known_stimulus_bbox is not None
            and self.state.missed_count < self.max_missed_frames
        )

    def detect(self, im: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        dets = self.detect_once_fn(im, None)
        obj_dets = dets.get("obj_dets")
        picked = select_stimulus_bbox(obj_dets, self.state.last_known_stimulus_bbox)
        if picked is not None:
            if not self.blue_guard_fn(im, picked):
                self.state.last_known_stimulus_bbox = picked
                self.state.missed_count = 0
            return dets

        if not self._can_propagate():
            return dets

        propagated_bbox = self.state.last_known_stimulus_bbox
        if propagated_bbox is None:
            return dets

        if self.blue_guard_fn(im, propagated_bbox):
            self._on_miss()
            return dets

        dets_temporal = self.detect_once_fn(im, np.asarray([propagated_bbox], dtype=np.float32))
        obj_dets_temporal = dets_temporal.get("obj_dets")
        picked_temporal = select_stimulus_bbox(obj_dets_temporal, propagated_bbox)
        if picked_temporal is not None and not self.blue_guard_fn(im, picked_temporal):
            self.state.last_known_stimulus_bbox = picked_temporal
            self.state.missed_count = 0
        else:
            self._on_miss()

        return dets_temporal
