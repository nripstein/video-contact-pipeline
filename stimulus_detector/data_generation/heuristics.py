from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.types import HandDetection, ObjectDetection, PseudoLabel


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0

    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return (inter / union) if union > 0 else 0.0


def _intersection_area(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _center_distance(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def match_hand_detection(
    obj_det: ObjectDetection,
    hand_detections: List[HandDetection],
) -> Tuple[Optional[HandDetection], Optional[float], Optional[float]]:
    if not hand_detections:
        return None, None, None

    ranked = []
    for hand in hand_detections:
        iou = _iou(obj_det.bbox_xyxy, hand.bbox_xyxy)
        dist = _center_distance(obj_det.bbox_xyxy, hand.bbox_xyxy)
        ranked.append((hand, iou, dist))

    ranked.sort(key=lambda item: (-item[1], item[2]))
    best_hand, best_iou, best_dist = ranked[0]
    return best_hand, float(best_iou), float(best_dist)


def apply_heuristic_filters(
    object_detections: List[ObjectDetection],
    hand_detections: List[HandDetection],
    config: Phase1Config,
) -> Tuple[List[PseudoLabel], List[Dict[str, object]]]:
    hands_by_frame: Dict[str, List[HandDetection]] = {}
    for hand in hand_detections:
        hands_by_frame.setdefault(hand.frame.frame_path, []).append(hand)

    kept_labels: List[PseudoLabel] = []
    audit_rows: List[Dict[str, object]] = []

    for obj in object_detections:
        frame_path = obj.frame.frame_path
        frame_hands = hands_by_frame.get(frame_path, [])

        reasons: List[str] = []

        if obj.confidence < config.min_confidence:
            reasons.append("low_confidence")

        aspect_ratio = obj.aspect_ratio()
        if not (config.aspect_ratio_min <= aspect_ratio <= config.aspect_ratio_max):
            reasons.append("aspect_ratio_out_of_range")

        matched_hand, matched_iou, matched_dist = match_hand_detection(obj, frame_hands)
        matched_hand_area = matched_hand.area() if matched_hand else None
        obj_area = obj.area()
        object_to_hand_area_ratio = None
        hand_overlap_ratio = None

        if config.require_hand_for_size and matched_hand is None:
            reasons.append("no_hand_in_frame")
        elif matched_hand is not None:
            safe_hand_area = max(1e-6, float(matched_hand_area))
            safe_obj_area = max(1e-6, float(obj_area))
            object_to_hand_area_ratio = obj_area / safe_hand_area
            if object_to_hand_area_ratio >= config.max_object_to_hand_area_ratio:
                reasons.append("object_too_large_relative_to_hand")

            inter_area = _intersection_area(obj.bbox_xyxy, matched_hand.bbox_xyxy)
            hand_overlap_ratio = inter_area / safe_obj_area
            if hand_overlap_ratio > config.max_hand_occlusion_ratio:
                reasons.append("object_too_occluded_by_hand")

        passed = len(reasons) == 0
        audit_rows.append(
            {
                "participant_id": obj.frame.participant_id,
                "video_id": obj.frame.video_id,
                "frame_idx": obj.frame.frame_idx,
                "frame_time_sec": obj.frame.frame_time_sec,
                "fps": obj.frame.fps,
                "frame_path": obj.frame.frame_path,
                "frame_name": Path(obj.frame.frame_path).name,
                "frame_width": obj.frame.width,
                "frame_height": obj.frame.height,
                "bbox_x1": obj.bbox_xyxy[0],
                "bbox_y1": obj.bbox_xyxy[1],
                "bbox_x2": obj.bbox_xyxy[2],
                "bbox_y2": obj.bbox_xyxy[3],
                "confidence": obj.confidence,
                "aspect_ratio": aspect_ratio,
                "object_area": obj_area,
                "matched_hand_area": matched_hand_area,
                "object_to_hand_area_ratio": object_to_hand_area_ratio,
                "matched_hand_iou": matched_iou,
                "matched_hand_center_dist": matched_dist,
                "hand_overlap_ratio": hand_overlap_ratio,
                "passed_heuristics": passed,
                "heuristics_fail_reasons": "|".join(reasons),
            }
        )

        if passed:
            kept_labels.append(
                PseudoLabel(
                    frame=obj.frame,
                    bbox_xyxy=obj.bbox_xyxy,
                    confidence=obj.confidence,
                    source=obj.source,
                    matched_hand_area=matched_hand_area,
                    matched_hand_iou=matched_iou,
                    matched_hand_center_dist=matched_dist,
                    filter_trace=["confidence", "aspect_ratio", "hand_size"],
                )
            )

    return kept_labels, audit_rows
