from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig


def get_blue_bbox_proportion(img: np.ndarray, bbox: List[int]) -> float:
    """
    Args:
        img (np.ndarray): BGR from OpenCV
        bbox (list[int]): Of the form [left, top, right, bottom]
    """
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the "blue" color in HSV
    LOWER_BLUE = np.array([90, 50, 50])
    UPPER_BLUE = np.array([130, 255, 255])
    # Create a mask that isolates the pixels within the specified blue range
    blue_mask = cv2.inRange(hsv_img, LOWER_BLUE, UPPER_BLUE)
    # Extract the bounding box area
    bbox_area = blue_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Count the number of blue pixels within the bounding box
    blue_pixels = np.count_nonzero(bbox_area)
    # Calculate the total area of the bounding box
    total_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    # Calculate the percentage of blue pixels within the bounding box
    blue_proportion = blue_pixels / total_area
    return blue_proportion


def _normalize_contact_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    if label in ("Portable Object", "Portable Object Contact"):
        return "Portable Object"
    if label in ("Stationary Object", "Stationary Object Contact"):
        return "Stationary Object"
    if label == "No Contact":
        return "No Contact"
    return None


def _clip_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def apply_blue_glove_filter(full_df: pd.DataFrame, config: PipelineConfig, image_dir: str) -> pd.DataFrame:
    """
    Relabels hand detections as "No Contact" if the hand bbox contains a high proportion of blue pixels.
    """
    if full_df.empty:
        return full_df

    df = full_df.copy()
    image_dir = str(Path(image_dir).expanduser())

    if "contact_label_raw" not in df.columns:
        df["contact_label_raw"] = df["contact_label"]
    if "contact_state_raw" not in df.columns:
        df["contact_state_raw"] = df["contact_state"]

    if "blue_prop" not in df.columns:
        df["blue_prop"] = None
    if "blue_glove_status" not in df.columns:
        df["blue_glove_status"] = "NA"

    image_cache: Dict[str, Optional[np.ndarray]] = {}

    for idx, row in df.iterrows():
        if row.get("detection_type") != "hand":
            continue
        if row.get("is_filtered") is True:
            continue

        frame_id = row.get("frame_id")
        if frame_id is None:
            continue

        if frame_id not in image_cache:
            img_path = Path(image_dir) / str(frame_id)
            image_cache[frame_id] = cv2.imread(str(img_path))

        img = image_cache[frame_id]
        if img is None:
            continue

        h, w = img.shape[:2]
        bbox = (
            int(row.get("bbox_x1")),
            int(row.get("bbox_y1")),
            int(row.get("bbox_x2")),
            int(row.get("bbox_y2")),
        )
        x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
        if x2 <= x1 or y2 <= y1:
            df.at[idx, "blue_prop"] = 0.0
            df.at[idx, "blue_glove_status"] = "participant"
            continue

        blue_prop = get_blue_bbox_proportion(img, [x1, y1, x2, y2])
        df.at[idx, "blue_prop"] = blue_prop

        if blue_prop >= config.blue_threshold:
            df.at[idx, "blue_glove_status"] = "experimenter"
            df.at[idx, "contact_label"] = "No Contact"
            df.at[idx, "contact_state"] = 0
        else:
            df.at[idx, "blue_glove_status"] = "participant"

    return df


def apply_object_size_filter(full_df: pd.DataFrame, config: PipelineConfig, image_dir: str) -> pd.DataFrame:
    """
    Marks oversized object detections as filtered based on bbox area ratio.
    """
    if full_df.empty:
        return full_df

    df = full_df.copy()
    image_dir = str(Path(image_dir).expanduser())

    if "is_filtered" not in df.columns:
        df["is_filtered"] = False
    if "filtered_by" not in df.columns:
        df["filtered_by"] = ""
    if "filtered_reason" not in df.columns:
        df["filtered_reason"] = ""

    frame_size_cache: Dict[str, Optional[tuple]] = {}

    for idx, row in df.iterrows():
        if row.get("detection_type") != "object":
            continue
        if row.get("is_filtered") is True:
            continue

        frame_id = row.get("frame_id")
        if frame_id is None:
            continue

        if frame_id not in frame_size_cache:
            img_path = Path(image_dir) / str(frame_id)
            img = cv2.imread(str(img_path))
            if img is None:
                frame_size_cache[frame_id] = None
            else:
                h, w = img.shape[:2]
                frame_size_cache[frame_id] = (w, h)

        size = frame_size_cache[frame_id]
        if size is None:
            continue
        width, height = size
        frame_area = width * height
        if frame_area <= 0:
            continue

        x1 = int(row.get("bbox_x1"))
        y1 = int(row.get("bbox_y1"))
        x2 = int(row.get("bbox_x2"))
        y2 = int(row.get("bbox_y2"))
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_ratio = bbox_area / frame_area

        if area_ratio > config.object_size_max_area_ratio:
            df.at[idx, "is_filtered"] = True
            df.at[idx, "filtered_by"] = "object_size_filter"
            df.at[idx, "filtered_reason"] = f"bbox_area_ratio>{config.object_size_max_area_ratio}"

    return df


def _bbox_center(x1: int, y1: int, x2: int, y2: int):
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def _bbox_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = _bbox_area(ax1, ay1, ax2, ay2)
    area_b = _bbox_area(bx1, by1, bx2, by2)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _offset_distance(hand_row, obj_row) -> float:
    hx1, hy1, hx2, hy2 = int(hand_row.get("bbox_x1")), int(hand_row.get("bbox_y1")), int(hand_row.get("bbox_x2")), int(hand_row.get("bbox_y2"))
    ox1, oy1, ox2, oy2 = int(obj_row.get("bbox_x1")), int(obj_row.get("bbox_y1")), int(obj_row.get("bbox_x2")), int(obj_row.get("bbox_y2"))
    hand_cc = np.array(_bbox_center(hx1, hy1, hx2, hy2))
    obj_cc = np.array(_bbox_center(ox1, oy1, ox2, oy2))

    mag = hand_row.get("offset_mag")
    ox = hand_row.get("offset_x")
    oy = hand_row.get("offset_y")
    try:
        mag = 0.0 if pd.isna(mag) else float(mag)
        ox = 0.0 if pd.isna(ox) else float(ox)
        oy = 0.0 if pd.isna(oy) else float(oy)
    except Exception:
        mag, ox, oy = 0.0, 0.0, 0.0

    point_cc = np.array([
        hand_cc[0] + mag * 10000 * ox,
        hand_cc[1] + mag * 10000 * oy,
    ])
    dist = np.sum((obj_cc - point_cc) ** 2, axis=0)
    return float(dist)


def _prepare_hand_object_ratio_columns(df: pd.DataFrame) -> None:
    if "contact_label_raw" not in df.columns:
        df["contact_label_raw"] = df["contact_label"]
    if "contact_state_raw" not in df.columns:
        df["contact_state_raw"] = df["contact_state"]

    for col in ["matched_object_conf", "matched_object_area", "hand_area", "area_ratio", "obj_match_method"]:
        if col not in df.columns:
            df[col] = None


def _best_matched_object_for_hand(hand_row, obj_rows: pd.DataFrame):
    candidates = []
    for _, obj in obj_rows.iterrows():
        conf = float(obj.get("confidence") or 0.0)
        dist = _offset_distance(hand_row, obj)
        hand_box = (
            int(hand_row.get("bbox_x1")),
            int(hand_row.get("bbox_y1")),
            int(hand_row.get("bbox_x2")),
            int(hand_row.get("bbox_y2")),
        )
        obj_box = (
            int(obj.get("bbox_x1")),
            int(obj.get("bbox_y1")),
            int(obj.get("bbox_x2")),
            int(obj.get("bbox_y2")),
        )
        iou = _iou(hand_box, obj_box)
        area = _bbox_area(*obj_box)
        candidates.append((obj, conf, dist, iou, area, obj_box))

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (
            -c[1],
            c[2],
            -c[3],
            -c[4],
            c[5][0],
            c[5][1],
            c[5][2],
            c[5][3],
        )
    )
    return candidates[0]


def _write_match_details(df: pd.DataFrame, idx: int, conf: float, obj_area: int, hand_area: int) -> None:
    ratio = (obj_area / hand_area) if hand_area > 0 else None
    df.at[idx, "matched_object_conf"] = conf
    df.at[idx, "matched_object_area"] = obj_area
    df.at[idx, "hand_area"] = hand_area
    df.at[idx, "area_ratio"] = ratio
    df.at[idx, "obj_match_method"] = "offset"


def apply_obj_bigger_than_hand_filter(full_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Relabels portable-object hand detections as No Contact when matched object is larger by ratio.
    """
    if full_df.empty:
        return full_df

    df = full_df.copy()
    _prepare_hand_object_ratio_columns(df)

    for frame_id, frame_df in df.groupby("frame_id"):
        obj_rows = frame_df[(frame_df["detection_type"] == "object") & (frame_df["is_filtered"] == False)]
        if obj_rows.empty:
            continue

        for idx, row in frame_df.iterrows():
            if row.get("detection_type") != "hand":
                continue
            if row.get("is_filtered") is True:
                continue
            if row.get("contact_label") != "Portable Object":
                continue

            best = _best_matched_object_for_hand(row, obj_rows)
            if best is None:
                continue
            _, conf, _, _, obj_area, _ = best

            hx1, hy1, hx2, hy2 = int(row.get("bbox_x1")), int(row.get("bbox_y1")), int(row.get("bbox_x2")), int(row.get("bbox_y2"))
            hand_area = _bbox_area(hx1, hy1, hx2, hy2)
            _write_match_details(df, idx, conf, obj_area, hand_area)

            if hand_area > 0 and obj_area > hand_area * config.obj_bigger_ratio_k:
                df.at[idx, "contact_label"] = "No Contact"
                df.at[idx, "contact_state"] = 0

    return df


def apply_obj_smaller_than_hand_filter(full_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Relabels portable-object hand detections as No Contact when matched object is too large by ratio.
    """
    if full_df.empty:
        return full_df

    df = full_df.copy()
    _prepare_hand_object_ratio_columns(df)
    if "small_object_rule_applied" not in df.columns:
        df["small_object_rule_applied"] = False

    for frame_id, frame_df in df.groupby("frame_id"):
        obj_rows = frame_df[(frame_df["detection_type"] == "object") & (frame_df["is_filtered"] == False)]
        if obj_rows.empty:
            continue

        for idx, row in frame_df.iterrows():
            if row.get("detection_type") != "hand":
                continue
            if row.get("is_filtered") is True:
                continue
            if row.get("contact_label") != "Portable Object":
                continue

            best = _best_matched_object_for_hand(row, obj_rows)
            if best is None:
                continue
            _, conf, _, _, obj_area, _ = best

            hx1, hy1, hx2, hy2 = int(row.get("bbox_x1")), int(row.get("bbox_y1")), int(row.get("bbox_x2")), int(row.get("bbox_y2"))
            hand_area = _bbox_area(hx1, hy1, hx2, hy2)
            _write_match_details(df, idx, conf, obj_area, hand_area)

            if hand_area > 0 and obj_area > hand_area * config.obj_smaller_ratio_factor:
                df.at[idx, "contact_label"] = "No Contact"
                df.at[idx, "contact_state"] = 0
                df.at[idx, "small_object_rule_applied"] = True

    return df
