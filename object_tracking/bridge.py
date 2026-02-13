from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from object_tracking.geometry import bbox_iou, center_distance
from object_tracking.motion import predict_next_bbox_linear
from object_tracking.state import (
    BBox,
    TRACKING_STATE_INACTIVE,
    TRACKING_STATE_LOST,
    TRACKING_STATE_TRACKING,
    TrackState,
)
from pipeline.config import PipelineConfig

LABEL_NO_CONTACT = "No Contact"
LABEL_PORTABLE = "Portable Object"
LABEL_STATIONARY = "Stationary Object"
LABEL_STATIONARY_CONTACT = "Stationary Object Contact"


def _confidence_to_probability(value: object) -> float:
    try:
        conf = float(value)
    except Exception:
        return 0.0
    if np.isnan(conf):
        return 0.0
    if conf <= 1.0:
        return float(max(0.0, min(1.0, conf)))
    return float(max(0.0, min(1.0, conf / 100.0)))


def _probability_threshold(value: object, default: float) -> float:
    try:
        threshold = float(value)
    except Exception:
        threshold = default
    if np.isnan(threshold):
        threshold = default
    if threshold > 1.0:
        threshold = threshold / 100.0
    return float(max(0.0, min(1.0, threshold)))


def _is_unfiltered(df: pd.DataFrame) -> pd.DataFrame:
    if "is_filtered" not in df.columns:
        return df
    return df[df["is_filtered"] != True]


def _bbox_from_row(row: pd.Series) -> BBox:
    return (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def _ensure_tracking_columns(df: pd.DataFrame) -> None:
    if "tracking_promoted" not in df.columns:
        df["tracking_promoted"] = False
    if "tracking_track_id" not in df.columns:
        df["tracking_track_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "tracking_iou" not in df.columns:
        df["tracking_iou"] = np.nan
    if "tracking_missed_count" not in df.columns:
        df["tracking_missed_count"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "tracking_state" not in df.columns:
        df["tracking_state"] = TRACKING_STATE_INACTIVE
    if "tracking_bbox_x1" not in df.columns:
        df["tracking_bbox_x1"] = np.nan
    if "tracking_bbox_y1" not in df.columns:
        df["tracking_bbox_y1"] = np.nan
    if "tracking_bbox_x2" not in df.columns:
        df["tracking_bbox_x2"] = np.nan
    if "tracking_bbox_y2" not in df.columns:
        df["tracking_bbox_y2"] = np.nan
    if "tracking_bbox_source" not in df.columns:
        df["tracking_bbox_source"] = "none"


def _set_frame_tracking_state(
    df: pd.DataFrame,
    frame_index: pd.Index,
    state: str,
    track: Optional[TrackState],
) -> None:
    if len(frame_index) == 0:
        return
    df.loc[frame_index, "tracking_state"] = state
    if track is None:
        return
    df.loc[frame_index, "tracking_track_id"] = int(track.track_id)
    df.loc[frame_index, "tracking_missed_count"] = int(track.missed_count)


def _set_frame_tracking_bbox(
    df: pd.DataFrame,
    frame_index: pd.Index,
    bbox: Optional[BBox],
    source: str,
) -> None:
    if len(frame_index) == 0:
        return

    if bbox is None:
        df.loc[frame_index, "tracking_bbox_x1"] = np.nan
        df.loc[frame_index, "tracking_bbox_y1"] = np.nan
        df.loc[frame_index, "tracking_bbox_x2"] = np.nan
        df.loc[frame_index, "tracking_bbox_y2"] = np.nan
        df.loc[frame_index, "tracking_bbox_source"] = "none"
        return

    x1, y1, x2, y2 = bbox
    df.loc[frame_index, "tracking_bbox_x1"] = float(x1)
    df.loc[frame_index, "tracking_bbox_y1"] = float(y1)
    df.loc[frame_index, "tracking_bbox_x2"] = float(x2)
    df.loc[frame_index, "tracking_bbox_y2"] = float(y2)
    df.loc[frame_index, "tracking_bbox_source"] = str(source)


def _select_init_object(frame_df: pd.DataFrame, config: PipelineConfig) -> Optional[pd.Series]:
    hand_df = frame_df[frame_df["detection_type"] == "hand"]
    hand_df = _is_unfiltered(hand_df)
    hand_df = hand_df[hand_df["contact_label"] == "Portable Object"]
    if "blue_glove_status" in hand_df.columns:
        hand_df = hand_df[hand_df["blue_glove_status"] != "experimenter"]
    if hand_df.empty:
        return None

    obj_df = frame_df[frame_df["detection_type"] == "object"].copy()
    obj_df = _is_unfiltered(obj_df)
    if obj_df.empty:
        return None
    obj_df["confidence_prob"] = obj_df["confidence"].map(_confidence_to_probability)
    init_threshold = _probability_threshold(config.tracking_init_obj_confidence, default=0.70)
    obj_df = obj_df[obj_df["confidence_prob"] >= init_threshold]
    if obj_df.empty:
        return None

    best_key = None
    best_object_idx = None
    for hand_idx, hand_row in hand_df.iterrows():
        hand_bbox = _bbox_from_row(hand_row)
        for obj_idx, obj_row in obj_df.iterrows():
            obj_bbox = _bbox_from_row(obj_row)
            conf = float(obj_row["confidence_prob"])
            dist = center_distance(hand_bbox, obj_bbox)
            iou = bbox_iou(hand_bbox, obj_bbox)
            key = (-conf, dist, -iou, str(obj_idx), str(hand_idx))
            if best_key is None or key < best_key:
                best_key = key
                best_object_idx = obj_idx

    if best_object_idx is None:
        return None
    return obj_df.loc[best_object_idx]


def _best_associated_object(
    frame_df: pd.DataFrame,
    predicted_bbox: BBox,
    min_iou: float,
) -> Tuple[Optional[pd.Series], float]:
    obj_df = frame_df[frame_df["detection_type"] == "object"].copy()
    obj_df = _is_unfiltered(obj_df)
    if obj_df.empty:
        return None, 0.0

    best_idx = None
    best_iou = 0.0
    best_conf = 0.0
    for idx, row in obj_df.iterrows():
        obj_bbox = _bbox_from_row(row)
        iou = bbox_iou(predicted_bbox, obj_bbox)
        conf = _confidence_to_probability(row.get("confidence"))
        if iou > best_iou or (iou == best_iou and conf > best_conf):
            best_idx = idx
            best_iou = iou
            best_conf = conf

    if best_idx is None or best_iou < min_iou:
        return None, 0.0
    return obj_df.loc[best_idx], float(best_iou)


def _best_contact_candidate(
    frame_df: pd.DataFrame,
    tracked_bbox: BBox,
) -> Tuple[Optional[int], float]:
    return _best_contact_candidate_for_labels(
        frame_df=frame_df,
        tracked_bbox=tracked_bbox,
        labels={LABEL_NO_CONTACT},
    )


def _best_contact_candidate_for_labels(
    frame_df: pd.DataFrame,
    tracked_bbox: BBox,
    labels: set[str],
) -> Tuple[Optional[int], float]:
    hand_df = frame_df[frame_df["detection_type"] == "hand"]
    hand_df = _is_unfiltered(hand_df)
    if hand_df.empty:
        return None, 0.0

    candidate_df = hand_df[hand_df["contact_label"].isin(labels)]
    if "blue_glove_status" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["blue_glove_status"] != "experimenter"]
    if candidate_df.empty:
        return None, 0.0

    best_idx = None
    best_iou = 0.0
    for idx, row in candidate_df.iterrows():
        hand_bbox = _bbox_from_row(row)
        iou = bbox_iou(tracked_bbox, hand_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx, float(best_iou)


def _frame_has_experimenter_blue_hand(frame_df: pd.DataFrame) -> bool:
    if "blue_glove_status" not in frame_df.columns:
        return False
    hand_df = frame_df[frame_df["detection_type"] == "hand"]
    hand_df = _is_unfiltered(hand_df)
    if hand_df.empty:
        return False
    return bool((hand_df["blue_glove_status"] == "experimenter").any())


def _normalize_thresholds(config: PipelineConfig) -> Tuple[int, int, float, float, int, float]:
    max_missed = max(0, int(config.tracking_max_missed_frames))
    no_contact_confirm = max(1, int(config.tracking_promotion_confirm_frames))
    stationary_confirm = max(1, int(config.tracking_stationary_confirm_frames))
    no_contact_iou_threshold = float(max(0.0, min(1.0, config.tracking_contact_iou_threshold)))
    stationary_iou_threshold = float(max(0.0, min(1.0, config.tracking_stationary_iou_threshold)))
    reassoc_iou = float(max(0.0, min(1.0, config.tracking_reassociate_iou_threshold)))
    return (
        max_missed,
        no_contact_confirm,
        no_contact_iou_threshold,
        reassoc_iou,
        stationary_confirm,
        stationary_iou_threshold,
    )


def apply_tracking_bridge(full_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if full_df.empty:
        return full_df

    (
        max_missed,
        no_contact_confirm_frames,
        no_contact_iou_threshold,
        reassoc_iou,
        stationary_confirm_frames,
        stationary_iou_threshold,
    ) = _normalize_thresholds(config)
    df = full_df.copy()
    _ensure_tracking_columns(df)

    frame_groups = df.groupby(["frame_number", "frame_id"], sort=False).groups
    frame_keys = sorted(frame_groups.keys(), key=lambda k: (int(k[0]), str(k[1])))

    active_track: Optional[TrackState] = None
    next_track_id = 1

    for frame_key in frame_keys:
        frame_index = frame_groups[frame_key]
        frame_df = df.loc[frame_index]

        if active_track is None:
            seed_object = _select_init_object(frame_df, config)
            if seed_object is None:
                _set_frame_tracking_state(df, frame_index, TRACKING_STATE_INACTIVE, None)
                _set_frame_tracking_bbox(df, frame_index, bbox=None, source="none")
                continue
            active_track = TrackState(
                track_id=next_track_id,
                last_bbox=_bbox_from_row(seed_object),
            )
            next_track_id += 1
            _set_frame_tracking_state(df, frame_index, TRACKING_STATE_TRACKING, active_track)
            _set_frame_tracking_bbox(
                df,
                frame_index,
                bbox=active_track.last_bbox,
                source="seed_detected",
            )
            continue

        predicted_bbox = predict_next_bbox_linear(active_track.prev_bbox, active_track.last_bbox)
        associated_obj, _ = _best_associated_object(frame_df, predicted_bbox, min_iou=reassoc_iou)

        if associated_obj is not None:
            active_track.prev_bbox = active_track.last_bbox
            active_track.last_bbox = _bbox_from_row(associated_obj)
            active_track.missed_count = 0
            active_track.iou_hit_streak = 0
            active_track.no_contact_hit_streak = 0
            active_track.stationary_hit_streak = 0
            _set_frame_tracking_state(df, frame_index, TRACKING_STATE_TRACKING, active_track)
            _set_frame_tracking_bbox(
                df,
                frame_index,
                bbox=active_track.last_bbox,
                source="associated_detected",
            )
            continue

        active_track.prev_bbox = active_track.last_bbox
        active_track.last_bbox = predicted_bbox
        active_track.missed_count += 1

        if active_track.missed_count > max_missed:
            _set_frame_tracking_state(df, frame_index, TRACKING_STATE_LOST, active_track)
            _set_frame_tracking_bbox(df, frame_index, bbox=None, source="none")
            active_track = None
            continue

        _set_frame_tracking_state(df, frame_index, TRACKING_STATE_TRACKING, active_track)
        _set_frame_tracking_bbox(
            df,
            frame_index,
            bbox=active_track.last_bbox,
            source="predicted",
        )
        if _frame_has_experimenter_blue_hand(frame_df):
            active_track.iou_hit_streak = 0
            active_track.no_contact_hit_streak = 0
            active_track.stationary_hit_streak = 0
            continue

        no_contact_idx, no_contact_iou = _best_contact_candidate(
            frame_df,
            active_track.last_bbox,
        )
        if no_contact_idx is not None and no_contact_iou >= no_contact_iou_threshold:
            active_track.no_contact_hit_streak += 1
        else:
            active_track.no_contact_hit_streak = 0

        stationary_idx: Optional[int] = None
        stationary_iou = 0.0
        if config.tracking_promote_stationary:
            stationary_idx, stationary_iou = _best_contact_candidate_for_labels(
                frame_df=frame_df,
                tracked_bbox=active_track.last_bbox,
                labels={LABEL_STATIONARY, LABEL_STATIONARY_CONTACT},
            )
            if stationary_idx is not None and stationary_iou >= stationary_iou_threshold:
                active_track.stationary_hit_streak += 1
            else:
                active_track.stationary_hit_streak = 0
        else:
            active_track.stationary_hit_streak = 0

        promotable = []
        if (
            no_contact_idx is not None
            and no_contact_iou >= no_contact_iou_threshold
            and active_track.no_contact_hit_streak >= no_contact_confirm_frames
        ):
            promotable.append((no_contact_idx, no_contact_iou))

        if (
            config.tracking_promote_stationary
            and stationary_idx is not None
            and stationary_iou >= stationary_iou_threshold
            and active_track.stationary_hit_streak >= stationary_confirm_frames
        ):
            promotable.append((stationary_idx, stationary_iou))

        if not promotable:
            active_track.iou_hit_streak = 0
            continue

        promotable.sort(key=lambda x: (-x[1], str(x[0])))
        candidate_idx, overlap_iou = promotable[0]
        active_track.iou_hit_streak += 1

        df.at[candidate_idx, "contact_label"] = LABEL_PORTABLE
        df.at[candidate_idx, "contact_state"] = 3
        df.at[candidate_idx, "tracking_promoted"] = True
        df.at[candidate_idx, "tracking_track_id"] = int(active_track.track_id)
        df.at[candidate_idx, "tracking_iou"] = float(overlap_iou)
        df.at[candidate_idx, "tracking_missed_count"] = int(active_track.missed_count)
        df.at[candidate_idx, "tracking_state"] = TRACKING_STATE_TRACKING

    return df
