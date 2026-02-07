from __future__ import annotations

from typing import Dict, List

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.filters import (
    apply_blue_glove_filter,
    apply_object_size_filter,
    apply_obj_bigger_than_hand_filter,
    apply_obj_smaller_than_hand_filter,
)

LABEL_PORTABLE = "Portable Object"
LABEL_STATIONARY = "Stationary Object"
LABEL_NONE = "No Contact"

PRIORITY = {
    LABEL_PORTABLE: 0,
    LABEL_STATIONARY: 1,
    LABEL_NONE: 2,
}


def _normalize_label(label: str) -> str:
    if label is None:
        return LABEL_NONE
    if label == "Portable Object Contact":
        return LABEL_PORTABLE
    if label == "Portable Object":
        return LABEL_PORTABLE
    if label == "Stationary Object Contact":
        return LABEL_STATIONARY
    if label == "Stationary Object":
        return LABEL_STATIONARY
    if label == "No Contact":
        return LABEL_NONE
    return LABEL_NONE


def _select_label(labels: List[str]) -> str:
    if not labels:
        return LABEL_NONE
    normalized = [_normalize_label(l) for l in labels]
    return min(normalized, key=lambda l: PRIORITY[l])


def _source_hand_for_frame(hand_sides: List[str]) -> str:
    sides = {s for s in hand_sides if s in {"Left", "Right"}}
    if not sides:
        return "NA"
    if sides == {"Left", "Right"}:
        return "Both"
    return next(iter(sides))


def condense_dataframe(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Condense full detections to one row per frame.
    """
    if full_df.empty:
        return pd.DataFrame(
            columns=["frame_id", "frame_number", "contact_label", "source_hand"]
        )

    frame_rows = (
        full_df[["frame_id", "frame_number"]]
        .drop_duplicates()
        .sort_values(by=["frame_number", "frame_id"])
    )

    condensed_rows = []

    for _, frame in frame_rows.iterrows():
        frame_id = frame["frame_id"]
        frame_number = int(frame["frame_number"])

        frame_df = full_df[full_df["frame_number"] == frame_number]
        hand_df = frame_df[
            (frame_df["detection_type"] == "hand")
            & (frame_df["is_filtered"] == False)
        ]

        labels = hand_df["contact_label"].dropna().tolist()
        contact_label = _select_label(labels)

        hand_sides = hand_df["hand_side"].dropna().tolist()
        source_hand = _source_hand_for_frame(hand_sides)

        condensed_rows.append(
            {
                "frame_id": frame_id,
                "frame_number": frame_number,
                "contact_label": contact_label,
                "source_hand": source_hand,
            }
        )

    condensed_df = pd.DataFrame(condensed_rows)
    condensed_df = condensed_df.sort_values(by=["frame_number", "frame_id"]).reset_index(drop=True)
    return condensed_df


def apply_detection_filters(full_df: pd.DataFrame, config: PipelineConfig, image_dir: str) -> pd.DataFrame:
    df = full_df
    if config.blue_glove_filter:
        df = apply_blue_glove_filter(df, config, image_dir)
    if config.object_size_filter:
        df = apply_object_size_filter(df, config, image_dir)
    if config.obj_smaller_than_hand_filter:
        df = apply_obj_smaller_than_hand_filter(df, config)
    if config.obj_bigger_than_hand_filter:
        df = apply_obj_bigger_than_hand_filter(df, config)
    return df
