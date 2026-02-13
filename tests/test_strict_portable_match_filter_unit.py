from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import PipelineConfig
from pipeline.filters import apply_strict_portable_match_filter
from pipeline.postprocessing import condense_dataframe


def _cfg(**kwargs) -> PipelineConfig:
    defaults = {
        "input_path": "dummy",
        "output_dir": "dummy_out",
        "strict_portable_match": True,
        "strict_portable_detected_iou_threshold": 0.05,
        "tracking_contact_iou_threshold": 0.15,
    }
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _hand_row(
    frame_number: int,
    frame_id: str,
    *,
    bbox: tuple[int, int, int, int] = (10, 10, 30, 30),
    contact_label: str = "Portable Object",
    blue_glove_status: str = "participant",
    tracking_bbox: tuple[float, float, float, float] | None = None,
    tracking_source: str = "none",
    hand_side: str = "Left",
) -> dict:
    contact_state = 3 if contact_label == "Portable Object" else 0
    row = {
        "frame_number": frame_number,
        "frame_id": frame_id,
        "detection_type": "hand",
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "confidence": 90.0,
        "contact_label": contact_label,
        "contact_state": contact_state,
        "hand_side": hand_side,
        "is_filtered": False,
        "blue_glove_status": blue_glove_status,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "offset_mag": 0.0,
    }
    if tracking_bbox is not None:
        row["tracking_bbox_x1"] = tracking_bbox[0]
        row["tracking_bbox_y1"] = tracking_bbox[1]
        row["tracking_bbox_x2"] = tracking_bbox[2]
        row["tracking_bbox_y2"] = tracking_bbox[3]
        row["tracking_bbox_source"] = tracking_source
    return row


def _object_row(
    frame_number: int,
    frame_id: str,
    *,
    bbox: tuple[int, int, int, int],
    confidence: float = 95.0,
) -> dict:
    return {
        "frame_number": frame_number,
        "frame_id": frame_id,
        "detection_type": "object",
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "confidence": confidence,
        "contact_label": None,
        "contact_state": None,
        "is_filtered": False,
        "blue_glove_status": "NA",
    }


def test_strict_filter_demotes_portable_without_detected_or_tracked_evidence():
    full_df = pd.DataFrame([_hand_row(1, "000001.png")])
    out_df = apply_strict_portable_match_filter(full_df, _cfg())
    hand = out_df.iloc[0]

    assert hand["contact_label"] == "No Contact"
    assert int(hand["contact_state"]) == 0
    assert bool(hand["portable_strict_eligible"]) is False
    assert hand["portable_evidence_source"] == "none"


def test_strict_filter_demotes_when_detected_iou_below_threshold():
    rows = [
        _hand_row(1, "000001.png", bbox=(0, 0, 10, 10)),
        _object_row(1, "000001.png", bbox=(100, 100, 120, 120)),
    ]
    out_df = apply_strict_portable_match_filter(pd.DataFrame(rows), _cfg())
    hand = out_df[out_df["detection_type"] == "hand"].iloc[0]

    assert hand["contact_label"] == "No Contact"
    assert int(hand["contact_state"]) == 0
    assert bool(hand["portable_strict_eligible"]) is False
    assert "detected_iou_below_threshold" in str(hand["strict_demote_reason"])


def test_strict_filter_keeps_portable_with_valid_detected_evidence():
    rows = [
        _hand_row(1, "000001.png", bbox=(0, 0, 10, 10)),
        _object_row(1, "000001.png", bbox=(1, 1, 9, 9)),
    ]
    out_df = apply_strict_portable_match_filter(pd.DataFrame(rows), _cfg())
    hand = out_df[out_df["detection_type"] == "hand"].iloc[0]

    assert hand["contact_label"] == "Portable Object"
    assert int(hand["contact_state"]) == 3
    assert bool(hand["portable_strict_eligible"]) is True
    assert hand["portable_evidence_source"] == "detected"
    assert float(hand["portable_evidence_iou"]) >= 0.05


def test_strict_filter_demotes_blue_experimenter_even_with_object_match():
    rows = [
        _hand_row(1, "000001.png", bbox=(0, 0, 10, 10), blue_glove_status="experimenter"),
        _object_row(1, "000001.png", bbox=(1, 1, 9, 9)),
    ]
    out_df = apply_strict_portable_match_filter(pd.DataFrame(rows), _cfg())
    hand = out_df[out_df["detection_type"] == "hand"].iloc[0]

    assert hand["contact_label"] == "No Contact"
    assert int(hand["contact_state"]) == 0
    assert hand["strict_demote_reason"] == "blue_experimenter"


def test_strict_filter_keeps_portable_with_valid_tracked_evidence():
    rows = [
        _hand_row(
            1,
            "000001.png",
            bbox=(10, 10, 30, 30),
            tracking_bbox=(12.0, 12.0, 28.0, 28.0),
            tracking_source="predicted",
        )
    ]
    out_df = apply_strict_portable_match_filter(pd.DataFrame(rows), _cfg())
    hand = out_df.iloc[0]

    assert hand["contact_label"] == "Portable Object"
    assert int(hand["contact_state"]) == 3
    assert bool(hand["portable_strict_eligible"]) is True
    assert hand["portable_evidence_source"] == "tracked"
    assert float(hand["portable_evidence_iou"]) >= 0.15


def test_strict_filter_demotes_when_tracked_iou_below_threshold():
    rows = [
        _hand_row(
            1,
            "000001.png",
            bbox=(10, 10, 30, 30),
            tracking_bbox=(100.0, 100.0, 120.0, 120.0),
            tracking_source="predicted",
        )
    ]
    out_df = apply_strict_portable_match_filter(pd.DataFrame(rows), _cfg())
    hand = out_df.iloc[0]

    assert hand["contact_label"] == "No Contact"
    assert int(hand["contact_state"]) == 0
    assert bool(hand["portable_strict_eligible"]) is False
    assert "tracked_iou_below_threshold" in str(hand["strict_demote_reason"])


def test_condensed_reflects_strict_demotion():
    rows = [
        _hand_row(1, "000001.png", bbox=(10, 10, 30, 30), contact_label="Portable Object"),
        _hand_row(2, "000002.png", bbox=(10, 10, 30, 30), contact_label="No Contact"),
    ]
    strict_df = apply_strict_portable_match_filter(pd.DataFrame(rows), _cfg())
    condensed = condense_dataframe(strict_df)

    frame1 = condensed[condensed["frame_number"] == 1].iloc[0]
    frame2 = condensed[condensed["frame_number"] == 2].iloc[0]
    assert frame1["contact_label"] == "No Contact"
    assert frame2["contact_label"] == "No Contact"
