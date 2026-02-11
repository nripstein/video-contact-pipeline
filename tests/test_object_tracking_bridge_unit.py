from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_tracking.bridge import apply_tracking_bridge
from pipeline.config import PipelineConfig


def _hand_row(
    frame_number: int,
    frame_id: str,
    bbox=(10, 10, 30, 30),
    contact_label: str = "No Contact",
    blue_glove_status: str = "participant",
):
    if contact_label == "No Contact":
        contact_state = 0
    elif contact_label in {"Stationary Object", "Stationary Object Contact"}:
        contact_state = 4
    else:
        contact_state = 3
    return {
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
        "is_filtered": False,
        "blue_glove_status": blue_glove_status,
    }


def _object_row(
    frame_number: int,
    frame_id: str,
    bbox=(12, 12, 28, 28),
    confidence: float = 95.0,
):
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


def _cfg(**kwargs) -> PipelineConfig:
    defaults = {
        "input_path": "dummy",
        "output_dir": "dummy_out",
        "tracking_bridge_enabled": True,
        "tracking_max_missed_frames": 8,
        "tracking_contact_iou_threshold": 0.15,
        "tracking_init_obj_confidence": 0.70,
        "tracking_promotion_confirm_frames": 2,
        "tracking_reassociate_iou_threshold": 0.10,
        "tracking_promote_stationary": False,
        "tracking_stationary_iou_threshold": 0.20,
        "tracking_stationary_confirm_frames": 2,
    }
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def test_tracking_bridge_promotes_after_two_miss_frames():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="No Contact"),
        _hand_row(3, "000003.png", contact_label="No Contact"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(full_df, _cfg())
    hand_f2 = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")].iloc[0]
    hand_f3 = out_df[(out_df["frame_number"] == 3) & (out_df["detection_type"] == "hand")].iloc[0]

    assert bool(hand_f2["tracking_promoted"]) is False
    assert hand_f2["contact_label"] == "No Contact"
    assert bool(hand_f3["tracking_promoted"]) is True
    assert hand_f3["contact_label"] == "Portable Object"
    assert int(hand_f3["contact_state"]) == 3


def test_tracking_bridge_does_not_init_with_low_confidence_object():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=60.0),
        _hand_row(2, "000002.png", contact_label="No Contact"),
        _hand_row(3, "000003.png", contact_label="No Contact"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(full_df, _cfg())
    hand_rows = out_df[out_df["detection_type"] == "hand"]
    assert not bool(hand_rows["tracking_promoted"].any())
    assert set(hand_rows["tracking_state"].tolist()) == {"inactive"}


def test_tracking_bridge_requires_portable_hand_for_strict_init():
    rows = [
        _hand_row(1, "000001.png", contact_label="No Contact"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="No Contact"),
        _hand_row(3, "000003.png", contact_label="No Contact"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(full_df, _cfg())
    assert not bool(out_df["tracking_promoted"].any())
    assert set(out_df["tracking_state"].tolist()) == {"inactive"}


def test_tracking_bridge_expires_track_after_gap_budget():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
    ]
    for frame_number in range(2, 12):
        rows.append(_hand_row(frame_number, f"{frame_number:06d}.png", contact_label="No Contact"))

    full_df = pd.DataFrame(rows)
    out_df = apply_tracking_bridge(full_df, _cfg(tracking_promotion_confirm_frames=99))

    frame10 = out_df[(out_df["frame_number"] == 10) & (out_df["detection_type"] == "hand")].iloc[0]
    frame11 = out_df[(out_df["frame_number"] == 11) & (out_df["detection_type"] == "hand")].iloc[0]
    assert frame10["tracking_state"] == "lost"
    assert frame11["tracking_state"] == "inactive"


def test_tracking_bridge_blocks_promotion_on_experimenter_blue_glove():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="No Contact", blue_glove_status="experimenter"),
        _hand_row(3, "000003.png", contact_label="No Contact", blue_glove_status="participant"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(full_df, _cfg())
    promoted = out_df[out_df["tracking_promoted"] == True]
    assert promoted.empty


def test_tracking_bridge_promotes_stationary_when_enabled():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="Stationary Object"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(
        full_df,
        _cfg(
            tracking_promote_stationary=True,
            tracking_stationary_iou_threshold=0.01,
            tracking_stationary_confirm_frames=1,
        ),
    )
    hand_f2 = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")].iloc[0]
    assert bool(hand_f2["tracking_promoted"]) is True
    assert hand_f2["contact_label"] == "Portable Object"
    assert int(hand_f2["contact_state"]) == 3


def test_tracking_bridge_does_not_promote_stationary_when_disabled():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="Stationary Object"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(
        full_df,
        _cfg(
            tracking_promote_stationary=False,
            tracking_stationary_iou_threshold=0.01,
            tracking_stationary_confirm_frames=1,
        ),
    )
    hand_f2 = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")].iloc[0]
    assert bool(hand_f2["tracking_promoted"]) is False
    assert hand_f2["contact_label"] == "Stationary Object"
    assert int(hand_f2["contact_state"]) == 4


def test_tracking_bridge_stationary_requires_iou_threshold():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="Stationary Object"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(
        full_df,
        _cfg(
            tracking_promote_stationary=True,
            tracking_stationary_iou_threshold=0.99,
            tracking_stationary_confirm_frames=1,
        ),
    )
    hand_f2 = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")].iloc[0]
    assert bool(hand_f2["tracking_promoted"]) is False
    assert hand_f2["contact_label"] == "Stationary Object"


def test_tracking_bridge_stationary_requires_streak():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="Stationary Object"),
        _hand_row(3, "000003.png", contact_label="Stationary Object"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(
        full_df,
        _cfg(
            tracking_promote_stationary=True,
            tracking_stationary_iou_threshold=0.01,
            tracking_stationary_confirm_frames=2,
        ),
    )
    hand_f2 = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")].iloc[0]
    hand_f3 = out_df[(out_df["frame_number"] == 3) & (out_df["detection_type"] == "hand")].iloc[0]
    assert bool(hand_f2["tracking_promoted"]) is False
    assert hand_f2["contact_label"] == "Stationary Object"
    assert bool(hand_f3["tracking_promoted"]) is True
    assert hand_f3["contact_label"] == "Portable Object"


def test_tracking_bridge_prefers_highest_iou_eligible_candidate():
    rows = [
        _hand_row(1, "000001.png", bbox=(10, 10, 30, 30), contact_label="Portable Object"),
        _object_row(1, "000001.png", bbox=(12, 12, 28, 28), confidence=95.0),
        _hand_row(2, "000002.png", bbox=(0, 0, 5, 5), contact_label="No Contact"),
        _hand_row(2, "000002.png", bbox=(11, 11, 29, 29), contact_label="Stationary Object"),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(
        full_df,
        _cfg(
            tracking_promote_stationary=True,
            tracking_contact_iou_threshold=0.01,
            tracking_promotion_confirm_frames=1,
            tracking_stationary_iou_threshold=0.01,
            tracking_stationary_confirm_frames=1,
        ),
    )
    frame2_hands = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")]
    promoted = frame2_hands[frame2_hands["tracking_promoted"] == True]
    assert len(promoted) == 1
    assert promoted.iloc[0]["contact_label"] == "Portable Object"
    assert int(promoted.iloc[0]["bbox_x1"]) == 11


def test_tracking_bridge_no_promotion_on_associated_object_frames():
    rows = [
        _hand_row(1, "000001.png", contact_label="Portable Object"),
        _object_row(1, "000001.png", confidence=95.0),
        _hand_row(2, "000002.png", contact_label="Stationary Object"),
        _object_row(2, "000002.png", confidence=95.0),
    ]
    full_df = pd.DataFrame(rows)

    out_df = apply_tracking_bridge(
        full_df,
        _cfg(
            tracking_promote_stationary=True,
            tracking_stationary_iou_threshold=0.01,
            tracking_stationary_confirm_frames=1,
        ),
    )
    hand_f2 = out_df[(out_df["frame_number"] == 2) & (out_df["detection_type"] == "hand")].iloc[0]
    assert bool(hand_f2["tracking_promoted"]) is False
    assert hand_f2["contact_label"] == "Stationary Object"
