from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import PipelineConfig
from pipeline.filters import apply_obj_smaller_than_hand_filter


def test_obj_smaller_than_hand_filter_unit():
    rows = [
        # Frame 1: object area 80 < hand area 100 -> keep portable.
        {
            "frame_id": "f1.png",
            "frame_number": 1,
            "detection_type": "hand",
            "contact_label": "Portable Object",
            "contact_state": 3,
            "is_filtered": False,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
            "offset_mag": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        {
            "frame_id": "f1.png",
            "frame_number": 1,
            "detection_type": "object",
            "is_filtered": False,
            "confidence": 0.9,
            "bbox_x1": 1,
            "bbox_y1": 1,
            "bbox_x2": 9,
            "bbox_y2": 11,
        },
        # Frame 2: object area 121 > hand area 100 -> relabel.
        {
            "frame_id": "f2.png",
            "frame_number": 2,
            "detection_type": "hand",
            "contact_label": "Portable Object",
            "contact_state": 3,
            "is_filtered": False,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
            "offset_mag": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        {
            "frame_id": "f2.png",
            "frame_number": 2,
            "detection_type": "object",
            "is_filtered": False,
            "confidence": 0.8,
            "bbox_x1": 5,
            "bbox_y1": 5,
            "bbox_x2": 16,
            "bbox_y2": 16,
        },
        # Frame 3: equality boundary (object area == hand area) -> keep portable.
        {
            "frame_id": "f3.png",
            "frame_number": 3,
            "detection_type": "hand",
            "contact_label": "Portable Object",
            "contact_state": 3,
            "is_filtered": False,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
            "offset_mag": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        {
            "frame_id": "f3.png",
            "frame_number": 3,
            "detection_type": "object",
            "is_filtered": False,
            "confidence": 0.7,
            "bbox_x1": 5,
            "bbox_y1": 5,
            "bbox_x2": 15,
            "bbox_y2": 15,
        },
        # Frame 4: no object -> unchanged.
        {
            "frame_id": "f4.png",
            "frame_number": 4,
            "detection_type": "hand",
            "contact_label": "Portable Object",
            "contact_state": 3,
            "is_filtered": False,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
            "offset_mag": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        # Frame 5: non-portable hand -> unchanged.
        {
            "frame_id": "f5.png",
            "frame_number": 5,
            "detection_type": "hand",
            "contact_label": "No Contact",
            "contact_state": 0,
            "is_filtered": False,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
            "offset_mag": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        {
            "frame_id": "f5.png",
            "frame_number": 5,
            "detection_type": "object",
            "is_filtered": False,
            "confidence": 0.9,
            "bbox_x1": 2,
            "bbox_y1": 2,
            "bbox_x2": 8,
            "bbox_y2": 8,
        },
    ]

    full_df = pd.DataFrame(rows)
    cfg = PipelineConfig(
        input_path=".",
        output_dir=None,
        obj_smaller_than_hand_filter=True,
        obj_smaller_ratio_factor=1.0,
    )

    filtered = apply_obj_smaller_than_hand_filter(full_df, cfg)

    f1_hand = filtered[(filtered["frame_id"] == "f1.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f1_hand["contact_label"] == "Portable Object"
    assert int(f1_hand["contact_state"]) == 3
    assert bool(f1_hand["small_object_rule_applied"]) is False

    f2_hand = filtered[(filtered["frame_id"] == "f2.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f2_hand["contact_label"] == "No Contact"
    assert int(f2_hand["contact_state"]) == 0
    assert bool(f2_hand["small_object_rule_applied"]) is True

    f3_hand = filtered[(filtered["frame_id"] == "f3.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f3_hand["contact_label"] == "Portable Object"
    assert int(f3_hand["contact_state"]) == 3
    assert bool(f3_hand["small_object_rule_applied"]) is False

    f4_hand = filtered[(filtered["frame_id"] == "f4.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f4_hand["contact_label"] == "Portable Object"
    assert int(f4_hand["contact_state"]) == 3
    assert bool(f4_hand["small_object_rule_applied"]) is False

    f5_hand = filtered[(filtered["frame_id"] == "f5.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f5_hand["contact_label"] == "No Contact"
    assert int(f5_hand["contact_state"]) == 0
    assert bool(f5_hand["small_object_rule_applied"]) is False
