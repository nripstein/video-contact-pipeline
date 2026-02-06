from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import PipelineConfig
from pipeline.filters import apply_obj_bigger_than_hand_filter


def test_obj_bigger_than_hand_filter_unit():
    # Frame 1: two objects, highest confidence should win even if farther.
    # Hand area = 100. Object A area = 200 (should trigger relabel if matched).
    # Object B area = 50 (would not trigger).
    frame1 = [
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
            "bbox_x1": 100,
            "bbox_y1": 100,
            "bbox_x2": 120,
            "bbox_y2": 120,
        },
        {
            "frame_id": "f1.png",
            "frame_number": 1,
            "detection_type": "object",
            "is_filtered": False,
            "confidence": 0.5,
            "bbox_x1": 5,
            "bbox_y1": 5,
            "bbox_x2": 12,
            "bbox_y2": 12,
        },
    ]

    # Frame 2: same confidence, closer distance should win.
    # Object A close and small => no relabel.
    # Object B far and large => would relabel if chosen.
    frame2 = [
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
            "confidence": 0.7,
            "bbox_x1": 6,
            "bbox_y1": 6,
            "bbox_x2": 12,
            "bbox_y2": 12,
        },
        {
            "frame_id": "f2.png",
            "frame_number": 2,
            "detection_type": "object",
            "is_filtered": False,
            "confidence": 0.7,
            "bbox_x1": 100,
            "bbox_y1": 100,
            "bbox_x2": 130,
            "bbox_y2": 130,
        },
    ]

    # Frame 3: no objects -> no relabel.
    frame3 = [
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
    ]

    full_df = pd.DataFrame(frame1 + frame2 + frame3)

    cfg = PipelineConfig(
        input_path=".",
        output_dir=None,
        obj_bigger_than_hand_filter=True,
        obj_bigger_ratio_k=1.0,
    )

    filtered = apply_obj_bigger_than_hand_filter(full_df, cfg)

    # Frame 1: pick higher confidence object => relabel
    f1_hand = filtered[(filtered["frame_id"] == "f1.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f1_hand["contact_label"] == "No Contact"
    assert int(f1_hand["contact_state"]) == 0

    # Frame 2: same confidence -> choose closer object => no relabel
    f2_hand = filtered[(filtered["frame_id"] == "f2.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f2_hand["contact_label"] == "Portable Object"
    assert int(f2_hand["contact_state"]) == 3

    # Frame 3: no objects => unchanged
    f3_hand = filtered[(filtered["frame_id"] == "f3.png") & (filtered["detection_type"] == "hand")].iloc[0]
    assert f3_hand["contact_label"] == "Portable Object"
    assert int(f3_hand["contact_state"]) == 3
