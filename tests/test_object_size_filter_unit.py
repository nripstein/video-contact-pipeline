from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import PipelineConfig
from pipeline.filters import apply_object_size_filter


def test_object_size_filter_marks_oversized(tmp_path: Path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_id = "000001.png"
    cv2.imwrite(str(image_dir / frame_id), img)

    full_df = pd.DataFrame(
        [
            {
                "frame_id": frame_id,
                "detection_type": "object",
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 20,
                "bbox_y2": 20,
                "is_filtered": False,
                "filtered_by": "",
                "filtered_reason": "",
            },
            {
                "frame_id": frame_id,
                "detection_type": "object",
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 100,
                "bbox_y2": 100,
                "is_filtered": False,
                "filtered_by": "",
                "filtered_reason": "",
            },
        ]
    )

    cfg = PipelineConfig(
        input_path=str(image_dir),
        output_dir=None,
        object_size_filter=True,
        object_size_max_area_ratio=0.5,
    )

    filtered = apply_object_size_filter(full_df, cfg, str(image_dir))

    assert len(filtered) == 2
    small = filtered.iloc[0]
    large = filtered.iloc[1]

    assert bool(small["is_filtered"]) is False
    assert small["filtered_by"] == ""

    assert bool(large["is_filtered"]) is True
    assert large["filtered_by"] == "object_size_filter"
