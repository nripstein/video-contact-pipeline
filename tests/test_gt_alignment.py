from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.visualization import load_gt_binary_from_csv


def test_load_gt_binary_from_csv_aligns_missing(tmp_path: Path):
    gt_path = tmp_path / "gt.csv"
    gt_df = pd.DataFrame(
        {
            "frame_number": [1, 3, 5],
            "gt_binary": [1, 0, 1],
        }
    )
    gt_df.to_csv(gt_path, index=False)

    frame_numbers = [1, 2, 3, 4, 5]
    aligned = load_gt_binary_from_csv(str(gt_path), frame_numbers)
    assert aligned.tolist() == [1, 0, 0, 0, 1]


def test_load_gt_binary_from_csv_frame_id_label(tmp_path: Path):
    gt_path = tmp_path / "gt_labels.csv"
    gt_df = pd.DataFrame(
        {
            "video_id": ["sv1_frames"] * 3,
            "frame_id": ["0_sv1.jpg", "1_sv1.jpg", "3_sv1.jpg"],
            "label": ["not_holding", "holding", "not_holding"],
            "fps": [60, 60, 60],
        }
    )
    gt_df.to_csv(gt_path, index=False)

    frame_numbers = [0, 1, 2, 3]
    aligned = load_gt_binary_from_csv(str(gt_path), frame_numbers)
    assert aligned.tolist() == [0, 1, 0, 0]
