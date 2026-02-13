from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from video_maker.contact_timeline_renderer import (
    align_secondary_prediction,
    load_gt_binary_aligned,
    load_pred_binary_from_condensed,
    render_contact_timeline_video,
)


def _write_dummy_frames(image_dir: Path, count: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        img = np.zeros((48, 80, 3), dtype=np.uint8)
        img[:, :, 1] = i * 20
        cv2.imwrite(str(image_dir / f"{i:06d}.png"), img)


def test_load_pred_and_secondary_alignment(tmp_path: Path):
    condensed = pd.DataFrame(
        {
            "frame_id": ["000001.png", "000002.png", "000003.png"],
            "frame_number": [1, 2, 3],
            "contact_label": ["No Contact", "Portable Object", "Stationary Object"],
            "source_hand": ["Left", "Left", "Left"],
        }
    )
    condensed_path = tmp_path / "detections_condensed.csv"
    condensed.to_csv(condensed_path, index=False)

    frame_numbers, pred = load_pred_binary_from_condensed(str(condensed_path))
    assert frame_numbers == [1, 2, 3]
    assert pred.tolist() == [0, 1, 0]

    secondary = pd.DataFrame(
        {
            "frame_id": ["000001.png", "000003.png"],
            "frame_number": [1, 3],
            "contact_label": ["Portable Object", "Portable Object"],
            "source_hand": ["Left", "Left"],
        }
    )
    secondary_path = tmp_path / "secondary.csv"
    secondary.to_csv(secondary_path, index=False)

    aligned = align_secondary_prediction(str(secondary_path), frame_numbers)
    assert aligned.tolist() == [1, 0, 1]


def test_render_contact_timeline_video_smoke(tmp_path: Path):
    image_dir = tmp_path / "frames"
    _write_dummy_frames(image_dir, 4)
    frame_numbers = [1, 2, 3, 4]
    pred = np.array([0, 1, 1, 0], dtype=np.uint8)
    gt = np.array([0, 1, 0, 0], dtype=np.uint8)
    secondary = np.array([0, 0, 1, 1], dtype=np.uint8)

    out_path = tmp_path / "visualizations" / "timeline.mp4"
    saved = render_contact_timeline_video(
        image_dir=str(image_dir),
        frame_numbers=frame_numbers,
        pred_binary=pred,
        output_video_path=str(out_path),
        fps=10.0,
        title="Test Timeline",
        gt_binary=gt,
        secondary_binary=secondary,
    )
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0


def test_cli_make_contact_timeline_video(tmp_path: Path):
    image_dir = tmp_path / "frames"
    _write_dummy_frames(image_dir, 3)

    condensed = pd.DataFrame(
        {
            "frame_id": ["000001.png", "000002.png", "000003.png"],
            "frame_number": [1, 2, 3],
            "contact_label": ["No Contact", "Portable Object", "No Contact"],
            "source_hand": ["Left", "Left", "Right"],
        }
    )
    condensed_path = tmp_path / "detections_condensed.csv"
    condensed.to_csv(condensed_path, index=False)

    gt = pd.DataFrame({"frame_number": [1, 2, 3], "gt_binary": [0, 1, 0]})
    gt_path = tmp_path / "gt.csv"
    gt.to_csv(gt_path, index=False)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "make_contact_timeline_video.py"),
        "--condensed-csv",
        str(condensed_path),
        "--image-dir",
        str(image_dir),
        "--gt-csv",
        str(gt_path),
        "--fps",
        "10",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    assert result.returncode == 0

    out_path = tmp_path / "visualizations" / "contact_timeline.mp4"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_load_gt_binary_aligned_uses_existing_formats(tmp_path: Path):
    gt_df = pd.DataFrame(
        {
            "frame_id": ["000001.png", "000002.png", "000003.png"],
            "label": ["not_holding", "holding", "not_holding"],
        }
    )
    gt_path = tmp_path / "gt_labels.csv"
    gt_df.to_csv(gt_path, index=False)

    aligned = load_gt_binary_aligned(str(gt_path), [1, 2, 3, 4])
    assert aligned.tolist() == [0, 1, 0, 0]
