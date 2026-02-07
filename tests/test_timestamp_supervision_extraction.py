from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timestamp_supervision_extraction.extract_timestamps import (
    extract_frames,
    filter_islands,
    find_islands,
    merge_predictions_and_metadata,
    select_central_frames,
)


def test_find_islands_breaks_on_label_and_frame_gaps():
    df = pd.DataFrame(
        {
            "frame_id": [0, 1, 2, 4, 5, 6, 8],
            "predicted_label": [0, 0, 0, 0, 1, 1, 1],
            "blue_glove_detected": [False, False, False, False, False, False, False],
        }
    )

    islands = find_islands(df)
    triples = [
        (
            island["predicted_label"],
            island["island_start_frame_id"],
            island["island_end_frame_id"],
            island["island_length_frames"],
        )
        for island in islands
    ]
    assert triples == [
        (0, 0, 2, 3),
        (0, 4, 4, 1),
        (1, 5, 6, 2),
        (1, 8, 8, 1),
    ]


def test_filter_islands_applies_min_length_and_blue_glove_veto():
    islands = [
        {
            "video_id": "v1",
            "predicted_label": 1,
            "island_start_frame_id": 0,
            "island_end_frame_id": 59,
            "island_length_frames": 60,
            "frame_ids": list(range(60)),
            "has_blue_glove_true": False,
        },
        {
            "video_id": "v1",
            "predicted_label": 1,
            "island_start_frame_id": 100,
            "island_end_frame_id": 159,
            "island_length_frames": 60,
            "frame_ids": list(range(100, 160)),
            "has_blue_glove_true": True,
        },
        {
            "video_id": "v1",
            "predicted_label": 0,
            "island_start_frame_id": 200,
            "island_end_frame_id": 229,
            "island_length_frames": 30,
            "frame_ids": list(range(200, 230)),
            "has_blue_glove_true": False,
        },
    ]

    kept, summary = filter_islands(islands, fps=60.0, min_seconds=1.0)
    assert len(kept) == 1
    assert kept[0]["island_start_frame_id"] == 0
    assert summary == {
        "num_islands_found": 3,
        "num_passed_length": 2,
        "num_removed_blue_glove": 1,
        "num_selected": 1,
    }


def test_select_central_frames_reproducible_with_seed():
    islands = [
        {
            "video_id": "v1",
            "predicted_label": 0,
            "island_start_frame_id": 10,
            "island_end_frame_id": 20,
            "island_length_frames": 11,
            "frame_ids": list(range(10, 21)),
            "has_blue_glove_true": False,
        },
        {
            "video_id": "v2",
            "predicted_label": 1,
            "island_start_frame_id": 5,
            "island_end_frame_id": 5,
            "island_length_frames": 1,
            "frame_ids": [5],
            "has_blue_glove_true": False,
        },
    ]

    selections_a = select_central_frames(islands, rng=np.random.default_rng(123))
    selections_b = select_central_frames(islands, rng=np.random.default_rng(123))

    assert selections_a == selections_b
    assert 10 <= selections_a[0]["frame_id"] <= 20
    assert selections_a[1]["frame_id"] == 5


def test_left_join_missing_metadata_treated_as_false():
    predictions_df = pd.DataFrame(
        {
            "frame_id": [0, 1, 2],
            "predicted_label": [1, 1, 1],
        }
    )
    metadata_df = pd.DataFrame(
        {
            "frame_id": [0],
            "blue_glove_detected": [False],
        }
    )

    merged = merge_predictions_and_metadata(predictions_df, metadata_df, join_mode="left")
    islands = find_islands(merged)
    for island in islands:
        island["video_id"] = "v1"
    kept, _ = filter_islands(islands, fps=1.0, min_seconds=1.0)

    assert len(kept) == 1
    assert kept[0]["predicted_label"] == 1


def test_extract_frames_uses_frames_dir_video_subdir(tmp_path: Path):
    import cv2

    frames_root = tmp_path / "frames"
    video_frames_dir = frames_root / "v1"
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    source_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(video_frames_dir / "000010.png"), source_img)

    stats = extract_frames(
        selections=[{"video_id": "v1", "frame_id": 10}],
        videos_dir=tmp_path / "videos",
        out_dir=tmp_path / "out",
        backend="opencv",
        image_format="jpg",
        frames_dir=frames_root,
    )

    assert stats["attempted"] == 1
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0
    assert (tmp_path / "out" / "extracted_frames" / "v1" / "frame_10.jpg").exists()


def test_extract_frames_uses_single_video_flat_frames_dir(tmp_path: Path):
    import cv2

    frames_root = tmp_path / "flat_frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    source_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_root / "000003.png"), source_img)

    stats = extract_frames(
        selections=[{"video_id": "only_video", "frame_id": 3}],
        videos_dir=tmp_path / "videos",
        out_dir=tmp_path / "out",
        backend="opencv",
        image_format="jpg",
        frames_dir=frames_root,
    )

    assert stats["attempted"] == 1
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0
    assert (
        tmp_path
        / "out"
        / "extracted_frames"
        / "only_video"
        / "frame_3.jpg"
    ).exists()
