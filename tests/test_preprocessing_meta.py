from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.main import prepare_frames


def test_preprocessing_meta_written(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "000001.png"), img)

    output_dir = tmp_path / "out"
    cfg = PipelineConfig(
        input_path=str(frames_dir),
        output_dir=str(output_dir),
        crop_square=8,
        flip_vertical=True,
    )

    processed_dir = prepare_frames(cfg)
    assert Path(processed_dir).exists()

    meta_path = output_dir / "preprocessing_meta.json"
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["original_width"] == 12
    assert meta["original_height"] == 10
    assert meta["processed_width"] == 8
    assert meta["processed_height"] == 8
    assert meta["crop_box"] == [2, 1, 10, 9]
    assert meta["flip_vertical"] is True
    assert meta["frame_ext"] == "png"
    assert meta["zero_pad"] is None
