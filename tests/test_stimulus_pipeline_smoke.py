from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.pipeline import run_phase1_data_generation
from stimulus_detector.data_generation.types import HandDetection, ObjectDetection


def _make_frame(path: Path):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class _DummyWrapper:
    def __init__(self, _config):
        pass

    def run_on_sequence(self, sequence):
        objects = []
        hands = []
        for frame in sequence.frame_records:
            objects.append(
                ObjectDetection(
                    frame=frame,
                    bbox_xyxy=(20.0, 20.0, 32.0, 32.0),
                    confidence=0.95,
                )
            )
            hands.append(
                HandDetection(
                    frame=frame,
                    bbox_xyxy=(10.0, 10.0, 48.0, 48.0),
                    confidence=0.99,
                )
            )
        return objects, hands


def test_phase1_pipeline_smoke(monkeypatch, tmp_path: Path):
    input_root = tmp_path / "input"
    sv1 = input_root / "sv1" / "clip_a"
    sv2 = input_root / "sv2" / "clip_b"
    sv1.mkdir(parents=True)
    sv2.mkdir(parents=True)

    _make_frame(sv1 / "000000.png")
    _make_frame(sv2 / "000000.png")

    monkeypatch.setattr("stimulus_detector.data_generation.pipeline.ShanInferenceWrapper", _DummyWrapper)

    config = Phase1Config(
        input_path=str(input_root),
        output_dir=str(tmp_path / "out"),
        filter_strategy="bbox_similarity",
        generate_visualizations=False,
        generate_stats=False,
        val_fraction=0.5,
        split_seed=1,
    )

    result = run_phase1_data_generation(config)

    assert result.selected_count == 2
    assert result.train_count == 1
    assert result.val_count == 1
    assert Path(result.train_coco_json).exists()
    assert Path(result.val_coco_json).exists()
    assert Path(result.raw_csv).exists()
    assert Path(result.heuristics_csv).exists()
    assert Path(result.selected_csv).exists()
