from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import PipelineConfig
from pipeline.main import run_pipeline


class FakeDetector:
    init_count = 0

    def __init__(self, *args, **kwargs):
        FakeDetector.init_count += 1

    def run_on_directory(self, image_dir: str):
        return pd.DataFrame(
            [
                {
                    "frame_id": "000001.png",
                    "frame_number": 1,
                    "detection_type": "hand",
                    "bbox_x1": 0,
                    "bbox_y1": 0,
                    "bbox_x2": 10,
                    "bbox_y2": 10,
                    "confidence": 1.0,
                    "contact_state": 0,
                    "contact_label": "No Contact",
                    "hand_side": "Left",
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                    "offset_mag": 0.0,
                    "blue_prop": None,
                    "blue_glove_status": "NA",
                    "is_filtered": False,
                    "filtered_by": "",
                    "filtered_reason": "",
                }
            ]
        )


def test_batch_model_loaded_once(tmp_path: Path, monkeypatch):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "a.mp4").write_text("x")
    (videos_dir / "b.mp4").write_text("x")

    output_dir = tmp_path / "out"

    # Patch preprocessing to avoid real frame extraction.
    def fake_prepare_frames(cfg):
        frames = tmp_path / "frames" / Path(cfg.input_path).stem
        frames.mkdir(parents=True, exist_ok=True)
        (frames / "000001.png").write_text("x")
        return str(frames)

    monkeypatch.setattr("pipeline.main.prepare_frames", fake_prepare_frames)
    monkeypatch.setattr("pipeline.main.run_inference", lambda cfg, image_dir, detector=None: detector.run_on_directory(image_dir))

    cfg = PipelineConfig(
        input_path=str(videos_dir),
        output_dir=str(output_dir),
        save_full_csv=False,
        save_condensed_csv=False,
        save_config=False,
    )

    FakeDetector.init_count = 0
    run_pipeline(cfg, detector_factory=lambda c: FakeDetector())

    assert FakeDetector.init_count == 1
    assert (output_dir / "a").exists()
    assert (output_dir / "b").exists()
