from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.frame_extractor import extract_video_frames


class _FakeCapture:
    def __init__(self, frames, fps: float):
        self._frames = frames
        self._fps = fps
        self._idx = 0

    def isOpened(self):
        return True

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def get(self, _prop):
        return self._fps

    def release(self):
        return None


def test_extract_video_frames_tracks_fps_and_time(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "sv1_clip.mp4"
    video_path.write_bytes(b"x")

    frames = [
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.zeros((10, 20, 3), dtype=np.uint8),
    ]

    monkeypatch.setattr(
        "stimulus_detector.data_generation.frame_extractor.cv2.VideoCapture",
        lambda _path: _FakeCapture(frames=frames, fps=30.0),
    )

    def _fake_imwrite(path, _frame):
        Path(path).write_bytes(b"frame")
        return True

    monkeypatch.setattr("stimulus_detector.data_generation.frame_extractor.cv2.imwrite", _fake_imwrite)

    config = Phase1Config(input_path=str(video_path), output_dir=str(tmp_path / "out"))
    seq = extract_video_frames(
        video_path=str(video_path),
        output_frames_root=tmp_path / "frames",
        config=config,
        participant_id="sv1",
    )

    assert abs(seq.fps - 30.0) < 1e-6
    assert len(seq.frame_records) == 3
    assert abs(seq.frame_records[1].frame_time_sec - (1.0 / 30.0)) < 1e-6


def test_extract_video_frames_fallback_fps(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "sv2_clip.mp4"
    video_path.write_bytes(b"x")

    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    monkeypatch.setattr(
        "stimulus_detector.data_generation.frame_extractor.cv2.VideoCapture",
        lambda _path: _FakeCapture(frames=frames, fps=0.0),
    )
    monkeypatch.setattr(
        "stimulus_detector.data_generation.frame_extractor.cv2.imwrite",
        lambda path, _frame: Path(path).write_bytes(b"frame") or True,
    )

    config = Phase1Config(
        input_path=str(video_path),
        output_dir=str(tmp_path / "out"),
        default_fps_if_unknown=60.0,
    )
    seq = extract_video_frames(
        video_path=str(video_path),
        output_frames_root=tmp_path / "frames",
        config=config,
        participant_id="sv2",
    )

    assert abs(seq.fps - 60.0) < 1e-6
    assert seq.frame_records[0].frame_time_sec == 0.0
