from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.data_generation.filters import BboxSimilarityFilter
from stimulus_detector.data_generation.types import FilterContext, FrameRecord, PseudoLabel


def _label(frame_idx: int, time_sec: float, bbox, width: int = 100) -> PseudoLabel:
    frame = FrameRecord(
        video_id="clip1",
        participant_id="sv1",
        frame_idx=frame_idx,
        frame_time_sec=time_sec,
        fps=60.0,
        frame_path=f"/tmp/{frame_idx:06d}.png",
        width=width,
        height=100,
    )
    return PseudoLabel(frame=frame, bbox_xyxy=bbox, confidence=0.9)


def test_bbox_similarity_filter_thresholds():
    filt = BboxSimilarityFilter()
    context = FilterContext(
        min_temporal_gap_sec=0.25,
        center_move_frac=0.10,
        area_change_frac=0.20,
        subsample_interval_sec=0.25,
    )

    first = _label(0, 0.00, (10, 10, 20, 20))
    near_time = _label(5, 0.10, (30, 10, 40, 20))
    low_move = _label(16, 0.30, (11, 10, 21, 20))
    low_area = _label(20, 0.40, (25, 10, 35, 20))
    keep = _label(30, 0.60, (45, 10, 60, 25))

    out = filt.filter([first, near_time, low_move, low_area, keep], context)

    assert [x.frame.frame_idx for x in out] == [0, 30]
