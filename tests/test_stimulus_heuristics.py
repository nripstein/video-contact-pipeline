from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.heuristics import apply_heuristic_filters, match_hand_detection
from stimulus_detector.data_generation.types import FrameRecord, HandDetection, ObjectDetection


def _frame(tmp_path: Path) -> FrameRecord:
    img = tmp_path / "000001.png"
    img.write_bytes(b"x")
    return FrameRecord(
        video_id="clip1",
        participant_id="sv1",
        frame_idx=1,
        frame_time_sec=1 / 60.0,
        fps=60.0,
        frame_path=str(img),
        width=100,
        height=100,
    )


def test_match_hand_prefers_iou_then_distance(tmp_path: Path):
    frame = _frame(tmp_path)
    obj = ObjectDetection(frame=frame, bbox_xyxy=(10, 10, 20, 20), confidence=0.9)

    hand_high_iou = HandDetection(frame=frame, bbox_xyxy=(9, 9, 21, 21), confidence=0.8)
    hand_low_iou = HandDetection(frame=frame, bbox_xyxy=(50, 50, 70, 70), confidence=0.9)

    best, iou, _ = match_hand_detection(obj, [hand_low_iou, hand_high_iou])
    assert best == hand_high_iou
    assert iou is not None and iou > 0


def test_heuristics_keep_only_valid_candidates(tmp_path: Path):
    frame = _frame(tmp_path)
    cfg = Phase1Config(input_path="/tmp/input", output_dir="/tmp/out")

    obj_ok = ObjectDetection(frame=frame, bbox_xyxy=(10, 10, 20, 20), confidence=0.95)
    obj_low_conf = ObjectDetection(frame=frame, bbox_xyxy=(10, 10, 20, 20), confidence=0.6)
    obj_bad_ar = ObjectDetection(frame=frame, bbox_xyxy=(10, 10, 30, 15), confidence=0.95)
    obj_too_big = ObjectDetection(frame=frame, bbox_xyxy=(5, 5, 40, 40), confidence=0.95)

    hand = HandDetection(frame=frame, bbox_xyxy=(0, 0, 30, 30), confidence=0.9)

    kept, audit = apply_heuristic_filters(
        object_detections=[obj_ok, obj_low_conf, obj_bad_ar, obj_too_big],
        hand_detections=[hand],
        config=cfg,
    )

    assert len(kept) == 1
    assert kept[0].bbox_xyxy == obj_ok.bbox_xyxy
    assert len(audit) == 4
    fail_reasons = [row["heuristics_fail_reasons"] for row in audit if not row["passed_heuristics"]]
    assert any("low_confidence" in reason for reason in fail_reasons)
    assert any("aspect_ratio_out_of_range" in reason for reason in fail_reasons)
    assert any("object_too_large_relative_to_hand" in reason for reason in fail_reasons)


def test_heuristics_apply_strict_object_hand_ratio(tmp_path: Path):
    frame = _frame(tmp_path)
    cfg = Phase1Config(
        input_path="/tmp/input",
        output_dir="/tmp/out",
        max_object_to_hand_area_ratio=0.5,
    )

    # area=400
    obj = ObjectDetection(frame=frame, bbox_xyxy=(10, 10, 30, 30), confidence=0.95)
    # area=625 -> ratio=0.64, should fail with threshold 0.5
    hand = HandDetection(frame=frame, bbox_xyxy=(5, 5, 30, 30), confidence=0.9)

    kept, audit = apply_heuristic_filters(
        object_detections=[obj],
        hand_detections=[hand],
        config=cfg,
    )

    assert len(kept) == 0
    assert len(audit) == 1
    assert "object_too_large_relative_to_hand" in audit[0]["heuristics_fail_reasons"]


def test_heuristics_filter_occluded_objects(tmp_path: Path):
    frame = _frame(tmp_path)
    cfg = Phase1Config(
        input_path="/tmp/input",
        output_dir="/tmp/out",
        max_hand_occlusion_ratio=0.4,
    )

    # fully overlapped by hand -> overlap ratio=1.0
    obj = ObjectDetection(frame=frame, bbox_xyxy=(10, 10, 20, 20), confidence=0.95)
    hand = HandDetection(frame=frame, bbox_xyxy=(0, 0, 30, 30), confidence=0.9)

    kept, audit = apply_heuristic_filters(
        object_detections=[obj],
        hand_detections=[hand],
        config=cfg,
    )

    assert len(kept) == 0
    assert len(audit) == 1
    assert "object_too_occluded_by_hand" in audit[0]["heuristics_fail_reasons"]
