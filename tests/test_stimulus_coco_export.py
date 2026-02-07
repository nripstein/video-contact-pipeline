from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.data_generation.exporters import export_coco_dataset
from stimulus_detector.data_generation.types import FrameRecord, PseudoLabel


def _label(tmp_path: Path, participant: str, video: str, frame_idx: int) -> PseudoLabel:
    img = tmp_path / f"{participant}_{video}_{frame_idx}.png"
    img.write_bytes(b"image")
    frame = FrameRecord(
        video_id=video,
        participant_id=participant,
        frame_idx=frame_idx,
        frame_time_sec=frame_idx / 60.0,
        fps=60.0,
        frame_path=str(img),
        width=100,
        height=80,
    )
    return PseudoLabel(frame=frame, bbox_xyxy=(10, 12, 30, 32), confidence=0.9)


def test_export_coco_dataset(tmp_path: Path):
    train_labels = [_label(tmp_path, "sv1", "clip1", 1)]
    val_labels = [_label(tmp_path, "sv2", "clip2", 2)]

    result = export_coco_dataset(
        train_labels=train_labels,
        val_labels=val_labels,
        output_dir=str(tmp_path / "out"),
        copy_images=True,
    )

    assert result.train_annotation_count == 1
    assert result.val_annotation_count == 1

    train_json = json.loads(Path(result.train_annotation_path).read_text(encoding="utf-8"))
    assert len(train_json["images"]) == 1
    assert len(train_json["annotations"]) == 1
    assert train_json["annotations"][0]["bbox"] == [10.0, 12.0, 20.0, 20.0]
