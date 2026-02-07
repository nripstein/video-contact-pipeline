from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    participant_id: str
    frame_idx: int
    frame_time_sec: float
    fps: float
    frame_path: str
    width: int
    height: int


@dataclass
class ObjectDetection:
    frame: FrameRecord
    bbox_xyxy: BBox
    confidence: float
    source: str = "shan_targetobject"

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        h = max(1e-6, y2 - y1)
        return max(0.0, x2 - x1) / h


@dataclass
class HandDetection:
    frame: FrameRecord
    bbox_xyxy: BBox
    confidence: float

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class PseudoLabel:
    frame: FrameRecord
    bbox_xyxy: BBox
    confidence: float
    source: str = "shan_targetobject"
    matched_hand_area: Optional[float] = None
    matched_hand_iou: Optional[float] = None
    matched_hand_center_dist: Optional[float] = None
    filter_trace: List[str] = field(default_factory=list)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        h = max(1e-6, y2 - y1)
        return max(0.0, x2 - x1) / h


@dataclass(frozen=True)
class FilterContext:
    min_temporal_gap_sec: float
    center_move_frac: float
    area_change_frac: float
    subsample_interval_sec: float


@dataclass(frozen=True)
class SequenceInput:
    kind: str
    path: str


@dataclass
class SequenceFrames:
    video_id: str
    participant_id: str
    fps: float
    frames_dir: str
    frame_records: List[FrameRecord]


@dataclass
class SplitResult:
    participant_to_split: dict
    train_labels: List[PseudoLabel]
    val_labels: List[PseudoLabel]
