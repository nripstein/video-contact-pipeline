from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class PipelineConfig:
    input_path: str
    output_dir: Optional[str]

    net: str = "res101"
    cfg_file: str = "cfgs/res101.yml"
    load_dir: str = "models"
    checksession: int = 1
    checkepoch: int = 8
    checkpoint: int = 132028
    cuda: bool = True

    thresh_hand: float = 0.5
    thresh_obj: float = 0.5

    crop_square: Optional[int] = 480
    flip_vertical: bool = True

    blue_glove_filter: bool = True
    blue_threshold: float = 0.5

    object_size_filter: bool = True
    object_size_max_area_ratio: float = 0.5

    obj_bigger_than_hand_filter: bool = False
    obj_bigger_ratio_k: float = 1.0

    obj_match_tiebreak: str = "conf_then_dist_then_iou"

    save_full_csv: bool = True
    save_condensed_csv: bool = True
    save_visualizations: bool = True
    gt_csv_path: Optional[str] = None
    save_config: bool = True
    show_progress: bool = True
    save_annotated_frames: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def save(self, path: "PathLike") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: "PathLike") -> "PipelineConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def resolve_output_dir(self) -> str:
        if self.output_dir:
            return str(Path(self.output_dir).expanduser())
        parent = Path(self.input_path).expanduser().parent
        return str(parent / "pipeline_output")


PathLike = Union[str, Path]
