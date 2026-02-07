from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Phase1Config:
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
    show_progress: bool = True

    frame_ext: str = "png"
    zero_pad: int = 6
    default_fps_if_unknown: float = 60.0

    participant_regex: str = r"(?i)(sv\d+)"
    val_fraction: float = 0.2
    split_seed: int = 42

    min_confidence: float = 0.7
    aspect_ratio_min: float = 0.8
    aspect_ratio_max: float = 1.2
    require_hand_for_size: bool = True
    max_object_to_hand_area_ratio: float = 1.0
    max_hand_occlusion_ratio: float = 1.0

    filter_strategy: str = "bbox_similarity"
    min_temporal_gap_sec: float = 0.25
    center_move_frac: float = 0.10
    area_change_frac: float = 0.20
    subsample_interval_sec: float = 0.25

    export_format: str = "coco"
    copy_images: bool = True
    generate_visualizations: bool = True
    generate_stats: bool = True
    overlay_sample_size: int = 100
    heatmap_bins: int = 5

    save_intermediate_csvs: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1Config":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str) -> "Phase1Config":
        cfg_path = Path(path).expanduser()
        suffix = cfg_path.suffix.lower()
        text = cfg_path.read_text(encoding="utf-8")
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config requested but PyYAML is not installed."
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def resolve_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir).expanduser() / "phase1"
        parent = Path(self.input_path).expanduser().parent
        return parent / "stimulus_detector_output" / "phase1"
