from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from stimulus_detector.data_generation.types import PseudoLabel


@dataclass
class CocoExportResult:
    train_annotation_path: str
    val_annotation_path: str
    train_image_count: int
    val_image_count: int
    train_annotation_count: int
    val_annotation_count: int


def _sanitize_token(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    text = "".join(safe).strip("_")
    return text or "unknown"


def _build_coco_for_split(
    labels: List[PseudoLabel],
    images_dir: Path,
    annotations_path: Path,
    copy_images: bool,
) -> Tuple[int, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 1, "name": "stimulus", "supercategory": "object"}]

    image_id = 1
    ann_id = 1
    used_names: Dict[str, int] = {}

    for label in labels:
        src = Path(label.frame.frame_path)
        suffix = src.suffix.lower() or ".png"
        base_name = (
            f"{_sanitize_token(label.frame.participant_id)}"
            f"__{_sanitize_token(label.frame.video_id)}"
            f"__{int(label.frame.frame_idx):06d}{suffix}"
        )
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        if count > 0:
            file_name = f"{Path(base_name).stem}_{count}{suffix}"
        else:
            file_name = base_name

        dst = images_dir / file_name
        if copy_images:
            shutil.copy2(src, dst)

        x1, y1, x2, y2 = label.bbox_xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        images.append(
            {
                "id": image_id,
                "file_name": file_name if copy_images else str(src),
                "width": int(label.frame.width),
                "height": int(label.frame.height),
                "participant_id": label.frame.participant_id,
                "video_id": label.frame.video_id,
                "frame_idx": int(label.frame.frame_idx),
                "fps": float(label.frame.fps),
                "source_frame_path": str(src),
            }
        )

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "confidence": float(label.confidence),
                "source": label.source,
            }
        )

        image_id += 1
        ann_id += 1

    coco = {
        "info": {"description": "Stimulus detector auto-generated dataset (Phase 1)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    annotations_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    return len(images), len(annotations)


def export_coco_dataset(
    train_labels: List[PseudoLabel],
    val_labels: List[PseudoLabel],
    output_dir: str,
    copy_images: bool = True,
) -> CocoExportResult:
    base = Path(output_dir).expanduser()
    train_images_dir = base / "dataset" / "train" / "images"
    val_images_dir = base / "dataset" / "val" / "images"
    ann_dir = base / "dataset" / "annotations"

    train_ann_path = ann_dir / "train_coco.json"
    val_ann_path = ann_dir / "val_coco.json"

    train_image_count, train_ann_count = _build_coco_for_split(
        labels=train_labels,
        images_dir=train_images_dir,
        annotations_path=train_ann_path,
        copy_images=copy_images,
    )
    val_image_count, val_ann_count = _build_coco_for_split(
        labels=val_labels,
        images_dir=val_images_dir,
        annotations_path=val_ann_path,
        copy_images=copy_images,
    )

    return CocoExportResult(
        train_annotation_path=str(train_ann_path),
        val_annotation_path=str(val_ann_path),
        train_image_count=train_image_count,
        val_image_count=val_image_count,
        train_annotation_count=train_ann_count,
        val_annotation_count=val_ann_count,
    )
