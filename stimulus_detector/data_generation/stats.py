from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stimulus_detector.data_generation.types import PseudoLabel, SplitResult


def _summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _center_entropy_bits(labels: List[PseudoLabel], bins: int) -> float:
    if not labels:
        return 0.0

    xs = []
    ys = []
    for label in labels:
        cx, cy = label.center()
        w = max(1.0, float(label.frame.width))
        h = max(1.0, float(label.frame.height))
        xs.append(min(max(cx / w, 0.0), 1.0))
        ys.append(min(max(cy / h, 0.0), 1.0))

    hist, _, _ = np.histogram2d(np.array(xs), np.array(ys), bins=bins, range=[[0, 1], [0, 1]])
    p = hist / np.sum(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def write_dataset_stats(
    raw_count: int,
    heuristics_pass_count: int,
    selected_labels: List[PseudoLabel],
    split_result: SplitResult,
    output_dir: str,
    bins: int,
) -> Dict[str, str]:
    out_dir = Path(output_dir).expanduser() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    confidences = np.array([label.confidence for label in selected_labels], dtype=float)
    areas = np.array([label.area() for label in selected_labels], dtype=float)
    aspect_ratios = np.array([label.aspect_ratio() for label in selected_labels], dtype=float)
    fps_values = [round(float(label.frame.fps), 3) for label in selected_labels]

    participant_rows = []
    per_participant: Dict[str, int] = {}
    for label in selected_labels:
        pid = label.frame.participant_id
        per_participant[pid] = per_participant.get(pid, 0) + 1

    for pid, count in sorted(per_participant.items()):
        split = split_result.participant_to_split.get(pid, "train")
        participant_rows.append({"participant_id": pid, "selected_frames": int(count), "split": split})

    part_df = pd.DataFrame(participant_rows)
    part_csv_path = out_dir / "dataset_stats_by_participant.csv"
    if part_df.empty:
        part_df = pd.DataFrame(columns=["participant_id", "selected_frames", "split"])
    part_df.to_csv(part_csv_path, index=False)

    stats = {
        "counts": {
            "raw_candidates": int(raw_count),
            "after_heuristics": int(heuristics_pass_count),
            "selected_final": int(len(selected_labels)),
            "train_selected": int(len(split_result.train_labels)),
            "val_selected": int(len(split_result.val_labels)),
        },
        "participants": {
            "n_participants": int(len(split_result.participant_to_split)),
            "participant_to_split": split_result.participant_to_split,
        },
        "confidence": _summary(confidences),
        "bbox_area": _summary(areas),
        "aspect_ratio": _summary(aspect_ratios),
        "fps_distribution": {str(k): int(v) for k, v in pd.Series(fps_values).value_counts().sort_index().items()} if fps_values else {},
        "pose_diversity": {
            "center_entropy_bits": _center_entropy_bits(selected_labels, bins=bins),
            "heatmap_bins": int(bins),
        },
    }

    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {
        "stats_json": str(stats_path),
        "stats_by_participant_csv": str(part_csv_path),
    }
