from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.exporters import CocoExportResult, export_coco_dataset
from stimulus_detector.data_generation.filters import build_filter, build_filter_context
from stimulus_detector.data_generation.frame_extractor import (
    discover_sequence_inputs,
    extract_participant_id,
    extract_video_frames,
    load_frames_directory,
)
from stimulus_detector.data_generation.heuristics import apply_heuristic_filters
from stimulus_detector.data_generation.shan_wrapper import ShanInferenceWrapper
from stimulus_detector.data_generation.splitter import split_labels_by_participant
from stimulus_detector.data_generation.stats import write_dataset_stats
from stimulus_detector.data_generation.types import ObjectDetection, PseudoLabel, SequenceFrames
from stimulus_detector.data_generation.visualize import save_pose_diversity_heatmap, save_selection_overlays


@dataclass
class Phase1RunResult:
    output_dir: str
    raw_csv: str
    heuristics_csv: str
    selected_csv: str
    train_coco_json: str
    val_coco_json: str
    train_image_count: int
    val_image_count: int
    train_annotation_count: int
    val_annotation_count: int
    selected_count: int
    train_count: int
    val_count: int


def _object_rows(detections: List[ObjectDetection]) -> List[Dict[str, object]]:
    rows = []
    for obj in detections:
        rows.append(
            {
                "participant_id": obj.frame.participant_id,
                "video_id": obj.frame.video_id,
                "frame_idx": obj.frame.frame_idx,
                "frame_time_sec": obj.frame.frame_time_sec,
                "fps": obj.frame.fps,
                "frame_path": obj.frame.frame_path,
                "frame_name": Path(obj.frame.frame_path).name,
                "frame_width": obj.frame.width,
                "frame_height": obj.frame.height,
                "bbox_x1": obj.bbox_xyxy[0],
                "bbox_y1": obj.bbox_xyxy[1],
                "bbox_x2": obj.bbox_xyxy[2],
                "bbox_y2": obj.bbox_xyxy[3],
                "confidence": obj.confidence,
                "source": obj.source,
            }
        )
    return rows


def _pseudo_label_rows(labels: List[PseudoLabel]) -> List[Dict[str, object]]:
    rows = []
    for label in labels:
        rows.append(
            {
                "participant_id": label.frame.participant_id,
                "video_id": label.frame.video_id,
                "frame_idx": label.frame.frame_idx,
                "frame_time_sec": label.frame.frame_time_sec,
                "fps": label.frame.fps,
                "frame_path": label.frame.frame_path,
                "frame_name": Path(label.frame.frame_path).name,
                "frame_width": label.frame.width,
                "frame_height": label.frame.height,
                "bbox_x1": label.bbox_xyxy[0],
                "bbox_y1": label.bbox_xyxy[1],
                "bbox_x2": label.bbox_xyxy[2],
                "bbox_y2": label.bbox_xyxy[3],
                "confidence": label.confidence,
                "matched_hand_area": label.matched_hand_area,
                "matched_hand_iou": label.matched_hand_iou,
                "matched_hand_center_dist": label.matched_hand_center_dist,
                "filter_trace": "|".join(label.filter_trace),
                "source": label.source,
            }
        )
    return rows


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _build_sequences(config: Phase1Config, output_dir: Path) -> List[SequenceFrames]:
    inputs = discover_sequence_inputs(config.input_path)
    frames_root = output_dir / "intermediate" / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    sequences: List[SequenceFrames] = []
    for seq_in in inputs:
        participant_id = extract_participant_id(seq_in.path, config.participant_regex)
        if seq_in.kind == "video":
            seq = extract_video_frames(seq_in.path, frames_root, config, participant_id)
        else:
            seq = load_frames_directory(seq_in.path, config, participant_id)
        sequences.append(seq)
    return sequences


def run_phase1_data_generation(config: Phase1Config) -> Phase1RunResult:
    output_dir = config.resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _build_sequences(config, output_dir)

    wrapper = ShanInferenceWrapper(config)

    raw_objects: List[ObjectDetection] = []
    all_hands = []
    for sequence in sequences:
        objects, hands = wrapper.run_on_sequence(sequence)
        raw_objects.extend(objects)
        all_hands.extend(hands)

    raw_csv_path = output_dir / "intermediate" / "detections_raw.csv"
    _write_csv(_object_rows(raw_objects), raw_csv_path)

    labels_after_heuristics: List[PseudoLabel] = []
    heuristics_rows: List[Dict[str, object]] = []
    for sequence in sequences:
        sequence_objects = [
            obj
            for obj in raw_objects
            if obj.frame.video_id == sequence.video_id and obj.frame.participant_id == sequence.participant_id
        ]
        sequence_hands = [
            hand
            for hand in all_hands
            if hand.frame.video_id == sequence.video_id and hand.frame.participant_id == sequence.participant_id
        ]
        seq_kept, seq_rows = apply_heuristic_filters(sequence_objects, sequence_hands, config)
        labels_after_heuristics.extend(seq_kept)
        heuristics_rows.extend(seq_rows)

    heuristics_csv_path = output_dir / "intermediate" / "detections_after_heuristics.csv"
    _write_csv(heuristics_rows, heuristics_csv_path)

    filter_impl = build_filter(config)
    filter_context = build_filter_context(config)

    selected_labels: List[PseudoLabel] = []
    for sequence in sequences:
        seq_labels = [
            lbl
            for lbl in labels_after_heuristics
            if lbl.frame.video_id == sequence.video_id and lbl.frame.participant_id == sequence.participant_id
        ]
        selected_labels.extend(filter_impl.filter(seq_labels, filter_context))

    selected_labels = sorted(
        selected_labels,
        key=lambda x: (
            x.frame.participant_id,
            x.frame.video_id,
            x.frame.frame_idx,
            Path(x.frame.frame_path).name,
        ),
    )

    selected_csv_path = output_dir / "intermediate" / "detections_selected.csv"
    _write_csv(_pseudo_label_rows(selected_labels), selected_csv_path)

    split_result = split_labels_by_participant(
        selected_labels,
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )

    if config.export_format.lower() != "coco":
        raise ValueError("Phase 1 currently supports export_format='coco' only.")

    export_result: CocoExportResult = export_coco_dataset(
        train_labels=split_result.train_labels,
        val_labels=split_result.val_labels,
        output_dir=str(output_dir),
        copy_images=config.copy_images,
    )

    if config.generate_visualizations:
        save_selection_overlays(
            labels=selected_labels,
            output_dir=str(output_dir),
            sample_size=config.overlay_sample_size,
        )
        save_pose_diversity_heatmap(
            labels=selected_labels,
            output_dir=str(output_dir),
            bins=config.heatmap_bins,
        )

    if config.generate_stats:
        write_dataset_stats(
            raw_count=len(raw_objects),
            heuristics_pass_count=len(labels_after_heuristics),
            selected_labels=selected_labels,
            split_result=split_result,
            output_dir=str(output_dir),
            bins=config.heatmap_bins,
        )

    run_summary = {
        "output_dir": str(output_dir),
        "selected_count": len(selected_labels),
        "train_count": len(split_result.train_labels),
        "val_count": len(split_result.val_labels),
        "train_coco_json": export_result.train_annotation_path,
        "val_coco_json": export_result.val_annotation_path,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    return Phase1RunResult(
        output_dir=str(output_dir),
        raw_csv=str(raw_csv_path),
        heuristics_csv=str(heuristics_csv_path),
        selected_csv=str(selected_csv_path),
        train_coco_json=export_result.train_annotation_path,
        val_coco_json=export_result.val_annotation_path,
        train_image_count=export_result.train_image_count,
        val_image_count=export_result.val_image_count,
        train_annotation_count=export_result.train_annotation_count,
        val_annotation_count=export_result.val_annotation_count,
        selected_count=len(selected_labels),
        train_count=len(split_result.train_labels),
        val_count=len(split_result.val_labels),
    )
