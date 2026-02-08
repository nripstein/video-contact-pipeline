from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from pipeline.preprocessing import get_sorted_image_list
from pipeline.visualization import load_gt_binary_from_csv, pred_binary_from_condensed

COLOR_HOLDING = (38, 140, 47)  # BGR
COLOR_NOT_HOLDING = (52, 28, 200)  # BGR
COLOR_TEXT = (32, 32, 32)
COLOR_PLAYHEAD = (0, 0, 0)


def load_pred_binary_from_condensed(condensed_csv: str) -> tuple[list[int], np.ndarray]:
    condensed_df = pd.read_csv(condensed_csv)
    ordered = condensed_df.sort_values(by=["frame_number"], kind="mergesort").reset_index(drop=True)
    frame_numbers = ordered["frame_number"].astype(int).tolist()
    pred = pred_binary_from_condensed(ordered).astype(np.uint8)
    return frame_numbers, pred


def align_secondary_prediction(
    secondary_condensed_csv: str,
    frame_numbers: Iterable[int],
) -> np.ndarray:
    secondary_df = pd.read_csv(secondary_condensed_csv)
    secondary_df = secondary_df.sort_values(by=["frame_number"], kind="mergesort").reset_index(drop=True)
    secondary_pred = pred_binary_from_condensed(secondary_df).astype(np.uint8)
    mapping = dict(zip(secondary_df["frame_number"].astype(int).tolist(), secondary_pred.tolist()))
    aligned = [int(mapping.get(int(frame_number), 0)) for frame_number in frame_numbers]
    return np.asarray(aligned, dtype=np.uint8)


def build_timeline_panel(
    frame_width: int,
    frame_index: int,
    tracks: list[np.ndarray],
    labels: list[str],
    left_label_width: int = 150,
    row_height: int = 28,
    row_gap: int = 6,
    top_pad: int = 10,
    bottom_pad: int = 10,
) -> np.ndarray:
    if not tracks:
        raise ValueError("tracks must be non-empty")
    if len(tracks) != len(labels):
        raise ValueError("tracks and labels must have the same length")

    timeline_width = max(1, frame_width - left_label_width)
    n_rows = len(tracks)
    panel_h = top_pad + bottom_pad + (n_rows * row_height) + ((n_rows - 1) * row_gap)
    panel = np.full((panel_h, frame_width, 3), 255, dtype=np.uint8)

    for row_idx, (track, label) in enumerate(zip(tracks, labels)):
        y0 = top_pad + row_idx * (row_height + row_gap)
        y1 = y0 + row_height

        if track.size == 0:
            continue

        x_edges = np.linspace(0, timeline_width, num=(track.size + 1), dtype=np.int32)
        for j in range(track.size):
            x0 = left_label_width + int(x_edges[j])
            x1 = left_label_width + int(x_edges[j + 1])
            color = COLOR_HOLDING if int(track[j]) == 1 else COLOR_NOT_HOLDING
            cv2.rectangle(panel, (x0, y0), (max(x0, x1 - 1), y1 - 1), color, thickness=-1)

        cv2.putText(
            panel,
            label,
            (10, y0 + int(row_height * 0.70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

    n_frames = tracks[0].size
    if n_frames > 0:
        ratio = min(max(frame_index / float(max(1, n_frames - 1)), 0.0), 1.0)
        x = left_label_width + int(round(ratio * (timeline_width - 1)))
        cv2.line(panel, (x, 0), (x, panel_h - 1), COLOR_PLAYHEAD, thickness=2)

    cv2.rectangle(panel, (0, 0), (frame_width - 1, panel_h - 1), (220, 220, 220), thickness=1)
    return panel


def compose_frame(
    frame: np.ndarray,
    timeline_panel: np.ndarray,
    title: str,
    frame_number: int,
    frame_index: int,
    n_frames: int,
) -> np.ndarray:
    frame_width = frame.shape[1]
    header_h = 56
    header = np.full((header_h, frame_width, 3), 255, dtype=np.uint8)
    header_text = f"{title} | frame={frame_number} ({frame_index + 1}/{n_frames})"
    cv2.putText(header, header_text, (12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT, 2, cv2.LINE_AA)
    return cv2.vconcat([frame, header, timeline_panel])


def render_contact_timeline_video(
    *,
    image_dir: str,
    frame_numbers: list[int],
    pred_binary: np.ndarray,
    output_video_path: str,
    fps: float = 30.0,
    title: str = "Contact Timeline",
    gt_binary: Optional[np.ndarray] = None,
    secondary_binary: Optional[np.ndarray] = None,
) -> str:
    image_paths = get_sorted_image_list(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in image_dir: {image_dir}")

    n = len(frame_numbers)
    if len(image_paths) < n:
        raise ValueError(
            f"Not enough images for predictions: images={len(image_paths)} predictions={n}. "
            f"image_dir={image_dir}"
        )
    if pred_binary.size != n:
        raise ValueError(f"pred_binary length mismatch: expected={n} got={pred_binary.size}")
    if gt_binary is not None and gt_binary.size != n:
        raise ValueError(f"gt_binary length mismatch: expected={n} got={gt_binary.size}")
    if secondary_binary is not None and secondary_binary.size != n:
        raise ValueError(f"secondary_binary length mismatch: expected={n} got={secondary_binary.size}")

    first = cv2.imread(image_paths[0])
    if first is None:
        raise ValueError(f"Failed to read frame image: {image_paths[0]}")
    frame_h, frame_w = first.shape[:2]

    labels = []
    tracks = []
    if gt_binary is not None:
        labels.append("GT")
        tracks.append(gt_binary.astype(np.uint8))
    labels.append("Pred")
    tracks.append(pred_binary.astype(np.uint8))
    if secondary_binary is not None:
        labels.append("Pred 2")
        tracks.append(secondary_binary.astype(np.uint8))

    panel_h = 10 + 10 + (len(tracks) * 28) + ((len(tracks) - 1) * 6)
    expected_size = (frame_w, frame_h + 56 + panel_h)
    output_video_path = str(Path(output_video_path).expanduser())
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), expected_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for: {output_video_path}")

    try:
        for i in range(n):
            image_path = image_paths[i]
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Failed to read frame image: {image_path}")
            timeline_panel = build_timeline_panel(
                frame_width=frame_w,
                frame_index=i,
                tracks=tracks,
                labels=labels,
            )
            composed = compose_frame(
                frame=frame,
                timeline_panel=timeline_panel,
                title=title,
                frame_number=int(frame_numbers[i]),
                frame_index=i,
                n_frames=n,
            )
            writer.write(composed)
    finally:
        writer.release()

    return output_video_path


def load_gt_binary_aligned(gt_csv_path: str, frame_numbers: list[int]) -> np.ndarray:
    return load_gt_binary_from_csv(gt_csv_path, frame_numbers).astype(np.uint8)
