from __future__ import annotations

import re
from pathlib import Path
from typing import List

import cv2

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.types import FrameRecord, SequenceFrames, SequenceInput

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def parse_frame_index(name: str, fallback: int) -> int:
    stem = Path(name).stem
    match = re.search(r"(\d+)", stem)
    if not match:
        return fallback
    return int(match.group(1))


def _sorted_image_paths(frame_dir: Path) -> List[Path]:
    image_paths = [p for p in frame_dir.iterdir() if p.is_file() and _is_image_file(p)]
    return sorted(image_paths, key=lambda p: (parse_frame_index(p.name, 10**12), p.name))


def _dir_has_images(path: Path) -> bool:
    return any(child.is_file() and _is_image_file(child) for child in path.iterdir())


def discover_sequence_inputs(input_path: str) -> List[SequenceInput]:
    root = Path(input_path).expanduser()
    if root.is_file():
        if not _is_video_file(root):
            raise ValueError(f"Unsupported input file: {root}")
        return [SequenceInput(kind="video", path=str(root))]

    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    video_paths = sorted([p for p in root.rglob("*") if p.is_file() and _is_video_file(p)])
    if video_paths:
        return [SequenceInput(kind="video", path=str(p)) for p in video_paths]

    frame_dirs: List[Path] = []
    if _dir_has_images(root):
        frame_dirs = [root]
    else:
        for subdir in sorted([p for p in root.rglob("*") if p.is_dir()]):
            if _dir_has_images(subdir):
                frame_dirs.append(subdir)

    if not frame_dirs:
        raise ValueError(f"No videos or frame directories found under: {root}")

    return [SequenceInput(kind="frames", path=str(p)) for p in frame_dirs]


def extract_participant_id(path_str: str, participant_regex: str) -> str:
    path = Path(path_str)
    match = re.search(participant_regex, str(path))
    if match:
        return match.group(1)
    return path.parent.name if path.parent.name else "unknown"


def extract_video_frames(
    video_path: str,
    output_frames_root: Path,
    config: Phase1Config,
    participant_id: str,
) -> SequenceFrames:
    path = Path(video_path).expanduser()
    video_id = path.stem
    frames_dir = output_frames_root / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = float(config.default_fps_if_unknown)

    frame_records: List[FrameRecord] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        filename = f"{frame_idx:0{config.zero_pad}d}.{config.frame_ext}"
        frame_path = frames_dir / filename
        cv2.imwrite(str(frame_path), frame)
        height, width = frame.shape[:2]
        frame_records.append(
            FrameRecord(
                video_id=video_id,
                participant_id=participant_id,
                frame_idx=frame_idx,
                frame_time_sec=(frame_idx / fps),
                fps=fps,
                frame_path=str(frame_path),
                width=int(width),
                height=int(height),
            )
        )
        frame_idx += 1

    cap.release()
    return SequenceFrames(
        video_id=video_id,
        participant_id=participant_id,
        fps=fps,
        frames_dir=str(frames_dir),
        frame_records=frame_records,
    )


def load_frames_directory(
    frame_dir: str,
    config: Phase1Config,
    participant_id: str,
) -> SequenceFrames:
    path = Path(frame_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {path}")

    fps = float(config.default_fps_if_unknown)
    image_paths = _sorted_image_paths(path)
    if not image_paths:
        raise ValueError(f"No images found in frame directory: {path}")

    video_id = path.name
    frame_records: List[FrameRecord] = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        frame_idx = parse_frame_index(img_path.name, idx)
        frame_records.append(
            FrameRecord(
                video_id=video_id,
                participant_id=participant_id,
                frame_idx=frame_idx,
                frame_time_sec=(frame_idx / fps),
                fps=fps,
                frame_path=str(img_path),
                width=int(width),
                height=int(height),
            )
        )

    frame_records = sorted(frame_records, key=lambda r: (r.frame_idx, Path(r.frame_path).name))

    return SequenceFrames(
        video_id=video_id,
        participant_id=participant_id,
        fps=fps,
        frames_dir=str(path),
        frame_records=frame_records,
    )
