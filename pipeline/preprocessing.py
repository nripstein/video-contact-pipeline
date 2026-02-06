from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv"}
DEFAULT_ZERO_PAD = 6
DEFAULT_FRAME_EXT = "png"


def _is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def resolve_input_to_image_dir(input_path: str, output_dir: str) -> str:
    """
    Resolve an input path (video or frames dir) into a directory of images.
    """
    input_path = str(Path(input_path).expanduser())
    output_dir = str(Path(output_dir).expanduser())

    if os.path.isdir(input_path):
        return input_path
    if os.path.isfile(input_path) and _is_video_file(input_path):
        return extract_frames(
            video_path=input_path,
            output_dir=output_dir,
            zero_pad=DEFAULT_ZERO_PAD,
            ext=DEFAULT_FRAME_EXT,
        )
    raise ValueError(f"Unsupported input path: {input_path}")


def extract_frames(video_path: str, output_dir: str, zero_pad: int = 6, ext: str = "png") -> str:
    """
    Extract all frames from a video file into output_dir/frames/<video_stem>/.
    """
    video_path = str(Path(video_path).expanduser())
    output_dir = str(Path(output_dir).expanduser())
    ext = ext.lstrip(".")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = Path(video_path).stem
    frames_dir = Path(output_dir) / "frames" / video_stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_index = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{frame_index:0{zero_pad}d}.{ext}"
        cv2.imwrite(str(frames_dir / filename), frame)
        frame_index += 1

    cap.release()
    return str(frames_dir)


def _compute_crop_box(
    original_width: int, original_height: int, square_size: Optional[int]
) -> Optional[Tuple[int, int, int, int]]:
    if square_size is None:
        return None
    if original_width < square_size or original_height < square_size:
        raise ValueError(
            f"Input smaller than crop size: "
            f"{original_width}x{original_height} < {square_size}x{square_size}"
        )
    x1 = (original_width - square_size) // 2
    y1 = (original_height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size
    return (x1, y1, x2, y2)


def crop_and_preprocess_directory(
    image_dir: str,
    output_dir: str,
    square_size: Optional[int],
    flip_vertical: bool,
) -> str:
    """
    Crop and/or flip images in a directory. Returns the directory of processed images.
    """
    image_dir = str(Path(image_dir).expanduser())
    output_dir = str(Path(output_dir).expanduser())

    if square_size is None and not flip_vertical:
        return image_dir

    image_list = get_sorted_image_list(image_dir)
    if not image_list:
        raise ValueError(f"No images found in {image_dir}")

    first_img = cv2.imread(image_list[0])
    if first_img is None:
        raise ValueError(f"Failed to read image: {image_list[0]}")

    original_height, original_width = first_img.shape[:2]
    crop_box = _compute_crop_box(original_width, original_height, square_size)

    stem = Path(image_dir).name
    processed_dir = Path(output_dir) / "processed_frames" / stem
    processed_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_list:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        h, w = img.shape[:2]
        if square_size is not None and (w < square_size or h < square_size):
            raise ValueError(
                f"Input smaller than crop size: {w}x{h} < {square_size}x{square_size}"
            )

        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            img = img[y1:y2, x1:x2]

        if flip_vertical:
            img = cv2.flip(img, 0)

        out_path = processed_dir / Path(img_path).name
        cv2.imwrite(str(out_path), img)

    return str(processed_dir)


def get_sorted_image_list(image_dir: str) -> List[str]:
    image_dir = str(Path(image_dir).expanduser())
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    entries = []
    for name in os.listdir(image_dir):
        if name.startswith("."):
            continue
        path = Path(image_dir) / name
        if path.is_file():
            entries.append(str(path))

    def _frame_key(path: str):
        name = Path(path).stem
        num = None
        current = ""
        for ch in name:
            if ch.isdigit():
                current += ch
            elif current:
                num = int(current)
                break
        if current and num is None:
            num = int(current)
        if num is not None:
            return (0, num, name)
        return (1, name, name)

    return sorted(entries, key=_frame_key)


def write_preprocessing_meta(
    output_dir: str,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
    crop_box: Optional[Tuple[int, int, int, int]],
    flip_vertical: bool,
    frame_ext: str,
    zero_pad: Optional[int],
) -> None:
    meta = {
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "crop_box": crop_box,
        "flip_vertical": flip_vertical,
        "frame_ext": frame_ext,
        "zero_pad": zero_pad,
    }
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "preprocessing_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
