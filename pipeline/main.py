from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import cv2

from pipeline.config import PipelineConfig
import pandas as pd

from pipeline.inference import HandObjectDetector
from pipeline.postprocessing import apply_detection_filters, condense_dataframe
from pipeline.preprocessing import (
    DEFAULT_FRAME_EXT,
    DEFAULT_ZERO_PAD,
    crop_and_preprocess_directory,
    get_sorted_image_list,
    resolve_input_to_image_dir,
    write_preprocessing_meta,
)
from pipeline.visualization import save_barcodes


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


def _read_first_image_dims(image_dir: str) -> Tuple[int, int]:
    image_list = get_sorted_image_list(image_dir)
    if not image_list:
        raise ValueError(f"No images found in {image_dir}")
    first_img = cv2.imread(image_list[0])
    if first_img is None:
        raise ValueError(f"Failed to read image: {image_list[0]}")
    height, width = first_img.shape[:2]
    return width, height


def _infer_frame_ext(image_dir: str) -> str:
    image_list = get_sorted_image_list(image_dir)
    if not image_list:
        raise ValueError(f"No images found in {image_dir}")
    return Path(image_list[0]).suffix.lstrip(".")


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def list_videos(input_dir: str) -> list[str]:
    input_dir = str(Path(input_dir).expanduser())
    if not Path(input_dir).is_dir():
        return []
    vids = []
    for entry in sorted(Path(input_dir).iterdir()):
        if entry.is_file() and is_video_file(str(entry)):
            vids.append(str(entry))
    return vids


def prepare_frames(config: PipelineConfig) -> str:
    output_dir = Path(config.resolve_output_dir()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(config.input_path).expanduser()
    is_video = input_path.is_file() and is_video_file(str(input_path))

    image_dir = resolve_input_to_image_dir(str(input_path), str(output_dir))
    processed_dir = crop_and_preprocess_directory(
        image_dir=image_dir,
        output_dir=str(output_dir),
        square_size=config.crop_square,
        flip_vertical=config.flip_vertical,
    )

    original_width, original_height = _read_first_image_dims(image_dir)
    processed_width, processed_height = _read_first_image_dims(processed_dir)
    crop_box = _compute_crop_box(original_width, original_height, config.crop_square)

    frame_ext = DEFAULT_FRAME_EXT if is_video else _infer_frame_ext(image_dir)
    zero_pad = DEFAULT_ZERO_PAD if is_video else None

    write_preprocessing_meta(
        output_dir=str(output_dir),
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
        crop_box=crop_box,
        flip_vertical=config.flip_vertical,
        frame_ext=frame_ext,
        zero_pad=zero_pad,
    )

    return processed_dir


def _sort_full_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_keys = [
        "frame_number",
        "detection_type",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "confidence",
    ]
    existing = [k for k in sort_keys if k in df.columns]
    return df.sort_values(by=existing, kind="mergesort").reset_index(drop=True)


def run_inference(config: PipelineConfig, image_dir: str, detector: Optional[HandObjectDetector] = None) -> pd.DataFrame:
    inference_config = replace(config, blue_glove_filter=False)
    if detector is None:
        detector = HandObjectDetector(inference_config)
    return detector.run_on_directory(image_dir)


def run_single_input(
    config: PipelineConfig,
    input_path: str,
    detector: Optional[HandObjectDetector] = None,
    do_condense: bool = True,
):
    single_config = replace(config, input_path=input_path)
    image_dir = prepare_frames(single_config)
    full_df = run_inference(single_config, image_dir, detector=detector)
    full_df = apply_detection_filters(full_df, single_config, image_dir)
    condensed_df = condense_dataframe(full_df) if do_condense else None

    output_dir = Path(single_config.resolve_output_dir()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if single_config.save_full_csv:
        full_sorted = _sort_full_df(full_df)
        full_path = output_dir / "detections_full.csv"
        full_sorted.to_csv(full_path, index=False)

    if do_condense and single_config.save_condensed_csv and condensed_df is not None:
        condensed_sorted = condensed_df.sort_values(
            by=["frame_number", "frame_id"], kind="mergesort"
        ).reset_index(drop=True)
        condensed_path = output_dir / "detections_condensed.csv"
        condensed_sorted.to_csv(condensed_path, index=False)

    if single_config.save_config:
        config_path = output_dir / "config.json"
        single_config.save(config_path)

    if single_config.save_visualizations and condensed_df is not None:
        save_barcodes(condensed_df, str(output_dir), gt_csv_path=single_config.gt_csv_path)

    return full_df, condensed_df


def run_pipeline(config: PipelineConfig, do_condense: bool = True, detector_factory=None):
    input_path = Path(config.input_path).expanduser()
    output_dir = Path(config.resolve_output_dir()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        videos = list_videos(str(input_path))
        if videos:
            detector_factory = detector_factory or (lambda c: HandObjectDetector(c))
            detector = detector_factory(replace(config, blue_glove_filter=False))
            results = []
            for video_path in videos:
                video_stem = Path(video_path).stem
                per_output = output_dir / video_stem
                per_config = replace(config, input_path=video_path, output_dir=str(per_output))
                results.append(run_single_input(per_config, video_path, detector=detector, do_condense=do_condense))
            return results

    return run_single_input(config, str(input_path), detector=None, do_condense=do_condense)
