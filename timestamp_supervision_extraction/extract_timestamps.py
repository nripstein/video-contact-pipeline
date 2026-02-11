from __future__ import annotations

import argparse
import logging
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("timestamp_supervision_extraction")

PREDICTIONS_FILENAME = "predictions.csv"
METADATA_FILENAME = "metadata.csv"

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

OUTPUT_COLUMNS = [
    "video_id",
    "frame_id",
    "predicted_label",
    "island_start_frame_id",
    "island_end_frame_id",
    "island_length_frames",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one representative timestamp per high-confidence island from "
            "per-frame predictions."
        )
    )
    parser.add_argument("--results_dir", required=True, help="Root directory containing results/**/predictions.csv")
    parser.add_argument("--videos_dir", default="videos", help="Directory containing source videos named <video_id>.mp4")
    parser.add_argument(
        "--frames_dir",
        default=None,
        help=(
            "Optional root directory of pre-extracted frames. "
            "Expected either <frames_dir>/<video_id>/* or a single flat frame directory "
            "for single-video runs."
        ),
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write CSV outputs and optional frames")
    parser.add_argument("--fps", type=float, default=60.0, help="FPS used to convert minimum island seconds to frames")
    parser.add_argument(
        "--min_island_seconds",
        type=float,
        default=1.0,
        help="Minimum stable island duration in seconds",
    )
    parser.add_argument(
        "--join_mode",
        choices=("inner", "left"),
        default="inner",
        help="Join mode for merging predictions.csv and metadata.csv on frame_id",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Retained for CLI compatibility; currently ignored because selection is deterministic.",
    )
    parser.add_argument(
        "--extract_frames",
        action="store_true",
        help="Extract selected timestamp frames from source videos",
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "ffmpeg"),
        default="opencv",
        help="Frame extraction backend to use with --extract_frames",
    )
    parser.add_argument(
        "--image_format",
        default="jpg",
        help="Output image extension (without dot), e.g., jpg or png",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _require_columns(df: pd.DataFrame, required: Sequence[str], csv_path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def _coerce_int_series(series: pd.Series, column_name: str, csv_path: Path) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    invalid_mask = coerced.isna()
    if invalid_mask.any():
        bad_values = series[invalid_mask].head(5).tolist()
        raise ValueError(
            f"{csv_path} has non-integer values in column '{column_name}'. "
            f"Examples: {bad_values}"
        )
    return coerced.astype(np.int64)


def _parse_bool_value(value: Any, csv_path: Path) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return bool(int(value))
    string_value = str(value).strip().lower()
    if string_value in TRUE_VALUES:
        return True
    if string_value in FALSE_VALUES:
        return False
    raise ValueError(
        f"{csv_path} contains invalid boolean value for 'blue_glove_detected': {value!r}"
    )


def _coerce_bool_series(series: pd.Series, csv_path: Path) -> pd.Series:
    return series.map(lambda value: _parse_bool_value(value, csv_path))


def discover_prediction_csvs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"results_dir is not a directory: {results_dir}")

    prediction_paths = sorted(results_dir.rglob(PREDICTIONS_FILENAME))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No '{PREDICTIONS_FILENAME}' files found under {results_dir}. "
            "Expected layout: results/**/predictions.csv with sibling metadata.csv"
        )
    return prediction_paths


def load_predictions_csv(predictions_csv: Path) -> pd.DataFrame:
    predictions_df = pd.read_csv(predictions_csv)
    _require_columns(predictions_df, ("frame_id", "predicted_label"), predictions_csv)

    predictions_df = predictions_df[["frame_id", "predicted_label"]].copy()
    predictions_df["frame_id"] = _coerce_int_series(predictions_df["frame_id"], "frame_id", predictions_csv)
    predictions_df["predicted_label"] = _coerce_int_series(
        predictions_df["predicted_label"], "predicted_label", predictions_csv
    )

    label_mask = predictions_df["predicted_label"].isin((0, 1))
    if not label_mask.all():
        bad_values = predictions_df.loc[~label_mask, "predicted_label"].head(5).tolist()
        raise ValueError(
            f"{predictions_csv} contains invalid predicted_label values. "
            f"Expected {{0,1}}; examples: {bad_values}"
        )
    return predictions_df


def load_metadata_csv(metadata_csv: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_csv)
    _require_columns(metadata_df, ("frame_id", "blue_glove_detected"), metadata_csv)

    metadata_df = metadata_df[["frame_id", "blue_glove_detected"]].copy()
    metadata_df["frame_id"] = _coerce_int_series(metadata_df["frame_id"], "frame_id", metadata_csv)
    metadata_df["blue_glove_detected"] = _coerce_bool_series(metadata_df["blue_glove_detected"], metadata_csv)
    return metadata_df


def merge_predictions_and_metadata(
    predictions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    join_mode: str = "inner",
) -> pd.DataFrame:
    merged = predictions_df.merge(metadata_df, on="frame_id", how=join_mode, sort=False)
    if join_mode == "left":
        merged["blue_glove_detected"] = merged["blue_glove_detected"].fillna(False)

    missing_blue = merged["blue_glove_detected"].isna()
    if missing_blue.any():
        raise ValueError(
            "Found missing blue_glove_detected values after merge. "
            "Use --join_mode left to keep missing metadata rows and default them to False."
        )

    merged["blue_glove_detected"] = merged["blue_glove_detected"].astype(bool)
    merged = merged.sort_values(by=["frame_id"], kind="mergesort").reset_index(drop=True)
    return merged


def find_islands(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    ordered = df.sort_values(by=["frame_id"], kind="mergesort").reset_index(drop=True)
    frame_ids = ordered["frame_id"].to_numpy(dtype=np.int64)
    labels = ordered["predicted_label"].to_numpy(dtype=np.int64)
    blue_glove = ordered["blue_glove_detected"].to_numpy(dtype=bool)

    islands: List[Dict[str, Any]] = []
    start_idx = 0

    def append_island(start: int, end: int) -> None:
        island_frame_ids = frame_ids[start : end + 1]
        island_blue = blue_glove[start : end + 1]
        islands.append(
            {
                "predicted_label": int(labels[start]),
                "island_start_frame_id": int(island_frame_ids[0]),
                "island_end_frame_id": int(island_frame_ids[-1]),
                "island_length_frames": int(len(island_frame_ids)),
                "frame_ids": [int(frame_id) for frame_id in island_frame_ids.tolist()],
                "has_blue_glove_true": bool(np.any(island_blue)),
            }
        )

    for idx in range(1, len(ordered)):
        same_label = labels[idx] == labels[idx - 1]
        strictly_contiguous = (frame_ids[idx] - frame_ids[idx - 1]) == 1
        if not (same_label and strictly_contiguous):
            append_island(start_idx, idx - 1)
            start_idx = idx

    append_island(start_idx, len(ordered) - 1)
    return islands


def filter_islands(
    islands: Sequence[Dict[str, Any]],
    fps: float,
    min_seconds: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    min_length_frames = int(math.ceil(min_seconds * fps))

    length_filtered = [
        island for island in islands if island["island_length_frames"] >= min_length_frames
    ]

    blue_glove_removed = 0
    kept: List[Dict[str, Any]] = []
    for island in length_filtered:
        if island["predicted_label"] == 1 and island["has_blue_glove_true"]:
            blue_glove_removed += 1
            continue
        kept.append(island)

    summary = {
        "num_islands_found": int(len(islands)),
        "num_passed_length": int(len(length_filtered)),
        "num_removed_blue_glove": int(blue_glove_removed),
        "num_selected": int(len(kept)),
    }
    return kept, summary


def _select_middle_index(length: int) -> int:
    if length <= 0:
        raise ValueError("Island length must be positive")
    return int(length // 2)


def select_central_frames(
    islands: Sequence[Dict[str, Any]],
    rng: np.random.Generator | None = None,
) -> List[Dict[str, Any]]:
    # Compatibility no-op: selection is deterministic and does not use randomness.
    _ = rng

    selections: List[Dict[str, Any]] = []
    for island in islands:
        frame_ids = island["frame_ids"]
        selected_idx = _select_middle_index(len(frame_ids))
        selected_frame_id = int(frame_ids[selected_idx])
        selections.append(
            {
                "video_id": island["video_id"],
                "frame_id": selected_frame_id,
                "predicted_label": int(island["predicted_label"]),
                "island_start_frame_id": int(island["island_start_frame_id"]),
                "island_end_frame_id": int(island["island_end_frame_id"]),
                "island_length_frames": int(island["island_length_frames"]),
            }
        )
    return selections


def _build_video_path(videos_dir: Path, video_id: str) -> Path:
    return videos_dir / f"{video_id}.mp4"


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _dir_has_images(path: Path) -> bool:
    return path.is_dir() and any(_is_image_file(child) for child in path.iterdir())


def _parse_frame_index_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)", Path(name).stem)
    if not match:
        return None
    return int(match.group(1))


def _index_frames_in_dir(frames_path: Path) -> Dict[int, Path]:
    index: Dict[int, Path] = {}
    for child in frames_path.iterdir():
        if not _is_image_file(child):
            continue
        frame_idx = _parse_frame_index_from_name(child.name)
        if frame_idx is None:
            continue
        existing = index.get(frame_idx)
        if existing is None or child.name < existing.name:
            index[frame_idx] = child
    return index


def _resolve_frame_source_dir(
    frames_root: Path,
    video_id: str,
    num_selected_videos: int,
) -> Path | None:
    candidate = frames_root / video_id
    if _dir_has_images(candidate):
        return candidate

    if num_selected_videos == 1 and _dir_has_images(frames_root):
        return frames_root
    return None


def _record_extraction_error(
    stats: Dict[str, Any],
    video_id: str,
    frame_id: int,
    reason: str,
) -> None:
    stats["failed"] += 1
    stats["errors"].append(
        {
            "video_id": video_id,
            "frame_id": int(frame_id),
            "reason": reason,
        }
    )


def _extract_with_opencv(
    by_video: Dict[str, List[Dict[str, Any]]],
    videos_dir: Path,
    extracted_root: Path,
    image_format: str,
    stats: Dict[str, Any],
) -> None:
    import cv2

    for video_id, records in by_video.items():
        video_path = _build_video_path(videos_dir, video_id)
        if not video_path.exists():
            for record in records:
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=int(record["frame_id"]),
                    reason=f"Video not found: {video_path}",
                )
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            for record in records:
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=int(record["frame_id"]),
                    reason=f"Failed to open video with OpenCV: {video_path}",
                )
            continue

        try:
            for record in records:
                frame_id = int(record["frame_id"])
                video_out_dir = extracted_root / video_id
                video_out_dir.mkdir(parents=True, exist_ok=True)
                output_path = video_out_dir / f"frame_{frame_id}.{image_format}"

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok or frame is None:
                    _record_extraction_error(
                        stats,
                        video_id=video_id,
                        frame_id=frame_id,
                        reason=f"Failed to read frame {frame_id} from {video_path}",
                    )
                    continue

                write_ok = cv2.imwrite(str(output_path), frame)
                if not write_ok:
                    _record_extraction_error(
                        stats,
                        video_id=video_id,
                        frame_id=frame_id,
                        reason=f"Failed to write image: {output_path}",
                    )
                    continue

                stats["succeeded"] += 1
        finally:
            cap.release()


def _extract_with_ffmpeg(
    by_video: Dict[str, List[Dict[str, Any]]],
    videos_dir: Path,
    extracted_root: Path,
    image_format: str,
    stats: Dict[str, Any],
) -> None:
    for video_id, records in by_video.items():
        video_path = _build_video_path(videos_dir, video_id)
        if not video_path.exists():
            for record in records:
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=int(record["frame_id"]),
                    reason=f"Video not found: {video_path}",
                )
            continue

        for record in records:
            frame_id = int(record["frame_id"])
            video_out_dir = extracted_root / video_id
            video_out_dir.mkdir(parents=True, exist_ok=True)
            output_path = video_out_dir / f"frame_{frame_id}.{image_format}"

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"select=eq(n\\,{frame_id})",
                "-vframes",
                "1",
                "-y",
                str(output_path),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError:
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=frame_id,
                    reason="ffmpeg executable not found on PATH",
                )
                continue

            if result.returncode != 0 or not output_path.exists():
                stderr = result.stderr.strip()
                reason = stderr if stderr else "ffmpeg failed to extract frame"
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=frame_id,
                    reason=reason,
                )
                continue

            stats["succeeded"] += 1


def _extract_from_frame_directories(
    by_video: Dict[str, List[Dict[str, Any]]],
    frames_dir: Path,
    extracted_root: Path,
    image_format: str,
    stats: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    import cv2

    unresolved_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    num_selected_videos = len(by_video)

    for video_id, records in by_video.items():
        source_dir = _resolve_frame_source_dir(
            frames_root=frames_dir,
            video_id=video_id,
            num_selected_videos=num_selected_videos,
        )
        if source_dir is None:
            unresolved_by_video[video_id].extend(records)
            continue

        frame_index = _index_frames_in_dir(source_dir)
        for record in records:
            frame_id = int(record["frame_id"])
            source_image = frame_index.get(frame_id)
            if source_image is None:
                unresolved_by_video[video_id].append(record)
                continue

            video_out_dir = extracted_root / video_id
            video_out_dir.mkdir(parents=True, exist_ok=True)
            output_path = video_out_dir / f"frame_{frame_id}.{image_format}"

            image = cv2.imread(str(source_image))
            if image is None:
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=frame_id,
                    reason=f"Failed to read source frame image: {source_image}",
                )
                continue

            write_ok = cv2.imwrite(str(output_path), image)
            if not write_ok:
                _record_extraction_error(
                    stats,
                    video_id=video_id,
                    frame_id=frame_id,
                    reason=f"Failed to write image: {output_path}",
                )
                continue

            stats["succeeded"] += 1

    return unresolved_by_video


def extract_frames(
    selections: Sequence[Dict[str, Any]],
    videos_dir: Path | str,
    out_dir: Path | str,
    backend: str = "opencv",
    image_format: str = "jpg",
    frames_dir: Path | str | None = None,
) -> Dict[str, Any]:
    videos_dir = Path(videos_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    extracted_root = out_dir / "extracted_frames"
    extracted_root.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "attempted": int(len(selections)),
        "succeeded": 0,
        "failed": 0,
        "errors": [],
    }

    if not selections:
        return stats

    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in selections:
        by_video[str(record["video_id"])].append(record)

    remaining_by_video = by_video
    if frames_dir is not None:
        frames_root = Path(frames_dir).expanduser()
        if not frames_root.exists():
            LOGGER.warning("frames_dir does not exist; falling back to video extraction: %s", frames_root)
        elif not frames_root.is_dir():
            LOGGER.warning("frames_dir is not a directory; falling back to video extraction: %s", frames_root)
        else:
            remaining_by_video = _extract_from_frame_directories(
                by_video=by_video,
                frames_dir=frames_root,
                extracted_root=extracted_root,
                image_format=image_format,
                stats=stats,
            )

    if not remaining_by_video:
        stats["failed"] = stats["attempted"] - stats["succeeded"]
        return stats

    if backend == "opencv":
        _extract_with_opencv(
            by_video=remaining_by_video,
            videos_dir=videos_dir,
            extracted_root=extracted_root,
            image_format=image_format,
            stats=stats,
        )
        stats["failed"] = stats["attempted"] - stats["succeeded"]
        return stats

    if backend == "ffmpeg":
        _extract_with_ffmpeg(
            by_video=remaining_by_video,
            videos_dir=videos_dir,
            extracted_root=extracted_root,
            image_format=image_format,
            stats=stats,
        )
        stats["failed"] = stats["attempted"] - stats["succeeded"]
        return stats

    raise ValueError(f"Unsupported extraction backend: {backend}")


def _iter_prediction_and_metadata_paths(results_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for predictions_csv in discover_prediction_csvs(results_dir):
        metadata_csv = predictions_csv.parent / METADATA_FILENAME
        if not metadata_csv.exists():
            raise FileNotFoundError(
                f"Missing sibling '{METADATA_FILENAME}' for {predictions_csv}"
            )
        yield predictions_csv, metadata_csv


def _write_per_video_csv(
    output_dir: Path,
    video_id: str,
    selections: Sequence[Dict[str, Any]],
) -> None:
    per_video_dir = output_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    output_path = per_video_dir / f"{video_id}_selected_timestamps.csv"
    per_video_df = pd.DataFrame(selections, columns=OUTPUT_COLUMNS)
    per_video_df = per_video_df.sort_values(
        by=["frame_id", "island_start_frame_id"], kind="mergesort"
    ).reset_index(drop=True)
    per_video_df.to_csv(output_path, index=False)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.min_island_seconds < 0:
        raise ValueError("--min_island_seconds must be >= 0")
    if not args.image_format.strip():
        raise ValueError("--image_format cannot be empty")


def run(args: argparse.Namespace) -> int:
    _validate_cli_args(args)

    results_dir = Path(args.results_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    videos_dir = Path(args.videos_dir).expanduser()
    image_format = args.image_format.lstrip(".").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.random_seed is not None:
        LOGGER.info("random_seed=%s is currently ignored; frame selection is deterministic.", args.random_seed)

    all_selections: List[Dict[str, Any]] = []
    overall_stats = {
        "videos_processed": 0,
        "num_islands_found": 0,
        "num_passed_length": 0,
        "num_removed_blue_glove": 0,
        "num_selected": 0,
    }

    for predictions_csv, metadata_csv in _iter_prediction_and_metadata_paths(results_dir):
        video_id = predictions_csv.parent.name
        predictions_df = load_predictions_csv(predictions_csv)
        metadata_df = load_metadata_csv(metadata_csv)
        merged_df = merge_predictions_and_metadata(
            predictions_df=predictions_df,
            metadata_df=metadata_df,
            join_mode=args.join_mode,
        )

        islands = find_islands(merged_df)
        for island in islands:
            island["video_id"] = video_id

        filtered_islands, video_stats = filter_islands(
            islands=islands,
            fps=args.fps,
            min_seconds=args.min_island_seconds,
        )
        video_selections = select_central_frames(filtered_islands, rng=None)
        _write_per_video_csv(output_dir=output_dir, video_id=video_id, selections=video_selections)
        all_selections.extend(video_selections)

        overall_stats["videos_processed"] += 1
        overall_stats["num_islands_found"] += video_stats["num_islands_found"]
        overall_stats["num_passed_length"] += video_stats["num_passed_length"]
        overall_stats["num_removed_blue_glove"] += video_stats["num_removed_blue_glove"]
        overall_stats["num_selected"] += video_stats["num_selected"]

        LOGGER.info(
            (
                "video_id=%s num_islands_found=%d num_passed_length=%d "
                "num_removed_blue_glove=%d num_selected=%d"
            ),
            video_id,
            video_stats["num_islands_found"],
            video_stats["num_passed_length"],
            video_stats["num_removed_blue_glove"],
            video_stats["num_selected"],
        )

    combined_df = pd.DataFrame(all_selections, columns=OUTPUT_COLUMNS)
    combined_df = combined_df.sort_values(
        by=["video_id", "frame_id", "island_start_frame_id"], kind="mergesort"
    ).reset_index(drop=True)
    combined_csv = output_dir / "selected_timestamps.csv"
    combined_df.to_csv(combined_csv, index=False)

    LOGGER.info(
        (
            "completed videos_processed=%d num_islands_found=%d num_passed_length=%d "
            "num_removed_blue_glove=%d num_selected=%d output_csv=%s"
        ),
        overall_stats["videos_processed"],
        overall_stats["num_islands_found"],
        overall_stats["num_passed_length"],
        overall_stats["num_removed_blue_glove"],
        overall_stats["num_selected"],
        combined_csv,
    )

    if not args.extract_frames:
        return 0

    extraction_stats = extract_frames(
        selections=all_selections,
        videos_dir=videos_dir,
        out_dir=output_dir,
        backend=args.backend,
        image_format=image_format,
        frames_dir=args.frames_dir,
    )
    LOGGER.info(
        "frame extraction attempted=%d succeeded=%d failed=%d",
        extraction_stats["attempted"],
        extraction_stats["succeeded"],
        extraction_stats["failed"],
    )
    for error in extraction_stats["errors"][:10]:
        LOGGER.error(
            "extract_failed video_id=%s frame_id=%d reason=%s",
            error["video_id"],
            error["frame_id"],
            error["reason"],
        )
    if len(extraction_stats["errors"]) > 10:
        LOGGER.error("additional_extraction_errors=%d", len(extraction_stats["errors"]) - 10)

    if extraction_stats["attempted"] > 0 and extraction_stats["succeeded"] == 0:
        raise RuntimeError("Frame extraction requested, but all frame extraction attempts failed.")
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        LOGGER.error("timestamp extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
