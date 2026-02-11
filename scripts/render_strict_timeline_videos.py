from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_maker.contact_timeline_renderer import (
    align_secondary_prediction,
    load_gt_binary_aligned,
    load_pred_binary_from_condensed,
    render_contact_timeline_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render timeline videos for strict predictions using video_maker. "
            "Optionally include baseline predictions as a secondary row."
        )
    )
    parser.add_argument(
        "--strict-run-root",
        required=True,
        help="Run root for strict predictions (must contain run_manifest.csv and predictions/).",
    )
    parser.add_argument(
        "--baseline-run-root",
        default=None,
        help="Optional baseline run root; if provided, renders baseline as secondary timeline row.",
    )
    parser.add_argument(
        "--dataset-keys",
        default=None,
        help="Optional comma-separated dataset keys to render.",
    )
    parser.add_argument(
        "--use-frames-det",
        action="store_true",
        help="Use <pred_dir>/visualizations/frames_det as image source instead of manifest input_dir.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output .mp4 filename under <strict_pred_dir>/visualizations/. "
            "Default is contact_timeline_strict.mp4 or contact_timeline_strict_vs_baseline.mp4."
        ),
    )
    parser.add_argument(
        "--fps-fallback",
        type=float,
        default=30.0,
        help="FPS fallback if GT does not include an fps column.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets whose output video already exists and is non-empty.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Rerender even if output video already exists.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Strict Timeline",
        help="Title prefix shown in the rendered video header.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tqdm():
    try:
        from tqdm import tqdm

        return tqdm
    except Exception:
        return None


def _parse_dataset_keys(text: Optional[str]) -> Optional[set[str]]:
    if not text:
        return None
    out = set()
    for item in text.split(","):
        key = item.strip()
        if key:
            out.add(key)
    return out


def _extract_numeric_token(text: object) -> Optional[int]:
    if text is None:
        return None
    match = re.search(r"(\d+)", str(text))
    if not match:
        return None
    return int(match.group(1))


def _extract_fps_from_gt(gt_csv: Path, fallback: float) -> float:
    if fallback <= 0:
        raise ValueError("fps fallback must be > 0.")
    gt_df = pd.read_csv(gt_csv, sep=None, engine="python")
    if "fps" not in gt_df.columns:
        return float(fallback)
    fps_values = pd.to_numeric(gt_df["fps"], errors="coerce").dropna()
    fps_values = fps_values[fps_values > 0]
    if fps_values.empty:
        return float(fallback)
    return float(fps_values.iloc[0])


def _load_manifest(run_root: Path) -> pd.DataFrame:
    manifest_path = run_root / "run_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame(columns=["dataset_key", "pred_dir", "input_dir", "gt_csv", "status"])
    df = pd.read_csv(manifest_path)
    required = {"dataset_key", "pred_dir", "input_dir", "gt_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")
    return df.sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)


def _build_baseline_index(baseline_root: Optional[Path]) -> Dict[str, Path]:
    if baseline_root is None:
        return {}
    pred_root = baseline_root / "predictions"
    if not pred_root.exists():
        raise FileNotFoundError(f"Baseline predictions dir missing: {pred_root}")
    index: Dict[str, Path] = {}
    for entry in sorted(pred_root.iterdir()):
        if entry.is_dir():
            condensed = entry / "detections_condensed.csv"
            if condensed.exists():
                index[entry.name] = condensed
    return index


def _manifest_meta_index(df: pd.DataFrame) -> Dict[str, Tuple[Path, Path]]:
    out: Dict[str, Tuple[Path, Path]] = {}
    if df.empty:
        return out
    for _, row in df.iterrows():
        key = str(row["dataset_key"])
        out[key] = (Path(row["input_dir"]).expanduser(), Path(row["gt_csv"]).expanduser())
    return out


def main() -> int:
    args = parse_args()
    strict_root = Path(args.strict_run_root).expanduser()
    baseline_root = Path(args.baseline_run_root).expanduser() if args.baseline_run_root else None
    selected = _parse_dataset_keys(args.dataset_keys)

    strict_manifest = _load_manifest(strict_root)
    baseline_manifest = _load_manifest(baseline_root) if baseline_root is not None else pd.DataFrame()
    strict_meta = _manifest_meta_index(strict_manifest)
    baseline_meta = _manifest_meta_index(baseline_manifest)
    baseline_index = _build_baseline_index(baseline_root)

    strict_pred_root = strict_root / "predictions"
    if not strict_pred_root.exists():
        raise FileNotFoundError(f"Missing strict predictions dir: {strict_pred_root}")
    dataset_keys = []
    for entry in sorted(strict_pred_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "detections_condensed.csv").exists():
            dataset_keys.append(entry.name)
    if selected is not None:
        dataset_keys = [k for k in dataset_keys if k in selected]
    if not dataset_keys:
        raise ValueError("No strict prediction datasets found after filtering.")

    output_name = args.output_name
    if output_name is None:
        output_name = (
            "contact_timeline_strict_vs_baseline.mp4"
            if baseline_root is not None
            else "contact_timeline_strict.mp4"
        )

    videos_dir = strict_root / "videos"
    logs_dir = videos_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    render_manifest_path = videos_dir / "strict_timeline_manifest.csv"

    rows: List[Dict[str, object]] = []
    iterable = list(dataset_keys)
    tqdm_fn = _tqdm()
    if tqdm_fn is not None:
        iterable = tqdm_fn(iterable, desc="render_videos", unit="ds")

    for dataset_key in iterable:
        strict_pred_dir = strict_pred_root / dataset_key
        strict_condensed = strict_pred_dir / "detections_condensed.csv"
        input_dir = None
        gt_csv = None
        if dataset_key in strict_meta:
            input_dir, gt_csv = strict_meta[dataset_key]
        elif dataset_key in baseline_meta:
            input_dir, gt_csv = baseline_meta[dataset_key]

        if args.use_frames_det:
            image_dir = strict_pred_dir / "visualizations" / "frames_det"
        else:
            image_dir = input_dir if input_dir is not None else Path("")

        baseline_condensed = baseline_index.get(dataset_key)
        output_video = strict_pred_dir / "visualizations" / output_name

        row: Dict[str, object] = {
            "dataset_key": dataset_key,
            "status": "pending",
            "error": "",
            "started_at_utc": _utc_now(),
            "ended_at_utc": None,
            "duration_sec": None,
            "strict_condensed_csv": str(strict_condensed),
            "baseline_condensed_csv": str(baseline_condensed) if baseline_condensed is not None else "",
            "gt_csv": str(gt_csv),
            "image_dir": str(image_dir),
            "output_video": str(output_video),
            "fps": None,
            "n_frames": None,
        }

        started = datetime.now(timezone.utc)
        try:
            if not strict_condensed.exists():
                raise FileNotFoundError(f"Missing strict condensed csv: {strict_condensed}")
            if baseline_root is not None and baseline_condensed is None:
                raise FileNotFoundError(f"Missing baseline condensed csv for dataset: {dataset_key}")
            if gt_csv is None:
                raise FileNotFoundError(
                    f"Missing GT/input metadata for dataset {dataset_key}; "
                    f"present in strict run_manifest.csv: {dataset_key in strict_meta}, "
                    f"baseline run_manifest.csv: {dataset_key in baseline_meta}"
                )
            if not gt_csv.exists():
                raise FileNotFoundError(f"Missing GT csv: {gt_csv}")
            if not image_dir.exists():
                raise FileNotFoundError(f"Missing image dir: {image_dir}")

            if args.skip_existing and output_video.exists() and output_video.stat().st_size > 0:
                row["status"] = "success"
                row["action"] = "skipped_existing"
            else:
                frame_numbers, strict_binary = load_pred_binary_from_condensed(str(strict_condensed))
                gt_binary = load_gt_binary_aligned(str(gt_csv), frame_numbers)
                secondary_binary = None
                if baseline_condensed is not None:
                    secondary_binary = align_secondary_prediction(str(baseline_condensed), frame_numbers)
                fps = _extract_fps_from_gt(gt_csv, fallback=float(args.fps_fallback))
                row["fps"] = float(fps)
                row["n_frames"] = int(len(frame_numbers))

                title = f"{args.title_prefix}: {dataset_key}"
                if baseline_condensed is not None:
                    title = f"{args.title_prefix} (strict vs baseline): {dataset_key}"

                saved = render_contact_timeline_video(
                    image_dir=str(image_dir),
                    frame_numbers=frame_numbers,
                    pred_binary=strict_binary,
                    output_video_path=str(output_video),
                    fps=float(fps),
                    title=title,
                    gt_binary=gt_binary,
                    secondary_binary=secondary_binary,
                )
                if not Path(saved).exists() or Path(saved).stat().st_size <= 0:
                    raise RuntimeError(f"Video not created or empty: {saved}")
                row["status"] = "success"
                row["action"] = "rendered"
        except Exception as exc:
            row["status"] = "failed"
            row["action"] = "error"
            row["error"] = str(exc)
        finally:
            ended = datetime.now(timezone.utc)
            row["ended_at_utc"] = ended.isoformat()
            row["duration_sec"] = round((ended - started).total_seconds(), 3)
            rows.append(row)
            pd.DataFrame(rows).to_csv(render_manifest_path, index=False)

    failed = [r for r in rows if r["status"] == "failed"]
    print(f"Render manifest: {render_manifest_path}")
    print(f"Datasets processed: {len(rows)}")
    print(f"Failed datasets: {len(failed)}")
    if failed:
        for row in failed:
            print(f"[failed] {row['dataset_key']}: {row['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
