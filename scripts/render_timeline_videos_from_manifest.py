from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render contact timeline videos for datasets listed in a run_manifest.csv, "
            "using frames_det as the visual source."
        )
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run root directory that contains run_manifest.csv and predictions/.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Optional explicit run_manifest.csv path (default: <run-root>/run_manifest.csv).",
    )
    parser.add_argument(
        "--dataset-keys",
        default=None,
        help="Optional comma-separated dataset keys to render (default: all successful entries).",
    )
    parser.add_argument(
        "--image-subdir",
        default="visualizations/frames_det",
        help="Image subdir under each pred_dir used as --image-dir for timeline rendering.",
    )
    parser.add_argument(
        "--output-name",
        default="contact_timeline_frames_det.mp4",
        help="Output video filename under each pred_dir/visualizations.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Contact Timeline (frames_det)",
        help="Prefix for timeline title. Dataset key is appended automatically.",
    )
    parser.add_argument(
        "--fps-fallback",
        type=float,
        default=30.0,
        help="Fallback FPS when GT fps cannot be parsed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets with an existing non-empty output video.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Always rerender videos even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without rendering videos.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dataset_keys(csv_text: str | None) -> set[str] | None:
    if not csv_text:
        return None
    out = set()
    for item in csv_text.split(","):
        key = item.strip()
        if key:
            out.add(key)
    return out


def _extract_numeric_token(text: object) -> int | None:
    if text is None:
        return None
    match = re.search(r"(\d+)", str(text))
    if not match:
        return None
    return int(match.group(1))


def _sorted_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    entries = [p for p in image_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    return sorted(entries, key=lambda p: (_extract_numeric_token(p.stem) is None, _extract_numeric_token(p.stem) or 0, p.name))


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


def _render_single_dataset(
    *,
    dataset_key: str,
    pred_dir: Path,
    gt_csv: Path,
    image_subdir: str,
    output_name: str,
    title_prefix: str,
    fps_fallback: float,
    skip_existing: bool,
    dry_run: bool,
    logs_dir: Path,
) -> Dict[str, object]:
    started = datetime.now(timezone.utc)
    row: Dict[str, object] = {
        "dataset_key": dataset_key,
        "status": "pending",
        "action": "",
        "error": "",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": None,
        "duration_sec": None,
        "pred_dir": str(pred_dir),
        "gt_csv": str(gt_csv),
        "image_dir": str(pred_dir / image_subdir),
        "condensed_csv": str(pred_dir / "detections_condensed.csv"),
        "output_video": str(pred_dir / "visualizations" / output_name),
        "fps": None,
        "n_frames_condensed": None,
        "n_images": None,
    }

    try:
        condensed_csv = pred_dir / "detections_condensed.csv"
        image_dir = pred_dir / image_subdir
        output_video = pred_dir / "visualizations" / output_name

        if not condensed_csv.exists():
            raise FileNotFoundError(f"Missing condensed csv: {condensed_csv}")
        if not gt_csv.exists():
            raise FileNotFoundError(f"Missing gt csv: {gt_csv}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {image_dir}")

        condensed_df = pd.read_csv(condensed_csv)
        n_frames = int(len(condensed_df))
        if n_frames <= 0:
            raise ValueError(f"Empty condensed csv: {condensed_csv}")
        row["n_frames_condensed"] = n_frames

        image_paths = _sorted_image_paths(image_dir)
        n_images = int(len(image_paths))
        row["n_images"] = n_images
        if n_images < n_frames:
            raise ValueError(
                f"Not enough images in {image_dir}: images={n_images}, frames={n_frames}."
            )

        fps = _extract_fps_from_gt(gt_csv, fallback=fps_fallback)
        row["fps"] = fps

        output_video.parent.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{dataset_key}.log"

        if skip_existing and output_video.exists() and output_video.stat().st_size > 0:
            row["status"] = "success"
            row["action"] = "skipped_existing"
            return row

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_contact_timeline_video.py"),
            "--condensed-csv",
            str(condensed_csv),
            "--image-dir",
            str(image_dir),
            "--gt-csv",
            str(gt_csv),
            "--fps",
            str(fps),
            "--output-video",
            str(output_video),
            "--title",
            f"{title_prefix}: {dataset_key}",
        ]

        if dry_run:
            row["status"] = "success"
            row["action"] = "dry_run"
            return row

        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        log_path.write_text(
            f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"make_contact_timeline_video.py failed with code {result.returncode}.")
        if not output_video.exists() or output_video.stat().st_size <= 0:
            raise RuntimeError(f"Output video was not created or is empty: {output_video}")

        row["status"] = "success"
        row["action"] = "rendered"
        return row
    except Exception as exc:
        row["status"] = "failed"
        row["action"] = "error"
        row["error"] = str(exc)
        return row
    finally:
        ended = datetime.now(timezone.utc)
        row["ended_at_utc"] = ended.isoformat()
        row["duration_sec"] = round((ended - started).total_seconds(), 3)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser()
    manifest_csv = Path(args.manifest_csv).expanduser() if args.manifest_csv else (run_root / "run_manifest.csv")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"run manifest not found: {manifest_csv}")
    if args.fps_fallback <= 0:
        raise ValueError("--fps-fallback must be > 0.")

    source_df = pd.read_csv(manifest_csv)
    if "dataset_key" not in source_df.columns:
        raise ValueError(f"{manifest_csv} missing required column dataset_key.")
    if "pred_dir" not in source_df.columns:
        raise ValueError(f"{manifest_csv} missing required column pred_dir.")
    if "gt_csv" not in source_df.columns:
        raise ValueError(f"{manifest_csv} missing required column gt_csv.")

    selected_keys = _parse_dataset_keys(args.dataset_keys)
    if selected_keys is not None:
        source_df = source_df[source_df["dataset_key"].isin(selected_keys)]

    if "status" in source_df.columns:
        source_df = source_df[source_df["status"] == "success"]

    source_df = source_df.sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)
    if source_df.empty:
        raise ValueError("No datasets selected for rendering after filters.")

    videos_dir = run_root / "videos"
    logs_dir = videos_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    render_manifest_path = videos_dir / "contact_timeline_frames_det_manifest.csv"

    rows: List[Dict[str, object]] = []
    for _, item in source_df.iterrows():
        row = _render_single_dataset(
            dataset_key=str(item["dataset_key"]),
            pred_dir=Path(item["pred_dir"]).expanduser(),
            gt_csv=Path(item["gt_csv"]).expanduser(),
            image_subdir=args.image_subdir,
            output_name=args.output_name,
            title_prefix=args.title_prefix,
            fps_fallback=float(args.fps_fallback),
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            logs_dir=logs_dir,
        )
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
