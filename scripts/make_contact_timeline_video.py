from __future__ import annotations

import argparse
import json
from pathlib import Path
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
        description="Render contact timeline MP4 from detections_condensed.csv and frame images."
    )
    parser.add_argument("--condensed-csv", required=True, help="Path to detections_condensed.csv")
    parser.add_argument("--image-dir", required=True, help="Path to frame directory")
    parser.add_argument("--gt-csv", default=None, help="Optional GT CSV path")
    parser.add_argument(
        "--secondary-condensed-csv",
        default=None,
        help="Optional second detections_condensed.csv to render as 'Pred 2' row",
    )
    parser.add_argument("--output-video", default=None, help="Optional explicit output .mp4 path")
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS (default: metadata or 30)")
    parser.add_argument("--title", default=None, help="Optional title shown in header")
    return parser.parse_args()


def _default_output_video(condensed_csv: str) -> str:
    condensed_path = Path(condensed_csv).expanduser()
    return str(condensed_path.parent / "visualizations" / "contact_timeline.mp4")


def _default_title(condensed_csv: str) -> str:
    parent = Path(condensed_csv).expanduser().parent
    return f"Contact Timeline: {parent.name}"


def _fps_from_preprocessing_meta(condensed_csv: str) -> float | None:
    parent = Path(condensed_csv).expanduser().parent
    meta_path = parent / "preprocessing_meta.json"
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("fps", "frame_rate", "video_fps"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if parsed > 0:
            return parsed
    return None


def main() -> int:
    args = parse_args()
    frame_numbers, pred_binary = load_pred_binary_from_condensed(args.condensed_csv)

    gt_binary = None
    if args.gt_csv:
        gt_binary = load_gt_binary_aligned(args.gt_csv, frame_numbers)

    secondary_binary = None
    if args.secondary_condensed_csv:
        secondary_binary = align_secondary_prediction(args.secondary_condensed_csv, frame_numbers)

    output_video = args.output_video or _default_output_video(args.condensed_csv)
    title = args.title or _default_title(args.condensed_csv)
    fps = args.fps or _fps_from_preprocessing_meta(args.condensed_csv) or 30.0

    output_path = render_contact_timeline_video(
        image_dir=args.image_dir,
        frame_numbers=frame_numbers,
        pred_binary=pred_binary,
        output_video_path=output_video,
        fps=fps,
        title=title,
        gt_binary=gt_binary,
        secondary_binary=secondary_binary,
    )
    print(f"Saved timeline video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
