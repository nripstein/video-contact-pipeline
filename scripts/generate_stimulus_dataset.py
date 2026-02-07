from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.pipeline import run_phase1_data_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a stimulus detection training dataset (Phase 1).")
    parser.add_argument("--input", dest="input_path", required=True, help="Video path, videos directory, or frames directory")
    parser.add_argument("--output-dir", dest="output_dir", required=True, help="Output root directory")
    parser.add_argument("--config", default=None, help="Optional JSON/YAML config file")

    parser.add_argument("--filter-strategy", default="bbox_similarity", choices=["bbox_similarity", "temporal_subsampling", "kmeans"])

    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--aspect-ratio-min", type=float, default=0.8)
    parser.add_argument("--aspect-ratio-max", type=float, default=1.2)
    parser.add_argument(
        "--max-object-hand-ratio",
        type=float,
        default=1.0,
        help="Require object_area/hand_area to be below this threshold. Lower is stricter (e.g., 0.6).",
    )
    parser.add_argument(
        "--max-hand-occlusion-ratio",
        type=float,
        default=1.0,
        help="Drop boxes when hand intersection/object_area exceeds this ratio. Lower is stricter (e.g., 0.5).",
    )

    parser.add_argument("--min-temporal-gap-sec", type=float, default=0.25)
    parser.add_argument("--center-move-frac", type=float, default=0.10)
    parser.add_argument("--area-change-frac", type=float, default=0.20)
    parser.add_argument("--subsample-interval-sec", type=float, default=0.25)

    parser.add_argument("--participant-regex", default=r"(?i)(sv\d+)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--default-fps",
        type=float,
        default=60.0,
        help="FPS to use for frame directories or when video FPS metadata is missing (commonly 60 or 30).",
    )

    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_config = (
        Phase1Config.load(args.config)
        if args.config
        else Phase1Config(input_path=args.input_path, output_dir=args.output_dir)
    )

    config = replace(
        base_config,
        input_path=args.input_path,
        output_dir=args.output_dir,
        filter_strategy=args.filter_strategy,
        min_confidence=args.min_confidence,
        aspect_ratio_min=args.aspect_ratio_min,
        aspect_ratio_max=args.aspect_ratio_max,
        max_object_to_hand_area_ratio=args.max_object_hand_ratio,
        max_hand_occlusion_ratio=args.max_hand_occlusion_ratio,
        min_temporal_gap_sec=args.min_temporal_gap_sec,
        center_move_frac=args.center_move_frac,
        area_change_frac=args.area_change_frac,
        subsample_interval_sec=args.subsample_interval_sec,
        participant_regex=args.participant_regex,
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        default_fps_if_unknown=args.default_fps,
        generate_visualizations=not args.no_viz,
        generate_stats=not args.no_stats,
        cuda=not args.cpu,
    )

    result = run_phase1_data_generation(config)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
