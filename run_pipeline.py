from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pipeline import PipelineConfig
from pipeline.main import prepare_frames, run_pipeline, is_video_file, list_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="New pipeline CLI (stub)")
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--thresh-hand", type=float, default=0.5)
    parser.add_argument("--thresh-obj", type=float, default=0.5)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--obj-bigger-filter", action="store_true")
    parser.add_argument("--obj-bigger-k", type=float, default=1.0)
    parser.add_argument("--blue-threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--no-blue-glove-filter", action="store_true")
    parser.add_argument("--no-object-size-filter", action="store_true")
    parser.add_argument("--object-size-max-area-ratio", type=float, default=0.5)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--gt-csv", dest="gt_csv_path", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-annotated-frames", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PipelineConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        thresh_hand=args.thresh_hand,
        thresh_obj=args.thresh_obj,
        crop_square=None if args.no_crop else 480,
        flip_vertical=not args.no_flip,
        obj_bigger_than_hand_filter=args.obj_bigger_filter,
        obj_bigger_ratio_k=args.obj_bigger_k,
        blue_threshold=args.blue_threshold,
        blue_glove_filter=not args.no_blue_glove_filter,
        object_size_filter=not args.no_object_size_filter,
        object_size_max_area_ratio=args.object_size_max_area_ratio,
        save_visualizations=not args.no_visualizations,
        gt_csv_path=args.gt_csv_path,
        show_progress=not args.no_progress,
        save_annotated_frames=args.save_annotated_frames,
    )

    config_dict = config.to_dict()
    config_dict["output_dir_resolved"] = config.resolve_output_dir()

    if args.verbose:
        print("Resolved pipeline config:")
    print(json.dumps(config_dict, indent=2, sort_keys=True))

    if args.preprocess_only:
        processed_dir = prepare_frames(config)
        print(f"Preprocessed frames: {processed_dir}")
        return 0
    if args.inference_only:
        full_df, _ = run_pipeline(config, do_condense=False)
        output_dir = config.resolve_output_dir()
        print(f"Wrote detections_full.csv to {output_dir}")
        return 0

    input_path = Path(config.input_path).expanduser()
    output_dir = Path(config.resolve_output_dir()).expanduser()
    if input_path.is_dir():
        videos = list_videos(str(input_path))
        if videos:
            for video_path in videos:
                video_stem = Path(video_path).stem
                per_output = output_dir / video_stem
                if args.skip_existing:
                    full_path = per_output / "detections_full.csv"
                    cond_path = per_output / "detections_condensed.csv"
                    if full_path.exists() and cond_path.exists():
                        print(f"[skip] {video_stem} -> {per_output}")
                        continue
                print(f"[run] {video_stem} -> {per_output}")
                run_pipeline(
                    replace(
                        config,
                        input_path=video_path,
                        output_dir=str(per_output),
                    )
                )
            return 0

    full_df, condensed_df = run_pipeline(config)
    output_dir = config.resolve_output_dir()
    print(f"Wrote detections_full.csv to {output_dir}")
    print(f"Wrote detections_condensed.csv to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
