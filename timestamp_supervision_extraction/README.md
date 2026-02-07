# Timestamp Supervision Extraction

Utility to extract one representative timestamp per stable prediction island from per-frame inference outputs.

## Expected Input Layout

- Prediction files: `results/**/predictions.csv`
- Metadata files: sibling `metadata.csv` next to each `predictions.csv`
- Source videos (for optional frame extraction): `videos/<video_id>.mp4` where `video_id` is the parent directory name of `predictions.csv`

Required columns:

- `predictions.csv`: `frame_id` (int), `predicted_label` (0/1)
- `metadata.csv`: `frame_id` (int), `blue_glove_detected` (bool-like)

## Usage

From repo root:

```bash
python -m timestamp_supervision_extraction.extract_timestamps \
  --results_dir results \
  --videos_dir videos \
  --output_dir timestamp_supervision_extraction/outputs \
  --fps 60 \
  --min_island_seconds 1.0 \
  --join_mode inner \
  --random_seed 42
```

With frame extraction:

```bash
python -m timestamp_supervision_extraction.extract_timestamps \
  --results_dir results \
  --videos_dir videos \
  --frames_dir preextracted_frames \
  --output_dir timestamp_supervision_extraction/outputs \
  --fps 60 \
  --min_island_seconds 1.0 \
  --extract_frames \
  --backend opencv \
  --image_format jpg
```

If `--frames_dir` is provided, extraction first looks for:

- `frames_dir/<video_id>/*` image files (multi-video layout), or
- image files directly in `frames_dir` when there is only one selected video (single-video flat layout).

Any unresolved selections then fall back to video extraction using `--videos_dir/<video_id>.mp4`.

## Island Definition

Rows are sorted by `frame_id`. Islands are maximal contiguous runs where:

- `predicted_label` stays constant, and
- successive `frame_id` values differ by exactly 1

Any frame gap (`delta != 1`) ends the current island.

## Filtering and Selection

1. Keep islands with `length_frames >= ceil(min_island_seconds * fps)`.
2. Blue glove veto: for label `1` islands, drop the island if any frame has `blue_glove_detected == True`.
3. Pick one representative frame per remaining island with center-biased sampling:
   - center `c = (L - 1) / 2`
   - sigma `= (L - 1) / 4`
   - sample normal, round with `np.rint`, resample while index is out of range

## Outputs

- Combined CSV: `output_dir/selected_timestamps.csv`
- Per-video CSVs: `output_dir/per_video/<video_id>_selected_timestamps.csv`
- Optional frames: `output_dir/extracted_frames/<video_id>/frame_<frame_id>.<image_format>`

CSV columns:

- `video_id`
- `frame_id`
- `predicted_label`
- `island_start_frame_id`
- `island_end_frame_id`
- `island_length_frames`

## Join Behavior

- `--join_mode inner`: only frames present in both CSVs are processed.
- `--join_mode left`: all prediction frames are kept, and missing `blue_glove_detected` defaults to `False`.

## Error Behavior

- Fails fast on malformed CSVs, missing required columns, or missing sibling `metadata.csv`.
- If `--extract_frames` is enabled, per-frame extraction errors are logged and processing continues.
- Exit is non-zero only if extraction was requested and all extraction attempts failed.
