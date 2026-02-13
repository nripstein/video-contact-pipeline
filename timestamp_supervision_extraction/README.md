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
3. Pick one representative frame per remaining island deterministically:
   - selected index `i = L // 2`
   - for odd `L`, this is the exact middle
   - for even `L`, this is the upper-middle frame

`--random_seed` is retained for CLI compatibility but is currently ignored because
selection is deterministic.

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

## Quantitative Evaluation of Selected Timestamps

You can evaluate selected timestamps against GT labels in either single- or multi-dataset mode.

### Single Dataset / Video

```bash
python -m timestamp_supervision_extraction.evaluate_selected_timestamps \
  --selected-csv timestamp_supervision_extraction/outputs/selected_timestamps.csv \
  --gt-csv /path/to/gt.csv \
  --dataset-key demo_video \
  --json-out timestamp_supervision_extraction/outputs/selected_metrics_demo.json
```

Accepted GT formats:

- `frame_number, gt_binary`
- `frame_id, label` where label maps to binary holding/not_holding semantics

### Multi Dataset (Manifest)

Create a manifest CSV with columns:

- `dataset_key`
- `selected_csv`
- `gt_csv`

Then run:

```bash
python -m timestamp_supervision_extraction.evaluate_selected_timestamps \
  --manifest-csv /path/to/metrics_manifest.csv \
  --output-dir /path/to/output_dir
```

Optional filtering:

```bash
python -m timestamp_supervision_extraction.evaluate_selected_timestamps \
  --manifest-csv /path/to/metrics_manifest.csv \
  --dataset-keys sv1,sv2,sv3 \
  --output-dir /path/to/output_dir
```

In multi mode, the command writes:

- `selected_timestamp_metrics_per_dataset.csv`
- `selected_timestamp_metrics_summary.json`

The summary JSON includes per-dataset metrics plus global summaries:

- `global_micro`: pooled over all selected frames from successful datasets
- `global_macro`: average of per-dataset metric values

If some datasets fail, valid datasets are still evaluated and written to outputs; exit code is non-zero.

### One-Command Baseline from Existing Run Manifest

If you already have a batch run with `run_manifest.csv` and per-dataset prediction outputs
(`detections_full.csv` + `detections_condensed.csv`), you can run confident-frame selection
and evaluation in one command:

```bash
python scripts/run_timestamp_supervision_baseline.py \
  --run-root /path/to/pipeline_run_root \
  --fps 60 \
  --min-island-seconds 1.0
```

Outputs are written under:

- `/path/to/pipeline_run_root/timestamp_supervision_baseline/prep_manifest.csv`
- `/path/to/pipeline_run_root/timestamp_supervision_baseline/selected_timestamps/`
- `/path/to/pipeline_run_root/timestamp_supervision_baseline/evaluation/selected_timestamp_metrics_per_dataset.csv`
- `/path/to/pipeline_run_root/timestamp_supervision_baseline/evaluation/selected_timestamp_metrics_summary.json`
