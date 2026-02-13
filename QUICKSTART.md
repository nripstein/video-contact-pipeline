# Quickstart

This guide covers the new pipeline for single inputs, frames directories, and batch folders.

## Prerequisites
- A working Python environment with the repo dependencies installed.
- C++/CUDA extensions built (`cd lib && CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop`).
- Model weights placed under `models/` (see README for the exact folder layout).
- Notebook workflow: see `docs/NOTEBOOKS.md`, `notebooks/pipeline_starter.ipynb`, and `notebooks/postprocess_only.ipynb`.
- Reproducible notebook bootstrap: `scripts/bootstrap_notebook.sh shan_et_al2`.

## Single video
```bash
python run_pipeline.py --input /path/to/video.mp4 --output-dir results/video_run/
```

## Frames directory
```bash
python run_pipeline.py --input /path/to/frames_dir --output-dir results/frames_run/
```

## Batch folder of videos
```bash
python run_pipeline.py --input /path/to/folder_of_videos --output-dir results/batch_run/
```

Outputs are written to:
```
results/batch_run/<video_stem>/
```

## Optional flags
- `--preprocess-only` : run preprocessing and exit
- `--inference-only` : run preprocessing + inference and exit (no condense)
- `--no-blue-glove-filter`
- `--no-object-size-filter`
- `--small-object-filter` : enable small-object ratio filter (default is off)
- `--obj-smaller-factor 2.0` : max allowed object/hand area ratio for portable contact (relabels to `No Contact` if object is larger)
- `--obj-bigger-filter --obj-bigger-k 1.0`
- `--tracking-bridge` : enable motion-only object tracking bridge (default is off)
- `--tracking-max-missed-frames 8` : maximum consecutive miss frames before track is dropped
- `--tracking-iou-threshold 0.15` : hand/tracked-object IoU threshold used for contact evidence
- `--tracking-init-obj-confidence 0.70` : minimum object confidence required to initialize a track
- `--tracking-promotion-confirm-frames 2` : required consecutive qualifying miss frames before promotion
- `--tracking-reassociate-iou-threshold 0.10` : minimum IoU to reassociate detections with active track
- `--tracking-promote-stationary` : allow miss-frame promotions from `Stationary Object( Contact)` to `Portable Object`
- `--tracking-stationary-iou-threshold 0.20` : IoU threshold for stationary promotions
- `--tracking-stationary-confirm-frames 2` : required streak length for stationary promotions
- `--condense-priority-strategy no_contact_first|portable_first` : duplicate-frame tie-break rule for `detections_condensed.csv` (default: `no_contact_first`)

Tracking bridge example:
```bash
python run_pipeline.py --input /path/to/video.mp4 --output-dir results/video_run/ --tracking-bridge
```

## Postprocess-only modes
Generate barcodes from an existing `detections_condensed.csv`:
```bash
python run_pipeline.py --barcodes-only --condensed-csv /path/to/detections_condensed.csv --output-dir /path/to/out
```

Generate annotated bbox frames from an existing `detections_full.csv`:
```bash
python run_pipeline.py --annotated-frames-only --full-csv /path/to/detections_full.csv --image-dir /path/to/frames --output-dir /path/to/out
```

Generate contact timeline video from an existing `detections_condensed.csv`:
```bash
python scripts/make_contact_timeline_video.py \
  --condensed-csv /path/to/detections_condensed.csv \
  --image-dir /path/to/frames \
  --gt-csv /path/to/gt.csv
```
Default output path:
```
<condensed_csv_parent>/visualizations/contact_timeline.mp4
```

## HSMM posteriors + confidence barcode
Run HSMM refinement and export posterior probabilities:
```bash
python scripts/run_hsmm_refinement.py \
  --condensed-csv /path/to/detections_condensed.csv \
  --gt-csv /path/to/gt.csv \
  --k-segments 11 \
  --alpha-during-trial 9.0 \
  --lambda-during-trial 0.1 \
  --alpha-between-trials 9.0 \
  --lambda-between-trials 0.075 \
  --fpr 0.1 \
  --fnr 0.1 \
  --return-posteriors \
  --no-progress
```

Outputs are written to:
```
<pred_dir>/hsmm_refinement/
```

Important files:
- `hsmm_refined_binary.csv` : original and refined binary predictions
- `hsmm_posteriors.csv` : frame-wise posterior probability `posterior_contact`
- `barcode_original_refined_gt.png` : binary barcode comparison
- `barcode_confidence_refined_gt.png` : 3-row confidence plot (confidence top, refined middle, GT bottom)

## Regression tests
```bash
pytest -q
TEST_INPUT="/path/to/input" pytest -q tests/test_inference_parity_full.py
TEST_INPUT="/path/to/input" pytest -q tests/test_condense_parity.py
TEST_INPUT="/path/to/input" pytest -q tests/test_blue_glove_parity.py
```

## Repeatable full workflow for shrunk thesis datasets
Use this when you want one command to run:
1) inference + condensed outputs + barcode plots + intersection metrics,
2) `frames_det` annotated images,
3) timeline videos rendered from `frames_det`.

```bash
python scripts/run_shrunk_full_workflow.py \
  --run-name 2026-02-08_shrunk_baseline \
  --profile baseline \
  --skip-existing \
  --no-progress
```

Common profile options:
- `--profile baseline` : parity-like flags (`--no-crop --no-flip --no-object-size-filter --no-small-object-filter`)
- `--profile default` : uses `run_pipeline.py` defaults
- `--profile tracking` : baseline + `--tracking-bridge`

### Exact Thesis-PDF Reproduction Command
Use this command to run all canonical shrunk datasets with the thesis-era condense priority (`no_contact_first`) and write outputs/metrics/barcodes to `results/all_preds_new_priority/`.

```bash
python scripts/run_shrunk_inference_batch.py \
  --run-root /home/nripstein/code/refactored-undergrad-thesis/Thesis-100-DOH/results/all_preds_new_priority \
  --data-root "/home/nripstein/Documents/thesis data/thesis labels" \
  --profile baseline \
  --condense-priority-strategy no_contact_first \
  --recompute-all
```

Add custom inference flags for experiments:
```bash
python scripts/run_shrunk_full_workflow.py \
  --run-name 2026-02-09_tracking_sweep \
  --profile baseline \
  --pipeline-arg=--tracking-bridge \
  --pipeline-arg=--tracking-max-missed-frames \
  --pipeline-arg=12 \
  --pipeline-arg=--tracking-iou-threshold \
  --pipeline-arg=0.20 \
  --recompute-all \
  --no-progress
```

Useful phase control:
- `--skip-inference` : reuse existing predictions/metrics in `run_manifest.csv`
- `--skip-frames-det` : skip annotated-frame generation
- `--skip-videos` : skip timeline video rendering
