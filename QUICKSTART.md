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

## Postprocess-only modes
Generate barcodes from an existing `detections_condensed.csv`:
```bash
python run_pipeline.py --barcodes-only --condensed-csv /path/to/detections_condensed.csv --output-dir /path/to/out
```

Generate annotated bbox frames from an existing `detections_full.csv`:
```bash
python run_pipeline.py --annotated-frames-only --full-csv /path/to/detections_full.csv --image-dir /path/to/frames --output-dir /path/to/out
```

## Regression tests
```bash
pytest -q
TEST_INPUT="/path/to/input" pytest -q tests/test_inference_parity_full.py
TEST_INPUT="/path/to/input" pytest -q tests/test_condense_parity.py
TEST_INPUT="/path/to/input" pytest -q tests/test_blue_glove_parity.py
```
