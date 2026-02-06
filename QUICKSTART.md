# Quickstart

This guide covers the new pipeline for single inputs, frames directories, and batch folders.

## Prerequisites
- A working Python environment with the repo dependencies installed.
- C++/CUDA extensions built (`cd lib && CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop`).
- Model weights placed under `models/` (see README for the exact folder layout).

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
- `--obj-bigger-filter --obj-bigger-k 1.0`

## Regression tests
```bash
pytest -q
TEST_INPUT="/path/to/input" pytest -q tests/test_inference_parity_full.py
TEST_INPUT="/path/to/input" pytest -q tests/test_condense_parity.py
TEST_INPUT="/path/to/input" pytest -q tests/test_blue_glove_parity.py
```
