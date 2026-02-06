# CLAUDE.md

Guide for AI assistants working on this codebase.

## Project Overview

Video-contact-pipeline automates detection of hand-object contact duration from egocentric video recordings. Built for neuroscience research (McMaster University Goldreich Lab), it replaces 250+ hours of manual video annotation with a Faster R-CNN-based detection pipeline achieving 87% frame-wise accuracy.

**Adapted from:** [100-DOH](https://github.com/ddshan/hand_object_detector) (CVPR 2020).

## Repository Structure

```
run_pipeline.py              # CLI entry point
pipeline/                    # Core pipeline modules
  config.py                  # PipelineConfig dataclass
  main.py                    # Pipeline orchestration
  preprocessing.py           # Frame extraction & preprocessing
  inference.py               # Faster R-CNN hand-object detection
  postprocessing.py          # Result filtering & condensation
  filters.py                 # Detection filters (blue glove, object size)
  metrics.py                 # Evaluation metrics (MoF, Edit, F1@k)
  visualization.py           # Barcode visualization output
lib/                         # Deep learning model code (Faster R-CNN, RoI ops, NMS)
  model/                     # Model architectures, C++/CUDA extensions
  setup.py                   # Build script for C++/CUDA extensions
tests/                       # 18 test files (unit + integration + parity)
scripts/                     # Utility scripts (evaluation, regression suite)
cfgs/                        # Model YAML configs (res101.yml is default)
archive/                     # Legacy code (reference only, do not modify)
docs/refs/ug-thesis.md       # Full thesis documentation
```

## Tech Stack

- **Python 3.8**, **PyTorch 1.12.1**, **CUDA 11.3**
- **Faster R-CNN** with **ResNet-101** backbone (default)
- C++/CUDA extension modules in `lib/` (compiled with GCC 10)
- Dependencies in `requirements.txt` (scipy, opencv-python, pillow, tqdm, etc.)

## Key Commands

### Run tests
```bash
pytest -q
```

### Run parity/integration tests (require test data)
```bash
TEST_INPUT="/path/to/input" pytest -q tests/test_inference_parity_full.py
TEST_INPUT="/path/to/input" scripts/run_parity_suite.sh
```

### Run the pipeline
```bash
# Single video
python run_pipeline.py --input /path/to/video.mp4 --output-dir results/run/

# Batch (folder of videos)
python run_pipeline.py --input /path/to/folder/ --output-dir results/batch/

# Preprocess only
python run_pipeline.py --input /path/to/input --output-dir results/run/ --preprocess-only

# Evaluate metrics against ground truth
python scripts/evaluate_metrics.py --pred detections_condensed.csv --gt gt.csv
```

### Build C++/CUDA extensions
```bash
cd lib
CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop
```

## Architecture & Data Flow

1. **Preprocessing** - Extract frames from video, apply crop (480x480 default) and vertical flip
2. **Inference** - Run Faster R-CNN to detect hands and objects per frame
3. **Filtering** - Apply blue glove filter, object size filter, etc.
4. **Postprocessing** - Condense per-detection results to one label per frame
5. **Output** - `detections_full.csv`, `detections_condensed.csv`, optional barcode visualizations

Contact labels: `"Portable Object"`, `"Stationary Object"`, `"No Contact"` (priority-based selection when multiple detections per frame).

## Code Conventions

- **Type hints** used throughout (`from __future__ import annotations`)
- **Dataclasses** for configuration (`PipelineConfig` in `pipeline/config.py`)
- **snake_case** for functions/variables, **PascalCase** for classes
- Private functions prefixed with `_`
- Paths support `~` expansion via `Path.expanduser()`
- Frame filenames are 1-indexed, zero-padded to 6 digits (000001.jpg, etc.)
- CSV files use UTF-8 encoding

## Configuration

All pipeline behavior is driven by `PipelineConfig` (dataclass, ~30 parameters). Key defaults:
- `net = "res101"`, `cfg_file = "cfgs/res101.yml"`
- `thresh_hand = 0.5`, `thresh_obj = 0.5`
- `crop_square = 480`, `flip_vertical = True`
- `blue_glove_filter = True`, `object_size_filter = True`
- Model checkpoint: session=1, epoch=8, checkpoint=132028

Config is set via CLI args in `run_pipeline.py` and serialized/loaded as JSON.

## Testing Notes

- **Unit tests** run without external data (`pytest -q` runs these)
- **Parity tests** require `TEST_INPUT` env var pointing to test video/frames
- **Golden file tests** compare against saved reference outputs in `tests/golden/`
- No CI/CD pipeline; regression testing is manual via `scripts/run_parity_suite.sh`
- No linter or formatter configured

## Things to Avoid

- Do not modify files under `archive/` (legacy reference code)
- Do not modify files under `lib/` unless fixing build issues (upstream model code)
- The model checkpoint parameters are fixed; do not change them without retraining
- `lib/` C++ extensions must be compiled before running inference
