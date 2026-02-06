# Notebook Setup and Usage

This guide configures Jupyter notebooks to call the pipeline functions directly from Python.

## 1) Environment Setup

Use the same environment as the CLI pipeline.

Recommended one-command bootstrap:

```bash
scripts/bootstrap_notebook.sh shan_et_al2
```

Manual equivalent:

```bash
conda activate shan_et_al2
conda env update -n shan_et_al2 -f environment.yml
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebooks.txt
python -m pip install -e .
cd lib
CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop
cd ..
python -m ipykernel install --user --name shan_et_al2 --display-name "Python (shan_et_al2)"
```

## 2) Launch Jupyter

Launch from the repository root so imports and relative paths are predictable:

```bash
jupyter lab
```

Open one of:
- `notebooks/pipeline_starter.ipynb` (inference + postprocess)
- `notebooks/postprocess_only.ipynb` (existing CSVs only)

Select kernel `Python (shan_et_al2)`.

## 3) Python API Imports

Use API imports directly in notebook cells:

```python
from pipeline.config import PipelineConfig
from pipeline.main import run_pipeline
from pipeline.visualization import save_barcodes, save_annotated_frames
```

## 4) Starter Workflows

The starter notebook includes:
- Inference run from `PipelineConfig` + `run_pipeline`
- DataFrame inspection (`full_df`, `condensed_df`)
- Barcode generation from `detections_condensed.csv`
- Annotated frame generation from `detections_full.csv`

The postprocess-only notebook includes:
- Barcode generation from an existing `detections_condensed.csv`
- Annotated frame generation from an existing `detections_full.csv` + `image_dir`

## 5) Common Issues

- `ModuleNotFoundError: pipeline`:
  - Install the local package in the active conda env:
    ```bash
    python -m pip install -e .
    ```
  - Launch Jupyter from the repo root.
  - Verify the selected kernel is `Python (shan_et_al2)`.

- `ImportError` for compiled extension (`model._C` / missing `_C*.so`):
  - Rebuild extensions:
    ```bash
    cd lib
    CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop
    ```

- Model weights not found:
  - Place expected checkpoints under `models/` (see `README.md`).

- Notebook runs but very slowly:
  - Confirm CUDA is available in the kernel environment.
