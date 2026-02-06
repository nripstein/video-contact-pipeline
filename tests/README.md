# Test Harness for Legacy Pipeline

This harness lets us freeze the current behavior before refactoring.

## Inputs
- Set `TEST_INPUT` to a small video file OR a directory of frames (same paths the old notebooks used).

## Generate golden outputs
```
TEST_INPUT=/path/to/input \
python scripts/run_old_pipeline.py --input "$TEST_INPUT" --output-dir tests/golden
```
This writes `tests/golden/detections_full.csv` and `tests/golden/detections_condensed.csv`.

## Run pytest
```
TEST_INPUT=/path/to/input pytest -q tests/test_old_golden.py
```
- If `tests/golden/` is missing, the comparison step is skipped.

## Manual CSV compare
```
python -m tests.compare_csvs tests/output/old/detections_full.csv tests/golden/detections_full.csv
python -m tests.compare_csvs tests/output/old/detections_condensed.csv tests/golden/detections_condensed.csv
```
Use `--sort-key column` (repeatable) to override default sort columns if needed.
