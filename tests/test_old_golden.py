from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from compare_csvs import compare_csvs


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_old_pipeline.py"
OUTPUT_DIR = REPO_ROOT / "tests" / "output" / "old"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
OUTPUT_FILES = ["detections_full.csv", "detections_condensed.csv"]


@pytest.mark.order("first")
def test_old_pipeline_matches_golden():
    test_input = os.getenv("TEST_INPUT")
    if not test_input:
        pytest.skip("Set TEST_INPUT to a small video file or frames directory to run this test.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        out_file = OUTPUT_DIR / name
        if out_file.exists():
            out_file.unlink()

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        test_input,
        "--output-dir",
        str(OUTPUT_DIR),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    for name in OUTPUT_FILES:
        assert (OUTPUT_DIR / name).exists(), f"Expected output file missing: {name}"

    if not GOLDEN_DIR.exists():
        pytest.skip("tests/golden/ not present; generate golden outputs first.")

    for name in OUTPUT_FILES:
        golden_file = GOLDEN_DIR / name
        if not golden_file.exists():
            pytest.skip(f"Golden file missing: {golden_file}. Generate per tests/README.md.")
        compare_csvs(OUTPUT_DIR / name, golden_file)
