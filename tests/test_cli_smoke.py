from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2


def _has_extension_build(repo_root: Path) -> bool:
    return any((repo_root / "lib" / "model").glob("_C*.so"))


def test_cli_help_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    if not _has_extension_build(repo_root):
        import pytest
        pytest.skip("C++/CUDA extensions not built; skip CLI smoke.")

    cmd = [sys.executable, str(repo_root / "run_pipeline.py"), "--help"]
    result = subprocess.run(cmd, cwd=repo_root)
    assert result.returncode == 0


def test_preprocess_only_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    if not _has_extension_build(repo_root):
        import pytest
        pytest.skip("C++/CUDA extensions not built; skip CLI smoke.")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "000001.png"), img)

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(repo_root / "run_pipeline.py"),
        "--input",
        str(frames_dir),
        "--output-dir",
        str(out_dir),
        "--preprocess-only",
        "--no-crop",
        "--no-flip",
    ]
    result = subprocess.run(cmd, cwd=repo_root)
    assert result.returncode == 0
    assert (out_dir / "preprocessing_meta.json").exists()
