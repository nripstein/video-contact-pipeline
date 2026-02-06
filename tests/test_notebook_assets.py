from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_docs_and_starter_exist():
    docs_path = REPO_ROOT / "docs" / "NOTEBOOKS.md"
    starter_path = REPO_ROOT / "notebooks" / "pipeline_starter.ipynb"
    postprocess_path = REPO_ROOT / "notebooks" / "postprocess_only.ipynb"
    env_path = REPO_ROOT / "environment.yml"
    bootstrap_path = REPO_ROOT / "scripts" / "bootstrap_notebook.sh"

    assert docs_path.exists(), f"Missing file: {docs_path}"
    assert starter_path.exists(), f"Missing file: {starter_path}"
    assert postprocess_path.exists(), f"Missing file: {postprocess_path}"
    assert env_path.exists(), f"Missing file: {env_path}"
    assert bootstrap_path.exists(), f"Missing file: {bootstrap_path}"


def test_notebook_docs_include_bootstrap_instructions():
    docs_path = REPO_ROOT / "docs" / "NOTEBOOKS.md"
    docs = docs_path.read_text(encoding="utf-8")

    required_snippets = [
        "scripts/bootstrap_notebook.sh shan_et_al2",
        "conda env update -n shan_et_al2 -f environment.yml",
        "python -m pip install -e .",
        "Python (shan_et_al2)",
    ]
    for snippet in required_snippets:
        assert snippet in docs, f"Missing docs snippet: {snippet}"


def _load_notebook_source(notebook_path: Path) -> str:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4
    cells = data.get("cells", [])
    assert cells, "Notebook has no cells"
    return "\n".join(
        "".join(cell.get("source", [])) for cell in cells if isinstance(cell, dict)
    )


def test_pipeline_starter_notebook_required_imports_present():
    notebook_path = REPO_ROOT / "notebooks" / "pipeline_starter.ipynb"
    all_source = _load_notebook_source(notebook_path)

    required_snippets = [
        "from pipeline.config import PipelineConfig",
        "from pipeline.main import run_pipeline",
        "from pipeline.visualization import save_barcodes, save_annotated_frames",
    ]
    for snippet in required_snippets:
        assert snippet in all_source, f"Missing notebook snippet: {snippet}"


def test_postprocess_only_notebook_required_imports_present():
    notebook_path = REPO_ROOT / "notebooks" / "postprocess_only.ipynb"
    all_source = _load_notebook_source(notebook_path)

    required_snippets = [
        "import pandas as pd",
        "from pipeline.visualization import save_barcodes, save_annotated_frames",
        "condensed_csv = '/path/to/detections_condensed.csv'",
        "full_csv = '/path/to/detections_full.csv'",
        "image_dir = '/path/to/frames_dir'",
    ]
    for snippet in required_snippets:
        assert snippet in all_source, f"Missing notebook snippet: {snippet}"
