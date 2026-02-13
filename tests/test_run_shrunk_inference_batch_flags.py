from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_shrunk_inference_batch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_shrunk_inference_batch", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_strict_flags_default_and_override():
    mod = _load_module()

    import sys

    original_argv = sys.argv
    sys.argv = ["run_shrunk_inference_batch.py"]
    try:
        args = mod.parse_args()
    finally:
        sys.argv = original_argv

    assert args.strict_portable_match is False
    assert args.strict_portable_detected_iou_threshold == 0.05
    assert args.condense_priority_strategy == "no_contact_first"

    original_argv = sys.argv
    sys.argv = [
        "run_shrunk_inference_batch.py",
        "--strict-portable-match",
        "--strict-portable-detected-iou-threshold",
        "0.11",
        "--condense-priority-strategy",
        "portable_first",
    ]
    try:
        args = mod.parse_args()
    finally:
        sys.argv = original_argv

    assert args.strict_portable_match is True
    assert args.strict_portable_detected_iou_threshold == 0.11
    assert args.condense_priority_strategy == "portable_first"
