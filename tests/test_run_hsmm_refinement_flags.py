from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_hsmm_refinement.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_hsmm_refinement", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_hsmm_new_defaults(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hsmm_refinement.py",
            "--condensed-csv",
            "pred.csv",
            "--gt-csv",
            "gt.csv",
            "--num-trials",
            "2",
            "--alpha-during-trial",
            "30.0",
            "--lambda-during-trial-per-sec",
            "150.0",
            "--alpha-between-trials",
            "32.0",
            "--lambda-between-trials-per-sec",
            "192.0",
            "--end-state",
            "1",
            "--fps",
            "60.0",
        ],
    )
    args = mod.parse_args()
    assert args.max_segment_frames == 540
    assert args.no_max_segment_cap is False
    assert args.numba_mode == "auto"
    assert args.return_posteriors is False


def test_parse_args_hsmm_new_overrides(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hsmm_refinement.py",
            "--condensed-csv",
            "pred.csv",
            "--gt-csv",
            "gt.csv",
            "--num-trials",
            "2",
            "--alpha-during-trial",
            "30.0",
            "--lambda-during-trial-per-sec",
            "150.0",
            "--alpha-between-trials",
            "32.0",
            "--lambda-between-trials-per-sec",
            "192.0",
            "--end-state",
            "1",
            "--fps",
            "60.0",
            "--max-segment-frames",
            "300",
            "--no-max-segment-cap",
            "--numba-mode",
            "off",
            "--return-posteriors",
        ],
    )
    args = mod.parse_args()
    assert args.max_segment_frames == 300
    assert args.no_max_segment_cap is True
    assert args.numba_mode == "off"
    assert args.return_posteriors is True


def test_main_legacy_cli_args_still_work(tmp_path: Path, monkeypatch):
    mod = _load_module()
    condensed_csv = tmp_path / "detections_condensed.csv"
    gt_csv = tmp_path / "gt.csv"
    out_dir = tmp_path / "out_legacy"

    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5, 6],
            "contact_label": [0, 0, 1, 1, 0, 0],
        }
    ).to_csv(condensed_csv, index=False)
    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5, 6],
            "gt_binary": [0, 0, 1, 1, 0, 0],
        }
    ).to_csv(gt_csv, index=False)

    monkeypatch.setattr(mod, "save_original_refined_gt_barcode", lambda **_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hsmm_refinement.py",
            "--condensed-csv",
            str(condensed_csv),
            "--gt-csv",
            str(gt_csv),
            "--k-segments",
            "2",
            "--alpha-during-trial",
            "30.0",
            "--lambda-during-trial",
            "2.5",
            "--alpha-between-trials",
            "32.0",
            "--lambda-between-trials",
            "3.2",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ],
    )

    rc = mod.main()
    assert rc == 0
    config = json.loads((out_dir / "hsmm_config_used.json").read_text())
    assert config["k_segments_derived"] == 2
    assert config["num_trials"] == 1
    assert config["fps"] == 1.0
    assert config["lambda_during_trial_per_frame"] == 2.5
    assert config["lambda_between_trials_per_frame"] == 3.2


def test_main_writes_new_hsmm_config_fields(tmp_path: Path, monkeypatch):
    mod = _load_module()
    condensed_csv = tmp_path / "detections_condensed.csv"
    gt_csv = tmp_path / "gt.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5, 6, 7, 8],
            "contact_label": [0, 0, 1, 1, 1, 0, 0, 1],
        }
    ).to_csv(condensed_csv, index=False)
    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5, 6, 7, 8],
            "gt_binary": [0, 0, 1, 1, 0, 0, 0, 1],
        }
    ).to_csv(gt_csv, index=False)

    def _fake_barcode(**kwargs):
        return None

    def _fake_confidence_barcode(**kwargs):
        Path(kwargs["save_path"]).write_bytes(b"")
        return kwargs["save_path"]

    monkeypatch.setattr(mod, "save_original_refined_gt_barcode", _fake_barcode)
    monkeypatch.setattr(mod, "save_confidence_refined_gt_barcode", _fake_confidence_barcode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hsmm_refinement.py",
            "--condensed-csv",
            str(condensed_csv),
            "--gt-csv",
            str(gt_csv),
            "--num-trials",
            "1",
            "--alpha-during-trial",
            "30.0",
            "--lambda-during-trial-per-sec",
            "150.0",
            "--alpha-between-trials",
            "32.0",
            "--lambda-between-trials-per-sec",
            "192.0",
            "--end-state",
            "1",
            "--fps",
            "60.0",
            "--fpr",
            "0.08",
            "--fnr",
            "0.08",
            "--max-segment-frames",
            "5",
            "--numba-mode",
            "off",
            "--return-posteriors",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ],
    )

    rc = mod.main()
    assert rc == 0

    config = json.loads((out_dir / "hsmm_config_used.json").read_text())
    assert config["max_segment_length_frames"] == 5
    assert config["numba_mode"] == "off"
    assert config["decoder_backend"] == "python"
    assert config["return_posteriors"] is True
    assert config["k_segments_derived"] == 2
    assert config["num_trials"] == 1
    assert config["start_state"] == 0
    assert config["end_state"] == 1
    assert config["fps"] == 60.0
    assert config["lambda_during_trial_per_sec"] == 150.0
    assert config["lambda_during_trial_per_frame"] == 2.5
    assert config["lambda_between_trials_per_sec"] == 192.0
    assert config["lambda_between_trials_per_frame"] == 3.2

    post_df = pd.read_csv(out_dir / "hsmm_posteriors.csv")
    assert list(post_df.columns) == ["frame_number", "posterior_contact"]
    assert len(post_df) == 8
    assert ((post_df["posterior_contact"] >= 0.0) & (post_df["posterior_contact"] <= 1.0)).all()
    assert (out_dir / "barcode_confidence_refined_gt.png").exists()


def test_main_without_posteriors_does_not_write_confidence_barcode(tmp_path: Path, monkeypatch):
    mod = _load_module()
    condensed_csv = tmp_path / "detections_condensed.csv"
    gt_csv = tmp_path / "gt.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5, 6, 7, 8],
            "contact_label": [0, 0, 1, 1, 1, 0, 0, 1],
        }
    ).to_csv(condensed_csv, index=False)
    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5, 6, 7, 8],
            "gt_binary": [0, 0, 1, 1, 0, 0, 0, 1],
        }
    ).to_csv(gt_csv, index=False)

    def _fake_barcode(**kwargs):
        return None

    monkeypatch.setattr(mod, "save_original_refined_gt_barcode", _fake_barcode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hsmm_refinement.py",
            "--condensed-csv",
            str(condensed_csv),
            "--gt-csv",
            str(gt_csv),
            "--num-trials",
            "1",
            "--alpha-during-trial",
            "30.0",
            "--lambda-during-trial-per-sec",
            "150.0",
            "--alpha-between-trials",
            "32.0",
            "--lambda-between-trials-per-sec",
            "192.0",
            "--end-state",
            "1",
            "--fps",
            "60.0",
            "--fpr",
            "0.08",
            "--fnr",
            "0.08",
            "--max-segment-frames",
            "5",
            "--numba-mode",
            "off",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ],
    )

    rc = mod.main()
    assert rc == 0
    assert not (out_dir / "hsmm_posteriors.csv").exists()
    assert not (out_dir / "barcode_confidence_refined_gt.png").exists()


def test_main_errors_on_k_segment_mismatch(tmp_path: Path, monkeypatch):
    mod = _load_module()
    condensed_csv = tmp_path / "detections_condensed.csv"
    gt_csv = tmp_path / "gt.csv"

    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4],
            "contact_label": [0, 1, 0, 1],
        }
    ).to_csv(condensed_csv, index=False)
    pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4],
            "gt_binary": [0, 1, 0, 1],
        }
    ).to_csv(gt_csv, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hsmm_refinement.py",
            "--condensed-csv",
            str(condensed_csv),
            "--gt-csv",
            str(gt_csv),
            "--k-segments",
            "5",
            "--num-trials",
            "1",
            "--start-state",
            "0",
            "--end-state",
            "1",
            "--alpha-during-trial",
            "30.0",
            "--lambda-during-trial-per-sec",
            "150.0",
            "--alpha-between-trials",
            "32.0",
            "--lambda-between-trials-per-sec",
            "192.0",
            "--fps",
            "60.0",
            "--no-progress",
        ],
    )

    with pytest.raises(ValueError, match="k_segments mismatch"):
        mod.main()
