from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sweep_timestamp_extractor_params.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sweep_timestamp_extractor_params", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sweep_dry_run_writes_expected_files(tmp_path: Path):
    mod = _load_module()

    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_csv = run_root / "run_manifest.csv"
    pd.DataFrame(
        [
            {"dataset_key": "nr_all_nr", "pred_dir": "predictions/nr_all_nr", "gt_csv": "gt/nr.csv", "status": "success"},
            {"dataset_key": "sv1", "pred_dir": "predictions/sv1", "gt_csv": "gt/sv1.csv", "status": "success"},
        ]
    ).to_csv(manifest_csv, index=False)

    out_dir = tmp_path / "sweep_out"
    args = argparse.Namespace(
        run_root=str(run_root),
        manifest_csv=str(manifest_csv),
        output_dir=str(out_dir),
        fps_group=["30:nr_all_nr", "60:sv1"],
        min_island_seconds_grid="1.0,2.0",
        join_mode_grid="inner",
        random_seed=42,
        strict_duplicates=False,
        dry_run=True,
    )
    rc = mod.run(args)
    assert rc == 0

    sweep_csv = out_dir / "sweep_results.csv"
    planned_json = out_dir / "planned_commands.json"
    metadata_json = out_dir / "sweep_metadata.json"

    assert sweep_csv.exists()
    assert planned_json.exists()
    assert metadata_json.exists()

    sweep_df = pd.read_csv(sweep_csv)
    # 2 min-island values x 1 join mode = 2 experiments.
    assert len(sweep_df) == 2
    assert set(sweep_df["experiment"].tolist()) == {"mins_1__join_inner", "mins_2__join_inner"}

    planned = json.loads(planned_json.read_text(encoding="utf-8"))
    # Per experiment: 2 group commands + 1 merged eval command = 3 commands.
    assert len(planned) == 6

    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    assert metadata["dry_run"] is True
    assert metadata["min_island_seconds_grid"] == [1.0, 2.0]
    assert len(metadata["fps_groups"]) == 2

