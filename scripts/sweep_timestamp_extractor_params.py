from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep timestamp extractor/evaluation settings across mixed-FPS dataset groups "
            "by invoking scripts/run_timestamp_supervision_baseline.py."
        )
    )
    parser.add_argument("--run-root", required=True, help="Run root containing run_manifest.csv and predictions.")
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Optional explicit manifest CSV (recommended if run_manifest pred_dir paths were fixed).",
    )
    parser.add_argument("--output-dir", required=True, help="Sweep output root.")
    parser.add_argument(
        "--fps-group",
        action="append",
        required=True,
        help=(
            "Group spec in form '<fps>:<dataset_key1,dataset_key2,...>'. "
            "Use multiple --fps-group args for mixed FPS."
        ),
    )
    parser.add_argument(
        "--min-island-seconds-grid",
        required=True,
        help="Comma-separated min-island-seconds values, e.g. '0.5,1.0,1.5,2.0'",
    )
    parser.add_argument(
        "--join-mode-grid",
        default="inner,left",
        help="Comma-separated join modes to test (subset of: inner,left).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Passed through for reproducibility/compat.")
    parser.add_argument("--strict-duplicates", action="store_true", help="Enable strict duplicate checks at eval.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_grid_floats(raw: str) -> List[float]:
    vals: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("Grid must contain at least one numeric value.")
    return vals


def _parse_grid_join_modes(raw: str) -> List[str]:
    allowed = {"inner", "left"}
    modes = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not modes:
        raise ValueError("join-mode grid is empty.")
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported join modes: {bad}. Allowed: {sorted(allowed)}")
    return modes


def _parse_fps_groups(group_specs: Sequence[str]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for idx, spec in enumerate(group_specs):
        if ":" not in spec:
            raise ValueError(
                f"Invalid --fps-group value: {spec!r}. Expected '<fps>:<dataset1,dataset2,...>'."
            )
        fps_raw, keys_raw = spec.split(":", 1)
        fps = float(fps_raw.strip())
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if fps <= 0:
            raise ValueError(f"FPS must be > 0 in --fps-group: {spec!r}")
        if not keys:
            raise ValueError(f"No dataset keys provided in --fps-group: {spec!r}")
        groups.append(
            {
                "group_index": idx,
                "group_name": f"group{idx+1}_fps{int(fps) if float(fps).is_integer() else fps}",
                "fps": fps,
                "dataset_keys": keys,
                "dataset_keys_csv": ",".join(keys),
                "raw_spec": spec,
            }
        )
    return groups


def _run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("$ " + " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _merge_eval_manifests(manifests: Sequence[Path], out_csv: Path, dry_run: bool) -> None:
    if dry_run:
        return
    rows = []
    for path in manifests:
        df = pd.read_csv(path)
        rows.append(df)
    merged = pd.concat(rows, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["dataset_key"], keep="first")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)


def _read_eval_summary(summary_json: Path) -> Dict[str, Any]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    global_micro = payload.get("global_micro")
    global_macro = payload.get("global_macro")
    return {
        "n_datasets_input": int(payload.get("n_datasets_input", 0)),
        "n_datasets_success": int(payload.get("n_datasets_success", 0)),
        "n_datasets_failed": int(payload.get("n_datasets_failed", 0)),
        "global_micro_accuracy": (
            float(global_micro["accuracy"]) if isinstance(global_micro, dict) and "accuracy" in global_micro else None
        ),
        "global_macro_accuracy": (
            float(global_macro["accuracy"]) if isinstance(global_macro, dict) and "accuracy" in global_macro else None
        ),
    }


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).expanduser()
    if not run_root.exists():
        raise FileNotFoundError(f"--run-root does not exist: {run_root}")

    manifest_csv = (
        Path(args.manifest_csv).expanduser() if args.manifest_csv else (run_root / "run_manifest.csv")
    )
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    min_seconds_grid = _parse_grid_floats(args.min_island_seconds_grid)
    join_modes = _parse_grid_join_modes(args.join_mode_grid)
    fps_groups = _parse_fps_groups(args.fps_group)

    sweep_rows: List[Dict[str, Any]] = []
    baseline_script = REPO_ROOT / "scripts" / "run_timestamp_supervision_baseline.py"
    if not baseline_script.exists():
        raise FileNotFoundError(f"Baseline script not found: {baseline_script}")

    planned: List[Dict[str, Any]] = []

    for min_seconds in min_seconds_grid:
        for join_mode in join_modes:
            exp_name = f"mins_{min_seconds:g}__join_{join_mode}"
            exp_root = output_root / exp_name
            group_manifests: List[Path] = []
            group_rcs: List[int] = []

            for group in fps_groups:
                group_root = exp_root / group["group_name"]
                cmd = [
                    sys.executable,
                    str(baseline_script),
                    "--run-root",
                    str(run_root),
                    "--manifest-csv",
                    str(manifest_csv),
                    "--dataset-keys",
                    group["dataset_keys_csv"],
                    "--fps",
                    str(group["fps"]),
                    "--min-island-seconds",
                    str(min_seconds),
                    "--join-mode",
                    str(join_mode),
                    "--random-seed",
                    str(args.random_seed),
                    "--output-dir",
                    str(group_root),
                ]
                if args.strict_duplicates:
                    cmd.append("--strict-duplicates")

                planned.append(
                    {
                        "experiment": exp_name,
                        "group_name": group["group_name"],
                        "command": cmd,
                    }
                )

                rc = _run_cmd(cmd, dry_run=bool(args.dry_run))
                group_rcs.append(int(rc))
                manifest_path = group_root / "evaluation_manifest.csv"
                if rc == 0 and manifest_path.exists():
                    group_manifests.append(manifest_path)

            eval_manifest = exp_root / "evaluation_manifest_all.csv"
            eval_output = exp_root / "evaluation_all"
            summary_json = eval_output / "selected_timestamp_metrics_summary.json"
            eval_rc = 1

            if not args.dry_run and group_manifests:
                _merge_eval_manifests(group_manifests, eval_manifest, dry_run=False)
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "timestamp_supervision_extraction.evaluate_selected_timestamps",
                    "--manifest-csv",
                    str(eval_manifest),
                    "--output-dir",
                    str(eval_output),
                ]
                eval_rc = _run_cmd(eval_cmd, dry_run=False)
            elif args.dry_run:
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "timestamp_supervision_extraction.evaluate_selected_timestamps",
                    "--manifest-csv",
                    str(eval_manifest),
                    "--output-dir",
                    str(eval_output),
                ]
                planned.append({"experiment": exp_name, "group_name": "__merged_eval__", "command": eval_cmd})
                eval_rc = 0

            row: Dict[str, Any] = {
                "experiment": exp_name,
                "min_island_seconds": float(min_seconds),
                "join_mode": join_mode,
                "n_groups": int(len(fps_groups)),
                "group_return_codes": ",".join(str(x) for x in group_rcs),
                "any_group_failed": bool(any(x != 0 for x in group_rcs)),
                "merged_eval_return_code": int(eval_rc),
                "eval_manifest_csv": str(eval_manifest),
                "eval_summary_json": str(summary_json),
            }
            if not args.dry_run and eval_rc == 0 and summary_json.exists():
                row.update(_read_eval_summary(summary_json))
            else:
                row.update(
                    {
                        "n_datasets_input": None,
                        "n_datasets_success": None,
                        "n_datasets_failed": None,
                        "global_micro_accuracy": None,
                        "global_macro_accuracy": None,
                    }
                )
            sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows)
    if "global_micro_accuracy" in sweep_df.columns:
        sweep_df = sweep_df.sort_values(
            by=["global_micro_accuracy", "any_group_failed", "experiment"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    sweep_csv = output_root / "sweep_results.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    planned_json = output_root / "planned_commands.json"
    planned_json.write_text(json.dumps(planned, indent=2), encoding="utf-8")

    metadata = {
        "generated_at_utc": _utc_now(),
        "run_root": str(run_root),
        "manifest_csv": str(manifest_csv),
        "output_dir": str(output_root),
        "min_island_seconds_grid": min_seconds_grid,
        "join_mode_grid": join_modes,
        "fps_groups": fps_groups,
        "dry_run": bool(args.dry_run),
        "strict_duplicates": bool(args.strict_duplicates),
        "random_seed": int(args.random_seed),
        "results_csv": str(sweep_csv),
    }
    metadata_json = output_root / "sweep_metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"sweep_results_csv: {sweep_csv}")
    print(f"planned_commands_json: {planned_json}")
    print(f"sweep_metadata_json: {metadata_json}")
    if not sweep_df.empty:
        top = sweep_df.head(5)
        print("top_experiments_by_global_micro:")
        for _, row in top.iterrows():
            print(
                f"  {row['experiment']} micro={row.get('global_micro_accuracy')} "
                f"macro={row.get('global_macro_accuracy')} "
                f"failed_datasets={row.get('n_datasets_failed')}"
            )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"timestamp sweep failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
