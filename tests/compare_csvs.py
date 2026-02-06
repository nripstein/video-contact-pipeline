"""
Utility to compare two CSV files with deterministic sorting and tolerance for floats.

Usage:
    python -m tests.compare_csvs a.csv b.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


FLOAT_TOL = 1e-6


def _choose_sort_keys(columns: Iterable[str]) -> List[str]:
    preferred = [
        "frame_id",
        "image",
        "type",
        "which",
        "contact_label_pred",
        "probability",
    ]
    cols = list(columns)
    keys = [c for c in preferred if c in cols]
    for c in sorted(cols):
        if c not in keys:
            keys.append(c)
    return keys


def _sort_df(df: pd.DataFrame, sort_keys: List[str]) -> pd.DataFrame:
    existing = [k for k in sort_keys if k in df.columns]
    return df.sort_values(by=existing).reset_index(drop=True)


def _values_equal(a, b) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        # Treat NaN comparisons as equal only if both are NaN
        if pd.isna(fa) and pd.isna(fb):
            return True
        return abs(fa - fb) <= FLOAT_TOL
    except Exception:
        return a == b


def _format_row(row: pd.Series, columns: Sequence[str]) -> str:
    parts = []
    for col in columns:
        val = row.get(col, None)
        parts.append(f"{col}={val!r}")
    return ", ".join(parts)


def _report_path_text(report_path: Path | None) -> str:
    return f" (report: {report_path})" if report_path else ""


def compare_csvs(
    left_path: Path,
    right_path: Path,
    sort_keys: List[str] | None = None,
    report_path: Path | None = None,
) -> None:
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)

    missing = set(left.columns) - set(right.columns)
    extra = set(right.columns) - set(left.columns)
    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"missing columns in right: {sorted(missing)}")
        if extra:
            msg_parts.append(f"extra columns in right: {sorted(extra)}")
        message = "; ".join(msg_parts)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(message + "\n")
        raise AssertionError(message + _report_path_text(report_path))

    if len(left) != len(right):
        message = f"row count differs: left={len(left)} right={len(right)}"
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(message + "\n")
        raise AssertionError(message + _report_path_text(report_path))

    keys = sort_keys or _choose_sort_keys(left.columns)
    left_sorted = _sort_df(left, keys)
    right_sorted = _sort_df(right, keys)

    for idx in range(len(left_sorted)):
        row_l = left_sorted.iloc[idx]
        row_r = right_sorted.iloc[idx]
        for col in left_sorted.columns:
            if not _values_equal(row_l[col], row_r[col]):
                sample_cols = list(left_sorted.columns[:10])
                details = [
                    f"mismatch at row {idx}, column '{col}': left={row_l[col]!r}, right={row_r[col]!r}",
                    f"sort keys: {keys}",
                    f"left row:  {_format_row(row_l, sample_cols)}",
                    f"right row: {_format_row(row_r, sample_cols)}",
                ]
                message = "\n".join(details)
                if report_path:
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(message + "\n")
                raise AssertionError(message + _report_path_text(report_path))


def main():
    parser = argparse.ArgumentParser(description="Compare two CSV files.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--sort-key", action="append", dest="sort_keys", help="Column to sort by (can repeat)")
    parser.add_argument("--report", type=Path, default=None, help="Write mismatch report to a file")
    args = parser.parse_args()

    compare_csvs(args.left, args.right, sort_keys=args.sort_keys, report_path=args.report)
    print("CSV files match.")


if __name__ == "__main__":
    main()
