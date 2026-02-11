from __future__ import annotations

import sys

import run_pipeline


def test_tracking_flags_default_values(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--input", "dummy"])
    args = run_pipeline.parse_args()

    assert args.tracking_bridge is False
    assert args.tracking_max_missed_frames == 8
    assert args.tracking_iou_threshold == 0.15
    assert args.tracking_init_obj_confidence == 0.70
    assert args.tracking_promotion_confirm_frames == 2
    assert args.tracking_reassociate_iou_threshold == 0.10
    assert args.tracking_promote_stationary is False
    assert args.tracking_stationary_iou_threshold == 0.20
    assert args.tracking_stationary_confirm_frames == 2
    assert args.strict_portable_match is False
    assert args.strict_portable_detected_iou_threshold == 0.05
    assert args.condense_priority_strategy == "no_contact_first"
    assert args.use_temporal_roi is False
    assert args.temporal_roi_max_missed_frames == 8


def test_tracking_flags_override_values(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--input",
            "dummy",
            "--tracking-bridge",
            "--tracking-max-missed-frames",
            "5",
            "--tracking-iou-threshold",
            "0.2",
            "--tracking-init-obj-confidence",
            "0.8",
            "--tracking-promotion-confirm-frames",
            "3",
            "--tracking-reassociate-iou-threshold",
            "0.25",
            "--tracking-promote-stationary",
            "--tracking-stationary-iou-threshold",
            "0.3",
            "--tracking-stationary-confirm-frames",
            "4",
            "--strict-portable-match",
            "--strict-portable-detected-iou-threshold",
            "0.07",
            "--condense-priority-strategy",
            "portable_first",
            "--use-temporal-roi",
            "--temporal-roi-max-missed-frames",
            "5",
        ],
    )
    args = run_pipeline.parse_args()

    assert args.tracking_bridge is True
    assert args.tracking_max_missed_frames == 5
    assert args.tracking_iou_threshold == 0.2
    assert args.tracking_init_obj_confidence == 0.8
    assert args.tracking_promotion_confirm_frames == 3
    assert args.tracking_reassociate_iou_threshold == 0.25
    assert args.tracking_promote_stationary is True
    assert args.tracking_stationary_iou_threshold == 0.3
    assert args.tracking_stationary_confirm_frames == 4
    assert args.strict_portable_match is True
    assert args.strict_portable_detected_iou_threshold == 0.07
    assert args.condense_priority_strategy == "portable_first"
    assert args.use_temporal_roi is True
    assert args.temporal_roi_max_missed_frames == 5


def test_temporal_roi_underscore_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--input",
            "dummy",
            "--use_temporal_roi",
        ],
    )
    args = run_pipeline.parse_args()
    assert args.use_temporal_roi is True
