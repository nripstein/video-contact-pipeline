from __future__ import annotations

from pipeline import PipelineConfig


def test_config_json_roundtrip(tmp_path):
    cfg = PipelineConfig(
        input_path="data/sample.mp4",
        output_dir="out_dir",
        net="res101",
        cfg_file="cfgs/res101.yml",
        load_dir="models",
        checksession=1,
        checkepoch=8,
        checkpoint=132028,
        cuda=True,
        thresh_hand=0.6,
        thresh_obj=0.2,
        crop_square=480,
        flip_vertical=False,
        blue_glove_filter=True,
        blue_threshold=0.5,
        object_size_filter=True,
        object_size_max_area_ratio=0.5,
        obj_bigger_than_hand_filter=False,
        obj_bigger_ratio_k=1.0,
        obj_smaller_than_hand_filter=True,
        obj_smaller_ratio_factor=2.0,
        obj_match_tiebreak="conf_then_dist_then_iou",
        strict_portable_match=True,
        strict_portable_detected_iou_threshold=0.07,
        tracking_bridge_enabled=True,
        tracking_max_missed_frames=12,
        tracking_contact_iou_threshold=0.22,
        tracking_init_obj_confidence=0.82,
        tracking_promotion_confirm_frames=3,
        tracking_reassociate_iou_threshold=0.18,
        tracking_promote_stationary=True,
        tracking_stationary_iou_threshold=0.24,
        tracking_stationary_confirm_frames=4,
        use_temporal_roi=True,
        temporal_roi_max_missed_frames=6,
        save_full_csv=True,
        save_condensed_csv=True,
        save_visualizations=False,
        save_config=True,
    )

    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = PipelineConfig.load(path)

    assert loaded.to_dict() == cfg.to_dict()
