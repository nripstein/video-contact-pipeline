from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.types import FilterContext, PseudoLabel


class DetectionFilter(ABC):
    @abstractmethod
    def filter(self, detections: List[PseudoLabel], context: FilterContext) -> List[PseudoLabel]:
        raise NotImplementedError


class BboxSimilarityFilter(DetectionFilter):
    def filter(self, detections: List[PseudoLabel], context: FilterContext) -> List[PseudoLabel]:
        if not detections:
            return []

        ordered = sorted(detections, key=lambda d: (d.frame.frame_time_sec, d.frame.frame_idx))
        kept = [ordered[0]]

        for det in ordered[1:]:
            last = kept[-1]

            time_gap = det.frame.frame_time_sec - last.frame.frame_time_sec
            if time_gap < context.min_temporal_gap_sec:
                continue

            curr_center = det.center()
            last_center = last.center()
            center_move = math.sqrt(
                (curr_center[0] - last_center[0]) ** 2 + (curr_center[1] - last_center[1]) ** 2
            )
            center_threshold = context.center_move_frac * float(max(1, det.frame.width))
            if center_move <= center_threshold:
                continue

            last_area = max(1e-6, last.area())
            area_change = abs(det.area() - last.area()) / last_area
            if area_change <= context.area_change_frac:
                continue

            det.filter_trace.append("bbox_similarity")
            kept.append(det)

        return kept


class KMeansClusterFilter(DetectionFilter):
    def filter(self, detections: List[PseudoLabel], context: FilterContext) -> List[PseudoLabel]:
        raise NotImplementedError("KMeansClusterFilter is planned for a later phase.")


class TemporalSubsamplingFilter(DetectionFilter):
    def filter(self, detections: List[PseudoLabel], context: FilterContext) -> List[PseudoLabel]:
        if not detections:
            return []

        ordered = sorted(detections, key=lambda d: (d.frame.frame_time_sec, d.frame.frame_idx))
        kept = [ordered[0]]
        last_time = ordered[0].frame.frame_time_sec

        for det in ordered[1:]:
            if det.frame.frame_time_sec - last_time >= context.subsample_interval_sec:
                det.filter_trace.append("temporal_subsample")
                kept.append(det)
                last_time = det.frame.frame_time_sec

        return kept


def build_filter(config: Phase1Config) -> DetectionFilter:
    strategy = config.filter_strategy.lower().strip()
    if strategy == "bbox_similarity":
        return BboxSimilarityFilter()
    if strategy == "temporal_subsampling":
        return TemporalSubsamplingFilter()
    if strategy == "kmeans":
        return KMeansClusterFilter()
    raise ValueError(f"Unknown filter strategy: {config.filter_strategy}")


def build_filter_context(config: Phase1Config) -> FilterContext:
    return FilterContext(
        min_temporal_gap_sec=config.min_temporal_gap_sec,
        center_move_frac=config.center_move_frac,
        area_change_frac=config.area_change_frac,
        subsample_interval_sec=config.subsample_interval_sec,
    )
