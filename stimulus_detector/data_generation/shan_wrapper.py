from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm

from stimulus_detector.config.phase1_config import Phase1Config
from stimulus_detector.data_generation.types import HandDetection, ObjectDetection, SequenceFrames


class ShanInferenceWrapper:
    def __init__(self, config: Phase1Config, detector: Optional[object] = None):
        self.config = config
        if detector is not None:
            self.detector = detector
        else:
            from pipeline.inference import HandObjectDetector

            self.detector = HandObjectDetector(self._to_pipeline_config(config))

    @staticmethod
    def _to_pipeline_config(config: Phase1Config):
        from pipeline.config import PipelineConfig

        return PipelineConfig(
            input_path=config.input_path,
            output_dir=config.output_dir,
            net=config.net,
            cfg_file=config.cfg_file,
            load_dir=config.load_dir,
            checksession=config.checksession,
            checkepoch=config.checkepoch,
            checkpoint=config.checkpoint,
            cuda=config.cuda,
            thresh_hand=config.thresh_hand,
            thresh_obj=config.thresh_obj,
            show_progress=False,
            blue_glove_filter=False,
            object_size_filter=False,
            save_visualizations=False,
            save_full_csv=False,
            save_condensed_csv=False,
            save_config=False,
        )

    @staticmethod
    def _xyxy_from_det_row(row: np.ndarray) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = row[:4].tolist()
        return (float(x1), float(y1), float(x2), float(y2))

    def run_on_sequence(
        self,
        sequence: SequenceFrames,
    ) -> Tuple[List[ObjectDetection], List[HandDetection]]:
        objects: List[ObjectDetection] = []
        hands: List[HandDetection] = []

        frame_iter = sequence.frame_records
        if self.config.show_progress:
            frame_iter = tqdm(frame_iter, desc=f"Shan inference: {sequence.video_id}")

        for frame_record in frame_iter:
            img = cv2.imread(frame_record.frame_path)
            if img is None:
                continue

            dets = self.detector.detect_single_image(img)
            obj_dets = dets.get("obj_dets")
            hand_dets = dets.get("hand_dets")

            if obj_dets is not None:
                for row in obj_dets:
                    objects.append(
                        ObjectDetection(
                            frame=frame_record,
                            bbox_xyxy=self._xyxy_from_det_row(row),
                            confidence=float(row[4]),
                            source="shan_targetobject",
                        )
                    )

            if hand_dets is not None:
                for row in hand_dets:
                    hands.append(
                        HandDetection(
                            frame=frame_record,
                            bbox_xyxy=self._xyxy_from_det_row(row),
                            confidence=float(row[4]),
                        )
                    )

        return objects, hands
