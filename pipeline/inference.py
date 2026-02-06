from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file

from pipeline.config import PipelineConfig
from pipeline.preprocessing import get_sorted_image_list


def _get_image_blob(im: np.ndarray):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    try:
        im_orig = im.astype(np.float32, copy=True)
    except Exception:
        print(type(im))
        print(im)
    # im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def get_blue_bbox_proportion(img: np.ndarray, bbox: List[int]) -> float:
    """
    Args:
        img (np.ndarray): BGR from OpenCV
        bbox (list[int]): Of the form [left, top, right, bottom]
    """
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the "blue" color in HSV
    LOWER_BLUE = np.array([90, 50, 50])
    UPPER_BLUE = np.array([130, 255, 255])
    # Create a mask that isolates the pixels within the specified blue range
    blue_mask = cv2.inRange(hsv_img, LOWER_BLUE, UPPER_BLUE)
    # Extract the bounding box area
    bbox_area = blue_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Count the number of blue pixels within the bounding box
    blue_pixels = np.count_nonzero(bbox_area)
    # Calculate the total area of the bounding box
    total_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    # Calculate the percentage of blue pixels within the bounding box
    blue_proportion = blue_pixels / total_area
    return blue_proportion


class HandObjectDetector:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.class_agnostic = False

        if config.cfg_file is not None:
            cfg_from_file(config.cfg_file)

        cfg.USE_GPU_NMS = config.cuda
        np.random.seed(cfg.RNG_SEED)

        model_dir = config.load_dir + "/" + config.net + "_handobj_100K" + "/" + "pascal_voc"
        if not os.path.exists(model_dir):
            raise Exception(
                'There is no input directory for loading network from ' +
                model_dir)
        load_name = os.path.join(
            model_dir,
            'faster_rcnn_{}_{}_{}.pth'.format(
                config.checksession,
                config.checkepoch,
                config.checkpoint))

        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
        set_cfgs = [
            'ANCHOR_SCALES',
            '[8, 16, 32, 64]',
            'ANCHOR_RATIOS',
            '[0.5, 1, 2]']

        if config.net == 'vgg16':
            fasterRCNN = vgg16(
                self.pascal_classes,
                pretrained=False,
                class_agnostic=self.class_agnostic)
        elif config.net == 'res101':
            fasterRCNN = resnet(
                self.pascal_classes,
                101,
                pretrained=False,
                class_agnostic=self.class_agnostic)
        elif config.net == 'res50':
            fasterRCNN = resnet(
                self.pascal_classes,
                50,
                pretrained=False,
                class_agnostic=self.class_agnostic)
        elif config.net == 'res152':
            fasterRCNN = resnet(
                self.pascal_classes,
                152,
                pretrained=False,
                class_agnostic=self.class_agnostic)
        else:
            print("network is not defined")
            raise RuntimeError("network is not defined")

        fasterRCNN.create_architecture()

        if config.cuda:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(
                load_name, map_location=(
                    lambda storage, loc: storage))
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)
        self.box_info = torch.FloatTensor(1)

        if config.cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        with torch.no_grad():
            if config.cuda:
                cfg.CUDA = True
            if config.cuda:
                fasterRCNN.cuda()
            fasterRCNN.eval()

        self.fasterRCNN = fasterRCNN
        self.thresh_hand = config.thresh_hand
        self.thresh_obj = config.thresh_obj

    def detect_single_image(self, im: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        obj_dets, hand_dets = None, None
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()
            self.box_info.resize_(1, 1, 5).zero_()

        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0]  # hand contact state info
        # offset vector (factored into a unit vector and a magnitude)
        offset_vector = loss_list[1][0].detach()
        lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

        # get hand contact
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and
                # stdev
                if self.class_agnostic:
                    if self.config.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda(
                        ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.config.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda(
                        ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(
                        1, -1, 4 * len(self.pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        max_per_image = 100
        thresh_hand = self.thresh_hand
        thresh_obj = self.thresh_obj

        for j in range(1, len(self.pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if self.pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
            elif self.pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat(
                    (cls_boxes,
                     cls_scores.unsqueeze(1),
                     contact_indices[inds],
                        offset_vector.squeeze(0)[inds],
                        lr[inds]),
                    1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :],
                           cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if self.pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                if self.pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()

        return {"hand_dets": hand_dets, "obj_dets": obj_dets}

    def run_on_directory(self, image_dir: str) -> pd.DataFrame:
        df_row_list: List[Dict[str, object]] = []
        image_list = get_sorted_image_list(image_dir)
        iterable = image_list
        if self.config.show_progress:
            iterable = tqdm(image_list, desc="Processing Images")
        state_map = {0: 'No Contact', 1: 'Self Contact', 2: 'Other Person Contact', 3: 'Portable Object', 4: 'Stationary Object Contact'}
        side_map2 = {0: 'Left', 1: 'Right'}

        for idx, img_path in enumerate(iterable):
            im = cv2.imread(img_path)
            if im is None:
                continue

            dets = self.detect_single_image(im)
            hand_dets = dets["hand_dets"]
            obj_dets = dets["obj_dets"]

            frame_id = Path(img_path).name
            frame_number = _parse_frame_number(frame_id, idx)

            if hand_dets is not None:
                for i in range(np.minimum(10, hand_dets.shape[0])):
                    bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
                    score = hand_dets[i, 4]
                    lr = int(hand_dets[i, 9])
                    state = int(hand_dets[i, 5])

                    blue_prop = None
                    blue_status = "NA"
                    if self.config.blue_glove_filter:
                        blue_prop = get_blue_bbox_proportion(im, bbox=bbox)
                        if blue_prop > self.config.blue_threshold:
                            state = 0
                            score = 1
                            blue_status = "blue_discard"

                    confidence = float(int(score * 100))
                    df_row_dict = {
                        "frame_id": frame_id,
                        "frame_number": frame_number,
                        "detection_type": "hand",
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "confidence": confidence,
                        "contact_state": state,
                        "contact_label": state_map[state],
                        "hand_side": side_map2[lr],
                        "offset_x": float(hand_dets[i, 6]),
                        "offset_y": float(hand_dets[i, 7]),
                        "offset_mag": float(hand_dets[i, 8]),
                        "blue_prop": blue_prop,
                        "blue_glove_status": blue_status,
                        "is_filtered": False,
                        "filtered_by": "",
                        "filtered_reason": "",
                    }
                    df_row_list.append(df_row_dict)

            if obj_dets is not None:
                for i in range(np.minimum(10, obj_dets.shape[0])):
                    bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
                    score = obj_dets[i, 4]
                    confidence = float(int(score * 100))
                    df_row_dict = {
                        "frame_id": frame_id,
                        "frame_number": frame_number,
                        "detection_type": "object",
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "confidence": confidence,
                        "contact_state": None,
                        "contact_label": None,
                        "hand_side": None,
                        "offset_x": None,
                        "offset_y": None,
                        "offset_mag": None,
                        "blue_prop": None,
                        "blue_glove_status": "NA",
                        "is_filtered": False,
                        "filtered_by": "",
                        "filtered_reason": "",
                    }
                    df_row_list.append(df_row_dict)

        return pd.DataFrame(df_row_list)


def _parse_frame_number(name: str, fallback: int) -> int:
    stem = Path(name).stem
    num = None
    current = ""
    for ch in stem:
        if ch.isdigit():
            current += ch
        elif current:
            num = int(current)
            break
    if current and num is None:
        num = int(current)
    return num if num is not None else fallback
