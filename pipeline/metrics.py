from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def frame_accuracy(pred: Sequence[int], gt: Sequence[int]) -> float:
    pred_arr = np.asarray(pred, dtype=int)
    gt_arr = np.asarray(gt, dtype=int)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError("pred and gt must have the same shape")
    if pred_arr.size == 0:
        return 0.0
    return float((pred_arr == gt_arr).mean())


def get_labels_start_end_time(frame_wise_labels: Sequence[int], bg_class: Iterable[int] = (0,)) -> Tuple[List[int], List[int], List[int]]:
    labels: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    last_label = frame_wise_labels[0]
    if last_label not in bg_class:
        labels.append(last_label)
        starts.append(0)
    for i, label in enumerate(frame_wise_labels):
        if label != last_label:
            if label not in bg_class:
                labels.append(label)
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = label
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def f_score(recognized: Sequence[int], ground_truth: Sequence[int], overlap: float, bg_class: Iterable[int] = (0,)) -> float:
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        iou = (1.0 * intersection / union) * np.array([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = int(np.argmax(iou)) if len(iou) else 0

        if len(iou) and iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(y_label) - np.sum(hits)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return float(f1 * 100.0)


def levenshtein(p: Sequence[int], y: Sequence[int], norm: bool = False) -> float:
    m, n = len(p), len(y)
    d = np.zeros((m + 1, n + 1), dtype=np.float64)

    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == y[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)

    if norm:
        return float((1 - d[m, n] / max(m, n)) * 100.0) if max(m, n) > 0 else 0.0
    return float(d[m, n])


def edit_score(recognized: Sequence[int], ground_truth: Sequence[int], bg_class: Iterable[int] = (0,), norm: bool = True) -> float:
    p, _, _ = get_labels_start_end_time(recognized, bg_class)
    y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return float(levenshtein(p, y, norm=norm))


def confusion_counts(pred: Sequence[int], gt: Sequence[int]) -> Tuple[int, int, int, int]:
    pred_arr = np.asarray(pred, dtype=int)
    gt_arr = np.asarray(gt, dtype=int)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError("pred and gt must have the same shape")
    tp = int(((pred_arr == 1) & (gt_arr == 1)).sum())
    tn = int(((pred_arr == 0) & (gt_arr == 0)).sum())
    fp = int(((pred_arr == 1) & (gt_arr == 0)).sum())
    fn = int(((pred_arr == 0) & (gt_arr == 1)).sum())
    return tp, fp, tn, fn
