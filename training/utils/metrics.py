"""mAP evaluation utilities."""

import numpy as np
import torch


def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU matrix between two sets of boxes [N, 4] and [M, 4] in x1y1x2y2 format."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


def compute_ap(recall, precision):
    """Compute AP from recall and precision curves (11-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    points = np.linspace(0, 1, 11)
    ap = 0
    for t in points:
        p = mpre[mrec >= t]
        ap += (p.max() if len(p) > 0 else 0) / 11

    return ap


def evaluate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """
    Compute mAP@iou_threshold.

    predictions: list of dicts per image, each with:
        - boxes: [N, 4] x1y1x2y2
        - scores: [N]
        - classes: [N]
    ground_truths: list of dicts per image, each with:
        - boxes: [M, 4] x1y1x2y2
        - classes: [M]

    Returns: mAP, per-class AP dict
    """
    aps = {}

    for cls in range(num_classes):
        all_scores = []
        all_tp = []
        n_gt = 0

        for pred, gt in zip(predictions, ground_truths):
            cls_mask = pred["classes"] == cls
            pred_boxes = pred["boxes"][cls_mask]
            pred_scores = pred["scores"][cls_mask]

            gt_mask = gt["classes"] == cls
            gt_boxes = gt["boxes"][gt_mask]
            n_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            sorted_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sorted_idx]
            pred_scores = pred_scores[sorted_idx]

            matched_gt = set()
            tp = np.zeros(len(pred_boxes))

            if len(gt_boxes) > 0:
                iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
                for i in range(len(pred_boxes)):
                    best_j = np.argmax(iou_matrix[i])
                    if iou_matrix[i, best_j] >= iou_threshold and best_j not in matched_gt:
                        tp[i] = 1
                        matched_gt.add(best_j)

            all_scores.extend(pred_scores.tolist())
            all_tp.extend(tp.tolist())

        if n_gt == 0:
            aps[cls] = 0.0
            continue

        sorted_idx = np.argsort(-np.array(all_scores))
        tp_sorted = np.array(all_tp)[sorted_idx]

        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(1 - tp_sorted)
        recall = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp)

        aps[cls] = compute_ap(recall, precision)

    map_val = np.mean(list(aps.values())) if aps else 0.0
    return map_val, aps
