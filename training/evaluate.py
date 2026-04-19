"""Evaluate trained DroneNet-Pico model on validation set."""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DroneNetPico
from dataset import DroneDataset
from dataset.drone_dataset import collate_fn
from utils.metrics import evaluate_map


def nms_numpy(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression on numpy arrays."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def evaluate(checkpoint_path, data_dir, labels_dir, num_classes=5, input_size=416,
             conf_threshold=0.25, iou_threshold=0.45):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DroneNetPico(num_classes=num_classes, input_size=input_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = DroneDataset(data_dir, labels_dir, input_size=input_size, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=collate_fn, pin_memory=True)

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            output = model(images)  # [B, N, 5+C]

            batch_size = images.shape[0]
            for b in range(batch_size):
                pred = output[b].cpu().numpy()

                # Filter by confidence
                cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
                obj_conf = pred[:, 4]
                cls_scores = pred[:, 5:]

                cls_ids = cls_scores.argmax(axis=1)
                cls_confs = cls_scores.max(axis=1)
                scores = obj_conf * cls_confs

                mask = scores > conf_threshold
                if mask.sum() == 0:
                    all_predictions.append({
                        "boxes": np.zeros((0, 4)),
                        "scores": np.zeros(0),
                        "classes": np.zeros(0, dtype=int),
                    })
                else:
                    boxes = np.stack([
                        cx[mask] - w[mask] / 2,
                        cy[mask] - h[mask] / 2,
                        cx[mask] + w[mask] / 2,
                        cy[mask] + h[mask] / 2,
                    ], axis=1)

                    filtered_scores = scores[mask]
                    filtered_classes = cls_ids[mask]

                    keep = nms_numpy(boxes, filtered_scores, iou_threshold)
                    all_predictions.append({
                        "boxes": boxes[keep],
                        "scores": filtered_scores[keep],
                        "classes": filtered_classes[keep],
                    })

                # Ground truth for this image
                img_targets = targets[targets[:, 0] == b]
                if len(img_targets) > 0:
                    gt_cls = img_targets[:, 1].numpy().astype(int)
                    gt_cx = img_targets[:, 2].numpy()
                    gt_cy = img_targets[:, 3].numpy()
                    gt_w = img_targets[:, 4].numpy()
                    gt_h = img_targets[:, 5].numpy()
                    gt_boxes = np.stack([
                        gt_cx - gt_w / 2,
                        gt_cy - gt_h / 2,
                        gt_cx + gt_w / 2,
                        gt_cy + gt_h / 2,
                    ], axis=1)
                    all_ground_truths.append({"boxes": gt_boxes, "classes": gt_cls})
                else:
                    all_ground_truths.append({"boxes": np.zeros((0, 4)), "classes": np.zeros(0, dtype=int)})

    class_names = ["drone_small", "drone_medium", "drone_large", "bird", "aircraft"]
    map_val, per_class_ap = evaluate_map(all_predictions, all_ground_truths, num_classes)

    print(f"\nmAP@0.5: {map_val:.4f}")
    for cls_id, ap in per_class_ap.items():
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  {name}: AP={ap:.4f}")

    return map_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/train/best.pt")
    parser.add_argument("--images", type=str, default="data/images/val")
    parser.add_argument("--labels", type=str, default="data/labels/val")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=416)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.images, args.labels, args.num_classes, args.input_size)
