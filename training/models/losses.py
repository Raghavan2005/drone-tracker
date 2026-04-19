"""Loss functions for DroneNet-Pico training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(pred, target):
    """Complete IoU loss between predicted and target boxes (x1y1x2y2 format)."""
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    union_area = pred_area + target_area - inter_area + 1e-7

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(px1, tx1)
    enc_y1 = torch.min(py1, ty1)
    enc_x2 = torch.max(px2, tx2)
    enc_y2 = torch.max(py2, ty2)

    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # Center distance
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # Aspect ratio
    pw, ph = px2 - px1, py2 - py1
    tw, th = tx2 - tx1, ty2 - ty1
    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / (th + 1e-7)) - torch.atan(pw / (ph + 1e-7))) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    return 1 - iou + rho2 / c2 + alpha * v


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=5, obj_weight=1.0, cls_weight=0.5, reg_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, targets, strides=[8, 16]):
        """
        predictions: list of [B, 5+C, H, W] tensors per scale
        targets: list of [num_objects, 6] tensors per image (class, cx, cy, w, h in pixels, batch_idx)
        """
        device = predictions[0].device
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_positive = 0

        batch_size = predictions[0].shape[0]

        for scale_idx, pred in enumerate(predictions):
            b, c, h, w = pred.shape
            stride = strides[scale_idx]

            pred = pred.permute(0, 2, 3, 1).reshape(b, h * w, c)
            reg_pred = pred[..., :4]
            obj_pred = pred[..., 4]
            cls_pred = pred[..., 5:]

            # Generate grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

            # Decode predictions to pixel space
            pred_xy = (reg_pred[..., :2].sigmoid() + grid.unsqueeze(0)) * stride
            pred_wh = reg_pred[..., 2:4].exp() * stride
            pred_x1y1 = pred_xy - pred_wh / 2
            pred_x2y2 = pred_xy + pred_wh / 2
            pred_boxes = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            # Build targets per image
            obj_target = torch.zeros(b, h * w, device=device)
            cls_target = torch.zeros(b, h * w, self.num_classes, device=device)
            reg_target = torch.zeros(b, h * w, 4, device=device)
            fg_mask = torch.zeros(b, h * w, dtype=torch.bool, device=device)

            for img_idx in range(batch_size):
                img_targets = targets[targets[:, 0] == img_idx]
                if len(img_targets) == 0:
                    continue

                gt_cls = img_targets[:, 1].long()
                gt_cx = img_targets[:, 2]
                gt_cy = img_targets[:, 3]
                gt_w = img_targets[:, 4]
                gt_h_val = img_targets[:, 5]

                gt_x1 = gt_cx - gt_w / 2
                gt_y1 = gt_cy - gt_h_val / 2
                gt_x2 = gt_cx + gt_w / 2
                gt_y2 = gt_cy + gt_h_val / 2

                # Assign to grid cells (center-based)
                gi = (gt_cx / stride).long().clamp(0, w - 1)
                gj = (gt_cy / stride).long().clamp(0, h - 1)
                indices = gj * w + gi

                for t_idx in range(len(img_targets)):
                    idx = indices[t_idx].item()
                    fg_mask[img_idx, idx] = True
                    obj_target[img_idx, idx] = 1.0
                    cls_target[img_idx, idx, gt_cls[t_idx]] = 1.0
                    reg_target[img_idx, idx] = torch.tensor(
                        [gt_x1[t_idx], gt_y1[t_idx], gt_x2[t_idx], gt_y2[t_idx]], device=device)

            # Objectness loss (all cells)
            total_obj_loss += self.bce(obj_pred, obj_target).mean()

            # Classification and regression loss (positive cells only)
            n_pos = fg_mask.sum().item()
            if n_pos > 0:
                total_positive += n_pos
                pos_cls_pred = cls_pred[fg_mask]
                pos_cls_target = cls_target[fg_mask]
                total_cls_loss += self.bce(pos_cls_pred, pos_cls_target).mean()

                pos_boxes = pred_boxes[fg_mask]
                pos_reg_target = reg_target[fg_mask]
                total_reg_loss += bbox_ciou(pos_boxes, pos_reg_target).mean()

        loss = (self.obj_weight * total_obj_loss +
                self.cls_weight * total_cls_loss +
                self.reg_weight * total_reg_loss)

        return loss, {
            "obj_loss": total_obj_loss.item(),
            "cls_loss": total_cls_loss.item(),
            "reg_loss": total_reg_loss.item(),
            "total_loss": loss.item(),
            "num_positive": total_positive,
        }
