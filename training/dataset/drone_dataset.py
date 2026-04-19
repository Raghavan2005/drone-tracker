"""YOLO-format dataset loader for drone detection."""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DroneDataset(Dataset):
    def __init__(self, images_dir, labels_dir, input_size=416, transforms=None,
                 mosaic_prob=0.0, augment=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.input_size = input_size
        self.transforms = transforms
        self.mosaic_prob = mosaic_prob
        self.augment = augment

        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.augment and random.random() < self.mosaic_prob:
            return self._load_mosaic(idx)
        return self._load_single(idx)

    def _load_single(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        labels = self._load_labels(label_path, w0, h0)

        if self.transforms:
            bboxes = labels[:, 1:5].tolist() if len(labels) > 0 else []
            class_labels = labels[:, 0].astype(int).tolist() if len(labels) > 0 else []

            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]

            if bboxes:
                labels = np.array([[c, *b] for c, b in zip(class_labels, bboxes)], dtype=np.float32)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

        image, labels = self._letterbox(image, labels)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Convert labels from YOLO normalized to pixel coordinates
        targets = self._labels_to_targets(labels, self.input_size, self.input_size)

        return torch.from_numpy(image), targets

    def _load_mosaic(self, idx):
        """4-image mosaic augmentation."""
        indices = [idx] + random.choices(range(len(self)), k=3)
        s = self.input_size
        xc = random.randint(s // 4, 3 * s // 4)
        yc = random.randint(s // 4, 3 * s // 4)

        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        all_labels = []

        placements = [
            (0, 0, xc, yc),
            (xc, 0, s, yc),
            (0, yc, xc, s),
            (xc, yc, s, s),
        ]

        for i, (x1, y1, x2, y2) in enumerate(placements):
            img_path = self.image_files[indices[i]]
            label_path = self.labels_dir / (img_path.stem + ".txt")

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]
            labels = self._load_labels(label_path, w0, h0)

            pw, ph = x2 - x1, y2 - y1
            img_resized = cv2.resize(img, (pw, ph))
            mosaic_img[y1:y2, x1:x2] = img_resized

            if len(labels) > 0:
                # Scale labels from normalized to patch coordinates
                scaled = labels.copy()
                scaled[:, 1] = labels[:, 1] * pw + x1  # cx
                scaled[:, 2] = labels[:, 2] * ph + y1  # cy
                scaled[:, 3] = labels[:, 3] * pw        # w
                scaled[:, 4] = labels[:, 4] * ph        # h
                # Normalize back to mosaic size
                scaled[:, 1] /= s
                scaled[:, 2] /= s
                scaled[:, 3] /= s
                scaled[:, 4] /= s
                all_labels.append(scaled)

        if all_labels:
            labels = np.concatenate(all_labels, axis=0)
            # Clip to valid range
            labels[:, 1:5] = np.clip(labels[:, 1:5], 0, 1)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        image = mosaic_img.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        targets = self._labels_to_targets(labels, s, s)
        return torch.from_numpy(image), targets

    def _load_labels(self, label_path, img_w, img_h):
        """Load YOLO format labels: class cx cy w h (all normalized)."""
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = float(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    labels.append([cls, cx, cy, w, h])

        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _letterbox(self, image, labels):
        """Resize with letterbox to input_size."""
        h, w = image.shape[:2]
        s = self.input_size
        scale = min(s / w, s / h)

        nw, nh = int(w * scale), int(h * scale)
        pad_x = (s - nw) // 2
        pad_y = (s - nh) // 2

        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        if len(labels) > 0:
            # Labels are already normalized [0,1], adjust for letterbox
            labels[:, 1] = (labels[:, 1] * nw + pad_x) / s
            labels[:, 2] = (labels[:, 2] * nh + pad_y) / s
            labels[:, 3] = labels[:, 3] * nw / s
            labels[:, 4] = labels[:, 4] * nh / s

        return canvas, labels

    def _labels_to_targets(self, labels, img_w, img_h):
        """Convert normalized YOLO labels to pixel-space targets tensor.
        Returns: [N, 5] tensor with [class, cx_px, cy_px, w_px, h_px]
        """
        if len(labels) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        targets = torch.from_numpy(labels).float()
        targets[:, 1] *= img_w
        targets[:, 2] *= img_h
        targets[:, 3] *= img_w
        targets[:, 4] *= img_h
        return targets


def collate_fn(batch):
    """Custom collate: add batch index to targets."""
    images = []
    targets = []

    for i, (img, tgt) in enumerate(batch):
        images.append(img)
        if len(tgt) > 0:
            batch_idx = torch.full((len(tgt), 1), i, dtype=torch.float32)
            targets.append(torch.cat([batch_idx, tgt], dim=1))

    images = torch.stack(images)
    if targets:
        targets = torch.cat(targets, dim=0)  # [N_total, 6]: batch_idx, class, cx, cy, w, h
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, targets
