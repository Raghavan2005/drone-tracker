"""End-to-end pipeline test: load model, run detection + tracking on video, save annotated output."""

import sys
import time

sys.path.insert(0, ".")

import cv2
import numpy as np
import torch

from training.models.drone_net import DroneNetPico


class SimpleTracker:
    """Minimal IoU tracker for testing (mirrors C++ ByteTrack logic)."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    "bbox": det[:4],
                    "conf": det[4],
                    "cls": int(det[5]),
                    "age": 0,
                    "hits": 1,
                }
                self.next_id += 1
            return self.tracks

        # Simple IoU matching
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        for i, tid in enumerate(track_ids):
            best_iou = 0.3
            best_j = -1
            for j, det in enumerate(detections):
                if j in matched_dets:
                    continue
                iou = self._iou(self.tracks[tid]["bbox"], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                self.tracks[tid]["bbox"] = detections[best_j][:4]
                self.tracks[tid]["conf"] = detections[best_j][4]
                self.tracks[tid]["cls"] = int(detections[best_j][5])
                self.tracks[tid]["age"] = 0
                self.tracks[tid]["hits"] += 1
                matched_tracks.add(tid)
                matched_dets.add(best_j)

        # Age unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > 15:
                    del self.tracks[tid]

        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self.next_id] = {
                    "bbox": det[:4],
                    "conf": det[4],
                    "cls": int(det[5]),
                    "age": 0,
                    "hits": 1,
                }
                self.next_id += 1

        return self.tracks

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0


def letterbox(img, size=416):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, px, py


def nms(detections, iou_thresh=0.45):
    if len(detections) == 0:
        return []
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    for d in dets:
        suppress = False
        for k in keep:
            if d[5] == k[5]:
                iou = SimpleTracker._iou(d[:4], k[:4])
                if iou > iou_thresh:
                    suppress = True
                    break
        if not suppress:
            keep.append(d)
    return keep


def detect(model, frame, device, conf_thresh=0.25, input_size=416):
    img, scale, px, py = letterbox(frame, input_size)
    blob = img.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    tensor = torch.from_numpy(blob).to(device)

    with torch.no_grad():
        output = model(tensor)  # [1, 3380, 10]

    preds = output[0].cpu().numpy()
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    obj = preds[:, 4]
    cls_scores = preds[:, 5:]
    cls_ids = cls_scores.argmax(axis=1)
    cls_confs = cls_scores.max(axis=1)
    scores = obj * cls_confs

    mask = scores > conf_thresh
    detections = []
    for i in np.where(mask)[0]:
        x1 = (cx[i] - w[i] / 2 - px) / scale
        y1 = (cy[i] - h[i] / 2 - py) / scale
        x2 = (cx[i] + w[i] / 2 - px) / scale
        y2 = (cy[i] + h[i] / 2 - py) / scale
        detections.append([x1, y1, x2, y2, scores[i], cls_ids[i]])

    return nms(detections)


CLASS_NAMES = ["drone_s", "drone_m", "drone_l", "bird", "aircraft"]
COLORS = [(0, 255, 0), (0, 200, 255), (0, 100, 255), (200, 200, 0), (200, 0, 200)]


def draw_results(frame, tracks, fps):
    # Status bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f" [TEST] FPS: {fps:.0f} | Tracks: {len(tracks)}",
        (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )

    for tid, t in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cls = t["cls"]
        conf = t["conf"]
        color = COLORS[cls % len(COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"D-{tid:02d} {CLASS_NAMES[cls]} {conf:.0%}"
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(frame, (x1, y1 - ts[1] - 6), (x1 + ts[0] + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/test_video.mp4")
    parser.add_argument("--model", default="runs/test/best.pt")
    parser.add_argument("--output", default="data/test_output.mp4")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--show", action="store_true", help="Display live window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DroneNetPico(num_classes=5, input_size=416).to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded")

    # GPU warmup
    if device.type == "cuda":
        dummy = torch.randn(1, 3, 416, 416, device=device)
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
            torch.cuda.synchronize()
        print("GPU warmup done")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open: {args.video}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracker = SimpleTracker()

    frame_count = 0
    total_det_time = 0
    total_detections = 0

    print(f"Running pipeline (conf={args.conf})...")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections = detect(model, frame, device, conf_thresh=args.conf)
        det_ms = (time.perf_counter() - t0) * 1000

        tracks = tracker.update(detections)
        total_det_time += det_ms
        total_detections += len(detections)
        frame_count += 1

        measured_fps = 1000.0 / det_ms if det_ms > 0 else 0
        annotated = draw_results(frame.copy(), tracks, measured_fps)
        writer.write(annotated)

        if args.show:
            cv2.imshow("Drone Tracker Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 30 == 0:
            avg_ms = total_det_time / frame_count
            print(
                f"  Frame {frame_count}/{total} | "
                f"Det: {det_ms:.1f}ms ({1000/avg_ms:.0f} avg FPS) | "
                f"Dets: {len(detections)} | Tracks: {len(tracks)}"
            )

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_ms = total_det_time / max(frame_count, 1)
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    print()
    print("=" * 60)
    print(f"  TEST COMPLETE")
    print(f"  Frames processed:   {frame_count}")
    print(f"  Total detections:   {total_detections}")
    print(f"  Avg detection time: {avg_ms:.1f} ms")
    print(f"  Avg FPS:            {avg_fps:.0f}")
    print(f"  Output saved:       {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
