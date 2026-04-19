"""Generate a synthetic test video with moving drone-like objects on a sky background."""

import argparse
import math
import random

import cv2
import numpy as np


def draw_drone(frame, cx, cy, size, angle):
    """Draw a simple drone shape (X with circle)."""
    s = int(size)
    color = (40, 40, 40)

    # Body (dark circle)
    cv2.circle(frame, (int(cx), int(cy)), s // 3, color, -1)

    # Arms (X shape)
    for a in [angle, angle + 90, angle + 180, angle + 270]:
        rad = math.radians(a)
        ex = int(cx + s * math.cos(rad))
        ey = int(cy + s * math.sin(rad))
        cv2.line(frame, (int(cx), int(cy)), (ex, ey), color, 2)
        # Propeller circles at arm tips
        cv2.circle(frame, (ex, ey), s // 4, color, 1)


class SimDrone:
    def __init__(self, w, h, drone_id):
        self.id = drone_id
        self.x = random.uniform(100, w - 100)
        self.y = random.uniform(80, h - 80)
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-2, 2)
        self.size = random.uniform(15, 40)
        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-2, 2)
        self.w = w
        self.h = h

    def update(self):
        # Slight random acceleration
        self.vx += random.gauss(0, 0.1)
        self.vy += random.gauss(0, 0.05)
        self.vx = max(-5, min(5, self.vx))
        self.vy = max(-3, min(3, self.vy))

        self.x += self.vx
        self.y += self.vy
        self.angle += self.spin

        # Bounce off edges
        if self.x < 50 or self.x > self.w - 50:
            self.vx *= -1
            self.x = max(50, min(self.w - 50, self.x))
        if self.y < 50 or self.y > self.h - 50:
            self.vy *= -1
            self.y = max(50, min(self.h - 50, self.y))

    def bbox(self):
        s = self.size * 1.5
        return (self.x - s, self.y - s, self.x + s, self.y + s)

    def draw(self, frame):
        draw_drone(frame, self.x, self.y, self.size, self.angle)


def generate_sky_background(w, h):
    """Generate a gradient sky background."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        r = int(135 + 80 * ratio)
        g = int(180 + 50 * ratio)
        b = int(235 - 20 * ratio)
        bg[y, :] = (b, g, r)
    return bg


def add_clouds(frame, t):
    """Add simple cloud shapes that drift."""
    overlay = frame.copy()
    for i in range(5):
        cx = int((200 + i * 250 + t * (0.3 + i * 0.1)) % (frame.shape[1] + 200) - 100)
        cy = int(80 + i * 60 + 20 * math.sin(t * 0.01 + i))
        cv2.ellipse(overlay, (cx, cy), (80 + i * 20, 30 + i * 5), 0, 0, 360, (230, 235, 245), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drone test video")
    parser.add_argument("--output", type=str, default="data/test_video.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--save-labels", action="store_true", help="Save YOLO labels for training")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_frames = args.fps * args.duration
    sky = generate_sky_background(args.width, args.height)

    drones = [SimDrone(args.width, args.height, i) for i in range(args.num_drones)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))

    if args.save_labels:
        img_dir = "data/images/train"
        lbl_dir = "data/labels/train"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    print(f"Generating {args.duration}s video: {args.width}x{args.height} @ {args.fps}fps, {args.num_drones} drones")

    for frame_idx in range(total_frames):
        frame = sky.copy()
        add_clouds(frame, frame_idx)

        labels = []
        for drone in drones:
            drone.update()
            drone.draw(frame)

            x1, y1, x2, y2 = drone.bbox()
            cx = (x1 + x2) / 2 / args.width
            cy = (y1 + y2) / 2 / args.height
            w = (x2 - x1) / args.width
            h = (y2 - y1) / args.height

            cls_id = 1 if drone.size > 25 else 0
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        writer.write(frame)

        if args.save_labels:
            fname = f"frame_{frame_idx:06d}"
            cv2.imwrite(f"{img_dir}/{fname}.jpg", frame)
            with open(f"{lbl_dir}/{fname}.txt", "w") as f:
                f.write("\n".join(labels) + "\n")

        if (frame_idx + 1) % args.fps == 0:
            sec = (frame_idx + 1) // args.fps
            print(f"  {sec}/{args.duration}s", end="\r")

    writer.release()
    print(f"\nVideo saved: {args.output} ({total_frames} frames)")

    if args.save_labels:
        print(f"Labels saved: {total_frames} frames in data/images/train + data/labels/train")


if __name__ == "__main__":
    main()
