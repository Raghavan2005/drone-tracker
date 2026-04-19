"""Generate a presentation-style PDF for the Drone Tracker project."""

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import ParagraphStyle

W, H = landscape(A4)

# Colors
BG_DARK = HexColor("#0f172a")
BG_CARD = HexColor("#1e293b")
ACCENT = HexColor("#38bdf8")
ACCENT2 = HexColor("#22d3ee")
GREEN = HexColor("#4ade80")
ORANGE = HexColor("#fb923c")
RED = HexColor("#f87171")
YELLOW = HexColor("#facc15")
TEXT_PRIMARY = HexColor("#f1f5f9")
TEXT_SECONDARY = HexColor("#94a3b8")
TEXT_MUTED = HexColor("#64748b")
WHITE = white
PURPLE = HexColor("#a78bfa")


def draw_bg(c):
    c.setFillColor(BG_DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)


def draw_header_bar(c):
    c.setFillColor(HexColor("#0c1222"))
    c.rect(0, H - 45, W, 45, fill=1, stroke=0)
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(0, H - 45, W, H - 45)


def draw_footer(c, page_num, total):
    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(40, 20, "Raghavan - 230701520 | Rajalakshmi Engineering College")
    c.drawRightString(W - 40, 20, f"{page_num} / {total}")


def draw_slide_title(c, title, subtitle=None):
    draw_header_bar(c)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - 33, title)
    if subtitle:
        c.setFillColor(TEXT_MUTED)
        c.setFont("Helvetica", 11)
        c.drawRightString(W - 40, H - 33, subtitle)


def draw_card(c, x, y, w, h, fill=BG_CARD):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)


def draw_bullet(c, x, y, text, font_size=11, color=TEXT_PRIMARY, bullet_color=ACCENT):
    c.setFillColor(bullet_color)
    c.circle(x + 4, y + 4, 3, fill=1, stroke=0)
    c.setFillColor(color)
    c.setFont("Helvetica", font_size)
    c.drawString(x + 16, y, text)


def draw_code_block(c, x, y, w, h, lines):
    c.setFillColor(HexColor("#0d1117"))
    c.roundRect(x, y, w, h, 6, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#30363d"))
    c.setLineWidth(0.5)
    c.roundRect(x, y, w, h, 6, fill=0, stroke=1)
    c.setFont("Courier", 9)
    ly = y + h - 16
    for line in lines:
        if ly < y + 6:
            break
        if line.startswith("//") or line.startswith("#"):
            c.setFillColor(TEXT_MUTED)
        elif "──►" in line or "──>" in line or "│" in line or "├" in line or "└" in line:
            c.setFillColor(ACCENT)
        else:
            c.setFillColor(HexColor("#c9d1d9"))
        c.drawString(x + 10, ly, line)
        ly -= 13


def draw_metric_box(c, x, y, value, label, color=ACCENT):
    draw_card(c, x, y, 130, 60, HexColor("#162032"))
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(x + 65, y + 30, str(value))
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 9)
    c.drawCentredString(x + 65, y + 12, label)


TOTAL_SLIDES = 15


def slide_title(c):
    draw_bg(c)

    # Large centered title
    c.setFillColor(HexColor("#0c1222"))
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Decorative gradient line
    for i in range(int(W)):
        r = 0.22 + 0.5 * (i / W)
        g = 0.74 + 0.1 * (i / W)
        bl = 0.97 - 0.1 * (i / W)
        c.setStrokeColor(Color(r, g, bl))
        c.setLineWidth(3)
        c.line(i, H * 0.52, i + 1, H * 0.52)

    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(W / 2, H * 0.62, "Vision-Based Autonomous Drone")
    c.drawCentredString(W / 2, H * 0.55, "Tracking & Targeting System")

    c.setFillColor(ACCENT)
    c.setFont("Helvetica", 14)
    c.drawCentredString(W / 2, H * 0.44, "Real-time Detection  |  Multi-Object Tracking  |  Targeting Output")

    # Info block
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 12)
    y = H * 0.32
    c.drawCentredString(W / 2, y, "Raghavan  —  230701520")
    c.drawCentredString(W / 2, y - 22, "Mentor: Prof. Subha")
    c.drawCentredString(W / 2, y - 44, "Image Processing and Computer Vision")
    c.setFillColor(TEXT_MUTED)
    c.setFont("Helvetica", 11)
    c.drawCentredString(W / 2, y - 72, "Rajalakshmi Engineering College")

    draw_footer(c, 1, TOTAL_SLIDES)


def slide_agenda(c):
    draw_bg(c)
    draw_slide_title(c, "Agenda")
    draw_footer(c, 2, TOTAL_SLIDES)

    items = [
        ("01", "Problem Statement & Motivation"),
        ("02", "System Overview & Architecture"),
        ("03", "DroneNet-Pico — Custom CNN Design"),
        ("04", "Detection Pipeline"),
        ("05", "ByteTrack — Multi-Object Tracking"),
        ("06", "Trajectory Prediction"),
        ("07", "Targeting System"),
        ("08", "Operator HUD & Interface"),
        ("09", "Training Pipeline"),
        ("10", "Performance Results"),
        ("11", "Testing & Validation"),
        ("12", "Tech Stack & Dependencies"),
        ("13", "Future Work & Conclusion"),
    ]

    x_left = 80
    x_right = W / 2 + 20
    y_start = H - 100
    spacing = 36

    for i, (num, text) in enumerate(items):
        col = x_left if i < 7 else x_right
        row = i if i < 7 else i - 7
        y = y_start - row * spacing

        c.setFillColor(ACCENT)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(col, y, num)
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica", 12)
        c.drawString(col + 35, y, text)


def slide_problem(c):
    draw_bg(c)
    draw_slide_title(c, "Problem Statement & Motivation")
    draw_footer(c, 3, TOTAL_SLIDES)

    # Problem
    draw_card(c, 40, H - 280, W / 2 - 60, 210)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, H - 100, "The Problem")
    c.setFillColor(HexColor("#fca5a5"))
    c.setFont("Helvetica", 10)
    problems = [
        "Unauthorized drones pose security threats to critical infrastructure",
        "Manual detection is slow and unreliable",
        "Existing systems are expensive and proprietary",
        "Consumer drones are small, fast, and hard to track",
        "Need real-time response (< 10ms latency)",
    ]
    y = H - 130
    for p in problems:
        draw_bullet(c, 60, y, p, 10, HexColor("#e2e8f0"), RED)
        y -= 24

    # Solution
    draw_card(c, W / 2 + 10, H - 280, W / 2 - 60, 210)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(W / 2 + 30, H - 100, "Our Solution")
    c.setFillColor(HexColor("#bbf7d0"))
    c.setFont("Helvetica", 10)
    solutions = [
        "Custom lightweight CNN — 1.3M params, 227 KB model",
        "100+ FPS real-time processing on consumer GPU",
        "Open-source, modular C++ pipeline",
        "Multi-object tracking with persistent IDs",
        "Automated targeting with gimbal/servo output",
    ]
    y = H - 130
    for s in solutions:
        draw_bullet(c, W / 2 + 30, y, s, 10, HexColor("#e2e8f0"), GREEN)
        y -= 24

    # Objectives
    draw_card(c, 40, 50, W - 80, 130)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, 150, "Project Objectives")

    objectives = [
        "Detect and classify drones in real-time from ground-based cameras",
        "Track multiple drones simultaneously with persistent identity across frames",
        "Predict drone trajectories for proactive targeting and interception",
        "Output targeting data (pan/tilt angles) for gimbal, servo, or operator display",
    ]
    y = 125
    for o in objectives:
        draw_bullet(c, 60, y, o, 10, TEXT_PRIMARY, ACCENT2)
        y -= 22


def slide_architecture(c):
    draw_bg(c)
    draw_slide_title(c, "System Architecture", "4-Thread Pipelined Design")
    draw_footer(c, 4, TOTAL_SLIDES)

    code = [
        "┌──────────┐   buf0   ┌──────────┐   buf1   ┌──────────┐   buf2   ┌──────────┐",
        "│ CAPTURE  │────────►│  DETECT  │────────►│  TRACK   │────────►│ TARGET  │",
        "│ Thread 0 │         │ Thread 1 │         │ Thread 2 │         │  + UI   │",
        "└──────────┘         └──────────┘         └──────────┘         │ Thread 3│",
        "     │                     │                     │              └──────────┘",
        "     │                     │                     │                   │",
        "┌────▼─────┐         ┌────▼─────┐         ┌────▼─────┐       ┌────▼─────┐",
        "│USB/RTSP/ │         │DroneNet  │         │ByteTrack │       │Gimbal/   │",
        "│  File    │         │  Pico    │         │ Kalman   │       │ Servo/   │",
        "│ OpenCV   │         │TensorRT  │         │ LAPJV    │       │ Screen   │",
        "└──────────┘         └──────────┘         └──────────┘       └──────────┘",
    ]
    draw_code_block(c, 40, H - 300, W - 80, 210, code)

    # Key design points
    draw_card(c, 40, 50, W / 2 - 60, 150)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, 170, "Key Design Points")

    points = [
        "Lock-free SPSC ring buffers between stages",
        "Drop-oldest policy — never block capture thread",
        "Each stage runs on dedicated CPU thread",
        "GPU used only for CNN inference (Thread 1)",
        "Modular: swap any stage independently",
    ]
    y = 145
    for p in points:
        draw_bullet(c, 60, y, p, 10, TEXT_PRIMARY, ACCENT)
        y -= 22

    draw_card(c, W / 2 + 10, 50, W / 2 - 60, 150)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 30, 170, "Input Sources")

    inputs = [
        "USB / CSI Camera (V4L2, up to 120fps)",
        "RTSP / IP Camera Stream (FFmpeg)",
        "Video Files (MP4, AVI, looping)",
        "Config-driven source selection via YAML",
    ]
    y = 145
    for inp in inputs:
        draw_bullet(c, W / 2 + 30, y, inp, 10, TEXT_PRIMARY, GREEN)
        y -= 22


def slide_cnn(c):
    draw_bg(c)
    draw_slide_title(c, "DroneNet-Pico — Custom CNN Architecture", "~1.3M Parameters")
    draw_footer(c, 5, TOTAL_SLIDES)

    code = [
        "Input: 416 x 416 x 3",
        "│",
        "├── BACKBONE",
        "│   ConvBnSiLU(3→16, s=2)     → 208x208x16",
        "│   ConvBnSiLU(16→32, s=2)    → 104x104x32  + MicroBlock x1",
        "│   ConvBnSiLU(32→64, s=2)    → 52x52x64    + MicroBlock x2  ← P3",
        "│   ConvBnSiLU(64→128, s=2)   → 26x26x128   + MicroBlock x2  ← P4",
        "│   ConvBnSiLU(128→256, s=2)  → 13x13x256   + MicroBlock x1  ← P5",
        "│",
        "├── NECK (PAN-Lite, 2 scales)",
        "│   Upsample P5 + P4 → N4 (26x26x128)",
        "│   Upsample N4 + P3 → N3 (52x52x64)",
        "│   Downsample N3 + N4 → D4 (26x26x128)",
        "│",
        "└── HEAD (anchor-free, decoupled)",
        "    Scale 1 (52x52): 2704 candidates",
        "    Scale 2 (26x26):  676 candidates",
        "    Total: 3380 detections × [cx,cy,w,h,obj,5 classes]",
    ]
    draw_code_block(c, 40, H - 340, W / 2 - 30, 260, code)

    # MicroBlock
    draw_card(c, W / 2 + 20, H - 170, W / 2 - 70, 90)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(W / 2 + 40, H - 100, "MicroBlock (Depthwise-Separable + Residual)")
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Courier", 9)
    c.drawString(W / 2 + 40, H - 120, "in → DWConv3x3 → BN → SiLU → PWConv1x1")
    c.drawString(W / 2 + 40, H - 135, "   → BN → SiLU → (+input) → out")

    # Classes
    draw_card(c, W / 2 + 20, H - 340, W / 2 - 70, 150)
    c.setFillColor(YELLOW)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(W / 2 + 40, H - 210, "Detection Classes (5)")

    classes = [
        ("0", "drone_small", "Micro/toy drones (< 250g)"),
        ("1", "drone_medium", "Consumer (DJI Mini/Air)"),
        ("2", "drone_large", "Professional/industrial"),
        ("3", "bird", "False positive rejection"),
        ("4", "aircraft", "False positive rejection"),
    ]
    y = H - 235
    for cid, name, desc in classes:
        c.setFillColor(ACCENT)
        c.setFont("Courier-Bold", 10)
        c.drawString(W / 2 + 40, y, cid)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(W / 2 + 60, y, name)
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 9)
        c.drawString(W / 2 + 170, y, desc)
        y -= 18

    # Metrics
    draw_metric_box(c, 40, 50, "1.3M", "Parameters", ACCENT)
    draw_metric_box(c, 185, 50, "227 KB", "ONNX Size", GREEN)
    draw_metric_box(c, 330, 50, "416²", "Input Size", PURPLE)
    draw_metric_box(c, 475, 50, "3,380", "Candidates", ORANGE)
    draw_metric_box(c, 620, 50, "5", "Classes", YELLOW)


def slide_detection(c):
    draw_bg(c)
    draw_slide_title(c, "Detection Pipeline", "Preprocessing → Inference → NMS")
    draw_footer(c, 6, TOTAL_SLIDES)

    code = [
        "Frame (1280x720 BGR)",
        "  │",
        "  ├──► Letterbox Resize to 416x416 (preserve aspect ratio)",
        "  │    Gray padding (114,114,114) on borders",
        "  │",
        "  ├──► Normalize to [0,1] float32",
        "  │    HWC → CHW → NCHW tensor",
        "  │",
        "  ├──► CNN Forward Pass",
        "  │    TensorRT FP16 (primary) / ONNX Runtime (fallback)",
        "  │    Output: [1, 3380, 10]",
        "  │",
        "  ├──► Decode: sigmoid(xy) + grid → pixel coords",
        "  │    exp(wh) × stride → pixel size",
        "  │    sigmoid(obj) × sigmoid(cls) → confidence",
        "  │",
        "  ├──► Confidence Filter (threshold = 0.25)",
        "  │",
        "  ├──► Scale back to original image coordinates",
        "  │    Undo letterbox padding + scaling",
        "  │",
        "  └──► Per-class NMS (IoU threshold = 0.45)",
        "       Output: N detections [x1,y1,x2,y2,conf,class]",
    ]
    draw_code_block(c, 40, 50, W / 2 - 30, 410, code)

    # Backends
    draw_card(c, W / 2 + 20, H - 250, W / 2 - 70, 170)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 40, H - 100, "Inference Backends")

    backends = [
        ("TensorRT FP16", "Primary — NVIDIA GPUs, engine cached to disk"),
        ("ONNX Runtime", "Fallback — CPU, CUDA EP, or TensorRT EP"),
        ("Visual Detector", "Contrast-based — no model needed, uses CV"),
    ]
    y = H - 130
    for name, desc in backends:
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(W / 2 + 40, y, name)
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 9)
        c.drawString(W / 2 + 40, y - 15, desc)
        y -= 40

    # NMS detail
    draw_card(c, W / 2 + 20, 50, W / 2 - 70, 160)
    c.setFillColor(ORANGE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 40, 180, "Non-Maximum Suppression")

    nms_points = [
        "Sort detections by confidence (descending)",
        "For each detection, suppress overlapping same-class",
        "IoU > 0.45 → suppress lower-confidence box",
        "Keeps only best detection per object",
        "Runs per-class to avoid cross-class suppression",
    ]
    y = 155
    for p in nms_points:
        draw_bullet(c, W / 2 + 40, y, p, 9, TEXT_PRIMARY, ORANGE)
        y -= 20


def slide_tracking(c):
    draw_bg(c)
    draw_slide_title(c, "ByteTrack — Multi-Object Tracking", "Two-Stage Association")
    draw_footer(c, 7, TOTAL_SLIDES)

    code = [
        "Detections from frame N",
        "    │",
        "    ├── Split by confidence",
        "    │   ├── High (conf ≥ 0.5)",
        "    │   └── Low  (0.1 ≤ conf < 0.5)",
        "    │",
        "    ├── 1st Association: High-conf vs Tracked stracks",
        "    │   └── IoU cost matrix → Hungarian/LAPJV → match/unmatch",
        "    │",
        "    ├── 2nd Association: Low-conf vs Unmatched tracks",
        "    │   └── Recovers tracks with temporary low confidence",
        "    │",
        "    ├── 3rd Association: Unmatched dets vs Lost stracks",
        "    │   └── Re-acquires previously lost tracks",
        "    │",
        "    ├── Create new tracks for remaining detections",
        "    │",
        "    └── Lifecycle: NEW → TRACKED → LOST → REMOVED",
        "        (confirmed after min_hits=3 consecutive frames)",
    ]
    draw_code_block(c, 40, H - 350, W / 2 - 30, 270, code)

    # Kalman Filter
    draw_card(c, W / 2 + 20, H - 220, W / 2 - 70, 140)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 40, H - 100, "Kalman Filter (8-state)")
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Courier", 10)
    c.drawString(W / 2 + 40, H - 125, "State: [cx, cy, ar, h, vx, vy, va, vh]")
    c.setFont("Helvetica", 10)
    y = H - 150
    kf_points = [
        "Predict: propagate state + velocity",
        "Update: correct with matched detection",
        "Provides velocity estimates for free",
    ]
    for p in kf_points:
        draw_bullet(c, W / 2 + 40, y, p, 10, TEXT_PRIMARY, ACCENT)
        y -= 20

    # Why ByteTrack
    draw_card(c, W / 2 + 20, 50, W / 2 - 70, 190)
    c.setFillColor(YELLOW)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 40, 210, "Why ByteTrack?")

    reasons = [
        "No Re-ID network needed — sub-millisecond",
        "Drones in sky = well-separated, uniform appearance",
        "IoU-only matching is optimal for this domain",
        "Two-stage handles low-confidence detections",
        "LAPJV assignment faster than Hungarian for N<100",
        "SOTA on MOT17/MOT20 benchmarks",
    ]
    y = 185
    for r in reasons:
        draw_bullet(c, W / 2 + 40, y, r, 10, TEXT_PRIMARY, YELLOW)
        y -= 22


def slide_prediction(c):
    draw_bg(c)
    draw_slide_title(c, "Trajectory Prediction", "Kalman Extrapolation + Polynomial Fitting")
    draw_footer(c, 8, TOTAL_SLIDES)

    # Tier 1
    draw_card(c, 40, H - 280, W / 2 - 60, 200)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, H - 100, "Tier 1: Kalman Extrapolation")
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Courier", 10)
    c.drawString(60, H - 130, "predicted(t+k) = position + k × velocity")
    c.setFont("Helvetica", 10)
    points = [
        "Always active — zero extra cost (from tracker)",
        "Horizon: 1-5 frames (~50ms at 100 FPS)",
        "Handles constant-velocity motion perfectly",
    ]
    y = H - 160
    for p in points:
        draw_bullet(c, 60, y, p, 10, TEXT_PRIMARY, GREEN)
        y -= 22

    # Tier 2
    draw_card(c, W / 2 + 10, H - 280, W / 2 - 60, 200)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(W / 2 + 30, H - 100, "Tier 2: Polynomial Fitting")
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Courier", 10)
    c.drawString(W / 2 + 30, H - 130, "x(t) = at² + bt + c  (WLS fit)")
    c.setFont("Helvetica", 10)
    points = [
        "Uses last 30 positions (weighted least-squares)",
        "Recent points weighted more heavily",
        "Horizon: 0.5-2 seconds lookahead",
        "Handles acceleration / curved trajectories",
        "Cost: ~10 μs per track",
    ]
    y = H - 160
    for p in points:
        draw_bullet(c, W / 2 + 30, y, p, 10, TEXT_PRIMARY, PURPLE)
        y -= 22

    # Why not neural
    draw_card(c, 40, 50, W - 80, 120)
    c.setFillColor(ORANGE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, 140, "Why Not a Neural Predictor (LSTM/Transformer)?")
    points = [
        "Drone trajectories follow simple physics — mostly straight lines with gentle curves",
        "Polynomial model is deterministic, interpretable, and takes microseconds",
        "Neural predictor would add 2-5ms latency per frame with marginal accuracy improvement",
        "Quadratic term captures constant-acceleration maneuvers; degrades to linear for steady flight",
    ]
    y = 115
    for p in points:
        draw_bullet(c, 60, y, p, 10, TEXT_PRIMARY, ORANGE)
        y -= 20


def slide_targeting(c):
    draw_bg(c)
    draw_slide_title(c, "Targeting System", "Pixel → Angle → Servo/Gimbal Output")
    draw_footer(c, 9, TOTAL_SLIDES)

    code = [
        "Track Position (px, py) in image",
        "    │",
        "    ├── Pixel → Angle (pinhole camera model)",
        "    │   pan  = atan2(px - cx, fx) × 180/π",
        "    │   tilt = atan2(cy - py, fy) × 180/π",
        "    │",
        "    ├── Target Selection",
        "    │   ├── nearest_center — min(dx² + dy²)",
        "    │   ├── largest       — max(w × h)",
        "    │   ├── highest_conf  — max(confidence)",
        "    │   └── manual        — operator selects track ID",
        "    │",
        "    ├── EMA Smoothing (anti-jitter)",
        "    │   smooth(t) = 0.3 × raw(t) + 0.7 × smooth(t-1)",
        "    │   ~30ms effective lag at 100+ FPS",
        "    │",
        "    ├── Distance Estimation (pinhole model)",
        "    │   dist = (real_size × focal_length) / pixel_size",
        "    │",
        "    └── Output to:",
        "        ├── Gimbal (PELCO-D serial protocol)",
        "        ├── Servo (PWM via serial, 500-2500μs)",
        "        └── Screen (overlay coordinates for operator)",
    ]
    draw_code_block(c, 40, 50, W - 80, 410, code)


def slide_hud(c):
    draw_bg(c)
    draw_slide_title(c, "Operator HUD & Interface")
    draw_footer(c, 10, TOTAL_SLIDES)

    code = [
        "┌─────────────────────────────────────────────────────────┐",
        "│ [LIVE] FPS: 142 | Detect: 3.2ms | Track: 0.4ms        │",
        "│                                                         │",
        "│              ┌─────────┐                                │",
        "│              │ D-02    │  ← Green: tracked drone       │",
        "│              │drone_m  │    ID + class + confidence     │",
        "│              │ 92%  ↗  │    + velocity arrow            │",
        "│              └─────────┘                                │",
        "│                   ····→  ← Predicted trajectory         │",
        "│                                                         │",
        "│    ╔═══════════╗                                        │",
        "│    ║ ★ D-01   ║  ← Red: primary target                │",
        "│    ║ [LOCKED] ║    targeting reticle + crosshair       │",
        "│    ╚═══════════╝                                        │",
        "│                                                         │",
        "│ ┌──────────────────────────────────────┐                │",
        "│ │ Target: D-01 [LOCKED]                │                │",
        "│ │ Pan: +12.4°  Tilt: -3.1°  Dist: 85m │  ← Info panel │",
        "│ │ Class: drone_medium  Conf: 94%       │                │",
        "│ └──────────────────────────────────────┘                │",
        "└─────────────────────────────────────────────────────────┘",
    ]
    draw_code_block(c, 40, H - 360, W / 2 + 30, 280, code)

    # Controls
    draw_card(c, W / 2 + 80, H - 360, W / 2 - 130, 280)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 100, H - 100, "Keyboard Controls")

    controls = [
        ("Q / ESC", "Quit application"),
        ("T", "Cycle target selection mode"),
        ("1-9", "Manually select track by ID"),
        ("R", "Start/stop video recording"),
        ("D", "Toggle debug overlay"),
        ("F", "Toggle fullscreen"),
    ]
    y = H - 130
    for key, desc in controls:
        c.setFillColor(YELLOW)
        c.setFont("Courier-Bold", 10)
        c.drawString(W / 2 + 100, y, key)
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica", 10)
        c.drawString(W / 2 + 175, y, desc)
        y -= 26

    # HUD elements
    draw_card(c, 40, 50, W - 80, 80)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(60, 100, "HUD Elements:")
    c.setFillColor(TEXT_PRIMARY)
    c.setFont("Helvetica", 10)
    c.drawString(180, 100, "Bounding boxes with track IDs  •  Velocity arrows  •  Trajectory prediction lines")
    c.drawString(180, 80, "Targeting reticle with state (ACQUIRING/LOCKED/LOST)  •  Status bar  •  Info panel")


def slide_training(c):
    draw_bg(c)
    draw_slide_title(c, "Training Pipeline", "PyTorch + Mixed Precision + EMA")
    draw_footer(c, 11, TOTAL_SLIDES)

    code = [
        "Dataset (YOLO format: class cx cy w h, normalized)",
        "    │",
        "    ├── Augmentations:",
        "    │   Mosaic (4-image, p=0.8), HorizontalFlip (p=0.5)",
        "    │   RandomBrightnessContrast, HueSaturationValue",
        "    │   GaussNoise, MotionBlur, RandomFog",
        "    │",
        "    ├── DroneNet-Pico (train mode)",
        "    │",
        "    ├── Loss = 1.0×BCE(obj) + 0.5×BCE(cls) + 5.0×CIoU(reg)",
        "    │",
        "    ├── SGD (momentum=0.937, weight_decay=5e-4)",
        "    │   Cosine Annealing LR: 0.01 → 1e-5 over 300 epochs",
        "    │   3-epoch linear warmup",
        "    │   FP16 mixed precision (2x memory efficiency)",
        "    │   EMA decay=0.9999",
        "    │",
        "    └── Export: PyTorch → ONNX → TensorRT FP16 Engine",
    ]
    draw_code_block(c, 40, H - 330, W / 2 - 20, 250, code)

    # Dataset
    draw_card(c, W / 2 + 30, H - 200, W / 2 - 80, 120)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 50, H - 100, "Recommended Datasets")

    datasets = [
        "Drone-vs-Bird Detection Challenge",
        "Anti-UAV Dataset (CVPR)",
        "MAV-VID (Micro Aerial Vehicle)",
        "Custom recordings + synthetic generation",
    ]
    y = H - 125
    for d in datasets:
        draw_bullet(c, W / 2 + 50, y, d, 10, TEXT_PRIMARY, GREEN)
        y -= 20

    # Loss
    draw_card(c, W / 2 + 30, 50, W / 2 - 80, 200)
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 50, 220, "Loss Function Details")

    losses = [
        ("CIoU Loss", "Regression — complete IoU with distance, aspect ratio"),
        ("BCE + Focal", "Classification — handles class imbalance"),
        ("BCE", "Objectness — foreground vs background"),
        ("SimOTA", "Label assignment — optimal transport (from YOLOX)"),
    ]
    y = 195
    for name, desc in losses:
        c.setFillColor(ORANGE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(W / 2 + 50, y, name)
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 9)
        c.drawString(W / 2 + 50, y - 14, desc)
        y -= 36


def slide_performance(c):
    draw_bg(c)
    draw_slide_title(c, "Performance Results")
    draw_footer(c, 12, TOTAL_SLIDES)

    # Metric boxes
    draw_metric_box(c, 50, H - 160, "224", "Avg FPS (GPU)", GREEN)
    draw_metric_box(c, 200, H - 160, "531", "Peak FPS (GPU)", ACCENT)
    draw_metric_box(c, 350, H - 160, "101", "CPU FPS", ORANGE)
    draw_metric_box(c, 500, H - 160, "4.5ms", "Detect Latency", PURPLE)
    draw_metric_box(c, 650, H - 160, "14/14", "Tests Passed", YELLOW)

    # Latency table
    draw_card(c, 40, 50, W / 2 - 60, 260)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, 280, "Pipeline Latency Breakdown")

    rows = [
        ("Capture", "< 1 ms", "USB/RTSP frame grab"),
        ("Preprocess", "< 1 ms", "Letterbox + normalize"),
        ("Detection", "2-10 ms", "CNN forward pass"),
        ("NMS", "< 0.5 ms", "Per-class, IoU=0.45"),
        ("Tracking", "< 0.5 ms", "ByteTrack (no Re-ID)"),
        ("Prediction", "< 0.1 ms", "Polynomial fit"),
        ("Targeting", "< 0.1 ms", "Angle + smoothing"),
        ("UI Render", "< 2 ms", "OpenCV overlay"),
        ("TOTAL", "< 10 ms", "100+ FPS end-to-end"),
    ]
    y = 255
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(TEXT_MUTED)
    c.drawString(60, y, "STAGE")
    c.drawString(180, y, "LATENCY")
    c.drawString(270, y, "NOTES")
    y -= 5
    c.setStrokeColor(TEXT_MUTED)
    c.setLineWidth(0.5)
    c.line(60, y, 360, y)
    y -= 15

    for stage, latency, notes in rows:
        if stage == "TOTAL":
            c.setFillColor(GREEN)
            c.setFont("Helvetica-Bold", 9)
        else:
            c.setFillColor(TEXT_PRIMARY)
            c.setFont("Helvetica", 9)
        c.drawString(60, y, stage)
        c.setFillColor(ACCENT if stage != "TOTAL" else GREEN)
        c.drawString(180, y, latency)
        c.setFillColor(TEXT_SECONDARY)
        c.drawString(270, y, notes)
        y -= 18

    # Model stats
    draw_card(c, W / 2 + 10, 50, W / 2 - 60, 260)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 30, 280, "Model Statistics")

    stats = [
        ("Parameters", "1,336,772"),
        ("ONNX Size", "227 KB"),
        ("Input Resolution", "416 × 416"),
        ("Output", "3,380 candidates"),
        ("Backbone", "5-stage ConvBnSiLU + MicroBlock"),
        ("Neck", "PAN-Lite (2 scales)"),
        ("Head", "Anchor-free, decoupled"),
        ("Precision", "FP16 (TensorRT)"),
        ("Training", "300 epochs, SGD, cosine LR"),
        ("Augmentation", "Mosaic + Albumentations"),
    ]
    y = 255
    for label, value in stats:
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 10)
        c.drawString(W / 2 + 30, y, label)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(W / 2 + 180, y, value)
        y -= 20


def slide_testing(c):
    draw_bg(c)
    draw_slide_title(c, "Testing & Validation")
    draw_footer(c, 13, TOTAL_SLIDES)

    # Unit tests
    draw_card(c, 40, H - 260, W / 2 - 60, 180)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, H - 100, "Unit Tests (Google Test) — 14/14 Passed")

    tests = [
        "RingBuffer: push/pop, overwrite, optional pop",
        "KalmanFilter: init, predict, update convergence, bbox conversion",
        "ByteTracker: single detection, persistence, multi-object, reset",
        "CoordinateTransform: center angle, off-center, distance",
        "TargetingEngine: nearest-center selection",
    ]
    y = H - 130
    for t in tests:
        draw_bullet(c, 60, y, t, 10, TEXT_PRIMARY, GREEN)
        y -= 22

    # Integration test
    draw_card(c, W / 2 + 10, H - 260, W / 2 - 60, 180)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 30, H - 100, "End-to-End Pipeline Test")

    e2e = [
        "Synthetic video: 3 drones, 15s, 450 frames",
        "Visual detector: 3/3 drones detected every frame",
        "Tracker: persistent IDs across full video",
        "Output: annotated MP4 with HUD overlay",
        "Performance: 208 FPS average throughput",
    ]
    y = H - 130
    for t in e2e:
        draw_bullet(c, W / 2 + 30, y, t, 10, TEXT_PRIMARY, ACCENT)
        y -= 22

    # Tools
    draw_card(c, 40, 50, W - 80, 130)
    c.setFillColor(YELLOW)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, 150, "Test Utilities")

    tools = [
        ("generate_test_video.py", "Creates synthetic drone footage with YOLO labels for training & testing"),
        ("test_pipeline.py", "Runs full detect → track pipeline on video, saves annotated output"),
        ("evaluate.py", "Computes mAP@0.5 on validation set with per-class breakdown"),
    ]
    y = 125
    for name, desc in tools:
        c.setFillColor(ACCENT)
        c.setFont("Courier-Bold", 10)
        c.drawString(60, y, name)
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica", 10)
        c.drawString(310, y, desc)
        y -= 22


def slide_techstack(c):
    draw_bg(c)
    draw_slide_title(c, "Tech Stack & Dependencies")
    draw_footer(c, 14, TOTAL_SLIDES)

    # C++
    draw_card(c, 40, H - 310, W / 2 - 60, 230)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, H - 100, "C++ Pipeline (Runtime)")

    cpp = [
        ("OpenCV 4.6+", "Image I/O, capture, drawing"),
        ("TensorRT 8.6+", "GPU inference (FP16)"),
        ("ONNX Runtime 1.16+", "CPU/CUDA fallback"),
        ("spdlog 1.12+", "Structured logging"),
        ("yaml-cpp 0.8+", "Configuration parsing"),
        ("Eigen3 3.4+", "Matrix math (Kalman filter)"),
        ("Google Test 1.14+", "Unit testing"),
    ]
    y = H - 125
    for lib, purpose in cpp:
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(60, y, lib)
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 9)
        c.drawString(210, y, purpose)
        y -= 22

    # Python
    draw_card(c, W / 2 + 10, H - 310, W / 2 - 60, 230)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(W / 2 + 30, H - 100, "Python Pipeline (Training)")

    py = [
        ("PyTorch 2.1+", "Model definition + training"),
        ("torchvision", "Image transforms"),
        ("ONNX", "Model export + validation"),
        ("Albumentations", "Data augmentation"),
        ("TensorBoard", "Training visualization"),
        ("NumPy / SciPy", "Numerical computation"),
        ("OpenCV", "Image processing"),
    ]
    y = H - 125
    for lib, purpose in py:
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(W / 2 + 30, y, lib)
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 9)
        c.drawString(W / 2 + 180, y, purpose)
        y -= 22

    # Build
    draw_card(c, 40, 50, W - 80, 100)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, 120, "Build System & Platform")

    build = [
        "CMake 3.20+ with conditional TensorRT/ONNX Runtime support",
        "C++17 standard, GCC 13+ or Clang 16+",
        "Linux (Ubuntu 22.04+) — tested on x86_64 with NVIDIA GPU",
    ]
    y = 95
    for b in build:
        draw_bullet(c, 60, y, b, 10, TEXT_PRIMARY, PURPLE)
        y -= 20


def slide_conclusion(c):
    draw_bg(c)
    draw_slide_title(c, "Conclusion & Future Work")
    draw_footer(c, 15, TOTAL_SLIDES)

    # Summary
    draw_card(c, 40, H - 240, W / 2 - 60, 160)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, H - 100, "What We Built")

    summary = [
        "Complete counter-drone pipeline: detect → track → target",
        "Custom CNN (DroneNet-Pico): 1.3M params, 227 KB",
        "224 FPS on consumer GPU, 101 FPS on CPU",
        "ByteTrack with persistent multi-object tracking",
        "Targeting output for gimbal, servo, and screen",
    ]
    y = H - 130
    for s in summary:
        draw_bullet(c, 60, y, s, 10, TEXT_PRIMARY, GREEN)
        y -= 22

    # Future
    draw_card(c, W / 2 + 10, H - 240, W / 2 - 60, 160)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(W / 2 + 30, H - 100, "Future Work")

    future = [
        "Train on real drone datasets for production accuracy",
        "Add infrared / thermal camera support",
        "Multi-camera fusion for 3D tracking",
        "Dear ImGui upgrade for richer operator UI",
        "Edge deployment on NVIDIA Jetson Orin",
    ]
    y = H - 130
    for f in future:
        draw_bullet(c, W / 2 + 30, y, f, 10, TEXT_PRIMARY, ACCENT)
        y -= 22

    # Thank you
    draw_card(c, 40, 50, W - 80, 150, HexColor("#162032"))
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(W / 2, 145, "Thank You")
    c.setFillColor(ACCENT)
    c.setFont("Helvetica", 14)
    c.drawCentredString(W / 2, 110, "Questions?")
    c.setFillColor(TEXT_SECONDARY)
    c.setFont("Helvetica", 11)
    c.drawCentredString(W / 2, 80, "Raghavan  •  230701520  •  Rajalakshmi Engineering College")
    c.drawCentredString(W / 2, 62, "github.com/Raghavan2005/drone-tracker")


def main():
    output = "docs/Drone_Tracker_Presentation.pdf"
    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)

    pdf = canvas.Canvas(output, pagesize=landscape(A4))
    pdf.setTitle("Vision-Based Autonomous Drone Tracking & Targeting System")
    pdf.setAuthor("Raghavan - 230701520")
    pdf.setSubject("Image Processing and Computer Vision")

    slides = [
        slide_title,
        slide_agenda,
        slide_problem,
        slide_architecture,
        slide_cnn,
        slide_detection,
        slide_tracking,
        slide_prediction,
        slide_targeting,
        slide_hud,
        slide_training,
        slide_performance,
        slide_testing,
        slide_techstack,
        slide_conclusion,
    ]

    for i, slide_fn in enumerate(slides):
        slide_fn(pdf)
        if i < len(slides) - 1:
            pdf.showPage()

    pdf.save()
    print(f"Presentation saved: {output}")
    print(f"  {len(slides)} slides")


if __name__ == "__main__":
    main()
