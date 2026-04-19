# Drone Tracker

Vision-based autonomous drone detection, tracking, and targeting system. Built for real-time counter-drone operations using a custom lightweight CNN and a multi-threaded C++ inference pipeline.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-14%2F14-brightgreen.svg)

---

## Architecture

```
                          ┌─────────────────────────────────────────────────────────┐
                          │              DRONE TRACKER PIPELINE                     │
                          │                                                         │
  ┌──────────┐   buf0    │  ┌──────────┐   buf1    ┌───────────┐   buf2    ┌─────┐│
  │  CAPTURE  │──────────►│  │  DETECT   │─────────►│   TRACK   │─────────►│ TGT ││
  │  Thread 0 │           │  │  Thread 1 │          │  Thread 2 │          │ + UI││
  └──────────┘           │  └──────────┘           └───────────┘          │Thr 3││
       │                  │       │                      │                  └─────┘│
       │                  │       │                      │                    │     │
  ┌────▼────┐            │  ┌────▼─────┐           ┌────▼─────┐        ┌───▼───┐ │
  │USB/RTSP │            │  │DroneNet  │           │ByteTrack │        │Gimbal │ │
  │  /File  │            │  │  Pico    │           │ Kalman   │        │Servo  │ │
  │ OpenCV  │            │  │TensorRT/ │           │ LAPJV    │        │Screen │ │
  │         │            │  │ONNX RT   │           │ Predict  │        │  HUD  │ │
  └─────────┘            │  └──────────┘           └──────────┘        └───────┘ │
                          └─────────────────────────────────────────────────────────┘

  ◄────── Lock-free SPSC Ring Buffers (drop-oldest, never block capture) ──────►
```

### Data Flow

```
Frame (1280x720 RGB)
  │
  ├─► Letterbox Resize (416x416) ──► CNN Forward Pass ──► NMS
  │                                                         │
  │                                    Detections [N x (x1,y1,x2,y2,conf,class)]
  │                                                         │
  │                         ┌───────────────────────────────┘
  │                         ▼
  │              ┌─────────────────────┐
  │              │     ByteTrack       │
  │              │                     │
  │              │  1st: High-conf det │──► IoU match ──► Kalman Update
  │              │  vs tracked stracks │
  │              │                     │
  │              │  2nd: Low-conf det  │──► IoU match ──► Kalman Update
  │              │  vs unmatched trks  │
  │              │                     │
  │              │  3rd: Unmatched det │──► IoU match ──► Re-track lost
  │              │  vs lost stracks    │
  │              └────────┬────────────┘
  │                       │
  │              Tracks [M x (id, bbox, velocity, class, history)]
  │                       │
  │              ┌────────▼────────────┐
  │              │ Trajectory Predict  │
  │              │                     │
  │              │ Tier 1: Kalman      │──► 1-5 frame extrapolation (free)
  │              │ Tier 2: Polynomial  │──► 0.5-2s quadratic fit
  │              └────────┬────────────┘
  │                       │
  │              ┌────────▼────────────┐
  │              │  Targeting Engine   │
  │              │                     │
  │              │ pixel ──► angle     │──► pan/tilt (atan2 + intrinsics)
  │              │ select primary      │──► nearest / largest / confidence
  │              │ EMA smooth          │──► anti-jitter (alpha=0.3)
  │              │ distance estimate   │──► pinhole model
  │              └────────┬────────────┘
  │                       │
  └───► Original Frame    │    TargetOutput
        + Overlays ◄──────┘    + Servo/Gimbal Commands
```

---

## DroneNet-Pico CNN

Custom anchor-free single-stage detector (~1.3M parameters) designed specifically for drone-in-sky detection.

### Model Architecture

```
Input: 416 x 416 x 3 (RGB)
│
├── BACKBONE
│   ├── ConvBnSiLU(3→16, s=2)     ──► 208x208x16
│   ├── ConvBnSiLU(16→32, s=2)    ──► 104x104x32
│   │   └── MicroBlock(32) x1
│   ├── ConvBnSiLU(32→64, s=2)    ──► 52x52x64      ← P3
│   │   └── MicroBlock(64) x2
│   ├── ConvBnSiLU(64→128, s=2)   ──► 26x26x128     ← P4
│   │   └── MicroBlock(128) x2
│   └── ConvBnSiLU(128→256, s=2)  ──► 13x13x256     ← P5
│       └── MicroBlock(256) x1
│
├── NECK (PAN-Lite, 2 scales)
│   ├── Upsample P5 + concat P4 → Conv ──► N4 (26x26x128)
│   ├── Upsample N4 + concat P3 → Conv ──► N3 (52x52x64)
│   └── Downsample N3 + concat N4 → Conv ──► D4 (26x26x128)
│
├── HEAD (anchor-free, decoupled)
│   ├── Scale 1 (52x52): cls + reg + obj ──► 2704 candidates
│   └── Scale 2 (26x26): cls + reg + obj ──►  676 candidates
│
└── Output: 3380 detections x [cx, cy, w, h, objectness, 5 classes]
```

### MicroBlock (Depthwise-Separable + Residual)

```
input ──► DWConv3x3 ──► BN ──► SiLU ──► PWConv1x1 ──► BN ──► SiLU ──► (+input) ──► output
```

### Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `drone_small` | Micro/toy drones (< 250g) |
| 1 | `drone_medium` | Consumer drones (DJI Mini/Air) |
| 2 | `drone_large` | Professional/industrial drones |
| 3 | `bird` | False positive rejection |
| 4 | `aircraft` | False positive rejection |

---

## Performance

```
┌──────────────────────────────────────────────────────────────────────┐
│                     INFERENCE SPEED (416x416)                        │
│                                                                      │
│  CPU (PyTorch, i7):     ████████████████████░░░░░░░░  101 FPS       │
│  ONNX Runtime (CPU):    ██████████████████████████░░░  ~150 FPS*    │
│  TensorRT FP16 (GPU):   ████████████████████████████   500+ FPS*   │
│                                                                      │
│  * Estimated based on model size (1.3M params, 227KB ONNX)          │
└──────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────┐
│         MODEL STATS               │
│                                   │
│  Parameters:   1,336,772          │
│  ONNX Size:    227 KB             │
│  Input:        416 x 416 x 3     │
│  Output:       3,380 detections   │
│  Classes:      5                  │
│  Unit Tests:   14/14 passing      │
└───────────────────────────────────┘
```

### Pipeline Latency Breakdown

```
Stage          │ Target    │ Notes
───────────────┼───────────┼──────────────────────────────────
Capture        │ < 1 ms    │ USB/RTSP frame grab
Preprocess     │ < 1 ms    │ Letterbox resize + normalize
Detection      │ 2-10 ms   │ CNN forward pass (GPU)
NMS            │ < 0.5 ms  │ Per-class NMS, IoU=0.45
Tracking       │ < 0.5 ms  │ ByteTrack (no Re-ID network)
Prediction     │ < 0.1 ms  │ Polynomial fit on history
Targeting      │ < 0.1 ms  │ Angle computation + smoothing
UI Render      │ < 2 ms    │ OpenCV overlay drawing
───────────────┼───────────┼──────────────────────────────────
Total          │ < 10 ms   │ 100+ FPS end-to-end
```

---

## Operator HUD

```
┌────────────────────────────────────────────────────────────────────┐
│ [LIVE] FPS: 142 | Detect: 3.2ms | Track: 0.4ms | Tracks: 2      │
│                                                                    │
│                      ┌─────────┐                                   │
│                      │ D-02    │◄── Green: tracked drone           │
│                      │drone_m  │    with ID + class                │
│                      │  92%  ↗ │    confidence + velocity          │
│                      └─────────┘                                   │
│                           ····→  ◄── Predicted trajectory          │
│                                                                    │
│        ╔═══════════╗                                               │
│        ║ ★ D-01   ║◄── Red: primary target                       │
│        ║ [LOCKED] ║    with targeting reticle                     │
│     ───╫─────╫─────╫───                                            │
│        ║     ↑     ║                                               │
│        ╚═══════════╝                                               │
│              ·                                                     │
│              · ◄── Predicted path                                  │
│                                                                    │
│ ┌────────────────────────────────────────────┐                     │
│ │ Target: D-01 [LOCKED]                      │                     │
│ │ Pan: +12.4°  Tilt: -3.1°  Dist: ~85m      │◄── Info panel       │
│ │ Class: drone_medium  Conf: 94%             │                     │
│ └────────────────────────────────────────────┘                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
drone-tracker/
├── CMakeLists.txt                  # Top-level build
├── config/
│   └── pipeline.yaml               # Full pipeline configuration
│
├── src/                             # C++ inference pipeline
│   ├── main.cpp                     # Entry point
│   ├── core/
│   │   ├── config.h/.cpp           # YAML config loader
│   │   ├── logger.h/.cpp           # spdlog wrapper
│   │   ├── pipeline.h/.cpp         # 4-thread pipeline orchestrator
│   │   ├── frame.h                 # Frame, Detection, Track, TargetOutput structs
│   │   ├── ring_buffer.h           # Lock-free SPSC ring buffer
│   │   └── timer.h                 # Scoped timer + FPS counter
│   ├── capture/
│   │   ├── capture_base.h          # Abstract interface
│   │   ├── capture_usb.h/.cpp      # USB/V4L2 camera
│   │   ├── capture_rtsp.h/.cpp     # RTSP/IP stream
│   │   ├── capture_file.h/.cpp     # Video file (with loop)
│   │   └── capture_factory.h/.cpp  # Config-driven factory
│   ├── detect/
│   │   ├── detector_base.h         # Abstract interface
│   │   ├── detector_tensorrt.h/.cpp # TensorRT FP16 backend
│   │   ├── detector_onnx.h/.cpp    # ONNX Runtime backend
│   │   ├── preprocessing.h/.cpp    # Letterbox, NMS
│   │   └── detector_factory.h/.cpp # Backend selection
│   ├── track/
│   │   ├── tracker.h/.cpp          # ByteTrack (two-stage association)
│   │   ├── kalman_filter.h/.cpp    # 8-state Kalman filter
│   │   ├── lapjv.h/.cpp            # Hungarian/Jonker-Volgenant assignment
│   │   └── strack.h/.cpp           # Single track lifecycle
│   ├── predict/
│   │   ├── trajectory_predictor.h/.cpp  # Kalman + polynomial prediction
│   │   └── motion_model.h/.cpp          # Weighted least-squares fitting
│   ├── target/
│   │   ├── targeting_engine.h/.cpp      # Target selection + angle output
│   │   ├── coordinate_transform.h/.cpp  # Pixel → pan/tilt angles
│   │   ├── gimbal_controller.h/.cpp     # PELCO-D serial protocol
│   │   └── servo_controller.h/.cpp      # PWM servo control
│   └── ui/
│       ├── overlay_renderer.h/.cpp # Composites all HUD elements
│       ├── ui_window.h/.cpp        # OpenCV display + recording
│       └── hud_elements.h/.cpp     # Boxes, reticle, trajectories, panels
│
├── training/                        # Python training pipeline
│   ├── models/
│   │   ├── drone_net.py            # DroneNet-Pico architecture
│   │   └── losses.py              # CIoU + Focal + BCE losses
│   ├── dataset/
│   │   ├── drone_dataset.py        # YOLO-format loader + mosaic
│   │   └── augmentations.py        # Albumentations pipeline
│   ├── utils/
│   │   ├── ema.py                  # Exponential moving average
│   │   └── metrics.py             # mAP computation
│   ├── train.py                    # Training loop (FP16, EMA, cosine LR)
│   ├── evaluate.py                 # Validation with mAP@0.5
│   ├── export_onnx.py              # PyTorch → ONNX export
│   ├── configs/
│   │   └── train_config.yaml       # Training hyperparameters
│   └── requirements.txt
│
├── models/                          # Exported model artifacts
│   └── drone_net_pico.onnx         # Pre-exported (untrained) ONNX model
├── tests/                           # Google Test unit tests
│   ├── test_ring_buffer.cpp
│   ├── test_kalman_filter.cpp
│   ├── test_tracker.cpp
│   └── test_targeting.cpp
├── scripts/
│   └── install_deps.sh             # System dependency installer
└── data/                            # Training data (gitignored)
```

---

## Quick Start

### 1. Install Dependencies

```bash
# System packages (Ubuntu/Debian)
sudo apt-get install -y cmake build-essential \
    libopencv-dev libspdlog-dev libyaml-cpp-dev libeigen3-dev

# Python training environment
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build C++ Pipeline

```bash
mkdir build && cd build

# Without GPU (CPU/ONNX only)
cmake .. -DUSE_TENSORRT=OFF -DUSE_ONNXRUNTIME=OFF
make -j$(nproc)

# With TensorRT (requires CUDA + TensorRT installed)
cmake .. -DUSE_TENSORRT=ON -DUSE_ONNXRUNTIME=ON
make -j$(nproc)
```

### 3. Train the Model

Prepare training data in YOLO format:

```
data/
├── images/
│   ├── train/    # Training images (.jpg/.png)
│   └── val/      # Validation images
└── labels/
    ├── train/    # One .txt per image: class cx cy w h (normalized)
    └── val/
```

Train:

```bash
cd training
source venv/bin/activate
python train.py --config configs/train_config.yaml
```

Export to ONNX:

```bash
python export_onnx.py --checkpoint runs/train/best.pt --output ../models/drone_net_pico.onnx
```

### 4. Run

```bash
# USB camera
./build/src/drone_tracker --config config/pipeline.yaml

# Video file (edit config/pipeline.yaml: source: "file", path: "path/to/video.mp4")
./build/src/drone_tracker --config config/pipeline.yaml
```

### 5. Run Tests

```bash
./build/tests/drone_tracker_tests
```

---

## Configuration

All settings are in `config/pipeline.yaml`:

```yaml
capture:
  source: "usb"           # "usb", "rtsp", or "file"
  device: 0               # Camera index
  width: 1280
  height: 720
  fps: 60

detector:
  backend: "tensorrt"      # "tensorrt" or "onnxruntime"
  model_path: "models/drone_net_pico.engine"
  input_size: 416
  confidence_threshold: 0.25
  nms_iou_threshold: 0.45

tracker:
  max_age: 30              # Frames before deleting lost track
  min_hits: 3              # Frames before confirming track
  high_threshold: 0.5      # High confidence split
  low_threshold: 0.1       # Low confidence split

targeting:
  selection_mode: "nearest_center"  # nearest_center/largest/highest_confidence/manual
  smoothing_alpha: 0.3     # EMA smoothing for servo output
  camera_fx: 1066.67       # Camera intrinsics (from calibration)
  camera_fy: 1066.67
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `T` | Cycle target selection mode |
| `1`-`9` | Manually select track by ID |
| `R` | Start/stop recording |
| `D` | Toggle debug overlay |

---

## Tracking: ByteTrack

Two-stage association without appearance model — optimal for drone-in-sky where objects are well-separated against uniform backgrounds.

```
                    Detections
                        │
            ┌───────────┴───────────┐
            │                       │
     High Confidence           Low Confidence
      (conf ≥ 0.5)            (0.1 ≤ conf < 0.5)
            │                       │
   ┌────────▼────────┐              │
   │ 1st Association │              │
   │ IoU match vs    │              │
   │ tracked stracks │              │
   └────┬───────┬────┘              │
        │       │                   │
   Matched  Unmatched              │
   (update)  tracks                 │
                │                   │
       ┌────────▼────────┐          │
       │ 2nd Association │◄─────────┘
       │ IoU match vs    │
       │ low-conf dets   │
       └────┬───────┬────┘
            │       │
       Matched  Still unmatched
       (update)     │
                    ▼
              Mark as LOST
              (delete after max_age frames)

   Unmatched high-conf detections
                │
       ┌────────▼────────┐
       │ 3rd Association │
       │ IoU match vs    │
       │ lost stracks    │
       └────┬───────┬────┘
            │       │
       Re-tracked   New track
       (recover)    (create)
```

---

## Trajectory Prediction

```
Position History (last 30 frames)
│
├── Tier 1: Kalman Extrapolation (always active)
│   └── predicted(t+k) = position + k × velocity
│   └── Cost: ~0 (already computed by tracker)
│   └── Horizon: 1-5 frames (~50ms at 100 FPS)
│
└── Tier 2: Polynomial Fitting (for longer horizons)
    └── Fit x(t) = at² + bt + c  (weighted least squares)
    └── Fit y(t) = at² + bt + c  (recent points weighted more)
    └── Horizon: 0.5-2 seconds
    └── Cost: ~10 μs per track
```

---

## Targeting System

```
                Track Position (px, py)
                        │
                ┌───────▼────────┐
                │   Pixel → Angle │
                │                 │
                │ pan  = atan2(px - cx, fx) × 180/π
                │ tilt = atan2(cy - py, fy) × 180/π
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Target Select  │
                │                 │
                │ Mode: nearest_center ──► min(dx² + dy²)
                │       largest        ──► max(w × h)
                │       highest_conf   ──► max(confidence)
                │       manual         ──► operator pick
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  EMA Smoothing  │
                │                 │
                │ smooth(t) = α × raw(t) + (1-α) × smooth(t-1)
                │ α = 0.3 at 100+ FPS → ~30ms effective lag
                └───────┬────────┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      ┌─────────┐ ┌──────────┐ ┌─────────┐
      │ Gimbal  │ │  Screen  │ │  Servo  │
      │ PELCO-D │ │ Overlay  │ │   PWM   │
      │ Serial  │ │  OpenCV  │ │ Serial  │
      └─────────┘ └──────────┘ └─────────┘
```

---

## Training Pipeline

```
Dataset (YOLO format)
    │
    ├── Mosaic Augmentation (4-image composite, p=0.8)
    ├── Albumentations:
    │   ├── HorizontalFlip (p=0.5)
    │   ├── RandomScale (0.5-1.5x)
    │   ├── RandomBrightnessContrast
    │   ├── HueSaturationValue
    │   ├── GaussNoise + MotionBlur
    │   └── RandomFog (p=0.1)
    │
    ▼
DroneNet-Pico (train mode)
    │
    ├── Loss = 1.0 × BCE(objectness)
    │        + 0.5 × BCE(classification)
    │        + 5.0 × CIoU(regression)
    │
    ├── Optimizer: SGD (momentum=0.937, weight_decay=5e-4)
    ├── LR Schedule: Cosine Annealing (0.01 → 1e-5)
    ├── Warmup: 3 epochs linear
    ├── Precision: FP16 mixed
    ├── EMA: decay=0.9999
    │
    ▼
Best Checkpoint (by loss)
    │
    ▼
ONNX Export (opset 17)
    │
    ▼
TensorRT Engine (FP16, cached)
```

---

## Dependencies

### C++ (System)

| Library | Purpose | Version |
|---------|---------|---------|
| OpenCV | Image I/O, capture, drawing | 4.5+ |
| TensorRT | GPU inference (FP16) | 8.6+ |
| ONNX Runtime | CPU/CUDA inference fallback | 1.16+ |
| spdlog | Structured logging | 1.12+ |
| yaml-cpp | Configuration parsing | 0.7+ |
| Eigen3 | Matrix math (Kalman filter) | 3.3+ |
| Google Test | Unit testing | 1.14+ |

### Python (Training)

| Library | Purpose |
|---------|---------|
| PyTorch | Model definition + training |
| torchvision | Image transforms |
| ONNX | Model export + validation |
| Albumentations | Training augmentations |
| TensorBoard | Training visualization |

---

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| CNN | Custom (DroneNet-Pico) | Domain-specific: drones-in-sky needs fewer features than COCO 80-class. 6x smaller than YOLOv8n. `DetectorBase` interface allows swapping to YOLO if needed. |
| Tracker | ByteTrack | No Re-ID needed — drones are separated blobs against sky. Sub-millisecond, no neural network overhead. |
| Prediction | Polynomial fit | Drone trajectories follow simple physics. Deterministic, microsecond-fast, sufficient for servo control. |
| Threading | Thread-per-stage | Overlaps ALL stages (CPU + GPU), not just GPU work. Lock-free ring buffers prevent blocking. |
| UI | OpenCV HighGUI | Zero extra dependencies for v1. Clean upgrade path to Dear ImGui via `OverlayRenderer` abstraction. |
| Neck | PAN-Lite (2 scales) | P5 (13x13) dropped — drones at extreme distance are noise anyway. Saves compute. |

---

## Recommended Datasets

| Dataset | Description |
|---------|-------------|
| Drone-vs-Bird | Annotated drone and bird footage |
| Anti-UAV (CVPR) | Infrared + visible drone video |
| MAV-VID | Micro aerial vehicle video dataset |
| Custom recording | Use `scripts/record_video.py` with cameras pointed at sky |

---

## License

MIT
