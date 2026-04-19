#pragma once

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace drone_tracker {

struct CaptureConfig {
    std::string source = "usb";
    int device = 0;
    std::string url;
    std::string path;
    int width = 1280;
    int height = 720;
    int fps = 60;
};

struct DetectorConfig {
    std::string backend = "tensorrt";
    std::string model_path = "models/drone_net_pico.engine";
    int input_size = 416;
    float confidence_threshold = 0.25f;
    float nms_iou_threshold = 0.45f;
    bool fp16 = true;
};

struct TrackerConfig {
    int max_age = 30;
    int min_hits = 3;
    float iou_threshold = 0.3f;
    float high_threshold = 0.5f;
    float low_threshold = 0.1f;
    int trajectory_history = 30;
};

struct PredictorConfig {
    std::string method = "polynomial";
    int horizon_frames = 10;
    int polynomial_order = 2;
};

struct TargetingConfig {
    std::string selection_mode = "nearest_center";
    float smoothing_alpha = 0.3f;
    float camera_fov_h_deg = 60.0f;
    float camera_fov_v_deg = 34.0f;
    float camera_fx = 1066.67f;
    float camera_fy = 1066.67f;
    float camera_cx = 640.0f;
    float camera_cy = 360.0f;
};

struct GimbalConfig {
    bool enabled = false;
    std::string serial_port = "/dev/ttyUSB0";
    int baud_rate = 9600;
    std::string protocol = "pelco_d";
    float pan_min = -180.0f;
    float pan_max = 180.0f;
    float tilt_min = -90.0f;
    float tilt_max = 90.0f;
};

struct UIConfig {
    bool fullscreen = false;
    bool show_fps = true;
    bool show_trajectories = true;
    bool show_debug = false;
    bool recording_enabled = false;
    std::string recording_path = "recordings/";
};

struct LoggingConfig {
    std::string level = "info";
    std::string file = "logs/drone_tracker.log";
    bool console = true;
};

struct Config {
    CaptureConfig capture;
    DetectorConfig detector;
    TrackerConfig tracker;
    PredictorConfig predictor;
    TargetingConfig targeting;
    GimbalConfig gimbal;
    UIConfig ui;
    LoggingConfig logging;

    static Config load(const std::string& path);
};

}  // namespace drone_tracker
