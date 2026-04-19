#include "core/config.h"

#include <stdexcept>

#include <spdlog/spdlog.h>

namespace drone_tracker {

namespace {

template <typename T>
T get_or(const YAML::Node& node, const std::string& key, const T& default_val) {
    if (node[key]) {
        return node[key].as<T>();
    }
    return default_val;
}

}  // namespace

Config Config::load(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config: " + std::string(e.what()));
    }

    Config cfg;

    if (auto n = root["capture"]) {
        cfg.capture.source = get_or<std::string>(n, "source", "usb");
        cfg.capture.device = get_or<int>(n, "device", 0);
        cfg.capture.url = get_or<std::string>(n, "url", "");
        cfg.capture.path = get_or<std::string>(n, "path", "");
        cfg.capture.width = get_or<int>(n, "width", 1280);
        cfg.capture.height = get_or<int>(n, "height", 720);
        cfg.capture.fps = get_or<int>(n, "fps", 60);
    }

    if (auto n = root["detector"]) {
        cfg.detector.backend = get_or<std::string>(n, "backend", "tensorrt");
        cfg.detector.model_path = get_or<std::string>(n, "model_path", "");
        cfg.detector.input_size = get_or<int>(n, "input_size", 416);
        cfg.detector.confidence_threshold = get_or<float>(n, "confidence_threshold", 0.25f);
        cfg.detector.nms_iou_threshold = get_or<float>(n, "nms_iou_threshold", 0.45f);
        cfg.detector.fp16 = get_or<bool>(n, "fp16", true);
    }

    if (auto n = root["tracker"]) {
        cfg.tracker.max_age = get_or<int>(n, "max_age", 30);
        cfg.tracker.min_hits = get_or<int>(n, "min_hits", 3);
        cfg.tracker.iou_threshold = get_or<float>(n, "iou_threshold", 0.3f);
        cfg.tracker.high_threshold = get_or<float>(n, "high_threshold", 0.5f);
        cfg.tracker.low_threshold = get_or<float>(n, "low_threshold", 0.1f);
        cfg.tracker.trajectory_history = get_or<int>(n, "trajectory_history", 30);
    }

    if (auto n = root["predictor"]) {
        cfg.predictor.method = get_or<std::string>(n, "method", "polynomial");
        cfg.predictor.horizon_frames = get_or<int>(n, "horizon_frames", 10);
        cfg.predictor.polynomial_order = get_or<int>(n, "polynomial_order", 2);
    }

    if (auto n = root["targeting"]) {
        cfg.targeting.selection_mode = get_or<std::string>(n, "selection_mode", "nearest_center");
        cfg.targeting.smoothing_alpha = get_or<float>(n, "smoothing_alpha", 0.3f);
        cfg.targeting.camera_fov_h_deg = get_or<float>(n, "camera_fov_h_deg", 60.0f);
        cfg.targeting.camera_fov_v_deg = get_or<float>(n, "camera_fov_v_deg", 34.0f);
        cfg.targeting.camera_fx = get_or<float>(n, "camera_fx", 1066.67f);
        cfg.targeting.camera_fy = get_or<float>(n, "camera_fy", 1066.67f);
        cfg.targeting.camera_cx = get_or<float>(n, "camera_cx", 640.0f);
        cfg.targeting.camera_cy = get_or<float>(n, "camera_cy", 360.0f);
    }

    if (auto n = root["gimbal"]) {
        cfg.gimbal.enabled = get_or<bool>(n, "enabled", false);
        cfg.gimbal.serial_port = get_or<std::string>(n, "serial_port", "/dev/ttyUSB0");
        cfg.gimbal.baud_rate = get_or<int>(n, "baud_rate", 9600);
        cfg.gimbal.protocol = get_or<std::string>(n, "protocol", "pelco_d");
        if (n["pan_range_deg"] && n["pan_range_deg"].IsSequence()) {
            cfg.gimbal.pan_min = n["pan_range_deg"][0].as<float>();
            cfg.gimbal.pan_max = n["pan_range_deg"][1].as<float>();
        }
        if (n["tilt_range_deg"] && n["tilt_range_deg"].IsSequence()) {
            cfg.gimbal.tilt_min = n["tilt_range_deg"][0].as<float>();
            cfg.gimbal.tilt_max = n["tilt_range_deg"][1].as<float>();
        }
    }

    if (auto n = root["ui"]) {
        cfg.ui.fullscreen = get_or<bool>(n, "fullscreen", false);
        cfg.ui.show_fps = get_or<bool>(n, "show_fps", true);
        cfg.ui.show_trajectories = get_or<bool>(n, "show_trajectories", true);
        cfg.ui.show_debug = get_or<bool>(n, "show_debug", false);
        cfg.ui.recording_enabled = get_or<bool>(n, "recording_enabled", false);
        cfg.ui.recording_path = get_or<std::string>(n, "recording_path", "recordings/");
    }

    if (auto n = root["logging"]) {
        cfg.logging.level = get_or<std::string>(n, "level", "info");
        cfg.logging.file = get_or<std::string>(n, "file", "logs/drone_tracker.log");
        cfg.logging.console = get_or<bool>(n, "console", true);
    }

    spdlog::info("Config loaded from {}", path);
    return cfg;
}

}  // namespace drone_tracker
