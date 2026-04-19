#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace drone_tracker {

struct Frame {
    uint64_t frame_id = 0;
    double timestamp_sec = 0.0;
    cv::Mat image;
    cv::Mat image_resized;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;

    float cx() const { return (x1 + x2) * 0.5f; }
    float cy() const { return (y1 + y2) * 0.5f; }
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
};

struct DetectionResult {
    uint64_t frame_id = 0;
    double timestamp_sec = 0.0;
    cv::Mat image;
    std::vector<Detection> detections;
};

enum class TargetState { ACQUIRING, LOCKED, LOST };

struct Track {
    int track_id = -1;
    float x1, y1, x2, y2;
    float vx = 0.0f;
    float vy = 0.0f;
    int class_id = -1;
    float confidence = 0.0f;
    int frames_tracked = 0;
    std::vector<cv::Point2f> trajectory_history;
    cv::Point2f predicted_position;
    std::array<cv::Point2f, 10> predicted_trajectory;
    int prediction_count = 0;

    float cx() const { return (x1 + x2) * 0.5f; }
    float cy() const { return (y1 + y2) * 0.5f; }
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
};

struct TrackingResult {
    uint64_t frame_id = 0;
    double timestamp_sec = 0.0;
    cv::Mat image;
    std::vector<Track> tracks;
};

struct TargetOutput {
    int track_id = -1;
    float pan_angle_deg = 0.0f;
    float tilt_angle_deg = 0.0f;
    float predicted_pan_deg = 0.0f;
    float predicted_tilt_deg = 0.0f;
    float screen_x = 0.0f;
    float screen_y = 0.0f;
    float bbox_x1 = 0, bbox_y1 = 0, bbox_x2 = 0, bbox_y2 = 0;
    float distance_estimate = -1.0f;
    int priority = 0;
    TargetState state = TargetState::LOST;
    int class_id = -1;
    float confidence = 0.0f;
};

}  // namespace drone_tracker
