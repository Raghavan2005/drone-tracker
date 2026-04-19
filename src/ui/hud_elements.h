#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/frame.h"

namespace drone_tracker {

struct PipelineMetrics {
    double capture_fps = 0;
    double detect_ms = 0;
    double track_ms = 0;
    double target_ms = 0;
    double total_fps = 0;
    int active_tracks = 0;
};

namespace hud {

void draw_detection_box(cv::Mat& frame, const Track& track, bool is_primary);
void draw_velocity_arrow(cv::Mat& frame, const Track& track);
void draw_trajectory(cv::Mat& frame, const Track& track);
void draw_prediction(cv::Mat& frame, const Track& track);
void draw_targeting_reticle(cv::Mat& frame, const TargetOutput& target);
void draw_status_bar(cv::Mat& frame, const PipelineMetrics& metrics);
void draw_info_panel(cv::Mat& frame, const TargetOutput& target);
void draw_alert(cv::Mat& frame, const std::string& message);

}  // namespace hud
}  // namespace drone_tracker
