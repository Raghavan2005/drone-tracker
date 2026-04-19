#include "ui/hud_elements.h"

#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace drone_tracker {

static const char* CLASS_NAMES[] = {"drone_s", "drone_m", "drone_l", "bird", "aircraft"};

static const cv::Scalar COLOR_TRACK = {0, 255, 0};
static const cv::Scalar COLOR_PRIMARY = {0, 0, 255};
static const cv::Scalar COLOR_PREDICTION = {255, 255, 0};
static const cv::Scalar COLOR_TRAJECTORY = {0, 200, 200};
static const cv::Scalar COLOR_TEXT_BG = {0, 0, 0};
static const cv::Scalar COLOR_ALERT = {0, 0, 255};
static const cv::Scalar COLOR_STATUS = {200, 200, 200};

namespace hud {

void draw_detection_box(cv::Mat& frame, const Track& track, bool is_primary) {
    auto color = is_primary ? COLOR_PRIMARY : COLOR_TRACK;
    int thickness = is_primary ? 3 : 2;

    cv::rectangle(frame, cv::Point(static_cast<int>(track.x1), static_cast<int>(track.y1)),
                  cv::Point(static_cast<int>(track.x2), static_cast<int>(track.y2)),
                  color, thickness);

    char label[64];
    const char* cls = (track.class_id >= 0 && track.class_id < 5) ? CLASS_NAMES[track.class_id] : "?";
    std::snprintf(label, sizeof(label), "D-%02d %s %.0f%%", track.track_id, cls, track.confidence * 100);

    int baseline = 0;
    auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(frame,
                  cv::Point(static_cast<int>(track.x1), static_cast<int>(track.y1) - text_size.height - 6),
                  cv::Point(static_cast<int>(track.x1) + text_size.width + 4, static_cast<int>(track.y1)),
                  COLOR_TEXT_BG, cv::FILLED);
    cv::putText(frame, label,
                cv::Point(static_cast<int>(track.x1) + 2, static_cast<int>(track.y1) - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

void draw_velocity_arrow(cv::Mat& frame, const Track& track) {
    float cx = track.cx();
    float cy = track.cy();
    float scale = 5.0f;
    cv::arrowedLine(frame,
                    cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
                    cv::Point(static_cast<int>(cx + track.vx * scale), static_cast<int>(cy + track.vy * scale)),
                    COLOR_TRACK, 2, cv::LINE_AA, 0, 0.3);
}

void draw_trajectory(cv::Mat& frame, const Track& track) {
    if (track.trajectory_history.size() < 2) return;

    for (size_t i = 1; i < track.trajectory_history.size(); i++) {
        float alpha = static_cast<float>(i) / track.trajectory_history.size();
        auto color = cv::Scalar(COLOR_TRAJECTORY[0] * alpha, COLOR_TRAJECTORY[1] * alpha, COLOR_TRAJECTORY[2] * alpha);
        cv::line(frame,
                 cv::Point(static_cast<int>(track.trajectory_history[i - 1].x),
                           static_cast<int>(track.trajectory_history[i - 1].y)),
                 cv::Point(static_cast<int>(track.trajectory_history[i].x),
                           static_cast<int>(track.trajectory_history[i].y)),
                 color, 1, cv::LINE_AA);
    }
}

void draw_prediction(cv::Mat& frame, const Track& track) {
    if (track.prediction_count < 1) return;

    cv::Point prev(static_cast<int>(track.cx()), static_cast<int>(track.cy()));
    for (int i = 0; i < track.prediction_count; i++) {
        cv::Point pt(static_cast<int>(track.predicted_trajectory[i].x),
                     static_cast<int>(track.predicted_trajectory[i].y));
        // Dotted line effect
        if (i % 2 == 0) {
            cv::line(frame, prev, pt, COLOR_PREDICTION, 2, cv::LINE_AA);
        }
        prev = pt;
    }

    // Draw circle at predicted position
    cv::circle(frame, cv::Point(static_cast<int>(track.predicted_trajectory[0].x),
                                 static_cast<int>(track.predicted_trajectory[0].y)),
               6, COLOR_PREDICTION, 2);
}

void draw_targeting_reticle(cv::Mat& frame, const TargetOutput& target) {
    if (target.track_id < 0) return;

    float cx = (target.bbox_x1 + target.bbox_x2) * 0.5f;
    float cy = (target.bbox_y1 + target.bbox_y2) * 0.5f;
    int icx = static_cast<int>(cx);
    int icy = static_cast<int>(cy);
    int r = static_cast<int>(std::max(target.bbox_x2 - target.bbox_x1, target.bbox_y2 - target.bbox_y1) * 0.6f);

    cv::circle(frame, cv::Point(icx, icy), r, COLOR_PRIMARY, 2, cv::LINE_AA);
    cv::circle(frame, cv::Point(icx, icy), r + 8, COLOR_PRIMARY, 1, cv::LINE_AA);

    int crosshair = r + 15;
    cv::line(frame, cv::Point(icx - crosshair, icy), cv::Point(icx - r - 10, icy), COLOR_PRIMARY, 2);
    cv::line(frame, cv::Point(icx + r + 10, icy), cv::Point(icx + crosshair, icy), COLOR_PRIMARY, 2);
    cv::line(frame, cv::Point(icx, icy - crosshair), cv::Point(icx, icy - r - 10), COLOR_PRIMARY, 2);
    cv::line(frame, cv::Point(icx, icy + r + 10), cv::Point(icx, icy + crosshair), COLOR_PRIMARY, 2);

    const char* state_str = "LOST";
    if (target.state == TargetState::LOCKED) state_str = "LOCKED";
    else if (target.state == TargetState::ACQUIRING) state_str = "ACQUIRING";

    char buf[32];
    std::snprintf(buf, sizeof(buf), "[%s]", state_str);
    cv::putText(frame, buf, cv::Point(icx - 30, icy + r + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRIMARY, 1);
}

void draw_status_bar(cv::Mat& frame, const PipelineMetrics& metrics) {
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 28), COLOR_TEXT_BG, cv::FILLED);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  " [LIVE] FPS: %.0f | Detect: %.1fms | Track: %.1fms | Tracks: %d",
                  metrics.total_fps, metrics.detect_ms, metrics.track_ms, metrics.active_tracks);

    cv::putText(frame, buf, cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, COLOR_STATUS, 1);
}

void draw_info_panel(cv::Mat& frame, const TargetOutput& target) {
    if (target.track_id < 0) return;

    int panel_h = 80;
    int panel_w = 380;
    int x = 10;
    int y = frame.rows - panel_h - 10;

    cv::rectangle(frame, cv::Point(x, y), cv::Point(x + panel_w, y + panel_h),
                  COLOR_TEXT_BG, cv::FILLED);
    cv::rectangle(frame, cv::Point(x, y), cv::Point(x + panel_w, y + panel_h),
                  COLOR_STATUS, 1);

    char line1[128], line2[128], line3[128];
    const char* state_str = (target.state == TargetState::LOCKED) ? "LOCKED" : "ACQUIRING";

    std::snprintf(line1, sizeof(line1), "Target: D-%02d [%s]", target.track_id, state_str);
    std::snprintf(line2, sizeof(line2), "Pan: %+.1f  Tilt: %+.1f  Dist: %.0fm",
                  target.pan_angle_deg, target.tilt_angle_deg,
                  target.distance_estimate > 0 ? target.distance_estimate : 0.0f);

    const char* cls = (target.class_id >= 0 && target.class_id < 5) ? CLASS_NAMES[target.class_id] : "?";
    std::snprintf(line3, sizeof(line3), "Class: %s  Conf: %.0f%%", cls, target.confidence * 100);

    cv::putText(frame, line1, cv::Point(x + 8, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_STATUS, 1);
    cv::putText(frame, line2, cv::Point(x + 8, y + 44), cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_STATUS, 1);
    cv::putText(frame, line3, cv::Point(x + 8, y + 66), cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_STATUS, 1);
}

void draw_alert(cv::Mat& frame, const std::string& message) {
    int baseline = 0;
    auto text_size = cv::getTextSize(message, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    int cx = (frame.cols - text_size.width) / 2;
    int cy = 60;

    cv::rectangle(frame, cv::Point(cx - 10, cy - text_size.height - 10),
                  cv::Point(cx + text_size.width + 10, cy + 10),
                  COLOR_ALERT, cv::FILLED);
    cv::putText(frame, message, cv::Point(cx, cy), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
}

}  // namespace hud
}  // namespace drone_tracker
