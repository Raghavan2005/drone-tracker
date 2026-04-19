#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "core/config.h"

namespace drone_tracker {

class UIWindow {
public:
    explicit UIWindow(const UIConfig& config);
    ~UIWindow();

    void show(const cv::Mat& frame);
    int wait_key(int delay_ms = 1);
    bool should_quit() const { return quit_; }

    void start_recording(int width, int height, double fps);
    void stop_recording();
    void write_frame(const cv::Mat& frame);
    bool is_recording() const { return recording_; }

private:
    UIConfig config_;
    std::string window_name_ = "Drone Tracker";
    bool quit_ = false;
    bool recording_ = false;
    cv::VideoWriter writer_;
};

}  // namespace drone_tracker
