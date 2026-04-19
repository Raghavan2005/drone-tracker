#include "ui/ui_window.h"

#include <filesystem>

#include <spdlog/spdlog.h>

namespace drone_tracker {

UIWindow::UIWindow(const UIConfig& config) : config_(config) {
    cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    if (config_.fullscreen) {
        cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }
}

UIWindow::~UIWindow() {
    stop_recording();
    cv::destroyWindow(window_name_);
}

void UIWindow::show(const cv::Mat& frame) {
    cv::imshow(window_name_, frame);
}

int UIWindow::wait_key(int delay_ms) {
    int key = cv::waitKey(delay_ms);
    if (key == 'q' || key == 'Q' || key == 27) {
        quit_ = true;
    }
    return key;
}

void UIWindow::start_recording(int width, int height, double fps) {
    if (recording_) return;

    std::filesystem::create_directories(config_.recording_path);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char filename[128];
    std::strftime(filename, sizeof(filename), "rec_%Y%m%d_%H%M%S.mp4", std::localtime(&time));

    std::string path = config_.recording_path + filename;
    writer_.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    if (writer_.isOpened()) {
        recording_ = true;
        spdlog::info("Recording started: {}", path);
    } else {
        spdlog::error("Failed to start recording: {}", path);
    }
}

void UIWindow::stop_recording() {
    if (recording_) {
        writer_.release();
        recording_ = false;
        spdlog::info("Recording stopped");
    }
}

void UIWindow::write_frame(const cv::Mat& frame) {
    if (recording_ && writer_.isOpened()) {
        writer_.write(frame);
    }
}

}  // namespace drone_tracker
