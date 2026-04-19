#include "capture/capture_usb.h"

#include <spdlog/spdlog.h>

namespace drone_tracker {

CaptureUSB::CaptureUSB(const CaptureConfig& config) : config_(config) {}

bool CaptureUSB::open() {
    cap_.open(config_.device, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        cap_.open(config_.device);
    }
    if (!cap_.isOpened()) {
        spdlog::error("Failed to open USB camera {}", config_.device);
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
    cap_.set(cv::CAP_PROP_FPS, config_.fps);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 2);

    spdlog::info("USB camera opened: {}x{} @ {} fps (device {})",
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)),
                 cap_.get(cv::CAP_PROP_FPS), config_.device);
    return true;
}

bool CaptureUSB::read(cv::Mat& frame) {
    return cap_.read(frame);
}

void CaptureUSB::release() {
    cap_.release();
}

bool CaptureUSB::is_opened() const {
    return cap_.isOpened();
}

int CaptureUSB::width() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int CaptureUSB::height() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double CaptureUSB::fps() const {
    return cap_.get(cv::CAP_PROP_FPS);
}

}  // namespace drone_tracker
