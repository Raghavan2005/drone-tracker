#include "capture/capture_rtsp.h"

#include <spdlog/spdlog.h>

namespace drone_tracker {

CaptureRTSP::CaptureRTSP(const CaptureConfig& config) : config_(config) {}

bool CaptureRTSP::open() {
    cap_.open(config_.url, cv::CAP_FFMPEG);
    if (!cap_.isOpened()) {
        spdlog::error("Failed to open RTSP stream: {}", config_.url);
        return false;
    }
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 2);

    spdlog::info("RTSP stream opened: {}x{} @ {} fps ({})",
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)),
                 cap_.get(cv::CAP_PROP_FPS), config_.url);
    return true;
}

bool CaptureRTSP::read(cv::Mat& frame) {
    return cap_.read(frame);
}

void CaptureRTSP::release() {
    cap_.release();
}

bool CaptureRTSP::is_opened() const {
    return cap_.isOpened();
}

int CaptureRTSP::width() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int CaptureRTSP::height() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double CaptureRTSP::fps() const {
    double f = cap_.get(cv::CAP_PROP_FPS);
    return f > 0 ? f : 30.0;
}

}  // namespace drone_tracker
