#include "capture/capture_file.h"

#include <spdlog/spdlog.h>

namespace drone_tracker {

CaptureFile::CaptureFile(const CaptureConfig& config) : config_(config) {}

bool CaptureFile::open() {
    cap_.open(config_.path);
    if (!cap_.isOpened()) {
        spdlog::error("Failed to open video file: {}", config_.path);
        return false;
    }

    spdlog::info("Video file opened: {}x{} @ {} fps, {} frames ({})",
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)),
                 cap_.get(cv::CAP_PROP_FPS),
                 static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT)),
                 config_.path);
    return true;
}

bool CaptureFile::read(cv::Mat& frame) {
    if (!cap_.read(frame)) {
        if (loop_) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            return cap_.read(frame);
        }
        return false;
    }
    return true;
}

void CaptureFile::release() {
    cap_.release();
}

bool CaptureFile::is_opened() const {
    return cap_.isOpened();
}

int CaptureFile::width() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int CaptureFile::height() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double CaptureFile::fps() const {
    double f = cap_.get(cv::CAP_PROP_FPS);
    return f > 0 ? f : 30.0;
}

}  // namespace drone_tracker
