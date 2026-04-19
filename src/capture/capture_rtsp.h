#pragma once

#include "capture/capture_base.h"
#include "core/config.h"

#include <opencv2/videoio.hpp>

namespace drone_tracker {

class CaptureRTSP : public CaptureBase {
public:
    explicit CaptureRTSP(const CaptureConfig& config);
    bool open() override;
    bool read(cv::Mat& frame) override;
    void release() override;
    bool is_opened() const override;
    int width() const override;
    int height() const override;
    double fps() const override;

private:
    CaptureConfig config_;
    cv::VideoCapture cap_;
};

}  // namespace drone_tracker
