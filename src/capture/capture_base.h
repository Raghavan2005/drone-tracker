#pragma once

#include <opencv2/core.hpp>

namespace drone_tracker {

class CaptureBase {
public:
    virtual ~CaptureBase() = default;
    virtual bool open() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual void release() = 0;
    virtual bool is_opened() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
};

}  // namespace drone_tracker
