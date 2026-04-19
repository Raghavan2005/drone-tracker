#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/config.h"
#include "core/frame.h"

namespace drone_tracker {

class DetectorBase {
public:
    virtual ~DetectorBase() = default;
    virtual bool init(const DetectorConfig& config) = 0;
    virtual std::vector<Detection> detect(const cv::Mat& preprocessed) = 0;
    virtual std::string backend_name() const = 0;
};

}  // namespace drone_tracker
