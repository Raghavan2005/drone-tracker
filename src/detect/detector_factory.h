#pragma once

#include <memory>

#include "core/config.h"
#include "detect/detector_base.h"

namespace drone_tracker {

std::unique_ptr<DetectorBase> create_detector(const DetectorConfig& config);

}  // namespace drone_tracker
