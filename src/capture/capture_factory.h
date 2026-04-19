#pragma once

#include <memory>

#include "capture/capture_base.h"
#include "core/config.h"

namespace drone_tracker {

std::unique_ptr<CaptureBase> create_capture(const CaptureConfig& config);

}  // namespace drone_tracker
