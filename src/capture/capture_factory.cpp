#include "capture/capture_factory.h"
#include "capture/capture_file.h"
#include "capture/capture_rtsp.h"
#include "capture/capture_usb.h"

#include <stdexcept>

#include <spdlog/spdlog.h>

namespace drone_tracker {

std::unique_ptr<CaptureBase> create_capture(const CaptureConfig& config) {
    if (config.source == "usb") {
        return std::make_unique<CaptureUSB>(config);
    } else if (config.source == "rtsp") {
        return std::make_unique<CaptureRTSP>(config);
    } else if (config.source == "file") {
        return std::make_unique<CaptureFile>(config);
    }
    throw std::runtime_error("Unknown capture source: " + config.source);
}

}  // namespace drone_tracker
