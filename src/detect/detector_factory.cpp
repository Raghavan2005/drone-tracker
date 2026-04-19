#include "detect/detector_factory.h"

#include <stdexcept>

#include <spdlog/spdlog.h>

#ifdef HAS_TENSORRT
#include "detect/detector_tensorrt.h"
#endif

#ifdef HAS_ONNXRUNTIME
#include "detect/detector_onnx.h"
#endif

namespace drone_tracker {

std::unique_ptr<DetectorBase> create_detector(const DetectorConfig& config) {
#ifdef HAS_TENSORRT
    if (config.backend == "tensorrt") {
        auto det = std::make_unique<DetectorTensorRT>();
        if (det->init(config)) {
            return det;
        }
        spdlog::warn("TensorRT init failed, trying ONNX Runtime fallback");
    }
#endif

#ifdef HAS_ONNXRUNTIME
    if (config.backend == "onnxruntime" || config.backend == "tensorrt") {
        auto det = std::make_unique<DetectorONNX>();
        if (det->init(config)) {
            return det;
        }
    }
#endif

    throw std::runtime_error("No detection backend available for: " + config.backend);
}

}  // namespace drone_tracker
