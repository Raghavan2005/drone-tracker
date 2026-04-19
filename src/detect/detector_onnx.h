#pragma once

#ifdef HAS_ONNXRUNTIME

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "detect/detector_base.h"

namespace drone_tracker {

class DetectorONNX : public DetectorBase {
public:
    bool init(const DetectorConfig& config) override;
    std::vector<Detection> detect(const cv::Mat& preprocessed) override;
    std::string backend_name() const override { return "ONNX Runtime"; }

private:
    std::vector<Detection> postprocess(const float* output, int num_detections);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "drone_tracker"};
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::string input_name_;
    std::string output_name_;
    int input_size_ = 416;
    int num_classes_ = 5;
    int num_detections_ = 0;
    float conf_threshold_ = 0.25f;
    float nms_threshold_ = 0.45f;
};

}  // namespace drone_tracker

#endif  // HAS_ONNXRUNTIME
