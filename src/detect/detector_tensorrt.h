#pragma once

#ifdef HAS_TENSORRT

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "detect/detector_base.h"

namespace drone_tracker {

class DetectorTensorRT : public DetectorBase {
public:
    ~DetectorTensorRT() override;
    bool init(const DetectorConfig& config) override;
    std::vector<Detection> detect(const cv::Mat& preprocessed) override;
    std::string backend_name() const override { return "TensorRT"; }

private:
    bool build_engine(const std::string& onnx_path);
    bool load_engine(const std::string& engine_path);
    bool save_engine(const std::string& engine_path);
    std::vector<Detection> postprocess(float* output, int num_detections);

    struct TRTDeleter {
        template <typename T>
        void operator()(T* p) const {
            if (p) p->destroy();
        }
    };

    std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> context_;

    void* buffers_[2] = {nullptr, nullptr};
    int input_size_ = 416;
    int num_classes_ = 5;
    int num_detections_ = 0;
    float conf_threshold_ = 0.25f;
    float nms_threshold_ = 0.45f;
};

}  // namespace drone_tracker

#endif  // HAS_TENSORRT
