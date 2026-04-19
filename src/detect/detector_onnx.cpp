#ifdef HAS_ONNXRUNTIME

#include "detect/detector_onnx.h"
#include "detect/preprocessing.h"

#include <spdlog/spdlog.h>

namespace drone_tracker {

bool DetectorONNX::init(const DetectorConfig& config) {
    input_size_ = config.input_size;
    conf_threshold_ = config.confidence_threshold;
    nms_threshold_ = config.nms_iou_threshold;

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, config.model_path.c_str(), options);
    } catch (const Ort::Exception& e) {
        spdlog::error("Failed to load ONNX model: {}", e.what());
        return false;
    }

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session_->GetInputNameAllocated(0, alloc);
    auto out_name = session_->GetOutputNameAllocated(0, alloc);
    input_name_ = in_name.get();
    output_name_ = out_name.get();

    auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() >= 2) {
        num_detections_ = static_cast<int>(output_shape[1]);
    }

    spdlog::info("ONNX Runtime detector ready: {}x{}, {} candidate detections",
                 input_size_, input_size_, num_detections_);
    return true;
}

std::vector<Detection> DetectorONNX::detect(const cv::Mat& preprocessed) {
    cv::Mat float_img;
    preprocessed.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // HWC -> CHW
    const int area = input_size_ * input_size_;
    std::vector<float> input_data(3 * area);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < area; i++) {
            input_data[c * area + i] = float_img.ptr<float>()[i * 3 + c];
        }
    }

    std::array<int64_t, 4> input_shape = {1, 3, input_size_, input_size_};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    const float* output = output_tensors[0].GetTensorData<float>();
    return postprocess(output, num_detections_);
}

std::vector<Detection> DetectorONNX::postprocess(const float* output, int num_det) {
    std::vector<Detection> detections;
    int stride = 5 + num_classes_;

    for (int i = 0; i < num_det; i++) {
        const float* row = output + i * stride;
        float obj_conf = row[4];
        if (obj_conf < conf_threshold_) continue;

        int best_cls = 0;
        float best_score = 0;
        for (int c = 0; c < num_classes_; c++) {
            if (row[5 + c] > best_score) {
                best_score = row[5 + c];
                best_cls = c;
            }
        }

        float final_conf = obj_conf * best_score;
        if (final_conf < conf_threshold_) continue;

        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        Detection det;
        det.x1 = cx - w * 0.5f;
        det.y1 = cy - h * 0.5f;
        det.x2 = cx + w * 0.5f;
        det.y2 = cy + h * 0.5f;
        det.confidence = final_conf;
        det.class_id = best_cls;
        detections.push_back(det);
    }

    nms(detections, nms_threshold_);
    return detections;
}

}  // namespace drone_tracker

#endif  // HAS_ONNXRUNTIME
