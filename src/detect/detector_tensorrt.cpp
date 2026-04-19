#ifdef HAS_TENSORRT

#include "detect/detector_tensorrt.h"
#include "detect/preprocessing.h"

#include <fstream>
#include <filesystem>

#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace drone_tracker {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            spdlog::warn("[TensorRT] {}", msg);
        }
    }
};

static TRTLogger g_trt_logger;

DetectorTensorRT::~DetectorTensorRT() {
    if (buffers_[0]) cudaFree(buffers_[0]);
    if (buffers_[1]) cudaFree(buffers_[1]);
}

bool DetectorTensorRT::init(const DetectorConfig& config) {
    input_size_ = config.input_size;
    conf_threshold_ = config.confidence_threshold;
    nms_threshold_ = config.nms_iou_threshold;

    std::string engine_path = config.model_path;
    std::string onnx_path = config.model_path;

    // If given an .onnx file, look for cached .engine or build one
    if (onnx_path.size() > 5 && onnx_path.substr(onnx_path.size() - 5) == ".onnx") {
        engine_path = onnx_path.substr(0, onnx_path.size() - 5) + ".engine";
    }

    if (std::filesystem::exists(engine_path)) {
        if (load_engine(engine_path)) {
            spdlog::info("TensorRT engine loaded from cache: {}", engine_path);
        } else {
            return false;
        }
    } else if (std::filesystem::exists(onnx_path)) {
        spdlog::info("Building TensorRT engine from ONNX (this may take a minute)...");
        if (!build_engine(onnx_path)) return false;
        save_engine(engine_path);
    } else {
        spdlog::error("No model file found: {} or {}", engine_path, onnx_path);
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        spdlog::error("Failed to create TensorRT execution context");
        return false;
    }

    // Allocate GPU buffers
    auto input_dims = engine_->getBindingDimensions(0);
    auto output_dims = engine_->getBindingDimensions(1);

    size_t input_bytes = 1;
    for (int i = 0; i < input_dims.nbDims; i++) input_bytes *= input_dims.d[i];
    input_bytes *= sizeof(float);

    size_t output_bytes = 1;
    for (int i = 0; i < output_dims.nbDims; i++) output_bytes *= output_dims.d[i];
    output_bytes *= sizeof(float);

    num_detections_ = output_dims.d[1];

    cudaMalloc(&buffers_[0], input_bytes);
    cudaMalloc(&buffers_[1], output_bytes);

    spdlog::info("TensorRT detector ready: {}x{}, {} candidate detections",
                 input_size_, input_size_, num_detections_);
    return true;
}

bool DetectorTensorRT::build_engine(const std::string& onnx_path) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder, TRTDeleter>(
        nvinfer1::createInferBuilder(g_trt_logger));
    if (!builder) return false;

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDeleter>(
        builder->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser, TRTDeleter>(
        nvonnxparser::createParser(*network, g_trt_logger));
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        spdlog::error("Failed to parse ONNX file: {}", onnx_path);
        return false;
    }

    auto config_ptr = std::unique_ptr<nvinfer1::IBuilderConfig, TRTDeleter>(
        builder->createBuilderConfig());
    config_ptr->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    config_ptr->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config_ptr));
    if (!plan) {
        spdlog::error("Failed to build TensorRT engine");
        return false;
    }

    runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
    engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

    return engine_ != nullptr;
}

bool DetectorTensorRT::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));

    return engine_ != nullptr;
}

bool DetectorTensorRT::save_engine(const std::string& engine_path) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!serialized) return false;

    std::ofstream file(engine_path, std::ios::binary);
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    spdlog::info("TensorRT engine saved to: {}", engine_path);
    return true;
}

std::vector<Detection> DetectorTensorRT::detect(const cv::Mat& preprocessed) {
    // Convert HWC uint8 to NCHW float32 and upload to GPU
    cv::Mat float_img;
    preprocessed.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // HWC -> CHW
    std::vector<float> input_data(3 * input_size_ * input_size_);
    const int area = input_size_ * input_size_;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < area; i++) {
            input_data[c * area + i] = float_img.ptr<float>()[i * 3 + c];
        }
    }

    cudaMemcpy(buffers_[0], input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    context_->executeV2(buffers_);

    // Download results
    int output_elements = num_detections_ * (5 + num_classes_);
    std::vector<float> output(output_elements);
    cudaMemcpy(output.data(), buffers_[1], output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return postprocess(output.data(), num_detections_);
}

std::vector<Detection> DetectorTensorRT::postprocess(float* output, int num_det) {
    std::vector<Detection> detections;
    int stride = 5 + num_classes_;

    for (int i = 0; i < num_det; i++) {
        float* row = output + i * stride;
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

#endif  // HAS_TENSORRT
