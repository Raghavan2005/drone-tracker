#include "detect/preprocessing.h"

#include <algorithm>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace drone_tracker {

void letterbox(const cv::Mat& src, cv::Mat& dst, int input_size,
               float& scale_x, float& scale_y, int& pad_x, int& pad_y) {
    float scale = std::min(static_cast<float>(input_size) / src.cols,
                           static_cast<float>(input_size) / src.rows);

    int new_w = static_cast<int>(src.cols * scale);
    int new_h = static_cast<int>(src.rows * scale);

    pad_x = (input_size - new_w) / 2;
    pad_y = (input_size - new_h) / 2;

    scale_x = scale;
    scale_y = scale;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    dst = cv::Mat(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(dst(cv::Rect(pad_x, pad_y, new_w, new_h)));
}

void scale_detections(std::vector<Detection>& detections,
                      float scale_x, float scale_y, int pad_x, int pad_y,
                      int orig_w, int orig_h) {
    for (auto& d : detections) {
        d.x1 = (d.x1 - pad_x) / scale_x;
        d.y1 = (d.y1 - pad_y) / scale_y;
        d.x2 = (d.x2 - pad_x) / scale_x;
        d.y2 = (d.y2 - pad_y) / scale_y;

        d.x1 = std::clamp(d.x1, 0.0f, static_cast<float>(orig_w));
        d.y1 = std::clamp(d.y1, 0.0f, static_cast<float>(orig_h));
        d.x2 = std::clamp(d.x2, 0.0f, static_cast<float>(orig_w));
        d.y2 = std::clamp(d.y2, 0.0f, static_cast<float>(orig_h));
    }
}

static float compute_iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float union_area = a.area() + b.area() - inter_area;

    return union_area > 0 ? inter_area / union_area : 0.0f;
}

void nms(std::vector<Detection>& detections, float iou_threshold) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (!suppressed[j] && detections[i].class_id == detections[j].class_id &&
                compute_iou(detections[i], detections[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    detections = std::move(result);
}

}  // namespace drone_tracker
