#pragma once

#include <opencv2/core.hpp>

#include "core/frame.h"

namespace drone_tracker {

// Letterbox resize: scale image to fit input_size while preserving aspect ratio
void letterbox(const cv::Mat& src, cv::Mat& dst, int input_size,
               float& scale_x, float& scale_y, int& pad_x, int& pad_y);

// Scale detections from model input space back to original image space
void scale_detections(std::vector<Detection>& detections,
                      float scale_x, float scale_y, int pad_x, int pad_y,
                      int orig_w, int orig_h);

// Apply non-maximum suppression
void nms(std::vector<Detection>& detections, float iou_threshold);

}  // namespace drone_tracker
