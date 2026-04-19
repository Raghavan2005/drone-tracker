#include "target/coordinate_transform.h"

#include <cmath>

namespace drone_tracker {

CoordinateTransform::CoordinateTransform(const TargetingConfig& config)
    : fx_(config.camera_fx), fy_(config.camera_fy),
      cx_(config.camera_cx), cy_(config.camera_cy) {}

void CoordinateTransform::pixel_to_angle(float px, float py, float& pan_deg, float& tilt_deg) const {
    pan_deg = std::atan2(px - cx_, fx_) * 180.0f / static_cast<float>(M_PI);
    tilt_deg = std::atan2(cy_ - py, fy_) * 180.0f / static_cast<float>(M_PI);
}

float CoordinateTransform::estimate_distance(float real_size_m, float pixel_size) const {
    if (pixel_size <= 0) return -1.0f;
    return (real_size_m * fx_) / pixel_size;
}

}  // namespace drone_tracker
