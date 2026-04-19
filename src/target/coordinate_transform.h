#pragma once

#include "core/config.h"

namespace drone_tracker {

class CoordinateTransform {
public:
    explicit CoordinateTransform(const TargetingConfig& config);

    // Convert pixel coordinates to pan/tilt angles (degrees)
    void pixel_to_angle(float px, float py, float& pan_deg, float& tilt_deg) const;

    // Estimate distance given known real-world size and pixel size
    float estimate_distance(float real_size_m, float pixel_size) const;

private:
    float fx_, fy_, cx_, cy_;
};

}  // namespace drone_tracker
