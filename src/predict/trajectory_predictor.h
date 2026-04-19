#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "core/config.h"
#include "core/frame.h"

namespace drone_tracker {

class TrajectoryPredictor {
public:
    explicit TrajectoryPredictor(const PredictorConfig& config);

    // Predict future positions for a track
    // Returns number of valid predictions written to track.predicted_trajectory
    int predict(Track& track) const;

private:
    PredictorConfig config_;
};

}  // namespace drone_tracker
