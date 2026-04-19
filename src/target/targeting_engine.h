#pragma once

#include <vector>

#include "core/config.h"
#include "core/frame.h"
#include "target/coordinate_transform.h"

namespace drone_tracker {

enum class SelectionMode { NEAREST_CENTER, LARGEST, HIGHEST_CONFIDENCE, MANUAL };

class TargetingEngine {
public:
    explicit TargetingEngine(const TargetingConfig& config);

    void update(const std::vector<Track>& tracks, int image_w, int image_h);
    TargetOutput primary_target() const { return primary_; }
    const std::vector<TargetOutput>& all_targets() const { return targets_; }

    void set_selection_mode(SelectionMode mode) { mode_ = mode; }
    void set_manual_target(int track_id) { manual_target_id_ = track_id; mode_ = SelectionMode::MANUAL; }
    SelectionMode selection_mode() const { return mode_; }

private:
    void select_primary(const std::vector<Track>& tracks, int image_w, int image_h);
    TargetOutput compute_target(const Track& track, int image_w, int image_h) const;

    TargetingConfig config_;
    CoordinateTransform transform_;
    SelectionMode mode_ = SelectionMode::NEAREST_CENTER;
    int manual_target_id_ = -1;

    TargetOutput primary_;
    std::vector<TargetOutput> targets_;

    // Smoothed angles for the primary target
    float smooth_pan_ = 0.0f;
    float smooth_tilt_ = 0.0f;
    int last_primary_id_ = -1;
};

}  // namespace drone_tracker
