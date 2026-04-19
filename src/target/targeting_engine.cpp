#include "target/targeting_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace drone_tracker {

// Approximate wingspan by class for distance estimation
static constexpr float DRONE_SIZES_M[] = {
    0.15f,  // drone_small
    0.30f,  // drone_medium
    0.60f,  // drone_large
    0.40f,  // bird (wingspan)
    3.00f,  // aircraft
};

TargetingEngine::TargetingEngine(const TargetingConfig& config)
    : config_(config), transform_(config) {
    if (config.selection_mode == "largest") mode_ = SelectionMode::LARGEST;
    else if (config.selection_mode == "highest_confidence") mode_ = SelectionMode::HIGHEST_CONFIDENCE;
    else if (config.selection_mode == "manual") mode_ = SelectionMode::MANUAL;
    else mode_ = SelectionMode::NEAREST_CENTER;
}

void TargetingEngine::update(const std::vector<Track>& tracks, int image_w, int image_h) {
    targets_.clear();
    primary_ = TargetOutput{};

    for (const auto& t : tracks) {
        targets_.push_back(compute_target(t, image_w, image_h));
    }

    if (!targets_.empty()) {
        select_primary(tracks, image_w, image_h);
    }
}

TargetOutput TargetingEngine::compute_target(const Track& track, int image_w, int image_h) const {
    TargetOutput out;
    out.track_id = track.track_id;
    out.class_id = track.class_id;
    out.confidence = track.confidence;

    out.bbox_x1 = track.x1;
    out.bbox_y1 = track.y1;
    out.bbox_x2 = track.x2;
    out.bbox_y2 = track.y2;

    out.screen_x = track.cx() / static_cast<float>(image_w);
    out.screen_y = track.cy() / static_cast<float>(image_h);

    transform_.pixel_to_angle(track.cx(), track.cy(), out.pan_angle_deg, out.tilt_angle_deg);

    if (track.prediction_count > 0) {
        transform_.pixel_to_angle(track.predicted_position.x, track.predicted_position.y,
                                   out.predicted_pan_deg, out.predicted_tilt_deg);
    }

    if (track.class_id >= 0 && track.class_id < 5) {
        float pixel_size = std::max(track.width(), track.height());
        out.distance_estimate = transform_.estimate_distance(DRONE_SIZES_M[track.class_id], pixel_size);
    }

    out.state = (track.frames_tracked >= 3) ? TargetState::LOCKED : TargetState::ACQUIRING;
    out.priority = 1;

    return out;
}

void TargetingEngine::select_primary(const std::vector<Track>& tracks, int image_w, int image_h) {
    int best_idx = -1;
    float best_score = std::numeric_limits<float>::max();

    float center_x = image_w * 0.5f;
    float center_y = image_h * 0.5f;

    for (size_t i = 0; i < targets_.size(); i++) {
        // Skip birds and aircraft for primary targeting
        if (targets_[i].class_id >= 3) continue;

        float score = 0;
        switch (mode_) {
            case SelectionMode::NEAREST_CENTER: {
                float dx = tracks[i].cx() - center_x;
                float dy = tracks[i].cy() - center_y;
                score = dx * dx + dy * dy;
                break;
            }
            case SelectionMode::LARGEST:
                score = -(tracks[i].width() * tracks[i].height());
                break;
            case SelectionMode::HIGHEST_CONFIDENCE:
                score = -tracks[i].confidence;
                break;
            case SelectionMode::MANUAL:
                score = (tracks[i].track_id == manual_target_id_) ? -1e6f : 0;
                break;
        }

        if (score < best_score) {
            best_score = score;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx >= 0) {
        primary_ = targets_[best_idx];

        // Apply exponential smoothing
        float alpha = config_.smoothing_alpha;
        if (primary_.track_id != last_primary_id_) {
            smooth_pan_ = primary_.pan_angle_deg;
            smooth_tilt_ = primary_.tilt_angle_deg;
            last_primary_id_ = primary_.track_id;
        } else {
            smooth_pan_ = alpha * primary_.pan_angle_deg + (1.0f - alpha) * smooth_pan_;
            smooth_tilt_ = alpha * primary_.tilt_angle_deg + (1.0f - alpha) * smooth_tilt_;
        }
        primary_.pan_angle_deg = smooth_pan_;
        primary_.tilt_angle_deg = smooth_tilt_;
    }
}

}  // namespace drone_tracker
