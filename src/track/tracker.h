#pragma once

#include <vector>

#include "core/config.h"
#include "core/frame.h"
#include "track/strack.h"

namespace drone_tracker {

class ByteTracker {
public:
    explicit ByteTracker(const TrackerConfig& config);

    std::vector<Track> update(const std::vector<Detection>& detections);
    void reset();

private:
    // Build IoU cost matrix (1 - IoU) between tracks and detections
    std::vector<std::vector<float>> iou_cost_matrix(
        const std::vector<STrack*>& tracks,
        const std::vector<Detection>& detections) const;

    TrackerConfig config_;
    std::vector<STrack> tracked_stracks_;
    std::vector<STrack> lost_stracks_;
    int next_id_ = 1;
};

}  // namespace drone_tracker
