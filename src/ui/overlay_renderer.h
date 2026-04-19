#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "core/config.h"
#include "core/frame.h"
#include "ui/hud_elements.h"

namespace drone_tracker {

class OverlayRenderer {
public:
    explicit OverlayRenderer(const UIConfig& config);

    void render(cv::Mat& frame,
                const std::vector<Track>& tracks,
                const TargetOutput& primary_target,
                const PipelineMetrics& metrics);

    void toggle_trajectories() { show_trajectories_ = !show_trajectories_; }
    void toggle_debug() { show_debug_ = !show_debug_; }

private:
    bool show_fps_;
    bool show_trajectories_;
    bool show_debug_;
};

}  // namespace drone_tracker
