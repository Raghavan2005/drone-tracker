#include "ui/overlay_renderer.h"

namespace drone_tracker {

OverlayRenderer::OverlayRenderer(const UIConfig& config)
    : show_fps_(config.show_fps),
      show_trajectories_(config.show_trajectories),
      show_debug_(config.show_debug) {}

void OverlayRenderer::render(cv::Mat& frame,
                              const std::vector<Track>& tracks,
                              const TargetOutput& primary_target,
                              const PipelineMetrics& metrics) {
    for (const auto& track : tracks) {
        bool is_primary = (track.track_id == primary_target.track_id);
        hud::draw_detection_box(frame, track, is_primary);
        hud::draw_velocity_arrow(frame, track);

        if (show_trajectories_) {
            hud::draw_trajectory(frame, track);
            hud::draw_prediction(frame, track);
        }
    }

    if (primary_target.track_id >= 0) {
        hud::draw_targeting_reticle(frame, primary_target);
        hud::draw_info_panel(frame, primary_target);
    }

    if (show_fps_) {
        hud::draw_status_bar(frame, metrics);
    }
}

}  // namespace drone_tracker
