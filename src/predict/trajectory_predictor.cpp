#include "predict/trajectory_predictor.h"
#include "predict/motion_model.h"

#include <algorithm>

namespace drone_tracker {

TrajectoryPredictor::TrajectoryPredictor(const PredictorConfig& config) : config_(config) {}

int TrajectoryPredictor::predict(Track& track) const {
    int n = static_cast<int>(track.trajectory_history.size());
    int horizon = std::min(config_.horizon_frames, static_cast<int>(track.predicted_trajectory.size()));

    if (n < 2) {
        // Not enough history, use velocity extrapolation
        float cx = track.cx();
        float cy = track.cy();
        for (int i = 0; i < horizon; i++) {
            cx += track.vx;
            cy += track.vy;
            track.predicted_trajectory[i] = cv::Point2f(cx, cy);
        }
        track.prediction_count = horizon;
        track.predicted_position = track.predicted_trajectory[0];
        return horizon;
    }

    if (config_.method == "kalman" || n < 4) {
        // Simple velocity extrapolation
        float cx = track.cx();
        float cy = track.cy();
        for (int i = 0; i < horizon; i++) {
            cx += track.vx;
            cy += track.vy;
            track.predicted_trajectory[i] = cv::Point2f(cx, cy);
        }
    } else {
        // Polynomial trajectory fitting
        std::vector<float> t_vals(n), x_vals(n), y_vals(n);
        for (int i = 0; i < n; i++) {
            t_vals[i] = static_cast<float>(i);
            x_vals[i] = track.trajectory_history[i].x;
            y_vals[i] = track.trajectory_history[i].y;
        }

        auto px = fit_polynomial_1d(t_vals, x_vals, config_.polynomial_order);
        auto py = fit_polynomial_1d(t_vals, y_vals, config_.polynomial_order);

        float t0 = static_cast<float>(n);
        for (int i = 0; i < horizon; i++) {
            float t = t0 + static_cast<float>(i + 1);
            float pred_x = px.a * t * t + px.b * t + px.c;
            float pred_y = py.a * t * t + py.b * t + py.c;
            track.predicted_trajectory[i] = cv::Point2f(pred_x, pred_y);
        }
    }

    track.prediction_count = horizon;
    if (horizon > 0) {
        track.predicted_position = track.predicted_trajectory[0];
    }
    return horizon;
}

}  // namespace drone_tracker
