#pragma once

#include <Eigen/Dense>

namespace drone_tracker {

// 8-state Kalman filter for bounding box tracking
// State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
class KalmanFilter {
public:
    using StateVector = Eigen::Matrix<float, 8, 1>;
    using StateMatrix = Eigen::Matrix<float, 8, 8>;
    using MeasVector = Eigen::Matrix<float, 4, 1>;
    using MeasMatrix = Eigen::Matrix<float, 4, 8>;

    KalmanFilter();

    void init(const MeasVector& measurement);
    void predict();
    void update(const MeasVector& measurement);

    MeasVector measurement() const;
    StateVector state() const { return x_; }
    StateMatrix covariance() const { return P_; }

    static MeasVector bbox_to_measurement(float x1, float y1, float x2, float y2);
    static void measurement_to_bbox(const MeasVector& m, float& x1, float& y1, float& x2, float& y2);

private:
    StateVector x_;
    StateMatrix P_;
    StateMatrix F_;      // Transition matrix
    MeasMatrix H_;       // Measurement matrix
    StateMatrix Q_;      // Process noise
    Eigen::Matrix<float, 4, 4> R_;  // Measurement noise
};

}  // namespace drone_tracker
