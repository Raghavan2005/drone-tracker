#include "track/kalman_filter.h"

#include <cmath>

namespace drone_tracker {

KalmanFilter::KalmanFilter() {
    F_ = StateMatrix::Identity();
    for (int i = 0; i < 4; i++) {
        F_(i, i + 4) = 1.0f;  // velocity terms
    }

    H_ = MeasMatrix::Zero();
    for (int i = 0; i < 4; i++) {
        H_(i, i) = 1.0f;
    }

    // Process noise
    Q_ = StateMatrix::Identity();
    for (int i = 0; i < 4; i++) Q_(i, i) = 1.0f;
    for (int i = 4; i < 8; i++) Q_(i, i) = 0.01f;

    // Measurement noise
    R_ = Eigen::Matrix<float, 4, 4>::Identity();
    R_(0, 0) = 1.0f;
    R_(1, 1) = 1.0f;
    R_(2, 2) = 10.0f;
    R_(3, 3) = 10.0f;
}

void KalmanFilter::init(const MeasVector& measurement) {
    x_ = StateVector::Zero();
    x_.head<4>() = measurement;

    P_ = StateMatrix::Identity() * 10.0f;
    P_(4, 4) = 100.0f;
    P_(5, 5) = 100.0f;
    P_(6, 6) = 100.0f;
    P_(7, 7) = 100.0f;
}

void KalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::update(const MeasVector& measurement) {
    Eigen::Matrix<float, 4, 4> S = H_ * P_ * H_.transpose() + R_;
    Eigen::Matrix<float, 8, 4> K = P_ * H_.transpose() * S.inverse();

    MeasVector y = measurement - H_ * x_;
    x_ = x_ + K * y;
    P_ = (StateMatrix::Identity() - K * H_) * P_;
}

KalmanFilter::MeasVector KalmanFilter::measurement() const {
    return H_ * x_;
}

KalmanFilter::MeasVector KalmanFilter::bbox_to_measurement(float x1, float y1, float x2, float y2) {
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = x2 - x1;
    float h = y2 - y1;
    float ar = (h > 0) ? w / h : 0.0f;
    return MeasVector(cx, cy, ar, h);
}

void KalmanFilter::measurement_to_bbox(const MeasVector& m, float& x1, float& y1, float& x2, float& y2) {
    float cx = m(0), cy = m(1), ar = m(2), h = m(3);
    float w = ar * h;
    x1 = cx - w * 0.5f;
    y1 = cy - h * 0.5f;
    x2 = cx + w * 0.5f;
    y2 = cy + h * 0.5f;
}

}  // namespace drone_tracker
