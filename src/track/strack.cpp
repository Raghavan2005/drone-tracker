#include "track/strack.h"

#include <algorithm>
#include <cmath>

namespace drone_tracker {

STrack::STrack(const Detection& det, int track_id)
    : track_id_(track_id),
      x1_(det.x1), y1_(det.y1), x2_(det.x2), y2_(det.y2),
      confidence_(det.confidence),
      class_id_(det.class_id) {
    auto meas = KalmanFilter::bbox_to_measurement(x1_, y1_, x2_, y2_);
    kf_.init(meas);
    trajectory_.push_back(cv::Point2f(det.cx(), det.cy()));
}

void STrack::predict() {
    float prev_cx = (x1_ + x2_) * 0.5f;
    float prev_cy = (y1_ + y2_) * 0.5f;

    kf_.predict();
    update_bbox_from_kf();
    frames_since_update_++;

    float new_cx = (x1_ + x2_) * 0.5f;
    float new_cy = (y1_ + y2_) * 0.5f;
    vx_ = new_cx - prev_cx;
    vy_ = new_cy - prev_cy;
}

void STrack::update(const Detection& det) {
    float prev_cx = (x1_ + x2_) * 0.5f;
    float prev_cy = (y1_ + y2_) * 0.5f;

    auto meas = KalmanFilter::bbox_to_measurement(det.x1, det.y1, det.x2, det.y2);
    kf_.update(meas);
    update_bbox_from_kf();

    float new_cx = (x1_ + x2_) * 0.5f;
    float new_cy = (y1_ + y2_) * 0.5f;
    vx_ = new_cx - prev_cx;
    vy_ = new_cy - prev_cy;

    confidence_ = det.confidence;
    class_id_ = det.class_id;
    state_ = TrackState::TRACKED;
    frames_since_update_ = 0;
    hit_count_++;

    trajectory_.push_back(cv::Point2f(new_cx, new_cy));
    if (trajectory_.size() > 60) {
        trajectory_.erase(trajectory_.begin());
    }
}

void STrack::mark_lost() {
    state_ = TrackState::LOST;
}

void STrack::mark_removed() {
    state_ = TrackState::REMOVED;
}

Detection STrack::to_detection() const {
    return Detection{x1_, y1_, x2_, y2_, confidence_, class_id_};
}

Track STrack::to_track() const {
    Track t;
    t.track_id = track_id_;
    t.x1 = x1_;
    t.y1 = y1_;
    t.x2 = x2_;
    t.y2 = y2_;
    t.vx = vx_;
    t.vy = vy_;
    t.class_id = class_id_;
    t.confidence = confidence_;
    t.frames_tracked = hit_count_;
    t.trajectory_history = trajectory_;
    t.predicted_position = cv::Point2f((x1_ + x2_) * 0.5f + vx_, (y1_ + y2_) * 0.5f + vy_);
    return t;
}

float STrack::iou(const STrack& other) const {
    return compute_iou(x1_, y1_, x2_, y2_, other.x1_, other.y1_, other.x2_, other.y2_);
}

float STrack::iou(const Detection& det) const {
    return compute_iou(x1_, y1_, x2_, y2_, det.x1, det.y1, det.x2, det.y2);
}

void STrack::update_bbox_from_kf() {
    auto meas = kf_.measurement();
    KalmanFilter::measurement_to_bbox(meas, x1_, y1_, x2_, y2_);
}

float STrack::compute_iou(float ax1, float ay1, float ax2, float ay2,
                           float bx1, float by1, float bx2, float by2) {
    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float a_area = (ax2 - ax1) * (ay2 - ay1);
    float b_area = (bx2 - bx1) * (by2 - by1);
    float union_area = a_area + b_area - inter_area;

    return union_area > 0 ? inter_area / union_area : 0.0f;
}

}  // namespace drone_tracker
