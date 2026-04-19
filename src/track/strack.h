#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "core/frame.h"
#include "track/kalman_filter.h"

namespace drone_tracker {

enum class TrackState { NEW, TRACKED, LOST, REMOVED };

class STrack {
public:
    STrack(const Detection& det, int track_id);

    void predict();
    void update(const Detection& det);
    void mark_lost();
    void mark_removed();

    Detection to_detection() const;
    Track to_track() const;

    float iou(const STrack& other) const;
    float iou(const Detection& det) const;

    int track_id() const { return track_id_; }
    TrackState state() const { return state_; }
    float confidence() const { return confidence_; }
    int class_id() const { return class_id_; }
    int frames_since_update() const { return frames_since_update_; }
    int hit_count() const { return hit_count_; }
    bool is_confirmed() const { return hit_count_ >= min_hits_; }

    float x1() const { return x1_; }
    float y1() const { return y1_; }
    float x2() const { return x2_; }
    float y2() const { return y2_; }

    static void set_min_hits(int n) { min_hits_ = n; }

private:
    void update_bbox_from_kf();
    static float compute_iou(float ax1, float ay1, float ax2, float ay2,
                              float bx1, float by1, float bx2, float by2);

    KalmanFilter kf_;
    int track_id_;
    TrackState state_ = TrackState::NEW;
    float x1_, y1_, x2_, y2_;
    float vx_ = 0, vy_ = 0;
    float confidence_;
    int class_id_;
    int hit_count_ = 1;
    int frames_since_update_ = 0;
    std::vector<cv::Point2f> trajectory_;

    static inline int min_hits_ = 3;
};

}  // namespace drone_tracker
