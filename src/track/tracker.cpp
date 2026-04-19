#include "track/tracker.h"
#include "track/lapjv.h"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace drone_tracker {

ByteTracker::ByteTracker(const TrackerConfig& config) : config_(config) {
    STrack::set_min_hits(config.min_hits);
}

std::vector<Track> ByteTracker::update(const std::vector<Detection>& detections) {
    // Split detections by confidence
    std::vector<Detection> det_high, det_low;
    for (const auto& d : detections) {
        if (d.confidence >= config_.high_threshold) {
            det_high.push_back(d);
        } else if (d.confidence >= config_.low_threshold) {
            det_low.push_back(d);
        }
    }

    // Predict all existing tracks
    for (auto& st : tracked_stracks_) st.predict();
    for (auto& st : lost_stracks_) st.predict();

    // ----- First association: high-confidence detections vs tracked tracks -----
    std::vector<STrack*> tracked_ptrs;
    for (auto& st : tracked_stracks_) tracked_ptrs.push_back(&st);

    std::vector<STrack*> unmatched_tracks;
    std::vector<Detection> unmatched_dets_high;

    if (!tracked_ptrs.empty() && !det_high.empty()) {
        auto cost1 = iou_cost_matrix(tracked_ptrs, det_high);
        std::vector<int> track_to_det, det_to_track;
        lapjv(cost1, track_to_det, det_to_track, 1.0f - config_.iou_threshold);

        for (size_t i = 0; i < tracked_ptrs.size(); i++) {
            if (track_to_det[i] >= 0) {
                tracked_ptrs[i]->update(det_high[track_to_det[i]]);
            } else {
                unmatched_tracks.push_back(tracked_ptrs[i]);
            }
        }
        for (size_t j = 0; j < det_high.size(); j++) {
            if (det_to_track[j] < 0) {
                unmatched_dets_high.push_back(det_high[j]);
            }
        }
    } else {
        for (auto* st : tracked_ptrs) unmatched_tracks.push_back(st);
        unmatched_dets_high = det_high;
    }

    // ----- Second association: low-confidence detections vs unmatched tracks -----
    if (!det_low.empty() && !unmatched_tracks.empty()) {
        auto cost2 = iou_cost_matrix(unmatched_tracks, det_low);
        std::vector<int> track_to_det2, det_to_track2;
        lapjv(cost2, track_to_det2, det_to_track2, 1.0f - config_.iou_threshold);

        std::vector<STrack*> still_unmatched;
        for (size_t i = 0; i < unmatched_tracks.size(); i++) {
            if (track_to_det2[i] >= 0) {
                unmatched_tracks[i]->update(det_low[track_to_det2[i]]);
            } else {
                still_unmatched.push_back(unmatched_tracks[i]);
            }
        }
        unmatched_tracks = still_unmatched;
    }

    // ----- Third association: unmatched high detections vs lost tracks -----
    if (!unmatched_dets_high.empty() && !lost_stracks_.empty()) {
        std::vector<STrack*> lost_ptrs;
        for (auto& st : lost_stracks_) lost_ptrs.push_back(&st);

        auto cost3 = iou_cost_matrix(lost_ptrs, unmatched_dets_high);
        std::vector<int> lost_to_det, det_to_lost;
        lapjv(cost3, lost_to_det, det_to_lost, 1.0f - config_.iou_threshold);

        std::vector<Detection> remaining_dets;
        for (size_t i = 0; i < lost_ptrs.size(); i++) {
            if (lost_to_det[i] >= 0) {
                lost_ptrs[i]->update(unmatched_dets_high[lost_to_det[i]]);
                tracked_stracks_.push_back(*lost_ptrs[i]);
            }
        }
        for (size_t j = 0; j < unmatched_dets_high.size(); j++) {
            if (det_to_lost[j] < 0) {
                remaining_dets.push_back(unmatched_dets_high[j]);
            }
        }
        unmatched_dets_high = remaining_dets;
    }

    // Mark unmatched tracks as lost
    for (auto* st : unmatched_tracks) {
        st->mark_lost();
    }

    // Create new tracks from remaining unmatched high-confidence detections
    for (const auto& det : unmatched_dets_high) {
        tracked_stracks_.emplace_back(det, next_id_++);
    }

    // Move lost tracks, remove old ones
    std::vector<STrack> new_lost;
    std::vector<STrack> new_tracked;

    for (auto& st : tracked_stracks_) {
        if (st.state() == TrackState::TRACKED || st.state() == TrackState::NEW) {
            new_tracked.push_back(st);
        } else if (st.state() == TrackState::LOST) {
            if (st.frames_since_update() < config_.max_age) {
                new_lost.push_back(st);
            }
        }
    }
    for (auto& st : lost_stracks_) {
        if (st.state() == TrackState::LOST && st.frames_since_update() < config_.max_age) {
            bool re_tracked = false;
            for (const auto& nt : new_tracked) {
                if (nt.track_id() == st.track_id()) { re_tracked = true; break; }
            }
            if (!re_tracked) {
                new_lost.push_back(st);
            }
        }
    }

    tracked_stracks_ = std::move(new_tracked);
    lost_stracks_ = std::move(new_lost);

    // Collect confirmed tracks for output
    std::vector<Track> output;
    for (const auto& st : tracked_stracks_) {
        if (st.is_confirmed()) {
            output.push_back(st.to_track());
        }
    }

    return output;
}

void ByteTracker::reset() {
    tracked_stracks_.clear();
    lost_stracks_.clear();
    next_id_ = 1;
}

std::vector<std::vector<float>> ByteTracker::iou_cost_matrix(
    const std::vector<STrack*>& tracks,
    const std::vector<Detection>& detections) const {

    std::vector<std::vector<float>> cost(tracks.size(), std::vector<float>(detections.size()));
    for (size_t i = 0; i < tracks.size(); i++) {
        for (size_t j = 0; j < detections.size(); j++) {
            cost[i][j] = 1.0f - tracks[i]->iou(detections[j]);
        }
    }
    return cost;
}

}  // namespace drone_tracker
