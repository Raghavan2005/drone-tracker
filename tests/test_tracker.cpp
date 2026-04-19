#include <gtest/gtest.h>

#include "core/frame.h"
#include "track/tracker.h"

using namespace drone_tracker;

TEST(ByteTracker, SingleDetection) {
    TrackerConfig cfg;
    cfg.min_hits = 1;
    ByteTracker tracker(cfg);

    std::vector<Detection> dets = {{100, 100, 200, 200, 0.9f, 0}};

    auto tracks = tracker.update(dets);
    // First frame: track created but may need min_hits
    // With min_hits=1, should be confirmed immediately
    EXPECT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks[0].track_id, 1);
}

TEST(ByteTracker, TrackPersistence) {
    TrackerConfig cfg;
    cfg.min_hits = 1;
    ByteTracker tracker(cfg);

    // Frame 1
    std::vector<Detection> dets1 = {{100, 100, 200, 200, 0.9f, 0}};
    auto tracks1 = tracker.update(dets1);

    // Frame 2: same detection slightly moved
    std::vector<Detection> dets2 = {{105, 105, 205, 205, 0.9f, 0}};
    auto tracks2 = tracker.update(dets2);

    EXPECT_EQ(tracks2.size(), 1u);
    EXPECT_EQ(tracks2[0].track_id, tracks1[0].track_id);
}

TEST(ByteTracker, MultipleDetections) {
    TrackerConfig cfg;
    cfg.min_hits = 1;
    ByteTracker tracker(cfg);

    std::vector<Detection> dets = {
        {100, 100, 200, 200, 0.9f, 0},
        {400, 400, 500, 500, 0.8f, 1},
    };

    auto tracks = tracker.update(dets);
    EXPECT_EQ(tracks.size(), 2u);
}

TEST(ByteTracker, Reset) {
    TrackerConfig cfg;
    cfg.min_hits = 1;
    ByteTracker tracker(cfg);

    std::vector<Detection> dets = {{100, 100, 200, 200, 0.9f, 0}};
    tracker.update(dets);
    tracker.reset();

    auto tracks = tracker.update(dets);
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks[0].track_id, 1);  // IDs reset
}
