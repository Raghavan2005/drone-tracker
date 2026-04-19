#include <gtest/gtest.h>

#include <cmath>

#include "target/coordinate_transform.h"
#include "target/targeting_engine.h"

using namespace drone_tracker;

TEST(CoordinateTransform, CenterPixelIsZeroAngle) {
    TargetingConfig cfg;
    cfg.camera_fx = 1000;
    cfg.camera_fy = 1000;
    cfg.camera_cx = 640;
    cfg.camera_cy = 360;

    CoordinateTransform transform(cfg);

    float pan, tilt;
    transform.pixel_to_angle(640, 360, pan, tilt);

    EXPECT_NEAR(pan, 0.0f, 0.01f);
    EXPECT_NEAR(tilt, 0.0f, 0.01f);
}

TEST(CoordinateTransform, OffCenterAngle) {
    TargetingConfig cfg;
    cfg.camera_fx = 1000;
    cfg.camera_fy = 1000;
    cfg.camera_cx = 640;
    cfg.camera_cy = 360;

    CoordinateTransform transform(cfg);

    float pan, tilt;
    transform.pixel_to_angle(1640, 360, pan, tilt);

    // 1000 pixels right of center with fx=1000 -> atan(1) = 45 degrees
    EXPECT_NEAR(pan, 45.0f, 0.1f);
}

TEST(CoordinateTransform, DistanceEstimation) {
    TargetingConfig cfg;
    cfg.camera_fx = 1000;
    cfg.camera_fy = 1000;
    cfg.camera_cx = 640;
    cfg.camera_cy = 360;

    CoordinateTransform transform(cfg);

    // A 0.3m object appearing as 30 pixels: distance = 0.3 * 1000 / 30 = 10m
    float dist = transform.estimate_distance(0.3f, 30.0f);
    EXPECT_NEAR(dist, 10.0f, 0.01f);
}

TEST(TargetingEngine, SelectsNearestCenter) {
    TargetingConfig cfg;
    cfg.selection_mode = "nearest_center";
    cfg.camera_fx = 1000;
    cfg.camera_fy = 1000;
    cfg.camera_cx = 640;
    cfg.camera_cy = 360;

    TargetingEngine engine(cfg);

    std::vector<Track> tracks = {
        {.track_id = 1, .x1 = 600, .y1 = 340, .x2 = 680, .y2 = 380,
         .class_id = 0, .confidence = 0.9f, .frames_tracked = 5},
        {.track_id = 2, .x1 = 100, .y1 = 100, .x2 = 180, .y2 = 180,
         .class_id = 1, .confidence = 0.95f, .frames_tracked = 5},
    };

    engine.update(tracks, 1280, 720);
    auto primary = engine.primary_target();

    EXPECT_EQ(primary.track_id, 1);  // Closer to center
}
