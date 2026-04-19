#include <gtest/gtest.h>

#include "track/kalman_filter.h"

using namespace drone_tracker;

TEST(KalmanFilter, InitAndPredict) {
    KalmanFilter kf;
    auto meas = KalmanFilter::bbox_to_measurement(100, 100, 200, 200);
    kf.init(meas);

    kf.predict();
    auto state = kf.state();

    // After one predict with zero velocity, position should stay roughly the same
    EXPECT_NEAR(state(0), 150.0f, 5.0f);  // cx
    EXPECT_NEAR(state(1), 150.0f, 5.0f);  // cy
}

TEST(KalmanFilter, UpdateConverges) {
    KalmanFilter kf;
    auto meas = KalmanFilter::bbox_to_measurement(100, 100, 200, 200);
    kf.init(meas);

    // Simulate moving target
    for (int i = 0; i < 10; i++) {
        kf.predict();
        float offset = static_cast<float>(i) * 5.0f;
        auto new_meas = KalmanFilter::bbox_to_measurement(100 + offset, 100, 200 + offset, 200);
        kf.update(new_meas);
    }

    auto state = kf.state();
    EXPECT_GT(state(4), 0);  // Should have positive x velocity
}

TEST(KalmanFilter, BboxConversion) {
    auto meas = KalmanFilter::bbox_to_measurement(10, 20, 110, 120);

    float x1, y1, x2, y2;
    KalmanFilter::measurement_to_bbox(meas, x1, y1, x2, y2);

    EXPECT_NEAR(x1, 10.0f, 1.0f);
    EXPECT_NEAR(y1, 20.0f, 1.0f);
    EXPECT_NEAR(x2, 110.0f, 1.0f);
    EXPECT_NEAR(y2, 120.0f, 1.0f);
}
