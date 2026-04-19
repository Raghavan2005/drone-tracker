#pragma once

#include <string>

#include "core/config.h"

namespace drone_tracker {

class ServoController {
public:
    explicit ServoController(const GimbalConfig& config);
    ~ServoController();

    bool open();
    void close();
    bool is_open() const { return fd_ >= 0; }

    void set_pan_tilt(float pan_deg, float tilt_deg);

private:
    void write_pwm(int channel, float angle_deg, float min_angle, float max_angle);

    GimbalConfig config_;
    int fd_ = -1;
};

}  // namespace drone_tracker
