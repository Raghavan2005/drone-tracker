#pragma once

#include <string>

#include "core/config.h"

namespace drone_tracker {

class GimbalController {
public:
    explicit GimbalController(const GimbalConfig& config);
    ~GimbalController();

    bool open();
    void close();
    bool is_open() const { return fd_ >= 0; }

    void set_angles(float pan_deg, float tilt_deg);
    void home();

private:
    void send_pelco_d(uint8_t cmd1, uint8_t cmd2, uint8_t data1, uint8_t data2);

    GimbalConfig config_;
    int fd_ = -1;
    float current_pan_ = 0;
    float current_tilt_ = 0;
};

}  // namespace drone_tracker
