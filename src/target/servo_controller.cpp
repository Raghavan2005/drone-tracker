#include "target/servo_controller.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <cmath>

#include <spdlog/spdlog.h>

namespace drone_tracker {

ServoController::ServoController(const GimbalConfig& config) : config_(config) {}

ServoController::~ServoController() {
    close();
}

bool ServoController::open() {
    if (!config_.enabled) return false;

    fd_ = ::open(config_.serial_port.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0) {
        spdlog::error("Failed to open servo serial port: {}", config_.serial_port);
        return false;
    }

    struct termios tty;
    std::memset(&tty, 0, sizeof(tty));
    tcgetattr(fd_, &tty);
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag &= ~(PARENB | CSTOPB);
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tcsetattr(fd_, TCSANOW, &tty);

    spdlog::info("Servo controller opened: {}", config_.serial_port);
    return true;
}

void ServoController::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void ServoController::set_pan_tilt(float pan_deg, float tilt_deg) {
    write_pwm(0, pan_deg, config_.pan_min, config_.pan_max);
    write_pwm(1, tilt_deg, config_.tilt_min, config_.tilt_max);
}

void ServoController::write_pwm(int channel, float angle_deg, float min_angle, float max_angle) {
    if (fd_ < 0) return;

    angle_deg = std::clamp(angle_deg, min_angle, max_angle);

    // Map angle to PWM pulse width: 500-2500 microseconds
    float normalized = (angle_deg - min_angle) / (max_angle - min_angle);
    int pulse_us = static_cast<int>(500.0f + normalized * 2000.0f);

    // Simple serial servo protocol: #<channel>P<pulse>T<time>\r\n
    char cmd[32];
    int len = std::snprintf(cmd, sizeof(cmd), "#%dP%dT20\r\n", channel, pulse_us);
    ::write(fd_, cmd, len);
}

}  // namespace drone_tracker
