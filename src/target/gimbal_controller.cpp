#include "target/gimbal_controller.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <cmath>

#include <spdlog/spdlog.h>

namespace drone_tracker {

GimbalController::GimbalController(const GimbalConfig& config) : config_(config) {}

GimbalController::~GimbalController() {
    close();
}

bool GimbalController::open() {
    if (!config_.enabled) return false;

    fd_ = ::open(config_.serial_port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        spdlog::error("Failed to open serial port: {}", config_.serial_port);
        return false;
    }

    struct termios tty;
    std::memset(&tty, 0, sizeof(tty));
    tcgetattr(fd_, &tty);

    speed_t baud;
    switch (config_.baud_rate) {
        case 2400: baud = B2400; break;
        case 4800: baud = B4800; break;
        case 9600: baud = B9600; break;
        case 19200: baud = B19200; break;
        case 115200: baud = B115200; break;
        default: baud = B9600;
    }

    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    tcsetattr(fd_, TCSANOW, &tty);

    spdlog::info("Gimbal serial port opened: {} @ {}", config_.serial_port, config_.baud_rate);
    return true;
}

void GimbalController::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void GimbalController::set_angles(float pan_deg, float tilt_deg) {
    pan_deg = std::clamp(pan_deg, config_.pan_min, config_.pan_max);
    tilt_deg = std::clamp(tilt_deg, config_.tilt_min, config_.tilt_max);

    if (config_.protocol == "pelco_d") {
        float dpan = pan_deg - current_pan_;
        float dtilt = tilt_deg - current_tilt_;

        uint8_t cmd1 = 0, cmd2 = 0;
        if (dpan > 0.1f) cmd2 |= 0x02;      // Pan right
        else if (dpan < -0.1f) cmd2 |= 0x04; // Pan left
        if (dtilt > 0.1f) cmd2 |= 0x08;      // Tilt up
        else if (dtilt < -0.1f) cmd2 |= 0x10; // Tilt down

        uint8_t pan_speed = static_cast<uint8_t>(std::min(63.0f, std::abs(dpan) * 5.0f));
        uint8_t tilt_speed = static_cast<uint8_t>(std::min(63.0f, std::abs(dtilt) * 5.0f));

        send_pelco_d(cmd1, cmd2, pan_speed, tilt_speed);
    }

    current_pan_ = pan_deg;
    current_tilt_ = tilt_deg;
}

void GimbalController::home() {
    set_angles(0, 0);
}

void GimbalController::send_pelco_d(uint8_t cmd1, uint8_t cmd2, uint8_t data1, uint8_t data2) {
    if (fd_ < 0) return;

    uint8_t addr = 0x01;
    uint8_t checksum = (addr + cmd1 + cmd2 + data1 + data2) & 0xFF;
    uint8_t packet[] = {0xFF, addr, cmd1, cmd2, data1, data2, checksum};

    ::write(fd_, packet, sizeof(packet));
}

}  // namespace drone_tracker
