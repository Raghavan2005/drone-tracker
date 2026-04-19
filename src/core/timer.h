#pragma once

#include <chrono>
#include <string>

#include <spdlog/spdlog.h>

namespace drone_tracker {

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        spdlog::trace("{}: {:.2f} ms", name_, ms);
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

class FpsCounter {
public:
    void tick() {
        auto now = std::chrono::high_resolution_clock::now();
        if (frame_count_ > 0) {
            double dt = std::chrono::duration<double>(now - last_time_).count();
            accumulated_time_ += dt;
            if (accumulated_time_ >= update_interval_) {
                fps_ = frame_count_ / accumulated_time_;
                frame_count_ = 0;
                accumulated_time_ = 0.0;
            }
        }
        last_time_ = now;
        frame_count_++;
    }

    double fps() const { return fps_; }

private:
    double fps_ = 0.0;
    int frame_count_ = 0;
    double accumulated_time_ = 0.0;
    double update_interval_ = 0.5;
    std::chrono::high_resolution_clock::time_point last_time_;
};

}  // namespace drone_tracker
