#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "capture/capture_base.h"
#include "core/config.h"
#include "core/frame.h"
#include "core/ring_buffer.h"
#include "core/timer.h"
#include "detect/detector_base.h"
#include "predict/trajectory_predictor.h"
#include "target/gimbal_controller.h"
#include "target/targeting_engine.h"
#include "track/tracker.h"
#include "ui/hud_elements.h"
#include "ui/overlay_renderer.h"
#include "ui/ui_window.h"

namespace drone_tracker {

class Pipeline {
public:
    explicit Pipeline(const Config& config);
    ~Pipeline();

    void start();
    void stop();
    bool is_running() const { return running_.load(); }

private:
    void capture_loop();
    void detect_loop();
    void track_loop();
    void target_ui_loop();

    void handle_key(int key);

    Config config_;

    std::unique_ptr<CaptureBase> capture_;
    std::unique_ptr<DetectorBase> detector_;
    std::unique_ptr<ByteTracker> tracker_;
    std::unique_ptr<TrajectoryPredictor> predictor_;
    std::unique_ptr<TargetingEngine> targeter_;
    std::unique_ptr<OverlayRenderer> renderer_;
    std::unique_ptr<UIWindow> ui_;
    std::unique_ptr<GimbalController> gimbal_;

    RingBuffer<Frame, 4> buf_capture_;
    RingBuffer<DetectionResult, 4> buf_detect_;
    RingBuffer<TrackingResult, 4> buf_track_;

    std::atomic<bool> running_{false};
    std::thread thread_capture_;
    std::thread thread_detect_;
    std::thread thread_track_;

    // Metrics
    std::atomic<double> metric_detect_ms_{0};
    std::atomic<double> metric_track_ms_{0};
    std::atomic<double> metric_target_ms_{0};
    FpsCounter fps_counter_;
};

}  // namespace drone_tracker
