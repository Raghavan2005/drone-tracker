#include "core/pipeline.h"

#include <chrono>

#include <spdlog/spdlog.h>

#include "capture/capture_factory.h"
#include "detect/detector_factory.h"
#include "detect/preprocessing.h"

namespace drone_tracker {

Pipeline::Pipeline(const Config& config) : config_(config) {}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    spdlog::info("Initializing pipeline...");

    capture_ = create_capture(config_.capture);
    if (!capture_->open()) {
        throw std::runtime_error("Failed to open capture source");
    }

    // Detector is optional (can run capture-only for testing)
    try {
        detector_ = create_detector(config_.detector);
    } catch (const std::exception& e) {
        spdlog::warn("Detector not available: {} — running in capture-only mode", e.what());
    }

    tracker_ = std::make_unique<ByteTracker>(config_.tracker);
    predictor_ = std::make_unique<TrajectoryPredictor>(config_.predictor);
    targeter_ = std::make_unique<TargetingEngine>(config_.targeting);
    renderer_ = std::make_unique<OverlayRenderer>(config_.ui);
    ui_ = std::make_unique<UIWindow>(config_.ui);

    if (config_.gimbal.enabled) {
        gimbal_ = std::make_unique<GimbalController>(config_.gimbal);
        gimbal_->open();
    }

    running_ = true;

    thread_capture_ = std::thread(&Pipeline::capture_loop, this);
    if (detector_) {
        thread_detect_ = std::thread(&Pipeline::detect_loop, this);
        thread_track_ = std::thread(&Pipeline::track_loop, this);
    }

    spdlog::info("Pipeline started");

    // Main thread handles UI (OpenCV requires main thread for display on some platforms)
    target_ui_loop();
}

void Pipeline::stop() {
    running_ = false;
    if (thread_capture_.joinable()) thread_capture_.join();
    if (thread_detect_.joinable()) thread_detect_.join();
    if (thread_track_.joinable()) thread_track_.join();

    if (capture_) capture_->release();
    if (gimbal_) gimbal_->close();

    spdlog::info("Pipeline stopped");
}

void Pipeline::capture_loop() {
    uint64_t frame_id = 0;
    spdlog::info("Capture thread started");

    while (running_) {
        Frame f;
        if (!capture_->read(f.image)) {
            spdlog::warn("Capture read failed");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        f.frame_id = frame_id++;
        f.timestamp_sec = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        if (detector_) {
            letterbox(f.image, f.image_resized, config_.detector.input_size,
                      f.scale_x, f.scale_y, f.pad_x, f.pad_y);
        }

        buf_capture_.push_overwrite(std::move(f));
    }
}

void Pipeline::detect_loop() {
    spdlog::info("Detection thread started");

    while (running_) {
        Frame f;
        if (!buf_capture_.try_pop(f)) {
            std::this_thread::yield();
            continue;
        }

        ScopedTimer timer("detect");
        auto detections = detector_->detect(f.image_resized);

        scale_detections(detections, f.scale_x, f.scale_y, f.pad_x, f.pad_y,
                         f.image.cols, f.image.rows);

        metric_detect_ms_ = timer.elapsed_ms();

        DetectionResult dr;
        dr.frame_id = f.frame_id;
        dr.timestamp_sec = f.timestamp_sec;
        dr.image = std::move(f.image);
        dr.detections = std::move(detections);

        buf_detect_.push_overwrite(std::move(dr));
    }
}

void Pipeline::track_loop() {
    spdlog::info("Track thread started");

    while (running_) {
        DetectionResult dr;
        if (!buf_detect_.try_pop(dr)) {
            std::this_thread::yield();
            continue;
        }

        ScopedTimer timer("track");
        auto tracks = tracker_->update(dr.detections);

        for (auto& t : tracks) {
            predictor_->predict(t);
        }

        metric_track_ms_ = timer.elapsed_ms();

        TrackingResult tr;
        tr.frame_id = dr.frame_id;
        tr.timestamp_sec = dr.timestamp_sec;
        tr.image = std::move(dr.image);
        tr.tracks = std::move(tracks);

        buf_track_.push_overwrite(std::move(tr));
    }
}

void Pipeline::target_ui_loop() {
    spdlog::info("Target/UI thread started (main thread)");
    FpsCounter display_fps;

    while (running_) {
        TrackingResult tr;
        bool has_tracking = buf_track_.try_pop(tr);

        if (!has_tracking) {
            // If no detector, read directly from capture for display
            if (!detector_) {
                Frame f;
                if (buf_capture_.try_pop(f)) {
                    display_fps.tick();
                    PipelineMetrics metrics;
                    metrics.total_fps = display_fps.fps();
                    renderer_->render(f.image, {}, TargetOutput{}, metrics);
                    ui_->show(f.image);
                }
            }
            int key = ui_->wait_key(1);
            if (ui_->should_quit()) { running_ = false; break; }
            handle_key(key);
            continue;
        }

        ScopedTimer timer("target");

        targeter_->update(tr.tracks, tr.image.cols, tr.image.rows);
        auto primary = targeter_->primary_target();

        metric_target_ms_ = timer.elapsed_ms();

        if (gimbal_ && gimbal_->is_open() && primary.track_id >= 0) {
            gimbal_->set_angles(primary.predicted_pan_deg, primary.predicted_tilt_deg);
        }

        display_fps.tick();
        PipelineMetrics metrics;
        metrics.total_fps = display_fps.fps();
        metrics.detect_ms = metric_detect_ms_.load();
        metrics.track_ms = metric_track_ms_.load();
        metrics.target_ms = metric_target_ms_.load();
        metrics.active_tracks = static_cast<int>(tr.tracks.size());

        renderer_->render(tr.image, tr.tracks, primary, metrics);

        if (ui_->is_recording()) {
            ui_->write_frame(tr.image);
        }

        ui_->show(tr.image);
        int key = ui_->wait_key(1);
        if (ui_->should_quit()) { running_ = false; break; }
        handle_key(key);
    }
}

void Pipeline::handle_key(int key) {
    if (key < 0) return;

    switch (key) {
        case 't': case 'T': {
            auto mode = targeter_->selection_mode();
            int next = (static_cast<int>(mode) + 1) % 4;
            targeter_->set_selection_mode(static_cast<SelectionMode>(next));
            spdlog::info("Target selection mode: {}", next);
            break;
        }
        case 'r': case 'R':
            if (ui_->is_recording()) {
                ui_->stop_recording();
            } else {
                ui_->start_recording(capture_->width(), capture_->height(), capture_->fps());
            }
            break;
        case 'd': case 'D':
            renderer_->toggle_debug();
            break;
        case 'f': case 'F':
            // Toggle fullscreen handled by OpenCV
            break;
        default:
            if (key >= '1' && key <= '9') {
                int id = key - '0';
                targeter_->set_manual_target(id);
                spdlog::info("Manual target: D-{:02d}", id);
            }
            break;
    }
}

}  // namespace drone_tracker
