set(CORE_SOURCES
    core/config.cpp
    core/logger.cpp
)

set(CAPTURE_SOURCES
    capture/capture_usb.cpp
    capture/capture_rtsp.cpp
    capture/capture_file.cpp
    capture/capture_factory.cpp
)

set(DETECT_SOURCES
    detect/preprocessing.cpp
    detect/detector_factory.cpp
)

set(TRACK_SOURCES
    track/tracker.cpp
    track/kalman_filter.cpp
    track/lapjv.cpp
    track/strack.cpp
)

set(PREDICT_SOURCES
    predict/trajectory_predictor.cpp
    predict/motion_model.cpp
)

set(TARGET_SOURCES
    target/targeting_engine.cpp
    target/gimbal_controller.cpp
    target/servo_controller.cpp
    target/coordinate_transform.cpp
)

set(UI_SOURCES
    ui/overlay_renderer.cpp
    ui/ui_window.cpp
    ui/hud_elements.cpp
)

add_library(drone_tracker_core STATIC
    ${CORE_SOURCES}
    ${CAPTURE_SOURCES}
    ${DETECT_SOURCES}
    ${TRACK_SOURCES}
    ${PREDICT_SOURCES}
    ${TARGET_SOURCES}
    ${UI_SOURCES}
    core/pipeline.cpp
)

target_include_directories(drone_tracker_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(drone_tracker_core PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
    spdlog::spdlog
    yaml-cpp
    Eigen3::Eigen
)

if(USE_TENSORRT)
    target_sources(drone_tracker_core PRIVATE detect/detector_tensorrt.cpp)
    target_link_libraries(drone_tracker_core PUBLIC ${NVINFER} ${NVONNXPARSER} ${CUDA_LIBRARIES})
    target_include_directories(drone_tracker_core PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

if(USE_ONNXRUNTIME)
    target_sources(drone_tracker_core PRIVATE detect/detector_onnx.cpp)
    target_link_libraries(drone_tracker_core PUBLIC ${ONNXRT_LIB})
    target_include_directories(drone_tracker_core PUBLIC ${ONNXRT_INCLUDE})
endif()

add_executable(drone_tracker main.cpp)
target_link_libraries(drone_tracker PRIVATE drone_tracker_core)
