include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(drone_tracker_tests
    test_ring_buffer.cpp
    test_kalman_filter.cpp
    test_tracker.cpp
    test_targeting.cpp
)

target_link_libraries(drone_tracker_tests PRIVATE
    drone_tracker_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(drone_tracker_tests)
