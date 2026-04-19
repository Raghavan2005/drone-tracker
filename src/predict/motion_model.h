#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace drone_tracker {

struct PolyCoeffs {
    float a, b, c;  // ax^2 + bx + c
};

// Fit a polynomial to a series of timed 2D points using weighted least squares
PolyCoeffs fit_polynomial_1d(const std::vector<float>& t, const std::vector<float>& vals, int order);

}  // namespace drone_tracker
