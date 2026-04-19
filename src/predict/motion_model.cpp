#include "predict/motion_model.h"

#include <Eigen/Dense>

#include <cmath>

namespace drone_tracker {

PolyCoeffs fit_polynomial_1d(const std::vector<float>& t, const std::vector<float>& vals, int order) {
    PolyCoeffs result{0.0f, 0.0f, 0.0f};
    int n = static_cast<int>(t.size());
    if (n < 2) {
        if (n == 1) result.c = vals[0];
        return result;
    }

    order = std::min(order, n - 1);
    int cols = order + 1;

    // Weight recent points more heavily
    Eigen::MatrixXf A(n, cols);
    Eigen::VectorXf b(n);
    Eigen::VectorXf w(n);

    for (int i = 0; i < n; i++) {
        float weight = 1.0f + static_cast<float>(i) / n;
        w(i) = weight;
        b(i) = vals[i] * weight;

        float ti = t[i];
        float power = 1.0f;
        for (int j = 0; j < cols; j++) {
            A(i, j) = power * weight;
            power *= ti;
        }
    }

    // Solve least squares: (A^T A) x = A^T b
    Eigen::VectorXf x = (A.transpose() * A).ldlt().solve(A.transpose() * b);

    if (cols > 0) result.c = x(0);
    if (cols > 1) result.b = x(1);
    if (cols > 2) result.a = x(2);

    return result;
}

}  // namespace drone_tracker
