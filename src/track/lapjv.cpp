#include "track/lapjv.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace drone_tracker {

// Simplified Jonker-Volgenant algorithm for linear assignment
// For small matrices typical in drone tracking (< 100 objects), this is efficient enough
void lapjv(const std::vector<std::vector<float>>& cost_matrix,
           std::vector<int>& row_to_col,
           std::vector<int>& col_to_row,
           float threshold) {
    int n_rows = static_cast<int>(cost_matrix.size());
    if (n_rows == 0) {
        row_to_col.clear();
        col_to_row.clear();
        return;
    }
    int n_cols = static_cast<int>(cost_matrix[0].size());
    if (n_cols == 0) {
        row_to_col.assign(n_rows, -1);
        col_to_row.clear();
        return;
    }

    int n = std::max(n_rows, n_cols);

    // Pad cost matrix to square
    std::vector<std::vector<float>> padded(n, std::vector<float>(n, threshold));
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            padded[i][j] = cost_matrix[i][j];
        }
    }

    // Hungarian algorithm (Munkres) for the padded square matrix
    std::vector<float> u(n + 1, 0), v(n + 1, 0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(n + 1, std::numeric_limits<float>::max());
        std::vector<bool> used(n + 1, false);

        do {
            used[j0] = true;
            int i0 = p[j0];
            float delta = std::numeric_limits<float>::max();
            int j1 = 0;

            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    float cur = padded[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    row_to_col.assign(n_rows, -1);
    col_to_row.assign(n_cols, -1);

    for (int j = 1; j <= n; j++) {
        int i = p[j] - 1;
        int col = j - 1;
        if (i < n_rows && col < n_cols && cost_matrix[i][col] < threshold) {
            row_to_col[i] = col;
            col_to_row[col] = i;
        }
    }
}

}  // namespace drone_tracker
