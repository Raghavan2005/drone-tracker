#pragma once

#include <vector>

namespace drone_tracker {

// Jonker-Volgenant linear assignment solver
// cost_matrix: rows x cols cost matrix (row = tracks, col = detections)
// Returns: assignment[row] = col, or -1 if unassigned
// row_to_col: assignment for each row (-1 if unmatched)
// col_to_row: assignment for each col (-1 if unmatched)
void lapjv(const std::vector<std::vector<float>>& cost_matrix,
           std::vector<int>& row_to_col,
           std::vector<int>& col_to_row,
           float threshold = 1e5f);

}  // namespace drone_tracker
