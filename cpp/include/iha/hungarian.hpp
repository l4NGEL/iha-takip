#pragma once

#include <utility>
#include <vector>

namespace iha {

// Dikdortgen maliyet matrisi icin en kucuk toplam maliyetli atama (Kuhn-Munkres, O(n^3)).
// cost[i][j] >= max_cost olan ciftler atanmaz. Doner: (satir, sutun) ciftleri.
std::vector<std::pair<int, int>> solveAssignment(const std::vector<std::vector<float>>& cost, float max_cost);

}  // namespace iha
