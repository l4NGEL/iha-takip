#include "iha/hungarian.hpp"

#include <algorithm>
#include <limits>

namespace iha {

std::vector<std::pair<int, int>> solveAssignment(const std::vector<std::vector<float>>& cost, float max_cost) {
    const int rows = static_cast<int>(cost.size());
    const int cols = rows ? static_cast<int>(cost[0].size()) : 0;
    if (rows == 0 || cols == 0) return {};

    // Kare matrise tamamla; esik ustu ve dolgu hucreleri ayni "atanamaz" maliyeti alir,
    // boylece algoritma onlari gercek bir eslesmeye tercih etmez.
    const int n = std::max(rows, cols);
    const double big = static_cast<double>(max_cost);
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, big));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) a[i + 1][j + 1] = std::min<double>(cost[i][j], big);

    // e-maxx potansiyelli Macar algoritmasi (1 tabanli).
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, inf);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
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
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<std::pair<int, int>> out;
    for (int j = 1; j <= n; ++j) {
        const int i = p[j];
        if (i >= 1 && i <= rows && j <= cols && cost[i - 1][j - 1] < max_cost) out.emplace_back(i - 1, j - 1);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace iha
