#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>

#include "iha/hungarian.hpp"

using Pairs = std::vector<std::pair<int, int>>;

TEST(Hungarian, PicksGlobalOptimumNotGreedy) {
    // Acgozlu (0,0)=1 secer ve toplam 1+10=11 olur; optimum (0,1)+(1,0)=2+3=5.
    const std::vector<std::vector<float>> c{{1, 2}, {3, 10}};
    EXPECT_EQ(iha::solveAssignment(c, 100), (Pairs{{0, 1}, {1, 0}}));
}

TEST(Hungarian, RectangularMoreColumns) {
    const std::vector<std::vector<float>> c{{5, 1, 9}};
    EXPECT_EQ(iha::solveAssignment(c, 100), (Pairs{{0, 1}}));
}

TEST(Hungarian, RectangularMoreRows) {
    const std::vector<std::vector<float>> c{{4}, {2}, {7}};
    EXPECT_EQ(iha::solveAssignment(c, 100), (Pairs{{1, 0}}));
}

TEST(Hungarian, RespectsMaxCost) {
    const std::vector<std::vector<float>> c{{0.5f, 3.f}, {3.f, 3.f}};
    EXPECT_EQ(iha::solveAssignment(c, 2.f), (Pairs{{0, 0}}));
}

TEST(Hungarian, EmptyInputs) {
    EXPECT_TRUE(iha::solveAssignment({}, 1.f).empty());
    EXPECT_TRUE(iha::solveAssignment({{}}, 1.f).empty());
}

TEST(Hungarian, MatchesBruteForceOnRandomSquares) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    for (int trial = 0; trial < 50; ++trial) {
        const int n = 2 + trial % 4;
        std::vector<std::vector<float>> c(n, std::vector<float>(n));
        for (auto& r : c)
            for (auto& v : r) v = U(rng);
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        float best = 1e9f;
        do {
            float s = 0;
            for (int i = 0; i < n; ++i) s += c[i][perm[i]];
            best = std::min(best, s);
        } while (std::next_permutation(perm.begin(), perm.end()));
        float got = 0;
        const auto pairs = iha::solveAssignment(c, 10.f);
        ASSERT_EQ(static_cast<int>(pairs.size()), n);
        for (const auto& [i, j] : pairs) got += c[i][j];
        EXPECT_NEAR(got, best, 1e-5f);
    }
}
