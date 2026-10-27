#include <gtest/gtest.h>

#include "iha/latency.hpp"

TEST(Latency, PercentileMatchesNumpyLinear) {
    // numpy.percentile([1,2,3,4], [50, 90]) = [2.5, 3.7]
    EXPECT_DOUBLE_EQ(iha::percentile({4, 1, 3, 2}, 50), 2.5);
    EXPECT_NEAR(iha::percentile({4, 1, 3, 2}, 90), 3.7, 1e-12);
    EXPECT_DOUBLE_EQ(iha::percentile({}, 50), 0.0);
}

TEST(Latency, SummaryKeepsInsertionOrder) {
    iha::LatencyStats s;
    s.add("yakalama", 1.0);
    s.add("cikarim", 5.0);
    s.add("yakalama", 3.0);
    const auto sum = s.summary();
    ASSERT_EQ(sum.size(), 2u);
    EXPECT_EQ(sum[0].name, "yakalama");
    EXPECT_EQ(sum[0].n, 2u);
    EXPECT_DOUBLE_EQ(sum[0].mean_ms, 2.0);
    EXPECT_DOUBLE_EQ(sum[0].max_ms, 3.0);
    EXPECT_NE(s.toJson().find("\"cikarim\""), std::string::npos);
}
