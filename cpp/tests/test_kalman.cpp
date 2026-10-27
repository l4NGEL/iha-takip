#include <gtest/gtest.h>

#include "iha/kalman.hpp"

using iha::Box;
using iha::KalmanBoxFilter;

TEST(Kalman, LearnsConstantVelocity) {
    KalmanBoxFilter kf(Box::fromCenter(100, 50, 10, 10));
    for (int t = 1; t <= 30; ++t) {
        kf.predict();
        kf.update(Box::fromCenter(100.f + 3.f * static_cast<float>(t), 50.f - 1.f * static_cast<float>(t), 10, 10));
    }
    EXPECT_NEAR(kf.vx(), 3.f, 0.1f);
    EXPECT_NEAR(kf.vy(), -1.f, 0.1f);
    kf.predict();
    EXPECT_NEAR(kf.box().cx(), 100.f + 3.f * 31, 0.5f);
    EXPECT_NEAR(kf.box().cy(), 50.f - 31, 0.5f);
}

TEST(Kalman, CovarianceStaysSymmetricPositive) {
    KalmanBoxFilter kf(Box::fromCenter(10, 10, 6, 6));
    for (int t = 0; t < 200; ++t) {
        kf.predict();
        if (t % 3) kf.update(Box::fromCenter(10.f + 0.5f * static_cast<float>(t), 10, 6, 6));
    }
    const auto& P = kf.covariance();
    EXPECT_LT((P - P.transpose()).cwiseAbs().maxCoeff(), 1e-3f);
    Eigen::SelfAdjointEigenSolver<KalmanBoxFilter::Mat8> es(P);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.f);
}

TEST(Kalman, UncertaintyGrowsWhileCoasting) {
    KalmanBoxFilter kf(Box::fromCenter(0, 0, 8, 8));
    kf.predict();
    kf.update(Box::fromCenter(0, 0, 8, 8));
    const Box probe = Box::fromCenter(6, 0, 8, 8);
    const float d_before = kf.centerMahalanobis2(probe);
    for (int i = 0; i < 10; ++i) kf.predict();
    EXPECT_LT(kf.centerMahalanobis2(probe), d_before);
}

TEST(Kalman, MinimumNoiseFloorForTinyTargets) {
    // 4 px'lik hedefte 2 px'lik sicrama tek karede imkansiz sayilmamali (Mahalanobis^2 < 9.2).
    KalmanBoxFilter kf(Box::fromCenter(50, 50, 4, 4));
    for (int i = 0; i < 5; ++i) {
        kf.predict();
        kf.update(Box::fromCenter(50, 50, 4, 4));
    }
    kf.predict();
    EXPECT_LT(kf.centerMahalanobis2(Box::fromCenter(52, 50, 4, 4)), 9.2f);
}

TEST(Kalman, SizeNeverCollapses) {
    KalmanBoxFilter kf(Box::fromCenter(0, 0, 10, 10));
    for (int i = 1; i <= 5; ++i) {
        kf.predict();
        kf.update(Box::fromCenter(0, 0, 10.f - 1.8f * static_cast<float>(i), 10.f - 1.8f * static_cast<float>(i)));
    }
    for (int i = 0; i < 20; ++i) kf.predict();
    EXPECT_GE(kf.box().w, 1.f);
    EXPECT_GE(kf.box().h, 1.f);
}
