#include <gtest/gtest.h>

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "iha/global_motion.hpp"

namespace {

cv::Mat texture(int w, int h, int seed) {
    cv::Mat m(h, w, CV_8UC1);
    cv::RNG rng(static_cast<uint64_t>(seed));
    rng.fill(m, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(m, m, cv::Size(0, 0), 2.0);
    cv::normalize(m, m, 0, 255, cv::NORM_MINMAX);
    return m;
}

// Kaynak goruntuyu A ile esle (p' = L p + t), sinirda yansitma.
cv::Mat warp(const cv::Mat& src, double ang_deg, double s, double tx, double ty) {
    const double r = ang_deg * CV_PI / 180.0;
    cv::Mat M = (cv::Mat_<double>(2, 3) << s * std::cos(r), -s * std::sin(r), tx, s * std::sin(r), s * std::cos(r), ty);
    cv::Mat out;
    cv::warpAffine(src, out, M, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

}  // namespace

TEST(GlobalMotion, RecoversTranslation) {
    iha::GlobalMotionEstimator est;
    const cv::Mat a = texture(640, 512, 1);
    EXPECT_FALSE(est.update(a).has_value());  // ilk kare
    const auto m = est.update(warp(a, 0, 1, 5.3, -2.7));
    ASSERT_TRUE(m.has_value());
    EXPECT_NEAR(m->A.tx, 5.3f, 0.1f);
    EXPECT_NEAR(m->A.ty, -2.7f, 0.1f);
    EXPECT_NEAR(m->A.a, 1.f, 1e-3f);
    EXPECT_GT(m->inliers, 100);
}

TEST(GlobalMotion, RecoversSmallRotationAndScale) {
    iha::GlobalMotionEstimator est;
    const cv::Mat a = texture(640, 512, 2);
    est.update(a);
    const auto m = est.update(warp(a, 1.0, 1.01, 3.0, 1.0));
    ASSERT_TRUE(m.has_value());
    EXPECT_NEAR(m->A.scale(), 1.01f, 2e-3f);
    EXPECT_NEAR(std::atan2(m->A.c, m->A.a) * 180.f / static_cast<float>(CV_PI), 1.f, 0.05f);
}

TEST(GlobalMotion, DownscaledEstimateIsInFullResolutionPixels) {
    iha::GlobalMotionParams p;
    p.downscale = 0.5f;
    iha::GlobalMotionEstimator est(p);
    const cv::Mat a = texture(640, 512, 3);
    est.update(a);
    const auto m = est.update(warp(a, 0, 1, 8.0, 4.0));
    ASSERT_TRUE(m.has_value());
    EXPECT_NEAR(m->A.tx, 8.f, 0.3f);
    EXPECT_NEAR(m->A.ty, 4.f, 0.3f);
}

TEST(GlobalMotion, TexturelessSkyGivesNoEstimate) {
    // Duz gokyuzu + tek kucuk sicak nokta (hedef): yeterli kose yok -> telafi yapilmaz.
    iha::GlobalMotionEstimator est;
    cv::Mat sky(512, 640, CV_8UC1, cv::Scalar(90));
    cv::circle(sky, {300, 200}, 3, cv::Scalar(250), -1);
    est.update(sky);
    cv::Mat sky2(512, 640, CV_8UC1, cv::Scalar(90));
    cv::circle(sky2, {306, 200}, 3, cv::Scalar(250), -1);
    EXPECT_FALSE(est.update(sky2).has_value());
}

TEST(GlobalMotion, MovingTargetDoesNotBiasBackgroundMotion) {
    // Arka plan 4 px saga kayiyor, hedef bolgesi tersine 20 px; RANSAC + maske hedefi eler.
    iha::GlobalMotionEstimator est;
    cv::Mat a = texture(640, 512, 4);
    est.update(a);
    cv::Mat b = warp(a, 0, 1, 4.0, 0.0);
    a(cv::Rect(300, 200, 40, 40)).copyTo(b(cv::Rect(280, 200, 40, 40)));
    const auto m = est.update(b, {iha::Box{300, 200, 40, 40}});
    ASSERT_TRUE(m.has_value());
    EXPECT_NEAR(m->A.tx, 4.f, 0.15f);
}
