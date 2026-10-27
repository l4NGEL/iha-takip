#include <gtest/gtest.h>

#include <cmath>

#include "iha/gimbal.hpp"

using iha::AngleError;
using iha::CameraModel;

namespace {
constexpr double kDeg = 3.14159265358979323846 / 180.0;
}

TEST(Camera, FromHfovEdgeIsHalfFov) {
    const auto cam = CameraModel::fromHfov(640, 512, 40.0);
    EXPECT_NEAR(iha::pixelToAngle(cam, 640, 256).az, 20.0 * kDeg, 1e-9);
    EXPECT_NEAR(iha::pixelToAngle(cam, 0, 256).az, -20.0 * kDeg, 1e-9);
}

TEST(Camera, CenterIsZeroAndSignsFollowConvention) {
    const auto cam = CameraModel::fromHfov(640, 512, 30.0);
    const auto c = iha::pixelToAngle(cam, 320, 256);
    EXPECT_DOUBLE_EQ(c.az, 0.0);
    EXPECT_DOUBLE_EQ(c.el, 0.0);
    const auto ur = iha::pixelToAngle(cam, 400, 200);  // sag ust
    EXPECT_GT(ur.az, 0.0);
    EXPECT_GT(ur.el, 0.0);
}

TEST(Camera, RoundTripAcrossImage) {
    const auto cam = CameraModel::fromHfov(640, 512, 60.0);
    for (double u : {0.0, 77.5, 320.0, 639.0})
        for (double v : {0.0, 100.25, 511.0}) {
            double u2, v2;
            iha::angleToPixel(cam, iha::pixelToAngle(cam, u, v), u2, v2);
            EXPECT_NEAR(u2, u, 1e-9);
            EXPECT_NEAR(v2, v, 1e-9);
        }
}

TEST(Camera, CornerElevationIsNotSmallAngleApproximation) {
    // Genis acida kosede el, (cy - v)/fy'nin atan'indan kucuktur (yaw ile donmus duzlem).
    const auto cam = CameraModel::fromHfov(640, 512, 90.0);
    const auto a = iha::pixelToAngle(cam, 640, 0);
    EXPECT_LT(a.el, std::atan(256.0 / cam.fy));
    EXPECT_NEAR(a.el, std::atan(256.0 / cam.fy * std::cos(a.az)), 1e-12);
}

TEST(Camera, LeadPredictionShiftsAlongVelocity) {
    const auto cam = CameraModel::fromHfov(640, 512, 30.0);
    const iha::Box b = iha::Box::fromCenter(320, 256, 8, 8);
    const auto now = iha::predictedAngle(cam, b, 4, 0, 0);
    const auto lead = iha::predictedAngle(cam, b, 4, 0, 2.5);
    EXPECT_DOUBLE_EQ(now.az, 0.0);
    EXPECT_NEAR(lead.az, std::atan(10.0 / cam.fx), 1e-12);
}

TEST(Pid, ProportionalAndOutputLimit) {
    iha::Pid pid({2.0, 0, 0, 0, 1.5});
    EXPECT_DOUBLE_EQ(pid.step(0.5, 0, 0.01), 1.0);
    EXPECT_DOUBLE_EQ(pid.step(5.0, 0, 0.01), 1.5);
}

TEST(Pid, IntegralIsClampedAntiWindup) {
    iha::Pid pid({0, 10.0, 0, 0.2, 0});
    double out = 0;
    for (int i = 0; i < 1000; ++i) out = pid.step(1.0, 0, 0.01);
    EXPECT_DOUBLE_EQ(out, 0.2);
}

TEST(Pid, DerivativeOnMeasurementHasNoSetpointKick) {
    iha::Pid pid({0, 0, 1.0, 0, 0});
    pid.step(0, 0, 0.01);
    EXPECT_DOUBLE_EQ(pid.step(10.0, 0, 0.01), 0.0);  // hata sicradi, olcum sabit
    EXPECT_DOUBLE_EQ(pid.step(10.0, 0.1, 0.01), -10.0);
}

TEST(GimbalAxis, RateAndAccelLimits) {
    iha::GimbalAxis ax({0.0, 2.0, 10.0});
    ax.step(100.0, 0.01);
    EXPECT_NEAR(ax.rate(), 0.1, 1e-12);  // ivme siniri: 10 rad/s^2 * 0.01 s
    for (int i = 0; i < 1000; ++i) ax.step(100.0, 0.01);
    EXPECT_NEAR(ax.rate(), 2.0, 1e-12);  // hiz siniri
}

TEST(GimbalAxis, ClosedLoopPidConvergesOnStep) {
    iha::GimbalAxis ax;
    // PI sifiri ki/kp = 0.5 rad/s: yaklasmada biriken integral ~2 s'de soner.
    iha::Pid pid({8.0, 4.0, 0.0, 0.5, 3.0});
    const double target = 5.0 * kDeg;
    for (int i = 0; i < 500; ++i) ax.step(pid.step(target - ax.angle(), ax.angle(), 0.01), 0.01);
    EXPECT_NEAR(ax.angle(), target, 0.05 * kDeg);
}
