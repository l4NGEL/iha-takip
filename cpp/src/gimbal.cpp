#include "iha/gimbal.hpp"

#include <algorithm>
#include <cmath>

namespace iha {

namespace {
constexpr double kPi = 3.14159265358979323846;
double clampAbs(double v, double lim) { return lim > 0 ? std::clamp(v, -lim, lim) : v; }
}  // namespace

CameraModel CameraModel::fromHfov(int width, int height, double hfov_deg) {
    CameraModel c;
    c.fx = c.fy = 0.5 * width / std::tan(0.5 * hfov_deg * kPi / 180.0);
    c.cx = 0.5 * width;
    c.cy = 0.5 * height;
    return c;
}

AngleError pixelToAngle(const CameraModel& cam, double u, double v) {
    // Isin: d = ((u-cx)/fx, (v-cy)/fy, 1). Yaw: atan2(dx, dz); pitch: atan2(-dy, hypot(dx, dz)).
    const double dx = (u - cam.cx) / cam.fx;
    const double dy = (v - cam.cy) / cam.fy;
    return {std::atan2(dx, 1.0), std::atan2(-dy, std::hypot(dx, 1.0))};
}

void angleToPixel(const CameraModel& cam, const AngleError& a, double& u, double& v) {
    const double dx = std::tan(a.az);
    const double dy = -std::tan(a.el) * std::hypot(dx, 1.0);
    u = cam.cx + dx * cam.fx;
    v = cam.cy + dy * cam.fy;
}

AngleError predictedAngle(const CameraModel& cam, const Box& box, double vx, double vy, double lead_frames) {
    return pixelToAngle(cam, box.cx() + vx * lead_frames, box.cy() + vy * lead_frames);
}

double Pid::step(double error, double measurement, double dt) {
    if (first_) {
        prev_meas_ = measurement;
        first_ = false;
    }
    integ_ = clampAbs(integ_ + p_.ki * error * dt, p_.i_limit);
    const double deriv = dt > 0 ? -(measurement - prev_meas_) / dt : 0.0;
    prev_meas_ = measurement;
    return clampAbs(p_.kp * error + integ_ + p_.kd * deriv, p_.out_limit);
}

void Pid::reset() {
    integ_ = 0;
    prev_meas_ = 0;
    first_ = true;
}

void GimbalAxis::step(double rate_cmd, double dt) {
    const double target = clampAbs(rate_cmd, p_.max_rate);
    double d_rate = (target - rate_) * (p_.tau > 0 ? std::min(1.0, dt / p_.tau) : 1.0);
    d_rate = clampAbs(d_rate, p_.max_accel * dt);
    rate_ += d_rate;
    angle_ += rate_ * dt;
}

}  // namespace iha
