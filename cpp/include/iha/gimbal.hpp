#pragma once

#include "iha/geometry.hpp"

namespace iha {

// Igne deligi kamera (bozulmasi giderilmis goruntu). fx, fy piksel cinsinden odak.
struct CameraModel {
    double fx = 0, fy = 0, cx = 0, cy = 0;

    // Yatay gorus acisindan (derece) kare pikselli model.
    static CameraModel fromHfov(int width, int height, double hfov_deg);
};

// Optik eksene gore aci hatasi (radyan). az > 0: hedef saga, el > 0: hedef yukari.
struct AngleError {
    double az = 0, el = 0;
};

// Pikselden aci: az = atan((u - cx) / fx), el = atan((cy - v) / fy * cos(az)).
// Elevasyon, azimutla donmus duzlemde olculur; boylece (az, el) tam bir iki eksenli
// (yaw-pitch) gimbal acisina karsilik gelir, kucuk aci yaklasimi yapilmaz.
AngleError pixelToAngle(const CameraModel& cam, double u, double v);
void angleToPixel(const CameraModel& cam, const AngleError& a, double& u, double& v);

// Hat gecikmesini telafi eden ileri ongoru: kutu merkezi + hiz * gecikme (kare).
AngleError predictedAngle(const CameraModel& cam, const Box& box, double vx, double vy, double lead_frames);

struct PidParams {
    double kp = 0, ki = 0, kd = 0;
    double i_limit = 0;    // integral terimin mutlak siniri (anti-windup)
    double out_limit = 0;  // cikis siniri (ornegin rad/s); 0 = sinirsiz
};

class Pid {
public:
    explicit Pid(const PidParams& p = {}) : p_(p) {}
    // Turev olcumden alinir (setpoint sicramasinda turev darbesi olmaz).
    double step(double error, double measurement, double dt);
    void reset();

private:
    PidParams p_;
    double integ_ = 0, prev_meas_ = 0;
    bool first_ = true;
};

// Tek eksen: hiz komutu -> birinci dereceden gecikmeli gercek hiz, hiz ve ivme sinirli.
struct AxisParams {
    double tau = 0.05;          // s, hiz dongusu zaman sabiti
    double max_rate = 3.0;      // rad/s
    double max_accel = 20.0;    // rad/s^2
};

class GimbalAxis {
public:
    explicit GimbalAxis(const AxisParams& p = {}) : p_(p) {}
    void step(double rate_cmd, double dt);
    double angle() const { return angle_; }
    double rate() const { return rate_; }
    void setAngle(double a) { angle_ = a; }

private:
    AxisParams p_;
    double angle_ = 0, rate_ = 0;
};

}  // namespace iha
