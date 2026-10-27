// Kapali dongu gimbal benzetimi: goruntu hatti cikisindan (kutu) aci hatasina, PID'den
// iki eksenli gimbal'e. Gercek bir gimbal yok; amac aci-hatasi katmaninin ve takipcinin
// (bosluk doldurma, hiz kestirimi) dongu performansina etkisini olcmek.
//
// Model:
//  * Kamera 30 fps; her kare yakalandiginda hedefin goreli acisi (hedef - gimbal) piksele
//    izdusurulur, Gauss piksel gurultusu eklenir, p_miss olasilikla tespit kaybolur
//    (varsayilan 0.2 = test kumesinde olculen recall 0.80).
//  * Tespit, hat gecikmesi (latency_ms) sonra kontrolcuye ulasir.
//  * Kontrolcu her sonucta komut uretir, sonraki sonuca kadar tutar (sifirinci derece tutma).
//  * Gimbal ekseni 1 kHz'de entegre edilir (hiz dongusu zaman sabiti, hiz/ivme siniri).
//  * Yaw ve pitch bagimsiz kabul edilir (kucuk elevasyonda gecerli yaklasim).
//
// Kiyaslanan kontrolculer:
//  raw         : her karenin ham tespitiyle PID; tespit yoksa komut 0.
//  sticky_nocmc: ByteTracker, kamera hareketi telafisi YOK, kilit yapiskan (eslesmesiz takip
//                coast_output kare boyunca kilitli kalir) -> iki eksik birlikte.
//  track_nocmc : ByteTracker, telafi yok, kilit olculen takibe yol verir.
//  (Olculen: iraksamayi yapiskan kilit yaratiyor; telafinin tek basina etkisi bu modelde gorulmuyor.)
//  track       : ByteTracker + gimbal enkoderinden kamera hareketi telafisi (kayipta ongoruyle devam).
//  lead        : track + gecikme telafisi (ileri ongoru) + hedef aci hizi ileri besleme.
//
//   gimbal_sim [--seeds 20] [--latency-ms 15] [--p-miss 0.2] [--noise-px 1.0] [--csv out.csv] [--json out.json]

#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "iha/args.hpp"
#include "iha/gimbal.hpp"
#include "iha/latency.hpp"
#include "iha/tracker.hpp"

using namespace iha;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

struct Target {
    double az, el, daz, del;  // rad, rad/s
};

// Manevrali gecis: yavas genis salinim + hizli kucuk salinim. En yuksek aci hizi ~16 derece/s.
Target targetAt(double t) {
    const double w1 = 2 * kPi * 0.1, w2 = 2 * kPi * 0.5, w3 = 2 * kPi * 0.3;
    return {(10 * std::sin(w1 * t) + 3 * std::sin(w2 * t)) * kDeg, (5 + 2 * std::sin(w3 * t)) * kDeg,
            (10 * w1 * std::cos(w1 * t) + 3 * w2 * std::cos(w2 * t)) * kDeg, (2 * w3 * std::cos(w3 * t)) * kDeg};
}

struct SimConfig {
    std::string ctrl;
    double fps = 30, latency_s = 0.015, p_miss = 0.2, noise_px = 1.0, hfov_deg = 30, duration_s = 30;
    double box_px = 8;
};

struct SimResult {
    double rms_deg = 0, p95_deg = 0;
    double in_fov = 0;      // hedefin kadrajda oldugu zaman orani
    double within_05 = 0;   // isaretleme hatasi < 0.5 derece zaman orani
    int lost_events = 0;    // kadrajdan cikis sayisi
};

struct Pending {
    double ready_t, capture_t;
    Detections dets;
    double g_az, g_el;  // yakalama anindaki gimbal acisi (enkoder)
};

SimResult runSim(const SimConfig& c, unsigned seed, std::ofstream* trace) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, c.noise_px);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const CameraModel cam = CameraModel::fromHfov(640, 512, c.hfov_deg);
    const double half_h = std::atan(0.5 * 640 / cam.fx), half_v = std::atan(0.5 * 512 / cam.fy);

    AxisParams ap;
    GimbalAxis yaw(ap), pitch(ap);
    const Target t0 = targetAt(0);
    yaw.setAngle(t0.az - 2 * kDeg);  // baslangicta 2 derece kayik
    pitch.setAngle(t0.el + 1 * kDeg);

    const PidParams pp{6.0, 2.0, 0.0, 0.5, ap.max_rate};
    Pid pid_az(pp), pid_el(pp);
    TrackerParams tp;
    tp.min_hits = 2;
    tp.coast_output = 10;
    tp.lock_yields_to_measured = c.ctrl != "sticky_nocmc";
    ByteTracker tracker(tp);

    const double dt = 0.001, frame_dt = 1.0 / c.fps;
    double next_frame = 0, cmd_az = 0, cmd_el = 0;
    double prev_g_az = yaw.angle(), prev_g_el = pitch.angle();
    std::deque<Pending> queue;
    std::vector<double> err;
    long n_in_fov = 0, n_steps = 0;
    bool was_in = true;
    SimResult r;

    for (double t = 0; t < c.duration_s; t += dt) {
        const Target tg = targetAt(t);
        const double rel_az = tg.az - yaw.angle(), rel_el = tg.el - pitch.angle();
        const bool in = std::abs(rel_az) < half_h && std::abs(rel_el) < half_v;

        if (t >= next_frame) {
            next_frame += frame_dt;
            Pending p{t + c.latency_s, t, {}, yaw.angle(), pitch.angle()};
            if (in && U(rng) >= c.p_miss) {
                double u, v;
                angleToPixel(cam, {rel_az, rel_el}, u, v);
                p.dets.push_back({Box::fromCenter(static_cast<float>(u + noise(rng)), static_cast<float>(v + noise(rng)),
                                                  static_cast<float>(c.box_px), static_cast<float>(c.box_px)),
                                  0.8f});
            }
            queue.push_back(std::move(p));
        }

        while (!queue.empty() && queue.front().ready_t <= t) {
            const Pending p = std::move(queue.front());
            queue.pop_front();
            std::optional<AngleError> e;
            double ff_az = 0, ff_el = 0;
            if (c.ctrl == "raw") {
                if (!p.dets.empty()) e = pixelToAngle(cam, p.dets[0].box.cx(), p.dets[0].box.cy());
            } else {
                if (c.ctrl != "track_nocmc" && c.ctrl != "sticky_nocmc") {
                    // Iki yakalama arasinda gimbal donusu -> sahnenin goruntudeki kaymasi.
                    tracker.compensateCameraMotion(static_cast<float>(-cam.fx * (p.g_az - prev_g_az)),
                                                   static_cast<float>(cam.fy * (p.g_el - prev_g_el)));
                }
                prev_g_az = p.g_az;
                prev_g_el = p.g_el;
                tracker.update(p.dets);
                if (auto o = tracker.target()) {
                    if (c.ctrl == "lead") {
                        // Telafiden sonra Kalman hizi hedefin dunya acisal hizidir (piksel/kare).
                        // Goreli hiz = hedef hizi - gimbal donusunun goruntudeki etkisi; yakalamadan
                        // bu yana gecen sure (hat gecikmesi) kadar ileri ongorulur.
                        const double lead_frames = (t - p.capture_t) * c.fps;
                        const double rel_vx = o->vx - yaw.rate() * cam.fx / c.fps;
                        const double rel_vy = o->vy + pitch.rate() * cam.fy / c.fps;
                        e = predictedAngle(cam, o->box, rel_vx, rel_vy, lead_frames);
                        ff_az = o->vx * c.fps / cam.fx;
                        ff_el = -o->vy * c.fps / cam.fy;
                    } else {
                        e = pixelToAngle(cam, o->box.cx(), o->box.cy());
                    }
                }
            }
            if (e) {
                // Olcum = gimbal acisi; hata dogrudan goruntuden gelir.
                cmd_az = ff_az + pid_az.step(e->az, yaw.angle(), frame_dt);
                cmd_el = ff_el + pid_el.step(e->el, pitch.angle(), frame_dt);
            } else {
                cmd_az = cmd_el = 0;  // hedef yok: gimbal durur
            }
        }

        yaw.step(cmd_az, dt);
        pitch.step(cmd_el, dt);

        const double pe = std::hypot(rel_az, rel_el) / kDeg;
        err.push_back(pe);
        ++n_steps;
        if (in) ++n_in_fov;
        if (was_in && !in) ++r.lost_events;
        was_in = in;
        if (trace && (n_steps % 10 == 0)) {
            *trace << c.ctrl << "," << seed << "," << t << "," << tg.az / kDeg << "," << tg.el / kDeg << ","
                   << yaw.angle() / kDeg << "," << pitch.angle() / kDeg << "," << pe << "\n";
        }
    }
    double ss = 0;
    long w05 = 0;
    for (double v : err) {
        ss += v * v;
        if (v < 0.5) ++w05;
    }
    r.rms_deg = std::sqrt(ss / static_cast<double>(err.size()));
    r.p95_deg = percentile(err, 95);
    r.in_fov = static_cast<double>(n_in_fov) / static_cast<double>(n_steps);
    r.within_05 = static_cast<double>(w05) / static_cast<double>(n_steps);
    return r;
}

}  // namespace

int main(int argc, char** argv) try {
    const Args a(argc, argv);
    const int seeds = a.i("seeds", 20);
    SimConfig base;
    base.latency_s = a.f("latency-ms", 15.f) / 1000.0;
    base.p_miss = a.f("p-miss", 0.2f);
    base.noise_px = a.f("noise-px", 1.0f);
    base.hfov_deg = a.f("hfov", 30.f);
    base.fps = a.f("fps", 30.f);

    std::ofstream trace;
    if (a.has("csv")) {
        trace.open(a.str("csv"));
        trace << "ctrl,seed,t,target_az,target_el,gimbal_az,gimbal_el,err_deg\n";
    }
    std::string json = "{\n  \"latency_ms\": " + std::to_string(base.latency_s * 1000) +
                       ", \"p_miss\": " + std::to_string(base.p_miss) + ", \"noise_px\": " +
                       std::to_string(base.noise_px) + ", \"seeds\": " + std::to_string(seeds) + ",\n";
    const std::vector<std::string> ctrls{"raw", "sticky_nocmc", "track_nocmc", "track", "lead"};
    std::printf("%-12s %10s %10s %10s %10s %8s\n", "ctrl", "rms_deg", "p95_deg", "<0.5deg", "in_fov", "lost");
    for (size_t ci = 0; ci < ctrls.size(); ++ci) {
        SimConfig c = base;
        c.ctrl = ctrls[ci];
        std::vector<double> rms, p95, w05, fov, lost;
        for (int s = 0; s < seeds; ++s) {
            const auto r = runSim(c, static_cast<unsigned>(1000 + s), (trace.is_open() && s == 0) ? &trace : nullptr);
            rms.push_back(r.rms_deg);
            p95.push_back(r.p95_deg);
            w05.push_back(r.within_05);
            fov.push_back(r.in_fov);
            lost.push_back(r.lost_events);
        }
        auto mean = [](const std::vector<double>& v) {
            double s = 0;
            for (double x : v) s += x;
            return s / static_cast<double>(v.size());
        };
        std::printf("%-12s %10.3f %10.3f %10.3f %10.3f %8.2f\n", c.ctrl.c_str(), mean(rms), mean(p95), mean(w05),
                    mean(fov), mean(lost));
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "  \"%s\": {\"rms_deg\": %.4f, \"rms_deg_p05\": %.4f, \"rms_deg_p95\": %.4f, \"p95_err_deg\": "
                      "%.4f, \"time_within_0.5deg\": %.4f, \"time_in_fov\": %.4f, \"lost_events\": %.2f}%s\n",
                      c.ctrl.c_str(), mean(rms), percentile(rms, 5), percentile(rms, 95), mean(p95), mean(w05),
                      mean(fov), mean(lost), ci + 1 < ctrls.size() ? "," : "");
        json += buf;
    }
    json += "}\n";
    if (a.has("json")) std::ofstream(a.str("json")) << json;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 1;
}
