// Canli hat: yakalama -> CUDA on isleme -> TensorRT cikarim -> ByteTrack -> aci hatasi -> UDP.
//
// Veri kumesi kipi (kare atlanmaz, her dizi icin takip sifirlanir):
//   iha_track --engine m.engine --root data/antiuav410/test --out-dir results/trk/test_trt
//       [--dump-dets results/trk/dets_test] [--udp 127.0.0.1:5600] [takip bayraklari]
// Akis kipi (video dosyasi, RTSP adresi ya da '!' iceren GStreamer hatti):
//   iha_track --engine m.engine --source rtsp://... [--max-frames N] [--udp ...]
//   Ayri yakalama is parcacigi yalniz en son kareyi tutar; isleme yetismezse eski kareler
//   atilir (gecikme birikmez, atilan kare sayisi raporlanir).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "iha/args.hpp"
#include "iha/gimbal.hpp"
#include "iha/latency.hpp"
#include "iha/pipeline_config.hpp"
#include "iha/protocol.hpp"
#include "iha/sequence_io.hpp"
#include "iha/tracker.hpp"
#include "iha/trt_detector.hpp"
#include "iha/udp.hpp"

namespace fs = std::filesystem;
using namespace iha;
using Clock = std::chrono::steady_clock;

namespace {

uint64_t nowUs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch()).count());
}

// Tek yuvali posta kutusu: yazar her zaman ezer, okuyucu en yeniyi alir.
class LatestFrame {
public:
    void put(cv::Mat m, uint64_t stamp) {
        std::lock_guard<std::mutex> l(mu_);
        if (has_) ++dropped_;
        frame_ = std::move(m);
        stamp_ = stamp;
        has_ = true;
        cv_.notify_one();
    }
    bool take(cv::Mat& m, uint64_t& stamp, const std::atomic<bool>& done) {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return has_ || done.load(); });
        if (!has_) return false;
        m = std::move(frame_);
        stamp = stamp_;
        has_ = false;
        return true;
    }
    void wake() { cv_.notify_all(); }
    long dropped() const { return dropped_; }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    cv::Mat frame_;
    uint64_t stamp_ = 0;
    bool has_ = false;
    long dropped_ = 0;
};

struct Pipeline {
    TrtDetector& det;
    ByteTracker tracker;
    CameraModel cam;
    std::unique_ptr<UdpSender> udp;
    LatencyStats stats;
    uint32_t seq = 0;
    double lead_frames = 0;
    bool measure = true;

    Pipeline(TrtDetector& d, const TrackerParams& tp, const CameraModel& c) : det(d), tracker(tp), cam(c) {}

    // stamp_us: karenin elde edildigi an (okuma/cozme sonrasi). Hat gecikmesi bundan olculur.
    std::optional<TargetOutput> step(const cv::Mat& img, uint64_t stamp_us, Detections* dets_out, AngleError* ang) {
        Stopwatch total;
        Detections dets = det.detect(img, measure ? &stats : nullptr);
        Stopwatch sw;
        tracker.update(dets);
        const auto t = tracker.target();
        if (measure) stats.add("takip", sw.lap());
        AngleError a{};
        if (t) a = lead_frames > 0 ? predictedAngle(cam, t->box, t->vx, t->vy, lead_frames)
                                   : pixelToAngle(cam, t->box.cx(), t->box.cy());
        if (udp) {
            AngleErrorMsg m;
            m.valid = t.has_value();
            m.measured = t && t->measured;
            m.seq = seq++;
            m.stamp_us = stamp_us;
            m.track_id = t ? t->track_id : -1;
            m.az = static_cast<float>(a.az);
            m.el = static_cast<float>(a.el);
            m.score = t ? t->score : 0.f;
            m.latency_us = static_cast<uint32_t>(nowUs() - stamp_us);
            const auto b = encode(m);
            udp->send(b.data(), b.size());
        }
        if (measure) {
            stats.add("aci_ve_gonderim", sw.lap());
            stats.add("uctan_uca", total.lap());
        }
        if (dets_out) *dets_out = std::move(dets);
        if (ang) *ang = a;
        return t;
    }
};

}  // namespace

int main(int argc, char** argv) try {
    const Args a(argc, argv);
    TrtDetector det(a.required("engine"), a.f("conf", 0.01f));
    det.setStageSync(!a.has("no-stage-sync"));
    const CameraModel cam = CameraModel::fromHfov(640, 512, a.f("hfov", 30.f));
    Pipeline pipe(det, trackerParamsFromArgs(a), cam);
    pipe.lead_frames = a.f("lead-frames", 0.f);
    if (a.has("udp")) {
        const std::string hp = a.str("udp");
        const auto colon = hp.rfind(':');
        pipe.udp = std::make_unique<UdpSender>(hp.substr(0, colon), static_cast<uint16_t>(std::stoi(hp.substr(colon + 1))));
    }
    const int warmup = a.i("warmup", 30);
    const std::string stats_path = a.str("stats", "");
    LatencyStats io_stats;

    auto warm = [&](const cv::Mat& img) {
        pipe.measure = false;
        for (int i = 0; i < warmup; ++i) det.detect(img);
        pipe.tracker.reset();
        pipe.measure = true;
    };

    long frames = 0;
    const auto t0 = Clock::now();
    if (a.has("root")) {
        const std::string out_dir = a.required("out-dir");
        fs::create_directories(out_dir);
        const std::string dump = a.str("dump-dets", "");
        if (!dump.empty()) fs::create_directories(dump);
        bool warmed = false;
        const auto seqs = listSequenceDirs(a.required("root"));
        int si = 0;
        for (const auto& seq_dir : seqs) {
            const std::string name = fs::path(seq_dir).filename().string();
            const int n = countFrames(seq_dir);
            pipe.tracker.reset();
            TrackCsvWriter tw((fs::path(out_dir) / (name + ".csv")).string());
            std::unique_ptr<DetectionCsvWriter> dw;
            if (!dump.empty()) dw = std::make_unique<DetectionCsvWriter>((fs::path(dump) / (name + ".csv")).string());
            for (int f = 0; f < n; ++f) {
                Stopwatch rd;
                const cv::Mat img = cv::imread((fs::path(seq_dir) / frameFilename(f)).string(), cv::IMREAD_COLOR);
                if (img.empty()) throw std::runtime_error("okunamadi: " + seq_dir + "/" + frameFilename(f));
                io_stats.add("okuma_cozme", rd.lap());
                if (!warmed) {
                    warm(img);
                    warmed = true;
                }
                Detections d;
                AngleError ang;
                const auto t = pipe.step(img, nowUs(), dw ? &d : nullptr, &ang);
                if (dw) dw->write(f, d);
                tw.write(f, t, ang);
                ++frames;
            }
            std::fprintf(stderr, "[%d/%zu] %s: %d kare\n", ++si, seqs.size(), name.c_str(), n);
        }
    } else {
        const std::string src = a.required("source");
        cv::VideoCapture cap;
        if (src.find('!') != std::string::npos)
            cap.open(src, cv::CAP_GSTREAMER);
        else
            cap.open(src, cv::CAP_ANY);
        if (!cap.isOpened()) throw std::runtime_error("kaynak acilamadi: " + src);
        const long max_frames = a.i("max-frames", 0);
        LatestFrame box;
        std::atomic<bool> done{false};
        std::thread grabber([&] {
            cv::Mat m;
            while (!done && cap.read(m)) box.put(m.clone(), nowUs());
            done = true;
            box.wake();
        });
        cv::Mat img;
        uint64_t stamp = 0;
        bool warmed = false;
        while (box.take(img, stamp, done)) {
            if (!warmed) {
                warm(img);
                warmed = true;
                continue;
            }
            io_stats.add("kuyruk_bekleme", static_cast<double>(nowUs() - stamp) / 1000.0);
            const auto t = pipe.step(img, stamp, nullptr, nullptr);
            ++frames;
            if (t && frames % 30 == 0)
                std::fprintf(stderr, "kare %ld: hedef #%d (%.0f, %.0f)\n", frames, t->track_id, t->box.cx(), t->box.cy());
            if (max_frames && frames >= max_frames) break;
        }
        done = true;
        box.wake();
        grabber.join();
        std::fprintf(stderr, "atilan kare (isleme yetismedi): %ld\n", box.dropped());
    }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("%ld kare, %.1f s, %.1f kare/sn (okuma dahil)\n", frames, s, static_cast<double>(frames) / s);
    std::string json = pipe.stats.toJson();
    const std::string io_json = io_stats.toJson();
    std::printf("hat:\n%s\ngirdi:\n%s", json.c_str(), io_json.c_str());
    if (!stats_path.empty()) {
        std::ofstream os(stats_path);
        os << "{\"pipeline\": " << json << ", \"io\": " << io_json << ", \"frames\": " << frames
           << ", \"wall_s\": " << s << ", \"stage_sync\": " << (a.has("no-stage-sync") ? "false" : "true") << "}\n";
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "hata: " << e.what() << "\n";
    return 1;
}
